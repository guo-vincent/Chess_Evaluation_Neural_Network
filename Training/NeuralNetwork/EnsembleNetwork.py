from keras.models import load_model
from keras import Model, Input
from keras.layers import Concatenate, Dense, Dropout, Layer, Input, Multiply, Lambda, Activation
from keras.callbacks import EarlyStopping
from CNN import read_matrix, enhance_board
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
import joblib
from config import MODEL_CONFIGS
import os

class UnscaleLayer(Layer):
    def __init__(self, scale, center, **kwargs):
        super(UnscaleLayer, self).__init__(**kwargs)
        self.scale = np.array(scale, dtype=np.float32)
        self.center = np.array(center, dtype=np.float32)

    def build(self, input_shape):
        self.scale_tensor = self.add_weight(
            name="scale", shape=self.scale.shape, trainable=False,
            initializer=tf.constant_initializer(self.scale)
        )
        self.center_tensor = self.add_weight(
            name="center", shape=self.center.shape, trainable=False,
            initializer=tf.constant_initializer(self.center)
        )

    def call(self, inputs):
        return inputs * self.scale_tensor + self.center_tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale.tolist(),
            "center": self.center.tolist(),
        })
        return config

# so we dont get a valueError of nonunique models
def rename_models(models, prefix):
    for i, model in enumerate(models):
        model._name = f"{prefix}_submodel_{i}"
    return models

def load_scalers(model_paths):
    scalers = []
    for path in model_paths:
        scaler_path = os.path.join(os.path.dirname(path), "scaler.pkl")
        scalers.append(joblib.load(scaler_path))
    return scalers

def build_extreme_aware_ensemble(submodels, submodel_scalers, freeze=True):
    inputs = Input(shape=(8, 8, 16))
    if freeze:
        for m in submodels:
            m.trainable = False

    # 1) Get each submodel’s unscaled output
    outs = []
    for m, scaler in zip(submodels, submodel_scalers):
        raw = m(inputs)
        unscaled = UnscaleLayer(scaler.scale_, scaler.center_)(raw)
        outs.append(unscaled)  # shape = (batch, 1)

    # Stack to shape (batch, n_models)
    stacked = Concatenate(axis=-1)(outs)

    # 2) “Normal” regressor head
    h_norm = Dense(64, activation='relu')(stacked)
    h_norm = Dropout(0.3)(h_norm)
    out_norm = Dense(1, name='out_normal')(h_norm)

    # 3) “Extreme” regressor head
    h_xtrm = Dense(64, activation='relu')(stacked)
    h_xtrm = Dropout(0.3)(h_xtrm)
    out_xtrm = Dense(1, name='out_extreme')(h_xtrm)

    # 4) Gating network: scalar in [0, 1]
    g_gate = Dense(64, activation='relu')(stacked)
    g_gate = Dropout(0.2)(g_gate)
    g_gate = Dense(16, activation='relu')(g_gate)
    g_gate = Dense(1, name='gate_logits')(g_gate)
    gate = Activation('sigmoid', name='gate')(g_gate)  # shape=(batch,1)

    # 5) Blend the two heads: (1−gate)*norm + gate*extreme
    part_norm = Multiply()([out_norm, Lambda(lambda x: 1.0 - x)(gate)])
    part_xtrm = Multiply()([out_xtrm, gate])
    final_out = Lambda(lambda x: x[0] + x[1], name='ensemble_output')([part_norm, part_xtrm])

    return Model(inputs=inputs, outputs=final_out)


def main(config):
    white_models = [load_model(p) for p in config['white_models']]
    black_models = [load_model(p) for p in config['black_models']]
    
    white_scalers = load_scalers(config['white_models'])
    black_scalers = load_scalers(config['black_models'])
    
    white_models = rename_models(white_models, "white")
    black_models = rename_models(black_models, "black")

    ensemble_white = build_extreme_aware_ensemble(white_models, white_scalers, freeze=True)
    ensemble_black = build_extreme_aware_ensemble(black_models, black_scalers, freeze=True)

    X_white_raw, y_white = read_matrix(config['white_data_path'], config['white_samples'])
    X_black_raw, y_black = read_matrix(config['black_data_path'], config['black_samples'])

    X_white = np.array([enhance_board(x) for x in X_white_raw])
    X_black = np.array([enhance_board(x) for x in X_black_raw])
    
    white_target_scaler = RobustScaler()
    black_target_scaler = RobustScaler()

    y_white_scaled = white_target_scaler.fit_transform(y_white.reshape(-1, 1)).flatten()
    y_black_scaled = black_target_scaler.fit_transform(y_black.reshape(-1, 1)).flatten()

    ensemble_white.compile(optimizer=Adam(learning_rate=config['learning_rate']),
                           loss='mse', metrics=['mae'])
    ensemble_black.compile(optimizer=Adam(learning_rate=config['learning_rate']),
                           loss='mse', metrics=['mae'])

    ensemble_white.fit(X_white, y_white_scaled, batch_size=config['batch_size'], epochs=config['epochs'])
    ensemble_black.fit(X_black, y_black_scaled, batch_size=config['batch_size'], epochs=config['epochs'])

    for model in white_models:
        for layer in model.layers[-3:]:
            layer.trainable = True
    ensemble_white.compile(optimizer=Adam(learning_rate=config['learning_rate'] * 0.1), 
                           loss='mse', metrics=['mae'])
    ensemble_white.fit(
        X_white, y_white_scaled,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    for model in black_models:
        for layer in model.layers[-3:]:
            layer.trainable = True
    ensemble_black.compile(optimizer=Adam(learning_rate=config['learning_rate'] * 0.1), 
                           loss='mse', metrics=['mae'])
    ensemble_black.fit(
        X_black, y_black_scaled,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    ensemble_white.save("Chess_Combined_Quick/ensemble_white.keras")
    ensemble_black.save("Chess_Combined_Quick/ensemble_black.keras")
    
    joblib.dump(white_target_scaler, "Chess_Combined_Quick/ensemble_white_target_scaler.pkl")
    joblib.dump(black_target_scaler, "Chess_Combined_Quick/ensemble_black_target_scaler.pkl")
    
if __name__ == "__main__":
    config = MODEL_CONFIGS["combined_quick_enqueue"]
    main(config)
