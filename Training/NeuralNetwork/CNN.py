# to run: python CNN.py --model_key model_name
import pandas as pd
import numpy as np
from keras import layers, models, Input, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm
import tensorflow as tf
import joblib
import argparse
import sys

class ExtremeValueMAE(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.2, name="extreme_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.mae = self.add_weight(name="mae", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.abs(y_true) > self.threshold
        if tf.reduce_any(mask):
            abs_errors = tf.abs(y_true - y_pred)
            selected_errors = tf.boolean_mask(abs_errors, mask)
            self.mae.assign_add(tf.reduce_sum(selected_errors))
            self.count.assign_add(tf.cast(tf.size(selected_errors), tf.float32))

    def result(self):
        return self.mae / tf.maximum(self.count, 1.0)

    def reset_state(self):
        self.mae.assign(0)
        self.count.assign(0)

def read_matrix(file_name, number_matrices, progress=True):
    """Reads White.csv/Black.csv and parses it into data that the neural network can read"""
    matrices = []
    evaluations = []

    reader = pd.read_csv(
        file_name,
        header=0,        # first line is “col0,…,Evaluation”
        chunksize=9,
        dtype=float,
        on_bad_lines='warn'
    )

    if progress:
        reader = tqdm(reader, total=number_matrices, desc="Reading matrices")

    for idx, df9 in enumerate(reader):
        if idx >= number_matrices:
            break

        board_block = df9.iloc[0:8, 0:8].to_numpy(dtype=np.float32)
        eval_value = df9.iloc[8]["Evaluation"]

        matrices.append(board_block)
        evaluations.append(eval_value)

    return matrices[:number_matrices], np.array(evaluations[:number_matrices], dtype=np.float32)

# ─────── Helper to compute sample‐weights for smart sampling ───────
def compute_sample_weights(model, X, y):
    preds = model.predict(X, batch_size=4096, verbose=0).flatten()
    errors = np.abs(preds - y)
    max_error = 0.3
    clipped = np.minimum(errors, max_error)
    eps = 1e-8
    weights = clipped + eps
    weights = weights / np.sum(weights)
    return weights

# ─────── Helper to split X, y into curriculum buckets ───────
def split_curriculum(X, y, q1=0.50, q2=0.90):
    abs_y = np.abs(y)
    thresh1 = np.quantile(abs_y, q1)
    thresh2 = np.quantile(abs_y, q2)

    easy_mask   = abs_y <=  thresh1
    medium_mask = (abs_y >  thresh1) & (abs_y <= thresh2)
    hard_mask   = abs_y >  thresh2

    X_easy,   y_easy   = X[easy_mask],   y[easy_mask]
    X_medium, y_medium = X[medium_mask], y[medium_mask]
    X_hard,   y_hard   = X[hard_mask],   y[hard_mask]

    return (X_easy, y_easy), (X_medium, y_medium), (X_hard, y_hard)

def progressive_curriculum(X, y, bins=4):
    abs_y = np.abs(y)
    quantiles = np.quantile(abs_y, [0, 0.3, 0.6, 0.9, 1.0])
    
    buckets = []
    for i in range(bins):
        mask = (abs_y >= quantiles[i]) & (abs_y < quantiles[i+1])
        if i == bins-1:
            mask = abs_y >= quantiles[i]
        X_bucket = X[mask]
        y_bucket = y[mask]
        if len(X_bucket) > 0:
            buckets.append((f"Bucket_{i}", X_bucket, y_bucket))
    
    return sorted(buckets, key=lambda x: np.mean(np.abs(x[2])))


def create_model(config):
    """
    Create CNN model.  For overfit_test, we disable
    BatchNorm, L2, and Dropout, and use a tiny network
    so it can truly memorize 8 samples.
    """
    inputs = Input(shape=(8, 8, 16))

    # Overfit‐test: no L2, no BatchNorm, no Dropout, 1 block, tiny filters
    if config['name'] == 'chess_eval_overfit_test':
        reg = None
        use_bn = False
        num_blocks = 1
    else:
        reg = tf.keras.regularizers.l2(1e-5)
        use_bn = True
        num_blocks = config['residual_blocks']

    # Helper for conditional BatchNorm:
    def maybe_bn(x):
        return layers.BatchNormalization()(x) if use_bn else x

    # First convolution
    x = layers.Conv2D(
        min(128, config['filters']),  # subject to config['filters']
        (3, 3),
        padding='same',
        kernel_regularizer=reg
    )(inputs)
    x = maybe_bn(x)
    x = layers.LeakyReLU(alpha=0.01)(x)

    # Residual blocks
    for _ in range(min(num_blocks, config['residual_blocks'])):
        shortcut = x
        x = layers.Conv2D(
            config['filters'], (3, 3), padding='same',
            kernel_regularizer=reg
        )(x)
        x = maybe_bn(x)
        x = layers.LeakyReLU(alpha=0.01)(x)

        x = layers.Conv2D(
            config['filters'], (3, 3), padding='same',
            kernel_regularizer=reg
        )(x)
        x = maybe_bn(x)

        x = layers.Add()([shortcut, x])
        x = layers.LeakyReLU(alpha=0.01)(x)

    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer
    x = layers.Dense(
        config['dense_units'],
        activation='relu',
        kernel_regularizer=reg
    )(x)
    x = maybe_bn(x)

    # Dropout only if NOT overfit_test
    if config['name'] == 'chess_eval_overfit_test':
        x = layers.Dropout(0.0)(x)
    else:
        x = layers.Dropout(config['dropout_rate'])(x)

    outputs = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs, outputs)


def create_positional_features():
    ranks = np.tile(np.arange(8).reshape(8, 1), (1, 8))
    files = np.tile(np.arange(8).reshape(1, 8), (8, 1))
    return ranks, files

def enhance_board(board):
    piece_types = [100, 9, 5, 4, 3, 1, -100, -9, -5, -4, -3, -1]
    channels = [(board == pt).astype(np.float32) for pt in piece_types]
    
    ranks, files = create_positional_features()
    channels.append(ranks / 3.5 - 1)
    channels.append(files / 3.5 - 1)
    
    def get_king_zone(king_value):
        zone = np.zeros((8, 8), dtype=np.float32)
        king_pos = np.argwhere(board == king_value)
        if len(king_pos) > 0:
            row, col = king_pos[0]
            row_start = max(0, row-1)
            row_end = min(8, row+2)  # +2 for inclusive slice
            col_start = max(0, col-1)
            col_end = min(8, col+2)
            zone[row_start:row_end, col_start:col_end] = 1.0
        return zone
    
    channels.append(get_king_zone(100))   # White king zone
    channels.append(get_king_zone(-100))  # Black king zone
    
    return np.stack(channels, axis=-1)

def main(config):
    """Main training function with configuration"""
    print(f"\n{'='*50}")
    print(f"Training {config['name']} model")
    print(f"{'='*50}\n")
    
    os.makedirs(config['save_path'], exist_ok=True)
    with open(os.path.join(config['save_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    matrices, y_data = read_matrix(
        config['data_path'], 
        config['samples'],
        progress=config.get('show_progress', True)
    )
    print(f"Loaded {len(matrices)} positions")
    
    if config.get('show_progress', True):
        iterator = tqdm(matrices, desc="Enhancing boards")
    else:
        iterator = matrices
        
    X_data = np.array([enhance_board(matrix) for matrix in iterator])
    
    if config['name'] == 'chess_eval_overfit_test':
        X_significant = X_data
        y_significant = y_data
    else:
        significant_mask = np.abs(y_data) > config.get('threshold', 100)
        X_significant = X_data[significant_mask]
        y_significant = y_data[significant_mask]
    
    print(f"Significant positions: {len(X_significant)}/{len(X_data)} "
          f"({len(X_significant)/len(X_data):.1%})")
    
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(y_data.reshape(-1, 1))
    joblib.dump(scaler, os.path.join(config['save_path'], "scaler.pkl"))
    
    y_scaled = scaler.transform(y_data.reshape(-1, 1)).flatten()
    y_sig_scaled = scaler.transform(y_significant.reshape(-1, 1)).flatten()
    
    seed = config.get('seed', 42)
    test_size = config.get('test_size', 0.1)
    
    if config['name'] == 'chess_eval_overfit_test':
        X_train_full, y_train_full = X_data, y_scaled
        X_val_full, y_val_full     = X_data, y_scaled
    else:
        X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
            X_data, y_scaled, test_size=test_size, random_state=seed
        )
    
    if config['name'] == 'chess_eval_overfit_test':
        X_train_sig, y_train_sig = X_significant, y_sig_scaled
        X_val_sig, y_val_sig     = X_significant, y_sig_scaled
    else:
        X_train_sig, X_val_sig, y_train_sig, y_val_sig = train_test_split(
            X_significant, y_sig_scaled, test_size=test_size, random_state=seed
        )

    model = create_model(config)
    mean_eval = np.mean(y_sig_scaled)
    model.layers[-1].bias.assign([mean_eval])
    model.summary()
    
    print("\n=== PHASE 1: Training on significant positions ===")
    optimizer = optimizers.Adam(learning_rate=config['learning_rate_phase1'])
    
    if config['name'] == 'chess_eval_overfit_test':
        phase1_loss = 'mse'
        phase1_metrics = ['mae']
    else:
        phase1_loss = tf.keras.losses.Huber(delta=1.0)
        phase1_metrics = ['mae', ExtremeValueMAE(threshold=0.2)]
    
    model.compile(optimizer=optimizer, loss=phase1_loss, metrics=phase1_metrics)
    
    checkpoint1 = callbacks.ModelCheckpoint(
        os.path.join(config['save_path'], "phase1_best.h5"),
        save_best_only=True,
        monitor='val_mae',
        mode='min'
    )
    early_stop1 = callbacks.EarlyStopping(
        monitor='val_mae',
        patience=config.get('patience', 5),
        restore_best_weights=True
    )
    
    if config['name'] == 'chess_eval_overfit_test':
        callbacks_for_phase1 = [checkpoint1]
    else:
        callbacks_for_phase1 = [checkpoint1, early_stop1]
    
    history1 = model.fit(
        X_train_sig, y_train_sig,
        validation_data=(X_val_sig, y_val_sig),
        epochs=config['epochs_phase1'],
        batch_size=config['batch_size'],
        callbacks=callbacks_for_phase1,
        verbose=config.get('verbose', 1)
    )
    
    if config['name'] == 'chess_eval_overfit_test':
        print("\nSkipping Phase 2 curriculum and final pass (overfit test only)")
        print(f"Evaluating on TRAINING SET of size {len(y_train_sig)}")
        loss, mae_scaled, *_ = model.evaluate(
            X_train_sig, y_train_sig,
            batch_size=config['batch_size'],
            verbose=0
        )
        try:
            mae_unscaled = scaler.inverse_transform([[mae_scaled]])[0][0]
            print(f"Overfit Test Results:")
            print(f"  → Scaled MAE:    {mae_scaled:.4f}")
            print(f"  → Unscaled MAE:  {mae_unscaled:.2f} (approx. centipawns)")
        except Exception as e:
            print(f"  → Could not unscale MAE: {e}")
        model.save(os.path.join(config['save_path'], "overfit_model.h5"))
        return mae_scaled
    
    print("\n=== PHASE 2A: Progressive Curriculum Learning ===")
    bins = config.get('curriculum_bins', 10)
    buckets = progressive_curriculum(X_train_full, y_train_full, bins=bins)

    checkpoint2 = callbacks.ModelCheckpoint(
        os.path.join(config['save_path'], "phase2_curriculum_best.h5"),
        save_best_only=True,
        monitor='val_mae',
        mode='min'
    )
    
    initial_lr = tf.keras.backend.get_value(model.optimizer.lr)
    current_lr = initial_lr

    for i, (bucket_name, X_bucket, y_bucket) in enumerate(buckets):
        current_lr *= 0.8
        tf.keras.backend.set_value(model.optimizer.lr, current_lr)
        
        print(f"\n  → Training Bucket {i+1}/{len(buckets)}: {bucket_name} "
              f"({len(y_bucket)} samples, |y|_mean={np.mean(np.abs(y_bucket)):.3f})")
        
        model.fit(
            X_bucket, y_bucket,
            validation_data=(X_val_full, y_val_full),
            epochs=config.get('epochs_curriculum', 2),
            batch_size=config['batch_size'],
            callbacks=[checkpoint2],
            verbose=config.get('verbose', 1)
        )
    
    tf.keras.backend.set_value(model.optimizer.lr, initial_lr)

    print("\n=== PHASE 2B: Final Smart-Sampling Resampling ===")
    model.load_weights(os.path.join(config['save_path'], "phase2_curriculum_best.h5"))
    huber = tf.keras.losses.Huber(delta=0.2)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['learning_rate_phase2']),
        loss='mse',
        metrics=['mae']
    )
    
    def resample_with_mix(X, y, model, mix=0.3):
        N = len(X)
        sample_probs = compute_sample_weights(model, X, y)

        n_error = int(N * (1.0 - mix))
        n_rand  = N - n_error

        idx_error = np.random.choice(
            np.arange(N), size=n_error, replace=True, p=sample_probs
        )
        idx_rand  = np.random.choice(
            np.arange(N), size=n_rand, replace=True
        )

        new_indices = np.concatenate([idx_error, idx_rand])
        np.random.shuffle(new_indices)
        return X[new_indices], y[new_indices]
    
    X_train_weighted, y_train_weighted = resample_with_mix(
        X_train_full, y_train_full, model, mix=0.3)
    print(f"  → Resampled {len(X_train_weighted)} training points based on prediction error.")
    
    checkpoint3 = callbacks.ModelCheckpoint(
        os.path.join(config['save_path'], "phase2_final_best.h5"),
        save_best_only=True,
        monitor='val_mae',
        mode='min'
    )
    early_stop3 = callbacks.EarlyStopping(
        monitor='val_mae',
        patience=config.get('patience', 3),
        restore_best_weights=True
    )
    final_epochs = config.get('epochs_final', 3)
    
    model.fit(
        X_train_weighted, y_train_weighted,
        validation_data=(X_val_full, y_val_full),
        epochs=final_epochs,
        batch_size=config['batch_size'],
        callbacks=[checkpoint3, early_stop3],
        verbose=config.get('verbose', 1)
    )
    
    print("\n=== FINAL EVALUATION on Validation Set ===")
    model.load_weights(os.path.join(config['save_path'], "phase2_final_best.h5"))
    loss, mae = model.evaluate(
        X_val_full, y_val_full,
        batch_size=config['batch_size'],
        verbose=0
    )
    print(f"\nFinal Validation MAE: {mae:.4f}")
    
    model.save(os.path.join(config['save_path'], "final_model.h5"))
    full_history = {
        'phase1': history1.history,
    }
    with open(os.path.join(config['save_path'], 'training_history.json'), 'w') as f:
        json.dump(full_history, f)
    
    if config.get('plot_history', True):
        plt.figure(figsize=(12, 8))
        x1 = np.arange(1, len(history1.history['mae']) + 1)
        y1_train = history1.history['mae']
        y1_val   = history1.history['val_mae']
        plt.plot(x1, y1_train, label='Phase1 Train MAE')
        plt.plot(x1, y1_val,   label='Phase1 Val MAE')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.title(f"{config['name']} - Training MAE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(config['save_path'], 'training_history.png'))
        plt.close()
    
    print(f"Model and artifacts saved to {config['save_path']}")
    return mae

if __name__ == "__main__":
    from config import MODEL_CONFIGS

    parser = argparse.ArgumentParser(description='Train chess evaluation model')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--model_key', type=str, help='Model key from config.py (e.g. white_quick_1)')
    parser.add_argument('--quick', action='store_true', help='Run quick test with small dataset')
    parser.add_argument('--overfit_test', action='store_true', help='Run overfit sanity check on small data')
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        config['name'] = os.path.splitext(os.path.basename(args.config))[0]

    elif args.model_key:
        if args.model_key not in MODEL_CONFIGS:
            print(f"Error: '{args.model_key}' not found in MODEL_CONFIGS.")
            print("Available keys:", ", ".join(MODEL_CONFIGS.keys()))
            sys.exit(1)

        config = MODEL_CONFIGS[args.model_key].copy()
        config['name'] = args.model_key
        
        # ─────────── **TUNE THESE FOR MORE CAPACITY / LONGER TRAINING** ───────────
        config['learning_rate_phase1'] = 5e-4        # slightly higher LR
        config['learning_rate_phase2'] = 5e-5
        config['epochs_phase1'] = 40                 # more epochs
        config['epochs_phase2'] = 5
        config.setdefault('epochs_curriculum', 3)
        config.setdefault('epochs_final', 3)
        config.setdefault('test_size', 0.1)
        config.setdefault('patience', 4)             # be a bit more patient
        config.setdefault('show_progress', True)
        config.setdefault('plot_history', True)

        # ─────────── **INCREASE CAPACITY** ───────────
        config.setdefault('residual_blocks', 4)       # try 4 → 6 if still underfitting
        config.setdefault('filters', 128)             # try 128 → 256
        config.setdefault('dense_units', 256)         # try 256 → 512
        config.setdefault('dropout_rate', 0.2)        # keep dropout moderate

    elif args.overfit_test:
        config = {
            'name': 'chess_eval_overfit_test',
            'data_path': "CSVFiles/Black.csv",
            'samples': 8,
            'save_path': "Overfit_Test_Model",
            'threshold': 0,
            'batch_size': 10,
            'residual_blocks': 1,
            'filters': 16,
            'dense_units': 32,
            'dropout_rate': 0.0,
            'learning_rate_phase1': 0.01,
            'learning_rate_phase2': 0.0,
            'epochs_phase1': 800,
            'epochs_phase2': 0,
            'epochs_curriculum': 0,
            'epochs_final': 0,
            'test_size': 0.0,
            'patience': 0,
            'show_progress': True,
            'plot_history': False,
            'curriculum_bins': 1
        }
    else:
        # ─────────── **DEFAULT “quicktest” STILL TWEAKED FOR NO UNDERFITTING** ───────────
        config = {
            'name': 'chess_eval_quicktest',
            'data_path': "CSVFiles/Black.csv",
            'samples': 1000,
            'save_path': "Chess_Model_QuickTest",
            'threshold': 200,
            'batch_size': 256,
            'residual_blocks': 3,       # bump from 2→3
            'filters': 64,              # bump from 32→64
            'dense_units': 128,         # bump from 64→128
            'dropout_rate': 0.1,
            'learning_rate_phase1': 5e-4,   # higher LR
            'learning_rate_phase2': 5e-5,
            'epochs_phase1': 20,        # longer than 5
            'epochs_phase2': 3,
            'epochs_curriculum': 1,
            'epochs_final': 1,
            'test_size': 0.2,
            'patience': 3,
            'show_progress': True,
            'plot_history': True,
            'curriculum_bins': 4
        }

    if args.quick:
        print("Running in quick test mode")
        config['samples'] = 1000
        config['epochs_phase1'] = 5
        config['epochs_phase2'] = 2
        config['epochs_curriculum'] = 1
        config['epochs_final'] = 1
        config['batch_size'] = 256
        config['show_progress'] = False
        config['save_path'] = config.get('save_path', 'Chess_Model_Quick') + "_Test"

    try:
        final_mae = main(config)
        print(f"Training completed successfully! Final MAE: {final_mae:.4f}")
    except Exception as e:
        print(f"Training failed: {str(e)}")
        sys.exit(1)
