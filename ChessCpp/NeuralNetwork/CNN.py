import pandas as pd
import numpy as np
from keras import layers, models, Input, callbacks, optimizers, regularizers, initializers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import json
import tensorflow as tf

# Takes on average 15 minutes to process.
total_number = 12956364 # Number of matrices in the file
black_matrices_total = 6483294
file_name_black = R"CSVFiles/Black.csv"
number_matrixes_black = 6483294    # Specify the number of matrices you want to read

def read_matrix(file_name, number_matrixes):
    # Rows to read: header + (number of matrices * 9 rows per matrix)
    rows_to_read = number_matrixes * 9 + 1

    data = pd.read_csv(file_name, nrows=rows_to_read)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        # Extract the board matrix and evaluation from the DataFrame
        board_matrix = data.iloc[start_row:start_row+8, :-1].to_numpy(dtype=np.float64)
        evaluation = data.iloc[start_row+8, -1]

        matrices.append((board_matrix, evaluation))
    
    return matrices

if __name__ == "__main__":
    matrices_black = read_matrix(file_name_black, number_matrixes_black)
    print("Done!")

    # Extract matrices and evaluations
    X_black = np.array([matrix/100 for matrix, _ in matrices_black])
    y_black = np.array([evaluation for _, evaluation in matrices_black])

    # Convert evaluations to numpy array
    y_black = y_black.reshape(-1, 1)

    # Normalize the evaluations using MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_black = scaler.fit_transform(y_black).flatten()

    # Reshape X_black to include the channel dimension
    X_black = X_black.reshape((X_black.shape[0], 8, 8, 1))  # (number_matrixes, 8, 8, 1)

    # Split data into training and validation sets. Set 1 has random state 42.
    X_train, X_val, y_train, y_val = train_test_split(X_black, normalized_y_black, test_size=0.2, random_state=42)
    
    # Root loss, since output values are in [-1, 1]. Epilson is needed to avoid nan loss through square rooting small negative values via float precision issues.
    def root_loss(y_true, y_pred, epsilon=1e-8): # Second tf.square term used to counterbalance overabundance of evaluations near 0 and convergence to evaluations near 0.
        diff = tf.abs(y_true - y_pred)
        safe_diff = tf.maximum(diff, epsilon)  # Ensure we don't get near zero
        sqrt_loss = tf.sqrt(safe_diff)
        scaling_factor = tf.square(tf.math.exp(tf.clip_by_value(y_true, 0, float('inf'))))  # Clipping to avoid negative values in exp
        return tf.reduce_mean(sqrt_loss * scaling_factor)

    model_black = models.Sequential([
        Input(shape=(8, 8, 1)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        
        layers.Conv2D(filters=64, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        
        layers.Conv2D(filters=64, kernel_size=(7, 7), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        layers.Dropout(0.1, seed=42),

        layers.Conv2D(filters=32, kernel_size=(8, 8), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        
        layers.Flatten(),
        
        layers.Dense(1024,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        
        layers.Dense(512,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),

        layers.Dense(216,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-8),
        layers.LeakyReLU(0.1),
        
        layers.Dense(1, activation='linear')
    ])

    # I wonder how much the learning rate can be increased. lr = .0005 seems to be stable.
    optimizer = optimizers.Adam(learning_rate=0.0005, amsgrad=True)

    model_black.compile(optimizer=optimizer, 
                        loss=root_loss, 
                        metrics=['mean_absolute_error'])

    # Callbacks
    print(model_black.summary())

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    reduceLr = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.4)

    history = model_black.fit(X_train, y_train,
                            epochs=50,
                            batch_size=2048,
                            validation_data=(X_val, y_val),
                            shuffle=True,
                            callbacks=[early_stopping, reduceLr])

    # Evaluate the model on the validation data
    loss, metric = model_black.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss: {loss}")
    print(f"Validation Metric: {metric}")

    model_black.export(R"Chess_Black_3")

    # Extract loss values from the history object
    epochs = range(1, len(history.history['loss']) + 1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Save the plots:
    with open(R'Chess_Black_3/training_history_black_3.json', 'w') as file:
        json.dump(history.history, file)

    # Create scatter plot for training loss
    plt.figure(figsize=(10, 6))
    plt.scatter(epochs, loss, s=1, alpha=0.5, label='Training Loss')
    
    # Create scatter plot for validation loss
    plt.scatter(epochs, val_loss, s=1, alpha=0.5, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Scatter Plot of Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()