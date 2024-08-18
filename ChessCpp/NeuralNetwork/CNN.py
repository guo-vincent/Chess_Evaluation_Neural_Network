import pandas as pd
import numpy as np
from keras import layers, models, Input, callbacks, optimizers, regularizers, losses
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import json

# Takes on average 15 minutes to process.
total_number = 12956364 # Number of matrices in the file
white_matrices_total = 6473070
black_matrices_total = 6483294
file_name_white = R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\CSVFiles\White.csv"
file_name_black = R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\CSVFiles\Black.csv"
number_matrixes_white = 6473070    # Specify the number of matrices you want to read
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
    matrices_white = read_matrix(file_name_white, number_matrixes_white)
    print("Done!")

    # Extract matrices and evaluations
    X_white = np.array([matrix/100 for matrix, _ in matrices_white])
    y_white = np.array([evaluation for _, evaluation in matrices_white])

    # Convert evaluations to numpy array
    y_white = y_white.reshape(-1, 1)

    # Normalize the evaluations using MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_white = scaler.fit_transform(y_white).flatten()

    # Reshape X_white to include the channel dimension
    X_white = X_white.reshape((X_white.shape[0], 8, 8, 1))  # (number_matrixes, 8, 8, 1)

    # Split data into training and validation sets. Set 1 has random state 42.
    X_train, X_val, y_train, y_val = train_test_split(X_white, normalized_y_white, test_size=0.2, random_state=42)

    model_white = models.Sequential([
        Input(shape=(8, 8, 1)),
        layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),
        
        layers.Conv2D(filters=64, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),

        layers.Conv2D(filters=64, kernel_size=(8, 8), kernel_regularizer=regularizers.l2(0.0001), padding='same'),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),
        
        layers.Flatten(),
        
        layers.Dense(1024,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),
        
        layers.Dense(512,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),

        layers.Dense(216,kernel_regularizer=regularizers.l2(0.0001)),
        layers.BatchNormalization(epsilon=1e-5),
        layers.LeakyReLU(0.1),
        
        layers.Dense(1, activation='linear')
    ])

    optimizer = optimizers.Adam(learning_rate=0.0002)

    # Compile the model
    model_white.compile(optimizer=optimizer,
                        loss=losses.MeanAbsoluteError(),
                        metrics=['mean_absolute_error'])

    # Print model summary
    print(model_white.summary())

    early_stopping = callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
    reduceLr = callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.1)

    # Fit the model
    history = model_white.fit(X_train, y_train,
                            epochs=25,
                            batch_size=1024,
                            validation_data=(X_val, y_val),
                            shuffle=True,
                            callbacks=[early_stopping, reduceLr])

    # Evaluate the model on the validation data
    loss, metric = model_white.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Metric: {metric}")

    model_white.export(R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\Chess_White_2")

    # Extract loss values from the history object
    epochs = range(1, len(history.history['loss']) + 1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Save the plots:
    with open(R'C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\Chess_White_2\training_history_white_2.json', 'w') as file:
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