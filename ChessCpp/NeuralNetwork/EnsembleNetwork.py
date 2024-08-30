import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from keras import layers, models, metrics, optimizers, callbacks, regularizers
import joblib
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt


total_number = 12956364 # Number of matrices in the file
white_matrices_total = 6473070
file_name_white = "CSVFiles/White.csv"
number_matrixes_white = 6473070    # Specify the number of matrices you want to read

def read_matrix(file_name, number_matrixes):
    # Rows to read: header + (number of matrices * 9 rows per matrix)
    rows_to_read = number_matrixes * 9 + 1

    data = pd.read_csv(file_name, nrows=rows_to_read)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        # Extract the board matrix and evaluation from the DataFrame
        board_matrix = data.iloc[start_row:start_row+8, :-1].to_numpy(dtype=np.float32)
        evaluation = data.iloc[start_row+8, -1]

        matrices.append((board_matrix, evaluation))
    
    return matrices

if __name__ == "__main__":
    matrices_white = read_matrix(file_name_white, number_matrixes_white)
    print("Finished reading from csv file!")

    # Extract matrices and evaluations
    X_white = np.array([matrix/100 for matrix, _ in matrices_white])
    y_white = np.array([evaluation for _, evaluation in matrices_white])
    del matrices_white

    # Convert evaluations to numpy array
    y_white = y_white.reshape(-1, 1)

    # Normalize the evaluations using MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_white = scaler.fit_transform(y_white).flatten()

    # Load Models
    model1 = tf.saved_model.load("Chess_White")
    model2 = tf.saved_model.load("Chess_White_2")
    model3 = tf.saved_model.load("Chess_White_3")

    # Create Predictions
    predictions1, predictions2, predictions3 = [], [], []
    infer1 = model1.signatures['serving_default']
    infer2 = model2.signatures['serving_default']
    infer3 = model3.signatures['serving_default']
    scaler = joblib.load("Scalers/ScalerWhite.pkl")
    for matrix in X_white:
        matrix = np.expand_dims(np.expand_dims(matrix, axis=-1), axis=0)  # Shape (1, 8, 8, 1)
        tensor_matrix = tf.convert_to_tensor(matrix)
        output_array1 = infer1(tensor_matrix)['output_0']
        output_array2 = infer2(tensor_matrix)['output_0']
        output_array3 = infer3(tensor_matrix)['output_0']
        predictions1.append(output_array1[0][0])
        predictions2.append(output_array2[0][0])
        predictions3.append(output_array3[0][0])
    
    print("Models loaded in. Starting hstack conversion")
    
    predictions = np.hstack((np.array(predictions1)[:, np.newaxis], 
                             np.array(predictions2)[:, np.newaxis], 
                             np.array(predictions3)[:, np.newaxis]))
    
    del predictions1, predictions2, predictions3
    
    X_train, X_val, y_train, y_val = train_test_split(predictions, normalized_y_white, test_size=0.2, random_state=42)

    print("Preprocessing done. Commencing training of model.")
    
    activation_layer = layers.LeakyReLU(0.1)

    input1 = layers.Input(shape=(3,))
    Dl1 = layers.Dense(1024, activation=activation_layer, kernel_regularizer=regularizers.l2(0.001))(input1)
    Dl2 = layers.Dense(512, activation=activation_layer, kernel_regularizer=regularizers.l2(0.001))(Dl1)
    Dl3 = layers.Dense(256, activation=activation_layer, kernel_regularizer=regularizers.l2(0.001))(Dl2)
    Dl4 = layers.Dense(256, activation=activation_layer, kernel_regularizer=regularizers.l2(0.001))(Dl3)
    output = layers.Dense(1, activation="linear")(Dl4)

    ensemble = models.Model(inputs=input1, outputs=output)

    def custom_loss(y_true, y_pred):
        diff = y_pred - y_true
        squared_diff = tf.square(diff)
        weighted_squared_diff = tf.exp(tf.abs(y_true) / 1000) * squared_diff
        return tf.reduce_mean(weighted_squared_diff)

    opt = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    ensemble.compile(optimizer=opt,
                    loss=custom_loss,
                    metrics=[metrics.MeanAbsoluteError()])

    ensemble.summary()

    early_stopping = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=7, restore_best_weights=True)
    reduceLr = callbacks.ReduceLROnPlateau(monitor='mean_absolute_error', patience=2, factor=0.1)

    history = ensemble.fit(X_train, y_train, epochs=1000, batch_size=2048, shuffle=True, callbacks=[early_stopping, reduceLr])

    loss, metric = ensemble.evaluate(x=X_val, y=y_val)
    print(f"Validation Loss: {loss}")
    print(f"Validation Metric: {metric}")
    
    ensemble.export(R"Chess_White_Ensemble")

    # Extract loss values from the history object
    epochs = range(1, len(history.history['loss']) + 1)
    loss = history.history['loss']
    val_loss = history.history.get('val_loss', [])  # Get val_loss if available

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Training Loss', color='blue')
    if val_loss:
        plt.plot(epochs, val_loss, label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(R'Chess_White_3/loss_plot_white_3.png')

    # Save the training history to a JSON file
    with open(R'Chess_White_3/training_history_white_3.json', 'w') as file:
        json.dump(history.history, file)

    plt.show()
        
        
        
        
        
        