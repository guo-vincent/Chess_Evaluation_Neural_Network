import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import joblib

total_number = 12956364 # Number of matrices in the file
white_matrices_total = 6473070
black_matrices_total = 6483294
file_name_white = R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\CSVFiles\White.csv"
file_name_black = R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\CSVFiles\Black.csv"
number_matrixes_white = 100000 #6473070    # Specify the number of matrices you want to read
number_matrixes_black = 6483294    # Specify the number of matrices you want to read

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
    print("Done!")

    # Extract matrices and evaluations
    X_white = np.array([matrix/100 for matrix, _ in matrices_white])
    y_white = np.array([evaluation for _, evaluation in matrices_white])

    # Convert evaluations to numpy array
    y_white = y_white.reshape(-1, 1)

    # Normalize the evaluations using MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_white = scaler.fit_transform(y_white).flatten()

    model1 = tf.saved_model.load(R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\Chess_White")
    model2 = tf.saved_model.load(R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\Chess_White_2")

    # Create a residual plot
    Predictions = []
    infer2 = model2.signatures['serving_default']
    scaler = joblib.load(R"C:\Chess_Engine\chess-engine\ChessCpp\NeuralNetwork\Scalers\ScalerWhite.pkl")
    for matrix in X_white:
        matrix = np.expand_dims(matrix, axis=-1)  # Shape (8, 8, 1)
        matrix = np.expand_dims(matrix, axis=0)   # Shape (1, 8, 8, 1)
        tensor_matrix = tf.convert_to_tensor(matrix)
        output_array = infer2(tensor_matrix)['output_0']
        inverse_transformed_output = scaler.inverse_transform(output_array)
        Predictions.append(inverse_transformed_output[0][0])

    # Residuals = [normalized_y_white[n] - Predictions[n] for n in range(len(normalized_y_white))]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_white, Predictions, s=1, alpha=0.5)

    # Draw the y = x line
    min_value = min(min(y_white), min(Predictions))
    max_value = max(max(y_white), max(Predictions))
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', linewidth=2, label="y = x")
    plt.xlabel("True Value")
    plt.ylabel("Estimated Model 1 Predictions")
    plt.title("True Value vs Model Prediction")
    plt.legend()
    plt.show()


