import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

def read_matrix(file_name, number_matrixes):
    # Rows to read: header + (number of matrices * 9 rows per matrix)
    rows_to_read = number_matrixes * 9+1

    data = pd.read_csv(file_name, nrows=rows_to_read)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        # Extract the board matrix and evaluation from the DataFrame
        board_matrix = data.iloc[start_row:start_row+8, :-1].to_numpy(dtype=np.float32)

        matrices.append(board_matrix)
    
    return matrices

# Code that allows for this file system to work:
def get_path(script_path, relative_path):
    script_dir = os.path.dirname(script_path)
    neuralnetwork_dir = os.path.dirname(script_dir)
    return os.path.join(neuralnetwork_dir, relative_path)

if __name__ == "__main__":
    matrices_white = read_matrix(get_path(__file__, "CSVFiles/test.csv"), 1)
    print(matrices_white[0])

    # Load the SavedModel
    model = tf.saved_model.load(get_path(__file__, "Chess_White_2"))

    # Add channel dimension (1)
    input_matrix = np.expand_dims(matrices_white[0]/100, axis=-1)  # Shape (8, 8, 1)

    # Add batch dimension (1)
    input_matrix = np.expand_dims(input_matrix, axis=0)   # Shape (1, 8, 8, 1)

    # Prepare the input data for inference
    input_tensor = tf.convert_to_tensor(input_matrix)

    # Access the default serving function
    infer = model.signatures['serving_default']

    print("Model input signature:")
    for key, value in infer.structured_input_signature[1].items():
        print(f"{key}: {value}")

    # Perform inference
    results = infer(input_tensor)
    scaler = joblib.load(get_path(__file__, "Scalers/ScalerWhite.pkl"))
    output_tensor = results['output_0'] 

    # Convert the tensor to a NumPy array
    output_array = output_tensor.numpy()

    # Apply the inverse transformation using the scaler
    inverse_transformed_output = scaler.inverse_transform(output_array)

    print(inverse_transformed_output[0][0])

"""
Model 1:
Chess::Board Fen: 4Qrq1/4N3/8/RK2P3/Bn2k3/1p2N1PP/6p1/2n5 w - - 0 1
Predicted: 602.52045
Expected: 2218.2622

Chess::Board Fen: r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1
Predicted: 74.52284
Expected: 51.300674

Chess::Board Fen: rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1
Predicted: 55.92586
Expected: 141.26765

Model 2:
Chess::Board Fen: 4Qrq1/4N3/8/RK2P3/Bn2k3/1p2N1PP/6p1/2n5 w - - 0 1
Predicted: 3100.3223
Expected: 2218.2622

Chess::Board Fen: r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 0 1
Predicted: 74.372444
Expected: 51.300674

Chess::Board Fen: rnbqkbnr/pppp1p1p/6p1/4p2Q/4P3/8/PPPP1PPP/RNB1KBNR w KQkq - 0 1
Predicted: 63.715527
Expected: 141.26765
"""
