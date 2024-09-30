import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

def read_matrix(file_name, number_matrixes=1):
    # Calculate number of rows to read (8 rows per matrix + 1 row for evaluation and side)
    rows_to_read = number_matrixes * 9

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_name, nrows=rows_to_read, header=0)

    matrices = []
    for k in range(number_matrixes):
        start_row = k * 9

        # Extract the board matrix (first 8 rows, excluding the "Evaluation" and "Side" columns)
        # Evaluation is an strictly unnessecary column, that was left over from model training. 
        #  
        board_matrix = data.iloc[start_row:start_row+8, :-2].to_numpy(dtype=np.float32)

        matrices.append(board_matrix)

    # Extract the "Side" value from the last row of the last matrix
    side_value = data.iloc[rows_to_read - 1, -1]
    
    return matrices, side_value

def white_model(filepath, number_to_read=1):
    matrices, side_value = read_matrix(filepath, number_to_read)
    
    # Return nothing if side is black
    if side_value != 'White':
        return None

    matrix = matrices[0]

    # Load the SavedModel
    model = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_White")
    model2 = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_White_2")
    model3 = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_White_3")
    meta_model = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_White_Ensemble")

    # Add channel dimension (1)
    input_matrix = np.expand_dims(matrix/100, axis=-1)  # Shape (8, 8, 1). Note that values are scaled down by 100. 

    # Add batch dimension (1)
    input_matrix = np.expand_dims(input_matrix, axis=0)   # Shape (1, 8, 8, 1)

    # Prepare the input data for inference
    input_tensor = tf.convert_to_tensor(input_matrix)

    # Access the default serving function
    infer = model.signatures['serving_default']
    infer2 = model2.signatures['serving_default']
    infer3 = model3.signatures['serving_default']
    meta_infer = meta_model.signatures['serving_default']

    # Perform inference
    results = infer(input_tensor)
    results2 = infer2(input_tensor)
    results3 = infer3(input_tensor)
    meta_results = meta_infer(tf.convert_to_tensor([[results['output_0'][0][0], results2['output_0'][0][0], results3['output_0'][0][0]]]))
    
    scaler = joblib.load("ChessCpp/NeuralNetwork/Scalers/ScalerWhite.pkl")
    output_tensor = meta_results['output_0'] 

    inverse_transformed_output = scaler.inverse_transform(output_tensor.numpy())

    return (inverse_transformed_output[0][0])

def black_model(filepath, number_to_read=1):
    matrices, side_value = read_matrix(filepath, number_to_read)
    
    # Return nothing if side is White
    if side_value != 'Black':
        return None

    matrix = matrices[0]

    # Load the SavedModel
    model = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_Black")
    model2 = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_Black_2")
    model3 = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_Black_3")
    meta_model = tf.saved_model.load("ChessCpp/NeuralNetwork/Chess_Black_Ensemble")

    # Add channel dimension (1)
    input_matrix = np.expand_dims(matrix/100, axis=-1)  # Shape (8, 8, 1). Note that values are scaled down by 100. 

    # Add batch dimension (1)
    input_matrix = np.expand_dims(input_matrix, axis=0)   # Shape (1, 8, 8, 1)

    # Prepare the input data for inference
    input_tensor = tf.convert_to_tensor(input_matrix)

    # Access the default serving function
    infer = model.signatures['serving_default']
    infer2 = model2.signatures['serving_default']
    infer3 = model3.signatures['serving_default']
    meta_infer = meta_model.signatures['serving_default']

    # Perform inference
    results = infer(input_tensor)
    results2 = infer2(input_tensor)
    results3 = infer3(input_tensor)
    meta_results = meta_infer(tf.convert_to_tensor([[results['output_0'][0][0], results2['output_0'][0][0], results3['output_0'][0][0]]]))
    
    scaler = joblib.load("ChessCpp/NeuralNetwork/Scalers/ScalerBlack.pkl")
    output_tensor = meta_results['output_0'] 

    inverse_transformed_output = scaler.inverse_transform(output_tensor.numpy())

    return (inverse_transformed_output[0][0])

if __name__ == "__main__":
    white_result = white_model("/Users/vincentguo/Chess_Engine/Chess_Evaluation_Neural_Network/ChessCpp/NeuralNetwork/CSVFiles/test.csv")
    if white_result:
        print("White model evaluation:", white_result)

    black_result = black_model("/Users/vincentguo/Chess_Engine/Chess_Evaluation_Neural_Network/ChessCpp/NeuralNetwork/CSVFiles/test.csv")
    if black_result:
        print("Black model evaluation:", black_result)
