import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
import os

# Code that allows for this file system to work:
def get_path(script_path, relative_path):
    script_dir = os.path.dirname(script_path)
    neuralnetwork_dir = os.path.dirname(script_dir)
    return os.path.join(neuralnetwork_dir, relative_path)

# Takes on average 15 minutes to process.
total_number = 12956364 # Number of matrices in the file
white_matrices_total = 6473070
black_matrices_total = 6483294
file_name_white = get_path(__file__, R"CSVFiles/WhiteFinal.csv")
file_name_black = get_path(__file__, R"CSVFiles/BlackFinal.csv")
number_matrixes_white = 6473070    # Specify the number of matrices you want to read
number_matrixes_black = 6483294    # Specify the number of matrices you want to read

def read_evaluations(file_name, number):
    matrices = []

    # Read the CSV file
    data = pd.read_csv(file_name, usecols=[1], header=None, nrows=number)

    # Iterate over each row
    for row in data.itertuples(index=False):
        value = int(row[0])
        matrices.append(value)

    return matrices

if __name__ == "__main__":
    matrices_white = read_evaluations(file_name_white, number_matrixes_white)

    y_white = np.array(matrices_white).reshape(-1, 1)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_white = scaler.fit_transform(y_white).flatten()

    joblib.dump(scaler, get_path(__file__, "Scalers/ScalerWhite.pkl"))

    matrices_black = read_evaluations(file_name_black, number_matrixes_black)

    y_black = np.array(matrices_black).reshape(-1, 1)

    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_black = scaler.fit_transform(y_black).flatten()

    joblib.dump(scaler, get_path(__file__, "Scalers/ScalerBlack.pkl"))
