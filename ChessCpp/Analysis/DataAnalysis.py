import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import kstest, kurtosis, skew
from heapq import nlargest
from sklearn import preprocessing
import os

# Code that allows for this file system to work:
def get_path(script_file, filename):
    """
    Returns the full path to a file located in the 'CSVfiles' directory relative to the script file.

    :param script_file: Path to the current script file
    :param filename: Name of the file to locate in the 'CSVfiles' directory
    :return: Full path to the specified file
    """
    script_dir = os.path.dirname(script_file)
    chesscpp_dir = os.path.dirname(script_dir)
    csv_dir = os.path.join(chesscpp_dir, 'NeuralNetwork', 'CSVFiles')
    return os.path.join(csv_dir, filename)

total_number = 12956364 # Number of matrices in the file
white_matrices_total = 6473070
black_matrices_total = 6483294
file_name_white = get_path(__file__, "WhiteFinal.csv")
file_name_black = get_path(__file__, "BlackFinal.csv")
number_matrixes_white = 6473070    # Specify the number of matrices you want to read
number_matrixes_black = 6483294#5  # Specify the number of matrices you want to read

min_eval_value = 999999
max_eval_value = -999999

def read_evaluations(file_name, number):
    global min_eval_value
    global max_eval_value

    matrices = []

    # Read the CSV file
    data = pd.read_csv(file_name, usecols=[1], header=None, nrows=number, skiprows=1)

    # Iterate over each row
    for row in data.itertuples(index=False):
        value = int(row[0])
        matrices.append(value)
        
        if value < min_eval_value:
            min_eval_value = value
        elif value > max_eval_value:
            max_eval_value = value

    return matrices

if __name__ == "__main__":
    # Read evaluations
    matrices_white = read_evaluations(file_name_black, number_matrixes_black)
    print("Done!")

    # Convert evaluations to numpy array
    y_white = np.array(matrices_white).reshape(-1, 1)

    # Normalize the evaluations using MinMaxScaler
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    normalized_y_white = scaler.fit_transform(y_white).flatten()

    normalized_y_white_list = normalized_y_white.tolist()
    eval_counts = Counter(normalized_y_white_list)

    # Prepare data for scatter plot
    eval_values = np.array(list(eval_counts.keys()))
    eval_frequencies = np.array(list(eval_counts.values()))

    print(f"Min_value: {min_eval_value}")
    print(f"Max_value: {max_eval_value}")
    print(f"Mean: {eval_values.mean()}")
    print(f"Standard Deviation: {eval_values.std()}")
    print(f"Kurtosis: {kurtosis(normalized_y_white_list)}")
    print(f"Skew: {skew(normalized_y_white_list)}")

    mu, std = np.mean(normalized_y_white), np.std(normalized_y_white)
    ks_statistic, p_value = kstest(normalized_y_white, 'norm', args=(mu, std))

    print(f"KS Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(eval_values, eval_frequencies, s=1, alpha=0.5)
    plt.xlabel('Evaluation Value')
    plt.ylabel('Frequency')
    plt.title('Scatter Plot of Evaluation Values')
    plt.grid(True)
    plt.show()

    res = nlargest(5, eval_counts, key=eval_counts.get)
    for n in res:
        print(f"Most common values: {n}")
        print(f"Frequency of {n}: {eval_counts[n]}")
    print('\n')

    for a in range(1, 11):
        greater_than_n = [n > a for n in eval_counts.values()]
        print(f"Total Unique Evaluations: {len(eval_counts)}")
        print(f"Evaluations with greater than {a} Examples: {sum(greater_than_n)}")


"""
White
Min_eval: -1694
Max_eval:  1736

Outliers in frequency:
Most common values: 0
Frequency of 0: 671910
Most common values: -13
Frequency of -13: 81919
Most common values: 13
Frequency of 13: 78768
Relatively Normal:
Most common values: -30
Frequency of -30: 19866
Most common values: -38
Frequency of -38: 19408

Black
Min_eval: -2241
Max_eval:  1712
"""