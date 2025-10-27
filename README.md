# Chess_Evaluation_Neural_Network

A chess engine that combines classical algorithms with a convolutional neural network (CNN) to evaluate chess positions.

> **Note:** This project was primarily developed and tested on Windows. Cross-platform compatibility (Mac or Linux) is not guaranteed.

## Setup

### C++ Dependencies

Update the submodules:

```bash
git submodule update --init --recursive
```

Create a .env file with the ONNXRUNTIME_ROOT variable pointing to your ONNX Runtime installation.
Example for Windows:
ONNXRUNTIME_ROOT=C:/onnxruntime-win-x64-1.21.0

## Python Dependencies

Python version: 3.8.10
Deactivate any other environment managers (e.g. conda) before downloading requirements.txt
Install required packages:

```bash
pip install -r requirements.txt
```

## Training the model

**If you only want to use the pre-trained models in this repository, you can skip this section.**

There already exists processed data in the form of a zip file. Unzip both files and start training right away.

Alternatively:

### 1. Download Training Data

Download chessData.csv from the Kaggle dataset:
<https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations/data>

Only chessData.csv is needed. The other CSV files can be deleted after extraction. Move chessData.csv to the following directory:

```swift
Training/NeuralNetwork/CSVFiles
```

### 2. Process the Data

Compile and run ProcessFile.cpp to convert chessData.csv into model-readable files.
This will generate two files in the CSVFiles folder: White.csv and Black.csv.
Processing usually takes ~10 minutes.

### 3. Train Submodels

Use CNN.py to train the submodels (3 submodels per side):

```bash
python CNN.py --model_key <model_name>
```

<model_name> is defined in config.py. Example:

```bash
python CNN.py --model_key white_quick_1
```

### 4. Train the Ensemble Model

The ensemble model combines the submodels:

```bash
python EnsembleNetwork.py --model_key <model_name>
```

Example:

```bash
python EnsembleNetwork.py --model_key combined_quick_enqueue
```

**Note: If you increase the number of submodels, you may need to adjust the code accordingly.**

### 5. Adjust Hyperparameters

Modify config.py (located inside the NeuralNetwork directory) to change learning rates, batch sizes, or other model parameters.

### 6. Convert Models to ONNX

Once training is complete, convert the models to .onnx format:

```bash
python ToOnnx.py
```

## Running the Program

To launch the chess engine UI:

```bash
python build.py --run
```

This should start the interactive interface.

### Notes

- Project tested on Windows only.
- Pre-trained models are included; training from scratch is optional.
- Hyperparameter tuning and model customization are controlled via config.py.