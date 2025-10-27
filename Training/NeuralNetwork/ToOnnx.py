import tensorflow as tf
import tf2onnx
import os
from keras.models import load_model
from EnsembleNetwork import UnscaleLayer  

custom_objects = {"UnscaleLayer": UnscaleLayer}

def convert_model_to_onnx(keras_model_path, onnx_model_path, input_signature):
    print(f"Loading Keras model: {keras_model_path}")
    model = load_model(keras_model_path, custom_objects=custom_objects, safe_mode=False)

    print("Converting to ONNX...")
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13
    )

    print(f"Saving ONNX model to: {onnx_model_path}")
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("Conversion complete.\n")

def main():
    os.makedirs("onnx_models", exist_ok=True)

    input_signature = [tf.TensorSpec([None, 8, 8, 16], tf.float32, name="input")]

    convert_model_to_onnx(
        "Chess_Combined_Quick/ensemble_white.keras",
        "onnx_models/ensemble_white.onnx",
        input_signature
    )

    convert_model_to_onnx(
        "Chess_Combined_Quick/ensemble_black.keras",
        "onnx_models/ensemble_black.onnx",
        input_signature
    )

if __name__ == "__main__":
    main()
