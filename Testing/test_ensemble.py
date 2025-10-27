import os
import sys
import argparse

this_file   = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.layers import Layer

from sklearn.metrics import mean_absolute_error, mean_squared_error

from Training.NeuralNetwork.CNN import read_matrix, enhance_board
from Training.NeuralNetwork.config import MODEL_CONFIGS

class UnscaleLayer(Layer):
    def __init__(self, scale, center, **kwargs):
        super(UnscaleLayer, self).__init__(**kwargs)
        self.scale = np.array(scale, dtype=np.float32)
        self.center = np.array(center, dtype=np.float32)

    def build(self, input_shape):
        self.scale_tensor = self.add_weight(
            name="scale", shape=self.scale.shape, trainable=False,
            initializer=tf.constant_initializer(self.scale)
        )
        self.center_tensor = self.add_weight(
            name="center", shape=self.center.shape, trainable=False,
            initializer=tf.constant_initializer(self.center)
        )

    def call(self, inputs):
        return inputs * self.scale_tensor + self.center_tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale.tolist(),
            "center": self.center.tolist(),
        })
        return config

def main():
    p = argparse.ArgumentParser(
        description="Test a *frozen ensemble* (white or black) on held-out data"
    )
    p.add_argument("--model_key", choices=["white_ensemble", "black_ensemble"],
                   required=True,
                   help="Which ensemble to test: white_ensemble or black_ensemble")
    p.add_argument("--test_samples", type=int, default=500,
                   help="How many positions to read from the CSV")
    p.add_argument("--no_plot", action="store_true",
                   help="Skip plotting; print MAE instead")
    args = p.parse_args()

    cfg = MODEL_CONFIGS["combined_quick_enqueue"]
    color = args.model_key.split("_")[0]  # 'white' or 'black'

    # ── locate model & scaler ───────────────────────────────────────────────
    model_fname  = f"ensemble_{color}.keras"
    scaler_fname = f"ensemble_{color}_target_scaler.pkl"

    model_path = os.path.join(
        project_root,
        "ChessCpp", "NeuralNetwork",
        cfg["save_path"],
        model_fname
    )
    scaler_path = os.path.join(
        project_root,
        "ChessCpp", "NeuralNetwork",
        cfg["save_path"],
        scaler_fname
    )

    if not os.path.isfile(model_path):
        print(f"Could not find ensemble model at `{model_path}`")
        sys.exit(1)

    # ── load the ensemble (with its UnscaleLayer) ───────────────────────────
    print(f"Loading ensemble from:\n    {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"UnscaleLayer": UnscaleLayer},
        safe_mode=False
    )

    # ── load the target‐scaler ───────────────────────────────────────────────
    if os.path.isfile(scaler_path):
        print(f"Loading target-scaler from:\n    {scaler_path}")
        target_scaler = joblib.load(scaler_path)
    else:
        print(f"No target-scaler found at `{scaler_path}`; will skip inverse-scaling.")
        target_scaler = None

    # ── read & preprocess test positions ─────────────────────────────────────
    csv_rel = cfg[f"{color}_data_path"]
    data_csv = os.path.join(
        project_root,
        "ChessCpp", "NeuralNetwork",
        csv_rel
    )
    if not os.path.isfile(data_csv):
        print(f"Cannot find CSV at `{data_csv}`")
        sys.exit(1)

    print(f"Reading first {args.test_samples} boards from:\n    {data_csv}")
    raw_boards, y_true = read_matrix(data_csv, args.test_samples, progress=False)
    
    X_test = np.stack([enhance_board(b) for b in raw_boards], axis=0)

    # ── predict ──────────────────────────────────────────────────────────────
    print("Running ensemble.predict(...)")
    y_pred_scaled = model.predict(X_test, batch_size=32).flatten()

    # ── inverse‐scale ────────────────────────────────
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
    else:
        y_pred = y_pred_scaled

    # ── metrics & output ─────────────────────────────────────────────────────
    mae = np.mean(np.abs(y_pred - y_true))
    print(f"\nSample MAE over {len(y_true)} positions: {mae:.4f} centipawns\n")

    if not args.no_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(y_true, label="True Eval", marker='o')
        plt.plot(y_pred, label="Predicted Eval", marker='x')
        plt.title(f"{color.title()} Ensemble: True vs Predicted (n={len(y_true)})")
        plt.xlabel("Sample index")
        plt.ylabel("Evaluation (cp)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("Skipped plotting (use `--no_plot` to print instead)")
        
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"MAE: {mae:.1f} cp,  RMSE: {np.sqrt(mse):.1f} cp")
    
    resid = y_pred - y_true
    plt.scatter(y_true, resid, alpha=0.3)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("True eval (cp)")
    plt.ylabel("Prediction error (cp)")
    plt.title("Residual vs. True")
    plt.show()

if __name__ == "__main__":
    main()
