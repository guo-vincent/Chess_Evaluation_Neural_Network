# e.g. python test_network.py --model_key white_quick_1
import os
import sys
import argparse

# ─────────────────── Ensure we can import from project_root ───────────────────
this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ─────────────────── Imports ───────────────────
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import joblib

from Training.NeuralNetwork.CNN import read_matrix, enhance_board
from Training.NeuralNetwork.config import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(
        description="Load a pretrained CNN (from MODEL_CONFIGS) "
                    "and compare its predictions against true evals."
    )
    parser.add_argument("--model_key", type=str, required=True)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    key = args.model_key
    if key not in MODEL_CONFIGS:
        print(f"Error: `{key}` not found in MODEL_CONFIGS. Available keys:")
        for k in MODEL_CONFIGS:
            print("  ", k)
        sys.exit(1)

    cfg = MODEL_CONFIGS[key]

    # ─────────────────── Paths ───────────────────
    csv_rel = cfg["data_path"]
    csv_full = os.path.join(project_root, "ChessCpp", "NeuralNetwork", csv_rel)
    model_dir = os.path.join(project_root, "ChessCpp", "NeuralNetwork", cfg["save_path"])
    model_path = os.path.join(model_dir, "final_model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    n_samples = min(args.test_samples, cfg.get("samples", args.test_samples))

    if not os.path.isfile(model_path):
        print(f"Error: Cannot find `{model_path}`. Did you train `{key}` already?")
        sys.exit(1)

    print(f"\n→ Loading model `{key}` from:\n    {model_path}")
    model = tf.keras.models.load_model(
        model_path
    )

    # ─────────────────── Load boards and true evals ───────────────────
    print(f"→ Reading {n_samples} boards from:\n    {csv_full}")
    raw_boards, true_evals = read_matrix(csv_full, n_samples, progress=False)

    X = np.stack([enhance_board(b) for b in raw_boards], axis=0)

    print("→ Running model.predict(...)")
    preds_scaled = model.predict(X, batch_size=32).flatten()

    # ─────────────────── Load and apply scaler ───────────────────
    if not os.path.isfile(scaler_path):
        print(f"⚠️  Scaler not found at: {scaler_path}")
        print("    Using raw prediction values.")
        y_pred = preds_scaled
    else:
        print(f"→ Loading scaler from:\n    {scaler_path}")
        scaler = joblib.load(scaler_path)
        y_pred = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    y_true = true_evals

    # ─────────────────── Output ───────────────────
    if not args.no_plot:
        plt.figure(figsize=(8, 6))
        plt.plot(y_true, label="True Evaluation", marker='o', linestyle='-')
        plt.plot(y_pred, label="Predicted Eval", marker='x', linestyle='--')
        plt.title(f"Predictions vs. True Evaluations ({key}, first {n_samples})")
        plt.xlabel("Index (0 to {})".format(n_samples - 1))
        plt.ylabel("Evaluation (original scale)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        mae = np.mean(np.abs(y_pred - y_true))
        print(f"\n   Sample MAE over {n_samples} boards (inverse-scaled): {mae:.4f}")

    print("\nDone.")
    
    print("Scaled preds (first 10):", preds_scaled[:10])
    print("→ Inverse-scaled (first 10):", 
        scaler.inverse_transform(preds_scaled[:10].reshape(-1,1)).flatten())
    print("True evals (first 10)      :", true_evals[:10])


if __name__ == "__main__":
    main()
