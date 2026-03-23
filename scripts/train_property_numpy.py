"""Train a lightweight property model from labeled CSV and export NPZ checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.embedding_model import RandomEmbeddingModel



def load_dataset(csv_path: Path) -> tuple[list[str], np.ndarray]:
    sequences: list[str] = []
    targets: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sequence", "stability", "activity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Dataset must contain columns: sequence, stability, activity")

        for row in reader:
            seq = str(row["sequence"]).strip()
            if not seq:
                continue
            sequences.append(seq)
            targets.append([float(row["stability"]), float(row["activity"])])

    if not sequences:
        raise ValueError("No valid rows found in dataset")

    return sequences, np.asarray(targets, dtype=np.float32)



def fit_ridge_regression(features: np.ndarray, targets: np.ndarray, ridge_alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit multivariate linear regression with L2 regularization."""
    n_samples, n_features = features.shape
    x_aug = np.concatenate([features, np.ones((n_samples, 1), dtype=np.float32)], axis=1)

    eye = np.eye(n_features + 1, dtype=np.float32)
    eye[-1, -1] = 0.0

    lhs = x_aug.T @ x_aug + ridge_alpha * eye
    rhs = x_aug.T @ targets

    params = np.linalg.solve(lhs, rhs)
    weights = params[:-1, :]
    bias = params[-1, :]
    return weights.astype(np.float32), bias.astype(np.float32)



def evaluate_rmse(features: np.ndarray, targets: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    preds = features @ weights + bias
    mse = np.mean((preds - targets) ** 2, axis=0)
    return np.sqrt(mse)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAPLE NumPy property predictor")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv", help="CSV dataset path")
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/property_linear.npz",
        help="Output NPZ checkpoint path",
    )
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--ridge-alpha", type=float, default=1e-3, help="L2 regularization strength")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    sequences, targets = load_dataset(data_path)
    embedder = RandomEmbeddingModel(embedding_dim=args.embedding_dim)
    features = np.stack([embedder.encode(seq) for seq in sequences]).astype(np.float32)

    weights, bias = fit_ridge_regression(features, targets, ridge_alpha=float(args.ridge_alpha))
    rmse = evaluate_rmse(features, targets, weights, bias)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        model_type="numpy_linear",
        embedding_dim=np.int32(args.embedding_dim),
        weights=weights,
        bias=bias,
    )

    print(f"Saved checkpoint: {output_path}")
    print(f"Train RMSE stability={rmse[0]:.4f}, activity={rmse[1]:.4f}")


if __name__ == "__main__":
    main()
