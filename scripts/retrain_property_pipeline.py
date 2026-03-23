"""Run property model retraining with validation-driven model selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retraining import select_best_trial
from models.embedding_model import build_embedding_model
from scripts.train_property_numpy import (
    fit_ridge_ensemble,
    fit_ridge_regression,
    load_dataset,
    predict_linear_ensemble,
    predict_linear,
    split_train_val,
    split_train_val_with_indices,
)
from utils.calibration import regression_ece
from utils.metrics import evaluate_property_metrics


def parse_alpha_grid(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("ridge alpha grid is empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPLE property retraining pipeline")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/property_retrain")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/property_linear_best.npz")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-backend", type=str, default="random")
    parser.add_argument("--embedding-model-id", type=str, default="")
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-pooling", type=str, default="mean")
    parser.add_argument("--disable-embedding-mock-fallback", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", type=str, default="random", help="Split strategy: random|scaffold")
    parser.add_argument("--scaffold-k", type=int, default=3, help="Scaffold key size for split_mode=scaffold")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Number of bootstrap ensemble members")
    parser.add_argument("--ece-bins", type=int, default=10, help="Bin count for regression ECE")
    parser.add_argument(
        "--val-index-file",
        type=str,
        default="",
        help="Optional JSON file with fixed validation indices",
    )
    parser.add_argument(
        "--ridge-alphas",
        type=str,
        default="1e-4,1e-3,1e-2,1e-1",
        help="Comma-separated ridge alpha candidates",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = Path(args.checkpoint_out)
    if not checkpoint_out.is_absolute():
        checkpoint_out = ROOT / checkpoint_out
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    sequences, targets = load_dataset(data_path)
    val_index_file = str(args.val_index_file).strip()
    if val_index_file:
        val_index_path = Path(val_index_file)
        if not val_index_path.is_absolute():
            val_index_path = ROOT / val_index_path
        split_payload = json.loads(val_index_path.read_text(encoding="utf-8"))
        val_idx = np.asarray(split_payload.get("val_indices", []), dtype=np.int64)
        if val_idx.size == 0:
            raise ValueError("val-index-file has empty val_indices")
        n = len(sequences)
        mask = np.ones((n,), dtype=bool)
        mask[val_idx] = False
        train_idx = np.arange(n, dtype=np.int64)[mask]
        train_sequences, train_targets, val_sequences, val_targets = split_train_val_with_indices(
            sequences,
            targets,
            train_idx=train_idx,
            val_idx=val_idx,
        )
    else:
        train_sequences, train_targets, val_sequences, val_targets = split_train_val(
            sequences,
            targets,
            val_ratio=float(args.val_ratio),
            seed=int(args.split_seed),
            split_mode=str(args.split_mode),
            scaffold_k=int(args.scaffold_k),
        )
    embedder = build_embedding_model(
        backend=str(args.embedding_backend),
        embedding_dim=int(args.embedding_dim),
        model_id=(str(args.embedding_model_id).strip() or None),
        device=str(args.embedding_device),
        pooling=str(args.embedding_pooling),
        allow_mock=(not args.disable_embedding_mock_fallback),
    )
    resolved_embedding_dim = int(embedder.embedding_dim)
    train_features = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
    val_features = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)

    alpha_grid = parse_alpha_grid(args.ridge_alphas)
    trials: list[dict] = []
    ensemble_size = max(1, int(args.ensemble_size))

    for alpha in alpha_grid:
        if ensemble_size > 1:
            weights, bias = fit_ridge_ensemble(
                train_features,
                train_targets,
                ridge_alpha=float(alpha),
                ensemble_size=ensemble_size,
                seed=int(args.split_seed),
            )
            val_preds, _ = predict_linear_ensemble(val_features, weights, bias)
        else:
            weights, bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=float(alpha))
            val_preds = predict_linear(val_features, weights, bias)
        val_metrics = evaluate_property_metrics(val_targets, val_preds)
        trial = {
            "ridge_alpha": float(alpha),
            "val_mean_rmse": float(val_metrics["mean"]["rmse"]),
            "val_mean_mae": float(val_metrics["mean"]["mae"]),
            "val_mean_r2": float(val_metrics["mean"]["r2"]),
            "val_mean_pearson": float(val_metrics["mean"]["pearson"]),
        }
        trials.append(trial)

    best_trial = select_best_trial(trials)
    best_alpha = float(best_trial["ridge_alpha"])
    if ensemble_size > 1:
        best_weights, best_bias = fit_ridge_ensemble(
            train_features,
            train_targets,
            ridge_alpha=best_alpha,
            ensemble_size=ensemble_size,
            seed=int(args.split_seed),
        )
        train_preds, train_unc = predict_linear_ensemble(train_features, best_weights, best_bias)
        val_preds, val_unc = predict_linear_ensemble(val_features, best_weights, best_bias)
    else:
        best_weights, best_bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=best_alpha)
        train_preds = predict_linear(train_features, best_weights, best_bias)
        val_preds = predict_linear(val_features, best_weights, best_bias)
        train_unc = np.zeros((train_preds.shape[0],), dtype=np.float32)
        val_unc = np.zeros((val_preds.shape[0],), dtype=np.float32)

    train_metrics = evaluate_property_metrics(train_targets, train_preds)
    val_metrics = evaluate_property_metrics(val_targets, val_preds)
    train_calibration = regression_ece(train_targets, train_preds, train_unc, num_bins=int(args.ece_bins))
    val_calibration = regression_ece(val_targets, val_preds, val_unc, num_bins=int(args.ece_bins))

    np.savez(
        checkpoint_out,
        model_type=("numpy_linear_ensemble" if ensemble_size > 1 else "numpy_linear"),
        embedding_dim=np.int32(resolved_embedding_dim),
        embedding_backend=np.array(str(args.embedding_backend)),
        embedding_model_id=np.array(str(args.embedding_model_id).strip()),
        embedding_pooling=np.array(str(args.embedding_pooling)),
        weights=best_weights,
        bias=best_bias,
    )

    report = {
        "dataset": str(data_path),
        "split": {
            "train_count": int(train_targets.shape[0]),
            "val_count": int(val_targets.shape[0]),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
            "val_index_file": (val_index_file or None),
        },
        "search_space": {"ridge_alphas": alpha_grid},
        "embedding": {
            "embedding_dim": resolved_embedding_dim,
            "embedding_backend": str(args.embedding_backend),
            "embedding_model_id": str(args.embedding_model_id).strip() or None,
            "embedding_pooling": str(args.embedding_pooling),
            "ensemble_size": int(ensemble_size),
        },
        "trials": trials,
        "best": {
            "ridge_alpha": best_alpha,
            "checkpoint": str(checkpoint_out),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "train_calibration": train_calibration,
            "val_calibration": val_calibration,
        },
    }
    report_path = output_dir / "retrain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved best checkpoint: {checkpoint_out}")
    print(f"Saved retrain report: {report_path}")
    print(f"Best ridge alpha: {best_alpha}")
    print(f"Best validation mean RMSE: {val_metrics['mean']['rmse']:.4f}")


if __name__ == "__main__":
    main()
