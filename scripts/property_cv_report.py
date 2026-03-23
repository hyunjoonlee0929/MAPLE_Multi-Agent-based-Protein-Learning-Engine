"""Run repeated split-seed validation and export reproducibility report."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.retraining import select_best_trial
from models.embedding_model import build_embedding_model
from scripts.retrain_property_pipeline import parse_alpha_grid
from scripts.train_property_numpy import (
    fit_ridge_ensemble,
    fit_ridge_regression,
    load_dataset,
    predict_linear_ensemble,
    predict_linear,
    split_train_val,
)
from utils.calibration import regression_ece
from utils.metrics import evaluate_property_metrics


def parse_seed_list(text: str) -> list[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("split seed list is empty")
    return values


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Property model cross-seed reproducibility report")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-backend", type=str, default="random")
    parser.add_argument("--embedding-model-id", type=str, default="")
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-pooling", type=str, default="mean")
    parser.add_argument("--disable-embedding-mock-fallback", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seeds", type=str, default="1,7,13,21,42")
    parser.add_argument("--split-mode", type=str, default="random", help="Split strategy: random|scaffold")
    parser.add_argument("--scaffold-k", type=int, default=3, help="Scaffold key size for split_mode=scaffold")
    parser.add_argument("--ensemble-size", type=int, default=1, help="Number of bootstrap ensemble members")
    parser.add_argument("--ece-bins", type=int, default=10, help="Bin count for regression ECE")
    parser.add_argument("--ridge-alphas", type=str, default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--output-dir", type=str, default="outputs/property_cv")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    split_seeds = parse_seed_list(args.split_seeds)
    alpha_grid = parse_alpha_grid(args.ridge_alphas)

    sequences, targets = load_dataset(data_path)
    embedder = build_embedding_model(
        backend=str(args.embedding_backend),
        embedding_dim=int(args.embedding_dim),
        model_id=(str(args.embedding_model_id).strip() or None),
        device=str(args.embedding_device),
        pooling=str(args.embedding_pooling),
        allow_mock=(not args.disable_embedding_mock_fallback),
    )
    resolved_embedding_dim = int(embedder.embedding_dim)

    runs: list[dict] = []
    ensemble_size = max(1, int(args.ensemble_size))
    for split_seed in split_seeds:
        train_sequences, train_targets, val_sequences, val_targets = split_train_val(
            sequences,
            targets,
            val_ratio=float(args.val_ratio),
            seed=int(split_seed),
            split_mode=str(args.split_mode),
            scaffold_k=int(args.scaffold_k),
        )
        train_features = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
        val_features = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)

        trials = []
        for alpha in alpha_grid:
            if ensemble_size > 1:
                weights, bias = fit_ridge_ensemble(
                    train_features,
                    train_targets,
                    ridge_alpha=float(alpha),
                    ensemble_size=ensemble_size,
                    seed=int(split_seed),
                )
                val_preds, _ = predict_linear_ensemble(val_features, weights, bias)
            else:
                weights, bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=float(alpha))
                val_preds = predict_linear(val_features, weights, bias)
            metrics = evaluate_property_metrics(val_targets, val_preds)
            trials.append(
                {
                    "ridge_alpha": float(alpha),
                    "val_mean_rmse": float(metrics["mean"]["rmse"]),
                    "val_mean_mae": float(metrics["mean"]["mae"]),
                    "val_mean_r2": float(metrics["mean"]["r2"]),
                    "val_mean_pearson": float(metrics["mean"]["pearson"]),
                }
            )

        best_trial = select_best_trial(trials)
        best_alpha = float(best_trial["ridge_alpha"])
        if ensemble_size > 1:
            weights, bias = fit_ridge_ensemble(
                train_features,
                train_targets,
                ridge_alpha=best_alpha,
                ensemble_size=ensemble_size,
                seed=int(split_seed),
            )
            train_preds, train_unc = predict_linear_ensemble(train_features, weights, bias)
            val_preds, val_unc = predict_linear_ensemble(val_features, weights, bias)
        else:
            weights, bias = fit_ridge_regression(train_features, train_targets, ridge_alpha=best_alpha)
            train_preds = predict_linear(train_features, weights, bias)
            val_preds = predict_linear(val_features, weights, bias)
            train_unc = np.zeros((train_preds.shape[0],), dtype=np.float32)
            val_unc = np.zeros((val_preds.shape[0],), dtype=np.float32)

        run = {
            "split_seed": int(split_seed),
            "train_count": int(train_targets.shape[0]),
            "val_count": int(val_targets.shape[0]),
            "best_alpha": best_alpha,
            "train_metrics": evaluate_property_metrics(train_targets, train_preds),
            "val_metrics": evaluate_property_metrics(val_targets, val_preds),
            "train_calibration": regression_ece(train_targets, train_preds, train_unc, num_bins=int(args.ece_bins)),
            "val_calibration": regression_ece(val_targets, val_preds, val_unc, num_bins=int(args.ece_bins)),
        }
        runs.append(run)

    val_rmse = [float(run["val_metrics"]["mean"]["rmse"]) for run in runs]
    val_mae = [float(run["val_metrics"]["mean"]["mae"]) for run in runs]
    val_r2 = [float(run["val_metrics"]["mean"]["r2"]) for run in runs]
    val_pearson = [float(run["val_metrics"]["mean"]["pearson"]) for run in runs]
    val_ece = [float(run.get("val_calibration", {}).get("ece", 0.0)) for run in runs]
    alpha_counter = Counter([str(run["best_alpha"]) for run in runs])

    summary = {
        "val_mean_rmse": _mean_std(val_rmse),
        "val_mean_mae": _mean_std(val_mae),
        "val_mean_r2": _mean_std(val_r2),
        "val_mean_pearson": _mean_std(val_pearson),
        "val_ece": _mean_std(val_ece),
        "best_alpha_frequency": dict(alpha_counter),
    }

    report = {
        "dataset": str(data_path),
        "config": {
            "embedding_dim": resolved_embedding_dim,
            "embedding_backend": str(args.embedding_backend),
            "embedding_model_id": str(args.embedding_model_id).strip() or None,
            "embedding_pooling": str(args.embedding_pooling),
            "ensemble_size": int(ensemble_size),
            "val_ratio": float(args.val_ratio),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
            "split_seeds": split_seeds,
            "ridge_alphas": alpha_grid,
        },
        "summary": summary,
        "runs": runs,
    }

    json_path = output_dir / "property_cv_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved CV report: {json_path}")
    print(f"Validation mean RMSE avg={summary['val_mean_rmse']['mean']:.4f} std={summary['val_mean_rmse']['std']:.4f}")


if __name__ == "__main__":
    main()
