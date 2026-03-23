"""Ingest DBTL test records and trigger automatic property-model retraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dbtl import load_dbtl_records, merge_dbtl_into_dataset
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


def _save_dataset_csv(path: Path, sequences: list[str], targets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "stability", "activity"])
        writer.writeheader()
        for seq, row in zip(sequences, targets):
            writer.writerow({"sequence": seq, "stability": float(row[0]), "activity": float(row[1])})


def _train_retrained_model(
    train_sequences: list[str],
    train_targets: np.ndarray,
    val_sequences: list[str],
    val_targets: np.ndarray,
    embedding_dim: int,
    embedding_backend: str,
    embedding_model_id: str | None,
    embedding_device: str,
    embedding_pooling: str,
    embedding_allow_mock: bool,
    ridge_alphas: list[float],
    ensemble_size: int,
    ece_bins: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    embedder = build_embedding_model(
        backend=embedding_backend,
        embedding_dim=embedding_dim,
        model_id=embedding_model_id,
        device=embedding_device,
        pooling=embedding_pooling,
        allow_mock=embedding_allow_mock,
    )
    resolved_embedding_dim = int(embedder.embedding_dim)
    train_x = np.stack([embedder.encode(seq) for seq in train_sequences]).astype(np.float32)
    val_x = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)

    trials = []
    for alpha in ridge_alphas:
        if ensemble_size > 1:
            w, b = fit_ridge_ensemble(
                train_x,
                train_targets,
                ridge_alpha=float(alpha),
                ensemble_size=ensemble_size,
                seed=42,
            )
            val_pred, _ = predict_linear_ensemble(val_x, w, b)
        else:
            w, b = fit_ridge_regression(train_x, train_targets, ridge_alpha=float(alpha))
            val_pred = predict_linear(val_x, w, b)
        val_metrics = evaluate_property_metrics(val_targets, val_pred)
        trials.append(
            {
                "ridge_alpha": float(alpha),
                "val_mean_rmse": float(val_metrics["mean"]["rmse"]),
                "val_mean_mae": float(val_metrics["mean"]["mae"]),
                "val_mean_r2": float(val_metrics["mean"]["r2"]),
                "val_mean_pearson": float(val_metrics["mean"]["pearson"]),
            }
        )

    best = select_best_trial(trials)
    best_alpha = float(best["ridge_alpha"])
    if ensemble_size > 1:
        weights, bias = fit_ridge_ensemble(
            train_x,
            train_targets,
            ridge_alpha=best_alpha,
            ensemble_size=ensemble_size,
            seed=42,
        )
        train_pred, train_unc = predict_linear_ensemble(train_x, weights, bias)
        val_pred, val_unc = predict_linear_ensemble(val_x, weights, bias)
    else:
        weights, bias = fit_ridge_regression(train_x, train_targets, ridge_alpha=best_alpha)
        train_pred = predict_linear(train_x, weights, bias)
        val_pred = predict_linear(val_x, weights, bias)
        train_unc = np.zeros((train_pred.shape[0],), dtype=np.float32)
        val_unc = np.zeros((val_pred.shape[0],), dtype=np.float32)
    summary = {
        "embedding_dim": resolved_embedding_dim,
        "embedding_backend": embedding_backend,
        "embedding_model_id": embedding_model_id,
        "embedding_pooling": embedding_pooling,
        "best_alpha": best_alpha,
        "trials": trials,
        "train_metrics": evaluate_property_metrics(train_targets, train_pred),
        "val_metrics": evaluate_property_metrics(val_targets, val_pred),
        "train_calibration": regression_ece(train_targets, train_pred, train_unc, num_bins=int(ece_bins)),
        "val_calibration": regression_ece(val_targets, val_pred, val_unc, num_bins=int(ece_bins)),
        "ensemble_size": int(ensemble_size),
    }
    return weights, bias, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DBTL ingestion + auto-retrain trigger")
    parser.add_argument("--seed-data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--dbtl-input", type=str, required=True, help="DBTL result file (csv/json)")
    parser.add_argument("--dbtl-format", type=str, default="auto", help="auto|csv|json")
    parser.add_argument("--output-dir", type=str, default="outputs/dbtl_ingest")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/property_linear_dbtl.npz")
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
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--ridge-alphas", type=str, default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--min-imported-records", type=int, default=1)
    args = parser.parse_args()

    seed_data_path = Path(args.seed_data)
    if not seed_data_path.is_absolute():
        seed_data_path = ROOT / seed_data_path

    dbtl_path = Path(args.dbtl_input)
    if not dbtl_path.is_absolute():
        dbtl_path = ROOT / dbtl_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_out = Path(args.checkpoint_out)
    if not checkpoint_out.is_absolute():
        checkpoint_out = ROOT / checkpoint_out
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    all_sequences, all_targets = load_dataset(seed_data_path)
    train_sequences, train_targets, val_sequences, val_targets = split_train_val(
        all_sequences,
        all_targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        scaffold_k=int(args.scaffold_k),
    )

    dbtl_records = load_dbtl_records(dbtl_path, fmt=str(args.dbtl_format))
    train_sequences, train_targets, val_sequences, val_targets, merge_stats = merge_dbtl_into_dataset(
        list(train_sequences),
        np.asarray(train_targets, dtype=np.float32),
        list(val_sequences),
        np.asarray(val_targets, dtype=np.float32),
        dbtl_records,
    )

    merged_train_csv = output_dir / "train_dataset_merged.csv"
    merged_val_csv = output_dir / "val_dataset_merged.csv"
    _save_dataset_csv(merged_train_csv, train_sequences, train_targets)
    _save_dataset_csv(merged_val_csv, val_sequences, val_targets)

    imported = int(merge_stats["imported_records"])
    retrain_triggered = imported >= int(args.min_imported_records)

    report = {
        "seed_dataset": str(seed_data_path),
        "dbtl_input": str(dbtl_path),
        "dbtl_format": str(args.dbtl_format),
        "merge_stats": merge_stats,
        "train_size": len(train_sequences),
        "val_size": len(val_sequences),
        "split_mode": str(args.split_mode),
        "scaffold_k": int(args.scaffold_k),
        "merged_train_csv": str(merged_train_csv),
        "merged_val_csv": str(merged_val_csv),
        "retrain_triggered": retrain_triggered,
        "checkpoint": None,
        "fit": None,
    }

    if retrain_triggered:
        ridge_alphas = parse_alpha_grid(args.ridge_alphas)
        weights, bias, fit_summary = _train_retrained_model(
            train_sequences=train_sequences,
            train_targets=train_targets,
            val_sequences=val_sequences,
            val_targets=val_targets,
            embedding_dim=int(args.embedding_dim),
            embedding_backend=str(args.embedding_backend),
            embedding_model_id=(str(args.embedding_model_id).strip() or None),
            embedding_device=str(args.embedding_device),
            embedding_pooling=str(args.embedding_pooling),
            embedding_allow_mock=(not args.disable_embedding_mock_fallback),
            ridge_alphas=ridge_alphas,
            ensemble_size=max(1, int(args.ensemble_size)),
            ece_bins=int(args.ece_bins),
        )
        np.savez(
            checkpoint_out,
            model_type=("numpy_linear_ensemble" if int(args.ensemble_size) > 1 else "numpy_linear"),
            embedding_dim=np.int32(fit_summary.get("embedding_dim", int(args.embedding_dim))),
            embedding_backend=np.array(str(args.embedding_backend)),
            embedding_model_id=np.array(str(args.embedding_model_id).strip()),
            embedding_pooling=np.array(str(args.embedding_pooling)),
            weights=weights.astype(np.float32),
            bias=bias.astype(np.float32),
        )
        report["checkpoint"] = str(checkpoint_out)
        report["fit"] = fit_summary

    report_path = output_dir / "dbtl_retrain_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved DBTL retrain report: {report_path}")
    print(f"Retrain triggered: {retrain_triggered}")
    if retrain_triggered:
        print(f"Saved checkpoint: {checkpoint_out}")


if __name__ == "__main__":
    main()
