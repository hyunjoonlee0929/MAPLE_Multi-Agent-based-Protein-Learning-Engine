"""Active learning cycle: propose batch, pseudo-label, retrain, and report."""

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

from core.active_learning import propose_active_learning_batch, synthetic_property_oracle
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


def _train_and_eval(
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
    report = {
        "best_alpha": best_alpha,
        "trials": trials,
        "train_metrics": evaluate_property_metrics(train_targets, train_pred),
        "val_metrics": evaluate_property_metrics(val_targets, val_pred),
        "train_calibration": regression_ece(train_targets, train_pred, train_unc, num_bins=int(ece_bins)),
        "val_calibration": regression_ece(val_targets, val_pred, val_unc, num_bins=int(ece_bins)),
        "ensemble_size": int(ensemble_size),
    }
    return weights, bias, report


def _save_augmented_csv(path: Path, sequences: list[str], targets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "stability", "activity"])
        writer.writeheader()
        for seq, vals in zip(sequences, targets):
            writer.writerow(
                {
                    "sequence": seq,
                    "stability": float(vals[0]),
                    "activity": float(vals[1]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPLE active learning cycle")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/active_learning")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/property_linear_active_learning.npz")
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
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pool-size", type=int, default=40)
    parser.add_argument("--mutation-rate", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--ridge-alphas", type=str, default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--seed", type=int, default=42)
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

    all_sequences, all_targets = load_dataset(data_path)
    train_seq, train_t, val_seq, val_t = split_train_val(
        all_sequences,
        all_targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        scaffold_k=int(args.scaffold_k),
    )

    train_sequences = list(train_seq)
    train_targets = np.asarray(train_t, dtype=np.float32)
    val_sequences = list(val_seq)
    val_targets = np.asarray(val_t, dtype=np.float32)

    alpha_grid = parse_alpha_grid(args.ridge_alphas)
    rounds_report = []
    weights = None
    bias = None

    for round_idx in range(int(args.rounds)):
        weights, bias, fit_report = _train_and_eval(
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
            ridge_alphas=alpha_grid,
            ensemble_size=max(1, int(args.ensemble_size)),
            ece_bins=int(args.ece_bins),
        )

        existing = set(train_sequences) | set(val_sequences)
        proposals = propose_active_learning_batch(
            train_sequences=train_sequences,
            train_targets=train_targets,
            existing_sequences=existing,
            embedding_dim=int(args.embedding_dim),
            batch_size=int(args.batch_size),
            pool_size=int(args.pool_size),
            mutation_rate=int(args.mutation_rate),
            beta=float(args.beta),
            random_seed=int(args.seed) + round_idx,
            embedding_backend=str(args.embedding_backend),
            embedding_model_id=(str(args.embedding_model_id).strip() or None),
            embedding_device=str(args.embedding_device),
            embedding_pooling=str(args.embedding_pooling),
            embedding_allow_mock=(not args.disable_embedding_mock_fallback),
        )

        acquired = []
        for item in proposals:
            seq = item["sequence"]
            pseudo = synthetic_property_oracle(seq)
            train_sequences.append(seq)
            train_targets = np.concatenate(
                [
                    train_targets,
                    np.asarray([[pseudo["stability"], pseudo["activity"]]], dtype=np.float32),
                ],
                axis=0,
            )
            acquired.append(
                {
                    "sequence": seq,
                    "acquisition": float(item["acquisition"]),
                    "pred_mean": float(item["pred_mean"]),
                    "novelty": float(item["novelty"]),
                    "pseudo_stability": float(pseudo["stability"]),
                    "pseudo_activity": float(pseudo["activity"]),
                }
            )

        rounds_report.append(
            {
                "round": round_idx,
                "train_size": len(train_sequences),
                "val_size": len(val_sequences),
                "fit": fit_report,
                "acquired_batch": acquired,
            }
        )

    if weights is None or bias is None:
        raise RuntimeError("active learning training did not run")

    np.savez(
        checkpoint_out,
        model_type=("numpy_linear_ensemble" if int(args.ensemble_size) > 1 else "numpy_linear"),
        embedding_dim=np.int32(args.embedding_dim),
        embedding_backend=np.array(str(args.embedding_backend)),
        embedding_model_id=np.array(str(args.embedding_model_id).strip()),
        embedding_pooling=np.array(str(args.embedding_pooling)),
        weights=weights.astype(np.float32),
        bias=bias.astype(np.float32),
    )

    augmented_csv = output_dir / "augmented_dataset.csv"
    _save_augmented_csv(augmented_csv, train_sequences, train_targets)

    report = {
        "seed_dataset": str(data_path),
        "final_checkpoint": str(checkpoint_out),
        "augmented_dataset": str(augmented_csv),
        "config": {
            "rounds": int(args.rounds),
            "batch_size": int(args.batch_size),
            "pool_size": int(args.pool_size),
            "mutation_rate": int(args.mutation_rate),
            "beta": float(args.beta),
            "embedding_dim": int(args.embedding_dim),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model_id": str(args.embedding_model_id).strip() or None,
            "embedding_pooling": str(args.embedding_pooling),
            "ensemble_size": int(max(1, int(args.ensemble_size))),
            "ece_bins": int(args.ece_bins),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
            "ridge_alphas": alpha_grid,
        },
        "rounds": rounds_report,
    }
    report_path = output_dir / "active_learning_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved active learning report: {report_path}")
    print(f"Saved augmented dataset: {augmented_csv}")
    print(f"Saved checkpoint: {checkpoint_out}")
    print(f"Final train size: {len(train_sequences)} | val size: {len(val_sequences)}")


if __name__ == "__main__":
    main()
