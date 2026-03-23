"""Run a closed-loop in-silico MAPLE campaign before DBTL integration."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.active_learning import synthetic_property_oracle
from core.campaign import append_labeled_records, select_novel_top_sequences
from core.retraining import select_best_trial
from main import load_config, run_maple
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


def _train_round_model(
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

    report = {
        "embedding_dim": resolved_embedding_dim,
        "embedding_backend": embedding_backend,
        "embedding_model_id": embedding_model_id,
        "embedding_pooling": embedding_pooling,
        "ensemble_size": int(ensemble_size),
        "best_alpha": best_alpha,
        "trials": trials,
        "train_metrics": evaluate_property_metrics(train_targets, train_pred),
        "val_metrics": evaluate_property_metrics(val_targets, val_pred),
        "train_calibration": regression_ece(train_targets, train_pred, train_unc, num_bins=int(ece_bins)),
        "val_calibration": regression_ece(val_targets, val_pred, val_unc, num_bins=int(ece_bins)),
    }
    return weights, bias, report


def _save_dataset_csv(path: Path, sequences: list[str], targets: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "stability", "activity"])
        writer.writeheader()
        for seq, row in zip(sequences, targets):
            writer.writerow({"sequence": seq, "stability": float(row[0]), "activity": float(row[1])})


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPLE closed-loop campaign")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/closed_loop_campaign")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--maple-iterations", type=int, default=3)
    parser.add_argument("--acquisition-batch-size", type=int, default=4)
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
    parser.add_argument("--selection-strategy", type=str, default="pareto_bo")
    parser.add_argument("--bo-beta", type=float, default=0.30)
    parser.add_argument("--bo-trials-per-parent", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mutation-rate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("campaign")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_sequences, all_targets = load_dataset(data_path)
    train_sequences, train_targets, val_sequences, val_targets = split_train_val(
        all_sequences,
        all_targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        scaffold_k=int(args.scaffold_k),
    )
    train_sequences = list(train_sequences)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    val_sequences = list(val_sequences)
    val_targets = np.asarray(val_targets, dtype=np.float32)

    ridge_alphas = parse_alpha_grid(args.ridge_alphas)
    seen = set(train_sequences) | set(val_sequences)
    rounds = []
    current_checkpoint = None

    for round_idx in range(int(args.rounds)):
        round_dir = output_dir / f"round_{round_idx:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = round_dir / "property_model.npz"

        w, b, fit_report = _train_round_model(
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
            checkpoint_path,
            model_type=("numpy_linear_ensemble" if int(args.ensemble_size) > 1 else "numpy_linear"),
            embedding_dim=np.int32(fit_report.get("embedding_dim", int(args.embedding_dim))),
            embedding_backend=np.array(str(args.embedding_backend)),
            embedding_model_id=np.array(str(args.embedding_model_id).strip()),
            embedding_pooling=np.array(str(args.embedding_pooling)),
            weights=w.astype(np.float32),
            bias=b.astype(np.float32),
        )
        current_checkpoint = checkpoint_path

        run_overrides = {
            "seed": int(args.seed) + round_idx,
            "num_iterations": int(args.maple_iterations),
            "num_candidates": int(args.num_candidates),
            "top_k": int(args.top_k),
            "mutation_rate": int(args.mutation_rate),
            "selection_strategy": str(args.selection_strategy),
            "bo_beta": float(args.bo_beta),
            "bo_trials_per_parent": int(args.bo_trials_per_parent),
            "embedding_dim": int(args.embedding_dim),
            "property_checkpoint": str(checkpoint_path),
        }
        final_state, resolved, run_artifacts = run_maple(
            config=config,
            overrides=run_overrides,
            output_dir=round_dir / "maple_run",
            logger=logger,
        )

        ranked = list(final_state.get("sequences", []))
        acquired_seqs = select_novel_top_sequences(
            ranked_sequences=ranked,
            existing_sequences=seen,
            batch_size=int(args.acquisition_batch_size),
        )
        acquired_records = []
        for seq in acquired_seqs:
            labels = synthetic_property_oracle(seq)
            acquired_records.append(
                {
                    "sequence": seq,
                    "stability": float(labels["stability"]),
                    "activity": float(labels["activity"]),
                }
            )
            seen.add(seq)

        train_sequences, train_targets = append_labeled_records(train_sequences, train_targets, acquired_records)

        rounds.append(
            {
                "round": round_idx,
                "checkpoint": str(checkpoint_path),
                "maple_artifacts": str(run_artifacts),
                "maple_best_sequence": (final_state.get("sequences", [None])[0] if final_state.get("sequences") else None),
                "maple_best_score": (float(final_state.get("scores", [0.0])[0]) if final_state.get("scores") else None),
                "fit": fit_report,
                "acquired_batch": acquired_records,
                "train_size_after_acquisition": len(train_sequences),
                "val_size": len(val_sequences),
                "resolved_runtime": resolved,
            }
        )

    final_dataset_path = output_dir / "train_dataset_final.csv"
    _save_dataset_csv(final_dataset_path, train_sequences, train_targets)

    report = {
        "seed_dataset": str(data_path),
        "output_dir": str(output_dir),
        "final_checkpoint": (str(current_checkpoint) if current_checkpoint else None),
        "final_train_dataset": str(final_dataset_path),
        "config": {
            "rounds": int(args.rounds),
            "maple_iterations": int(args.maple_iterations),
            "acquisition_batch_size": int(args.acquisition_batch_size),
            "selection_strategy": str(args.selection_strategy),
            "bo_beta": float(args.bo_beta),
            "bo_trials_per_parent": int(args.bo_trials_per_parent),
            "num_candidates": int(args.num_candidates),
            "top_k": int(args.top_k),
            "mutation_rate": int(args.mutation_rate),
            "embedding_dim": int(args.embedding_dim),
            "embedding_backend": str(args.embedding_backend),
            "embedding_model_id": str(args.embedding_model_id).strip() or None,
            "embedding_pooling": str(args.embedding_pooling),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
            "ensemble_size": int(max(1, int(args.ensemble_size))),
            "ece_bins": int(args.ece_bins),
            "ridge_alphas": ridge_alphas,
            "seed": int(args.seed),
        },
        "rounds": rounds,
    }
    report_path = output_dir / "campaign_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved campaign report: {report_path}")
    print(f"Saved final train dataset: {final_dataset_path}")
    print(f"Final checkpoint: {report['final_checkpoint']}")


if __name__ == "__main__":
    main()
