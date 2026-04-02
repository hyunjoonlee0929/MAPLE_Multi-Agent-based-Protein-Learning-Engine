"""Run reproducible benchmark across public dataset CSV manifests."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.public_benchmark import aggregate_rows, benchmark_markdown, load_benchmark_manifest
from core.retraining import select_best_trial
from models.embedding_model import build_embedding_model
from scripts.retrain_property_pipeline import parse_alpha_grid
from scripts.train_property_numpy import (
    fit_ridge_ensemble,
    fit_ridge_regression,
    load_dataset,
    predict_linear,
    predict_linear_ensemble,
    split_train_val,
)
from utils.calibration import regression_ece
from utils.metrics import evaluate_property_metrics


def parse_seed_list(text: str) -> list[int]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    if not vals:
        raise ValueError("split seed list is empty")
    return vals


def parse_backend_list(text: str) -> list[str]:
    vals = []
    for item in text.split(","):
        item = item.strip().lower()
        if item:
            vals.append(item)
    if not vals:
        raise ValueError("embedding backend list is empty")
    return vals


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr))


def _resolve_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def _best_alpha_mode(runs: list[dict]) -> float:
    counter = Counter(str(run.get("best_alpha")) for run in runs)
    if not counter:
        return 0.0
    top = counter.most_common(1)[0][0]
    return float(top)


def _run_one_split(
    sequences: list[str],
    targets: np.ndarray,
    split_seed: int,
    val_ratio: float,
    split_mode: str,
    scaffold_k: int,
    embedder,
    alpha_grid: list[float],
    ensemble_size: int,
    ece_bins: int,
) -> dict:
    train_seq, train_t, val_seq, val_t = split_train_val(
        sequences,
        targets,
        val_ratio=val_ratio,
        seed=split_seed,
        split_mode=split_mode,
        scaffold_k=scaffold_k,
    )
    train_x = np.stack([embedder.encode(seq) for seq in train_seq]).astype(np.float32)
    val_x = np.stack([embedder.encode(seq) for seq in val_seq]).astype(np.float32)

    trials = []
    for alpha in alpha_grid:
        if ensemble_size > 1:
            w, b = fit_ridge_ensemble(train_x, train_t, ridge_alpha=float(alpha), ensemble_size=ensemble_size, seed=split_seed)
            val_pred, _ = predict_linear_ensemble(val_x, w, b)
        else:
            w, b = fit_ridge_regression(train_x, train_t, ridge_alpha=float(alpha))
            val_pred = predict_linear(val_x, w, b)
        val_metrics = evaluate_property_metrics(val_t, val_pred)
        trials.append(
            {
                "ridge_alpha": float(alpha),
                "val_mean_rmse": float(val_metrics["mean"]["rmse"]),
                "val_mean_mae": float(val_metrics["mean"]["mae"]),
                "val_mean_r2": float(val_metrics["mean"]["r2"]),
            }
        )

    best_trial = select_best_trial(trials)
    best_alpha = float(best_trial["ridge_alpha"])
    if ensemble_size > 1:
        w, b = fit_ridge_ensemble(train_x, train_t, ridge_alpha=best_alpha, ensemble_size=ensemble_size, seed=split_seed)
        train_pred, train_unc = predict_linear_ensemble(train_x, w, b)
        val_pred, val_unc = predict_linear_ensemble(val_x, w, b)
    else:
        w, b = fit_ridge_regression(train_x, train_t, ridge_alpha=best_alpha)
        train_pred = predict_linear(train_x, w, b)
        val_pred = predict_linear(val_x, w, b)
        train_unc = np.zeros((train_pred.shape[0],), dtype=np.float32)
        val_unc = np.zeros((val_pred.shape[0],), dtype=np.float32)

    train_metrics = evaluate_property_metrics(train_t, train_pred)
    val_metrics = evaluate_property_metrics(val_t, val_pred)
    train_cal = regression_ece(train_t, train_pred, train_unc, num_bins=ece_bins)
    val_cal = regression_ece(val_t, val_pred, val_unc, num_bins=ece_bins)
    return {
        "split_seed": int(split_seed),
        "train_count": int(train_t.shape[0]),
        "val_count": int(val_t.shape[0]),
        "best_alpha": best_alpha,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_calibration": train_cal,
        "val_calibration": val_cal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Public dataset benchmark for MAPLE property pipeline")
    parser.add_argument("--manifest", type=str, default="benchmarks/public_datasets_manifest.json")
    parser.add_argument("--output-dir", type=str, default="outputs/public_benchmark")
    parser.add_argument("--embedding-backends", type=str, default="random")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-pooling", type=str, default="mean")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-mode", type=str, default="scaffold")
    parser.add_argument("--scaffold-k", type=int, default=3)
    parser.add_argument("--split-seeds", type=str, default="1,7,13,21,42")
    parser.add_argument("--ridge-alphas", type=str, default="1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--embedding-model-id-map", type=str, default="")
    parser.add_argument("--disable-embedding-mock-fallback", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = load_benchmark_manifest(manifest_path)
    backends = parse_backend_list(args.embedding_backends)
    split_seeds = parse_seed_list(args.split_seeds)
    alpha_grid = parse_alpha_grid(args.ridge_alphas)
    ensemble_size = max(1, int(args.ensemble_size))

    model_id_map = {}
    map_text = str(args.embedding_model_id_map).strip()
    if map_text:
        model_id_map = json.loads(map_text)
        if not isinstance(model_id_map, dict):
            raise ValueError("--embedding-model-id-map must be a JSON object")

    benchmark_runs: list[dict] = []
    summary_rows: list[dict] = []

    for ds in dataset_entries:
        data_path = _resolve_path(ds["path"])
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        sequences, targets = load_dataset(data_path)
        for backend in backends:
            embedder = build_embedding_model(
                backend=backend,
                embedding_dim=int(args.embedding_dim),
                model_id=(str(model_id_map.get(backend, "")).strip() or None),
                device=str(args.embedding_device),
                pooling=str(args.embedding_pooling),
                allow_mock=(not args.disable_embedding_mock_fallback),
            )

            runs = [
                _run_one_split(
                    sequences=sequences,
                    targets=targets,
                    split_seed=int(split_seed),
                    val_ratio=float(args.val_ratio),
                    split_mode=str(args.split_mode),
                    scaffold_k=int(args.scaffold_k),
                    embedder=embedder,
                    alpha_grid=alpha_grid,
                    ensemble_size=ensemble_size,
                    ece_bins=int(args.ece_bins),
                )
                for split_seed in split_seeds
            ]

            val_rmse = [float(run["val_metrics"]["mean"]["rmse"]) for run in runs]
            val_mae = [float(run["val_metrics"]["mean"]["mae"]) for run in runs]
            val_r2 = [float(run["val_metrics"]["mean"]["r2"]) for run in runs]
            val_ece = [float(run["val_calibration"]["ece"]) for run in runs]
            rmse_mean, rmse_std = _mean_std(val_rmse)
            mae_mean, mae_std = _mean_std(val_mae)
            r2_mean, r2_std = _mean_std(val_r2)
            ece_mean, ece_std = _mean_std(val_ece)

            entry = {
                "dataset": ds["name"],
                "dataset_path": str(data_path),
                "source_url": ds.get("source_url"),
                "license": ds.get("license"),
                "embedding_backend": backend,
                "embedding_model_id": model_id_map.get(backend),
                "embedding_dim": int(embedder.embedding_dim),
                "split_mode": str(args.split_mode),
                "scaffold_k": int(args.scaffold_k),
                "val_ratio": float(args.val_ratio),
                "split_seeds": split_seeds,
                "alpha_grid": alpha_grid,
                "ensemble_size": int(ensemble_size),
                "ece_bins": int(args.ece_bins),
                "runs": runs,
            }
            benchmark_runs.append(entry)
            summary_rows.append(
                {
                    "dataset": ds["name"],
                    "dataset_path": str(data_path),
                    "source_url": ds.get("source_url"),
                    "license": ds.get("license"),
                    "embedding_backend": backend,
                    "embedding_model_id": model_id_map.get(backend),
                    "embedding_dim": int(embedder.embedding_dim),
                    "split_mode": str(args.split_mode),
                    "scaffold_k": int(args.scaffold_k),
                    "val_ratio": float(args.val_ratio),
                    "split_seeds_csv": ",".join(str(s) for s in split_seeds),
                    "ensemble_size": int(ensemble_size),
                    "ece_bins": int(args.ece_bins),
                    "best_alpha_mode": _best_alpha_mode(runs),
                    "val_rmse_mean": rmse_mean,
                    "val_rmse_std": rmse_std,
                    "val_mae_mean": mae_mean,
                    "val_mae_std": mae_std,
                    "val_r2_mean": r2_mean,
                    "val_r2_std": r2_std,
                    "val_ece_mean": ece_mean,
                    "val_ece_std": ece_std,
                }
            )

    ranked_rows = aggregate_rows(summary_rows)
    payload = {
        "manifest": str(manifest_path),
        "config": {
            "embedding_backends": backends,
            "embedding_device": str(args.embedding_device),
            "embedding_pooling": str(args.embedding_pooling),
            "val_ratio": float(args.val_ratio),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
            "split_seeds": split_seeds,
            "ridge_alphas": alpha_grid,
            "ensemble_size": int(ensemble_size),
            "ece_bins": int(args.ece_bins),
            "embedding_allow_mock": (not args.disable_embedding_mock_fallback),
        },
        "ranked_summary": ranked_rows,
        "runs": benchmark_runs,
    }

    json_path = output_dir / "public_benchmark_report.json"
    csv_path = output_dir / "public_benchmark_leaderboard.csv"
    md_path = output_dir / "public_benchmark_table.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "dataset",
                "dataset_path",
                "source_url",
                "license",
                "embedding_backend",
                "embedding_model_id",
                "embedding_dim",
                "split_mode",
                "scaffold_k",
                "val_ratio",
                "split_seeds_csv",
                "ensemble_size",
                "ece_bins",
                "best_alpha_mode",
                "val_rmse_mean",
                "val_rmse_std",
                "val_mae_mean",
                "val_mae_std",
                "val_r2_mean",
                "val_r2_std",
                "val_ece_mean",
                "val_ece_std",
            ],
        )
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow(row)

    md_path.write_text(benchmark_markdown(ranked_rows), encoding="utf-8")

    print(f"Saved benchmark report JSON: {json_path}")
    print(f"Saved benchmark leaderboard CSV: {csv_path}")
    print(f"Saved benchmark table Markdown: {md_path}")
    if ranked_rows:
        best = ranked_rows[0]
        print(
            "Best row: "
            f"dataset={best['dataset']} backend={best['embedding_backend']} "
            f"val_rmse_mean={float(best['val_rmse_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
