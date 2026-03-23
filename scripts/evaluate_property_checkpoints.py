"""Evaluate one or more property checkpoints on a fixed validation split."""

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

from core.validation import rank_by_val_rmse
from models.embedding_model import build_embedding_model
from models.property_model import PropertyPredictor
from scripts.train_property_numpy import load_dataset, split_train_val
from utils.calibration import regression_ece
from utils.metrics import evaluate_property_metrics


def parse_checkpoint_list(text: str) -> list[str]:
    items = [item.strip() for item in text.split(",")]
    return [item for item in items if item]


def infer_embedding_dim(checkpoint_path: str, fallback_dim: int) -> int:
    path = Path(checkpoint_path)
    if path.suffix.lower() != ".npz":
        return int(fallback_dim)
    data = np.load(path, allow_pickle=False)
    if "embedding_dim" in data:
        return int(data["embedding_dim"])
    if "weights" in data:
        weights = np.asarray(data["weights"])
        if weights.ndim == 2:
            return int(weights.shape[0])
        if weights.ndim == 3:
            return int(weights.shape[1])
    return int(fallback_dim)


def infer_embedding_backend(checkpoint_path: str, fallback_backend: str) -> str:
    path = Path(checkpoint_path)
    if path.suffix.lower() != ".npz":
        return str(fallback_backend)
    data = np.load(path, allow_pickle=False)
    raw = data.get("embedding_backend")
    if raw is None:
        return str(fallback_backend)
    if isinstance(raw, np.ndarray):
        if raw.shape == ():
            return str(raw.item())
        return str(raw.reshape(-1)[0])
    return str(raw)


def infer_embedding_model_id(checkpoint_path: str) -> str | None:
    path = Path(checkpoint_path)
    if path.suffix.lower() != ".npz":
        return None
    data = np.load(path, allow_pickle=False)
    raw = data.get("embedding_model_id")
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        value = str(raw.item()) if raw.shape == () else str(raw.reshape(-1)[0])
    else:
        value = str(raw)
    value = value.strip()
    return value or None


def evaluate_checkpoint(
    checkpoint_path: str,
    val_sequences: list[str],
    val_targets: np.ndarray,
    fallback_embedding_dim: int,
    fallback_backend: str,
    embedding_device: str,
    embedding_pooling: str,
) -> dict:
    emb_dim = infer_embedding_dim(checkpoint_path, fallback_embedding_dim)
    backend = infer_embedding_backend(checkpoint_path, fallback_backend)
    model_id = infer_embedding_model_id(checkpoint_path)
    embedder = build_embedding_model(
        backend=backend,
        embedding_dim=emb_dim,
        model_id=model_id,
        device=embedding_device,
        pooling=embedding_pooling,
        allow_mock=True,
    )
    features = np.stack([embedder.encode(seq) for seq in val_sequences]).astype(np.float32)
    predictor = PropertyPredictor(
        embedding_dim=emb_dim,
        checkpoint_path=checkpoint_path,
        uncertainty_samples=5,
        uncertainty_noise=0.02,
    )
    preds, unc = predictor.predict_with_uncertainty(features)
    preds = np.asarray(preds, dtype=np.float32)
    unc = np.asarray(unc, dtype=np.float32)
    metrics = evaluate_property_metrics(val_targets, preds)
    calibration = regression_ece(val_targets, preds, unc, num_bins=10)
    return {
        "checkpoint": checkpoint_path,
        "embedding_dim": int(embedder.embedding_dim),
        "embedding_backend": backend,
        "embedding_model_id": model_id,
        "val_metrics": metrics,
        "uncertainty_mean": float(np.mean(unc)) if unc.size else 0.0,
        "val_ece": float(calibration.get("ece", 0.0)),
    }


def export_leaderboard_csv(results: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "checkpoint",
                "embedding_dim",
                "embedding_backend",
                "embedding_model_id",
                "val_rmse_mean",
                "val_mae_mean",
                "val_r2_mean",
                "val_pearson_mean",
                "val_ece",
                "uncertainty_mean",
            ],
        )
        writer.writeheader()
        for idx, row in enumerate(results, start=1):
            mean = row["val_metrics"]["mean"]
            writer.writerow(
                {
                    "rank": idx,
                    "checkpoint": row["checkpoint"],
                    "embedding_dim": row["embedding_dim"],
                    "embedding_backend": row.get("embedding_backend"),
                    "embedding_model_id": row.get("embedding_model_id"),
                    "val_rmse_mean": mean["rmse"],
                    "val_mae_mean": mean["mae"],
                    "val_r2_mean": mean["r2"],
                    "val_pearson_mean": mean["pearson"],
                    "val_ece": row.get("val_ece"),
                    "uncertainty_mean": row.get("uncertainty_mean"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate property checkpoints on fixed validation split")
    parser.add_argument("--data", type=str, default="data/sample_property_labels.csv")
    parser.add_argument("--checkpoints", type=str, required=True, help="Comma-separated checkpoint paths")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-mode", type=str, default="random", help="Split strategy: random|scaffold")
    parser.add_argument("--scaffold-k", type=int, default=3, help="Scaffold key size for split_mode=scaffold")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--embedding-backend", type=str, default="random")
    parser.add_argument("--embedding-device", type=str, default="cpu")
    parser.add_argument("--embedding-pooling", type=str, default="mean")
    parser.add_argument("--output-dir", type=str, default="outputs/property_validation")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = parse_checkpoint_list(args.checkpoints)
    if not checkpoints:
        raise ValueError("No checkpoints provided")

    resolved_checkpoints: list[str] = []
    for ckpt in checkpoints:
        p = Path(ckpt)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        resolved_checkpoints.append(str(p))

    sequences, targets = load_dataset(data_path)
    _, _, val_sequences, val_targets = split_train_val(
        sequences,
        targets,
        val_ratio=float(args.val_ratio),
        seed=int(args.split_seed),
        split_mode=str(args.split_mode),
        scaffold_k=int(args.scaffold_k),
    )

    raw_results = [
        evaluate_checkpoint(
            checkpoint_path=ckpt,
            val_sequences=val_sequences,
            val_targets=val_targets,
            fallback_embedding_dim=int(args.embedding_dim),
            fallback_backend=str(args.embedding_backend),
            embedding_device=str(args.embedding_device),
            embedding_pooling=str(args.embedding_pooling),
        )
        for ckpt in resolved_checkpoints
    ]
    ranked = rank_by_val_rmse(raw_results)

    payload = {
        "dataset": str(data_path),
        "split": {
            "val_count": int(val_targets.shape[0]),
            "val_ratio": float(args.val_ratio),
            "split_seed": int(args.split_seed),
            "split_mode": str(args.split_mode),
            "scaffold_k": int(args.scaffold_k),
        },
        "ranked_results": ranked,
        "best": ranked[0] if ranked else None,
    }

    json_path = output_dir / "validation_leaderboard.json"
    csv_path = output_dir / "validation_leaderboard.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    export_leaderboard_csv(ranked, csv_path)

    print(f"Saved leaderboard JSON: {json_path}")
    print(f"Saved leaderboard CSV: {csv_path}")
    print(f"Best checkpoint: {payload['best']['checkpoint']}")
    print(f"Best val mean RMSE: {payload['best']['val_metrics']['mean']['rmse']:.4f}")


if __name__ == "__main__":
    main()
