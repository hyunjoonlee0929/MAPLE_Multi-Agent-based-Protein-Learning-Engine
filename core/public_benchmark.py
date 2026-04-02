"""Helpers for public-dataset benchmark reporting."""

from __future__ import annotations

import json
from pathlib import Path


def load_benchmark_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object")
    datasets = payload.get("datasets", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Benchmark manifest must include non-empty 'datasets' list")
    out: list[dict] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        data_path = str(item.get("path", "")).strip()
        if not name or not data_path:
            continue
        out.append(
            {
                "name": name,
                "path": data_path,
                "source_url": str(item.get("source_url", "")).strip() or None,
                "license": str(item.get("license", "")).strip() or None,
                "notes": str(item.get("notes", "")).strip() or None,
            }
        )
    if not out:
        raise ValueError("No valid dataset entries found in manifest")
    return out


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Sort benchmark rows by val_rmse_mean and return new list."""
    ranked = sorted(rows, key=lambda r: float(r.get("val_rmse_mean", 1e9)))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def benchmark_markdown(rows: list[dict]) -> str:
    if not rows:
        return "# Public Dataset Benchmark\n\nNo benchmark rows."

    headers = [
        "Rank",
        "Dataset",
        "Embedding",
        "Split",
        "Val RMSE (mean±std)",
        "Val MAE (mean±std)",
        "Val R2 (mean±std)",
        "Val ECE (mean±std)",
        "Best Alpha",
    ]
    out = ["# Public Dataset Benchmark", "", "| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    for row in rows:
        out.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank")),
                    str(row.get("dataset")),
                    str(row.get("embedding_backend")),
                    str(row.get("split_mode")),
                    f"{float(row.get('val_rmse_mean', 0.0)):.4f} ± {float(row.get('val_rmse_std', 0.0)):.4f}",
                    f"{float(row.get('val_mae_mean', 0.0)):.4f} ± {float(row.get('val_mae_std', 0.0)):.4f}",
                    f"{float(row.get('val_r2_mean', 0.0)):.4f} ± {float(row.get('val_r2_std', 0.0)):.4f}",
                    f"{float(row.get('val_ece_mean', 0.0)):.4f} ± {float(row.get('val_ece_std', 0.0)):.4f}",
                    str(row.get("best_alpha_mode")),
                ]
            )
            + " |"
        )
    return "\n".join(out) + "\n"
