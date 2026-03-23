"""Validation and model ranking helpers for property prediction."""

from __future__ import annotations


def rank_by_val_rmse(results: list[dict]) -> list[dict]:
    """Return results sorted by mean validation RMSE (ascending)."""
    return sorted(results, key=lambda r: float(r.get("val_metrics", {}).get("mean", {}).get("rmse", 1e9)))


def leaderboard_rows(payload: dict) -> list[dict]:
    """Flatten leaderboard payload to table rows."""
    rows = []
    for idx, item in enumerate(payload.get("ranked_results", []), start=1):
        mean = item.get("val_metrics", {}).get("mean", {})
        rows.append(
            {
                "rank": idx,
                "checkpoint": item.get("checkpoint"),
                "embedding_dim": item.get("embedding_dim"),
                "val_rmse_mean": mean.get("rmse"),
                "val_mae_mean": mean.get("mae"),
                "val_r2_mean": mean.get("r2"),
                "val_pearson_mean": mean.get("pearson"),
            }
        )
    return rows


def cv_run_rows(payload: dict) -> list[dict]:
    """Flatten CV payload per-seed runs for table/chart."""
    rows = []
    for run in payload.get("runs", []):
        mean = run.get("val_metrics", {}).get("mean", {})
        rows.append(
            {
                "split_seed": run.get("split_seed"),
                "best_alpha": run.get("best_alpha"),
                "val_rmse_mean": mean.get("rmse"),
                "val_mae_mean": mean.get("mae"),
                "val_r2_mean": mean.get("r2"),
                "val_pearson_mean": mean.get("pearson"),
            }
        )
    return rows
