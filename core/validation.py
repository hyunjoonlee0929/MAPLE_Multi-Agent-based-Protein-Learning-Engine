"""Validation and model ranking helpers for property prediction."""

from __future__ import annotations


def rank_by_val_rmse(results: list[dict]) -> list[dict]:
    """Return results sorted by mean validation RMSE (ascending)."""
    return sorted(results, key=lambda r: float(r.get("val_metrics", {}).get("mean", {}).get("rmse", 1e9)))
