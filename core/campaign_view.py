"""View helpers for closed-loop campaign report visualization."""

from __future__ import annotations


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def campaign_round_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("rounds", []):
        fit = row.get("fit", {})
        val_mean = fit.get("val_metrics", {}).get("mean", {})
        train_mean = fit.get("train_metrics", {}).get("mean", {})
        batch = row.get("acquired_batch", [])
        rows.append(
            {
                "round": int(row.get("round", 0)),
                "maple_best_score": _safe_float(row.get("maple_best_score")),
                "train_size_after_acquisition": int(row.get("train_size_after_acquisition", 0)),
                "val_size": int(row.get("val_size", 0)),
                "best_alpha": _safe_float(fit.get("best_alpha")),
                "val_rmse_mean": _safe_float(val_mean.get("rmse")),
                "train_rmse_mean": _safe_float(train_mean.get("rmse")),
                "acquired_count": len(batch),
                "acquired_stability_mean": (
                    sum(_safe_float(item.get("stability")) for item in batch) / len(batch)
                    if batch
                    else 0.0
                ),
                "acquired_activity_mean": (
                    sum(_safe_float(item.get("activity")) for item in batch) / len(batch)
                    if batch
                    else 0.0
                ),
            }
        )
    return rows


def campaign_acquisition_rows(payload: dict) -> list[dict]:
    rows = []
    for row in payload.get("rounds", []):
        round_idx = int(row.get("round", 0))
        for item in row.get("acquired_batch", []):
            rows.append(
                {
                    "round": round_idx,
                    "sequence": item.get("sequence"),
                    "stability": _safe_float(item.get("stability")),
                    "activity": _safe_float(item.get("activity")),
                }
            )
    return rows
