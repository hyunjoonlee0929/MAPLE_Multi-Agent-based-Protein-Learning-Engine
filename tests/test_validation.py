from __future__ import annotations

from core.validation import rank_by_val_rmse
from scripts.evaluate_property_checkpoints import parse_checkpoint_list


def test_rank_by_val_rmse_orders_lowest_first() -> None:
    results = [
        {"checkpoint": "a.npz", "val_metrics": {"mean": {"rmse": 0.3}}},
        {"checkpoint": "b.npz", "val_metrics": {"mean": {"rmse": 0.1}}},
        {"checkpoint": "c.npz", "val_metrics": {"mean": {"rmse": 0.2}}},
    ]
    ranked = rank_by_val_rmse(results)
    assert ranked[0]["checkpoint"] == "b.npz"
    assert ranked[1]["checkpoint"] == "c.npz"
    assert ranked[2]["checkpoint"] == "a.npz"


def test_parse_checkpoint_list_filters_empty_items() -> None:
    parsed = parse_checkpoint_list("a.npz, ,b.npz,, c.npz")
    assert parsed == ["a.npz", "b.npz", "c.npz"]
