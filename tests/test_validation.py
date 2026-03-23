from __future__ import annotations

from core.validation import cv_run_rows, leaderboard_rows, rank_by_val_rmse
from scripts.property_cv_report import parse_seed_list
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


def test_parse_seed_list_parses_ints() -> None:
    seeds = parse_seed_list("1, 7,13")
    assert seeds == [1, 7, 13]


def test_leaderboard_rows_flattens_ranked_results() -> None:
    payload = {
        "ranked_results": [
            {
                "checkpoint": "a.npz",
                "embedding_dim": 128,
                "val_metrics": {"mean": {"rmse": 0.1, "mae": 0.08, "r2": 0.2, "pearson": 0.4}},
            }
        ]
    }
    rows = leaderboard_rows(payload)
    assert rows[0]["rank"] == 1
    assert rows[0]["checkpoint"] == "a.npz"
    assert abs(rows[0]["val_rmse_mean"] - 0.1) < 1e-8


def test_cv_run_rows_flattens_runs() -> None:
    payload = {
        "runs": [
            {
                "split_seed": 42,
                "best_alpha": 0.001,
                "val_metrics": {"mean": {"rmse": 0.11, "mae": 0.09, "r2": 0.1, "pearson": 0.3}},
            }
        ]
    }
    rows = cv_run_rows(payload)
    assert rows[0]["split_seed"] == 42
    assert abs(rows[0]["val_rmse_mean"] - 0.11) < 1e-8
