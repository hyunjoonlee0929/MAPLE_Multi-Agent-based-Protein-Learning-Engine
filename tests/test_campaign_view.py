from __future__ import annotations

from core.campaign_view import campaign_acquisition_rows, campaign_round_rows


def test_campaign_round_rows_extracts_round_metrics() -> None:
    payload = {
        "rounds": [
            {
                "round": 0,
                "maple_best_score": 0.8,
                "train_size_after_acquisition": 24,
                "val_size": 4,
                "fit": {
                    "best_alpha": 0.001,
                    "train_metrics": {"mean": {"rmse": 0.01}},
                    "val_metrics": {"mean": {"rmse": 0.05}},
                },
                "acquired_batch": [
                    {"sequence": "AAA", "stability": 0.6, "activity": 0.4},
                    {"sequence": "AAT", "stability": 0.7, "activity": 0.5},
                ],
            }
        ]
    }
    rows = campaign_round_rows(payload)
    assert len(rows) == 1
    assert rows[0]["round"] == 0
    assert rows[0]["acquired_count"] == 2
    assert abs(rows[0]["val_rmse_mean"] - 0.05) < 1e-8


def test_campaign_acquisition_rows_flattens_sequences() -> None:
    payload = {"rounds": [{"round": 1, "acquired_batch": [{"sequence": "AAA", "stability": 0.6, "activity": 0.4}]}]}
    rows = campaign_acquisition_rows(payload)
    assert len(rows) == 1
    assert rows[0]["round"] == 1
    assert rows[0]["sequence"] == "AAA"
