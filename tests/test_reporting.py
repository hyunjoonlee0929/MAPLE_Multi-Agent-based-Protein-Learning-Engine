from __future__ import annotations

import json
from pathlib import Path

from core.reporting import export_final_summary, export_history_csv, export_history_json



def test_reporting_exports_json_csv_and_summary(tmp_path: Path) -> None:
    history = [
        {
            "iteration": 0,
            "best_sequence": "AAAA",
            "best_score": 0.9,
            "mean_score": 0.5,
            "num_candidates": 5,
            "constraint_pass_rate": 0.8,
            "constraint_passed": 4,
            "constraint_total": 5,
            "constraint_mode": "hard",
        }
    ]
    final_state = {
        "sequences": ["AAAA"],
        "scores": [0.9],
        "history": history,
    }

    json_path = tmp_path / "history.json"
    csv_path = tmp_path / "history.csv"
    summary_path = tmp_path / "summary.json"

    export_history_json(history, json_path)
    export_history_csv(history, csv_path)
    export_final_summary(final_state, summary_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert summary_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["best_sequence"] == "AAAA"
    assert payload["best_score"] == 0.9

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "constraint_pass_rate" in csv_text
    assert "constraint_mode" in csv_text
