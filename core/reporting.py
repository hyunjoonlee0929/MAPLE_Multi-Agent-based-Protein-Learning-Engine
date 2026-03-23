"""Reporting and artifact export helpers for MAPLE runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path



def export_history_json(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)



def export_history_csv(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["iteration", "best_sequence", "best_score", "mean_score", "num_candidates"]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in headers})



def export_final_summary(final_state: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_sequence": final_state["sequences"][0] if final_state.get("sequences") else None,
        "best_score": final_state["scores"][0] if final_state.get("scores") else None,
        "history_entries": len(final_state.get("history", [])),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
