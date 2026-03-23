from __future__ import annotations

from core.phase import build_phase_report


def test_phase_report_stays_phase2_when_gates_unmet() -> None:
    final_state = {
        "history": [
            {
                "best_score": 0.5,
                "constraint_pass_rate": 0.4,
                "structure_external_rate": 0.0,
                "structure_error_fallback_rate": 0.3,
            }
        ],
        "scores": [0.5],
    }
    report = build_phase_report(final_state, {"num_iterations": 5})

    assert report["current_phase"] == "phase2"
    assert report["phase3_ready"] is False
    assert "best_score" in report["phase3_unmet_gates"]


def test_phase_report_enters_phase3_when_all_gates_met() -> None:
    final_state = {
        "history": [
            {
                "best_score": 0.9,
                "constraint_pass_rate": 0.8,
                "structure_external_rate": 0.95,
                "structure_error_fallback_rate": 0.01,
            }
        ],
        "scores": [0.9],
    }
    report = build_phase_report(final_state, {"num_iterations": 12})

    assert report["current_phase"] == "phase3"
    assert report["phase3_ready"] is True
    assert report["phase3_unmet_gates"] == []
