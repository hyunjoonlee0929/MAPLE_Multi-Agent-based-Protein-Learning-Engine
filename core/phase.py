"""Phase transition gates and readiness assessment for MAPLE."""

from __future__ import annotations


PHASE_LABELS = {
    "phase1": "MVP Baseline",
    "phase2": "Adapter-Integrated Optimization",
    "phase3": "Model-Driven Design",
}


PHASE3_GATES = {
    "num_iterations_min": 10,
    "best_score_min": 0.75,
    "constraint_pass_rate_min": 0.70,
    "structure_external_rate_min": 0.80,
    "structure_error_fallback_rate_max": 0.05,
}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_phase_report(final_state: dict, resolved: dict | None = None) -> dict:
    """Build phase status and Phase-3 readiness report from a run result."""
    resolved = resolved or {}
    history = final_state.get("history", [])
    last = history[-1] if history else {}
    scores = final_state.get("scores", [])

    num_iterations = int(resolved.get("num_iterations", len(history)))
    best_score = _safe_float(scores[0], 0.0) if scores else _safe_float(last.get("best_score"), 0.0)
    constraint_pass_rate = _safe_float(last.get("constraint_pass_rate"), 0.0)
    structure_external_rate = _safe_float(last.get("structure_external_rate"), 0.0)
    structure_error_fallback_rate = _safe_float(last.get("structure_error_fallback_rate"), 1.0)

    gates = {
        "num_iterations": num_iterations >= PHASE3_GATES["num_iterations_min"],
        "best_score": best_score >= PHASE3_GATES["best_score_min"],
        "constraint_pass_rate": constraint_pass_rate >= PHASE3_GATES["constraint_pass_rate_min"],
        "structure_external_rate": structure_external_rate >= PHASE3_GATES["structure_external_rate_min"],
        "structure_error_fallback_rate": structure_error_fallback_rate <= PHASE3_GATES["structure_error_fallback_rate_max"],
    }
    unmet = [name for name, passed in gates.items() if not passed]
    phase3_ready = len(unmet) == 0

    current_phase = "phase3" if phase3_ready else "phase2"
    next_phase = None if phase3_ready else "phase3"

    return {
        "current_phase": current_phase,
        "current_phase_label": PHASE_LABELS[current_phase],
        "next_phase": next_phase,
        "next_phase_label": (PHASE_LABELS[next_phase] if next_phase else None),
        "phase3_ready": phase3_ready,
        "phase3_gates": gates,
        "phase3_unmet_gates": unmet,
        "thresholds": dict(PHASE3_GATES),
        "observed": {
            "num_iterations": num_iterations,
            "best_score": best_score,
            "constraint_pass_rate": constraint_pass_rate,
            "structure_external_rate": structure_external_rate,
            "structure_error_fallback_rate": structure_error_fallback_rate,
        },
        "transition_decision": ("enter_phase3" if phase3_ready else "stay_phase2"),
    }
