"""Planner agent to validate and initialize shared execution context."""

from __future__ import annotations

from core.state import validate_state_shape


class PlannerAgent:
    """Ensures state and runtime configuration are ready for each iteration."""

    def __init__(self, default_num_candidates: int = 8, default_top_k: int = 3) -> None:
        self.default_num_candidates = default_num_candidates
        self.default_top_k = default_top_k

    def run(self, state: dict) -> dict:
        validate_state_shape(state)

        config = state.setdefault("config", {})
        config.setdefault("num_candidates", self.default_num_candidates)
        config.setdefault("top_k", self.default_top_k)
        config.setdefault("mutation_rate", 1)

        config.setdefault("w_stability", 0.35)
        config.setdefault("w_activity", 0.35)
        config.setdefault("w_uncertainty", 0.10)
        config.setdefault("w_structure", 0.10)
        config.setdefault("w_plddt", 0.05)
        config.setdefault("w_ptm", 0.03)
        config.setdefault("w_pae", 0.02)

        return state
