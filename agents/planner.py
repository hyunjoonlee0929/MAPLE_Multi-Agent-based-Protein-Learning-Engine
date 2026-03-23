"""Planner agent to validate and initialize shared execution context."""

from __future__ import annotations

from core.state import validate_state_shape


class PlannerAgent:
    """Ensures state and runtime configuration are ready for each iteration."""

    PRESET_WEIGHTS = {
        "balanced": {
            "w_stability": 0.35,
            "w_activity": 0.35,
            "w_uncertainty": 0.10,
            "w_structure": 0.10,
            "w_plddt": 0.05,
            "w_ptm": 0.03,
            "w_pae": 0.02,
        },
        "exploration": {
            "w_stability": 0.25,
            "w_activity": 0.25,
            "w_uncertainty": 0.25,
            "w_structure": 0.10,
            "w_plddt": 0.07,
            "w_ptm": 0.05,
            "w_pae": 0.03,
        },
        "structure_first": {
            "w_stability": 0.20,
            "w_activity": 0.20,
            "w_uncertainty": 0.05,
            "w_structure": 0.25,
            "w_plddt": 0.15,
            "w_ptm": 0.10,
            "w_pae": 0.05,
        },
        "activity_first": {
            "w_stability": 0.15,
            "w_activity": 0.50,
            "w_uncertainty": 0.10,
            "w_structure": 0.10,
            "w_plddt": 0.07,
            "w_ptm": 0.05,
            "w_pae": 0.03,
        },
    }

    WEIGHT_KEYS = [
        "w_stability",
        "w_activity",
        "w_uncertainty",
        "w_structure",
        "w_plddt",
        "w_ptm",
        "w_pae",
    ]

    def __init__(self, default_num_candidates: int = 8, default_top_k: int = 3) -> None:
        self.default_num_candidates = default_num_candidates
        self.default_top_k = default_top_k

    def _apply_weight_preset(self, config: dict) -> None:
        preset = str(config.get("scoring_preset", "balanced")).strip().lower()
        if preset not in self.PRESET_WEIGHTS:
            preset = "balanced"
            config["scoring_preset"] = preset

        if not bool(config.get("use_weight_preset", True)):
            return

        for k, v in self.PRESET_WEIGHTS[preset].items():
            config.setdefault(k, v)

    def _normalize_weights(self, config: dict) -> None:
        if not bool(config.get("normalize_score_weights", True)):
            return

        weights = [float(config.get(k, 0.0)) for k in self.WEIGHT_KEYS]
        total = sum(max(0.0, w) for w in weights)
        if total <= 0.0:
            # fallback to balanced preset if malformed config zeros everything out
            for k, v in self.PRESET_WEIGHTS["balanced"].items():
                config[k] = v
            return

        for k in self.WEIGHT_KEYS:
            config[k] = max(0.0, float(config.get(k, 0.0))) / total

    def run(self, state: dict) -> dict:
        validate_state_shape(state)

        config = state.setdefault("config", {})
        config.setdefault("num_candidates", self.default_num_candidates)
        config.setdefault("top_k", self.default_top_k)
        config.setdefault("mutation_rate", 1)

        config.setdefault("scoring_preset", "balanced")
        config.setdefault("use_weight_preset", True)
        config.setdefault("normalize_score_weights", True)

        self._apply_weight_preset(config)
        self._normalize_weights(config)

        config.setdefault("constraint_enabled", False)
        config.setdefault("constraint_mode", "hard")
        config.setdefault("constraint_penalty", 0.20)
        config.setdefault("bo_beta", 0.30)
        config.setdefault("bo_trials_per_parent", 8)

        return state
