"""Optimization agent for evolutionary sequence improvement."""

from __future__ import annotations

import random

from utils.diversity import select_diverse_sequences
from utils.mutation import random_mutation


class OptimizationAgent:
    """Selects top candidates and generates next iteration sequences."""

    def __init__(self, random_seed: int = 19) -> None:
        self.random_seed = random_seed

    def _as_float(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _passes_constraints(self, prop: dict, structure: dict, config: dict) -> bool:
        if not bool(config.get("constraint_enabled", False)):
            return True

        min_stability = self._as_float(config.get("min_stability"))
        min_activity = self._as_float(config.get("min_activity"))
        min_structure_confidence = self._as_float(config.get("min_structure_confidence"))
        min_plddt = self._as_float(config.get("min_plddt"))
        min_ptm = self._as_float(config.get("min_ptm"))
        max_pae = self._as_float(config.get("max_pae"))

        stability = self._as_float(prop.get("stability"))
        activity = self._as_float(prop.get("activity"))
        confidence = self._as_float(structure.get("confidence"))
        plddt = self._as_float(structure.get("plddt_mean"))
        ptm = self._as_float(structure.get("ptm"))
        pae = self._as_float(structure.get("pae_mean"))

        if min_stability is not None and (stability is None or stability < min_stability):
            return False
        if min_activity is not None and (activity is None or activity < min_activity):
            return False
        if min_structure_confidence is not None and (confidence is None or confidence < min_structure_confidence):
            return False
        if min_plddt is not None and (plddt is None or plddt < min_plddt):
            return False
        if min_ptm is not None and (ptm is None or ptm < min_ptm):
            return False
        if max_pae is not None and (pae is None or pae > max_pae):
            return False

        return True

    def _select_elites(self, ranked_sequences: list[str], top_k: int, strategy: str, min_distance: int) -> list[str]:
        if not ranked_sequences:
            return []

        if strategy == "diverse" and min_distance > 0:
            return select_diverse_sequences(ranked_sequences, top_k=top_k, min_distance=min_distance)

        return ranked_sequences[: max(1, min(top_k, len(ranked_sequences)))]

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        top_k = int(config.get("top_k", 3))
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        min_distance = int(config.get("min_hamming_distance", 0))
        strategy = str(config.get("selection_strategy", "elitist")).strip().lower()
        iteration = int(state.get("iteration", 0))

        sequences = state.get("sequences", [])
        structures = state.get("structures", [])
        properties = state.get("properties", [])

        if not sequences:
            state["next_sequences"] = []
            return state

        ranked_aligned = list(zip(sequences, structures, properties))
        constrained_sequences = [
            seq for seq, structure, prop in ranked_aligned if self._passes_constraints(prop, structure, config)
        ]

        if constrained_sequences:
            candidate_pool = constrained_sequences
        else:
            candidate_pool = sequences

        state["constraint_summary"] = {
            "enabled": bool(config.get("constraint_enabled", False)),
            "passed": len(constrained_sequences),
            "total": len(sequences),
        }

        elites = self._select_elites(
            ranked_sequences=candidate_pool,
            top_k=max(1, min(top_k, len(candidate_pool))),
            strategy=strategy,
            min_distance=min_distance,
        )

        rng = random.Random(self.random_seed + iteration)
        next_sequences = list(elites)
        while len(next_sequences) < num_candidates:
            parent = rng.choice(elites)
            child = random_mutation(parent, num_mutations=mutation_rate, rng=rng)
            next_sequences.append(child)

        state["next_sequences"] = next_sequences
        return state
