"""Optimization agent for evolutionary sequence improvement."""

from __future__ import annotations

import random

from utils.mutation import random_mutation


class OptimizationAgent:
    """Selects top candidates and generates next iteration sequences."""

    def __init__(self, random_seed: int = 19) -> None:
        self.random_seed = random_seed

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        top_k = int(config.get("top_k", 3))
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        iteration = int(state.get("iteration", 0))

        sequences = state.get("sequences", [])
        if not sequences:
            state["next_sequences"] = []
            return state

        elites = sequences[: max(1, min(top_k, len(sequences)))]
        rng = random.Random(self.random_seed + iteration)

        next_sequences = list(elites)
        while len(next_sequences) < num_candidates:
            parent = rng.choice(elites)
            child = random_mutation(parent, num_mutations=mutation_rate, rng=rng)
            next_sequences.append(child)

        state["next_sequences"] = next_sequences
        return state
