"""Sequence agent for candidate generation and mutation."""

from __future__ import annotations

import random

from utils.mutation import guided_mutation, random_mutation


class SequenceAgent:
    """Generates candidate protein sequences for current iteration."""

    def __init__(self, random_seed: int = 7) -> None:
        self.random_seed = random_seed

    def run(self, state: dict) -> dict:
        config = state.get("config", {})
        num_candidates = int(config.get("num_candidates", 8))
        mutation_rate = int(config.get("mutation_rate", 1))
        iteration = int(state.get("iteration", 0))

        rng = random.Random(self.random_seed + iteration)
        existing = state.get("next_sequences")

        if existing:
            pool = list(existing)
        else:
            pool = list(state.get("sequences", []))

        if not pool:
            raise ValueError("SequenceAgent requires at least one seed sequence")

        generated: list[str] = []
        while len(generated) < num_candidates:
            parent = rng.choice(pool)
            if rng.random() < 0.5:
                child = random_mutation(parent, num_mutations=mutation_rate, rng=rng)
            else:
                child = guided_mutation(
                    parent,
                    guidance_strength=0.6,
                    base_mutations=mutation_rate,
                    rng=rng,
                )
            generated.append(child)

        state["sequences"] = generated
        state["structures"] = []
        state["embeddings"] = []
        state["properties"] = []
        state["scores"] = []
        state["next_sequences"] = None
        return state
