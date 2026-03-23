"""Mutation utilities for protein sequence exploration."""

from __future__ import annotations

import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"



def random_mutation(sequence: str, num_mutations: int = 1, rng: random.Random | None = None) -> str:
    """Apply random amino-acid substitutions to a sequence."""
    if not sequence:
        return sequence

    rng = rng or random.Random()
    seq_list = list(sequence)
    positions = rng.sample(range(len(seq_list)), k=min(num_mutations, len(seq_list)))

    for pos in positions:
        current = seq_list[pos]
        alternatives = [aa for aa in AMINO_ACIDS if aa != current]
        seq_list[pos] = rng.choice(alternatives)

    return "".join(seq_list)



def guided_mutation(
    sequence: str,
    guidance_strength: float = 0.5,
    base_mutations: int = 1,
    rng: random.Random | None = None,
) -> str:
    """A simple guided mutation heuristic for MVP.

    Higher guidance_strength increases mutation count moderately.
    """
    rng = rng or random.Random()
    extra = 1 if rng.random() < max(0.0, min(1.0, guidance_strength)) else 0
    return random_mutation(sequence, num_mutations=base_mutations + extra, rng=rng)
