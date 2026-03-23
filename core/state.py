"""Shared state definitions and helpers for the MAPLE pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np


State = dict[str, Any]


REQUIRED_KEYS = {
    "sequences": list,
    "structures": list,
    "embeddings": list,
    "properties": list,
    "scores": list,
    "history": list,
}


def create_initial_state(seed_sequence: str) -> State:
    """Create a valid initial shared state with one seed sequence."""
    return {
        "sequences": [seed_sequence],
        "structures": [],
        "embeddings": [],
        "properties": [],
        "scores": [],
        "history": [],
        "iteration": 0,
        "next_sequences": None,
    }


def validate_state_shape(state: State) -> None:
    """Validate required keys and expected value types for state."""
    for key, expected_type in REQUIRED_KEYS.items():
        if key not in state:
            raise KeyError(f"Missing required state key: {key}")
        if not isinstance(state[key], expected_type):
            raise TypeError(
                f"State key '{key}' must be {expected_type.__name__}, got {type(state[key]).__name__}"
            )



def ensure_numpy_embeddings(state: State) -> None:
    """Safety check to ensure all embeddings are numpy arrays."""
    for idx, emb in enumerate(state["embeddings"]):
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"Embedding at index {idx} is not np.ndarray")
