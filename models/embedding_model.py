"""Embedding model abstraction for protein sequences."""

from __future__ import annotations

import hashlib

import numpy as np


class RandomEmbeddingModel:
    """Deterministic random embedding model using sequence hash as seed."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_dim = embedding_dim

    def encode(self, sequence: str) -> np.ndarray:
        """Return a deterministic pseudo-random embedding for sequence."""
        digest = hashlib.md5(sequence.encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, size=(self.embedding_dim,)).astype(np.float32)
