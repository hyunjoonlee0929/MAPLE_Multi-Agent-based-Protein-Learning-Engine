from __future__ import annotations

import random

import numpy as np

from models.embedding_model import RandomEmbeddingModel
from utils.bo import propose_bo_mutations
from utils.pareto import select_top_by_pareto


def test_select_top_by_pareto_returns_front_members() -> None:
    points = np.asarray(
        [
            [0.9, 0.1, -0.1, 0.9],  # non-dominated
            [0.1, 0.9, -0.1, 0.9],  # non-dominated
            [0.5, 0.5, -0.5, 0.5],  # dominated
        ],
        dtype=np.float32,
    )
    selected = select_top_by_pareto(points, top_k=2)
    assert set(selected) == {0, 1}


def test_propose_bo_mutations_generates_sequences() -> None:
    rng = random.Random(7)
    embedder = RandomEmbeddingModel(embedding_dim=8)
    parents = ["MKTFFV", "MKTFFI"]
    train_embeddings = np.stack([embedder.encode(seq) for seq in ["MKTFFV", "MKTFFI", "MKTFFL"]]).astype(np.float32)
    train_scores = [0.9, 0.8, 0.7]

    proposed = propose_bo_mutations(
        parents=parents,
        train_embeddings=train_embeddings,
        train_scores=train_scores,
        embedding_model=embedder,
        num_to_generate=4,
        mutation_rate=1,
        rng=rng,
        beta=0.3,
        trials_per_parent=5,
    )
    assert 0 < len(proposed) <= 4
