"""Property agent for embedding generation and property prediction."""

from __future__ import annotations

import numpy as np

from models.embedding_model import RandomEmbeddingModel
from models.property_model import PropertyPredictor


class PropertyAgent:
    """Predicts stability and activity from sequence embeddings."""

    def __init__(self, embedding_dim: int = 128) -> None:
        self.embedding_model = RandomEmbeddingModel(embedding_dim=embedding_dim)
        self.predictor = PropertyPredictor(embedding_dim=embedding_dim)

    def run(self, state: dict) -> dict:
        sequences = state.get("sequences", [])
        embeddings = [self.embedding_model.encode(seq) for seq in sequences]

        if embeddings:
            batch = np.stack(embeddings).astype(np.float32)
            preds = np.asarray(self.predictor.predict(batch), dtype=np.float32)
            properties = [
                {
                    "stability": float(item[0]),
                    "activity": float(item[1]),
                }
                for item in preds
            ]
        else:
            properties = []

        state["embeddings"] = embeddings
        state["properties"] = properties
        return state
