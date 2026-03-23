"""Property predictor with PyTorch primary path and NumPy fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = object
    TORCH_AVAILABLE = False


class PropertyPredictorLike(Protocol):
    """Protocol for property prediction backends."""

    def predict(self, embedding_batch):
        """Return Nx2 array where cols are [stability, activity]."""


if TORCH_AVAILABLE:

    class TorchPropertyPredictor(nn.Module):
        """Small MLP for predicting stability and activity."""

        def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
            super().__init__()
            torch.manual_seed(42)
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

        @torch.no_grad()
        def predict(self, embedding_batch):
            self.eval()
            if isinstance(embedding_batch, np.ndarray):
                embedding_batch = torch.from_numpy(embedding_batch).float()
            return self.forward(embedding_batch).cpu().numpy()

        def load_checkpoint(self, checkpoint_path: str | Path) -> None:
            state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
            self.load_state_dict(state_dict)


class NumpyPropertyPredictor:
    """NumPy fallback predictor when PyTorch is unavailable."""

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 64) -> None:
        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, 0.1, size=(embedding_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = rng.normal(0, 0.1, size=(hidden_dim, 2))
        self.b2 = np.zeros((2,), dtype=np.float32)

    def predict(self, embedding_batch):
        x = np.asarray(embedding_batch, dtype=np.float32)
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


class PropertyPredictor:
    """Factory-style predictor wrapper with optional checkpoint loading."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        checkpoint_path: str | None = None,
    ) -> None:
        if TORCH_AVAILABLE:
            backend = TorchPropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            if checkpoint_path:
                backend.load_checkpoint(checkpoint_path)
            self.backend: PropertyPredictorLike = backend
        else:
            if checkpoint_path:
                raise RuntimeError(
                    "checkpoint_path requires PyTorch runtime, but torch is not installed"
                )
            self.backend = NumpyPropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )

    def predict(self, embedding_batch):
        return self.backend.predict(embedding_batch)
