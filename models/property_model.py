"""Property predictor with PyTorch primary path and NumPy fallback."""

from __future__ import annotations

import numpy as np


try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None
    nn = object
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class PropertyPredictor(nn.Module):
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

else:

    class PropertyPredictor:
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
