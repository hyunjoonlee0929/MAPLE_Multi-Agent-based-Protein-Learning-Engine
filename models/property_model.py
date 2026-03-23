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
    """NumPy predictor supporting random init or NPZ checkpoint loading."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        weights: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> None:
        self.mode = "mlp"

        if weights is not None and bias is not None:
            self.mode = "linear"
            self.linear_weights = np.asarray(weights, dtype=np.float32)
            self.linear_bias = np.asarray(bias, dtype=np.float32)
            return

        rng = np.random.default_rng(42)
        self.w1 = rng.normal(0, 0.1, size=(embedding_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.w2 = rng.normal(0, 0.1, size=(hidden_dim, 2)).astype(np.float32)
        self.b2 = np.zeros((2,), dtype=np.float32)

    @classmethod
    def from_npz(cls, checkpoint_path: str | Path) -> "NumpyPropertyPredictor":
        data = np.load(Path(checkpoint_path), allow_pickle=False)
        if "weights" not in data or "bias" not in data:
            raise ValueError("NPZ checkpoint must include 'weights' and 'bias'")
        return cls(weights=data["weights"], bias=data["bias"])

    def predict(self, embedding_batch):
        x = np.asarray(embedding_batch, dtype=np.float32)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)

        if self.mode == "linear":
            return x @ self.linear_weights + self.linear_bias

        h = np.maximum(0.0, x @ self.w1 + self.b1)
        return h @ self.w2 + self.b2


class PropertyPredictor:
    """Factory-style predictor wrapper with optional checkpoint loading."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        checkpoint_path: str | None = None,
        uncertainty_samples: int = 5,
        uncertainty_noise: float = 0.02,
    ) -> None:
        self.uncertainty_samples = max(1, uncertainty_samples)
        self.uncertainty_noise = max(0.0, uncertainty_noise)

        ckpt = str(checkpoint_path).strip() if checkpoint_path else ""
        use_numpy_npz = ckpt.endswith(".npz")

        if use_numpy_npz:
            self.backend: PropertyPredictorLike = NumpyPropertyPredictor.from_npz(ckpt)
            return

        if TORCH_AVAILABLE:
            backend = TorchPropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )
            if ckpt:
                backend.load_checkpoint(ckpt)
            self.backend = backend
        else:
            if ckpt:
                raise RuntimeError(
                    "Non-NPZ checkpoint_path requires PyTorch runtime, but torch is not installed"
                )
            self.backend = NumpyPropertyPredictor(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )

    def predict(self, embedding_batch):
        return self.backend.predict(embedding_batch)

    def predict_with_uncertainty(self, embedding_batch) -> tuple[np.ndarray, np.ndarray]:
        """Estimate predictive uncertainty via noisy input Monte Carlo passes."""
        x = np.asarray(embedding_batch, dtype=np.float32)
        if x.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

        if self.uncertainty_samples <= 1 or self.uncertainty_noise <= 0:
            preds = np.asarray(self.predict(x), dtype=np.float32)
            unc = np.zeros((preds.shape[0],), dtype=np.float32)
            return preds, unc

        samples = []
        for _ in range(self.uncertainty_samples):
            noise = np.random.normal(0.0, self.uncertainty_noise, size=x.shape).astype(np.float32)
            sample = np.asarray(self.predict(x + noise), dtype=np.float32)
            samples.append(sample)

        stacked = np.stack(samples, axis=0)
        mean_preds = np.mean(stacked, axis=0)
        unc = np.mean(np.std(stacked, axis=0), axis=1)
        return mean_preds.astype(np.float32), unc.astype(np.float32)
