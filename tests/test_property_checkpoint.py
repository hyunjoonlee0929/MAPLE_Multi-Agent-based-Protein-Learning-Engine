from __future__ import annotations

from pathlib import Path

import numpy as np

from models.property_model import PropertyPredictor



def test_property_predictor_loads_numpy_linear_checkpoint(tmp_path: Path) -> None:
    weights = np.array([[2.0, -1.0], [0.5, 1.5]], dtype=np.float32)
    bias = np.array([0.1, -0.2], dtype=np.float32)

    ckpt = tmp_path / "property_linear.npz"
    np.savez(ckpt, model_type="numpy_linear", embedding_dim=np.int32(2), weights=weights, bias=bias)

    predictor = PropertyPredictor(
        embedding_dim=2,
        checkpoint_path=str(ckpt),
        uncertainty_samples=1,
        uncertainty_noise=0.0,
    )

    x = np.array([[1.0, 2.0]], dtype=np.float32)
    preds, unc = predictor.predict_with_uncertainty(x)

    expected = x @ weights + bias
    assert np.allclose(preds, expected)
    assert float(unc[0]) == 0.0
