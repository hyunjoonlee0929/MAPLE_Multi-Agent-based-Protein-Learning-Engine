from __future__ import annotations

import numpy as np

from utils.calibration import regression_ece


def test_regression_ece_returns_non_negative_value() -> None:
    y_true = np.array([[0.5, 0.1], [0.4, 0.2], [0.3, 0.3]], dtype=np.float32)
    y_pred = np.array([[0.45, 0.15], [0.35, 0.25], [0.25, 0.35]], dtype=np.float32)
    unc = np.array([0.05, 0.06, 0.08], dtype=np.float32)

    out = regression_ece(y_true, y_pred, unc, num_bins=4)
    assert "ece" in out
    assert float(out["ece"]) >= 0.0
    assert len(out["bins"]) > 0
