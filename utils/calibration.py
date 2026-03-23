"""Calibration utilities for regression uncertainty."""

from __future__ import annotations

import numpy as np


def regression_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    num_bins: int = 10,
) -> dict:
    """Compute ECE-like calibration for regression.

    We compare mean predicted uncertainty vs mean absolute error per uncertainty bin.
    """
    yt = np.asarray(y_true, dtype=np.float32)
    yp = np.asarray(y_pred, dtype=np.float32)
    unc = np.asarray(uncertainty, dtype=np.float32).reshape(-1)

    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.ndim != 2:
        raise ValueError("y_true/y_pred must be 2D arrays")
    if yt.shape[0] != unc.shape[0]:
        raise ValueError("uncertainty length must match sample count")

    sample_error = np.mean(np.abs(yp - yt), axis=1)
    num_bins = max(1, int(num_bins))

    lo = float(np.min(unc))
    hi = float(np.max(unc))
    if hi <= lo + 1e-12:
        mean_u = float(np.mean(unc))
        mean_e = float(np.mean(sample_error))
        return {
            "ece": float(abs(mean_u - mean_e)),
            "num_bins": 1,
            "bins": [
                {
                    "start": lo,
                    "end": hi,
                    "count": int(len(unc)),
                    "mean_uncertainty": mean_u,
                    "mean_abs_error": mean_e,
                    "gap": float(abs(mean_u - mean_e)),
                }
            ],
        }

    edges = np.linspace(lo, hi, num_bins + 1)
    total = float(len(unc))
    ece = 0.0
    bins = []
    for b in range(num_bins):
        start = float(edges[b])
        end = float(edges[b + 1])
        if b == num_bins - 1:
            mask = (unc >= start) & (unc <= end)
        else:
            mask = (unc >= start) & (unc < end)
        count = int(np.sum(mask))
        if count == 0:
            continue
        mean_u = float(np.mean(unc[mask]))
        mean_e = float(np.mean(sample_error[mask]))
        gap = float(abs(mean_u - mean_e))
        ece += (float(count) / total) * gap
        bins.append(
            {
                "start": start,
                "end": end,
                "count": count,
                "mean_uncertainty": mean_u,
                "mean_abs_error": mean_e,
                "gap": gap,
            }
        )

    return {
        "ece": float(ece),
        "num_bins": int(num_bins),
        "bins": bins,
    }
