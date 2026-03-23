"""Scoring and normalization helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np



def minmax_normalize(values: Iterable[float]) -> np.ndarray:
    """Min-max normalize values into [0, 1]."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if np.isclose(vmin, vmax):
        return np.ones_like(arr) * 0.5

    return (arr - vmin) / (vmax - vmin)



def combined_score(
    stability: Iterable[float],
    activity: Iterable[float],
    w_stability: float = 0.5,
    w_activity: float = 0.5,
) -> np.ndarray:
    """Compute weighted score from normalized stability and activity."""
    s_norm = minmax_normalize(stability)
    a_norm = minmax_normalize(activity)
    return w_stability * s_norm + w_activity * a_norm



def combined_score_with_structure_quality(
    stability: Iterable[float],
    activity: Iterable[float],
    uncertainty: Iterable[float],
    structure_confidence: Iterable[float],
    plddt_mean: Iterable[float],
    ptm: Iterable[float],
    pae_mean: Iterable[float],
    w_stability: float = 0.35,
    w_activity: float = 0.35,
    w_uncertainty: float = 0.10,
    w_structure: float = 0.10,
    w_plddt: float = 0.05,
    w_ptm: float = 0.03,
    w_pae: float = 0.02,
) -> np.ndarray:
    """Weighted score with uncertainty and structure quality signals."""
    s_norm = minmax_normalize(stability)
    a_norm = minmax_normalize(activity)
    u_norm = minmax_normalize(uncertainty)
    c_norm = minmax_normalize(structure_confidence)
    plddt_norm = minmax_normalize(plddt_mean)
    ptm_norm = minmax_normalize(ptm)
    pae_inv_norm = 1.0 - minmax_normalize(pae_mean)

    return (
        w_stability * s_norm
        + w_activity * a_norm
        + w_uncertainty * u_norm
        + w_structure * c_norm
        + w_plddt * plddt_norm
        + w_ptm * ptm_norm
        + w_pae * pae_inv_norm
    )
