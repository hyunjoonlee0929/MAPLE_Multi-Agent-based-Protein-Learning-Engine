"""Lightweight BO-style candidate proposal utilities."""

from __future__ import annotations

import random

import numpy as np

from utils.mutation import random_mutation


def _fit_linear_surrogate(x: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-3) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    n, d = x.shape
    x_aug = np.concatenate([x, np.ones((n, 1), dtype=np.float32)], axis=1)

    eye = np.eye(d + 1, dtype=np.float32)
    eye[-1, -1] = 0.0
    lhs = x_aug.T @ x_aug + float(ridge_alpha) * eye
    rhs = x_aug.T @ y
    params = np.linalg.solve(lhs, rhs).reshape(-1)

    w = params[:-1]
    b = float(params[-1])
    return w.astype(np.float32), b


def _predict_linear(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32) + float(b)


def _distance_uncertainty(x: np.ndarray, train_x: np.ndarray) -> np.ndarray:
    """Novelty proxy: nearest-neighbor distance to observed embeddings."""
    x = np.asarray(x, dtype=np.float32)
    train_x = np.asarray(train_x, dtype=np.float32)
    if train_x.size == 0:
        return np.ones((x.shape[0],), dtype=np.float32)

    dists = []
    for row in x:
        diff = train_x - row
        nn = float(np.min(np.sqrt(np.sum(diff * diff, axis=1))))
        dists.append(nn)
    arr = np.asarray(dists, dtype=np.float32)
    scale = float(np.mean(arr)) + 1e-6
    return arr / scale


def propose_bo_mutations(
    parents: list[str],
    train_embeddings: np.ndarray,
    train_scores: list[float],
    embedding_model,
    num_to_generate: int,
    mutation_rate: int,
    rng: random.Random,
    beta: float = 0.30,
    trials_per_parent: int = 8,
) -> list[str]:
    """Generate mutations scored by mean+beta*uncertainty acquisition."""
    if num_to_generate <= 0 or not parents:
        return []

    train_x = np.asarray(train_embeddings, dtype=np.float32)
    y = np.asarray(train_scores, dtype=np.float32)
    if train_x.ndim != 2 or train_x.shape[0] == 0 or train_x.shape[0] != y.shape[0]:
        return []

    w, b = _fit_linear_surrogate(train_x, y, ridge_alpha=1e-3)

    candidates: list[tuple[str, float]] = []
    for parent in parents:
        for _ in range(max(1, int(trials_per_parent))):
            child = random_mutation(parent, num_mutations=max(1, int(mutation_rate)), rng=rng)
            emb = np.asarray(embedding_model.encode(child), dtype=np.float32).reshape(1, -1)
            mean_pred = float(_predict_linear(emb, w, b)[0])
            unc = float(_distance_uncertainty(emb, train_x)[0])
            acquisition = mean_pred + float(beta) * unc
            candidates.append((child, acquisition))

    best_by_seq: dict[str, float] = {}
    for seq, acq in candidates:
        prev = best_by_seq.get(seq)
        if prev is None or acq > prev:
            best_by_seq[seq] = acq

    ranked = sorted(best_by_seq.items(), key=lambda x: x[1], reverse=True)
    return [seq for seq, _ in ranked[:num_to_generate]]
