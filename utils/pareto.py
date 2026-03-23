"""Pareto utilities for multi-objective optimization."""

from __future__ import annotations

import math

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True when vector a Pareto-dominates vector b for maximization."""
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_sort(points: np.ndarray) -> list[list[int]]:
    """Compute Pareto fronts for maximization objectives."""
    n = int(points.shape[0])
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    dominated_count = [0 for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(points[i], points[j]):
                dominates_list[i].append(j)
            elif dominates(points[j], points[i]):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            fronts[0].append(i)

    f_idx = 0
    while f_idx < len(fronts) and fronts[f_idx]:
        next_front: list[int] = []
        for p in fronts[f_idx]:
            for q in dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        f_idx += 1

    return fronts


def crowding_distance(points: np.ndarray, indices: list[int]) -> dict[int, float]:
    """Compute NSGA-II style crowding distance for one front."""
    if not indices:
        return {}
    if len(indices) <= 2:
        return {idx: math.inf for idx in indices}

    m = int(points.shape[1])
    dist = {idx: 0.0 for idx in indices}
    subset = points[indices]

    for obj in range(m):
        order = np.argsort(subset[:, obj])
        sorted_indices = [indices[int(i)] for i in order]
        min_v = float(points[sorted_indices[0], obj])
        max_v = float(points[sorted_indices[-1], obj])
        denom = max(max_v - min_v, 1e-12)

        dist[sorted_indices[0]] = math.inf
        dist[sorted_indices[-1]] = math.inf
        for k in range(1, len(sorted_indices) - 1):
            if math.isinf(dist[sorted_indices[k]]):
                continue
            prev_v = float(points[sorted_indices[k - 1], obj])
            next_v = float(points[sorted_indices[k + 1], obj])
            dist[sorted_indices[k]] += (next_v - prev_v) / denom

    return dist


def select_top_by_pareto(points: np.ndarray, top_k: int) -> list[int]:
    """Select top_k indices by Pareto front rank and crowding diversity."""
    if points.size == 0 or top_k <= 0:
        return []

    fronts = non_dominated_sort(points)
    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= top_k:
            chosen.extend(front)
            continue

        remaining = top_k - len(chosen)
        if remaining <= 0:
            break
        distances = crowding_distance(points, front)
        ranked = sorted(front, key=lambda i: (-distances.get(i, 0.0), i))
        chosen.extend(ranked[:remaining])
        break

    return chosen
