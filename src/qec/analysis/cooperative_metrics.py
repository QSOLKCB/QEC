"""Deterministic cooperative exploration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_key(region: Any) -> tuple[float, ...]:
    return tuple(np.asarray(region, dtype=np.float64).tolist())


def agent_region_overlap(assignments: dict[str, np.ndarray | None]) -> np.float64:
    """Fraction of agents assigned duplicate regions."""
    seen: set[tuple[float, ...]] = set()
    overlap = 0
    total = 0
    for agent_id in sorted(assignments.keys()):
        region = assignments[agent_id]
        if region is None:
            continue
        total += 1
        key = _as_key(region)
        if key in seen:
            overlap += 1
        else:
            seen.add(key)
    if total == 0:
        return np.float64(0.0)
    return np.float64(overlap / float(total))


def agent_spacing_distance(assignments: dict[str, np.ndarray | None]) -> np.float64:
    """Minimum pairwise Euclidean distance among assigned regions."""
    vectors = [
        np.asarray(assignments[aid], dtype=np.float64)
        for aid in sorted(assignments.keys())
        if assignments[aid] is not None
    ]
    if len(vectors) < 2:
        return np.float64(0.0)

    min_dist = float("inf")
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            diff = vectors[i] - vectors[j]
            d = float(np.sqrt(np.dot(diff, diff)))
            if d < min_dist:
                min_dist = d
    return np.float64(min_dist)


def cooperative_coverage(
    explored_regions: list[np.ndarray | list[float] | tuple[float, ...]],
    known_regions: list[np.ndarray | list[float] | tuple[float, ...]],
) -> np.float64:
    """coverage = unique explored regions / unique known regions."""
    known = {_as_key(r) for r in known_regions if r is not None}
    explored = {_as_key(r) for r in explored_regions if r is not None}
    if not known:
        return np.float64(0.0)
    return np.float64(len(explored.intersection(known)) / float(len(known)))


def frontier_exploration_rate(
    explored_regions: list[np.ndarray | list[float] | tuple[float, ...]],
    frontier_regions: list[np.ndarray | list[float] | tuple[float, ...]],
) -> np.float64:
    """Fraction of unique frontier regions covered by explored regions."""
    frontiers = {_as_key(r) for r in frontier_regions if r is not None}
    explored = {_as_key(r) for r in explored_regions if r is not None}
    if not frontiers:
        return np.float64(0.0)
    return np.float64(len(explored.intersection(frontiers)) / float(len(frontiers)))
