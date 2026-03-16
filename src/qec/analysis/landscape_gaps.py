"""v43.0.0 — Deterministic spectral landscape gap detection."""

from __future__ import annotations

import numpy as np


def _centers_from_memory(memory) -> np.ndarray:
    """Extract landscape centers as float64 array from memory-like input."""
    if hasattr(memory, "centers") and callable(memory.centers):
        centers = memory.centers()
    else:
        centers = memory

    centers_arr = np.asarray(centers, dtype=np.float64)
    if centers_arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if centers_arr.ndim == 1:
        centers_arr = centers_arr.reshape(1, -1)
    return centers_arr


def detect_landscape_gaps(memory, gap_radius: float, max_gaps: int) -> list[np.ndarray]:
    """Detect under-explored spectral targets from memory center separations.

    Deterministic algorithm:
    1) read center vectors from ``memory.centers()``;
    2) compute pairwise center distances;
    3) keep pairs whose separation exceeds ``gap_radius``;
    4) emit midpoint candidates in deterministic distance-descending order.
    """
    centers = _centers_from_memory(memory)
    if centers.shape[0] < 2:
        return []

    radius_sq = float(gap_radius) * float(gap_radius)
    pairs: list[tuple[float, int, int]] = []
    n = int(centers.shape[0])

    for i in range(n - 1):
        ci = centers[i]
        for j in range(i + 1, n):
            diff = ci - centers[j]
            dist_sq = float(np.dot(diff, diff))
            if dist_sq > radius_sq:
                pairs.append((dist_sq, i, j))

    if not pairs:
        return []

    ordered_pairs = sorted(pairs, key=lambda x: (-x[0], x[1], x[2]))

    gaps: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for _, i, j in ordered_pairs:
        midpoint = ((centers[i] + centers[j]) * 0.5).astype(np.float64, copy=False)
        key = tuple(float(v) for v in midpoint.tolist())
        if key in seen:
            continue
        seen.add(key)
        gaps.append(midpoint)
        if len(gaps) >= int(max_gaps):
            break
    return gaps
