"""Deterministic frontier detection in spectral landscape memory."""

from __future__ import annotations

from typing import Any

import numpy as np


def _extract_points(memory: Any, key: str) -> list[np.ndarray]:
    if memory is None:
        return []
    if isinstance(memory, dict):
        values = memory.get(key, [])
        return [np.asarray(v, dtype=np.float64) for v in values]
    values = getattr(memory, key, [])
    return [np.asarray(v, dtype=np.float64) for v in values]


def _nearest_center_distance_sq(point: np.ndarray, centers: list[np.ndarray]) -> float:
    if not centers:
        return float("inf")
    best = float("inf")
    for center in centers:
        diff = point - center
        dist_sq = float(np.dot(diff, diff))
        if dist_sq < best:
            best = dist_sq
    return best


def detect_spectral_frontiers(memory: Any, threshold: float) -> list[np.ndarray]:
    """Return candidate frontier spectra farther than threshold from known centers."""
    centers = _extract_points(memory, "region_centers")
    if centers is None or len(centers) == 0:
        return []

    candidates = _extract_points(memory, "candidate_regions")
    if not candidates:
        candidates = list(centers)

    threshold_sq = float(threshold) * float(threshold)
    frontiers: list[tuple[int, np.ndarray]] = []
    for idx, point in enumerate(candidates):
        if _nearest_center_distance_sq(point, centers) > threshold_sq:
            frontiers.append((idx, np.asarray(point, dtype=np.float64)))

    # Deterministic order: by original candidate index.
    return [point for _, point in sorted(frontiers, key=lambda kv: kv[0])]
