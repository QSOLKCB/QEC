"""Deterministic basin escape-direction utilities."""

from __future__ import annotations

import numpy as np


def estimate_escape_direction(
    current_state: np.ndarray | list[float],
    basin_center: np.ndarray | list[float],
    other_centers: np.ndarray | list[list[float]] | list[np.ndarray] | None = None,
) -> np.ndarray:
    """Compute a unit vector for boundary-aware escape from current basin."""
    current = np.asarray(current_state, dtype=np.float64)
    center = np.asarray(basin_center, dtype=np.float64)

    # default radial escape
    direction = current - center

    if other_centers is not None and len(other_centers) > 0:
        centers = np.asarray(other_centers, dtype=np.float64)
        dists2 = np.sum((centers - current) ** 2, axis=1)
        nearest = centers[int(np.argmin(dists2))]
        boundary_dir = nearest - center
        if float(np.linalg.norm(boundary_dir)) > 0.0:
            direction = boundary_dir

    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros_like(current, dtype=np.float64)
    return (direction / norm).astype(np.float64, copy=False)
