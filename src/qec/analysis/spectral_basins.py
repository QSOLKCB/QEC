"""Deterministic spectral basin identification utilities."""

from __future__ import annotations

import numpy as np


def identify_spectral_basins(
    trajectory: np.ndarray,
    threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition trajectory states into distance-threshold spectral basins.

    The algorithm is deterministic and order-preserving:
    each state is assigned to the first existing basin center within
    ``threshold`` Euclidean distance; otherwise a new basin is created
    with that state as center.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError("trajectory must be a 2D array")
    if traj.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, traj.shape[1]), dtype=np.float64)

    basins: list[np.ndarray] = []
    assignments: list[int] = []
    distance_threshold = float(threshold)
    threshold2 = distance_threshold * distance_threshold

    capacity = 16
    center_dim = traj.shape[1]
    centers_array = np.empty((capacity, center_dim), dtype=np.float64)
    num_centers = 0

    for state in traj:
        if num_centers > 0:
            active_centers = centers_array[:num_centers]
            dists2 = np.sum((active_centers - state) ** 2, axis=1)
            within = dists2 < threshold2
            if np.any(within):
                assignments.append(int(np.flatnonzero(within)[0]))
                continue

        if num_centers == capacity:
            capacity *= 2
            new_centers = np.empty((capacity, center_dim), dtype=np.float64)
            new_centers[:num_centers] = centers_array[:num_centers]
            centers_array = new_centers

        centers_array[num_centers] = state
        basins.append(np.asarray(state, dtype=np.float64))
        assignments.append(num_centers)
        num_centers += 1

    basin_centers = centers_array[:num_centers]
    return (
        np.asarray(assignments, dtype=np.int64),
        np.asarray(basin_centers, dtype=np.float64),
    )
