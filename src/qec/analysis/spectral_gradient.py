"""Deterministic spectral gradient estimation from recorded trajectories."""

from __future__ import annotations

import numpy as np


def estimate_spectral_gradient(trajectory) -> np.ndarray:
    """Estimate a deterministic local gradient direction in spectral space."""
    traj = np.asarray(trajectory, dtype=np.float64)

    if traj.ndim == 1:
        traj = traj.reshape(1, -1)

    if traj.size == 0:
        return np.zeros((0,), dtype=np.float64)

    if len(traj) < 2:
        return np.zeros_like(traj[0], dtype=np.float64)

    deltas = traj[1:] - traj[:-1]
    return np.mean(deltas, axis=0, dtype=np.float64)
