"""Deterministic exploration-state classification for basin trajectories."""

from __future__ import annotations

import numpy as np


def analyze_exploration_state(
    trajectory: np.ndarray | list[list[float]] | list[float],
    basin_assignments: list[int] | np.ndarray,
    window: int = 10,
) -> str:
    """Classify recent search behavior into an exploration state.

    Parameters
    ----------
    trajectory : array-like
        Recent spectral trajectory samples.
    basin_assignments : list[int] | np.ndarray
        Basin ID sequence by generation.
    window : int
        Classification lookback window.
    """
    w = int(max(1, window))
    assignments = np.asarray(basin_assignments, dtype=np.int64)
    traj = np.asarray(trajectory, dtype=np.float64)

    if assignments.size < w or traj.shape[0] < w:
        return "GLOBAL_EXPLORATION"

    recent_assignments = assignments[-w:]
    if np.all(recent_assignments == recent_assignments[0]):
        return "BASIN_STAGNATION"

    if recent_assignments.size >= 2 and recent_assignments[-1] != recent_assignments[-2]:
        return "BASIN_TRANSITION"

    return "LOCAL_OPTIMIZATION"


__all__ = ["analyze_exploration_state"]
