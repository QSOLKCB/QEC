"""Deterministic exploration metrics for basin-navigation diagnostics."""

from __future__ import annotations

import numpy as np


def basin_switch_rate(assignments: list[int] | np.ndarray, window: int = 10) -> float:
    """Return switch-rate over the trailing assignment window."""
    w = int(max(1, window))
    arr = np.asarray(assignments, dtype=np.int64)
    if arr.size <= 1:
        return 0.0
    recent = arr[-w:]
    if recent.size <= 1:
        return 0.0
    switches = np.count_nonzero(recent[1:] != recent[:-1])
    denom = float(max(1, recent.size - 1))
    return float(np.float64(switches) / np.float64(denom))


def mean_basin_duration(assignments: list[int] | np.ndarray) -> float:
    """Return mean contiguous basin duration based on assignment runs."""
    arr = np.asarray(assignments, dtype=np.int64)
    if arr.size == 0:
        return 0.0

    durations: list[float] = []
    run_len = 1
    for idx in range(1, int(arr.size)):
        if int(arr[idx]) == int(arr[idx - 1]):
            run_len += 1
        else:
            durations.append(float(run_len))
            run_len = 1
    durations.append(float(run_len))

    return float(np.mean(np.asarray(durations, dtype=np.float64), dtype=np.float64))


def exploration_entropy(assignments: list[int] | np.ndarray) -> float:
    """Compute normalized entropy over visited basin assignments."""
    arr = np.asarray(assignments, dtype=np.int64)
    if arr.size == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    probs = counts.astype(np.float64) / np.float64(arr.size)
    entropy = -np.sum(probs * np.log2(np.maximum(probs, np.finfo(np.float64).tiny)), dtype=np.float64)
    k = int(counts.size)
    if k <= 1:
        return 0.0
    return float(np.float64(entropy) / np.log2(np.float64(k)))


__all__ = ["basin_switch_rate", "mean_basin_duration", "exploration_entropy"]
