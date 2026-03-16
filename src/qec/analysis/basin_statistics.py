"""Deterministic statistics for spectral basin assignments."""

from __future__ import annotations

import numpy as np


def basin_sizes(assignments: np.ndarray) -> dict[int, int]:
    """Count number of states per basin id."""
    labels = np.asarray(assignments, dtype=np.int64)
    if labels.size == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {
        int(k): int(v)
        for k, v in zip(unique.tolist(), counts.tolist())
    }
