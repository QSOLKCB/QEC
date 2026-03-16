"""Deterministic basin transition detection."""

from __future__ import annotations

import numpy as np


def detect_basin_transitions(assignments: np.ndarray) -> list[int]:
    """Return trajectory indices where basin assignment changes."""
    labels = np.asarray(assignments, dtype=np.int64)
    transitions: list[int] = []
    for idx in range(1, labels.shape[0]):
        if int(labels[idx]) != int(labels[idx - 1]):
            transitions.append(int(idx))
    return transitions
