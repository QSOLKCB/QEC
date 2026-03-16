"""Deterministic spectral target proposal for basin escape."""

from __future__ import annotations

import numpy as np


def propose_escape_step(
    current_spectrum: np.ndarray | list[float],
    escape_direction: np.ndarray | list[float],
    step: float = 0.3,
) -> np.ndarray:
    """Generate a spectral target outside the current basin."""
    current = np.asarray(current_spectrum, dtype=np.float64)
    direction = np.asarray(escape_direction, dtype=np.float64)
    return (current + float(step) * direction).astype(np.float64, copy=False)
