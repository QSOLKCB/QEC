"""Shared deterministic test utilities for QEC test suite."""

from __future__ import annotations

import numpy as np


def simple_parity_matrix() -> np.ndarray:
    """Return a small deterministic parity check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
    ], dtype=np.float64)


def received_vector(n: int = 5) -> np.ndarray:
    """Return a deterministic received vector of length n.

    The vector is constructed by tiling a fixed base pattern, ensuring
    consistent behavior for all n >= 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    base = np.array([1.0, -1.0, 1.0, 0.0, -1.0], dtype=np.float64)
    repeats = (n + len(base) - 1) // len(base)
    tiled = np.tile(base, repeats)
    return tiled[:n]
