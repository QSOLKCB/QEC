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
    """Return a simple deterministic received vector."""
    return np.array([1.0, -1.0, 1.0, 0.0, -1.0], dtype=np.float64)[:n]
