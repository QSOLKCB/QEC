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


def minimal_parity_matrix_3x5() -> np.ndarray:
    """Canonical minimal parity-check matrix (3x5) for fast structural tests."""
    return np.array([
        [1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ], dtype=np.float64)


def minimal_parity_matrix_4x6() -> np.ndarray:
    """Slightly richer matrix for degeneracy / edge-case testing."""
    return np.array([
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 0],
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


# --- Ternary assertion layer ---

def to_ternary(arr: np.ndarray) -> np.ndarray:
    """Map values to ternary domain: +1 = correct, 0 = neutral, -1 = error."""
    out = np.zeros_like(arr, dtype=np.int8)
    out[arr > 0] = 1
    out[arr < 0] = -1
    return out


def assert_no_ternary_errors(arr: np.ndarray) -> None:
    """Fail if any -1 present in ternary mapping."""
    tern = to_ternary(arr)
    assert not np.any(tern == -1), "Detected ternary error state"


def assert_strict_ternary_success(arr: np.ndarray) -> None:
    """All values must map to +1 in ternary domain."""
    tern = to_ternary(arr)
    assert np.min(tern) == 1, "Not all entries are strictly valid"


# --- Deterministic lightweight caching ---

_cache: dict[str, np.ndarray] = {}


def deterministic_array_cache(key: str, arr_fn: object) -> np.ndarray:
    """Cache deterministic array-producing functions.

    Returns a read-only cached copy to prevent mutation.
    """
    if key in _cache:
        return _cache[key]
    arr = arr_fn()
    arr.setflags(write=False)
    _cache[key] = arr
    return arr
