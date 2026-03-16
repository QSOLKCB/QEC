"""Deterministic spectrum helpers for non-backtracking matrices."""

from __future__ import annotations

import numpy as np


def leading_nb_eigenmode(B: np.ndarray) -> tuple[np.complex128, np.ndarray]:
    """Return dominant eigenvalue/eigenvector by magnitude.

    Deterministic tie-break: first occurrence in ``np.linalg.eig`` output.
    """
    B_arr = np.asarray(B, dtype=np.float64)
    if B_arr.ndim != 2 or B_arr.shape[0] != B_arr.shape[1]:
        raise ValueError("B must be a square matrix")
    if B_arr.shape[0] == 0:
        return np.complex128(0.0), np.zeros(0, dtype=np.complex128)

    vals, vecs = np.linalg.eig(B_arr)
    mags = np.abs(vals)
    idx = int(np.argmax(mags))
    return np.complex128(vals[idx]), np.asarray(vecs[:, idx], dtype=np.complex128)
