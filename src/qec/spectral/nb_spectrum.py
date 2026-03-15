"""Deterministic NB spectral mode selection utilities."""

from __future__ import annotations

import numpy as np


def select_unstable_nb_modes(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    k_modes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top-k unstable NB modes by |eigenvalue| with deterministic ties."""
    vals = np.asarray(eigenvalues, dtype=np.complex128)
    vecs = np.asarray(eigenvectors, dtype=np.complex128)

    if vals.size == 0 or vecs.size == 0:
        return vecs[:, :0], vals[:0]

    k = max(1, int(k_modes))
    mags = np.abs(vals)
    order = np.lexsort((np.arange(len(mags), dtype=np.int64), -mags))
    selected = order[: min(k, order.size)]
    return vecs[:, selected], vals[selected]
