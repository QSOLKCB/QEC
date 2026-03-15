"""Deterministic non-backtracking spectrum helpers."""

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
_ROUND = 12


def compute_nb_spectral_gap(eigenvalues):
    """Return dominant non-backtracking spectral gap with deterministic rounding."""
    ev = np.asarray(eigenvalues, dtype=np.complex128)
    mags = np.abs(ev)
    if len(mags) < 2:
        return 0.0
    order = np.argsort(-mags)
    gap = float(mags[order[0]] - mags[order[1]])
    return round(gap, _ROUND)
