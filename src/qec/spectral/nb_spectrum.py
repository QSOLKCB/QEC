"""Deterministic non-backtracking spectrum helpers."""

from __future__ import annotations

import numpy as np

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
