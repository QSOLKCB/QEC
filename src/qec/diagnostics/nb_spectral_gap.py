"""
v8.1.0 — Non-Backtracking Spectral Gap Diagnostic.

Computes the spectral gap of the non-backtracking operator, defined as
the difference between the two largest eigenvalue magnitudes.

A large spectral gap indicates clear separation between the dominant
mode and the bulk — typically associated with stable BP convergence.
A small gap suggests near-degeneracy and potential instability.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigs

from qec.diagnostics._spectral_utils import build_nb_operator
from qec.diagnostics.spectral_nb import _TannerGraph

_ROUND = 12


def compute_nb_spectral_gap(H: np.ndarray) -> float:
    """Compute the non-backtracking spectral gap.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Spectral gap: |lambda_1| - |lambda_2|.
        Returns 0.0 if fewer than 2 eigenvalues exist.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    graph = _TannerGraph(H_arr)

    op, directed_edges = build_nb_operator(graph)
    n_edges = len(directed_edges)

    if n_edges < 2:
        return 0.0

    # Request top 2 eigenvalues by largest real part
    k = min(2, n_edges - 1)
    vals, _ = eigs(op, k=k, which="LM", tol=1e-6)

    magnitudes = np.sort(np.abs(vals))[::-1]

    if len(magnitudes) < 2:
        return 0.0

    gap = float(magnitudes[0] - magnitudes[1])
    return round(gap, _ROUND)
