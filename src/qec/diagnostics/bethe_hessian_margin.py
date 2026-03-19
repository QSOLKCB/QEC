"""
v8.1.0 — Bethe Hessian Margin Diagnostic.

Computes the margin of the smallest Bethe Hessian eigenvalue from zero.
A negative margin indicates structural degeneracy in the Tanner graph
that can cause BP convergence failure.

Reuses the existing ``compute_bethe_hessian`` implementation.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from qec.diagnostics.bethe_hessian import compute_bethe_hessian

_ROUND = 12


def compute_bethe_hessian_margin(H: np.ndarray) -> float:
    """Compute the Bethe Hessian margin.

    The margin is the smallest eigenvalue of the Bethe Hessian matrix.
    Negative values indicate detectable community structure below the
    spectral threshold, signaling potential BP instability.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Smallest eigenvalue of the Bethe Hessian.
    """
    result = compute_bethe_hessian(H)
    return round(float(result["min_eigenvalue"]), _ROUND)
