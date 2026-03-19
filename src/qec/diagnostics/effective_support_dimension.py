"""
v8.1.0 — Effective Support Dimension Diagnostic.

Computes the effective support dimension of the dominant NB eigenvector,
defined as the exponential of the spectral entropy:

    D_eff = exp(entropy)

This gives an effective count of edges that carry significant energy.
When the eigenvector is uniformly spread, D_eff equals the number of
directed edges.  When localized, D_eff approaches 1.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from qec.diagnostics.spectral_entropy import compute_spectral_entropy

_ROUND = 12


def compute_effective_support_dimension(H: np.ndarray) -> float:
    """Compute the effective support dimension.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Effective number of edges carrying significant energy.
        Range: [1, 2|E|].
    """
    entropy = compute_spectral_entropy(H)
    return round(float(np.exp(entropy)), _ROUND)
