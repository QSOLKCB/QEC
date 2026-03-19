"""
v8.2.0 — Spectral-Only Critical Line Estimator.

Predicts the BP critical spectral radius using only NB spectral
information (no BP decoding required).

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import compute_nb_spectrum
from qec.diagnostics._spectral_utils import compute_ipr


_ROUND = 12


def predict_spectral_critical_radius(
    H: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> dict[str, Any]:
    """Predict the BP critical spectral radius from the NB spectrum.

    Combines the dominant eigenvalue, IPR, and spectral entropy:

        lambda_pred = lambda_nb * (1 - alpha * IPR + beta * H)

    where H is the Shannon entropy of the edge energy distribution.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    alpha : float
        IPR weighting coefficient.
    beta : float
        Entropy weighting coefficient.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``predicted_critical_radius``.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    spectrum = compute_nb_spectrum(H_arr)

    lambda_nb = spectrum["spectral_radius"]
    eigenvector = spectrum["eigenvector"]
    ipr = compute_ipr(eigenvector)

    # Spectral entropy
    edge_energy = np.abs(eigenvector) ** 2
    total = edge_energy.sum()
    if total > 0:
        p = edge_energy / total
        mask = p > 0
        entropy = float(-np.sum(p[mask] * np.log(p[mask])))
    else:
        entropy = 0.0

    lambda_pred = lambda_nb * (1.0 - alpha * ipr + beta * entropy)

    return {
        "predicted_critical_radius": round(float(lambda_pred), _ROUND),
    }
