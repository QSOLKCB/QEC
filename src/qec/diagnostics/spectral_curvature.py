"""
v8.1.0 — Spectral Curvature Diagnostic.

Computes spectral curvature as the variance of the log-energy
distribution of the dominant NB eigenvector.

High curvature indicates sharp peaks in the energy landscape,
suggesting localized instability modes.  Low curvature indicates
a smooth energy profile associated with stable BP convergence.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

import numpy as np

from qec.diagnostics._spectral_utils import (
    compute_nb_dominant_eigenpair,
)
from qec.diagnostics.spectral_nb import _TannerGraph

_ROUND = 12


def compute_spectral_curvature(H: np.ndarray) -> float:
    """Compute spectral curvature of the NB edge energy distribution.

    Curvature is the variance of log(|v_e|^2) over edges with
    nonzero energy.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Variance of the log-energy distribution.
        Returns 0.0 if fewer than 2 edges have nonzero energy.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    graph = _TannerGraph(H_arr)

    spectral_radius, eigenvector, directed_edges = (
        compute_nb_dominant_eigenpair(graph)
    )

    if len(eigenvector) == 0:
        return 0.0

    # Normalize eigenvector
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm

    # Edge energy: |v_e|^2
    edge_energy = np.abs(eigenvector) ** 2

    # Filter nonzero entries for log computation
    nonzero = edge_energy[edge_energy > 0]

    if len(nonzero) < 2:
        return 0.0

    log_energy = np.log(nonzero)
    curvature = float(np.var(log_energy))

    return round(curvature, _ROUND)
