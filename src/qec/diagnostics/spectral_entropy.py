"""
v8.1.0 — Spectral Entropy Diagnostic.

Computes the Shannon entropy of the normalized edge energy distribution
from the dominant non-backtracking eigenvector.

High entropy indicates energy is spread uniformly across edges (stable).
Low entropy indicates energy is concentrated on few edges (unstable).

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


def compute_spectral_entropy(H: np.ndarray) -> float:
    """Compute spectral entropy of the NB edge energy distribution.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    float
        Shannon entropy of the normalized edge energy distribution.
        Range: [0, log(2|E|)].
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

    # Edge energy distribution: |v_e|^2
    edge_energy = np.abs(eigenvector) ** 2
    total = edge_energy.sum()

    if total == 0.0:
        return 0.0

    # Normalize to probability distribution
    p = edge_energy / total

    # Shannon entropy: -sum(p * log(p)), skip zero entries
    mask = p > 0
    entropy = -np.sum(p[mask] * np.log(p[mask]))

    return round(float(entropy), _ROUND)
