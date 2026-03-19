"""
v9.5.0 — Spectral Mutation Pressure Map.

Computes per-edge mutation pressure from the non-backtracking dominant
eigenvector.  Edges with high spectral energy are candidates for
guided mutation.

Layer 3 — Discovery.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import compute_nb_spectrum


def compute_spectral_mutation_pressure(
    H: np.ndarray,
) -> dict[str, Any]:
    """Compute per-edge spectral mutation pressure.

    Uses the magnitude of the dominant non-backtracking eigenvector
    to estimate mutation pressure on each directed edge.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``edge_pressure`` : np.ndarray — normalized pressure per
          directed edge.
        - ``max_pressure`` : float — maximum pressure value.
    """
    spectrum = compute_nb_spectrum(H)
    leading_vec = spectrum["eigenvector"]

    pressure = np.abs(leading_vec)

    # Normalize to [0, 1]
    total = pressure.sum()
    if total > 0:
        pressure = pressure / total

    max_pressure = float(pressure.max()) if len(pressure) > 0 else 0.0

    return {
        "edge_pressure": pressure,
        "max_pressure": max_pressure,
    }
