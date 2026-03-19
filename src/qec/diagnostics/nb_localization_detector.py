"""
v8.0.0 — NB Eigenvector Localization Detector.

Detects localization of the dominant non-backtracking eigenvector
on Tanner graph substructures.  Uses IPR and participation ratio
thresholds to classify eigenvector components as localized or
delocalized.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Does not import from experiments (Layer 5) or bench (Layer 6).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import compute_nb_spectrum


_ROUND = 12


def detect_nb_eigenvector_localization(
    H: np.ndarray,
    *,
    ipr_threshold: float = 0.1,
    energy_fraction_threshold: float = 0.5,
) -> dict[str, Any]:
    """Detect localization of the dominant NB eigenvector.

    Identifies directed edges where eigenvector energy is concentrated
    beyond what a delocalized eigenvector would exhibit.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    ipr_threshold : float
        IPR above this value indicates localization.
    energy_fraction_threshold : float
        Fraction of total energy concentrated in top-k edges
        above which localization is declared.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``is_localized`` : bool
        - ``ipr`` : float
        - ``eeec`` : float
        - ``num_localized_edges`` : int
        - ``localized_edge_indices`` : list[int]
        - ``max_edge_energy`` : float
        - ``mean_edge_energy`` : float
        - ``localization_ratio`` : float
    """
    spectrum = compute_nb_spectrum(H)

    ipr = spectrum["ipr"]
    eeec = spectrum["eeec"]
    edge_energy = spectrum["edge_energy"]

    n_edges = len(edge_energy)
    if n_edges == 0:
        return {
            "is_localized": False,
            "ipr": 0.0,
            "eeec": 0.0,
            "num_localized_edges": 0,
            "localized_edge_indices": [],
            "max_edge_energy": 0.0,
            "mean_edge_energy": 0.0,
            "localization_ratio": 0.0,
        }

    total_energy = float(edge_energy.sum())
    mean_energy = total_energy / n_edges if n_edges > 0 else 0.0

    # Threshold: edges with energy > 2 * mean are considered localized
    threshold = 2.0 * mean_energy
    localized_indices = []
    for i in range(n_edges):
        if edge_energy[i] > threshold:
            localized_indices.append(i)

    # Sort deterministically
    localized_indices.sort()

    localized_energy = sum(float(edge_energy[i]) for i in localized_indices)
    localization_ratio = localized_energy / total_energy if total_energy > 0 else 0.0

    is_localized = (
        ipr > ipr_threshold
        or localization_ratio > energy_fraction_threshold
    )

    max_energy = float(np.max(edge_energy)) if n_edges > 0 else 0.0

    return {
        "is_localized": bool(is_localized),
        "ipr": round(float(ipr), _ROUND),
        "eeec": round(float(eeec), _ROUND),
        "num_localized_edges": len(localized_indices),
        "localized_edge_indices": localized_indices,
        "max_edge_energy": round(max_energy, _ROUND),
        "mean_edge_energy": round(mean_energy, _ROUND),
        "localization_ratio": round(localization_ratio, _ROUND),
    }
