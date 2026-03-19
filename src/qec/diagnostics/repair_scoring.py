"""
v8.3.0 — Repair Scoring.

Scores a repair candidate by computing spectral metrics before and
after applying the proposed edge swap.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics


_ROUND = 12


def _apply_repair(
    H: np.ndarray,
    candidate: dict[str, Any],
) -> np.ndarray:
    """Apply a repair candidate to H and return the modified matrix.

    Does not mutate the input matrix.
    """
    H_new = H.copy()
    remove_vi, remove_ci = candidate["remove_edge"]
    add_vi, add_ci = candidate["add_edge"]

    H_new[remove_ci, remove_vi] = 0.0
    H_new[add_ci, add_vi] = 1.0

    return H_new


def score_repair_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
) -> dict[str, float]:
    """Score a repair candidate by comparing spectral metrics.

    Computes spectral metrics before and after the proposed repair,
    then returns the delta for each metric.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    candidate : dict[str, Any]
        Repair candidate with ``remove_edge`` and ``add_edge``.

    Returns
    -------
    dict[str, float]
        Dictionary with delta values for each spectral metric.
        Negative deltas indicate improvement (reduction).
    """
    H_arr = np.asarray(H, dtype=np.float64)

    metrics_before = compute_spectral_metrics(H_arr)
    H_repaired = _apply_repair(H_arr, candidate)
    metrics_after = compute_spectral_metrics(H_repaired)

    return {
        "delta_spectral_radius": round(
            metrics_after["spectral_radius"] - metrics_before["spectral_radius"],
            _ROUND,
        ),
        "delta_entropy": round(
            metrics_after["entropy"] - metrics_before["entropy"],
            _ROUND,
        ),
        "delta_curvature": round(
            metrics_after["curvature"] - metrics_before["curvature"],
            _ROUND,
        ),
        "delta_cycle_density": round(
            metrics_after["cycle_density"] - metrics_before["cycle_density"],
            _ROUND,
        ),
        "delta_sis": round(
            metrics_after["sis"] - metrics_before["sis"],
            _ROUND,
        ),
    }
