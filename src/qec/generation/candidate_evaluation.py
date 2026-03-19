"""
v8.4.0 — Spectral Candidate Evaluation.

Evaluates a candidate Tanner graph by computing spectral stability
metrics and classifying its predicted BP regime.

Layer 3 — Generation.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics.stability_classifier import (
    classify_from_parity_check,
    classify_tanner_graph_stability,
)


_ROUND = 12

_REGIME_LABELS = {
    1: "stable",
    0: "metastable",
    -1: "unstable",
}


def _instability_score(metrics: dict[str, Any]) -> float:
    """Compute a scalar instability score from spectral metrics.

    Higher score = more unstable.  Combines spectral_radius and SIS
    (higher is worse) minus entropy and bethe_margin (higher is better).
    """
    return round(
        metrics["spectral_radius"]
        + metrics["sis"]
        - metrics["entropy"]
        - metrics["bethe_margin"],
        _ROUND,
    )


def evaluate_tanner_graph_candidate(H: np.ndarray) -> dict[str, Any]:
    """Evaluate a candidate Tanner graph using spectral metrics.

    Procedure:
    1. Compute spectral metrics via ``compute_spectral_metrics``.
    2. Classify stability regime via ``classify_from_parity_check``.
    3. Compute scalar instability score.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``spectral_gap`` : float
        - ``bethe_margin`` : float
        - ``support_dimension`` : float
        - ``curvature`` : float
        - ``cycle_density`` : float
        - ``sis`` : float
        - ``instability_score`` : float
        - ``predicted_regime`` : str
    """
    H_arr = np.asarray(H, dtype=np.float64)
    metrics = compute_spectral_metrics(H_arr)
    regime_code = classify_tanner_graph_stability(metrics)
    instability = _instability_score(metrics)

    return {
        "spectral_radius": metrics["spectral_radius"],
        "entropy": metrics["entropy"],
        "spectral_gap": metrics["spectral_gap"],
        "bethe_margin": metrics["bethe_margin"],
        "support_dimension": metrics["support_dimension"],
        "curvature": metrics["curvature"],
        "cycle_density": metrics["cycle_density"],
        "sis": metrics["sis"],
        "instability_score": instability,
        "predicted_regime": _REGIME_LABELS.get(regime_code, "unknown"),
    }
