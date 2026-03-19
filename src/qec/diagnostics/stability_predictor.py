"""
v8.2.0 — BP Stability Predictor.

Predicts whether BP will converge on a given Tanner graph using
a pre-fitted linear stability boundary.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics
from qec.diagnostics.stability_boundary import _FEATURE_KEYS


_ROUND = 12


def predict_bp_stability(
    H: np.ndarray,
    boundary: dict[str, Any],
) -> dict[str, Any]:
    """Predict BP convergence from spectral metrics and a fitted boundary.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    boundary : dict[str, Any]
        Output of ``estimate_stability_boundary``, containing
        ``weights`` and ``bias``.

    Returns
    -------
    dict[str, Any]
        Dictionary with ``score``, ``predicted_converged``, and ``metrics``.
    """
    metrics = compute_spectral_metrics(H)

    weights = boundary.get("weights", [0.0] * len(_FEATURE_KEYS))
    bias = boundary.get("bias", 0.0)

    feature_values = [metrics.get(key, 0.0) for key in _FEATURE_KEYS]
    score = float(np.dot(weights, feature_values)) + bias

    return {
        "score": round(score, _ROUND),
        "predicted_converged": bool(score > 0.0),
        "metrics": metrics,
    }
