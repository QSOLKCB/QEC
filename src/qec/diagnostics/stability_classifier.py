"""
v8.1.0 — Ternary Tanner Graph Stability Classifier.

Classifies Tanner graph stability into three regimes based on
spectral metrics:

    +1 : stable      — BP expected to converge reliably
     0 : metastable  — BP may converge slowly or intermittently
    -1 : unstable    — BP expected to fail or oscillate

Classification uses deterministic threshold rules on the spectral
instability score (SIS), spectral gap, and Bethe Hessian margin.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_metrics import compute_spectral_metrics


def classify_tanner_graph_stability(metrics: dict[str, Any]) -> int:
    """Classify Tanner graph stability from precomputed spectral metrics.

    Classification rules (evaluated in order):

    1. If SIS > 0.1 AND bethe_margin < -1.0 → unstable (-1)
    2. If SIS > 0.05 OR bethe_margin < 0.0 → metastable (0)
    3. Otherwise → stable (+1)

    Parameters
    ----------
    metrics : dict[str, Any]
        Dictionary from ``compute_spectral_metrics``.

    Returns
    -------
    int
        +1 (stable), 0 (metastable), or -1 (unstable).
    """
    sis = float(metrics["sis"])
    bethe_margin = float(metrics["bethe_margin"])

    # Rule 1: unstable
    if sis > 0.1 and bethe_margin < -1.0:
        return -1

    # Rule 2: metastable
    if sis > 0.05 or bethe_margin < 0.0:
        return 0

    # Rule 3: stable
    return 1


def classify_from_parity_check(H: np.ndarray) -> int:
    """Classify Tanner graph stability directly from a parity-check matrix.

    Convenience function that computes spectral metrics and then
    classifies stability.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    int
        +1 (stable), 0 (metastable), or -1 (unstable).
    """
    metrics = compute_spectral_metrics(H)
    return classify_tanner_graph_stability(metrics)
