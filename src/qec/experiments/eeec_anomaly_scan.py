"""
v7.6.1 — EEEC Anomaly Detection Experiment.

Detects localized trapping-set regimes where spectral radius appears
normal but eigenvector localization is high.

An EEEC anomaly is detected when all three conditions hold:

  spectral_radius < radius_threshold (1.5)
  eeec > eeec_threshold (0.25)
  ipr > ipr_threshold (0.05)

This ensures anomalies are flagged only when both eigenvector energy
concentration (EEEC) and eigenvector localization (IPR) are high,
avoiding dense-graph false positives.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import compute_nb_spectrum


_ROUND = 12

# ── Default thresholds ────────────────────────────────────────────

DEFAULT_RADIUS_THRESHOLD = 1.5
DEFAULT_EEEC_THRESHOLD = 0.25
DEFAULT_IPR_THRESHOLD = 0.05


# ── Core: anomaly detection ──────────────────────────────────────


def detect_eeec_anomaly(
    H: np.ndarray,
    *,
    radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
    eeec_threshold: float = DEFAULT_EEEC_THRESHOLD,
    ipr_threshold: float = DEFAULT_IPR_THRESHOLD,
) -> dict[str, Any]:
    """Detect EEEC anomaly for a parity-check matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    radius_threshold : float
        Upper bound on spectral radius for anomaly detection.
    eeec_threshold : float
        Lower bound on EEEC for anomaly detection.
    ipr_threshold : float
        Lower bound on IPR for anomaly detection.

    Returns
    -------
    dict[str, Any]
        JSON-serializable result with keys:

        - ``eeec_anomaly_detected`` : bool
        - ``spectral_radius`` : float
        - ``eeec`` : float
        - ``ipr`` : float
        - ``radius_threshold`` : float
        - ``eeec_threshold`` : float
        - ``ipr_threshold`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)

    spectrum = compute_nb_spectrum(H_arr)

    spectral_radius = spectrum["spectral_radius"]
    eeec = spectrum["eeec"]
    ipr = spectrum["ipr"]

    anomaly = (
        spectral_radius < radius_threshold
        and eeec > eeec_threshold
        and ipr > ipr_threshold
    )

    return {
        "eeec_anomaly_detected": bool(anomaly),
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "eeec": round(float(eeec), _ROUND),
        "ipr": round(float(ipr), _ROUND),
        "radius_threshold": radius_threshold,
        "eeec_threshold": eeec_threshold,
        "ipr_threshold": ipr_threshold,
    }


# ── Batch scan ───────────────────────────────────────────────────


def run_eeec_anomaly_scan(
    matrices: list[np.ndarray],
    *,
    radius_threshold: float = DEFAULT_RADIUS_THRESHOLD,
    eeec_threshold: float = DEFAULT_EEEC_THRESHOLD,
    ipr_threshold: float = DEFAULT_IPR_THRESHOLD,
) -> dict[str, Any]:
    """Scan multiple parity-check matrices for EEEC anomalies.

    Parameters
    ----------
    matrices : list[np.ndarray]
        List of binary parity-check matrices.
    radius_threshold : float
        Upper bound on spectral radius.
    eeec_threshold : float
        Lower bound on EEEC.
    ipr_threshold : float
        Lower bound on IPR.

    Returns
    -------
    dict[str, Any]
        JSON-serializable scan result with keys:

        - ``num_matrices`` : int
        - ``num_anomalies`` : int
        - ``anomaly_indices`` : list[int]
        - ``results`` : list[dict]
    """
    results = []
    anomaly_indices = []

    for i, H in enumerate(matrices):
        result = detect_eeec_anomaly(
            H,
            radius_threshold=radius_threshold,
            eeec_threshold=eeec_threshold,
            ipr_threshold=ipr_threshold,
        )
        results.append(result)
        if result["eeec_anomaly_detected"]:
            anomaly_indices.append(i)

    return {
        "num_matrices": len(matrices),
        "num_anomalies": len(anomaly_indices),
        "anomaly_indices": anomaly_indices,
        "results": results,
    }
