"""Deterministic non-backtracking spectral threshold predictor."""

from __future__ import annotations

from typing import Any


def _metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, 0.0))


def predict_threshold_from_spectrum(metrics: dict[str, Any]) -> dict[str, float]:
    nb_radius = _metric(metrics, "nb_spectral_radius")
    bethe_negative_mass = _metric(metrics, "bethe_negative_mass")
    flow_ipr = _metric(metrics, "flow_ipr")
    spec_entropy = _metric(metrics, "spectral_entropy")
    trap_similarity = _metric(metrics, "trap_similarity")

    score = (
        -0.6 * nb_radius
        + 0.4 * bethe_negative_mass
        - 0.3 * flow_ipr
        + 0.2 * spec_entropy
        - 0.2 * trap_similarity
    )
    threshold_est = max(0.0, min(0.1, 0.03 + score))

    return {
        "predicted_threshold": round(float(threshold_est), 12),
        "prediction_score": round(float(score), 12),
    }
