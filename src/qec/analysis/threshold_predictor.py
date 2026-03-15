"""Deterministic lightweight predictor for spectral threshold quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_ROUND = 12


@dataclass(frozen=True)
class ThresholdPrediction:
    predicted_threshold: float
    score: float


def predict_threshold_quality(
    spectral_radius: float,
    bethe_negative_mass: float,
    flow_ipr: float,
    spectral_entropy_value: float,
    trap_similarity: float,
) -> ThresholdPrediction:
    """Lightweight deterministic predictor for threshold quality."""
    score = (
        -0.5 * float(np.float64(spectral_radius))
        + 0.4 * float(np.float64(bethe_negative_mass))
        - 0.3 * float(np.float64(flow_ipr))
        + 0.2 * float(np.float64(spectral_entropy_value))
        - 0.3 * float(np.float64(trap_similarity))
    )
    score = round(float(np.float64(score)), _ROUND)

    predicted = max(0.0, min(1.0, 0.02 + score))
    predicted = round(float(np.float64(predicted)), _ROUND)

    return ThresholdPrediction(predicted_threshold=predicted, score=score)
