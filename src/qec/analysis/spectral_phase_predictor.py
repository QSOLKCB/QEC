"""Deterministic spectral phase-diagram threshold predictor."""

from __future__ import annotations

import numpy as np

_ROUND = 12


class SpectralPhasePredictor:
    """Linear spectral predictor for BP threshold ranking."""

    def __init__(self) -> None:
        self.coefficients = np.array([0.42, -0.31, 0.27, -0.18], dtype=np.float64)
        self.bias = np.float64(0.12)

    def predict(self, features: np.ndarray) -> float:
        f = np.asarray(features, dtype=np.float64)
        score = float(np.dot(self.coefficients, f) + self.bias)
        return round(score, _ROUND)


def spectral_feature_vector(metrics: dict[str, float]) -> np.ndarray:
    """Build deterministic feature vector from spectral diagnostics."""
    return np.array(
        [
            metrics["nb_spectral_radius"],
            metrics["bethe_min_eigenvalue"],
            metrics["bp_stability_score"],
            metrics["ipr_localization_score"],
        ],
        dtype=np.float64,
    )

