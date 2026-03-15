"""Deterministic spectral surrogate for BP phase-diagram curves."""

from __future__ import annotations

from typing import Any

import numpy as np

_ROUND = 12

PHASE_GRID = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08], dtype=np.float64)


class SpectralPhaseDiagramSurrogate:
    """Lightweight deterministic linear surrogate for BP success curves."""

    def __init__(self) -> None:
        self.weights = np.array(
            [
                [0.42, -0.31, 0.18, 0.11],
                [0.39, -0.28, 0.19, 0.12],
                [0.36, -0.25, 0.21, 0.13],
                [0.32, -0.22, 0.23, 0.14],
                [0.28, -0.18, 0.25, 0.16],
                [0.23, -0.15, 0.27, 0.18],
                [0.18, -0.12, 0.29, 0.20],
                [0.12, -0.09, 0.31, 0.23],
            ],
            dtype=np.float64,
        )
        self.bias = np.array([0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20, 0.22], dtype=np.float64)

    def predict(self, features: np.ndarray | list[float]) -> np.ndarray:
        f = np.asarray(features, dtype=np.float64)
        result = self.weights @ f + self.bias
        return np.round(result, _ROUND)


def _metric(metrics: dict[str, Any], key: str) -> float:
    return float(np.float64(metrics.get(key, 0.0)))


def spectral_feature_vector(metrics: dict[str, Any]) -> np.ndarray:
    """Build deterministic spectral feature vector used by the surrogate."""
    return np.array(
        [
            _metric(metrics, "nb_spectral_radius"),
            _metric(metrics, "bethe_hessian_min_eigenvalue"),
            _metric(metrics, "bp_stability_score"),
            _metric(metrics, "ipr_localization"),
        ],
        dtype=np.float64,
    )


def surrogate_threshold(grid: np.ndarray, curve: np.ndarray) -> float:
    """Extract first phase-grid value where success probability drops below 0.5."""
    g = np.asarray(grid, dtype=np.float64)
    c = np.asarray(curve, dtype=np.float64)
    for i in range(len(c)):
        if float(c[i]) < 0.5:
            return round(float(np.float64(g[i])), _ROUND)
    return round(float(np.float64(g[-1])), _ROUND)
