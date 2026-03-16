"""Deterministic expected-improvement utilities for Bayesian candidate ranking."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normal_pdf(z: float) -> float:
    z64 = float(np.float64(z))
    return float(math.exp(-0.5 * z64 * z64) / math.sqrt(2.0 * math.pi))


def _normal_cdf(z: float) -> float:
    z64 = float(np.float64(z))
    return float(0.5 * (1.0 + math.erf(z64 / math.sqrt(2.0))))


def expected_improvement(mean: float, sigma: float, best_value: float) -> float:
    """Compute expected improvement with deterministic normal CDF/PDF."""
    mu = float(np.float64(mean))
    std = float(np.float64(sigma))
    best = float(np.float64(best_value))

    if std <= 0.0:
        return 0.0

    z = float((mu - best) / std)
    improvement = (mu - best) * _normal_cdf(z) + std * _normal_pdf(z)
    return float(np.float64(improvement))


def rank_candidates_bayesian(
    candidates: list[dict[str, Any]],
    model: Any,
    best_value: float,
) -> list[dict[str, Any]]:
    """Rank candidates by expected improvement using stable index tie-breaks."""
    if not candidates:
        return []

    enriched: list[dict[str, Any]] = []
    eis = np.zeros((len(candidates),), dtype=np.float64)
    indices = np.arange(len(candidates), dtype=np.int64)

    for idx, candidate in enumerate(candidates):
        objectives = candidate.get("objectives", {})
        spectrum = np.asarray(
            [
                float(objectives.get("spectral_radius", 0.0)),
                float(objectives.get("bethe_margin", 0.0)),
                float(objectives.get("ipr_localization", 0.0)),
                float(objectives.get("entropy", 0.0)),
            ],
            dtype=np.float64,
        )
        mean, sigma = model.predict(spectrum)
        ei = expected_improvement(mean, sigma, best_value)
        updated = dict(candidate)
        updated["bayesian_prediction_mean"] = float(np.float64(mean))
        updated["bayesian_prediction_uncertainty"] = float(np.float64(sigma))
        updated["expected_improvement"] = float(np.float64(ei))
        enriched.append(updated)
        eis[idx] = float(np.float64(ei))

    order = np.lexsort((indices, -eis))
    return [enriched[int(i)] for i in order.tolist()]
