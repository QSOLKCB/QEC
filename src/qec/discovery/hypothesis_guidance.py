"""Deterministic hypothesis-guided spectral bias computation."""

from __future__ import annotations

from typing import Any

import numpy as np


_FEATURE_INDEX = {
    "spectral_radius": 0,
    "bethe_margin": 1,
    "bp_stability": 1,
    "motif_frequency": 3,
    "cycle_density": 0,
    "mean_degree": 3,
}


def _feature_alignment(spectrum: np.ndarray, feature_name: str) -> float:
    index = _FEATURE_INDEX.get(feature_name)
    if index is None or spectrum.size == 0:
        return 0.0
    value = float(np.float64(spectrum[index]))
    if feature_name in ("spectral_radius", "cycle_density"):
        value = -value
    return float(np.float64(np.tanh(value)))


def compute_hypothesis_bias(spectrum: np.ndarray, hypotheses: list[dict[str, Any]]) -> float:
    """Compute float64 hypothesis bias value for a candidate spectrum."""
    spec = np.asarray(spectrum, dtype=np.float64)
    if len(hypotheses) == 0:
        return 0.0

    weighted = np.zeros((len(hypotheses),), dtype=np.float64)
    weights = np.zeros((len(hypotheses),), dtype=np.float64)

    for i, hypothesis in enumerate(hypotheses):
        feature_name = str(hypothesis.get("feature_name", ""))
        corr = float(np.float64(hypothesis.get("correlation_strength", 0.0)))
        conf = float(np.float64(hypothesis.get("confidence_score", 0.0)))
        alignment = _feature_alignment(spec, feature_name)
        weighted[i] = np.float64(np.sign(corr) * alignment * conf)
        weights[i] = np.float64(conf)

    denom = float(np.sum(weights, dtype=np.float64))
    if denom <= 0.0:
        return 0.0
    return float(np.float64(np.sum(weighted, dtype=np.float64) / denom))



def compute_weighted_scheduler_score(
    exploration_score: float,
    hypothesis_bias: float,
    hypothesis_weight: float,
) -> float:
    """Compute weighted deterministic hypothesis-guided scheduler score."""
    weight = float(np.float64(min(max(float(np.float64(hypothesis_weight)), 0.0), 1.0)))
    explore = float(np.float64(exploration_score))
    bias = float(np.float64(hypothesis_bias))
    return float(np.float64((np.float64(1.0) - np.float64(weight)) * explore + np.float64(weight) * bias))


def select_hypothesis_guided_candidate(
    exploration_scores: np.ndarray,
    hypothesis_biases: np.ndarray,
    hypothesis_weight: float,
) -> int:
    """Select deterministic best candidate with np.lexsort tie-breaking."""
    explore = np.asarray(exploration_scores, dtype=np.float64)
    biases = np.asarray(hypothesis_biases, dtype=np.float64)
    if explore.size == 0:
        return -1
    combined = np.asarray(
        [
            compute_weighted_scheduler_score(
                float(explore[i]),
                float(biases[i]) if i < biases.size else 0.0,
                hypothesis_weight,
            )
            for i in range(explore.size)
        ],
        dtype=np.float64,
    )
    indices = np.arange(combined.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -combined))
    return int(order[0])
