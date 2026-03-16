"""Deterministic adaptive mutation operator weighting."""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_OPERATORS = [
    "edge_swap",
    "local_rewire",
    "cycle_break",
    "degree_preserving_rotation",
    "seeded_reconstruction",
    "cycle_guided_mutation",
    "spectral_pressure_guided_mutation",
]


def compute_operator_weights(
    operator_stats: dict[str, dict[str, Any]],
    regional_similarity: dict[str, float] | np.ndarray | list[float] | None,
    *,
    base_weight: float = 1.0,
) -> dict[str, float]:
    """Compute normalized deterministic operator weights."""
    similarity_map: dict[str, np.float64] = {}
    if isinstance(regional_similarity, dict):
        similarity_map = {str(k): np.float64(float(v)) for k, v in regional_similarity.items()}
    elif regional_similarity is None:
        similarity_map = {}
    else:
        arr = np.asarray(regional_similarity, dtype=np.float64).reshape(-1)
        for i, op in enumerate(_DEFAULT_OPERATORS):
            similarity_map[op] = np.float64(arr[i] if i < arr.size else 0.0)

    weights = np.zeros(len(_DEFAULT_OPERATORS), dtype=np.float64)
    for i, op in enumerate(_DEFAULT_OPERATORS):
        rec = operator_stats.get(op, {}) if operator_stats else {}
        success_rate = np.float64(float(rec.get("success_rate", rec.get("operator_success_rate", 0.0))))
        sim = np.float64(float(similarity_map.get(op, 0.0)))
        weights[i] = np.float64(base_weight) * (np.float64(1.0) + success_rate) * (np.float64(1.0) + sim)

    total = np.float64(np.sum(weights, dtype=np.float64))
    if total <= 0.0:
        weights = np.full(weights.shape, 1.0 / float(len(weights)), dtype=np.float64)
    else:
        weights = weights / total

    return {op: float(weights[i]) for i, op in enumerate(_DEFAULT_OPERATORS)}


def deterministic_weighted_choice(weights: dict[str, float]) -> str:
    """Choose deterministically by max weight with stable name tie-break."""
    if not weights:
        return _DEFAULT_OPERATORS[0]
    ordered = sorted(
        ((str(op), float(w)) for op, w in weights.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ordered[0][0]
