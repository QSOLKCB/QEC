"""Deterministic adaptive operator weighting for mutation selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_adaptive_operator_weights(
    operator_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute deterministic normalized operator weights from stabilized stats."""
    if not operator_stats:
        return {}

    names = [str(name) for name, _ in sorted(operator_stats.items(), key=lambda item: item[0])]
    raw_weights: list[float] = []
    for name in names:
        stats = operator_stats.get(name, {})
        attempts = float(np.float64(stats.get("attempts", 0.0)))
        successes = float(np.float64(stats.get("successes", 0.0)))
        success_rate = float(np.float64(successes) / np.float64(attempts)) if attempts > 0.0 else 0.0
        raw_weights.append(float(np.float64(1.0) + np.float64(success_rate)))

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights_sum = float(np.sum(weights, dtype=np.float64))
    if weights_sum <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64) / np.float64(len(weights))
    else:
        weights = weights / np.float64(weights_sum)

    return {name: float(weights[idx]) for idx, name in enumerate(names)}


def compute_operator_weights(
    operator_stats: dict[str, dict[str, Any]],
    regional_similarity: dict[str, float] | None = None,
) -> dict[str, float]:
    """Backward-compatible deterministic operator weight computation."""
    if not operator_stats:
        return {}

    regional_similarity = regional_similarity or {}
    names = [str(name) for name, _ in sorted(operator_stats.items(), key=lambda item: item[0])]
    raw_weights: list[float] = []
    for name in names:
        stats = operator_stats.get(name, {})
        attempts = float(np.float64(stats.get("attempts", 0.0)))
        successes = float(np.float64(stats.get("successes", 0.0)))
        fallback_sr = float(np.float64(stats.get("success_rate", 0.0)))
        success_rate = fallback_sr
        if attempts > 0.0:
            success_rate = float(np.float64(successes) / np.float64(attempts))

        similarity = float(np.float64(regional_similarity.get(name, 0.0)))
        weight = float((np.float64(1.0) + np.float64(success_rate)) * (np.float64(1.0) + np.float64(similarity)))
        raw_weights.append(weight)

    weights = np.asarray(raw_weights, dtype=np.float64)
    weights_sum = float(np.sum(weights, dtype=np.float64))
    if weights_sum <= 0.0:
        weights = np.ones_like(weights, dtype=np.float64) / np.float64(len(weights))
    else:
        weights = weights / np.float64(weights_sum)

    return {name: float(weights[idx]) for idx, name in enumerate(names)}


def deterministic_weighted_choice(
    options: dict[str, float] | list[Any] | tuple[Any, ...],
    weights: list[float] | np.ndarray | None = None,
    seed: int = 0,
) -> Any:
    """Deterministic weighted choice with seeded NumPy RNG."""
    if isinstance(options, dict):
        ordered = sorted(options.items(), key=lambda item: item[0])
        option_values = [k for k, _ in ordered]
        weight_values = np.asarray([v for _, v in ordered], dtype=np.float64)
    else:
        option_values = list(options)
        if weights is None:
            weight_values = np.ones(len(option_values), dtype=np.float64)
        else:
            weight_values = np.asarray(weights, dtype=np.float64)

    if len(option_values) == 0:
        raise ValueError("deterministic_weighted_choice requires non-empty options")

    rng = np.random.default_rng(int(seed))

    weight_values = np.asarray(weight_values, dtype=np.float64).reshape(-1)
    if weight_values.size != len(option_values):
        raise ValueError("weights must match options length")
    weights_sum = float(np.sum(weight_values, dtype=np.float64))
    if weights_sum <= 0.0:
        p = np.ones_like(weight_values, dtype=np.float64) / np.float64(len(option_values))
    else:
        p = weight_values / np.float64(weights_sum)

    index = int(rng.choice(len(option_values), p=p))
    return option_values[index]


__all__ = [
    "compute_adaptive_operator_weights",
    "compute_operator_weights",
    "deterministic_weighted_choice",
]
