"""Deterministic adaptive operator weighting for mutation selection."""

from __future__ import annotations

import numpy as np


def compute_adaptive_operator_weights(
    operator_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute deterministic normalized operator weights from stabilized stats."""
    if not operator_stats:
        return {}

    names = sorted(operator_stats)
    raw_weights: list[float] = []
    for name in names:
        stats = operator_stats[name]
        attempts = float(np.float64(stats.get("attempts", 0.0)))
        successes = float(np.float64(stats.get("successes", 0.0)))
        success_rate = float(
            np.float64(successes) / np.float64(attempts)
            if attempts > 0.0
            else np.float64(0.0)
        )
        base_weight = float(np.float64(1.0))
        regional_similarity = float(np.float64(0.0))
        weight = float(
            np.float64(base_weight)
            * (np.float64(1.0) + np.float64(success_rate))
            * (np.float64(1.0) + np.float64(regional_similarity))
        )
        raw_weights.append(weight)

    total = float(np.sum(np.asarray(raw_weights, dtype=np.float64), dtype=np.float64))
    if total <= 0.0:
        uniform = float(np.float64(1.0) / np.float64(len(names)))
        return {name: uniform for name in names}

    return {
        name: float(np.float64(weight) / np.float64(total))
        for name, weight in zip(names, raw_weights)
    }


__all__ = ["compute_adaptive_operator_weights"]
