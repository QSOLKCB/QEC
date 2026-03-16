"""Deterministic autonomous scheduler with optional information-gain strategy."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_combined_score(
    exploration_score: float,
    hypothesis_bias: float = 0.0,
    *,
    strategy: str = "default",
    hypothesis_weight: float = 0.5,
) -> float:
    """Compute deterministic combined candidate score."""
    explore = np.float64(exploration_score)
    bias = np.float64(hypothesis_bias)
    weight = np.float64(min(max(float(np.float64(hypothesis_weight)), 0.0), 1.0))
    if strategy == "hypothesis_guided":
        return float(np.float64((np.float64(1.0) - weight) * explore + weight * bias))
    return float(np.float64(explore))


def select_best_candidate(
    candidates: list[dict[str, Any]],
    *,
    strategy: str = "default",
    hypothesis_weight: float = 0.5,
) -> int:
    """Select the best candidate index with deterministic tie-breaking."""
    if not candidates:
        return -1

    scores = np.asarray(
        [
            compute_combined_score(
                item.get("exploration_score", 0.0),
                item.get("hypothesis_bias", 0.0),
                strategy=strategy,
                hypothesis_weight=hypothesis_weight,
            )
            for item in candidates
        ],
        dtype=np.float64,
    )
    indices = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -scores))
    return int(order[0])
