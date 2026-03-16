"""Deterministic adaptive operator-weight utilities."""

from __future__ import annotations

import numpy as np


def compute_adaptive_operator_weights(
    base_weights: np.ndarray,
    success_rates: np.ndarray,
    regional_similarity: float,
    cluster_similarity: float = 0.0,
    *,
    enable_stability_guard: bool = True,
) -> np.ndarray:
    """Compute normalized deterministic adaptive weights."""
    base = np.asarray(base_weights, dtype=np.float64)
    success = np.asarray(success_rates, dtype=np.float64)
    if base.shape != success.shape:
        raise ValueError("base_weights and success_rates must match shape")

    weights = (
        base
        * (1.0 + success)
        * (1.0 + float(np.float64(regional_similarity)))
        * (1.0 + float(np.float64(cluster_similarity)))
    ).astype(np.float64, copy=False)

    if enable_stability_guard:
        weights = weights + np.float64(1e-6)

    total = np.sum(weights, dtype=np.float64)
    if float(total) <= 0.0:
        return np.full(base.shape[0], 1.0 / float(max(1, base.shape[0])), dtype=np.float64)
    return (weights / total).astype(np.float64, copy=False)


def deterministic_weighted_choice(
    items: list[str],
    weights: np.ndarray,
    *,
    seed: int,
) -> str:
    """Choose one item deterministically from weighted probabilities."""
    w = np.asarray(weights, dtype=np.float64)
    if len(items) == 0:
        raise ValueError("items cannot be empty")
    if w.shape[0] != len(items):
        raise ValueError("weights length must equal items length")

    order = np.argsort(np.arange(len(items)), kind="stable")
    probs = w[order]
    probs = probs / np.sum(probs, dtype=np.float64)
    cdf = np.cumsum(probs, dtype=np.float64)
    rng = np.random.RandomState(int(seed))
    u = np.float64(rng.rand())
    idx = int(np.searchsorted(cdf, u, side="left"))
    idx = min(max(idx, 0), len(items) - 1)
    return items[int(order[idx])]
