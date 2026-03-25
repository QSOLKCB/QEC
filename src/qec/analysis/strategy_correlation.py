"""v101.9.0 — Pairwise strategy correlation for redundancy detection.

Computes a correlation score between two strategies based on their
metric vectors.  High correlation indicates redundant strategies
that can be pruned.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical computation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


CORRELATION_KEYS: List[str] = [
    "design_score",
    "confidence_efficiency",
    "temporal_stability",
    "trust_modulation",
]


def compute_strategy_correlation(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute pairwise correlation between two strategies.

    Definition::

        v_a = [design_score, confidence_efficiency, temporal_stability, trust_modulation]
        v_b = [same keys]
        corr = clamp(1 - mean(abs(v_a - v_b)), 0.0, 1.0)

    A value of 1.0 means identical metric profiles; 0.0 means
    completely different.

    Parameters
    ----------
    a, b : dict
        Strategy dicts with a ``"metrics"`` sub-dict.

    Returns
    -------
    float
        Correlation in [0.0, 1.0], rounded to 12 decimals.
    """
    a_metrics = a.get("metrics", {})
    b_metrics = b.get("metrics", {})

    n = len(CORRELATION_KEYS)
    total_diff = 0.0

    for key in CORRELATION_KEYS:
        a_val = float(a_metrics.get(key, 0.0))
        b_val = float(b_metrics.get(key, 0.0))
        total_diff += abs(a_val - b_val)

    mean_diff = total_diff / n if n > 0 else 0.0
    corr = 1.0 - mean_diff
    corr = max(0.0, min(1.0, corr))
    corr = round(corr, 12)

    return corr


def prune_redundant(
    strategies: List[Dict[str, Any]],
    threshold: float = 0.98,
) -> List[Dict[str, Any]]:
    """Remove redundant strategies based on pairwise correlation.

    For each pair with correlation > threshold, keeps the strategy with:
    1. Lower consistency_gap (preferred)
    2. Higher design_score (tie-break)

    Comparison order is deterministic: strategies are compared in
    sorted name order; the first survivor wins.

    Does not mutate inputs.

    Parameters
    ----------
    strategies : list of dict
        Candidate strategies.
    threshold : float
        Correlation threshold above which strategies are considered redundant.

    Returns
    -------
    list of dict
        Strategies with redundant entries removed, sorted by name.
    """
    if not strategies:
        return []

    # Sort by name for deterministic processing
    sorted_strats = sorted(strategies, key=lambda s: s.get("name", ""))

    n = len(sorted_strats)
    removed = [False] * n

    for i in range(n):
        if removed[i]:
            continue
        for j in range(i + 1, n):
            if removed[j]:
                continue

            corr = compute_strategy_correlation(sorted_strats[i], sorted_strats[j])
            if corr > threshold:
                # Decide which to remove
                i_metrics = sorted_strats[i].get("metrics", {})
                j_metrics = sorted_strats[j].get("metrics", {})

                i_gap = float(i_metrics.get("consistency_gap", 1.0))
                j_gap = float(j_metrics.get("consistency_gap", 1.0))

                if i_gap < j_gap:
                    removed[j] = True
                elif j_gap < i_gap:
                    removed[i] = True
                    break  # i is removed, stop comparing i
                else:
                    # Tie-break: higher design_score wins
                    i_ds = float(i_metrics.get("design_score", 0.0))
                    j_ds = float(j_metrics.get("design_score", 0.0))
                    if i_ds >= j_ds:
                        removed[j] = True
                    else:
                        removed[i] = True
                        break  # i is removed

    result = [sorted_strats[i] for i in range(n) if not removed[i]]
    result.sort(key=lambda s: s.get("name", ""))

    return result


__all__ = [
    "CORRELATION_KEYS",
    "compute_strategy_correlation",
    "prune_redundant",
]
