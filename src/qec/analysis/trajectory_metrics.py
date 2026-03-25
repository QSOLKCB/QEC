"""v102.2.0 — Trajectory metrics for strategy histories.

Computes per-strategy trajectory statistics (mean, variance, stability,
trend, oscillation) from metric histories built by strategy_history.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compute_trajectory_metrics(history: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
    """Compute trajectory metrics per strategy from score histories.

    Parameters
    ----------
    history : dict
        Output of ``build_strategy_history``.  Keyed by strategy name,
        each value maps metric names to lists of float values.

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Each value contains:

        - ``mean_score`` : float — mean of design_score values
        - ``variance_score`` : float — population variance of design_score
        - ``stability`` : float — 1 / (1 + variance_score)
        - ``trend`` : float — final - initial design_score (0.0 if < 2 values)
        - ``oscillation`` : int — sign changes in consecutive score deltas
    """
    result: Dict[str, Dict[str, Any]] = {}

    for name in sorted(history.keys()):
        scores = history[name].get("design_score", [])
        mean_score = _mean(scores)
        variance_score = _variance(scores, mean_score)
        stability = round(1.0 / (1.0 + variance_score), 12)
        trend = round(scores[-1] - scores[0], 12) if len(scores) >= 2 else 0.0
        oscillation = _count_sign_changes(scores)

        result[name] = {
            "mean_score": round(mean_score, 12),
            "variance_score": round(variance_score, 12),
            "stability": stability,
            "trend": trend,
            "oscillation": oscillation,
        }

    return result


def _mean(values: List[float]) -> float:
    """Compute arithmetic mean.  Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: List[float], mean: float) -> float:
    """Compute population variance.  Returns 0.0 for empty list."""
    if not values:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / len(values)


def _count_sign_changes(values: List[float]) -> int:
    """Count the number of sign changes in consecutive deltas."""
    if len(values) < 3:
        return 0
    deltas = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    changes = 0
    for i in range(len(deltas) - 1):
        if deltas[i] * deltas[i + 1] < 0:
            changes += 1
    return changes


__all__ = ["compute_trajectory_metrics"]
