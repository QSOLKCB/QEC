"""Deterministic performance metrics for benchmarking — v101.0.0.

Pure functions for computing cumulative scores, convergence rates,
stability variance, and final performance from score sequences.

All functions are deterministic, bounded, and never mutate inputs.

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import List, Optional


def compute_cumulative_score(scores: List[float]) -> List[float]:
    """Compute running cumulative average of scores.

    Parameters
    ----------
    scores : list of float
        Per-step score values.

    Returns
    -------
    list of float
        Cumulative average at each step. Empty list if input is empty.
    """
    if not scores:
        return []
    result = []
    running_sum = 0.0
    for i, s in enumerate(scores):
        running_sum += s
        result.append(running_sum / (i + 1))
    return result


def compute_convergence_rate(scores: List[float]) -> float:
    """Compute convergence rate as mean absolute step-to-step change.

    Lower values indicate faster convergence (less change between steps).

    Parameters
    ----------
    scores : list of float
        Per-step score values.

    Returns
    -------
    float
        Mean absolute difference between consecutive scores.
        0.0 if fewer than 2 scores.
    """
    if len(scores) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(scores)):
        total += abs(scores[i] - scores[i - 1])
    return total / (len(scores) - 1)


def compute_stability_variance(scores: List[float]) -> float:
    """Compute variance of score sequence as a stability measure.

    Lower variance indicates more stable performance.

    Parameters
    ----------
    scores : list of float
        Per-step score values.

    Returns
    -------
    float
        Population variance of scores. 0.0 if fewer than 1 score.
    """
    if not scores:
        return 0.0
    n = len(scores)
    mean = sum(scores) / n
    return sum((s - mean) ** 2 for s in scores) / n


def compute_final_performance(
    scores: List[float],
    window: Optional[int] = None,
) -> float:
    """Compute final performance as mean of last *window* scores.

    Parameters
    ----------
    scores : list of float
        Per-step score values.
    window : int or None
        Number of trailing scores to average. Defaults to min(5, len(scores)).

    Returns
    -------
    float
        Mean of the last *window* scores. 0.0 if input is empty.
    """
    if not scores:
        return 0.0
    if window is None:
        window = min(5, len(scores))
    window = max(1, min(window, len(scores)))
    tail = scores[-window:]
    return sum(tail) / len(tail)
