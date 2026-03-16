"""Deterministic operator statistics for adaptive mutation learning."""

from __future__ import annotations

import numpy as np


def update_operator_success(
    operator_stats: dict[str, dict[str, float]],
    operator_name: str,
    improvement: float | None,
) -> None:
    """Update per-operator attempts, successes, and improvement aggregates."""
    if improvement is None:
        raise ValueError("Operator success update requires computed improvement.")

    stats = operator_stats.setdefault(
        str(operator_name),
        {
            "attempts": float(np.float64(0.0)),
            "successes": float(np.float64(0.0)),
            "total_improvement": float(np.float64(0.0)),
            "mean_improvement": float(np.float64(0.0)),
        },
    )

    improvement64 = float(np.float64(improvement))
    effective_improvement = float(np.float64(max(improvement64, 0.0)))

    attempts = float(np.float64(stats["attempts"]) + np.float64(1.0))
    successes = float(
        np.float64(stats["successes"])
        + np.float64(1.0 if improvement64 > 0.0 else 0.0),
    )
    total_improvement = float(
        np.float64(stats["total_improvement"]) + np.float64(effective_improvement),
    )
    mean_improvement = float(
        np.float64(total_improvement) / np.float64(attempts)
        if attempts > 0.0
        else np.float64(0.0)
    )

    stats["attempts"] = attempts
    stats["successes"] = successes
    stats["total_improvement"] = total_improvement
    stats["mean_improvement"] = mean_improvement


def summarize_operator_statistics(
    operator_stats: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Return a deterministic copy of operator statistics."""
    summary: dict[str, dict[str, float]] = {}
    for name in sorted(operator_stats):
        stats = operator_stats[name]
        attempts = float(np.float64(stats.get("attempts", 0.0)))
        successes = float(np.float64(stats.get("successes", 0.0)))
        total_improvement = float(np.float64(stats.get("total_improvement", 0.0)))
        mean_improvement = float(
            np.float64(total_improvement) / np.float64(attempts)
            if attempts > 0.0
            else np.float64(0.0)
        )
        summary[name] = {
            "attempts": attempts,
            "successes": successes,
            "total_improvement": total_improvement,
            "mean_improvement": mean_improvement,
        }
    return summary


__all__ = ["update_operator_success", "summarize_operator_statistics"]

