"""Deterministic mutation operator statistics tracking."""

from __future__ import annotations

from typing import Any

import numpy as np


def update_operator_success(
    stats: dict[str, dict[str, np.float64]] | None,
    operator_name: str,
    improvement: float,
) -> dict[str, Any]:
    """Update attempt/success/improvement aggregates for one operator."""
    updated: dict[str, dict[str, np.float64]] = {} if stats is None else dict(stats)
    key = str(operator_name)
    rec = updated.get(
        key,
        {
            "attempts": np.float64(0.0),
            "successes": np.float64(0.0),
            "total_improvement": np.float64(0.0),
        },
    )
    rec = {
        "attempts": np.float64(rec.get("attempts", 0.0)) + np.float64(1.0),
        "successes": np.float64(rec.get("successes", 0.0)),
        "total_improvement": np.float64(rec.get("total_improvement", 0.0)),
    }
    gain = np.float64(improvement)
    if gain > 0.0:
        rec["successes"] = rec["successes"] + np.float64(1.0)
        rec["total_improvement"] = rec["total_improvement"] + gain

    attempts = rec["attempts"]
    rec["success_rate"] = np.float64(rec["successes"] / attempts) if attempts > 0 else np.float64(0.0)
    rec["mean_improvement"] = np.float64(rec["total_improvement"] / attempts) if attempts > 0 else np.float64(0.0)

    # Backward-compatible aliases.
    rec["operator_attempts"] = np.float64(rec["attempts"])
    rec["operator_successes"] = np.float64(rec["successes"])
    rec["operator_improvement_sum"] = np.float64(rec["total_improvement"])
    rec["operator_success_rate"] = np.float64(rec["success_rate"])
    rec["operator_mean_improvement"] = np.float64(rec["mean_improvement"])

    updated[key] = rec
    return updated
