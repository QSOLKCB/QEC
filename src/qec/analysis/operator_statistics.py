"""Deterministic operator success statistics utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def update_operator_success(
    stats: dict[str, dict[str, int]],
    operator: str,
    success: bool,
) -> dict[str, dict[str, int]]:
    """Update deterministic operator success/attempt counters."""
    key = str(operator)
    rec = stats.get(key, {"attempts": 0, "successes": 0})
    attempts = int(rec.get("attempts", 0)) + 1
    successes = int(rec.get("successes", 0)) + (1 if bool(success) else 0)
    stats[key] = {"attempts": attempts, "successes": successes}
    return stats


def summarize_operator_statistics(
    stats: dict[str, dict[str, int]],
    operator: str,
) -> float:
    """Return deterministic success-rate summary for one operator."""
    rec = stats.get(str(operator), {"attempts": 0, "successes": 0})
    attempts = int(rec.get("attempts", 0))
    if attempts <= 0:
        return 0.0
    rate = np.float64(rec.get("successes", 0)) / np.float64(attempts)
    return float(min(max(float(rate), 0.0), 1.0))


__all__ = ["update_operator_success", "summarize_operator_statistics"]
