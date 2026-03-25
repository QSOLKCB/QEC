"""v101.9.0 — Temporal revival detection for structure-aware dominance.

Detects drop-then-recovery patterns in score time series.
A revival indicates a strategy that recovered from a temporary dip,
which may signal resilience or instability depending on context.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical computation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def detect_revival(history: List[float]) -> Dict[str, Any]:
    """Detect a drop-then-recovery pattern in a score time series.

    A revival occurs when the series drops to a minimum then recovers
    to a higher final value.

    Revival strength is defined as::

        strength = clamp((final - min_value) / (max_value - min_value), 0.0, 1.0)

    where max_value is the maximum observed in the series.  If max equals
    min (flat series), strength is 0.0.

    Parameters
    ----------
    history : list of float
        Time series of score values.  Must have at least 2 elements
        for revival detection.

    Returns
    -------
    dict
        ``has_revival`` : bool
        ``revival_strength`` : float in [0, 1]
        ``min_value`` : float
        ``recovered_to`` : float (final value)
    """
    if len(history) < 2:
        return {
            "has_revival": False,
            "revival_strength": 0.0,
            "min_value": float(history[0]) if history else 0.0,
            "recovered_to": float(history[0]) if history else 0.0,
        }

    values = [float(v) for v in history]
    min_value = min(values)
    max_value = max(values)
    final_value = values[-1]

    # Find the index of the first occurrence of min_value
    min_idx = values.index(min_value)

    # Revival requires: min occurs before the end AND final > min
    has_revival = min_idx < len(values) - 1 and final_value > min_value

    if has_revival and max_value > min_value:
        strength = (final_value - min_value) / (max_value - min_value)
        strength = max(0.0, min(1.0, strength))
    else:
        strength = 0.0

    strength = round(strength, 12)

    return {
        "has_revival": has_revival,
        "revival_strength": strength,
        "min_value": round(min_value, 12),
        "recovered_to": round(final_value, 12),
    }


def enrich_with_revival(
    strategy: Dict[str, Any],
    history: List[float],
) -> Dict[str, Any]:
    """Return a copy of *strategy* with revival info added to metrics.

    Does not mutate the input.

    Parameters
    ----------
    strategy : dict
        Strategy dict with a ``"metrics"`` sub-dict.
    history : list of float
        Score history for this strategy.

    Returns
    -------
    dict
        New strategy dict with revival fields in ``metrics``.
    """
    result = dict(strategy)
    result["metrics"] = dict(result.get("metrics", {}))
    revival = detect_revival(history)
    result["metrics"]["has_revival"] = revival["has_revival"]
    result["metrics"]["revival_strength"] = revival["revival_strength"]
    return result


__all__ = [
    "detect_revival",
    "enrich_with_revival",
]
