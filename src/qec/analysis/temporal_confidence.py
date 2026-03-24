"""v101.2.0 — Temporal confidence tracking and trust signal derivation.

Tracks confidence over time, measures stability/volatility, derives
a bounded trust signal, and optionally modulates scoring using trust.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import List


# ---------------------------------------------------------------------------
# Task 1 — Temporal Confidence Tracker
# ---------------------------------------------------------------------------


def update_confidence_history(
    history: List[float],
    new_confidence: float,
    max_len: int = 10,
) -> List[float]:
    """Append confidence, enforce bounded FIFO history.

    Returns a new list (input not mutated) containing the most recent
    *max_len* entries after appending *new_confidence*.

    Parameters
    ----------
    history : list of float
        Existing confidence history (oldest first).
    new_confidence : float
        New confidence value to append.
    max_len : int
        Maximum history length (default 10).

    Returns
    -------
    list of float
        Updated history with at most *max_len* entries.
    """
    new_history = list(history) + [float(new_confidence)]
    if len(new_history) > max_len:
        new_history = new_history[-max_len:]
    return new_history


# ---------------------------------------------------------------------------
# Task 2 — Volatility & Stability
# ---------------------------------------------------------------------------


def compute_confidence_variance(history: List[float]) -> float:
    """Return variance (>= 0), safe for small lists.

    Returns 0.0 for empty or single-element histories.

    Parameters
    ----------
    history : list of float
        Confidence history values.

    Returns
    -------
    float
        Population variance, >= 0.
    """
    if len(history) < 2:
        return 0.0

    values = [float(v) for v in history]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return max(0.0, variance)


def compute_confidence_stability(history: List[float]) -> float:
    """Return stability in [0, 1] = 1 / (1 + variance).

    Returns 1.0 for empty or single-element histories (maximally stable).

    Parameters
    ----------
    history : list of float
        Confidence history values.

    Returns
    -------
    float
        Stability in [0.0, 1.0].
    """
    if len(history) < 2:
        return 1.0

    variance = compute_confidence_variance(history)
    stability = 1.0 / (1.0 + variance)
    return max(0.0, min(1.0, stability))


# ---------------------------------------------------------------------------
# Task 3 — Trend Signal
# ---------------------------------------------------------------------------


def compute_confidence_trend(history: List[float]) -> float:
    """Return trend in [-1, 1] (increasing vs decreasing).

    Computed as ``last - first``, clamped to [-1, 1].
    Returns 0.0 for empty or single-element histories.

    Parameters
    ----------
    history : list of float
        Confidence history values.

    Returns
    -------
    float
        Trend in [-1.0, 1.0].
    """
    if len(history) < 2:
        return 0.0

    trend = float(history[-1]) - float(history[0])
    return max(-1.0, min(1.0, trend))


# ---------------------------------------------------------------------------
# Task 4 — Trust Signal
# ---------------------------------------------------------------------------


def compute_trust_signal(
    stability: float,
    trend: float,
) -> float:
    """Combine stability and trend into a trust signal in [0, 1].

    Formula::

        trend_factor = 0.5 + 0.5 * trend   # maps [-1,1] -> [0,1]
        trust = stability * trend_factor

    Interpretation:
    - high stability + positive trend -> high trust
    - unstable or declining -> low trust

    Parameters
    ----------
    stability : float
        Confidence stability in [0, 1].
    trend : float
        Confidence trend in [-1, 1].

    Returns
    -------
    float
        Trust signal in [0.0, 1.0].
    """
    s = max(0.0, min(1.0, float(stability)))
    t = max(-1.0, min(1.0, float(trend)))

    trend_factor = 0.5 + 0.5 * t
    trust = s * trend_factor
    return max(0.0, min(1.0, trust))


# ---------------------------------------------------------------------------
# Task 5 — Trust Modulation
# ---------------------------------------------------------------------------


def compute_trust_modulation(trust: float) -> float:
    """Compute optional trust modulation factor.

    Formula::

        trust_modulation = 0.9 + 0.2 * trust

    Range: [0.9, 1.1].  Neutral (1.0) when trust = 0.5.

    Parameters
    ----------
    trust : float
        Trust signal in [0, 1].

    Returns
    -------
    float
        Trust modulation factor in [0.9, 1.1].
    """
    t = max(0.0, min(1.0, float(trust)))
    modulation = 0.9 + 0.2 * t
    return max(0.9, min(1.1, modulation))


__all__ = [
    "update_confidence_history",
    "compute_confidence_variance",
    "compute_confidence_stability",
    "compute_confidence_trend",
    "compute_trust_signal",
    "compute_trust_modulation",
]
