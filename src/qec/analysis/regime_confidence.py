"""v101.3.0 — Regime-aware confidence tracking and local trust signals.

Tracks confidence history per regime_key, computes local stability/trend/trust,
blends global and local trust, and derives regime-specific trust modulation.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from qec.analysis.temporal_confidence import (
    compute_confidence_stability,
    compute_confidence_trend,
    compute_trust_signal,
)


# ---------------------------------------------------------------------------
# Task 1 — Regime Confidence Memory
# ---------------------------------------------------------------------------


def update_regime_confidence_history(
    memory: dict,
    regime_key: tuple,
    confidence: float,
    max_len: int = 10,
) -> dict:
    """Append confidence to regime-specific FIFO history.

    Returns a new dict (input not mutated) with the updated history
    for *regime_key*.  Other regime entries are preserved unchanged.

    Parameters
    ----------
    memory : dict
        Mapping of regime_key -> list of float (confidence history).
    regime_key : tuple
        Regime identifier (e.g. ('stable', 'basin_2')).
    confidence : float
        New confidence value to append.
    max_len : int
        Maximum history length per regime (default 10).

    Returns
    -------
    dict
        New memory dict with updated history for *regime_key*.
    """
    new_memory = {k: list(v) for k, v in memory.items()}
    history = list(new_memory.get(regime_key, []))
    history.append(float(confidence))
    if len(history) > max_len:
        history = history[-max_len:]
    new_memory[regime_key] = history
    return new_memory


# ---------------------------------------------------------------------------
# Task 2 — Local Trust Computation
# ---------------------------------------------------------------------------


def compute_regime_trust(history: List[float]) -> Dict[str, float]:
    """Compute local trust signals from a regime's confidence history.

    Reuses v101.2 functions for stability, trend, and trust derivation.

    Parameters
    ----------
    history : list of float
        Confidence history for a specific regime (oldest first).

    Returns
    -------
    dict[str, float]
        Keys: ``stability``, ``trend``, ``trust``.
    """
    stability = compute_confidence_stability(history)
    trend = compute_confidence_trend(history)
    trust = compute_trust_signal(stability, trend)
    return {
        "stability": stability,
        "trend": trend,
        "trust": trust,
    }


# ---------------------------------------------------------------------------
# Task 3 — Global + Local Blending
# ---------------------------------------------------------------------------


def blend_trust_signals(
    global_trust: float,
    local_trust: float,
    alpha: float = 0.5,
) -> float:
    """Blend global and local trust signals.

    Formula::

        blended_trust = alpha * local_trust + (1 - alpha) * global_trust

    Result clamped to [0, 1].

    Parameters
    ----------
    global_trust : float
        Global trust signal in [0, 1].
    local_trust : float
        Regime-local trust signal in [0, 1].
    alpha : float
        Blending weight for local trust (default 0.5).

    Returns
    -------
    float
        Blended trust in [0.0, 1.0].
    """
    g = max(0.0, min(1.0, float(global_trust)))
    l = max(0.0, min(1.0, float(local_trust)))
    a = max(0.0, min(1.0, float(alpha)))
    blended = a * l + (1.0 - a) * g
    return max(0.0, min(1.0, blended))


# ---------------------------------------------------------------------------
# Task 4 — Regime Trust Modulation
# ---------------------------------------------------------------------------


def compute_regime_trust_modulation(trust: float) -> float:
    """Compute regime-specific trust modulation factor.

    Formula::

        regime_trust_modulation = 0.9 + 0.2 * trust

    Range: [0.9, 1.1].  Neutral (1.0) when trust = 0.5.

    Parameters
    ----------
    trust : float
        Trust signal in [0, 1].

    Returns
    -------
    float
        Regime trust modulation factor in [0.9, 1.1].
    """
    t = max(0.0, min(1.0, float(trust)))
    modulation = 0.9 + 0.2 * t
    return max(0.9, min(1.1, modulation))


__all__ = [
    "update_regime_confidence_history",
    "compute_regime_trust",
    "blend_trust_signals",
    "compute_regime_trust_modulation",
]
