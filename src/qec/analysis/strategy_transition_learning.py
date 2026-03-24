"""Deterministic transition learning layer (v99.3.0).

Records and learns from regime transitions to bias future strategy selection.
When a strategy is applied in one regime and the system transitions to another,
the outcome is recorded as a transition event.  Accumulated transition
statistics inform future strategy scoring via a multiplicative bias.

Transition key:
    (regime_before, attractor_before, strategy_id, regime_after, attractor_after)

Transition memory entry:
    {"count": int, "mean_delta": float, "success_rate": float}

All operations are deterministic, pure-functional (no mutation), and bounded.
Dependencies: stdlib only.  No randomness, no ML, no mutation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Transition key type: 5-tuple
TransitionKey = Tuple[str, str, str, str, str]


# ---------------------------------------------------------------------------
# 1. Transition key computation
# ---------------------------------------------------------------------------

def compute_transition_key(
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
    regime_after: str,
    attractor_after: str,
) -> TransitionKey:
    """Compute a deterministic, hashable transition key.

    Parameters
    ----------
    regime_before : str
        Regime label before strategy application.
    attractor_before : str
        Attractor identifier before strategy application.
    strategy_id : str
        Strategy that was applied.
    regime_after : str
        Regime label after strategy application.
    attractor_after : str
        Attractor identifier after strategy application.

    Returns
    -------
    tuple of (str, str, str, str, str)
        Deterministic, hashable transition key.
    """
    return (
        str(regime_before),
        str(attractor_before),
        str(strategy_id),
        str(regime_after),
        str(attractor_after),
    )


# ---------------------------------------------------------------------------
# 2. Transition memory update
# ---------------------------------------------------------------------------

def update_transition_memory(
    memory: Dict[TransitionKey, Dict[str, Any]],
    key: TransitionKey,
    improvement_score: float,
) -> Dict[TransitionKey, Dict[str, Any]]:
    """Record a transition outcome using deterministic incremental update.

    Returns a **new** memory dict — the input is never mutated.

    The success rule is: ``improvement_score > 0``.

    Incremental update formulas::

        new_count = old_count + 1
        new_mean_delta = (old_mean_delta * old_count + improvement_score) / new_count
        old_successes = old_success_rate * old_count
        new_success_rate = (old_successes + is_success) / new_count

    Parameters
    ----------
    memory : dict
        Maps transition key -> ``{"count": int, "mean_delta": float,
        "success_rate": float}``.
    key : tuple
        Transition key from :func:`compute_transition_key`.
    improvement_score : float
        Improvement score from evaluation (positive = success).

    Returns
    -------
    dict
        New memory with updated statistics.
    """
    score = float(improvement_score)
    is_success = 1.0 if score > 0.0 else 0.0

    # Shallow copy top-level dict
    new_memory = dict(memory)

    if key in new_memory:
        old = new_memory[key]
        old_count = int(old["count"])
        new_count = old_count + 1

        # Deterministic incremental mean
        new_mean_delta = (
            float(old["mean_delta"]) * old_count + score
        ) / new_count

        # Deterministic incremental success rate
        old_successes = float(old["success_rate"]) * old_count
        new_success_rate = (old_successes + is_success) / new_count

        new_memory[key] = {
            "count": new_count,
            "mean_delta": new_mean_delta,
            "success_rate": new_success_rate,
        }
    else:
        new_memory[key] = {
            "count": 1,
            "mean_delta": score,
            "success_rate": is_success,
        }

    return new_memory


# ---------------------------------------------------------------------------
# 3. Transition bias computation
# ---------------------------------------------------------------------------

# Bias formula: transition_bias = 0.8 + 0.4 * aggregate_success_rate
# Range: [0.8, 1.2]
#   success_rate = 0.0  -> bias = 0.8  (penalize failing transitions)
#   success_rate = 0.5  -> bias = 1.0  (neutral)
#   success_rate = 1.0  -> bias = 1.2  (reward successful transitions)
_BIAS_BASE: float = 0.8
_BIAS_SCALE: float = 0.4


def compute_transition_bias(
    memory: Dict[TransitionKey, Dict[str, Any]],
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
) -> float:
    """Compute transition bias for a strategy in the current regime context.

    Aggregates all transition entries matching
    ``(regime_before, attractor_before, strategy_id, *, *)``
    using count-weighted success rates.

    Fallback: if no transition history exists, returns 1.0 (neutral).

    Formula::

        transition_bias = 0.8 + 0.4 * aggregate_success_rate

    Parameters
    ----------
    memory : dict
        Transition memory from :func:`update_transition_memory`.
    regime_before : str
        Current regime label.
    attractor_before : str
        Current attractor identifier.
    strategy_id : str
        Strategy to evaluate.

    Returns
    -------
    float
        Transition bias in [0.8, 1.2].  Returns 1.0 if no history.
    """
    r_before = str(regime_before)
    a_before = str(attractor_before)
    s_id = str(strategy_id)

    # Collect matching entries with deterministic iteration order
    total_count = 0
    weighted_success = 0.0

    for key in sorted(memory.keys(), key=lambda k: str(k)):
        if (
            len(key) == 5
            and key[0] == r_before
            and key[1] == a_before
            and key[2] == s_id
        ):
            entry = memory[key]
            count = int(entry["count"])
            total_count += count
            weighted_success += float(entry["success_rate"]) * count

    if total_count == 0:
        return 1.0

    aggregate_success = weighted_success / total_count
    return _BIAS_BASE + _BIAS_SCALE * aggregate_success


# ---------------------------------------------------------------------------
# 4. High-level transition recording
# ---------------------------------------------------------------------------

def record_transition_outcome(
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
    regime_after: str,
    attractor_after: str,
    improvement_score: float,
) -> Dict[TransitionKey, Dict[str, Any]]:
    """Record a complete transition outcome.

    Convenience wrapper that computes the transition key and updates memory.

    Parameters
    ----------
    transition_memory : dict
        Existing transition memory.
    regime_before, attractor_before : str
        Regime context before strategy application.
    strategy_id : str
        Strategy that was applied.
    regime_after, attractor_after : str
        Regime context after strategy application.
    improvement_score : float
        Improvement score from evaluation.

    Returns
    -------
    dict
        New transition memory with the outcome recorded.
    """
    key = compute_transition_key(
        regime_before, attractor_before, strategy_id,
        regime_after, attractor_after,
    )
    return update_transition_memory(transition_memory, key, improvement_score)
