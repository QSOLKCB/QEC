"""Deterministic multi-step evaluation layer (v99.4.0).

Extends v99.3.0 transition learning with two-step lookahead scoring.
For each candidate strategy, estimates the expected value of applying it
now *and* the best follow-up strategy, using existing transition memory.

Two-step value formula::

    For strategy S in current state:
      step1_outcomes = all recorded (next_state, mean_delta, success_rate, count)
      For each next_state:
        best_delta2 = max mean_delta of any strategy from next_state
      two_step_value = weighted_mean(delta1 + best_delta2 * success_rate)

The second step is gated by success_rate to reduce optimistic bias:
only reliable transitions contribute their full follow-up value.

Multi-step factor::

    multi_step_factor = 1 + alpha * normalized_two_step_value
    alpha = 0.2, clamped to [0.8, 1.2]
    fallback = 1.0 when no data

All operations are deterministic, pure-functional, and bounded.
Dependencies: stdlib only.  No randomness, no ML, no recursion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# Type alias for transition memory keys
TransitionKey = Tuple[str, str, str, str, str]

# Horizon is fixed at 2 (current step + one lookahead).
# No recursion, no nested multi-step calls.
HORIZON: int = 2

# Multi-step constants
_ALPHA: float = 0.2
_FACTOR_MIN: float = 0.8
_FACTOR_MAX: float = 1.2


# ---------------------------------------------------------------------------
# 1. Expected transition outcomes
# ---------------------------------------------------------------------------

def get_expected_transition_outcomes(
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
) -> List[Tuple[Tuple[str, str], float, float, int]]:
    """Return deterministic list of expected outcomes for a strategy.

    Scans transition memory for all entries matching
    ``(regime_before, attractor_before, strategy_id, *, *)``.

    Parameters
    ----------
    regime_before : str
        Current regime label.
    attractor_before : str
        Current attractor identifier.
    strategy_id : str
        Strategy to evaluate.
    transition_memory : dict
        Transition memory from
        :func:`~qec.analysis.strategy_transition_learning.update_transition_memory`.

    Returns
    -------
    list of (next_state, mean_delta, success_rate, count)
        Each entry is ``((regime_after, attractor_after), mean_delta,
        success_rate, count)``.
        Sorted by key for deterministic ordering.  Empty list if no data.
    """
    r_before = str(regime_before)
    a_before = str(attractor_before)
    s_id = str(strategy_id)

    outcomes: List[Tuple[Tuple[str, str], float, float, int]] = []

    for key in sorted(transition_memory.keys(), key=lambda k: str(k)):
        if (
            len(key) == 5
            and key[0] == r_before
            and key[1] == a_before
            and key[2] == s_id
        ):
            entry = transition_memory[key]
            next_state = (str(key[3]), str(key[4]))
            mean_delta = float(entry["mean_delta"])
            success_rate = float(entry["success_rate"])
            count = int(entry["count"])
            outcomes.append((next_state, mean_delta, success_rate, count))

    return outcomes


# ---------------------------------------------------------------------------
# 2. Best follow-up delta from a given state
# ---------------------------------------------------------------------------

def _best_follow_up_delta(
    regime: str,
    attractor: str,
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
) -> float:
    """Find the best mean_delta achievable from a given state.

    Scans all strategies that have been tried from
    ``(regime, attractor, *, *, *)`` and returns the maximum
    count-weighted mean_delta.

    Parameters
    ----------
    regime : str
        Regime of the state to search from.
    attractor : str
        Attractor of the state to search from.
    transition_memory : dict
        Transition memory.

    Returns
    -------
    float
        Best mean_delta across all strategies from this state.
        Returns 0.0 if no data.
    """
    r = str(regime)
    a = str(attractor)

    # Aggregate per-strategy: weighted mean_delta across all outcomes
    strategy_deltas: Dict[str, Tuple[float, int]] = {}  # sid -> (sum_weighted, total_count)

    for key in sorted(transition_memory.keys(), key=lambda k: str(k)):
        if len(key) == 5 and key[0] == r and key[1] == a:
            sid = key[2]
            entry = transition_memory[key]
            count = int(entry["count"])
            delta = float(entry["mean_delta"])
            if sid not in strategy_deltas:
                strategy_deltas[sid] = (0.0, 0)
            prev_sum, prev_count = strategy_deltas[sid]
            strategy_deltas[sid] = (prev_sum + delta * count, prev_count + count)

    if not strategy_deltas:
        return 0.0

    # Compute weighted mean per strategy, return best
    best = None
    for sid in sorted(strategy_deltas.keys()):
        weighted_sum, total_count = strategy_deltas[sid]
        if total_count > 0:
            mean = weighted_sum / total_count
            if best is None or mean > best:
                best = mean

    return best if best is not None else 0.0


# ---------------------------------------------------------------------------
# 3. Two-step value computation
# ---------------------------------------------------------------------------

def _precompute_best_follow_up_deltas(
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
) -> Dict[Tuple[str, str], float]:
    """Precompute best follow-up mean_delta for every (regime, attractor) state.

    Single O(N) scan of transition_memory.  For each origin state
    ``(regime, attractor)``, aggregates per-strategy weighted mean_delta
    and keeps the maximum.

    Parameters
    ----------
    transition_memory : dict
        Transition memory.

    Returns
    -------
    dict
        Maps ``(regime, attractor)`` -> best achievable mean_delta.
    """
    # Accumulate per (state, strategy): (weighted_sum, total_count)
    accum: Dict[Tuple[str, str, str], Tuple[float, int]] = {}

    for key in sorted(transition_memory.keys(), key=lambda k: str(k)):
        if len(key) != 5:
            continue
        state_key = (str(key[0]), str(key[1]))
        sid = str(key[2])
        entry = transition_memory[key]
        count = int(entry["count"])
        delta = float(entry["mean_delta"])

        acc_key = (state_key[0], state_key[1], sid)
        if acc_key not in accum:
            accum[acc_key] = (0.0, 0)
        prev_sum, prev_count = accum[acc_key]
        accum[acc_key] = (prev_sum + delta * count, prev_count + count)

    # Reduce to best per state
    best: Dict[Tuple[str, str], float] = {}
    for acc_key in sorted(accum.keys()):
        state = (acc_key[0], acc_key[1])
        weighted_sum, total_count = accum[acc_key]
        if total_count > 0:
            mean = weighted_sum / total_count
            if state not in best or mean > best[state]:
                best[state] = mean

    return best


def compute_two_step_value(
    regime_before: str,
    attractor_before: str,
    strategy_id: str,
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
) -> float:
    """Compute two-step expected value for a strategy.

    For each recorded next_state from applying ``strategy_id`` in the
    current state, finds the best follow-up strategy and computes::

        two_step_value = weighted_mean(delta1 + best_delta2 * success_rate)

    weighted by transition count.  The follow-up delta is gated by
    ``success_rate`` to reduce optimistic bias — unreliable transitions
    contribute less follow-up value.

    Parameters
    ----------
    regime_before : str
        Current regime.
    attractor_before : str
        Current attractor.
    strategy_id : str
        Strategy to evaluate.
    transition_memory : dict
        Transition memory.

    Returns
    -------
    float
        Two-step expected value.  Returns 0.0 if no data.
    """
    outcomes = get_expected_transition_outcomes(
        regime_before, attractor_before, strategy_id, transition_memory,
    )

    if not outcomes:
        return 0.0

    # Precompute best follow-up deltas in one pass (O(N) total)
    best_follow_ups = _precompute_best_follow_up_deltas(transition_memory)

    # Collect count-weighted (delta1 + best_delta2 * success_rate) values
    weighted_sum = 0.0
    total_count = 0

    for next_state, mean_delta, success_rate, count in outcomes:
        best_delta2 = best_follow_ups.get(
            (str(next_state[0]), str(next_state[1])), 0.0,
        )

        combined = mean_delta + best_delta2 * success_rate
        weighted_sum += combined * count
        total_count += count

    if total_count == 0:
        return 0.0

    return weighted_sum / total_count


# ---------------------------------------------------------------------------
# 4. Multi-step factor
# ---------------------------------------------------------------------------

def compute_multi_step_factor(
    two_step_value: float,
    max_abs_value: float = 1.0,
) -> float:
    """Compute multiplicative multi-step factor from two-step value.

    Formula::

        normalized = two_step_value / max(|max_abs_value|, 1e-9)
        normalized = clamp(normalized, -1, 1)
        factor = 1 + alpha * normalized
        factor = clamp(factor, 0.8, 1.2)

    Parameters
    ----------
    two_step_value : float
        Raw two-step value from :func:`compute_two_step_value`.
    max_abs_value : float
        Normalization denominator.  Should be the maximum absolute
        two-step value across all candidate strategies, or 1.0
        if only one strategy or no normalization needed.

    Returns
    -------
    float
        Multi-step factor in [0.8, 1.2].  Returns 1.0 when
        two_step_value is 0.
    """
    if two_step_value == 0.0:
        return 1.0

    denom = max(abs(max_abs_value), 1e-9)
    normalized = two_step_value / denom
    normalized = max(-1.0, min(1.0, normalized))

    factor = 1.0 + _ALPHA * normalized
    return max(_FACTOR_MIN, min(_FACTOR_MAX, factor))


# ---------------------------------------------------------------------------
# 5. Batch multi-step factors for strategy set
# ---------------------------------------------------------------------------

def compute_multi_step_factors(
    regime_before: str,
    attractor_before: str,
    strategy_ids: List[str],
    transition_memory: Dict[TransitionKey, Dict[str, Any]],
) -> Dict[str, float]:
    """Compute multi-step factors for all candidate strategies.

    Normalizes two-step values across the candidate set so that
    the factor reflects relative multi-step advantage.

    Parameters
    ----------
    regime_before : str
        Current regime.
    attractor_before : str
        Current attractor.
    strategy_ids : list of str
        Strategy identifiers to evaluate.
    transition_memory : dict
        Transition memory.

    Returns
    -------
    dict
        Maps strategy_id -> multi_step_factor in [0.8, 1.2].
        Strategies with no data get 1.0.
    """
    if not strategy_ids or not transition_memory:
        return {sid: 1.0 for sid in sorted(strategy_ids)}

    # Compute raw two-step values (deterministic order)
    raw_values: Dict[str, float] = {}
    for sid in sorted(strategy_ids):
        raw_values[sid] = compute_two_step_value(
            regime_before, attractor_before, sid, transition_memory,
        )

    # Find max absolute value for normalization
    abs_values = [abs(v) for v in raw_values.values() if v != 0.0]
    max_abs = max(abs_values) if abs_values else 1.0

    # Compute factors
    factors: Dict[str, float] = {}
    for sid in sorted(strategy_ids):
        factors[sid] = compute_multi_step_factor(raw_values[sid], max_abs)

    return factors
