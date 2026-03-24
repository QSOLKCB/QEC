"""Per-strategy memory and local adaptation layer (v99.3.0).

Extends v99.0 global adaptation to per-strategy memory, allowing the
system to prefer or avoid specific strategies based on their individual
performance history.

v99.2.0 adds regime-aware memory indexing and stability-weighted
strategy evaluation.  Memory keys are upgraded from flat strategy IDs
to (regime_key, strategy_id) tuples, with fallback to global aggregation
when regime-specific data is unavailable.

v99.3.0 adds transition learning: a multiplicative bias derived from
historical transition outcomes.  When transition memory is available,
strategy scoring uses the formula::

    final_score = base_score * stability_weight * transition_bias

where transition_bias defaults to 1.0 (neutral) when no history exists.

Each strategy carries its own performance record, enabling targeted
adaptation rather than uniform global bias.

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.strategy_adaptation import compute_adaptation_bias
from qec.analysis.strategy_transition import score_strategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_CAP: int = 10


# ---------------------------------------------------------------------------
# 1. Memory update
# ---------------------------------------------------------------------------

def update_strategy_memory(
    memory: Dict[str, List[Dict[str, Any]]],
    strategy_id: str,
    evaluation: Dict[str, Any],
    cap: int = DEFAULT_MEMORY_CAP,
) -> Dict[str, List[Dict[str, Any]]]:
    """Append an evaluation record to a strategy's memory.

    Returns a **new** memory dict — the input is never mutated.

    Parameters
    ----------
    memory : dict
        Maps strategy_id -> list of ``{"score": float, "outcome": str}``.
    strategy_id : str
        Strategy to update.
    evaluation : dict
        Must contain ``"score"`` (float) and ``"outcome"`` (str).
    cap : int
        Maximum number of records per strategy (default 10).

    Returns
    -------
    dict
        New memory with the record appended and cap enforced.
    """
    new_memory = {k: list(v) for k, v in memory.items()}

    record = {
        "score": float(evaluation["score"]),
        "outcome": str(evaluation["outcome"]),
    }

    if strategy_id not in new_memory:
        new_memory[strategy_id] = []

    entries = list(new_memory[strategy_id])
    entries.append(record)

    # Enforce cap — keep most recent entries
    if len(entries) > cap:
        entries = entries[-cap:]

    new_memory[strategy_id] = entries
    return new_memory


# ---------------------------------------------------------------------------
# 2. Strategy performance score
# ---------------------------------------------------------------------------

def compute_strategy_performance(
    memory: Dict[str, List[Dict[str, Any]]],
    strategy_id: str,
) -> float:
    """Compute a performance score for a strategy from its memory.

    If no history exists, returns 0.0.

    Formula::

        performance = 0.6 * avg_score
                    + 0.3 * improvement_rate
                    - 0.1 * instability_penalty

    Clamped to [-1, 1].

    Parameters
    ----------
    memory : dict
        Per-strategy memory.
    strategy_id : str
        Strategy to evaluate.

    Returns
    -------
    float
        Performance in [-1, 1].
    """
    records = memory.get(strategy_id, [])
    if not records:
        return 0.0

    scores = [float(r["score"]) for r in records]
    n = len(scores)

    avg_score = sum(scores) / n

    # Improvement rate: fraction of "improved" outcomes
    improvement_rate = sum(
        1 for r in records if r.get("outcome") == "improved"
    ) / n

    # Stability: fraction of scores with |score| < 0.1
    stability = sum(1 for s in scores if abs(s) < 0.1) / n
    instability_penalty = 1.0 - stability

    performance = (
        0.6 * avg_score
        + 0.3 * improvement_rate
        - 0.1 * instability_penalty
    )

    return max(-1.0, min(1.0, performance))


# ---------------------------------------------------------------------------
# 3. Strategy bias
# ---------------------------------------------------------------------------

def compute_strategy_bias(
    memory: Dict[str, List[Dict[str, Any]]],
    strategy_id: str,
) -> float:
    """Compute a local bias for a strategy from its performance history.

    bias = 0.2 * performance, clamped to [-0.2, +0.2].

    Parameters
    ----------
    memory : dict
        Per-strategy memory.
    strategy_id : str
        Strategy to evaluate.

    Returns
    -------
    float
        Bias in [-0.2, +0.2].
    """
    perf = compute_strategy_performance(memory, strategy_id)
    bias = 0.2 * perf
    return max(-0.2, min(0.2, bias))


# ---------------------------------------------------------------------------
# 4. Memory-aware scoring
# ---------------------------------------------------------------------------

def score_strategy_with_memory(
    strategy: Any,
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
    memory: Dict[Any, List[Dict[str, Any]]],
    strategy_id: str,
    regime_key: Optional[Tuple[str, str]] = None,
    transition_memory: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a strategy with both global and per-strategy bias.

    When *regime_key* and *transition_memory* are both provided, uses
    the v99.3.0 multiplicative formula::

        score = base_score * stability_weight * transition_bias

    where ``stability_weight`` comes from regime-aware evaluation and
    ``transition_bias`` comes from transition learning history.

    When *regime_key* is provided without transition memory, uses
    the v99.2.0 additive formula (backward compatible).

    When neither is provided, uses flat memory bias (v99.1.0).

    Parameters
    ----------
    strategy : dict or object
        Strategy to score.
    state : dict
        Current system state.
    history : list of dict
        Global evaluation history.
    memory : dict
        Per-strategy memory (flat or regime-indexed).
    strategy_id : str
        ID of this strategy in *memory*.
    regime_key : tuple, optional
        Regime key from :func:`compute_regime_key`.  When provided,
        activates regime-aware scoring with stability weighting.
    transition_memory : dict, optional
        Transition memory from
        :func:`~qec.analysis.strategy_transition_learning.update_transition_memory`.
        When provided together with *regime_key*, activates v99.3.0
        multiplicative transition bias.

    Returns
    -------
    dict
        ``score``, ``global_bias``, ``local_bias``, ``transition_bias``.
    """
    base_score = score_strategy(state, strategy)
    global_bias = compute_adaptation_bias(history)
    transition_bias = 1.0

    if regime_key is not None and transition_memory is not None:
        # v99.3.0: multiplicative formula with transition learning
        from qec.analysis.strategy_transition_learning import (
            compute_transition_bias,
        )
        regime_score = compute_regime_aware_score(memory, regime_key, strategy_id)
        stability_weight = regime_score["stability_weight"]
        transition_bias = compute_transition_bias(
            transition_memory, regime_key[0], regime_key[1], strategy_id,
        )
        score = max(0.0, min(1.0, base_score * stability_weight * transition_bias))
        local_bias = score - base_score  # effective local adjustment
    elif regime_key is not None:
        # v99.2.0: additive formula with stability weighting
        regime_score = compute_regime_aware_score(memory, regime_key, strategy_id)
        local_bias = 0.2 * regime_score["final_score"]
        local_bias = max(-0.2, min(0.2, local_bias))
        score = max(0.0, min(1.0, base_score + global_bias + local_bias))
    else:
        # v99.1.0: flat memory bias
        local_bias = compute_strategy_bias(memory, strategy_id)
        score = max(0.0, min(1.0, base_score + global_bias + local_bias))

    return {
        "score": score,
        "global_bias": global_bias,
        "local_bias": local_bias,
        "transition_bias": transition_bias,
    }


# ---------------------------------------------------------------------------
# 5. Memory-aware selection helpers
# ---------------------------------------------------------------------------

def _get_confidence(strategy: Any) -> float:
    """Extract confidence from a strategy."""
    if isinstance(strategy, dict):
        return float(strategy.get("confidence", 0.0))
    return float(getattr(strategy, "confidence", 0.0))


def _count_actions(strategy: Any) -> int:
    """Count param entries (simplicity measure)."""
    if isinstance(strategy, dict):
        params = strategy.get("params", {})
    else:
        params = getattr(strategy, "params", {})
    return max(len(params), 1)


# ---------------------------------------------------------------------------
# 6. Memory-aware selection
# ---------------------------------------------------------------------------

def select_strategy_with_memory(
    strategies: Dict[str, Any],
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
    memory: Dict[Any, List[Dict[str, Any]]],
    regime_key: Optional[Tuple[str, str]] = None,
    transition_memory: Optional[Dict[Any, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Select the best strategy using per-strategy memory bias.

    For each strategy, computes a memory-aware score.  When
    *transition_memory* is provided, uses v99.3.0 multiplicative
    formula.  Tie-breaking is deterministic:
    score -> confidence -> simplicity -> id.

    Parameters
    ----------
    strategies : dict
        Maps strategy id -> strategy.
    state : dict
        Current system state.
    history : list of dict
        Global evaluation history.
    memory : dict
        Per-strategy memory (flat or regime-indexed).
    regime_key : tuple, optional
        Regime key from :func:`compute_regime_key`.
    transition_memory : dict, optional
        Transition memory for v99.3.0 transition learning.

    Returns
    -------
    dict
        ``selected``, ``score``, ``global_bias``, ``local_bias``,
        ``transition_bias``.
    """
    if not strategies:
        return {
            "selected": "",
            "score": 0.0,
            "global_bias": 0.0,
            "local_bias": 0.0,
            "transition_bias": 1.0,
        }

    scored: list = []
    for sid in sorted(strategies.keys()):
        s = strategies[sid]
        result = score_strategy_with_memory(
            s, state, history, memory, sid,
            regime_key=regime_key,
            transition_memory=transition_memory,
        )
        conf = _get_confidence(s)
        n_actions = _count_actions(s)
        scored.append((
            -result["score"],
            -conf,
            n_actions,
            sid,
            s,
            result["global_bias"],
            result["local_bias"],
            result["transition_bias"],
        ))

    scored.sort()
    best = scored[0]

    return {
        "selected": best[3],
        "score": -best[0],
        "global_bias": best[5],
        "local_bias": best[6],
        "transition_bias": best[7],
    }


# ---------------------------------------------------------------------------
# 7. Regime key computation (v99.2.0)
# ---------------------------------------------------------------------------

_BASIN_THRESHOLDS: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def compute_attractor_id(basin_score: float) -> str:
    """Compute a deterministic attractor identifier from basin score.

    Quantizes basin_score into discrete buckets for stable indexing.
    Thresholds: 0.2, 0.4, 0.6, 0.8.

    Parameters
    ----------
    basin_score : float
        Basin score in [0, 1].

    Returns
    -------
    str
        Deterministic attractor identifier (e.g. ``"basin_0"`` .. ``"basin_4"``).
    """
    score = float(basin_score)
    bucket = 0
    for threshold in _BASIN_THRESHOLDS:
        if score >= threshold:
            bucket += 1
        else:
            break
    return f"basin_{bucket}"


def compute_regime_key(
    regime_label: str,
    attractor_id: str,
) -> Tuple[str, str]:
    """Compute a deterministic, hashable regime key.

    Parameters
    ----------
    regime_label : str
        Current regime classification (e.g. ``"unstable"``).
    attractor_id : str
        Attractor identifier from :func:`compute_attractor_id`.

    Returns
    -------
    tuple of (str, str)
        ``(regime_label, attractor_id)`` — hashable and deterministic.
    """
    return (str(regime_label), str(attractor_id))


# ---------------------------------------------------------------------------
# 8. Regime-indexed memory (v99.2.0)
# ---------------------------------------------------------------------------

def update_regime_memory(
    memory: Dict[Any, List[Dict[str, Any]]],
    regime_key: Tuple[str, str],
    strategy_id: str,
    event: Dict[str, Any],
    cap: int = DEFAULT_MEMORY_CAP,
) -> Dict[Any, List[Dict[str, Any]]]:
    """Append an event to regime-indexed memory.

    Returns a **new** memory dict — the input is never mutated.

    The memory key is ``(regime_key, strategy_id)`` where regime_key is
    a tuple from :func:`compute_regime_key`.

    Parameters
    ----------
    memory : dict
        Maps ``(regime_key, strategy_id)`` → list of events.
    regime_key : tuple
        Output of :func:`compute_regime_key`.
    strategy_id : str
        Strategy identifier.
    event : dict
        Must contain ``"step"`` (int), ``"score"`` (float),
        ``"metrics"`` (dict of str → float).
    cap : int
        Maximum events per key (default 10).

    Returns
    -------
    dict
        New memory with event appended and cap enforced.
    """
    new_memory = {k: list(v) for k, v in memory.items()}

    record = {
        "step": int(event["step"]),
        "score": float(event["score"]),
        "metrics": {str(k): float(v) for k, v in event.get("metrics", {}).items()},
    }

    key = (regime_key, strategy_id)
    if key not in new_memory:
        new_memory[key] = []

    entries = list(new_memory[key])
    entries.append(record)

    if len(entries) > cap:
        entries = entries[-cap:]

    new_memory[key] = entries
    return new_memory


def query_regime_memory(
    memory: Dict[Any, List[Dict[str, Any]]],
    regime_key: Tuple[str, str],
    strategy_id: str,
) -> List[Dict[str, Any]]:
    """Query regime-indexed memory with fallback to global strategy memory.

    Lookup order:

    1. ``memory[(regime_key, strategy_id)]`` — exact regime match
    2. Aggregate all ``memory[(..., strategy_id)]`` entries — global fallback
    3. Empty list

    Parameters
    ----------
    memory : dict
        Regime-indexed memory.
    regime_key : tuple
        Current regime key.
    strategy_id : str
        Strategy to query.

    Returns
    -------
    list
        Event records (copies), or empty list.
    """
    key = (regime_key, strategy_id)
    if key in memory and memory[key]:
        return list(memory[key])

    # Fallback: aggregate all entries for this strategy across regimes.
    # Sort keys by string representation for deterministic ordering.
    fallback: List[Dict[str, Any]] = []
    for k in sorted(memory.keys(), key=lambda x: str(x)):
        if isinstance(k, tuple) and len(k) == 2 and k[1] == strategy_id:
            fallback.extend(memory[k])

    return fallback


# ---------------------------------------------------------------------------
# 9. Stability weighting (v99.2.0)
# ---------------------------------------------------------------------------

def compute_stability_weight(scores: List[float]) -> float:
    """Compute stability weight from a list of scores.

    Formula::

        stability = 1 / (1 + variance(scores))

    Parameters
    ----------
    scores : list of float
        Score values from memory events.

    Returns
    -------
    float
        Stability weight in (0, 1].  Returns 1.0 for empty or
        single-entry lists.
    """
    if len(scores) <= 1:
        return 1.0

    n = len(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n

    # Safe division: variance >= 0, so denominator >= 1
    return 1.0 / (1.0 + variance)


# ---------------------------------------------------------------------------
# 10. Regime-aware scoring (v99.2.0)
# ---------------------------------------------------------------------------

def compute_regime_aware_score(
    memory: Dict[Any, List[Dict[str, Any]]],
    regime_key: Tuple[str, str],
    strategy_id: str,
) -> Dict[str, Any]:
    """Compute stability-weighted score for a strategy in a regime context.

    Formula::

        final_score = mean_score * stability_weight

    where ``stability_weight = 1 / (1 + variance(scores))``.

    Parameters
    ----------
    memory : dict
        Regime-indexed memory.
    regime_key : tuple
        Current regime key.
    strategy_id : str
        Strategy to evaluate.

    Returns
    -------
    dict
        ``"mean_score"``, ``"stability_weight"``, ``"final_score"``,
        ``"n_events"``.
    """
    events = query_regime_memory(memory, regime_key, strategy_id)

    if not events:
        return {
            "mean_score": 0.0,
            "stability_weight": 1.0,
            "final_score": 0.0,
            "n_events": 0,
        }

    scores = [float(e["score"]) for e in events]
    mean_score = sum(scores) / len(scores)
    stability = compute_stability_weight(scores)
    final_score = mean_score * stability

    return {
        "mean_score": mean_score,
        "stability_weight": stability,
        "final_score": final_score,
        "n_events": len(events),
    }
