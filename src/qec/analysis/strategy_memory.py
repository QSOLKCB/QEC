"""Per-strategy memory and local adaptation layer (v99.1.0).

Extends v99.0 global adaptation to per-strategy memory, allowing the
system to prefer or avoid specific strategies based on their individual
performance history.

Each strategy carries its own performance record, enabling targeted
adaptation rather than uniform global bias.

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

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
    memory: Dict[str, List[Dict[str, Any]]],
    strategy_id: str,
) -> Dict[str, Any]:
    """Score a strategy with both global and per-strategy bias.

    score = base_score + global_bias + local_bias, clamped to [0, 1].

    Parameters
    ----------
    strategy : dict or object
        Strategy to score.
    state : dict
        Current system state.
    history : list of dict
        Global evaluation history.
    memory : dict
        Per-strategy memory.
    strategy_id : str
        ID of this strategy in *memory*.

    Returns
    -------
    dict
        ``score``, ``global_bias``, ``local_bias``.
    """
    base_score = score_strategy(state, strategy)
    global_bias = compute_adaptation_bias(history)
    local_bias = compute_strategy_bias(memory, strategy_id)

    score = max(0.0, min(1.0, base_score + global_bias + local_bias))

    return {
        "score": score,
        "global_bias": global_bias,
        "local_bias": local_bias,
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
    memory: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Select the best strategy using per-strategy memory bias.

    For each strategy, computes a memory-aware score (base + global + local).
    Tie-breaking is deterministic: score -> confidence -> simplicity -> id.

    Parameters
    ----------
    strategies : dict
        Maps strategy id -> strategy.
    state : dict
        Current system state.
    history : list of dict
        Global evaluation history.
    memory : dict
        Per-strategy memory.

    Returns
    -------
    dict
        ``selected``, ``score``, ``global_bias``, ``local_bias``.
    """
    if not strategies:
        return {
            "selected": "",
            "score": 0.0,
            "global_bias": 0.0,
            "local_bias": 0.0,
        }

    scored: list = []
    for sid in sorted(strategies.keys()):
        s = strategies[sid]
        result = score_strategy_with_memory(s, state, history, memory, sid)
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
        ))

    scored.sort()
    best = scored[0]

    return {
        "selected": best[3],
        "score": -best[0],
        "global_bias": best[5],
        "local_bias": best[6],
    }
