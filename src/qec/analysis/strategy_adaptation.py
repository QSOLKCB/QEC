"""Deterministic adaptation layer for strategy selection (v99.0.0).

Extends the strategy feedback loop from sense -> decide -> evaluate
to sense -> decide -> evaluate -> adapt.

Biases future strategy selection using evaluation history through
deterministic feedback shaping.  No randomness, no mutation, no ML.

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qec.analysis.strategy_transition import score_strategy


# ---------------------------------------------------------------------------
# Regime-aware evaluation weight overrides
# ---------------------------------------------------------------------------

REGIME_EVAL_WEIGHTS: Dict[str, Dict[str, float]] = {
    "unstable": {"basin": 0.6, "curvature": -0.3},
    "oscillatory": {"resonance": -0.4, "curvature": -0.2},
    "stable": {"consistency": 0.4, "divergence": -0.2},
}


# ---------------------------------------------------------------------------
# 1. History scoring
# ---------------------------------------------------------------------------

def compute_strategy_history_score(
    history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute aggregate scores from evaluation history.

    Parameters
    ----------
    history : list of dict
        Each entry must have ``"score"`` (float) and ``"direction"`` (str).

    Returns
    -------
    dict
        ``avg_score``, ``improvement_rate``, ``stability``.
    """
    if not history:
        return {
            "avg_score": 0.0,
            "improvement_rate": 0.0,
            "stability": 1.0,
        }

    scores = [float(h["score"]) for h in history]
    n = len(scores)

    avg_score = round(sum(scores) / n, 12)
    improvement_rate = sum(1 for h in history if h.get("direction") == "improved") / n
    stability = sum(1 for s in scores if abs(s) < 0.1) / n

    return {
        "avg_score": avg_score,
        "improvement_rate": improvement_rate,
        "stability": stability,
    }


# ---------------------------------------------------------------------------
# 2. Adaptation bias
# ---------------------------------------------------------------------------

def compute_adaptation_bias(
    history: List[Dict[str, Any]],
) -> float:
    """Compute a deterministic bias from evaluation history.

    The bias is a small adjustment in [-0.2, +0.2] that nudges
    future strategy scores based on past performance.

    Formula::

        bias = 0.15 * avg_score
             + 0.05 * improvement_rate
             - 0.05 * instability_penalty

    where instability_penalty = 1.0 - stability.

    Parameters
    ----------
    history : list of dict
        Evaluation history entries with ``"score"`` and ``"direction"``.

    Returns
    -------
    float
        Bias in [-0.2, +0.2].
    """
    if not history:
        return 0.0

    h_score = compute_strategy_history_score(history)

    instability_penalty = 1.0 - h_score["stability"]

    raw_bias = (
        0.15 * h_score["avg_score"]
        + 0.05 * h_score["improvement_rate"]
        - 0.05 * instability_penalty
    )

    return max(-0.2, min(0.2, raw_bias))


# ---------------------------------------------------------------------------
# 3. Regime-aware weight override
# ---------------------------------------------------------------------------

def get_regime_weights(regime: str) -> Dict[str, float]:
    """Return regime-specific evaluation weight adjustments.

    Parameters
    ----------
    regime : str
        Current regime name (e.g. ``"unstable"``, ``"stable"``).

    Returns
    -------
    dict
        Weight adjustments for evaluation dimensions, or empty dict.
    """
    return dict(REGIME_EVAL_WEIGHTS.get(regime, {}))


# ---------------------------------------------------------------------------
# 4. Adaptive strategy scoring
# ---------------------------------------------------------------------------

def score_strategy_adaptive(
    strategy: Any,
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score a strategy with adaptive bias from evaluation history.

    Computes:
        adaptive_score = base_score + bias

    Clamped to [0, 1].

    Parameters
    ----------
    strategy : dict or object
        Strategy to score.
    state : dict
        Current system state (from ``extract_state``).
    history : list of dict
        Evaluation history.

    Returns
    -------
    dict
        ``score`` (float), ``bias`` (float), ``base_score`` (float).
    """
    base_score = score_strategy(state, strategy)
    bias = compute_adaptation_bias(history)

    adaptive_score = max(0.0, min(1.0, base_score + bias))

    return {
        "score": adaptive_score,
        "bias": bias,
        "base_score": base_score,
    }


# ---------------------------------------------------------------------------
# 5. Multi-step trajectory scoring
# ---------------------------------------------------------------------------

def compute_trajectory_score(
    history: List[Dict[str, Any]],
) -> float:
    """Compute a trajectory score favoring recent evaluations.

    Uses linearly increasing weights [1, 2, 3, ..., N] to weight
    evaluation scores, giving more influence to recent entries.

    Parameters
    ----------
    history : list of dict
        Evaluation history entries with ``"score"``.

    Returns
    -------
    float
        Weighted mean of scores, or 0.0 if history is empty.
    """
    if not history:
        return 0.0

    n = len(history)
    weights = list(range(1, n + 1))
    total_weight = sum(weights)

    weighted_sum = sum(
        w * float(h["score"]) for w, h in zip(weights, history)
    )

    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# 6. Master adaptive selection
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


def select_strategy_adaptive(
    strategies: Dict[str, Any],
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the best strategy using adaptive scoring.

    For each strategy:
    1. Compute adaptive score (base + bias)
    2. Add small trajectory bonus (0.05 * trajectory_score)

    Tie-breaking (deterministic):
    1. Higher combined score
    2. Higher confidence
    3. Fewer params (simpler)
    4. Lexicographic strategy id

    Parameters
    ----------
    strategies : dict
        Maps strategy id -> strategy.
    state : dict
        Current system state.
    history : list of dict
        Evaluation history.

    Returns
    -------
    dict
        ``selected`` (str), ``score`` (float), ``bias`` (float),
        ``trajectory_score`` (float), ``strategy`` (the selected strategy).
    """
    if not strategies:
        return {
            "selected": "",
            "score": 0.0,
            "bias": 0.0,
            "trajectory_score": 0.0,
            "strategy": None,
        }

    traj_score = compute_trajectory_score(history)
    traj_bonus = 0.05 * traj_score

    scored: list = []
    bias_val = 0.0
    for sid in sorted(strategies.keys()):
        s = strategies[sid]
        adaptive = score_strategy_adaptive(s, state, history)
        bias_val = adaptive["bias"]
        combined = max(0.0, min(1.0, adaptive["score"] + traj_bonus))
        conf = _get_confidence(s)
        n_actions = _count_actions(s)
        scored.append((-combined, -conf, n_actions, sid, s))

    scored.sort()
    best = scored[0]

    return {
        "selected": best[3],
        "score": -best[0],
        "bias": bias_val,
        "trajectory_score": traj_score,
        "strategy": best[4],
    }
