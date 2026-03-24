"""Basin-aware strategy transition layer (v98.8.1).

Deterministic control layer that selects strategies based on current regime
and basin score, and computes transitions between strategies to move toward
more stable basins.

Uses existing metrics only — no new signals introduced.
Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Centralized scoring configuration
# ---------------------------------------------------------------------------

STRATEGY_WEIGHTS: Dict[str, float] = {
    "phi": 0.4,
    "consistency": 0.3,
    "curvature": -0.3,
    "divergence": -0.2,
    "resonance": -0.2,
    "complexity": -0.2,
}

REGIME_ACTION_PREFERENCES: Dict[str, Dict[str, float]] = {
    "unstable": {"reduce_instability": 1.0},
    "oscillatory": {"reduce_oscillation": 1.0},
    "transitional": {"increase_stability": 1.0},
    "stable": {"minimal": 1.0},
}

# Per-regime metric adjustment weights for score_strategy.
# These small adjustments are added to the base regime×action score.
_REGIME_METRIC_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "unstable": {"curvature": 0.05, "complexity": 0.05},
    "oscillatory": {"resonance": 0.05},
    "transitional": {"divergence": 0.05, "consistency": -0.03},
    "stable": {},
}
_STABLE_FLAT_PENALTY: float = -0.05


# ---------------------------------------------------------------------------
# Centralized transition labels
# ---------------------------------------------------------------------------

TRANSITION_LABELS: Dict[tuple, str] = {
    ("unstable", "stable"): "increase_stability",
    ("unstable", "transitional"): "increase_stability",
    ("unstable", "mixed"): "reduce_instability",
    ("unstable", "oscillatory"): "reduce_instability",
    ("oscillatory", "stable"): "reduce_oscillation",
    ("oscillatory", "transitional"): "reduce_oscillation",
    ("oscillatory", "mixed"): "reduce_oscillation",
    ("transitional", "stable"): "increase_stability",
    ("mixed", "stable"): "increase_stability",
}

DEFAULT_TRANSITION_LABEL: str = "reduce_instability"


# ---------------------------------------------------------------------------
# 1. State extraction
# ---------------------------------------------------------------------------

def extract_state(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat state dict from nested metrics.

    Parameters
    ----------
    metrics : dict
        Must contain "attractor" (with "regime", "basin_score"),
        "field" (with "phi_alignment", "curvature", "resonance", "complexity"),
        and "multiscale" (with "scale_consistency", "scale_divergence").

    Returns
    -------
    dict
        Flat state with keys: regime, basin_score, phi, consistency,
        divergence, curvature, resonance, complexity.
    """
    attractor = metrics.get("attractor", {})
    field = metrics.get("field", {})
    multi = metrics.get("multiscale", {})
    curv = field.get("curvature", {})

    return {
        "regime": attractor.get("regime", "mixed"),
        "basin_score": float(attractor.get("basin_score", 0.0)),
        "phi": float(field.get("phi_alignment", 0.0)),
        "consistency": float(multi.get("scale_consistency", 0.0)),
        "divergence": float(multi.get("scale_divergence", 0.0)),
        "curvature": float(curv.get("abs_curvature", 0.0))
        if isinstance(curv, dict)
        else float(curv),
        "resonance": float(field.get("resonance", 0.0)),
        "complexity": float(field.get("complexity", 0.0)),
    }


# ---------------------------------------------------------------------------
# 2. Strategy scoring
# ---------------------------------------------------------------------------

# Fixed action-type scores per regime.  Each regime maps action types to a
# base score.  Higher is better.
_REGIME_ACTION_SCORES: Dict[str, Dict[str, float]] = {
    "unstable": {
        "damping": 0.9,
        "scaling": 0.5,
        "rotation": 0.3,
        "reweight": 0.6,
        "adjust_damping": 0.9,
    },
    "oscillatory": {
        "damping": 0.8,
        "scaling": 0.4,
        "rotation": 0.6,
        "reweight": 0.5,
        "adjust_damping": 0.7,
    },
    "transitional": {
        "damping": 0.4,
        "scaling": 0.7,
        "rotation": 0.5,
        "reweight": 0.8,
        "adjust_damping": 0.5,
    },
    "stable": {
        "damping": 0.1,
        "scaling": 0.2,
        "rotation": 0.1,
        "reweight": 0.1,
        "adjust_damping": 0.1,
    },
    "mixed": {
        "damping": 0.5,
        "scaling": 0.5,
        "rotation": 0.5,
        "reweight": 0.5,
        "adjust_damping": 0.5,
    },
}


def _get_action_type(strategy: Any) -> str:
    """Extract action_type from a strategy (dict or object)."""
    if isinstance(strategy, dict):
        return str(strategy.get("action_type", ""))
    return str(getattr(strategy, "action_type", ""))


def _get_confidence(strategy: Any) -> float:
    """Extract confidence from a strategy, defaulting to 0.0."""
    if isinstance(strategy, dict):
        return float(strategy.get("confidence", 0.0))
    return float(getattr(strategy, "confidence", 0.0))


def _get_params(strategy: Any) -> Dict[str, Any]:
    """Extract params dict from a strategy."""
    if isinstance(strategy, dict):
        return dict(strategy.get("params", {}))
    return dict(getattr(strategy, "params", {}))


def _count_actions(strategy: Any) -> int:
    """Count the number of actions/params in a strategy (simplicity)."""
    params = _get_params(strategy)
    return max(len(params), 1)


def score_strategy(state: Dict[str, Any], strategy: Any) -> float:
    """Score a strategy given current system state.

    Heuristic, deterministic scoring:
    - Looks up base score from regime × action_type table
    - Adjusts based on state metrics to prefer strategies that address
      the dominant instability signal.

    Parameters
    ----------
    state : dict
        Output of :func:`extract_state`.
    strategy : dict or object
        Must have ``action_type`` and ``params``.

    Returns
    -------
    float
        Score in [0, 1].  Higher is better.
    """
    regime = state.get("regime", "mixed")
    action_type = _get_action_type(strategy)

    regime_scores = _REGIME_ACTION_SCORES.get(regime, _REGIME_ACTION_SCORES["mixed"])
    base = regime_scores.get(action_type, 0.3)

    # Metric-driven adjustments (small, bounded) using centralized weights
    adjustment = 0.0
    metric_weights = _REGIME_METRIC_ADJUSTMENTS.get(regime, {})
    for metric, weight in sorted(metric_weights.items()):
        adjustment += weight * state.get(metric, 0.0)
    if regime == "stable":
        adjustment += _STABLE_FLAT_PENALTY

    score = base + adjustment
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 3. Strategy selection
# ---------------------------------------------------------------------------

def select_strategy(
    state: Dict[str, Any],
    strategies: Dict[str, Any],
) -> Dict[str, Any]:
    """Select the best strategy for the current state.

    Tie-breaking (deterministic):
    1. Higher confidence
    2. Fewer params (simpler)
    3. Lexicographic strategy id

    Parameters
    ----------
    state : dict
        Output of :func:`extract_state`.
    strategies : dict
        Maps strategy id → strategy (dict or object).

    Returns
    -------
    dict
        ``{"id": str, "strategy": <strategy>, "score": float}``
    """
    if not strategies:
        return {"id": "", "strategy": None, "score": 0.0}

    scored: List[tuple] = []
    for sid in sorted(strategies.keys()):
        s = strategies[sid]
        sc = score_strategy(state, s)
        conf = _get_confidence(s)
        n_actions = _count_actions(s)
        # sort key: (-score, -confidence, n_actions, sid)
        scored.append((-sc, -conf, n_actions, sid, s))

    scored.sort()
    best = scored[0]
    return {
        "id": best[3],
        "strategy": best[4],
        "score": -best[0],
    }


# ---------------------------------------------------------------------------
# 4. Transition detection
# ---------------------------------------------------------------------------

def should_transition(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> bool:
    """Determine whether a strategy transition is warranted.

    A transition is triggered if:
    - The regime has changed, OR
    - The basin_score has decreased.

    Parameters
    ----------
    prev_state, curr_state : dict
        Outputs of :func:`extract_state`.

    Returns
    -------
    bool
    """
    if prev_state.get("regime") != curr_state.get("regime"):
        return True
    if curr_state.get("basin_score", 0.0) < prev_state.get("basin_score", 0.0):
        return True
    return False


# ---------------------------------------------------------------------------
# 5. Transition computation
# ---------------------------------------------------------------------------

def compute_transition(
    prev_strategy: Dict[str, Any],
    new_strategy: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]] = None,
    curr_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a transition record between two strategy selections.

    Parameters
    ----------
    prev_strategy, new_strategy : dict
        Output of :func:`select_strategy` (must have "id").
    prev_state, curr_state : dict, optional
        States used to determine the transition label.

    Returns
    -------
    dict
        ``{"from": str, "to": str, "change": str}``
    """
    from_id = prev_strategy.get("id", "")
    to_id = new_strategy.get("id", "")

    # Fallback: when regime pair is not in TRANSITION_LABELS, use
    # DEFAULT_TRANSITION_LABEL to ensure explicit, deterministic behaviour.
    change = DEFAULT_TRANSITION_LABEL
    if prev_state and curr_state:
        prev_regime = prev_state.get("regime", "mixed")
        curr_regime = curr_state.get("regime", "mixed")
        change = TRANSITION_LABELS.get(
            (prev_regime, curr_regime), DEFAULT_TRANSITION_LABEL,
        )

    return {
        "from": from_id,
        "to": to_id,
        "change": change,
    }


# ---------------------------------------------------------------------------
# 6. Master function
# ---------------------------------------------------------------------------

def select_next_strategy(
    metrics: Dict[str, Any],
    strategies: Dict[str, Any],
    prev_strategy: Optional[Dict[str, Any]] = None,
    prev_state: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    memory: Optional[Dict[str, list]] = None,
) -> Dict[str, Any]:
    """Select the next strategy and compute any transition.

    This is the main entry point for the strategy transition layer.

    Parameters
    ----------
    metrics : dict
        Full metrics dict with "attractor", "field", "multiscale" keys.
    strategies : dict
        Maps strategy id → strategy (dict or object).
    prev_strategy : dict, optional
        Previous output of :func:`select_strategy`.
    prev_state : dict, optional
        Previous output of :func:`extract_state`.
    history : list of dict, optional
        Evaluation history.  When provided, uses adaptive selection
        (biased by history) instead of base scoring.
    memory : dict, optional
        Per-strategy memory.  When provided together with *history*,
        uses memory-aware selection with per-strategy bias.

    Returns
    -------
    dict
        ``{"strategy": <selected>, "state": dict, "transition": dict|None}``
        When adaptive selection is used, also includes ``"adaptation"``
        with bias and trajectory_score.
    """
    state = extract_state(metrics)

    adaptation_info = None
    if history and memory is not None:
        from qec.analysis.strategy_memory import select_strategy_with_memory
        mem_result = select_strategy_with_memory(
            strategies, state, history, memory,
        )
        selected = {
            "id": mem_result["selected"],
            "strategy": strategies.get(mem_result["selected"]),
            "score": mem_result["score"],
        }
        adaptation_info = {
            "bias": mem_result["global_bias"],
            "trajectory_score": 0.0,
            "local_bias": mem_result["local_bias"],
        }
    elif history:
        from qec.analysis.strategy_adaptation import select_strategy_adaptive
        adaptive = select_strategy_adaptive(strategies, state, history)
        selected = {
            "id": adaptive["selected"],
            "strategy": adaptive["strategy"],
            "score": adaptive["score"],
        }
        adaptation_info = {
            "bias": adaptive["bias"],
            "trajectory_score": adaptive["trajectory_score"],
        }
    else:
        selected = select_strategy(state, strategies)

    transition = None
    if prev_strategy is not None:
        if prev_state is not None:
            # Full transition detection with regime comparison and magnitude
            if should_transition(prev_state, state):
                transition = compute_transition(
                    prev_strategy, selected, prev_state, state,
                )
        else:
            # Partial input: prev_state missing — detect transition by
            # strategy change only, skip magnitude calculation.
            if prev_strategy.get("id", "") != selected.get("id", ""):
                transition = compute_transition(prev_strategy, selected)

    result = {
        "strategy": selected,
        "state": state,
        "transition": transition,
    }
    if adaptation_info is not None:
        result["adaptation"] = adaptation_info
    return result
