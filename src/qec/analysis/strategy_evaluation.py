"""Deterministic strategy evaluation and feedback loop (v98.9.0).

Measures whether a chosen strategy improved system stability by comparing
before/after metric states.  Produces structured feedback signals and
optionally tracks simple historical performance.

Observation-only: does not modify strategies, decoder inputs, or core logic.

Dependencies: stdlib only (no numpy required).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. Extract evaluation state
# ---------------------------------------------------------------------------

def extract_eval_state(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a flat evaluation state from full metrics.

    Parameters
    ----------
    metrics : dict
        Must contain ``attractor``, ``field``, and ``multiscale`` sub-dicts
        as produced by the experiment pipeline.

    Returns
    -------
    dict
        Flat state with regime, basin_score, phi, consistency, divergence,
        curvature, resonance, complexity.
    """
    attractor = metrics["attractor"]
    field = metrics["field"]
    multiscale = metrics["multiscale"]

    return {
        "regime": attractor["regime"],
        "basin_score": float(attractor["basin_score"]),
        "phi": float(field["phi_alignment"]),
        "consistency": float(multiscale["scale_consistency"]),
        "divergence": float(multiscale["scale_divergence"]),
        "curvature": float(field["curvature"]["abs_curvature"]),
        "resonance": float(field["resonance"]),
        "complexity": float(field["complexity"]),
    }


# ---------------------------------------------------------------------------
# 2. Compute state delta
# ---------------------------------------------------------------------------

_DELTA_KEYS = (
    "basin_score",
    "phi",
    "consistency",
    "divergence",
    "curvature",
    "resonance",
    "complexity",
)


def compute_state_delta(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> Dict[str, float]:
    """Compute element-wise deltas between two evaluation states.

    Returns
    -------
    dict
        Keys are ``<field>_delta`` for each numeric field.
    """
    return {
        f"{key}_delta": float(curr_state[key]) - float(prev_state[key])
        for key in _DELTA_KEYS
    }


# ---------------------------------------------------------------------------
# 3. Evaluate improvement
# ---------------------------------------------------------------------------

def evaluate_improvement(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Score whether the system improved between two states.

    Improvement score formula::

        score = +0.5 * basin_delta
                +0.2 * consistency_delta
                -0.2 * curvature_delta
                -0.1 * divergence_delta

    The score is clamped to [-1, 1].

    Returns
    -------
    dict
        ``improved`` (bool), ``score`` (float), ``direction`` (str).
    """
    delta = compute_state_delta(prev_state, curr_state)

    raw = (
        0.5 * delta["basin_score_delta"]
        + 0.2 * delta["consistency_delta"]
        - 0.2 * delta["curvature_delta"]
        - 0.1 * delta["divergence_delta"]
    )

    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, raw))

    if score > 0.01:
        direction = "improved"
    elif score < -0.01:
        direction = "degraded"
    else:
        direction = "neutral"

    return {
        "improved": score > 0.01,
        "score": score,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# 4. Classify outcome
# ---------------------------------------------------------------------------

_STABLE_REGIMES = frozenset({"stable"})
_UNSTABLE_REGIMES = frozenset({"unstable"})
_OSCILLATORY_REGIMES = frozenset({"oscillatory"})


def classify_outcome(
    prev_state: Dict[str, Any],
    curr_state: Dict[str, Any],
) -> str:
    """Classify the transition outcome between two states.

    Returns one of:
        ``"stabilized"`` -- stable to stable with higher basin score
        ``"recovered"``  -- unstable to stable
        ``"damped"``     -- oscillatory to stable
        ``"regressed"``  -- stable to unstable
        ``"neutral"``    -- no significant regime change
    """
    prev_regime = prev_state["regime"]
    curr_regime = curr_state["regime"]

    if curr_regime in _STABLE_REGIMES:
        if prev_regime in _UNSTABLE_REGIMES:
            return "recovered"
        if prev_regime in _OSCILLATORY_REGIMES:
            return "damped"
        if prev_regime in _STABLE_REGIMES:
            if curr_state["basin_score"] > prev_state["basin_score"] + 0.01:
                return "stabilized"
            return "neutral"

    if curr_regime in _UNSTABLE_REGIMES and prev_regime in _STABLE_REGIMES:
        return "regressed"

    return "neutral"


# ---------------------------------------------------------------------------
# 5. Track history
# ---------------------------------------------------------------------------

_MAX_HISTORY = 20


def update_history(
    history: List[Dict[str, Any]],
    evaluation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Append an evaluation record to history, capping at *_MAX_HISTORY*.

    Returns a new list (does not mutate the input).
    """
    updated = list(history)
    updated.append(dict(evaluation))
    if len(updated) > _MAX_HISTORY:
        updated = updated[-_MAX_HISTORY:]
    return updated


# ---------------------------------------------------------------------------
# 6. Master evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    prev_metrics: Dict[str, Any],
    curr_metrics: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Evaluate whether a strategy improved system stability.

    Parameters
    ----------
    prev_metrics : dict
        Full metrics dict (field + multiscale + attractor) *before* strategy.
    curr_metrics : dict
        Full metrics dict *after* strategy.
    history : list, optional
        Previous evaluation records.  If provided, the new evaluation is
        appended and the updated history is included in the result.

    Returns
    -------
    dict
        ``delta``, ``evaluation``, ``outcome``, and optionally ``history``.
    """
    prev_state = extract_eval_state(prev_metrics)
    curr_state = extract_eval_state(curr_metrics)

    delta = compute_state_delta(prev_state, curr_state)
    evaluation = evaluate_improvement(prev_state, curr_state)
    outcome = classify_outcome(prev_state, curr_state)

    result: Dict[str, Any] = {
        "delta": delta,
        "evaluation": evaluation,
        "outcome": outcome,
    }

    if history is not None:
        result["history"] = update_history(history, evaluation)

    return result
