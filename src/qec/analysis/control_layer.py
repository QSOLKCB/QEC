"""v103.0.0 — Control layer and intervention modeling.

Provides:
- deterministic intervention application on state vectors
- system-level intervention simulation
- response metrics (before/after comparison)
- objective evaluation (stability, escape, sync targets)
- deterministic optimizer (exhaustive candidate evaluation)

Extends the multistate + coupled_dynamics pipeline from
"what happens naturally" to "what happens if we intervene?"

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- normalized where applicable

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

# Valid intervention actions.
VALID_ACTIONS = ("boost_stability", "reduce_escape", "force_transition")

# Default strength bounds.
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


def _normalize_membership(membership: Dict[str, float]) -> Dict[str, float]:
    """Normalize membership weights to sum to 1.0.

    Returns a new dict with sorted keys and normalized values.
    If all weights are zero, returns uniform zero weights.
    """
    total = sum(membership.values()) + 1e-12
    result: Dict[str, float] = {}
    for key in sorted(membership.keys()):
        result[key] = _round(membership[key] / total)
    return result


# ---------------------------------------------------------------------------
# Public API — Intervention Application
# ---------------------------------------------------------------------------


def apply_intervention(state_vector: dict, rule: dict) -> dict:
    """Apply a deterministic intervention rule to a state vector.

    Parameters
    ----------
    state_vector : dict
        Must contain ``"ternary"`` and ``"membership"`` sub-dicts.
        This is the output format of ``build_state_vector`` from
        ``multistate.py``.
    rule : dict
        Intervention rule with keys:

        - ``"target"`` : str — strategy name (informational)
        - ``"action"`` : str — one of ``"boost_stability"``,
          ``"reduce_escape"``, ``"force_transition"``
        - ``"strength"`` : float — intervention magnitude in [0, 1]

    Returns
    -------
    dict
        Modified state vector with the same structure. Membership
        weights are re-normalized after modification.

    Raises
    ------
    ValueError
        If action is not recognized or strength is out of bounds.
    """
    action = str(rule.get("action", ""))
    if action not in VALID_ACTIONS:
        raise ValueError(
            f"Unknown intervention action: {action!r}. "
            f"Must be one of {VALID_ACTIONS}"
        )

    strength = float(rule.get("strength", 0.0))
    if strength < MIN_STRENGTH or strength > MAX_STRENGTH:
        raise ValueError(
            f"Intervention strength {strength} out of bounds "
            f"[{MIN_STRENGTH}, {MAX_STRENGTH}]"
        )

    # Deep copy to avoid mutating input.
    ternary = dict(state_vector.get("ternary", {}))
    membership = dict(state_vector.get("membership", {}))

    if action == "boost_stability":
        # Increase attractor weights, shift stability state upward.
        for phase in sorted(membership.keys()):
            if phase in ("strong_attractor", "weak_attractor"):
                membership[phase] = membership.get(phase, 0.0) + strength
            elif phase == "transient":
                membership[phase] = _clamp(
                    membership.get(phase, 0.0) - strength * 0.5
                )

        # Nudge stability ternary state toward +1.
        current = ternary.get("stability_state", 0)
        if strength > 0.5 and current < 1:
            ternary["stability_state"] = min(current + 1, 1)

    elif action == "reduce_escape":
        # Decrease transient weight, redistribute to basin.
        transient_w = membership.get("transient", 0.0)
        reduction = _clamp(transient_w * strength, 0.0, transient_w)
        membership["transient"] = transient_w - reduction
        membership["basin"] = membership.get("basin", 0.0) + reduction

        # Nudge phase state toward stability.
        current_phase = ternary.get("phase_state", 0)
        if strength > 0.5 and current_phase < 0:
            ternary["phase_state"] = current_phase + 1

    elif action == "force_transition":
        # Shift weight from dominant phase to transient, forcing change.
        # Find the dominant phase.
        dominant_phase = ""
        dominant_weight = -1.0
        for phase in sorted(membership.keys()):
            w = membership.get(phase, 0.0)
            if w > dominant_weight:
                dominant_weight = w
                dominant_phase = phase

        if dominant_phase and dominant_weight > 0:
            shift = _clamp(dominant_weight * strength, 0.0, dominant_weight)
            membership[dominant_phase] = dominant_weight - shift
            membership["transient"] = membership.get("transient", 0.0) + shift

        # Nudge trend state toward change.
        current_trend = ternary.get("trend_state", 0)
        if strength > 0.5 and current_trend == 0:
            ternary["trend_state"] = -1

    # Re-normalize membership.
    membership = _normalize_membership(membership)

    return {
        "ternary": ternary,
        "membership": membership,
    }


# ---------------------------------------------------------------------------
# Public API — System-Level Simulation
# ---------------------------------------------------------------------------


def simulate_intervention(
    runs: List[Dict[str, Any]],
    interventions: List[Dict[str, Any]],
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Simulate interventions on the multi-state system.

    Pipeline: runs -> multistate -> apply interventions -> recompute metrics.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    interventions : list of dict
        Each intervention must contain ``"target"`` (strategy name),
        ``"action"``, and ``"strength"``.
    multistate_result : dict, optional
        Precomputed output of ``run_multistate_analysis``. If *None*,
        computes it from *runs*.

    Returns
    -------
    dict
        Contains:

        - ``"before"`` : dict — original multistate per strategy
        - ``"after"`` : dict — modified multistate per strategy
        - ``"interventions_applied"`` : list of dict — interventions used
    """
    # Lazy import to avoid circular dependencies.
    from qec.analysis.strategy_adapter import run_multistate_analysis

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)

    before = multistate_result.get("multistate", {})

    # Deep copy the before state for modification.
    after: Dict[str, Dict[str, Any]] = {}
    for name in sorted(before.keys()):
        sv = before[name]
        after[name] = {
            "ternary": dict(sv.get("ternary", {})),
            "membership": dict(sv.get("membership", {})),
        }

    # Apply each intervention to the targeted strategy.
    applied: List[Dict[str, Any]] = []
    for intervention in interventions:
        target = str(intervention.get("target", ""))
        if target not in after:
            continue

        after[target] = apply_intervention(after[target], intervention)
        applied.append({
            "target": target,
            "action": str(intervention.get("action", "")),
            "strength": float(intervention.get("strength", 0.0)),
        })

    return {
        "before": before,
        "after": after,
        "interventions_applied": applied,
    }


# ---------------------------------------------------------------------------
# Public API — Response Metrics
# ---------------------------------------------------------------------------


def evaluate_intervention(before: dict, after: dict) -> dict:
    """Compute response metrics comparing before/after state.

    Parameters
    ----------
    before : dict
        Multistate dict keyed by strategy name.
    after : dict
        Multistate dict keyed by strategy name (post-intervention).

    Returns
    -------
    dict
        Per-strategy response metrics:

        - ``"delta_stability"`` : float — change in stability state
        - ``"delta_phase"`` : float — change in phase state
        - ``"delta_trend"`` : float — change in trend state
        - ``"delta_attractor_weight"`` : float — change in attractor membership
        - ``"delta_transient_weight"`` : float — change in transient membership
    """
    all_names = sorted(set(list(before.keys()) + list(after.keys())))

    result: Dict[str, Dict[str, float]] = {}
    for name in all_names:
        bv = before.get(name, {})
        av = after.get(name, {})

        bt = bv.get("ternary", {})
        at = av.get("ternary", {})
        bm = bv.get("membership", {})
        am = av.get("membership", {})

        # Ternary deltas.
        d_stability = at.get("stability_state", 0) - bt.get("stability_state", 0)
        d_phase = at.get("phase_state", 0) - bt.get("phase_state", 0)
        d_trend = at.get("trend_state", 0) - bt.get("trend_state", 0)

        # Membership deltas.
        b_attractor = (
            bm.get("strong_attractor", 0.0) + bm.get("weak_attractor", 0.0)
        )
        a_attractor = (
            am.get("strong_attractor", 0.0) + am.get("weak_attractor", 0.0)
        )
        d_attractor = _round(a_attractor - b_attractor)

        d_transient = _round(
            am.get("transient", 0.0) - bm.get("transient", 0.0)
        )

        result[name] = {
            "delta_stability": _round(float(d_stability)),
            "delta_phase": _round(float(d_phase)),
            "delta_trend": _round(float(d_trend)),
            "delta_attractor_weight": d_attractor,
            "delta_transient_weight": d_transient,
        }

    return result


# ---------------------------------------------------------------------------
# Public API — Objective Evaluation
# ---------------------------------------------------------------------------


def evaluate_objective(state: dict, objective: dict) -> float:
    """Score a state vector against a control objective.

    Parameters
    ----------
    state : dict
        Multistate dict for a single strategy, with ``"ternary"``
        and ``"membership"`` sub-dicts.
    objective : dict
        Control objective. Supported keys:

        - ``"maximize"`` : str — metric to maximize
          (``"stability"``, ``"attractor_weight"``)
        - ``"minimize"`` : str — metric to minimize
          (``"escape"``, ``"transient_weight"``)
        - ``"target_sync"`` : float — target synchronization value

    Returns
    -------
    float
        Objective score in [0, 1]. Higher is better.
    """
    ternary = state.get("ternary", {})
    membership = state.get("membership", {})

    score = 0.0
    n_objectives = 0

    # Maximize objectives.
    maximize = objective.get("maximize", "")
    if maximize == "stability":
        # Map stability_state from {-1, 0, +1} to {0.0, 0.5, 1.0}.
        raw = ternary.get("stability_state", 0)
        score += (raw + 1) / 2.0
        n_objectives += 1
    elif maximize == "attractor_weight":
        w = membership.get("strong_attractor", 0.0) + membership.get(
            "weak_attractor", 0.0
        )
        score += _clamp(w)
        n_objectives += 1

    # Minimize objectives.
    minimize = objective.get("minimize", "")
    if minimize == "escape":
        # Less transient weight = better.
        w = membership.get("transient", 0.0)
        score += _clamp(1.0 - w)
        n_objectives += 1
    elif minimize == "transient_weight":
        w = membership.get("transient", 0.0)
        score += _clamp(1.0 - w)
        n_objectives += 1

    # Target sync objectives (evaluated against phase_state).
    target_sync = objective.get("target_sync", None)
    if target_sync is not None:
        target_sync = float(target_sync)
        # Use phase_state mapped to [0, 1] as proxy.
        raw_phase = ternary.get("phase_state", 0)
        phase_val = (raw_phase + 1) / 2.0
        # Score is 1 - |deviation| from target.
        deviation = abs(phase_val - target_sync)
        score += _clamp(1.0 - deviation)
        n_objectives += 1

    if n_objectives == 0:
        return _round(0.0)

    return _round(score / n_objectives)


# ---------------------------------------------------------------------------
# Public API — Deterministic Optimizer
# ---------------------------------------------------------------------------


def find_best_intervention(
    runs: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    objective: Optional[Dict[str, Any]] = None,
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find the best intervention from a candidate set.

    Evaluates each candidate intervention deterministically and
    returns the one with the highest objective score.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    candidates : list of dict
        Each candidate is an intervention rule with ``"target"``,
        ``"action"``, and ``"strength"``.
    objective : dict, optional
        Control objective for scoring. Defaults to
        ``{"maximize": "stability", "minimize": "escape"}``.
    multistate_result : dict, optional
        Precomputed multistate. Computed from *runs* if *None*.

    Returns
    -------
    dict
        Contains:

        - ``"best_intervention"`` : dict — the winning candidate
        - ``"best_score"`` : float — its objective score
        - ``"all_scores"`` : list of dict — all candidates with scores
    """
    from qec.analysis.strategy_adapter import run_multistate_analysis

    if objective is None:
        objective = {"maximize": "stability", "minimize": "escape"}

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)

    all_scores: List[Dict[str, Any]] = []
    best_score = -1.0
    best_intervention: Dict[str, Any] = {}

    for candidate in candidates:
        # Simulate this single intervention.
        sim = simulate_intervention(
            runs,
            [candidate],
            multistate_result=multistate_result,
        )

        after = sim.get("after", {})
        target = str(candidate.get("target", ""))

        # Score the target strategy's post-intervention state.
        if target in after:
            score = evaluate_objective(after[target], objective)
        else:
            score = 0.0

        entry = {
            "intervention": {
                "target": target,
                "action": str(candidate.get("action", "")),
                "strength": float(candidate.get("strength", 0.0)),
            },
            "score": score,
        }
        all_scores.append(entry)

        if score > best_score:
            best_score = score
            best_intervention = entry["intervention"]

    return {
        "best_intervention": best_intervention,
        "best_score": _round(best_score),
        "all_scores": all_scores,
    }


__all__ = [
    "MAX_STRENGTH",
    "MIN_STRENGTH",
    "ROUND_PRECISION",
    "VALID_ACTIONS",
    "apply_intervention",
    "evaluate_intervention",
    "evaluate_objective",
    "find_best_intervention",
    "simulate_intervention",
]
