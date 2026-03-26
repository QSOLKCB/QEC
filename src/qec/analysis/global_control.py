"""v103.2.0 — Multi-strategy feedback and global control.

Provides:
- global objective evaluation across all strategies
- multi-strategy intervention set generation
- joint intervention application with normalization
- global feedback loop with coordinated interventions
- conflict resolution between competing interventions
- global convergence detection

Extends the single-strategy feedback loop (v103.1) to system-level
coordination: multiple strategies share a global objective, interventions
are coordinated across strategies, and conflicts are resolved
deterministically.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- rule-based only (no stochastic search, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

# Default objective weights.
DEFAULT_STABILITY_WEIGHT = 0.3
DEFAULT_ATTRACTOR_WEIGHT = 0.3
DEFAULT_TRANSIENT_WEIGHT = 0.2
DEFAULT_SYNC_WEIGHT = 0.2

# Convergence thresholds.
GLOBAL_SCORE_DELTA_THRESHOLD = 0.01
SYNC_DELTA_THRESHOLD = 0.01

# Default maximum global feedback steps.
DEFAULT_MAX_STEPS = 5

# Conflict resolution priority (higher index = higher priority).
ACTION_PRIORITY = {
    "boost_stability": 2,
    "reduce_escape": 1,
    "force_transition": 0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Public API — Global Objective Function
# ---------------------------------------------------------------------------


def evaluate_global_objective(
    multistate: Dict[str, Dict[str, Any]],
    coupled: Dict[str, Any],
    objective: Dict[str, Any],
) -> float:
    """Evaluate a global objective function across all strategies.

    Combines average stability, average attractor weight, inverse
    transient weight, and synchronization into a single score.

    Parameters
    ----------
    multistate : dict
        Keyed by strategy name.  Each value is a state vector with
        ``"ternary"`` and ``"membership"`` sub-dicts.
    coupled : dict
        Output of ``run_coupled_dynamics_analysis`` or
        ``build_coupled_summary``.  Must contain ``"coupled_summary"``
        (or be the summary dict itself).
    objective : dict
        Weights for combining metrics.  Supported keys:

        - ``"w_stability"`` : float — weight for average stability
        - ``"w_attractor"`` : float — weight for average attractor
        - ``"w_transient"`` : float — weight for inverse transient
        - ``"w_sync"`` : float — weight for average synchronization

        Missing keys default to module-level constants.

    Returns
    -------
    float
        Global objective score in [0, 1].  Higher is better.
    """
    w1 = float(objective.get("w_stability", DEFAULT_STABILITY_WEIGHT))
    w2 = float(objective.get("w_attractor", DEFAULT_ATTRACTOR_WEIGHT))
    w3 = float(objective.get("w_transient", DEFAULT_TRANSIENT_WEIGHT))
    w4 = float(objective.get("w_sync", DEFAULT_SYNC_WEIGHT))

    # --- Average stability across strategies ---
    total_stability = 0.0
    total_attractor = 0.0
    total_transient = 0.0
    count = 0

    for name in sorted(multistate.keys()):
        sv = multistate[name]
        ternary = sv.get("ternary", {})
        membership = sv.get("membership", {})

        # Map stability_state {-1, 0, +1} -> {0.0, 0.5, 1.0}.
        stability = (ternary.get("stability_state", 0) + 1) / 2.0
        total_stability += stability

        attractor = (
            membership.get("strong_attractor", 0.0)
            + membership.get("weak_attractor", 0.0)
        )
        total_attractor += attractor

        total_transient += membership.get("transient", 0.0)
        count += 1

    if count == 0:
        return _round(0.0)

    avg_stability = total_stability / count
    avg_attractor = total_attractor / count
    avg_transient = total_transient / count

    # --- Average synchronization from coupled dynamics ---
    coupled_summary = coupled.get("coupled_summary", coupled)
    total_sync = 0.0
    sync_count = 0
    for pair_key in sorted(coupled_summary.keys(), key=str):
        info = coupled_summary[pair_key]
        if isinstance(info, dict) and "sync_ratio" in info:
            total_sync += float(info["sync_ratio"])
            sync_count += 1

    avg_sync = total_sync / sync_count if sync_count > 0 else 0.0

    # --- Weighted combination ---
    score = (
        w1 * avg_stability
        + w2 * avg_attractor
        - w3 * avg_transient
        + w4 * avg_sync
    )

    return _round(_clamp(score))


# ---------------------------------------------------------------------------
# Public API — Multi-Strategy Intervention Set
# ---------------------------------------------------------------------------


def generate_global_candidates(
    strategy_names: List[str],
    actions: Optional[List[str]] = None,
    strengths: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Generate a set of simultaneous multi-strategy interventions.

    Each candidate is a list of per-strategy interventions representing
    a coordinated joint intervention across all strategies.

    Parameters
    ----------
    strategy_names : list of str
        Names of strategies to target.
    actions : list of str, optional
        Valid intervention actions.  Defaults to the canonical set.
    strengths : list of float, optional
        Strength values to enumerate.  Defaults to ``[0.3, 0.6, 0.9]``.

    Returns
    -------
    list of dict
        Each dict has ``"target"``, ``"action"``, ``"strength"`` keys.
        The list represents all individual intervention atoms that can
        be composed into joint interventions.
    """
    if actions is None:
        actions = ["boost_stability", "reduce_escape", "force_transition"]
    if strengths is None:
        strengths = [0.3, 0.6, 0.9]

    candidates: List[Dict[str, Any]] = []
    for name in sorted(strategy_names):
        for action in sorted(actions):
            for strength in sorted(strengths):
                candidates.append({
                    "target": name,
                    "action": action,
                    "strength": _round(float(strength)),
                })

    return candidates


# ---------------------------------------------------------------------------
# Public API — Apply Joint Interventions
# ---------------------------------------------------------------------------


def apply_global_intervention(
    state: Dict[str, Dict[str, Any]],
    interventions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Apply a set of interventions to a multistate dict.

    Interventions are applied sequentially in deterministic order
    (sorted by target, then action, then strength).  After each
    intervention, the affected strategy's membership is re-normalized.

    Parameters
    ----------
    state : dict
        Multistate dict keyed by strategy name.
    interventions : list of dict
        Each must have ``"target"``, ``"action"``, ``"strength"``.

    Returns
    -------
    dict
        New multistate dict with interventions applied.
        Original *state* is not mutated.
    """
    from qec.analysis.control_layer import apply_intervention

    # Deep copy to avoid mutating input.
    result: Dict[str, Dict[str, Any]] = {}
    for name in sorted(state.keys()):
        sv = state[name]
        result[name] = {
            "ternary": dict(sv.get("ternary", {})),
            "membership": dict(sv.get("membership", {})),
        }

    # Sort interventions for deterministic ordering.
    sorted_interventions = sorted(
        interventions,
        key=lambda x: (
            str(x.get("target", "")),
            str(x.get("action", "")),
            float(x.get("strength", 0.0)),
        ),
    )

    for intervention in sorted_interventions:
        target = str(intervention.get("target", ""))
        if target not in result:
            continue

        result[target] = apply_intervention(result[target], intervention)

    return result


# ---------------------------------------------------------------------------
# Public API — Conflict Resolution
# ---------------------------------------------------------------------------


def resolve_conflicts(
    interventions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve conflicts between competing interventions.

    Rules (deterministic):
    - Same target + multiple actions: keep highest priority action.
      Priority: boost_stability > reduce_escape > force_transition.
    - Same target + same action + multiple strengths: keep highest
      strength.

    Parameters
    ----------
    interventions : list of dict
        Each must have ``"target"``, ``"action"``, ``"strength"``.

    Returns
    -------
    list of dict
        Conflict-free interventions (new list, inputs not mutated).
        Sorted by target name for deterministic ordering.
    """
    # Group by target.
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for intervention in interventions:
        target = str(intervention.get("target", ""))
        by_target.setdefault(target, []).append(intervention)

    resolved: List[Dict[str, Any]] = []
    for target in sorted(by_target.keys()):
        group = by_target[target]

        # Pick the intervention with the highest priority action.
        # Among same-priority, pick highest strength.
        best = max(
            group,
            key=lambda x: (
                ACTION_PRIORITY.get(str(x.get("action", "")), -1),
                float(x.get("strength", 0.0)),
            ),
        )

        resolved.append({
            "target": target,
            "action": str(best.get("action", "")),
            "strength": _round(float(best.get("strength", 0.0))),
        })

    return resolved


# ---------------------------------------------------------------------------
# Public API — Global Convergence Detection
# ---------------------------------------------------------------------------


def detect_global_convergence(
    states: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Detect convergence in a sequence of global states.

    Criteria:
    - Small change in global score between consecutive steps.
    - Cycle detection (repeated state pattern).

    Parameters
    ----------
    states : list of dict
        Each entry is a dict with ``"score"`` (float) and
        ``"sync"`` (float) keys from the global feedback loop.

    Returns
    -------
    dict
        Contains:

        - ``"converged"`` : bool
        - ``"step"`` : int — step at which convergence detected
        - ``"type"`` : str — ``"stable"``, ``"cycle"``, or ``"max_steps"``
    """
    if len(states) < 2:
        return {"converged": False, "step": 0, "type": "max_steps"}

    # Check for stable convergence (small score delta).
    for i in range(1, len(states)):
        prev_score = float(states[i - 1].get("score", 0.0))
        curr_score = float(states[i].get("score", 0.0))
        prev_sync = float(states[i - 1].get("sync", 0.0))
        curr_sync = float(states[i].get("sync", 0.0))

        score_delta = abs(curr_score - prev_score)
        sync_delta = abs(curr_sync - prev_sync)

        if (
            score_delta < GLOBAL_SCORE_DELTA_THRESHOLD
            and sync_delta < SYNC_DELTA_THRESHOLD
        ):
            return {"converged": True, "step": i, "type": "stable"}

    # Check for cycle detection.
    for i in range(2, len(states)):
        for j in range(0, i - 1):
            score_i = float(states[i].get("score", 0.0))
            score_j = float(states[j].get("score", 0.0))
            sync_i = float(states[i].get("sync", 0.0))
            sync_j = float(states[j].get("sync", 0.0))

            if (
                abs(score_i - score_j) < GLOBAL_SCORE_DELTA_THRESHOLD
                and abs(sync_i - sync_j) < SYNC_DELTA_THRESHOLD
            ):
                return {"converged": True, "step": i, "type": "cycle"}

    return {"converged": False, "step": len(states) - 1, "type": "max_steps"}


# ---------------------------------------------------------------------------
# Public API — Global Feedback Loop
# ---------------------------------------------------------------------------


def run_global_feedback(
    runs: List[Dict[str, Any]],
    objective: Dict[str, Any],
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a global multi-strategy feedback loop.

    Pipeline::

        runs -> multistate -> coupled_dynamics -> loop:
            generate candidates per strategy
            build joint interventions (all combinations of best-per-target)
            resolve conflicts
            evaluate global objective for each joint intervention
            select best joint intervention
            apply
            detect convergence
            repeat

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    objective : dict
        Global objective weights (w_stability, w_attractor, w_transient,
        w_sync).
    max_steps : int
        Maximum number of feedback iterations.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.

    Returns
    -------
    dict
        Contains:

        - ``"states"`` : list of dict — multistate at each step
        - ``"interventions"`` : list of list — joint interventions at each step
        - ``"scores"`` : list of float — global score at each step
        - ``"convergence"`` : dict — convergence detection result
        - ``"steps_taken"`` : int — number of steps executed
    """
    # Lazy imports to avoid circular dependencies.
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)

    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs,
            multistate_result=multistate_result,
        )

    current_multistate = multistate_result.get("multistate", {})
    strategy_names = sorted(current_multistate.keys())
    coupled_summary = coupled_result.get("coupled_summary", {})

    # Generate candidate atoms.
    candidate_atoms = generate_global_candidates(strategy_names)

    # Compute initial global score.
    initial_score = evaluate_global_objective(
        current_multistate, coupled_result, objective,
    )

    # Compute initial sync for convergence tracking.
    initial_sync = _compute_avg_sync(coupled_summary)

    states: List[Dict[str, Any]] = [current_multistate]
    interventions_history: List[List[Dict[str, Any]]] = []
    scores: List[float] = [initial_score]
    convergence_states: List[Dict[str, Any]] = [
        {"score": initial_score, "sync": initial_sync},
    ]

    for step in range(max_steps):
        # Find the best intervention for each strategy independently.
        best_per_target = _find_best_per_target(
            current_multistate, candidate_atoms, coupled_result, objective,
        )

        if not best_per_target:
            break

        # Resolve conflicts among the selected interventions.
        resolved = resolve_conflicts(best_per_target)

        if not resolved:
            break

        # Apply the joint intervention.
        new_multistate = apply_global_intervention(current_multistate, resolved)

        # Compute new global score.
        new_score = evaluate_global_objective(
            new_multistate, coupled_result, objective,
        )

        # Track state.
        new_sync = _compute_avg_sync(coupled_summary)
        states.append(new_multistate)
        interventions_history.append(resolved)
        scores.append(new_score)
        convergence_states.append({"score": new_score, "sync": new_sync})

        current_multistate = new_multistate

        # Check for convergence.
        convergence = detect_global_convergence(convergence_states)
        if convergence.get("converged", False):
            return {
                "states": states,
                "interventions": interventions_history,
                "scores": scores,
                "convergence": convergence,
                "steps_taken": step + 1,
            }

    # Reached max steps.
    convergence = detect_global_convergence(convergence_states)
    if not convergence.get("converged", False):
        convergence = {
            "converged": False,
            "step": len(states) - 1,
            "type": "max_steps",
        }

    return {
        "states": states,
        "interventions": interventions_history,
        "scores": scores,
        "convergence": convergence,
        "steps_taken": len(interventions_history),
    }


def _find_best_per_target(
    multistate: Dict[str, Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    coupled_result: Dict[str, Any],
    objective: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Find the best intervention for each target strategy.

    Evaluates each candidate by simulating it on the multistate and
    scoring with the global objective.  Returns the best candidate
    per target.
    """
    # Group candidates by target.
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        target = str(candidate.get("target", ""))
        by_target.setdefault(target, []).append(candidate)

    best_per_target: List[Dict[str, Any]] = []

    for target in sorted(by_target.keys()):
        group = by_target[target]
        best_score = -1.0
        best_candidate: Optional[Dict[str, Any]] = None

        for candidate in group:
            # Simulate applying just this one intervention.
            trial_state = apply_global_intervention(multistate, [candidate])
            trial_score = evaluate_global_objective(
                trial_state, coupled_result, objective,
            )

            if trial_score > best_score:
                best_score = trial_score
                best_candidate = candidate

        if best_candidate is not None:
            best_per_target.append(dict(best_candidate))

    return best_per_target


def _compute_avg_sync(coupled_summary: Dict) -> float:
    """Compute average sync ratio from coupled summary."""
    total = 0.0
    count = 0
    for pair_key in sorted(coupled_summary.keys(), key=str):
        info = coupled_summary[pair_key]
        if isinstance(info, dict) and "sync_ratio" in info:
            total += float(info["sync_ratio"])
            count += 1
    if count == 0:
        return _round(0.0)
    return _round(total / count)


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_global_control_summary(result: Dict[str, Any]) -> str:
    """Format global control results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_global_feedback``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Global Control ===")

    interventions = result.get("interventions", [])
    scores = result.get("scores", [])
    convergence = result.get("convergence", {})

    for i, step_interventions in enumerate(interventions):
        step_num = i + 1
        score = scores[i + 1] if i + 1 < len(scores) else 0.0

        lines.append(f"Step {step_num}:")
        lines.append("  Apply:")
        for intervention in step_interventions:
            target = intervention.get("target", "?")
            action = intervention.get("action", "?")
            strength = intervention.get("strength", 0.0)
            lines.append(f"    {target}: {action}({strength})")
        lines.append(f"  Score: {score:.4f}")

    # Convergence info.
    conv_type = convergence.get("type", "max_steps")
    converged = convergence.get("converged", False)

    if converged:
        lines.append(f"Converged: {conv_type}")
    else:
        lines.append(f"Did not converge after {len(interventions)} steps")

    # Final score.
    if scores:
        lines.append(f"Final Score: {scores[-1]:.4f}")

    return "\n".join(lines)


__all__ = [
    "ACTION_PRIORITY",
    "DEFAULT_ATTRACTOR_WEIGHT",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_STABILITY_WEIGHT",
    "DEFAULT_SYNC_WEIGHT",
    "DEFAULT_TRANSIENT_WEIGHT",
    "GLOBAL_SCORE_DELTA_THRESHOLD",
    "ROUND_PRECISION",
    "SYNC_DELTA_THRESHOLD",
    "apply_global_intervention",
    "detect_global_convergence",
    "evaluate_global_objective",
    "format_global_control_summary",
    "generate_global_candidates",
    "resolve_conflicts",
    "run_global_feedback",
]
