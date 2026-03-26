"""v103.3.0 — Hierarchical control and policy-based routing.

Provides:
- control policy evaluation (local, global, hybrid modes)
- local vs global decision routing
- hybrid merge logic with conflict resolution
- hierarchical control loop combining v103.1 and v103.2
- hierarchical convergence detection
- built-in policy types (stability-first, sync-first, balanced)

Extends the feedback control (v103.1) and global coordination (v103.2)
into a layered control architecture with deterministic policy routing:

    local_control <-> global_control
            |
       policy_router
            |
       final intervention

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- rule-based only (no stochastic routing, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

# Default maximum hierarchical feedback steps.
DEFAULT_MAX_STEPS = 5

# Convergence thresholds.
SCORE_DELTA_THRESHOLD = 0.01
MODE_STABILITY_WINDOW = 3

# Built-in policy names.
POLICY_STABILITY_FIRST = "stability_first"
POLICY_SYNC_FIRST = "sync_first"
POLICY_BALANCED = "balanced"

# Default thresholds for built-in policies.
DEFAULT_INSTABILITY_THRESHOLD = 0.5
DEFAULT_SYNC_THRESHOLD = 0.5

# Conflict resolution priority (higher = stronger intervention).
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


def _compute_avg_stability(multistate: Dict[str, Any]) -> float:
    """Compute average stability score across all strategies."""
    if not multistate:
        return 0.0
    total = 0.0
    count = 0
    for name in sorted(multistate.keys()):
        sv = multistate[name]
        ternary = sv.get("ternary", {})
        stability = (ternary.get("stability_state", 0) + 1) / 2.0
        total += stability
        count += 1
    if count == 0:
        return 0.0
    return _round(total / count)


def _compute_avg_sync(coupled_summary: Dict[str, Any]) -> float:
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
# Public API — Control Policy Evaluation
# ---------------------------------------------------------------------------


def evaluate_control_policy(
    state: dict,
    global_state: dict,
    policy: dict,
) -> str:
    """Evaluate a control policy to determine routing mode.

    Examines local state metrics and global coordination state to
    decide whether control should be routed locally, globally, or
    through a hybrid merge.

    Parameters
    ----------
    state : dict
        Local multistate dict (keyed by strategy name, each with
        ``"ternary"`` and ``"membership"`` sub-dicts).
    global_state : dict
        Global coordination state with keys:

        - ``"avg_sync"`` : float — average synchronization ratio
        - ``"avg_stability"`` : float — average stability score
        - ``"coupled_summary"`` : dict — coupled dynamics summary

    policy : dict
        Control policy with keys:

        - ``"mode"`` : str — ``"local"``, ``"global"``, or ``"hybrid"``
        - ``"priority"`` : str — ``"stability"``, ``"synchronization"``,
          or ``"balanced"``
        - ``"thresholds"`` : dict — threshold values for switching

    Returns
    -------
    str
        One of ``"local"``, ``"global"``, or ``"hybrid"``.
    """
    mode = str(policy.get("mode", "hybrid"))

    # Fixed modes: return directly.
    if mode in ("local", "global"):
        return mode

    # Hybrid mode: use priority and thresholds to decide.
    priority = str(policy.get("priority", "balanced"))
    thresholds = policy.get("thresholds", {})

    instability_threshold = float(
        thresholds.get("instability", DEFAULT_INSTABILITY_THRESHOLD)
    )
    sync_threshold = float(
        thresholds.get("sync", DEFAULT_SYNC_THRESHOLD)
    )

    avg_stability = float(global_state.get("avg_stability", 0.5))
    avg_sync = float(global_state.get("avg_sync", 0.5))

    if priority == "stability":
        # Prefer local unless instability detected.
        if avg_stability < instability_threshold:
            return "global"
        return "local"

    elif priority == "synchronization":
        # Prefer global if sync below threshold.
        if avg_sync < sync_threshold:
            return "global"
        return "local"

    else:
        # Balanced: hybrid if both metrics are moderate,
        # global if either is critically low.
        if avg_stability < instability_threshold and avg_sync < sync_threshold:
            return "global"
        if avg_stability < instability_threshold or avg_sync < sync_threshold:
            return "hybrid"
        return "local"


# ---------------------------------------------------------------------------
# Public API — Decision Routing
# ---------------------------------------------------------------------------


def route_control(
    local_action: dict,
    global_action: dict,
    policy: dict,
) -> dict:
    """Route between local and global control actions.

    Parameters
    ----------
    local_action : dict
        Local intervention with ``"interventions"`` (list of dict)
        and ``"score"`` (float).
    global_action : dict
        Global intervention with ``"interventions"`` (list of dict)
        and ``"score"`` (float).
    policy : dict
        Control policy with ``"mode"`` key.

    Returns
    -------
    dict
        Routed action with ``"interventions"``, ``"score"``, and
        ``"mode"`` keys.
    """
    mode = str(policy.get("mode", "hybrid"))

    if mode == "local":
        return {
            "interventions": list(local_action.get("interventions", [])),
            "score": float(local_action.get("score", 0.0)),
            "mode": "local",
        }

    if mode == "global":
        return {
            "interventions": list(global_action.get("interventions", [])),
            "score": float(global_action.get("score", 0.0)),
            "mode": "global",
        }

    # Hybrid mode: merge interventions.
    merged = merge_interventions(
        local_action.get("interventions", []),
        global_action.get("interventions", []),
    )

    # Score is the max of local and global scores.
    local_score = float(local_action.get("score", 0.0))
    global_score = float(global_action.get("score", 0.0))
    score = _round(max(local_score, global_score))

    return {
        "interventions": merged,
        "score": score,
        "mode": "hybrid",
    }


# ---------------------------------------------------------------------------
# Public API — Hybrid Merge Logic
# ---------------------------------------------------------------------------


def merge_interventions(
    local: List[Dict[str, Any]],
    global_: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge local and global intervention sets.

    Rules:
    - Combine both sets.
    - Deduplicate by target.
    - If conflict (same target, different actions): use priority ordering.
    - If same target, same action: keep stronger intervention.

    Parameters
    ----------
    local : list of dict
        Local interventions.
    global_ : list of dict
        Global interventions.

    Returns
    -------
    list of dict
        Merged, deduplicated, conflict-resolved interventions.
        Sorted by target for deterministic ordering.
    """
    # Group all interventions by target.
    by_target: Dict[str, List[Dict[str, Any]]] = {}
    for intervention in local:
        target = str(intervention.get("target", ""))
        by_target.setdefault(target, []).append(dict(intervention))
    for intervention in global_:
        target = str(intervention.get("target", ""))
        by_target.setdefault(target, []).append(dict(intervention))

    # Resolve per target.
    merged: List[Dict[str, Any]] = []
    for target in sorted(by_target.keys()):
        group = by_target[target]
        if not group:
            continue

        # Pick the intervention with the highest priority action.
        # Among same-priority, pick highest strength.
        best = max(
            group,
            key=lambda x: (
                ACTION_PRIORITY.get(str(x.get("action", "")), -1),
                float(x.get("strength", 0.0)),
            ),
        )

        merged.append({
            "target": target,
            "action": str(best.get("action", "")),
            "strength": _round(float(best.get("strength", 0.0))),
        })

    return merged


# ---------------------------------------------------------------------------
# Public API — Hierarchical Convergence Detection
# ---------------------------------------------------------------------------


def detect_hierarchical_convergence(
    states: List[Dict[str, Any]],
    scores: List[float],
) -> dict:
    """Detect convergence in a hierarchical control sequence.

    Criteria:
    - Stable score (small delta between consecutive steps).
    - Stable policy selection (no oscillation between local/global).
    - No oscillation pattern detected.

    Parameters
    ----------
    states : list of dict
        Each entry has ``"mode"`` (str) and ``"multistate"`` (dict).
    scores : list of float
        Objective score at each step.

    Returns
    -------
    dict
        Contains:

        - ``"converged"`` : bool
        - ``"step"`` : int — step at which convergence detected
        - ``"type"`` : str — ``"stable"``, ``"mode_stable"``,
          ``"oscillation"``, or ``"max_steps"``
    """
    if len(scores) < 2:
        return {"converged": False, "step": 0, "type": "max_steps"}

    # Check for score stability.
    for i in range(1, len(scores)):
        if abs(scores[i] - scores[i - 1]) < SCORE_DELTA_THRESHOLD:
            # Also check mode stability if enough history.
            if _modes_stable(states, i):
                return {"converged": True, "step": i, "type": "stable"}

    # Check for oscillation (alternating modes).
    if len(states) >= 4:
        modes = [str(s.get("mode", "")) for s in states]
        for i in range(3, len(modes)):
            # A-B-A-B pattern.
            if (
                modes[i] == modes[i - 2]
                and modes[i - 1] == modes[i - 3]
                and modes[i] != modes[i - 1]
            ):
                return {"converged": True, "step": i, "type": "oscillation"}

    return {
        "converged": False,
        "step": len(scores) - 1,
        "type": "max_steps",
    }


def _modes_stable(states: List[Dict[str, Any]], up_to: int) -> bool:
    """Check if the mode has been stable in the recent window."""
    if up_to < MODE_STABILITY_WINDOW:
        # Not enough history; accept as stable.
        return True

    modes = [
        str(states[i].get("mode", ""))
        for i in range(max(0, up_to - MODE_STABILITY_WINDOW + 1), up_to + 1)
    ]
    return len(set(modes)) == 1


# ---------------------------------------------------------------------------
# Public API — Built-in Policies
# ---------------------------------------------------------------------------


def get_builtin_policy(name: str) -> dict:
    """Return a built-in control policy by name.

    Parameters
    ----------
    name : str
        One of ``"stability_first"``, ``"sync_first"``, or ``"balanced"``.

    Returns
    -------
    dict
        Policy dict with ``"mode"``, ``"priority"``, and ``"thresholds"``.

    Raises
    ------
    ValueError
        If *name* is not a recognized built-in policy.
    """
    if name == POLICY_STABILITY_FIRST:
        return {
            "mode": "hybrid",
            "priority": "stability",
            "thresholds": {
                "instability": DEFAULT_INSTABILITY_THRESHOLD,
                "sync": DEFAULT_SYNC_THRESHOLD,
            },
        }
    elif name == POLICY_SYNC_FIRST:
        return {
            "mode": "hybrid",
            "priority": "synchronization",
            "thresholds": {
                "instability": DEFAULT_INSTABILITY_THRESHOLD,
                "sync": DEFAULT_SYNC_THRESHOLD,
            },
        }
    elif name == POLICY_BALANCED:
        return {
            "mode": "hybrid",
            "priority": "balanced",
            "thresholds": {
                "instability": DEFAULT_INSTABILITY_THRESHOLD,
                "sync": DEFAULT_SYNC_THRESHOLD,
            },
        }
    else:
        raise ValueError(
            f"Unknown built-in policy: {name!r}. "
            f"Must be one of: {POLICY_STABILITY_FIRST!r}, "
            f"{POLICY_SYNC_FIRST!r}, {POLICY_BALANCED!r}"
        )


# ---------------------------------------------------------------------------
# Public API — Hierarchical Control Loop
# ---------------------------------------------------------------------------


def run_hierarchical_control(
    runs: List[Dict[str, Any]],
    objective: Dict[str, Any],
    policy: Dict[str, Any],
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a hierarchical control loop combining local and global feedback.

    Pipeline::

        runs -> multistate -> coupled_dynamics -> loop:
            local_feedback (v103.1)
            global_feedback (v103.2)
            evaluate_control_policy -> route decision
            merge or select interventions
            apply intervention
            evaluate
            detect convergence
            repeat

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    objective : dict
        Control objective. Used for both local and global scoring.
    policy : dict
        Control policy with ``"mode"``, ``"priority"``, ``"thresholds"``.
    max_steps : int
        Maximum number of hierarchical feedback iterations.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.

    Returns
    -------
    dict
        Contains:

        - ``"states"`` : list of dict — state at each step (with mode)
        - ``"local_actions"`` : list of dict — local actions at each step
        - ``"global_actions"`` : list of list — global actions at each step
        - ``"final_actions"`` : list of dict — routed actions at each step
        - ``"scores"`` : list of float — objective score at each step
        - ``"convergence"`` : dict — convergence detection result
        - ``"steps_taken"`` : int — number of steps executed
    """
    # Lazy imports to avoid circular dependencies.
    from qec.analysis.feedback_control import run_feedback_control
    from qec.analysis.global_control import (
        apply_global_intervention,
        evaluate_global_objective,
        run_global_feedback,
    )
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    # Build multistate and coupled dynamics if not provided.
    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)

    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs,
            multistate_result=multistate_result,
        )

    current_multistate = multistate_result.get("multistate", {})
    coupled_summary = coupled_result.get("coupled_summary", {})

    # Build local objective from global objective.
    local_objective = _build_local_objective(objective)

    # Run local feedback (v103.1) — one pass to get local recommendations.
    local_result = run_feedback_control(
        runs,
        local_objective,
        max_steps=1,
        multistate_result=multistate_result,
    )

    # Run global feedback (v103.2) — one pass to get global recommendations.
    global_result = run_global_feedback(
        runs,
        objective,
        max_steps=1,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )

    # Compute initial score.
    initial_score = evaluate_global_objective(
        current_multistate, coupled_result, objective,
    )

    # Initialize tracking.
    states: List[Dict[str, Any]] = [
        {"mode": "initial", "multistate": current_multistate},
    ]
    local_actions: List[Dict[str, Any]] = []
    global_actions: List[List[Dict[str, Any]]] = []
    final_actions: List[Dict[str, Any]] = []
    scores: List[float] = [initial_score]

    for step in range(max_steps):
        # Compute global state for policy evaluation.
        avg_stability = _compute_avg_stability(current_multistate)
        avg_sync = _compute_avg_sync(coupled_summary)

        global_state = {
            "avg_stability": avg_stability,
            "avg_sync": avg_sync,
            "coupled_summary": coupled_summary,
        }

        # Evaluate policy to determine routing mode.
        effective_mode = evaluate_control_policy(
            current_multistate, global_state, policy,
        )

        # Build local action from local feedback result.
        local_interventions = local_result.get("interventions", [])
        local_score = local_result.get("scores", [0.0])[-1] if local_result.get("scores") else 0.0
        local_action = {
            "interventions": local_interventions[-1:] if local_interventions else [],
            "score": local_score,
        }

        # Build global action from global feedback result.
        global_interventions = global_result.get("interventions", [])
        global_score = global_result.get("scores", [0.0])[-1] if global_result.get("scores") else 0.0
        global_action_list = global_interventions[-1] if global_interventions else []
        global_action = {
            "interventions": global_action_list,
            "score": global_score,
        }

        # Route control based on effective mode.
        routed_policy = dict(policy)
        routed_policy["mode"] = effective_mode
        routed = route_control(local_action, global_action, routed_policy)

        # Track actions.
        local_actions.append(local_action)
        global_actions.append(global_action_list if isinstance(global_action_list, list) else [global_action_list])
        final_actions.append(routed)

        # Apply the routed interventions.
        routed_interventions = routed.get("interventions", [])
        if routed_interventions:
            new_multistate = apply_global_intervention(
                current_multistate, routed_interventions,
            )
        else:
            new_multistate = current_multistate

        # Compute new score.
        new_score = evaluate_global_objective(
            new_multistate, coupled_result, objective,
        )

        states.append({
            "mode": routed.get("mode", "hybrid"),
            "multistate": new_multistate,
        })
        scores.append(new_score)

        current_multistate = new_multistate

        # Re-run local and global for next step using updated state.
        ms_wrapper = {"multistate": current_multistate}
        local_result = run_feedback_control(
            runs,
            local_objective,
            max_steps=1,
            multistate_result=ms_wrapper,
        )
        global_result = run_global_feedback(
            runs,
            objective,
            max_steps=1,
            multistate_result=ms_wrapper,
            coupled_result=coupled_result,
        )

        # Check for convergence.
        convergence = detect_hierarchical_convergence(states, scores)
        if convergence.get("converged", False):
            return {
                "states": states,
                "local_actions": local_actions,
                "global_actions": global_actions,
                "final_actions": final_actions,
                "scores": scores,
                "convergence": convergence,
                "steps_taken": step + 1,
            }

    # Reached max steps.
    convergence = detect_hierarchical_convergence(states, scores)
    if not convergence.get("converged", False):
        convergence = {
            "converged": False,
            "step": len(scores) - 1,
            "type": "max_steps",
        }

    return {
        "states": states,
        "local_actions": local_actions,
        "global_actions": global_actions,
        "final_actions": final_actions,
        "scores": scores,
        "convergence": convergence,
        "steps_taken": len(final_actions),
    }


def _build_local_objective(global_objective: Dict[str, Any]) -> Dict[str, Any]:
    """Build a local objective from global objective weights.

    Maps global weights to the local objective format used by
    ``feedback_control.run_feedback_control``.
    """
    # The local feedback uses {"maximize": ..., "minimize": ...} format.
    # Default: maximize stability, minimize escape.
    return {"maximize": "stability", "minimize": "escape"}


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_hierarchical_control_summary(result: Dict[str, Any]) -> str:
    """Format hierarchical control results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_hierarchical_control``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Hierarchical Control ===")

    local_actions = result.get("local_actions", [])
    global_actions = result.get("global_actions", [])
    final_actions = result.get("final_actions", [])
    scores = result.get("scores", [])
    states = result.get("states", [])

    for i in range(len(final_actions)):
        step_num = i + 1
        lines.append("")
        lines.append(f"Step {step_num}:")

        # Local actions.
        la = local_actions[i] if i < len(local_actions) else {}
        la_interventions = la.get("interventions", [])
        if la_interventions:
            for intervention in la_interventions:
                target = intervention.get("target", "?")
                action = intervention.get("action", "?")
                strength = intervention.get("strength", 0.0)
                lines.append(
                    f"  Local:  {target} {action}({strength})"
                )
        else:
            lines.append("  Local:  (none)")

        # Global actions.
        ga = global_actions[i] if i < len(global_actions) else []
        if ga:
            for intervention in ga:
                target = intervention.get("target", "?")
                action = intervention.get("action", "?")
                strength = intervention.get("strength", 0.0)
                lines.append(
                    f"  Global: {target} {action}({strength})"
                )
        else:
            lines.append("  Global: (none)")

        # Final (routed) actions.
        fa = final_actions[i] if i < len(final_actions) else {}
        fa_interventions = fa.get("interventions", [])
        if fa_interventions:
            parts = []
            for intervention in fa_interventions:
                target = intervention.get("target", "?")
                action = intervention.get("action", "?")
                strength = intervention.get("strength", 0.0)
                parts.append(f"{target} {action}({strength})")
            lines.append(f"  Final:  {', '.join(parts)}")
        else:
            lines.append("  Final:  (none)")

        # Mode.
        mode = fa.get("mode", "?")
        lines.append(f"  Mode: {mode}")

        # Check for mode switch.
        if i > 0:
            prev_mode = final_actions[i - 1].get("mode", "?") if i - 1 < len(final_actions) else "?"
            if mode != prev_mode:
                lines.append(f"  Mode switched -> {mode}")

    # Final score and convergence.
    lines.append("")
    if scores:
        lines.append(f"Final Score: {scores[-1]:.4f}")

    convergence = result.get("convergence", {})
    conv_type = convergence.get("type", "max_steps")
    converged = convergence.get("converged", False)

    if converged:
        lines.append(f"Converged: {conv_type}")
    else:
        lines.append(f"Did not converge after {len(final_actions)} steps")

    return "\n".join(lines)


__all__ = [
    "ACTION_PRIORITY",
    "DEFAULT_INSTABILITY_THRESHOLD",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SYNC_THRESHOLD",
    "MODE_STABILITY_WINDOW",
    "POLICY_BALANCED",
    "POLICY_STABILITY_FIRST",
    "POLICY_SYNC_FIRST",
    "ROUND_PRECISION",
    "SCORE_DELTA_THRESHOLD",
    "detect_hierarchical_convergence",
    "evaluate_control_policy",
    "format_hierarchical_control_summary",
    "get_builtin_policy",
    "merge_interventions",
    "route_control",
    "run_hierarchical_control",
]
