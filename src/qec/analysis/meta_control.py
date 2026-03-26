"""v103.5.0 — Meta-control and policy selection engine.

Provides:
- per-step policy evaluation across multiple policies
- deterministic policy selection (highest score wins)
- meta-control loop with policy switching
- policy switching detection (stable, oscillation, frequent)
- meta-convergence detection

Implements control over control: instead of fixing a single policy,
the meta-control layer dynamically selects the best policy at each
step based on current state and objective scores.

Pipeline::

    runs -> multistate -> coupled_dynamics -> loop:
        evaluate all policies (one-step hierarchical control each)
        select best policy
        apply selected intervention
        update state
        detect convergence
        repeat

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- rule-based only (no stochastic selection, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUND_PRECISION = 12

DEFAULT_META_MAX_STEPS = 5

# Convergence thresholds.
META_SCORE_DELTA = 0.01
META_POLICY_WINDOW = 3

# Switching detection thresholds.
SWITCHING_FREQUENT_RATIO = 0.5
SWITCHING_OSCILLATION_MIN = 4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Public API — Policy Evaluation per State
# ---------------------------------------------------------------------------


def evaluate_policies_step(
    state: Dict[str, Any],
    policies: list,
    objective: Dict[str, Any],
    runs: List[Dict[str, Any]],
    *,
    coupled_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate each policy for one step and return scores and actions.

    For each policy, runs a single-step hierarchical control pass
    (NOT a full loop) and computes the resulting objective score.

    Parameters
    ----------
    state : dict
        Current multistate dict (keyed by strategy name).
    policies : list of Policy
        Policies to evaluate.
    objective : dict
        Global objective weights.
    runs : list of dict
        Run data (each with ``"strategies"`` key).
    coupled_result : dict, optional
        Precomputed coupled dynamics result.

    Returns
    -------
    dict
        Mapping from policy name to ``{"score": float, "action": list}``.
        Sorted deterministically by policy name.
    """
    from qec.analysis.hierarchical_control import (
        evaluate_control_policy,
        merge_interventions,
        route_control,
        _compute_avg_stability,
        _compute_avg_sync,
    )
    from qec.analysis.feedback_control import run_feedback_control
    from qec.analysis.global_control import (
        evaluate_global_objective,
        run_global_feedback,
    )

    ms_wrapper = {"multistate": dict(state)}
    coupled_summary = (
        coupled_result.get("coupled_summary", {}) if coupled_result else {}
    )

    # Compute global metrics once (shared across policies).
    avg_stability = _compute_avg_stability(state)
    avg_sync = _compute_avg_sync(coupled_summary)
    global_state = {
        "avg_stability": avg_stability,
        "avg_sync": avg_sync,
        "coupled_summary": coupled_summary,
    }

    # Run local and global feedback once (shared across policies).
    local_objective = {"maximize": "stability", "minimize": "escape"}
    local_result = run_feedback_control(
        runs, local_objective, max_steps=1, multistate_result=ms_wrapper,
    )
    global_result = run_global_feedback(
        runs, objective, max_steps=1,
        multistate_result=ms_wrapper, coupled_result=coupled_result,
    )

    local_interventions = local_result.get("interventions", [])
    local_score = (
        local_result.get("scores", [0.0])[-1]
        if local_result.get("scores")
        else 0.0
    )
    local_action = {
        "interventions": local_interventions[-1:] if local_interventions else [],
        "score": local_score,
    }

    global_interventions = global_result.get("interventions", [])
    global_score = (
        global_result.get("scores", [0.0])[-1]
        if global_result.get("scores")
        else 0.0
    )
    global_action_list = (
        global_interventions[-1] if global_interventions else []
    )
    global_action = {
        "interventions": global_action_list,
        "score": global_score,
    }

    results: Dict[str, Dict[str, Any]] = {}

    for policy in sorted(policies, key=lambda p: p.name):
        policy_dict = policy.to_dict()

        # Evaluate policy to get routing mode.
        effective_mode = evaluate_control_policy(
            state, global_state, policy_dict,
        )

        # Route control based on effective mode.
        routed_policy = dict(policy_dict)
        routed_policy["mode"] = effective_mode
        routed = route_control(local_action, global_action, routed_policy)

        # Compute score after applying routed interventions.
        routed_interventions = routed.get("interventions", [])
        if routed_interventions:
            from qec.analysis.global_control import apply_global_intervention

            new_multistate = apply_global_intervention(
                state, routed_interventions,
            )
        else:
            new_multistate = state

        score = evaluate_global_objective(
            new_multistate, coupled_result or {}, objective,
        )

        results[policy.name] = {
            "score": _round(score),
            "action": list(routed.get("interventions", [])),
        }

    return results


# ---------------------------------------------------------------------------
# Public API — Policy Selection
# ---------------------------------------------------------------------------


def select_policy(results: Dict[str, Dict[str, Any]]) -> str:
    """Select the best policy from evaluation results.

    Selection rule (deterministic):
    - Highest score wins.
    - Tie: fewer actions (simpler intervention preferred).
    - Still tied: lexicographic policy name order.

    Parameters
    ----------
    results : dict
        Output of ``evaluate_policies_step``.

    Returns
    -------
    str
        Name of the selected policy.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("Cannot select from empty results")

    candidates = sorted(results.keys())

    best_name = candidates[0]
    best_score = results[best_name].get("score", 0.0)
    best_actions = len(results[best_name].get("action", []))

    for name in candidates[1:]:
        score = results[name].get("score", 0.0)
        n_actions = len(results[name].get("action", []))

        if (
            score > best_score
            or (score == best_score and n_actions < best_actions)
            or (
                score == best_score
                and n_actions == best_actions
                and name < best_name
            )
        ):
            best_name = name
            best_score = score
            best_actions = n_actions

    return best_name


# ---------------------------------------------------------------------------
# Public API — Meta-Control Loop
# ---------------------------------------------------------------------------


def run_meta_control(
    runs: List[Dict[str, Any]],
    policies: list,
    objective: Dict[str, Any],
    max_steps: int = DEFAULT_META_MAX_STEPS,
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the meta-control loop with dynamic policy selection.

    Pipeline::

        runs -> multistate -> coupled_dynamics -> loop:
            evaluate all policies (one-step each)
            select best policy
            apply selected intervention
            update state
            detect meta-convergence
            repeat

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    policies : list of Policy
        Policies to select among.
    objective : dict
        Global objective weights.
    max_steps : int
        Maximum meta-control iterations.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.

    Returns
    -------
    dict
        Contains:

        - ``"states"`` : list of dict — multistate at each step
        - ``"policies"`` : list of str — selected policy at each step
        - ``"actions"`` : list of list — interventions at each step
        - ``"scores"`` : list of float — objective score at each step
        - ``"evaluations"`` : list of dict — per-policy scores at each step
        - ``"convergence"`` : dict — convergence detection result
        - ``"switching"`` : dict — policy switching analysis
        - ``"steps_taken"`` : int
    """
    from qec.analysis.global_control import (
        apply_global_intervention,
        evaluate_global_objective,
    )
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)
    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs, multistate_result=multistate_result,
        )

    current_multistate = multistate_result.get("multistate", {})

    # Initial score.
    initial_score = evaluate_global_objective(
        current_multistate, coupled_result, objective,
    )

    # Tracking lists.
    states: List[Dict[str, Any]] = [dict(current_multistate)]
    selected_policies: List[str] = []
    all_actions: List[List[Dict[str, Any]]] = []
    scores: List[float] = [_round(initial_score)]
    evaluations: List[Dict[str, Dict[str, Any]]] = []

    for step in range(max_steps):
        # Evaluate all policies for current state.
        step_results = evaluate_policies_step(
            current_multistate,
            policies,
            objective,
            runs,
            coupled_result=coupled_result,
        )
        evaluations.append(step_results)

        # Select best policy.
        best_policy_name = select_policy(step_results)
        selected_policies.append(best_policy_name)

        # Get the selected policy's action.
        best_action = list(step_results[best_policy_name].get("action", []))
        all_actions.append(best_action)

        # Apply intervention.
        if best_action:
            new_multistate = apply_global_intervention(
                current_multistate, best_action,
            )
        else:
            new_multistate = current_multistate

        # Compute new score.
        new_score = evaluate_global_objective(
            new_multistate, coupled_result, objective,
        )

        states.append(dict(new_multistate))
        scores.append(_round(new_score))
        current_multistate = new_multistate

        # Check meta-convergence.
        convergence = detect_meta_convergence(
            states, selected_policies, scores,
        )
        if convergence.get("converged", False):
            switching = detect_policy_switching(selected_policies)
            return {
                "states": states,
                "policies": selected_policies,
                "actions": all_actions,
                "scores": scores,
                "evaluations": evaluations,
                "convergence": convergence,
                "switching": switching,
                "steps_taken": step + 1,
            }

    # Max steps reached.
    convergence = detect_meta_convergence(
        states, selected_policies, scores,
    )
    if not convergence.get("converged", False):
        convergence = {
            "converged": False,
            "step": len(scores) - 1,
            "type": "max_steps",
        }

    switching = detect_policy_switching(selected_policies)

    return {
        "states": states,
        "policies": selected_policies,
        "actions": all_actions,
        "scores": scores,
        "evaluations": evaluations,
        "convergence": convergence,
        "switching": switching,
        "steps_taken": len(selected_policies),
    }


# ---------------------------------------------------------------------------
# Public API — Policy Switching Detection
# ---------------------------------------------------------------------------


def detect_policy_switching(history: List[str]) -> Dict[str, Any]:
    """Detect policy switching patterns in the selection history.

    Detects:
    - **stable**: same policy selected throughout
    - **oscillation**: alternating A-B-A-B pattern
    - **frequent**: switching ratio exceeds threshold

    Parameters
    ----------
    history : list of str
        Policy names selected at each step.

    Returns
    -------
    dict
        Contains:

        - ``"pattern"`` : str — ``"stable"``, ``"oscillation"``,
          ``"frequent"``, or ``"none"``
        - ``"switches"`` : int — number of policy switches
        - ``"dominant_policy"`` : str — most frequently selected policy
        - ``"switch_ratio"`` : float — fraction of steps with a switch
    """
    if not history:
        return {
            "pattern": "none",
            "switches": 0,
            "dominant_policy": "",
            "switch_ratio": 0.0,
        }

    if len(history) == 1:
        return {
            "pattern": "stable",
            "switches": 0,
            "dominant_policy": history[0],
            "switch_ratio": 0.0,
        }

    # Count switches.
    switches = 0
    for i in range(1, len(history)):
        if history[i] != history[i - 1]:
            switches += 1

    switch_ratio = _round(switches / (len(history) - 1))

    # Find dominant policy (most frequent; tie: lexicographic).
    counts: Dict[str, int] = {}
    for name in history:
        counts[name] = counts.get(name, 0) + 1
    dominant = max(
        sorted(counts.keys()),
        key=lambda n: counts[n],
    )

    # Detect patterns.
    if switches == 0:
        pattern = "stable"
    elif _is_oscillation(history):
        pattern = "oscillation"
    elif switch_ratio >= SWITCHING_FREQUENT_RATIO:
        pattern = "frequent"
    else:
        pattern = "none"

    return {
        "pattern": pattern,
        "switches": switches,
        "dominant_policy": dominant,
        "switch_ratio": switch_ratio,
    }


def _is_oscillation(history: List[str]) -> bool:
    """Check for A-B-A-B oscillation pattern."""
    if len(history) < SWITCHING_OSCILLATION_MIN:
        return False

    # Check if the last 4 entries form an A-B-A-B pattern.
    for i in range(len(history) - 3):
        a, b, c, d = history[i], history[i + 1], history[i + 2], history[i + 3]
        if a == c and b == d and a != b:
            return True

    return False


# ---------------------------------------------------------------------------
# Public API — Meta-Convergence Detection
# ---------------------------------------------------------------------------


def detect_meta_convergence(
    states: List[Dict[str, Any]],
    policies: List[str],
    scores: List[float],
) -> Dict[str, Any]:
    """Detect convergence in the meta-control loop.

    Criteria:
    - **stable**: score delta below threshold AND policy selection stable
    - **score_stable**: score delta below threshold (policy may vary)
    - **oscillation**: alternating policy pattern detected
    - **max_steps**: none of the above

    Parameters
    ----------
    states : list of dict
        Multistate at each step.
    policies : list of str
        Selected policy at each step.
    scores : list of float
        Objective score at each step.

    Returns
    -------
    dict
        Contains:

        - ``"converged"`` : bool
        - ``"step"`` : int — step at which convergence detected
        - ``"type"`` : str — convergence type
    """
    if len(scores) < 2:
        return {"converged": False, "step": 0, "type": "max_steps"}

    # Check score stability.
    for i in range(1, len(scores)):
        if abs(scores[i] - scores[i - 1]) < META_SCORE_DELTA:
            # Also check if policy is stable in recent window.
            if _policy_stable(policies, i):
                return {"converged": True, "step": i, "type": "stable"}

    # Check for policy oscillation.
    if len(policies) >= SWITCHING_OSCILLATION_MIN:
        if _is_oscillation(policies):
            return {
                "converged": True,
                "step": len(policies) - 1,
                "type": "oscillation",
            }

    return {
        "converged": False,
        "step": len(scores) - 1,
        "type": "max_steps",
    }


def _policy_stable(policies: List[str], up_to: int) -> bool:
    """Check if policy selection is stable in the recent window."""
    if not policies:
        return True
    if up_to < META_POLICY_WINDOW:
        return True

    # Look at the last META_POLICY_WINDOW entries (0-indexed into policies).
    # policies[i] corresponds to step i+1 (step 0 has no policy).
    start = max(0, up_to - META_POLICY_WINDOW)
    # Ensure we don't exceed policies list bounds.
    end = min(up_to, len(policies))
    if start >= end:
        return True

    window = policies[start:end]
    return len(set(window)) == 1


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_meta_control_summary(result: Dict[str, Any]) -> str:
    """Format meta-control results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_meta_control``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Meta Control ===")

    selected_policies = result.get("policies", [])
    scores = result.get("scores", [])
    evaluations = result.get("evaluations", [])

    for i, policy_name in enumerate(selected_policies):
        step_num = i + 1
        step_score = scores[i + 1] if i + 1 < len(scores) else 0.0

        lines.append("")
        lines.append(f"Step {step_num}:")
        lines.append(f"  Selected Policy: {policy_name}")
        lines.append(f"  Score: {step_score:.2f}")

        # Show per-policy scores for this step.
        if i < len(evaluations):
            eval_step = evaluations[i]
            for pname in sorted(eval_step.keys()):
                if pname != policy_name:
                    pscore = eval_step[pname].get("score", 0.0)
                    lines.append(f"    {pname}: {pscore:.2f}")

        # Note policy switch.
        if i > 0 and selected_policies[i] != selected_policies[i - 1]:
            lines.append(
                f"  Policy switched: {selected_policies[i - 1]} -> {policy_name}"
            )

    # Final summary.
    lines.append("")
    if scores:
        lines.append(f"Final Score: {scores[-1]:.2f}")

    convergence = result.get("convergence", {})
    conv_type = convergence.get("type", "max_steps")
    converged = convergence.get("converged", False)

    if converged:
        lines.append(f"Converged: {conv_type}")
    else:
        lines.append(f"Did not converge after {len(selected_policies)} steps")

    switching = result.get("switching", {})
    pattern = switching.get("pattern", "none")
    if pattern != "none":
        dominant = switching.get("dominant_policy", "")
        if pattern == "stable":
            lines.append(f"Policy stabilized: {dominant}")
        elif pattern == "oscillation":
            lines.append(f"Policy oscillation detected (dominant: {dominant})")
        elif pattern == "frequent":
            lines.append(f"Frequent switching detected (dominant: {dominant})")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_META_MAX_STEPS",
    "META_POLICY_WINDOW",
    "META_SCORE_DELTA",
    "ROUND_PRECISION",
    "SWITCHING_FREQUENT_RATIO",
    "SWITCHING_OSCILLATION_MIN",
    "detect_meta_convergence",
    "detect_policy_switching",
    "evaluate_policies_step",
    "format_meta_control_summary",
    "run_meta_control",
    "select_policy",
]
