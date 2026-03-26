"""v103.1.0 — Feedback control and closed-loop adaptation.

Provides:
- iterative feedback control loop (multi-step intervention)
- convergence detection (stable, cycle, max_steps)
- feedback-based candidate adjustment
- stability scoring

Extends the control layer from single-shot intervention
(state -> intervention -> new state) to closed-loop control
(state -> intervention -> feedback -> next intervention -> ...).

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

# Convergence thresholds.
STABILITY_DELTA_THRESHOLD = 0.01
ATTRACTOR_DELTA_THRESHOLD = 0.01

# Default maximum feedback steps.
DEFAULT_MAX_STEPS = 5

# Overshoot detection threshold (score decreased by this much).
OVERSHOOT_THRESHOLD = 0.02

# Minimum strength after reduction.
MIN_REDUCED_STRENGTH = 0.05

# Ineffective threshold (score change below this is ineffective).
INEFFECTIVE_THRESHOLD = 0.005


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
# Public API — Stability Score
# ---------------------------------------------------------------------------


def compute_stability_score(state: dict) -> float:
    """Compute a scalar stability score from a state vector.

    Combines stability, attractor weight, and transient weight into
    a single score in [0, 1].

    Parameters
    ----------
    state : dict
        State vector with ``"ternary"`` and ``"membership"`` sub-dicts.

    Returns
    -------
    float
        Stability score in [0, 1]. Higher is more stable.
    """
    ternary = state.get("ternary", {})
    membership = state.get("membership", {})

    # Map stability_state from {-1, 0, +1} to {0.0, 0.5, 1.0}.
    stability = (ternary.get("stability_state", 0) + 1) / 2.0

    # Attractor weight: strong + weak.
    attractor_weight = (
        membership.get("strong_attractor", 0.0)
        + membership.get("weak_attractor", 0.0)
    )

    # Transient weight (penalty).
    transient_weight = membership.get("transient", 0.0)

    score = _clamp(stability + attractor_weight - transient_weight)
    return _round(score)


# ---------------------------------------------------------------------------
# Public API — Feedback Step
# ---------------------------------------------------------------------------


def feedback_step(
    state: dict,
    objective: dict,
    candidates: List[Dict[str, Any]],
    runs: List[Dict[str, Any]],
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
) -> Tuple[dict, dict]:
    """Execute one feedback step: evaluate, intervene, return new state.

    Parameters
    ----------
    state : dict
        Current multistate dict (keyed by strategy name).
    objective : dict
        Control objective for scoring.
    candidates : list of dict
        Candidate interventions.
    runs : list of dict
        Original run data.
    multistate_result : dict, optional
        Precomputed multistate result for reuse.

    Returns
    -------
    tuple of (dict, dict)
        ``(new_state, chosen_intervention)`` where *new_state* is the
        updated multistate dict and *chosen_intervention* is the rule
        that was applied.
    """
    from qec.analysis.control_layer import (
        evaluate_objective,
        find_best_intervention,
        simulate_intervention,
    )

    if not candidates:
        return state, {}

    # Build a multistate_result wrapper if needed.
    if multistate_result is None:
        multistate_result = {"multistate": state}

    # Find the best intervention from candidates.
    best_result = find_best_intervention(
        runs,
        candidates,
        objective,
        multistate_result=multistate_result,
    )

    chosen = best_result.get("best_intervention", {})
    if not chosen:
        return state, {}

    # Apply the chosen intervention.
    sim = simulate_intervention(
        runs,
        [chosen],
        multistate_result=multistate_result,
    )

    new_state = sim.get("after", state)
    return new_state, chosen


# ---------------------------------------------------------------------------
# Public API — Convergence Detection
# ---------------------------------------------------------------------------


def detect_convergence(states: List[Dict[str, Any]]) -> dict:
    """Detect convergence in a sequence of states.

    Checks for:
    - Small change (stability and attractor weight deltas below threshold)
    - Cycle detection (repeated state patterns)

    Parameters
    ----------
    states : list of dict
        Sequence of multistate dicts (one per step). Each is keyed
        by strategy name with state vectors as values.

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

    # Check for small-change convergence between consecutive states.
    for i in range(1, len(states)):
        prev = states[i - 1]
        curr = states[i]

        if _states_close(prev, curr):
            return {"converged": True, "step": i, "type": "stable"}

    # Check for cycle detection: does current state match any earlier state?
    for i in range(2, len(states)):
        for j in range(0, i - 1):
            if _states_close(states[i], states[j]):
                return {"converged": True, "step": i, "type": "cycle"}

    return {"converged": False, "step": len(states) - 1, "type": "max_steps"}


def _states_close(state_a: dict, state_b: dict) -> bool:
    """Check if two multistate dicts are close enough to be considered equal."""
    all_keys = sorted(set(list(state_a.keys()) + list(state_b.keys())))

    for key in all_keys:
        sv_a = state_a.get(key, {})
        sv_b = state_b.get(key, {})

        mem_a = sv_a.get("membership", {})
        mem_b = sv_b.get("membership", {})
        ter_a = sv_a.get("ternary", {})
        ter_b = sv_b.get("ternary", {})

        # Check stability ternary state.
        stab_a = ter_a.get("stability_state", 0)
        stab_b = ter_b.get("stability_state", 0)
        if stab_a != stab_b:
            return False

        # Check attractor weight delta.
        attr_a = mem_a.get("strong_attractor", 0.0) + mem_a.get(
            "weak_attractor", 0.0
        )
        attr_b = mem_b.get("strong_attractor", 0.0) + mem_b.get(
            "weak_attractor", 0.0
        )
        if abs(attr_a - attr_b) >= ATTRACTOR_DELTA_THRESHOLD:
            return False

        # Check transient weight delta.
        trans_a = mem_a.get("transient", 0.0)
        trans_b = mem_b.get("transient", 0.0)
        if abs(trans_a - trans_b) >= STABILITY_DELTA_THRESHOLD:
            return False

    return True


# ---------------------------------------------------------------------------
# Public API — Feedback Adjustment
# ---------------------------------------------------------------------------


def adjust_candidates(
    candidates: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Adjust candidate interventions based on feedback history.

    Rules (deterministic, rule-based only):
    - If the last intervention caused a score decrease (overshoot),
      reduce strength of matching candidates by half.
    - Remove candidates whose action+target appeared in history
      but produced negligible improvement (ineffective).
    - Preserve deterministic ordering.

    Parameters
    ----------
    candidates : list of dict
        Current candidate interventions.
    history : list of dict
        Feedback history entries, each with ``"intervention"`` and
        ``"score"`` keys. Must be ordered chronologically.

    Returns
    -------
    list of dict
        Adjusted candidates (new list, inputs not mutated).
    """
    if not history or not candidates:
        return list(candidates)

    # Identify overshooting: score decreased in last step.
    overshoot_keys: set = set()
    ineffective_keys: set = set()

    if len(history) >= 2:
        prev_score = history[-2].get("score", 0.0)
        curr_score = history[-1].get("score", 0.0)

        if curr_score < prev_score - OVERSHOOT_THRESHOLD:
            # Last intervention overshot — mark its action+target.
            last_int = history[-1].get("intervention", {})
            key = (
                str(last_int.get("target", "")),
                str(last_int.get("action", "")),
            )
            overshoot_keys.add(key)

    # Identify ineffective interventions across all history.
    for i in range(1, len(history)):
        prev_score = history[i - 1].get("score", 0.0)
        curr_score = history[i].get("score", 0.0)
        delta = abs(curr_score - prev_score)

        if delta < INEFFECTIVE_THRESHOLD:
            int_entry = history[i].get("intervention", {})
            key = (
                str(int_entry.get("target", "")),
                str(int_entry.get("action", "")),
            )
            ineffective_keys.add(key)

    # Build adjusted candidates.
    adjusted: List[Dict[str, Any]] = []
    for candidate in candidates:
        key = (
            str(candidate.get("target", "")),
            str(candidate.get("action", "")),
        )

        # Skip ineffective candidates.
        if key in ineffective_keys:
            continue

        # Reduce strength for overshooting candidates.
        if key in overshoot_keys:
            new_strength = max(
                MIN_REDUCED_STRENGTH,
                float(candidate.get("strength", 0.0)) * 0.5,
            )
            adjusted.append({
                "target": candidate.get("target", ""),
                "action": candidate.get("action", ""),
                "strength": _round(new_strength),
            })
        else:
            adjusted.append(dict(candidate))

    return adjusted


# ---------------------------------------------------------------------------
# Public API — Iterative Control Loop
# ---------------------------------------------------------------------------


def run_feedback_control(
    runs: List[Dict[str, Any]],
    objective: Dict[str, Any],
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run an iterative feedback control loop.

    Pipeline: runs -> multistate -> loop(intervene, evaluate, adjust)
    until convergence or max_steps.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    objective : dict
        Control objective for scoring.
    max_steps : int
        Maximum number of feedback iterations.
    multistate_result : dict, optional
        Precomputed multistate result.

    Returns
    -------
    dict
        Contains:

        - ``"states"`` : list of dict — state at each step
        - ``"interventions"`` : list of dict — intervention at each step
        - ``"scores"`` : list of float — stability score at each step
        - ``"convergence"`` : dict — convergence detection result
        - ``"steps_taken"`` : int — number of steps executed
    """
    from qec.analysis.control_layer import (
        VALID_ACTIONS,
        evaluate_objective,
    )

    # Build initial multistate if not provided.
    if multistate_result is None:
        from qec.analysis.strategy_adapter import run_multistate_analysis

        multistate_result = run_multistate_analysis(runs)

    current_state = multistate_result.get("multistate", {})
    strategy_names = sorted(current_state.keys())

    # Generate initial candidates.
    candidates: List[Dict[str, Any]] = []
    actions = ("boost_stability", "reduce_escape", "force_transition")
    strengths = (0.3, 0.6, 0.9)
    for name in strategy_names:
        for action in actions:
            for strength in strengths:
                candidates.append({
                    "target": name,
                    "action": action,
                    "strength": strength,
                })

    # Compute initial score.
    initial_score = _compute_avg_score(current_state, objective)

    states: List[Dict[str, Any]] = [current_state]
    interventions: List[Dict[str, Any]] = []
    scores: List[float] = [initial_score]
    history: List[Dict[str, Any]] = []

    for step in range(max_steps):
        # Adjust candidates based on feedback history.
        if history:
            candidates = adjust_candidates(candidates, history)

        if not candidates:
            break

        # Build a multistate wrapper for the current state.
        ms_wrapper = {"multistate": current_state}

        # Execute one feedback step.
        new_state, chosen = feedback_step(
            current_state,
            objective,
            candidates,
            runs,
            multistate_result=ms_wrapper,
        )

        if not chosen:
            break

        # Compute score for the new state.
        new_score = _compute_avg_score(new_state, objective)

        states.append(new_state)
        interventions.append(chosen)
        scores.append(new_score)
        history.append({
            "intervention": chosen,
            "score": new_score,
        })

        current_state = new_state

        # Check for convergence.
        convergence = detect_convergence(states)
        if convergence.get("converged", False):
            return {
                "states": states,
                "interventions": interventions,
                "scores": scores,
                "convergence": convergence,
                "steps_taken": step + 1,
            }

    # Reached max steps without convergence.
    convergence = detect_convergence(states)
    if not convergence.get("converged", False):
        convergence = {
            "converged": False,
            "step": len(states) - 1,
            "type": "max_steps",
        }

    return {
        "states": states,
        "interventions": interventions,
        "scores": scores,
        "convergence": convergence,
        "steps_taken": len(interventions),
    }


def _compute_avg_score(
    multistate: Dict[str, Any],
    objective: Dict[str, Any],
) -> float:
    """Compute average objective score across all strategies."""
    from qec.analysis.control_layer import evaluate_objective

    if not multistate:
        return _round(0.0)

    total = 0.0
    count = 0
    for name in sorted(multistate.keys()):
        total += evaluate_objective(multistate[name], objective)
        count += 1

    if count == 0:
        return _round(0.0)
    return _round(total / count)


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_feedback_summary(result: Dict[str, Any]) -> str:
    """Format feedback control results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_feedback_control``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Feedback Control ===")

    interventions = result.get("interventions", [])
    scores = result.get("scores", [])
    convergence = result.get("convergence", {})

    for i, intervention in enumerate(interventions):
        step_num = i + 1
        action = intervention.get("action", "?")
        target = intervention.get("target", "?")
        strength = intervention.get("strength", 0.0)

        # Score after this step (index i+1 in scores, since scores[0]
        # is the initial score).
        score = scores[i + 1] if i + 1 < len(scores) else 0.0

        lines.append(f"Step {step_num}:")
        lines.append(f"  Intervention: {action}({target}) [strength={strength}]")
        lines.append(f"  Score: {score:.4f}")

    # Convergence info.
    conv_type = convergence.get("type", "max_steps")
    conv_step = convergence.get("step", 0)
    converged = convergence.get("converged", False)

    if converged:
        lines.append(f"Converged at step {conv_step} ({conv_type})")
    else:
        lines.append(f"Did not converge after {len(interventions)} steps")

    # Final state summary.
    states = result.get("states", [])
    if states:
        final_state = states[-1]
        lines.append("")
        lines.append("Final State:")
        for name in sorted(final_state.keys()):
            sv = final_state[name]
            mem = sv.get("membership", {})
            stab = sv.get("ternary", {}).get("stability_state", 0)
            attr_w = mem.get("strong_attractor", 0.0) + mem.get(
                "weak_attractor", 0.0
            )
            lines.append(
                f"  {name}: stability={stab} attractor={attr_w:.4f}"
            )

    return "\n".join(lines)


__all__ = [
    "ATTRACTOR_DELTA_THRESHOLD",
    "DEFAULT_MAX_STEPS",
    "INEFFECTIVE_THRESHOLD",
    "MIN_REDUCED_STRENGTH",
    "OVERSHOOT_THRESHOLD",
    "ROUND_PRECISION",
    "STABILITY_DELTA_THRESHOLD",
    "adjust_candidates",
    "compute_stability_score",
    "detect_convergence",
    "feedback_step",
    "format_feedback_summary",
    "run_feedback_control",
]
