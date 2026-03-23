"""Correction hysteresis and friction analysis (v96.3.0).

Upgrades from:
    correction -> evaluation -> invariant theory
to:
    correction -> switching dynamics -> hysteresis -> friction (dissipation)

Detects:
  - oscillatory correction behavior
  - mode switching instability
  - invariant conflict churn
  - energy-like dissipation during correction

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No probabilistic scoring.
"""

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------


def extract_correction_traces(
    data: Any,
) -> List[Dict[str, Any]]:
    """Extract correction traces from alignment/application data.

    Accepts a dict with 'applications' key (invariant_application output).

    Returns list of trace dicts:
        {
          "dfa_type": str,
          "n": Optional[int],
          "states_before": [...],
          "states_after": [...],
          "modes": [...],
          "projection_distances": [...],
          "stability_efficiency": float,
          "compression_efficiency": float,
        }
    """
    if not isinstance(data, dict):
        return []

    applications = data.get("applications", [])
    if not applications:
        return []

    # Group applications by (dfa_type, n) to build per-system traces.
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for app in applications:
        dfa_type = app.get("dfa_type", "")
        n = app.get("n")
        key = (dfa_type, n)
        groups.setdefault(key, []).append(app)

    traces: List[Dict[str, Any]] = []
    for key in sorted(groups.keys(), key=lambda k: (k[0], str(k[1]))):
        apps = groups[key]
        dfa_type, n = key

        states_before = []
        states_after = []
        modes = []
        projection_distances = []
        stab_effs = []
        comp_effs = []

        for app in apps:
            before_mode = app.get("before_mode", "none")
            after_mode = app.get("after_mode", "none")
            states_before.append(before_mode)
            states_after.append(after_mode)
            modes.append(after_mode)

            # Compute projection distance from metrics delta.
            before_m = app.get("before_metrics", {})
            after_m = app.get("after_metrics", {})
            b_stab = before_m.get("stability_efficiency", 0.0)
            a_stab = after_m.get("stability_efficiency", 0.0)
            b_comp = before_m.get("compression_efficiency", 0.0)
            a_comp = after_m.get("compression_efficiency", 0.0)

            dist = abs(a_stab - b_stab) + abs(a_comp - b_comp)
            projection_distances.append(dist)
            stab_effs.append(a_stab)
            comp_effs.append(a_comp)

        avg_stab = sum(stab_effs) / len(stab_effs) if stab_effs else 0.0
        avg_comp = sum(comp_effs) / len(comp_effs) if comp_effs else 0.0

        traces.append({
            "dfa_type": dfa_type,
            "n": n,
            "states_before": list(states_before),
            "states_after": list(states_after),
            "modes": list(modes),
            "projection_distances": list(projection_distances),
            "stability_efficiency": avg_stab,
            "compression_efficiency": avg_comp,
        })

    return traces


# ---------------------------------------------------------------------------
# PART 2 — HYSTERESIS DETECTION
# ---------------------------------------------------------------------------


def detect_oscillation(
    states_before: List[str],
    states_after: List[str],
) -> Dict[str, Any]:
    """Detect oscillation loops in correction traces.

    Counts transitions where state A -> state B -> state A (the system
    transitions to B then returns to A within the sequence).

    Returns:
        {"oscillation_count": int, "oscillation_ratio": float}
    """
    if not states_before or not states_after:
        return {"oscillation_count": 0, "oscillation_ratio": 0.0}

    # Build transition sequence: list of (before, after) pairs.
    n = min(len(states_before), len(states_after))
    transitions = [(states_before[i], states_after[i]) for i in range(n)]

    oscillation_count = 0
    for i in range(len(transitions) - 1):
        a, b = transitions[i]
        c, d = transitions[i + 1]
        # A->B then B->A pattern (or C->D reverses A->B).
        if a == d and b == c:
            oscillation_count += 1

    max_possible = max(len(transitions) - 1, 1)
    ratio = oscillation_count / max_possible

    return {
        "oscillation_count": oscillation_count,
        "oscillation_ratio": ratio,
    }


def detect_hysteresis(
    states_before: List[str],
    states_after: List[str],
) -> Dict[str, Any]:
    """Detect history dependence in correction traces.

    Same input state leading to different outputs depending on prior path.
    Counts duplicate state_before values that map to different state_after.

    Returns:
        {"hysteresis_events": int, "hysteresis_ratio": float}
    """
    if not states_before or not states_after:
        return {"hysteresis_events": 0, "hysteresis_ratio": 0.0}

    n = min(len(states_before), len(states_after))

    # Map each state_before to the set of state_after values seen.
    state_map: Dict[str, set] = {}
    for i in range(n):
        sb = states_before[i]
        sa = states_after[i]
        state_map.setdefault(sb, set()).add(sa)

    # Hysteresis events: states_before that map to >1 distinct state_after.
    hysteresis_events = sum(
        1 for outputs in state_map.values() if len(outputs) > 1
    )

    total_inputs = max(len(state_map), 1)
    ratio = hysteresis_events / total_inputs

    return {
        "hysteresis_events": hysteresis_events,
        "hysteresis_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# PART 3 — MODE SWITCH FRICTION
# ---------------------------------------------------------------------------


def compute_mode_switches(
    modes: List[str],
) -> Dict[str, Any]:
    """Count mode switches in a correction trace.

    Returns:
        {"switch_count": int, "switch_ratio": float}
    """
    if len(modes) <= 1:
        return {"switch_count": 0, "switch_ratio": 0.0}

    switch_count = sum(
        1 for i in range(len(modes) - 1) if modes[i] != modes[i + 1]
    )
    max_possible = max(len(modes) - 1, 1)
    ratio = switch_count / max_possible

    return {"switch_count": switch_count, "switch_ratio": ratio}


def switching_instability_score(
    modes: List[str],
) -> float:
    """Compute switching instability score.

    Detects frequent back-and-forth patterns (A->B->A->B).
    Higher instability = higher score.

    Returns a float in [0.0, 1.0].
    """
    if len(modes) <= 2:
        return 0.0

    # Count back-and-forth: A->B->A pattern.
    reversal_count = 0
    for i in range(len(modes) - 2):
        if modes[i] == modes[i + 2] and modes[i] != modes[i + 1]:
            reversal_count += 1

    max_possible = max(len(modes) - 2, 1)
    return reversal_count / max_possible


# ---------------------------------------------------------------------------
# PART 4 — PROJECTION CHURN (ENERGY DISSIPATION)
# ---------------------------------------------------------------------------


def compute_projection_churn(
    projection_distances: List[float],
    stability_efficiency: float,
) -> Dict[str, Any]:
    """Compute projection churn (wasted correction effort).

    Large total projection with low stability gain = wasted effort.
    churn = total_projection_distance * (1 - stability_efficiency)

    Returns:
        {"total_projection": float, "churn_score": float}
    """
    total_projection = sum(abs(d) for d in projection_distances)
    churn_score = total_projection * (1.0 - stability_efficiency)

    return {
        "total_projection": total_projection,
        "churn_score": churn_score,
    }


# ---------------------------------------------------------------------------
# PART 5 — INVARIANT CONFLICT DISSIPATION
# ---------------------------------------------------------------------------


def invariant_conflict_dissipation(
    invariants: List[str],
    interaction_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute dissipation penalty from invariant conflicts.

    Uses interaction data (from detect_interactions) to penalize
    conflicting invariant pairs present in the current trace.

    Args:
        invariants: list of invariant type strings active in this trace.
        interaction_data: list of interaction dicts with 'pair', 'type',
                          'evidence_count' keys.

    Returns:
        {"conflict_count": int, "conflict_penalty": float}
    """
    if not invariants or not interaction_data:
        return {"conflict_count": 0, "conflict_penalty": 0.0}

    inv_set = set(invariants)
    conflict_count = 0
    total_evidence = 0

    for interaction in interaction_data:
        if interaction.get("type") != "conflict":
            continue
        pair = interaction.get("pair", ())
        if len(pair) == 2 and pair[0] in inv_set and pair[1] in inv_set:
            conflict_count += 1
            total_evidence += interaction.get("evidence_count", 1)

    # Penalty: 0.5 per conflict pair, scaled by evidence.
    conflict_penalty = conflict_count * 0.5
    if total_evidence > conflict_count:
        conflict_penalty += (total_evidence - conflict_count) * 0.1

    return {
        "conflict_count": conflict_count,
        "conflict_penalty": conflict_penalty,
    }


# ---------------------------------------------------------------------------
# PART 5b — LOOP TWIST SCORE
# ---------------------------------------------------------------------------


def compute_loop_twist(
    states_before: List[str],
    states_after: List[str],
) -> Dict[str, Any]:
    """Detect orientation-flipping loops in correction traces.

    A twist occurs when consecutive transitions form an A->B, B->A loop
    BUT the state_after at the return differs from the original state_before,
    indicating the system returned to the same label but through a different
    representational path (partial mismatch in surrounding context).

    Specifically: at positions i and i+1, if before[i]==after[i+1] and
    after[i]==before[i+1] (oscillation pattern) AND additionally
    after[i+1] != before[i+1] (the return state differs from its input),
    that is a twist.

    Returns:
        {"twist_count": int, "twist_ratio": float}
    """
    if not states_before or not states_after:
        return {"twist_count": 0, "twist_ratio": 0.0}

    n = min(len(states_before), len(states_after))
    if n < 2:
        return {"twist_count": 0, "twist_ratio": 0.0}

    twist_count = 0
    for i in range(n - 1):
        a_before = states_before[i]
        a_after = states_after[i]
        b_before = states_before[i + 1]
        b_after = states_after[i + 1]
        # Oscillation: A->B then B->A pattern
        if a_before == b_after and a_after == b_before:
            # Twist: the return destination differs from its own input
            if b_after != b_before:
                twist_count += 1

    max_possible = max(n - 1, 1)
    return {
        "twist_count": twist_count,
        "twist_ratio": twist_count / max_possible,
    }


# ---------------------------------------------------------------------------
# PART 5c — NONLOCAL INFLUENCE SCORE
# ---------------------------------------------------------------------------


def compute_nonlocal_influence(
    states_before: List[str],
    states_after: List[str],
) -> Dict[str, Any]:
    """Detect indirect (nonlocal) effects in correction traces.

    A nonlocal event occurs when a state changes (before != after) without
    a direct local cause — approximated by the previous step having no
    change (states_before[i-1] == states_after[i-1]) while the current
    step does change.

    Returns:
        {"nonlocal_events": int, "nonlocal_ratio": float}
    """
    if not states_before or not states_after:
        return {"nonlocal_events": 0, "nonlocal_ratio": 0.0}

    n = min(len(states_before), len(states_after))
    if n < 2:
        return {"nonlocal_events": 0, "nonlocal_ratio": 0.0}

    nonlocal_events = 0
    for i in range(1, n):
        prev_unchanged = (states_before[i - 1] == states_after[i - 1])
        curr_changed = (states_before[i] != states_after[i])
        if prev_unchanged and curr_changed:
            nonlocal_events += 1

    max_possible = max(n - 1, 1)
    return {
        "nonlocal_events": nonlocal_events,
        "nonlocal_ratio": nonlocal_events / max_possible,
    }


# ---------------------------------------------------------------------------
# PART 5d — CORRECTION ACCELERATION
# ---------------------------------------------------------------------------


def compute_acceleration(
    projection_distances: List[float],
) -> Dict[str, Any]:
    """Compute correction acceleration from projection distances.

    Acceleration measures how rapidly the correction effort changes between
    consecutive steps: mean of |delta| where delta is the difference between
    consecutive projection distances.

    Returns:
        {"mean_delta": float, "acceleration_score": float}
    """
    if len(projection_distances) < 2:
        return {"mean_delta": 0.0, "acceleration_score": 0.0}

    dists = np.asarray(projection_distances, dtype=np.float64)
    deltas = np.abs(np.diff(dists))
    mean_delta = float(np.mean(deltas))

    # Normalize to [0, 1] range: cap at 2.0 for scoring.
    acceleration_score = min(mean_delta, 2.0) / 2.0

    return {
        "mean_delta": mean_delta,
        "acceleration_score": acceleration_score,
    }


# ---------------------------------------------------------------------------
# PART 6 — TOTAL FRICTION SCORE
# ---------------------------------------------------------------------------


def compute_friction_score(
    trace: Dict[str, Any],
    interaction_data: Optional[List[Dict[str, Any]]] = None,
    invariants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute total friction score for a correction trace.

    Deterministic additive combination of all components.
    No weight tuning — simple sum.

    Returns:
        {
          "friction_score": float,
          "components": {
            "oscillation": float,
            "hysteresis": float,
            "switching": float,
            "churn": float,
            "conflict": float,
          }
        }
    """
    states_before = trace.get("states_before", [])
    states_after = trace.get("states_after", [])
    modes = trace.get("modes", [])
    proj_dists = trace.get("projection_distances", [])
    stab_eff = trace.get("stability_efficiency", 0.0)

    osc = detect_oscillation(states_before, states_after)
    hyst = detect_hysteresis(states_before, states_after)
    sw = compute_mode_switches(modes)
    instab = switching_instability_score(modes)
    churn = compute_projection_churn(proj_dists, stab_eff)

    # Normalize churn to [0, ~1] range: cap at 2.0 for scoring.
    churn_normalized = min(churn["churn_score"], 2.0) / 2.0

    conflict = invariant_conflict_dissipation(
        invariants or [], interaction_data or [],
    )

    friction = (
        osc["oscillation_ratio"]
        + hyst["hysteresis_ratio"]
        + sw["switch_ratio"]
        + instab
        + churn_normalized
        + conflict["conflict_penalty"]
    )

    # Extended metrics (additive, do not alter friction_score).
    twist = compute_loop_twist(states_before, states_after)
    nonlocal_inf = compute_nonlocal_influence(states_before, states_after)
    accel = compute_acceleration(proj_dists)

    return {
        "friction_score": friction,
        "components": {
            "oscillation": osc["oscillation_ratio"],
            "hysteresis": hyst["hysteresis_ratio"],
            "switching": sw["switch_ratio"],
            "churn": churn_normalized,
            "conflict": conflict["conflict_penalty"],
        },
        "extended_components": {
            "twist": twist,
            "nonlocal": nonlocal_inf,
            "acceleration": accel,
        },
    }


# ---------------------------------------------------------------------------
# PART 7 — CLASSIFICATION
# ---------------------------------------------------------------------------


def classify_dynamics(
    friction_score: float,
    extended_components: Optional[Dict[str, Any]] = None,
) -> str:
    """Label correction dynamics regime.

    Base rules:
        < 1.0  -> "stable"
        1.0-2.5 -> "adaptive"
        >= 2.5  -> "frictional"

    Extended labels (appended with '+' separator):
        twist_ratio > 0.3   -> "+twisted"
        nonlocal_ratio > 0.3 -> "+nonlocal"
        acceleration_score high (> 0.3) -> "+accelerated"
    """
    if friction_score < 1.0:
        base = "stable"
    elif friction_score <= 2.5:
        base = "adaptive"
    else:
        base = "frictional"

    if extended_components is None:
        return base

    labels = [base]
    twist = extended_components.get("twist", {})
    if twist.get("twist_ratio", 0.0) > 0.3:
        labels.append("twisted")

    nonlocal_inf = extended_components.get("nonlocal", {})
    if nonlocal_inf.get("nonlocal_ratio", 0.0) > 0.3:
        labels.append("nonlocal")

    accel = extended_components.get("acceleration", {})
    if accel.get("acceleration_score", 0.0) > 0.3:
        labels.append("accelerated")

    return "+".join(labels)


# ---------------------------------------------------------------------------
# PART 8 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_correction_dynamics(
    data: Any,
    interaction_data: Optional[List[Dict[str, Any]]] = None,
    invariants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run full correction dynamics pipeline.

    Pipeline:
      1. extract traces
      2. compute all components
      3. compute friction score
      4. classify dynamics

    Args:
        data: application output (dict with 'applications' key).
        interaction_data: optional interaction list from detect_interactions.
        invariants: optional list of active invariant types.

    Returns:
        {"results": [{"dfa_type", "n", "friction_score", "regime",
                       "components"}]}
    """
    traces = extract_correction_traces(data)

    results: List[Dict[str, Any]] = []
    for trace in traces:
        friction = compute_friction_score(
            trace, interaction_data, invariants,
        )
        extended = friction.get("extended_components", {})
        regime = classify_dynamics(friction["friction_score"], extended)

        results.append({
            "dfa_type": trace["dfa_type"],
            "n": trace["n"],
            "friction_score": friction["friction_score"],
            "regime": regime,
            "components": friction["components"],
            "extended": extended,
        })

    return {"results": results}


# ---------------------------------------------------------------------------
# PART 9 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_dynamics_report(report: Dict[str, Any]) -> str:
    """Format correction dynamics report as text.

    Returns formatted string (does not print directly).
    """
    lines = ["=== Correction Dynamics ==="]

    for result in report.get("results", []):
        dfa_type = result.get("dfa_type", "unknown")
        n = result.get("n")
        n_str = f" (n={n})" if n is not None else ""
        lines.append(f"DFA: {dfa_type}{n_str}")
        lines.append(f"  friction_score: {result['friction_score']:.1f}")
        lines.append(f"  regime: {result['regime']}")
        lines.append("  components:")
        for comp_name in sorted(result.get("components", {}).keys()):
            val = result["components"][comp_name]
            lines.append(f"    {comp_name}: {val:.1f}")
        extended = result.get("extended", {})
        if extended:
            lines.append("  extended:")
            twist = extended.get("twist", {})
            lines.append(f"    twist: {twist.get('twist_ratio', 0.0):.2f}")
            nonlocal_inf = extended.get("nonlocal", {})
            lines.append(f"    nonlocal: {nonlocal_inf.get('nonlocal_ratio', 0.0):.2f}")
            accel = extended.get("acceleration", {})
            lines.append(f"    acceleration: {accel.get('acceleration_score', 0.0):.2f}")
        lines.append("---")

    return "\n".join(lines)
