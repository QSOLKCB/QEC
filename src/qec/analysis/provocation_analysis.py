"""v104.5.0 — Provocation analysis for diagnosis-driven baseline intervention.

Implements the provocation pipeline:
    differential_diagnosis
    -> baseline intervention selection
    -> intervention application
    -> response characterization
    -> diagnosis revision

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Response classifications
# ---------------------------------------------------------------------------

RESPONSE_CLASSES = [
    "improved",
    "worsened",
    "unchanged",
    "destabilized",
    "revealed_new_mode",
]

# ---------------------------------------------------------------------------
# Baseline intervention mapping (diagnosis -> low-strength actions)
# ---------------------------------------------------------------------------

_BASELINE_MAP: Dict[str, List[Dict[str, Any]]] = {
    "oscillatory_trap": [
        {"action": "reduce_escape", "strength": 0.2},
        {"action": "boost_stability", "strength": 0.2},
    ],
    "metastable_plateau": [
        {"action": "force_transition", "strength": 0.2},
    ],
    "basin_switch_instability": [
        {"action": "boost_stability", "strength": 0.2},
    ],
    "control_overshoot": [
        {"action": "reduce_escape", "strength": 0.2},
    ],
    "underconstrained_dynamics": [
        {"action": "boost_stability", "strength": 0.2},
    ],
    "slow_convergence": [
        {"action": "boost_stability", "strength": 0.2},
    ],
    "healthy_convergence": [],
}


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1. Baseline intervention selection
# ---------------------------------------------------------------------------


def get_baseline_interventions(diagnosis: dict) -> list:
    """Select deterministic baseline interventions for a diagnosis.

    Parameters
    ----------
    diagnosis : dict
        Output of ``run_differential_diagnosis``.  Must contain
        ``primary_diagnosis``.

    Returns
    -------
    list of dict
        Each dict has ``action`` and ``strength`` keys.
        Returns empty list for ``healthy_convergence``.
    """
    primary = str(diagnosis.get("primary_diagnosis", "healthy_convergence"))
    interventions = _BASELINE_MAP.get(primary, [])
    # Return copies to avoid mutation of module-level data.
    return [dict(i) for i in interventions]


# ---------------------------------------------------------------------------
# 2. Apply baseline interventions
# ---------------------------------------------------------------------------


def apply_baseline_interventions(
    runs: list,
    interventions: list,
) -> dict:
    """Apply baseline interventions and compute before/after state.

    Reuses the existing control layer infrastructure.  Does not mutate
    *runs* or *interventions*.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    interventions : list of dict
        Each must have ``action`` and ``strength``.  The ``target`` key
        is filled in automatically for each strategy present.

    Returns
    -------
    dict
        Contains ``before``, ``after``, ``interventions_applied``,
        and ``response`` (per-strategy deltas).
    """
    from qec.analysis.strategy_adapter import run_multistate_analysis
    from qec.analysis.control_layer import (
        apply_intervention,
        evaluate_intervention,
    )

    multistate_result = run_multistate_analysis(runs)
    before = multistate_result.get("multistate", {})

    # Deep copy the before state.
    after: Dict[str, Dict[str, Any]] = {}
    for name in sorted(before.keys()):
        sv = before[name]
        after[name] = {
            "ternary": dict(sv.get("ternary", {})),
            "membership": dict(sv.get("membership", {})),
        }

    applied: List[Dict[str, Any]] = []
    for name in sorted(after.keys()):
        for intervention in interventions:
            rule = {
                "target": name,
                "action": str(intervention.get("action", "")),
                "strength": float(intervention.get("strength", 0.0)),
            }
            after[name] = apply_intervention(after[name], rule)
            applied.append(rule)

    response = evaluate_intervention(before, after)

    return {
        "before": before,
        "after": after,
        "interventions_applied": applied,
        "response": response,
    }


# ---------------------------------------------------------------------------
# 3. Response characterization
# ---------------------------------------------------------------------------


def characterize_response(before: dict, after: dict) -> dict:
    """Characterize the system response to a baseline intervention.

    Computes deltas for key metrics and classifies the overall response.

    Parameters
    ----------
    before : dict
        Pre-intervention multistate dict keyed by strategy name.
    after : dict
        Post-intervention multistate dict keyed by strategy name.

    Returns
    -------
    dict
        Contains ``deltas`` (per-metric), ``classification`` (str),
        and ``per_strategy`` details.
    """
    all_names = sorted(set(list(before.keys()) + list(after.keys())))

    # Aggregate deltas across strategies.
    total_stability = 0.0
    total_attractor = 0.0
    total_transient = 0.0
    total_phase = 0.0
    count = max(len(all_names), 1)

    per_strategy: Dict[str, Dict[str, float]] = {}

    for name in all_names:
        bv = before.get(name, {})
        av = after.get(name, {})
        bt = bv.get("ternary", {})
        at = av.get("ternary", {})
        bm = bv.get("membership", {})
        am = av.get("membership", {})

        d_stability = float(at.get("stability_state", 0) - bt.get("stability_state", 0))
        d_phase = float(at.get("phase_state", 0) - bt.get("phase_state", 0))

        b_attractor = bm.get("strong_attractor", 0.0) + bm.get("weak_attractor", 0.0)
        a_attractor = am.get("strong_attractor", 0.0) + am.get("weak_attractor", 0.0)
        d_attractor = a_attractor - b_attractor

        d_transient = am.get("transient", 0.0) - bm.get("transient", 0.0)

        per_strategy[name] = {
            "delta_stability": _round(d_stability),
            "delta_phase": _round(d_phase),
            "delta_attractor_weight": _round(d_attractor),
            "delta_transient_weight": _round(d_transient),
        }

        total_stability += d_stability
        total_attractor += d_attractor
        total_transient += d_transient
        total_phase += d_phase

    avg_stability = _round(total_stability / count)
    avg_attractor = _round(total_attractor / count)
    avg_transient = _round(total_transient / count)
    avg_phase = _round(total_phase / count)

    deltas = {
        "stability": avg_stability,
        "attractor_weight": avg_attractor,
        "transient_weight": avg_transient,
        "phase": avg_phase,
    }

    classification = _classify_response(deltas)

    return {
        "deltas": deltas,
        "classification": classification,
        "per_strategy": per_strategy,
    }


def _classify_response(deltas: dict) -> str:
    """Classify the aggregate response into one of the response classes.

    Rules (evaluated in order):
    - destabilized: stability decreased AND transient increased
    - improved: stability increased OR (attractor increased AND transient decreased)
    - worsened: stability decreased OR transient increased
    - revealed_new_mode: phase changed significantly
    - unchanged: no significant change
    """
    d_stability = deltas.get("stability", 0.0)
    d_attractor = deltas.get("attractor_weight", 0.0)
    d_transient = deltas.get("transient_weight", 0.0)
    d_phase = deltas.get("phase", 0.0)

    # Destabilized: simultaneous stability loss and transient gain.
    if d_stability < 0.0 and d_transient > 0.0:
        return "destabilized"

    # Improved: stability gain or attractor gain with transient reduction.
    if d_stability > 0.0:
        return "improved"
    if d_attractor > 0.0 and d_transient < 0.0:
        return "improved"

    # Worsened: stability loss or transient gain (but not destabilized).
    if d_stability < 0.0:
        return "worsened"
    if d_transient > 0.0:
        return "worsened"

    # Revealed new mode: phase change without stability/transient change.
    if abs(d_phase) > 0.0:
        return "revealed_new_mode"

    return "unchanged"


# ---------------------------------------------------------------------------
# 4. Diagnosis revision from response
# ---------------------------------------------------------------------------


def revise_diagnosis_from_response(
    diagnosis: dict,
    response: dict,
) -> dict:
    """Revise the diagnosis based on the observed provocation response.

    Rules:
    - improved: increase confidence in primary diagnosis
    - worsened: decrease confidence, elevate alternative
    - destabilized: elevate alternative diagnosis
    - revealed_new_mode: set revealed_mode flag
    - unchanged: no revision

    Parameters
    ----------
    diagnosis : dict
        Output of ``run_differential_diagnosis``.
    response : dict
        Output of ``characterize_response``.

    Returns
    -------
    dict
        Contains ``revised_diagnosis``, ``revised_confidence``,
        ``confidence_shift``, ``revealed_mode``, ``revision_reason``.
    """
    primary = str(diagnosis.get("primary_diagnosis", "unknown"))
    confidence = float(diagnosis.get("diagnosis_confidence", 0.0))
    ranked = list(diagnosis.get("ranked", []))
    classification = str(response.get("classification", "unchanged"))

    revised_diagnosis = primary
    confidence_shift = 0.0
    revealed_mode = ""
    reason = "no revision needed"

    if classification == "improved":
        # Baseline helped — confirms diagnosis.
        confidence_shift = 0.12
        reason = "baseline intervention improved suspected failure mode"

    elif classification == "worsened":
        # Baseline made it worse — suspect alternative.
        confidence_shift = -0.10
        reason = "baseline worsened condition; alternative diagnosis elevated"
        # Elevate the second-ranked diagnosis if available.
        if len(ranked) >= 2:
            alt_name = ranked[1][0]
            alt_score = ranked[1][1]
            if alt_score > 0.0:
                revised_diagnosis = alt_name
                reason = (
                    f"baseline worsened condition; elevated {alt_name} "
                    f"(score {alt_score:.2f})"
                )

    elif classification == "destabilized":
        # System destabilized — strong signal for alternative.
        confidence_shift = -0.15
        reason = "baseline destabilized system; alternative diagnosis elevated"
        if len(ranked) >= 2:
            alt_name = ranked[1][0]
            revised_diagnosis = alt_name
            reason = (
                f"destabilization under baseline; elevated {alt_name}"
            )

    elif classification == "revealed_new_mode":
        # New mode revealed.
        confidence_shift = -0.05
        revealed_mode = "phase_shift_detected"
        reason = "baseline revealed new mode not matching primary diagnosis"

    revised_confidence = _round(_clamp(confidence + confidence_shift))

    return {
        "revised_diagnosis": revised_diagnosis,
        "revised_confidence": revised_confidence,
        "confidence_shift": _round(confidence_shift),
        "revealed_mode": revealed_mode,
        "revision_reason": reason,
    }


# ---------------------------------------------------------------------------
# 5. Full provocation pipeline
# ---------------------------------------------------------------------------


def run_provocation_analysis(runs: list) -> dict:
    """Run the full provocation analysis pipeline.

    Pipeline:
        system_diagnostics
        -> differential_diagnosis
        -> baseline interventions
        -> response characterization
        -> revised diagnosis

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.

    Returns
    -------
    dict
        Contains ``diagnosis``, ``interventions``, ``intervention_result``,
        ``response``, ``revision``.
    """
    from qec.analysis.differential_diagnosis import run_differential_diagnosis
    from qec.analysis.trajectory_geometry import (
        run_trajectory_geometry_analysis,
    )

    # Build a lightweight diagnosis input without calling run_system_diagnostics
    # to avoid circular dependency (system_diagnostics -> provocation -> system_diagnostics).
    tg_result = run_trajectory_geometry_analysis(runs)

    # Extract minimal global metrics needed for diagnosis.
    from qec.analysis.strategy_adapter import (
        run_trajectory_analysis,
        run_feedback_analysis,
        run_taxonomy_analysis,
        run_strategy_graph_analysis,
        run_multistate_analysis,
        run_coupled_dynamics_analysis,
    )

    trajectory_result = run_trajectory_analysis(runs)
    feedback_result = run_feedback_analysis(
        runs,
        multistate_result=run_multistate_analysis(runs),
    )

    # Compute minimal global metrics for diagnosis.
    traj_metrics = trajectory_result.get("trajectory_metrics", {})
    stabilities = [
        traj_metrics[n].get("stability", 0.0) for n in sorted(traj_metrics.keys())
    ]
    system_stability = sum(stabilities) / max(len(stabilities), 1)

    steps = feedback_result.get("steps", [])
    if steps:
        final_step = steps[-1]
        deltas = final_step.get("stability_deltas", {})
        total = len(deltas)
        converged_count = sum(1 for n in sorted(deltas.keys()) if abs(deltas[n]) <= 0.01)
        convergence_rate = converged_count / max(total, 1)
    else:
        convergence_rate = 1.0 if feedback_result.get("converged", False) else 0.0

    variances = [
        traj_metrics[n].get("variance_score", 0.0) for n in sorted(traj_metrics.keys())
    ]
    volatility_score = sum(variances) / max(len(variances), 1)

    rot = tg_result.get("rotation_metrics", {})
    geom_pred = tg_result.get("predictions", {})

    global_metrics = {
        "system_stability": system_stability,
        "convergence_rate": convergence_rate,
        "volatility_score": volatility_score,
        "topology_type": "unknown",
        "angular_velocity": rot.get("angular_velocity", 0.0),
        "spiral_score": rot.get("spiral_score", 0.0),
        "basin_switch_risk": geom_pred.get("basin_switch_risk", "low"),
    }

    diagnosis_input = {
        "global_metrics": global_metrics,
        "trajectory_geometry": tg_result,
    }
    diagnosis = run_differential_diagnosis(diagnosis_input)

    interventions = get_baseline_interventions(diagnosis)

    if not interventions:
        # Healthy convergence — no provocation needed.
        return {
            "diagnosis": diagnosis,
            "interventions": [],
            "intervention_result": {},
            "response": {
                "deltas": {},
                "classification": "unchanged",
                "per_strategy": {},
            },
            "revision": {
                "revised_diagnosis": diagnosis.get("primary_diagnosis", "unknown"),
                "revised_confidence": diagnosis.get("diagnosis_confidence", 0.0),
                "confidence_shift": 0.0,
                "revealed_mode": "",
                "revision_reason": "healthy convergence; no provocation needed",
            },
        }

    intervention_result = apply_baseline_interventions(runs, interventions)

    response = characterize_response(
        intervention_result["before"],
        intervention_result["after"],
    )

    revision = revise_diagnosis_from_response(diagnosis, response)

    return {
        "diagnosis": diagnosis,
        "interventions": interventions,
        "intervention_result": intervention_result,
        "response": response,
        "revision": revision,
    }


# ---------------------------------------------------------------------------
# 6. Formatter
# ---------------------------------------------------------------------------


def format_provocation_analysis(result: dict) -> str:
    """Format provocation analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_provocation_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    diagnosis = result.get("diagnosis", {})
    interventions = result.get("interventions", [])
    response = result.get("response", {})
    revision = result.get("revision", {})

    primary = diagnosis.get("primary_diagnosis", "unknown")
    deltas = response.get("deltas", {})
    classification = response.get("classification", "unchanged")

    lines.append("")
    lines.append("=== Provocation Analysis ===")
    lines.append("")
    lines.append(f"Primary Diagnosis: {primary}")

    if interventions:
        lines.append("Baseline Intervention:")
        for iv in interventions:
            lines.append(f"  {iv['action']}({iv['strength']})")
    else:
        lines.append("Baseline Intervention: none (healthy convergence)")

    lines.append("")
    lines.append("Response:")
    for key in sorted(deltas.keys()):
        val = deltas[key]
        sign = "+" if val >= 0 else ""
        lines.append(f"  {_format_metric_name(key)}: {sign}{val:.2f}")
    lines.append(f"  Classification: {classification}")

    lines.append("")
    revised = revision.get("revised_diagnosis", primary)
    shift = revision.get("confidence_shift", 0.0)
    sign = "+" if shift >= 0 else ""
    lines.append("Revised Diagnosis:")
    lines.append(f"  {revised}")
    lines.append(f"Confidence Shift: {sign}{shift:.2f}")

    revealed = revision.get("revealed_mode", "")
    if revealed:
        lines.append(f"Revealed Mode: {revealed}")

    return "\n".join(lines)


def _format_metric_name(name: str) -> str:
    """Format a metric name for display."""
    return name.replace("_", " ").title()


__all__ = [
    "RESPONSE_CLASSES",
    "ROUND_PRECISION",
    "apply_baseline_interventions",
    "characterize_response",
    "format_provocation_analysis",
    "get_baseline_interventions",
    "revise_diagnosis_from_response",
    "run_provocation_analysis",
]
