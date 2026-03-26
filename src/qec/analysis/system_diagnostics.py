"""v104.0.0 — Unified system diagnostics layer.

Synthesizes all analysis modules into a single coherent report.

Pipeline:
    trajectory -> regime -> taxonomy -> evolution -> transition_graph
    -> phase_space -> flow_geometry -> multistate -> coupled_dynamics
    -> control / feedback / global / hierarchical
    -> policy_memory -> policy_clustering -> strategy_graph

Extracts global metrics:
    - dominant_strategy
    - system_stability (avg across strategies)
    - convergence_rate
    - volatility_score
    - topology_type
    - best_policy
    - best_archetype

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ROUND_PRECISION = 12


def run_system_diagnostics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run unified system diagnostics across all analysis layers.

    Executes the full analysis pipeline and synthesizes global metrics
    into a single coherent report.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains all sub-results plus a ``"global_metrics"`` key with
        synthesized system-level metrics.
    """
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_control_analysis,
        run_evolution_analysis,
        run_feedback_analysis,
        run_flow_geometry_analysis,
        run_global_control_analysis,
        run_hierarchical_control_analysis,
        run_multistate_analysis,
        run_phase_space_analysis,
        run_policy_clustering_analysis,
        run_policy_memory_analysis,
        run_strategy_graph_analysis,
        run_taxonomy_analysis,
        run_transition_graph_analysis,
        run_trajectory_analysis,
    )

    # --- Stage 1: trajectory -> regime -> taxonomy ---
    trajectory_result = run_trajectory_analysis(runs)
    taxonomy_result = run_taxonomy_analysis(runs)
    evolution_result = run_evolution_analysis(runs)
    transition_graph_result = run_transition_graph_analysis(runs)

    # --- Stage 2: phase_space -> flow_geometry ---
    phase_space_result = run_phase_space_analysis(runs)
    flow_geometry_result = run_flow_geometry_analysis(
        runs,
        phase_space_result=phase_space_result,
    )

    # --- Stage 3: multistate -> coupled_dynamics ---
    multistate_result = run_multistate_analysis(
        runs,
        phase_space_result=phase_space_result,
    )
    coupled_result = run_coupled_dynamics_analysis(
        runs,
        multistate_result=multistate_result,
    )

    # --- Stage 4: control / feedback / global / hierarchical ---
    control_result = run_control_analysis(
        runs,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )
    feedback_result = run_feedback_analysis(
        runs,
        multistate_result=multistate_result,
    )
    global_control_result = run_global_control_analysis(
        runs,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )
    hierarchical_result = run_hierarchical_control_analysis(
        runs,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )

    # --- Stage 5: policy_memory -> policy_clustering -> strategy_graph ---
    policy_memory_result = run_policy_memory_analysis(
        runs,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )
    policy_clustering_result = run_policy_clustering_analysis(
        runs,
        memory=policy_memory_result.get("memory"),
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )
    strategy_graph_result = run_strategy_graph_analysis(
        runs,
        multistate_result=multistate_result,
        coupled_result=coupled_result,
    )

    # --- Stage 6: trajectory geometry ---
    from qec.analysis.trajectory_geometry import run_trajectory_geometry_analysis
    trajectory_geometry_result = run_trajectory_geometry_analysis(runs)

    # --- Stage 7: differential diagnosis ---
    from qec.analysis.differential_diagnosis import run_differential_diagnosis

    # --- Synthesize global metrics ---
    global_metrics = _extract_global_metrics(
        trajectory_result=trajectory_result,
        taxonomy_result=taxonomy_result,
        evolution_result=evolution_result,
        phase_space_result=phase_space_result,
        feedback_result=feedback_result,
        hierarchical_result=hierarchical_result,
        policy_memory_result=policy_memory_result,
        policy_clustering_result=policy_clustering_result,
        strategy_graph_result=strategy_graph_result,
        trajectory_geometry_result=trajectory_geometry_result,
    )

    # Run differential diagnosis on the assembled result.
    diagnosis_input = {
        "global_metrics": global_metrics,
        "trajectory_geometry": trajectory_geometry_result,
    }
    diagnosis_result = run_differential_diagnosis(diagnosis_input)

    # Enrich global metrics with primary diagnosis.
    global_metrics["primary_diagnosis"] = diagnosis_result.get(
        "primary_diagnosis", "unknown"
    )
    global_metrics["diagnosis_confidence"] = diagnosis_result.get(
        "diagnosis_confidence", 0.0
    )

    # --- Stage 8: provocation + treatment + invariants ---
    from qec.analysis.provocation_analysis import run_provocation_analysis
    from qec.analysis.treatment_planning import run_treatment_planning
    from qec.analysis.treatment_invariants import (
        extract_treatment_invariants,
        score_invariants,
    )

    provocation_result = run_provocation_analysis(runs)
    treatment_result = run_treatment_planning(runs)

    raw_invariants = extract_treatment_invariants(
        diagnosis_result, provocation_result, treatment_result,
    )
    scored_inv = score_invariants(raw_invariants)
    scored_list = scored_inv.get("scored_invariants", [])

    # Enrich global metrics with provocation/treatment/invariant data.
    prov_response = provocation_result.get("response", {})
    prov_revision = provocation_result.get("revision", {})
    best_treatment = treatment_result.get("best_treatment", {})
    best_candidate = best_treatment.get("candidate", {})

    global_metrics["baseline_response_class"] = prov_response.get(
        "classification", "unchanged"
    )
    global_metrics["revised_diagnosis"] = prov_revision.get(
        "revised_diagnosis", global_metrics["primary_diagnosis"]
    )
    global_metrics["diagnosis_shift"] = round(
        prov_revision.get("confidence_shift", 0.0), ROUND_PRECISION
    )
    global_metrics["best_treatment"] = (
        f"{best_candidate.get('action', 'none')}"
        f"({best_candidate.get('strength', 0.0)})"
        if best_candidate else "none"
    )
    global_metrics["treatment_score"] = round(
        best_treatment.get("score", 0.0), ROUND_PRECISION
    )
    global_metrics["invariant_count"] = len(scored_list)
    global_metrics["strongest_invariant"] = (
        scored_list[0].get("name", "none") if scored_list else "none"
    )

    return {
        "trajectory": trajectory_result,
        "taxonomy": taxonomy_result,
        "evolution": evolution_result,
        "transition_graph": transition_graph_result,
        "phase_space": phase_space_result,
        "flow_geometry": flow_geometry_result,
        "multistate": multistate_result,
        "coupled_dynamics": coupled_result,
        "control": control_result,
        "feedback": feedback_result,
        "global_control": global_control_result,
        "hierarchical": hierarchical_result,
        "policy_memory": policy_memory_result,
        "policy_clustering": policy_clustering_result,
        "strategy_graph": strategy_graph_result,
        "trajectory_geometry": trajectory_geometry_result,
        "differential_diagnosis": diagnosis_result,
        "provocation": provocation_result,
        "treatment": treatment_result,
        "treatment_invariants": raw_invariants,
        "scored_invariants": scored_list,
        "global_metrics": global_metrics,
    }


def _extract_global_metrics(
    *,
    trajectory_result: Dict[str, Any],
    taxonomy_result: Dict[str, Any],
    evolution_result: Dict[str, Any],
    phase_space_result: Dict[str, Any],
    feedback_result: Dict[str, Any],
    hierarchical_result: Dict[str, Any],
    policy_memory_result: Dict[str, Any],
    policy_clustering_result: Dict[str, Any],
    strategy_graph_result: Dict[str, Any],
    trajectory_geometry_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract synthesized global metrics from all sub-results.

    Parameters
    ----------
    trajectory_result : dict
        Output of trajectory analysis.
    taxonomy_result : dict
        Output of taxonomy analysis.
    evolution_result : dict
        Output of evolution analysis.
    phase_space_result : dict
        Output of phase space analysis.
    feedback_result : dict
        Output of feedback analysis.
    hierarchical_result : dict
        Output of hierarchical control analysis.
    policy_memory_result : dict
        Output of policy memory analysis.
    policy_clustering_result : dict
        Output of policy clustering analysis.
    strategy_graph_result : dict
        Output of strategy graph analysis.
    trajectory_geometry_result : dict, optional
        Output of trajectory geometry analysis.

    Returns
    -------
    dict
        Synthesized global metrics.
    """
    # --- dominant_strategy ---
    dominant_strategy = _compute_dominant_strategy(taxonomy_result)

    # --- system_stability (avg stability across all strategies) ---
    system_stability = _compute_system_stability(trajectory_result)

    # --- convergence_rate ---
    convergence_rate = _compute_convergence_rate(feedback_result)

    # --- volatility_score ---
    volatility_score = _compute_volatility_score(trajectory_result)

    # --- topology_type ---
    topology_type = strategy_graph_result.get("topology", "unknown")

    # --- best_policy ---
    best_policy = _compute_best_policy(policy_memory_result)

    # --- best_archetype ---
    best_archetype = _compute_best_archetype(policy_clustering_result)

    # --- geometry metrics ---
    geom = trajectory_geometry_result or {}
    rot = geom.get("rotation_metrics", {})
    geom_pred = geom.get("predictions", {})

    result = {
        "dominant_strategy": dominant_strategy,
        "system_stability": round(system_stability, ROUND_PRECISION),
        "convergence_rate": round(convergence_rate, ROUND_PRECISION),
        "volatility_score": round(volatility_score, ROUND_PRECISION),
        "topology_type": topology_type,
        "best_policy": best_policy,
        "best_archetype": best_archetype,
        "angular_velocity": rot.get("angular_velocity", 0.0),
        "spiral_score": rot.get("spiral_score", 0.0),
        "basin_switch_risk": geom_pred.get("basin_switch_risk", "low"),
    }

    return result


def _compute_dominant_strategy(taxonomy_result: Dict[str, Any]) -> str:
    """Find the dominant strategy type from taxonomy classification.

    Selects the type with highest confidence. Ties are broken
    by lexicographic ordering of strategy names for determinism.

    Parameters
    ----------
    taxonomy_result : dict
        Must contain ``"taxonomy"`` key.

    Returns
    -------
    str
        Name of the dominant strategy type, or ``"unknown"``.
    """
    taxonomy = taxonomy_result.get("taxonomy", {})
    if not taxonomy:
        return "unknown"

    # Count occurrences of each type, weighted by confidence.
    type_scores: Dict[str, float] = {}
    for name in sorted(taxonomy.keys()):
        entry = taxonomy[name]
        stype = entry.get("type", "unknown")
        confidence = entry.get("confidence", 0.0)
        if stype not in type_scores:
            type_scores[stype] = 0.0
        type_scores[stype] += confidence

    if not type_scores:
        return "unknown"

    # Select type with highest aggregate confidence.
    # Sort by (-score, type_name) for determinism.
    best_type = sorted(type_scores.keys(), key=lambda t: (-type_scores[t], t))[0]
    return best_type


def _compute_system_stability(trajectory_result: Dict[str, Any]) -> float:
    """Compute average stability across all strategies.

    Parameters
    ----------
    trajectory_result : dict
        Must contain ``"trajectory_metrics"`` key.

    Returns
    -------
    float
        Average stability score, or ``0.0`` if no strategies.
    """
    traj_metrics = trajectory_result.get("trajectory_metrics", {})
    if not traj_metrics:
        return 0.0

    stabilities = []
    for name in sorted(traj_metrics.keys()):
        stabilities.append(traj_metrics[name].get("stability", 0.0))

    if not stabilities:
        return 0.0

    return sum(stabilities) / len(stabilities)


def _compute_convergence_rate(feedback_result: Dict[str, Any]) -> float:
    """Compute convergence rate from feedback control results.

    Uses the ratio of converged strategies to total strategies.

    Parameters
    ----------
    feedback_result : dict
        Output of feedback analysis.

    Returns
    -------
    float
        Convergence rate in [0, 1], or ``0.0`` if unavailable.
    """
    steps = feedback_result.get("steps", [])
    if not steps:
        # Fall back to convergence flag.
        converged = feedback_result.get("converged", False)
        return 1.0 if converged else 0.0

    # Count strategies that converged (stability delta below threshold).
    final_step = steps[-1] if steps else {}
    deltas = final_step.get("stability_deltas", {})

    if not deltas:
        converged = feedback_result.get("converged", False)
        return 1.0 if converged else 0.0

    threshold = 0.01  # STABILITY_DELTA_THRESHOLD from feedback_control.
    converged_count = 0
    total = 0
    for name in sorted(deltas.keys()):
        total += 1
        if abs(deltas[name]) <= threshold:
            converged_count += 1

    if total == 0:
        return 0.0

    return converged_count / total


def _compute_volatility_score(trajectory_result: Dict[str, Any]) -> float:
    """Compute system volatility as average variance across strategies.

    Parameters
    ----------
    trajectory_result : dict
        Must contain ``"trajectory_metrics"`` key.

    Returns
    -------
    float
        Average variance score, or ``0.0`` if no strategies.
    """
    traj_metrics = trajectory_result.get("trajectory_metrics", {})
    if not traj_metrics:
        return 0.0

    variances = []
    for name in sorted(traj_metrics.keys()):
        variances.append(traj_metrics[name].get("variance_score", 0.0))

    if not variances:
        return 0.0

    return sum(variances) / len(variances)


def _compute_best_policy(policy_memory_result: Dict[str, Any]) -> str:
    """Extract the best policy name from memory.

    Parameters
    ----------
    policy_memory_result : dict
        Must contain ``"memory"`` key.

    Returns
    -------
    str
        Name of the best policy, or ``"none"``.
    """
    memory = policy_memory_result.get("memory", {})
    policies = memory.get("policies", {})
    if not policies:
        return "none"

    # Find policy with highest score.
    best_name = "none"
    best_score = -1.0
    for name in sorted(policies.keys()):
        score = policies[name].get("score", 0.0)
        if score > best_score or (score == best_score and name < best_name):
            best_score = score
            best_name = name

    return best_name


def _compute_best_archetype(policy_clustering_result: Dict[str, Any]) -> str:
    """Extract the best archetype from clustering results.

    Parameters
    ----------
    policy_clustering_result : dict
        Must contain ``"ranked_archetypes"`` key.

    Returns
    -------
    str
        Name of the best archetype, or ``"none"``.
    """
    ranked = policy_clustering_result.get("ranked_archetypes", [])
    if not ranked:
        return "none"

    # Ranked archetypes are sorted best-first.
    best = ranked[0]
    if isinstance(best, dict):
        return best.get("name", best.get("archetype", "archetype_0"))
    return str(best)


def format_system_diagnostics(result: Dict[str, Any]) -> str:
    """Format system diagnostics results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_system_diagnostics``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    gm = result.get("global_metrics", {})

    lines.append("")
    lines.append("=== System Diagnostics ===")
    lines.append("")
    lines.append(f"Stability: {gm.get('system_stability', 0.0):.2f}")
    lines.append(f"Volatility: {gm.get('volatility_score', 0.0):.2f}")
    lines.append(f"Topology: {gm.get('topology_type', 'unknown')}")
    lines.append("")
    lines.append(f"Best Policy: {gm.get('best_policy', 'none')}")
    lines.append(f"Best Archetype: {gm.get('best_archetype', 'none')}")
    lines.append("")
    lines.append(f"Dominant Strategy: {gm.get('dominant_strategy', 'unknown')}")
    lines.append(f"Convergence Rate: {gm.get('convergence_rate', 0.0):.2f}")
    lines.append("")
    lines.append(f"Angular Velocity: {gm.get('angular_velocity', 0.0):.2f}")
    lines.append(f"Spiral Score: {gm.get('spiral_score', 0.0):.2f}")
    lines.append(f"Basin Switch Risk: {gm.get('basin_switch_risk', 'low')}")
    lines.append("")
    lines.append(f"Primary Diagnosis: {gm.get('primary_diagnosis', 'unknown')}")
    lines.append(f"Diagnosis Confidence: {gm.get('diagnosis_confidence', 0.0):.2f}")
    lines.append(f"Baseline Response: {gm.get('baseline_response_class', 'unknown')}")
    lines.append(f"Revised Diagnosis: {gm.get('revised_diagnosis', 'unknown')}")
    lines.append(f"Diagnosis Shift: {gm.get('diagnosis_shift', 0.0):+.2f}")
    lines.append("")
    lines.append(f"Best Treatment: {gm.get('best_treatment', 'none')}")
    lines.append(f"Treatment Score: {gm.get('treatment_score', 0.0):.2f}")
    lines.append(f"Invariant Count: {gm.get('invariant_count', 0)}")
    lines.append(f"Strongest Invariant: {gm.get('strongest_invariant', 'none')}")

    return "\n".join(lines)
