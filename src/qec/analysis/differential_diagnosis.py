"""v104.2.0 — Differential diagnosis engine for decoder behavior.

Identifies likely failure modes, ranks competing explanations, and
generates human-readable explanations for observed system behavior.

Pipeline:
    system_diagnostics + trajectory_geometry
    -> diagnostic features
    -> rule-based scoring
    -> ranked diagnoses
    -> explanations

Failure modes:
    - oscillatory_trap
    - metastable_plateau
    - basin_switch_instability
    - control_overshoot
    - underconstrained_dynamics
    - slow_convergence
    - healthy_convergence

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

ROUND_PRECISION = 12

FAILURE_MODES = [
    "oscillatory_trap",
    "metastable_plateau",
    "basin_switch_instability",
    "control_overshoot",
    "underconstrained_dynamics",
    "slow_convergence",
    "healthy_convergence",
]


# ---------------------------------------------------------------------------
# 1. Diagnostic Feature Extraction
# ---------------------------------------------------------------------------


def extract_diagnostic_features(result: dict) -> dict:
    """Extract diagnostic features from system diagnostics and geometry results.

    Pulls features from existing outputs: angular_velocity, spiral_score,
    curvature, axis_lock, coupling metrics, topology, convergence_rate,
    stability.

    Parameters
    ----------
    result : dict
        Combined result dict containing keys from ``run_system_diagnostics``
        and/or ``run_trajectory_geometry_analysis``.  Gracefully defaults
        to 0.0 / "unknown" for missing keys.

    Returns
    -------
    dict
        Flat dict of diagnostic features with deterministic keys.
    """
    gm = result.get("global_metrics", {})
    tg = result.get("trajectory_geometry", {})
    rot = tg.get("rotation_metrics", {})
    coupling = tg.get("coupling_metrics", {})
    predictions = tg.get("predictions", {})

    return {
        "angular_velocity": float(rot.get("angular_velocity", 0.0)),
        "spiral_score": float(rot.get("spiral_score", 0.0)),
        "curvature": float(rot.get("curvature", 0.0)),
        "axis_lock": float(rot.get("axis_lock", 0.0)),
        "total_displacement": float(rot.get("total_displacement", 0.0)),
        "displacement_variance": float(rot.get("displacement_variance", 0.0)),
        "mean_instability": float(rot.get("mean_instability", 0.0)),
        "plane_coupling_score": float(coupling.get("plane_coupling_score", 0.0)),
        "dimensional_activity": int(coupling.get("dimensional_activity", 0)),
        "convergence_rate": float(gm.get("convergence_rate", 0.0)),
        "system_stability": float(gm.get("system_stability", 0.0)),
        "volatility_score": float(gm.get("volatility_score", 0.0)),
        "topology_type": str(gm.get("topology_type", "unknown")),
        "convergence_prediction": str(predictions.get("convergence", "unknown")),
        "oscillation_prediction": str(predictions.get("oscillation", "unknown")),
        "basin_switch_risk": str(predictions.get("basin_switch_risk", "unknown")),
        "metastable_prediction": str(predictions.get("metastable", "unknown")),
        "control_signal": float(rot.get("control_signal", 0.0)),
    }


# ---------------------------------------------------------------------------
# 2. Rule-Based Scoring (Deterministic)
# ---------------------------------------------------------------------------

# Thresholds for scoring rules.
_HIGH = 0.5
_MODERATE = 0.3
_LOW = 0.15


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _score_oscillatory_trap(f: dict) -> float:
    """Score: high angular_velocity, low convergence_rate, moderate-high stability."""
    s = 0.0
    # High angular velocity is the primary signal.
    if f["angular_velocity"] >= _HIGH:
        s += 0.4
    elif f["angular_velocity"] >= _MODERATE:
        s += 0.2
    # Low convergence rate.
    if f["convergence_rate"] < _LOW:
        s += 0.25
    elif f["convergence_rate"] < _MODERATE:
        s += 0.1
    # Moderate-high stability (trapped, not diverging).
    if f["system_stability"] >= _MODERATE:
        s += 0.2
    # Oscillation prediction reinforcement.
    if f["oscillation_prediction"] == "high":
        s += 0.15
    elif f["oscillation_prediction"] == "moderate":
        s += 0.05
    return round(_clamp(s), ROUND_PRECISION)


def _score_metastable_plateau(f: dict) -> float:
    """Score: low movement, high stability, low curvature."""
    s = 0.0
    # Low total displacement (not moving).
    if f["total_displacement"] < _LOW:
        s += 0.35
    elif f["total_displacement"] < _MODERATE:
        s += 0.15
    # High stability.
    if f["system_stability"] >= _HIGH:
        s += 0.25
    elif f["system_stability"] >= _MODERATE:
        s += 0.1
    # Low curvature.
    if f["curvature"] < _LOW:
        s += 0.2
    elif f["curvature"] < _MODERATE:
        s += 0.1
    # Metastable prediction reinforcement.
    if f["metastable_prediction"] == "likely":
        s += 0.2
    elif f["metastable_prediction"] == "moderate":
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


def _score_basin_switch_instability(f: dict) -> float:
    """Score: high curvature, high coupling, medium stability."""
    s = 0.0
    # High curvature.
    if f["curvature"] >= _HIGH:
        s += 0.3
    elif f["curvature"] >= _MODERATE:
        s += 0.15
    # High coupling.
    if f["plane_coupling_score"] >= _HIGH:
        s += 0.25
    elif f["plane_coupling_score"] >= _MODERATE:
        s += 0.1
    # Medium stability (not fully stable, not diverging).
    stab = f["system_stability"]
    if _LOW <= stab <= 0.7:
        s += 0.2
    # Basin switch risk prediction.
    if f["basin_switch_risk"] == "high":
        s += 0.25
    elif f["basin_switch_risk"] == "medium":
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


def _score_control_overshoot(f: dict) -> float:
    """Score: high control signal, high curvature, oscillatory signals."""
    s = 0.0
    # High control signal.
    if f["control_signal"] >= _HIGH:
        s += 0.3
    elif f["control_signal"] >= _MODERATE:
        s += 0.15
    # High curvature (sharp direction changes from overcorrection).
    if f["curvature"] >= _HIGH:
        s += 0.25
    elif f["curvature"] >= _MODERATE:
        s += 0.1
    # High angular velocity (oscillatory from overshoot).
    if f["angular_velocity"] >= _HIGH:
        s += 0.2
    elif f["angular_velocity"] >= _MODERATE:
        s += 0.1
    # High volatility.
    if f["volatility_score"] >= _HIGH:
        s += 0.15
    elif f["volatility_score"] >= _MODERATE:
        s += 0.05
    # Displacement variance (erratic steps).
    if f["displacement_variance"] >= _MODERATE:
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


def _score_underconstrained_dynamics(f: dict) -> float:
    """Score: low coupling, low axis_lock, high dimensional activity."""
    s = 0.0
    # Low coupling (dimensions not coordinated).
    if f["plane_coupling_score"] < _LOW:
        s += 0.3
    elif f["plane_coupling_score"] < _MODERATE:
        s += 0.15
    # Low axis lock (motion spread across many axes).
    if f["axis_lock"] < _LOW:
        s += 0.25
    elif f["axis_lock"] < _MODERATE:
        s += 0.1
    # High dimensional activity.
    if f["dimensional_activity"] >= 6:
        s += 0.2
    elif f["dimensional_activity"] >= 4:
        s += 0.1
    # Low stability.
    if f["system_stability"] < _LOW:
        s += 0.15
    elif f["system_stability"] < _MODERATE:
        s += 0.05
    # Low convergence.
    if f["convergence_rate"] < _LOW:
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


def _score_slow_convergence(f: dict) -> float:
    """Score: moderate spiral, low convergence rate, moderate stability."""
    s = 0.0
    # Moderate spiral (some inward trend, but slow).
    sp = f["spiral_score"]
    if _LOW <= sp < _HIGH:
        s += 0.3
    elif sp >= _HIGH:
        s += 0.1  # High spiral is more like healthy convergence.
    # Low convergence rate.
    if f["convergence_rate"] < _MODERATE:
        s += 0.25
    elif f["convergence_rate"] < _HIGH:
        s += 0.1
    # Moderate stability.
    stab = f["system_stability"]
    if _MODERATE <= stab < 0.8:
        s += 0.2
    # Convergence prediction.
    if f["convergence_prediction"] == "moderate":
        s += 0.15
    elif f["convergence_prediction"] == "low":
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


def _score_healthy_convergence(f: dict) -> float:
    """Score: high spiral, high convergence rate, high stability."""
    s = 0.0
    # High spiral score.
    if f["spiral_score"] >= _HIGH:
        s += 0.3
    elif f["spiral_score"] >= _MODERATE:
        s += 0.15
    # High convergence rate.
    if f["convergence_rate"] >= 0.8:
        s += 0.3
    elif f["convergence_rate"] >= _HIGH:
        s += 0.15
    # High stability.
    if f["system_stability"] >= 0.8:
        s += 0.2
    elif f["system_stability"] >= _HIGH:
        s += 0.1
    # Low angular velocity (smooth convergence).
    if f["angular_velocity"] < _LOW:
        s += 0.1
    # Convergence prediction.
    if f["convergence_prediction"] == "likely":
        s += 0.1
    return round(_clamp(s), ROUND_PRECISION)


_SCORERS = {
    "oscillatory_trap": _score_oscillatory_trap,
    "metastable_plateau": _score_metastable_plateau,
    "basin_switch_instability": _score_basin_switch_instability,
    "control_overshoot": _score_control_overshoot,
    "underconstrained_dynamics": _score_underconstrained_dynamics,
    "slow_convergence": _score_slow_convergence,
    "healthy_convergence": _score_healthy_convergence,
}


def score_failure_modes(features: dict) -> dict:
    """Score all failure modes using deterministic rule-based scoring.

    Parameters
    ----------
    features : dict
        Output of ``extract_diagnostic_features``.

    Returns
    -------
    dict
        Mapping from failure mode name to score in [0, 1].
    """
    scores: Dict[str, float] = {}
    for mode in FAILURE_MODES:
        scorer = _SCORERS[mode]
        scores[mode] = scorer(features)
    return scores


# ---------------------------------------------------------------------------
# 3. Ranking
# ---------------------------------------------------------------------------


def rank_diagnoses(scores: dict) -> list:
    """Rank diagnoses by score descending, then name ascending for ties.

    Parameters
    ----------
    scores : dict
        Mapping from failure mode name to score.

    Returns
    -------
    list of tuple
        List of (name, score) tuples sorted by score DESC, name ASC.
    """
    items: List[Tuple[str, float]] = []
    for name in sorted(scores.keys()):
        items.append((name, scores[name]))
    # Sort by (-score, name) for deterministic ordering.
    items.sort(key=lambda x: (-x[1], x[0]))
    return items


# ---------------------------------------------------------------------------
# 4. Explanation Layer
# ---------------------------------------------------------------------------

_EXPLANATIONS = {
    "oscillatory_trap": {
        "high_angular_velocity": "high angular velocity ({value:.2f}) indicates persistent rotation",
        "low_convergence_rate": "low convergence rate ({value:.2f}) suggests no inward progress",
        "high_stability": "moderate-high stability ({value:.2f}) indicates trapped state, not divergence",
        "summary": "Persistent rotation without descent toward a fixed point.",
    },
    "metastable_plateau": {
        "low_displacement": "low total displacement ({value:.2f}) indicates minimal movement",
        "high_stability": "high stability ({value:.2f}) suggests a flat energy region",
        "low_curvature": "low curvature ({value:.2f}) confirms flat trajectory",
        "summary": "System stuck on a flat energy plateau with no gradient to follow.",
    },
    "basin_switch_instability": {
        "high_curvature": "high curvature ({value:.2f}) indicates sharp trajectory changes",
        "high_coupling": "high coupling ({value:.2f}) shows correlated multi-axis shifts",
        "basin_risk": "basin switch risk is {value}",
        "summary": "Trajectory crosses basin boundaries causing erratic state jumps.",
    },
    "control_overshoot": {
        "high_control": "high control signal ({value:.2f}) drives overcorrection",
        "high_curvature": "high curvature ({value:.2f}) from sharp direction reversals",
        "oscillatory": "angular velocity ({value:.2f}) confirms oscillatory overcorrection",
        "summary": "Control interventions overshoot the target, causing oscillation.",
    },
    "underconstrained_dynamics": {
        "low_coupling": "low coupling ({value:.2f}) indicates uncoordinated dimensions",
        "low_axis_lock": "low axis lock ({value:.2f}) shows diffuse motion",
        "high_activity": "high dimensional activity ({value}) indicates many active axes",
        "summary": "Dynamics are spread across many unconstrained dimensions.",
    },
    "slow_convergence": {
        "moderate_spiral": "moderate spiral score ({value:.2f}) shows partial inward trend",
        "low_convergence": "low convergence rate ({value:.2f}) indicates slow progress",
        "summary": "System converges but at an impractically slow rate.",
    },
    "healthy_convergence": {
        "high_spiral": "high spiral score ({value:.2f}) shows strong inward motion",
        "high_convergence": "high convergence rate ({value:.2f}) confirms rapid convergence",
        "high_stability": "high stability ({value:.2f}) indicates stable fixed point",
        "summary": "System converges efficiently toward a stable fixed point.",
    },
}


def explain_diagnosis(features: dict, diagnosis: str) -> str:
    """Generate a human-readable explanation for a diagnosis.

    Parameters
    ----------
    features : dict
        Output of ``extract_diagnostic_features``.
    diagnosis : str
        One of the failure mode names from ``FAILURE_MODES``.

    Returns
    -------
    str
        Multi-line explanation string.
    """
    if diagnosis not in _EXPLANATIONS:
        return f"Unknown diagnosis: {diagnosis}"

    templates = _EXPLANATIONS[diagnosis]
    lines: List[str] = []

    if diagnosis == "oscillatory_trap":
        if features["angular_velocity"] >= _MODERATE:
            lines.append(
                "- " + templates["high_angular_velocity"].format(
                    value=features["angular_velocity"]
                )
            )
        if features["convergence_rate"] < _MODERATE:
            lines.append(
                "- " + templates["low_convergence_rate"].format(
                    value=features["convergence_rate"]
                )
            )
        if features["system_stability"] >= _MODERATE:
            lines.append(
                "- " + templates["high_stability"].format(
                    value=features["system_stability"]
                )
            )

    elif diagnosis == "metastable_plateau":
        if features["total_displacement"] < _MODERATE:
            lines.append(
                "- " + templates["low_displacement"].format(
                    value=features["total_displacement"]
                )
            )
        if features["system_stability"] >= _MODERATE:
            lines.append(
                "- " + templates["high_stability"].format(
                    value=features["system_stability"]
                )
            )
        if features["curvature"] < _MODERATE:
            lines.append(
                "- " + templates["low_curvature"].format(
                    value=features["curvature"]
                )
            )

    elif diagnosis == "basin_switch_instability":
        if features["curvature"] >= _MODERATE:
            lines.append(
                "- " + templates["high_curvature"].format(
                    value=features["curvature"]
                )
            )
        if features["plane_coupling_score"] >= _MODERATE:
            lines.append(
                "- " + templates["high_coupling"].format(
                    value=features["plane_coupling_score"]
                )
            )
        lines.append(
            "- " + templates["basin_risk"].format(
                value=features["basin_switch_risk"]
            )
        )

    elif diagnosis == "control_overshoot":
        if features["control_signal"] >= _MODERATE:
            lines.append(
                "- " + templates["high_control"].format(
                    value=features["control_signal"]
                )
            )
        if features["curvature"] >= _MODERATE:
            lines.append(
                "- " + templates["high_curvature"].format(
                    value=features["curvature"]
                )
            )
        if features["angular_velocity"] >= _MODERATE:
            lines.append(
                "- " + templates["oscillatory"].format(
                    value=features["angular_velocity"]
                )
            )

    elif diagnosis == "underconstrained_dynamics":
        if features["plane_coupling_score"] < _MODERATE:
            lines.append(
                "- " + templates["low_coupling"].format(
                    value=features["plane_coupling_score"]
                )
            )
        if features["axis_lock"] < _MODERATE:
            lines.append(
                "- " + templates["low_axis_lock"].format(
                    value=features["axis_lock"]
                )
            )
        if features["dimensional_activity"] >= 4:
            lines.append(
                "- " + templates["high_activity"].format(
                    value=features["dimensional_activity"]
                )
            )

    elif diagnosis == "slow_convergence":
        if features["spiral_score"] >= _LOW:
            lines.append(
                "- " + templates["moderate_spiral"].format(
                    value=features["spiral_score"]
                )
            )
        if features["convergence_rate"] < _HIGH:
            lines.append(
                "- " + templates["low_convergence"].format(
                    value=features["convergence_rate"]
                )
            )

    elif diagnosis == "healthy_convergence":
        if features["spiral_score"] >= _MODERATE:
            lines.append(
                "- " + templates["high_spiral"].format(
                    value=features["spiral_score"]
                )
            )
        if features["convergence_rate"] >= _HIGH:
            lines.append(
                "- " + templates["high_convergence"].format(
                    value=features["convergence_rate"]
                )
            )
        if features["system_stability"] >= _HIGH:
            lines.append(
                "- " + templates["high_stability"].format(
                    value=features["system_stability"]
                )
            )

    summary = templates["summary"]

    if lines:
        return "\n".join(lines) + "\n" + summary
    return summary


# ---------------------------------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------------------------------


def run_differential_diagnosis(
    result: dict,
    *,
    refined_thresholds: dict | None = None,
) -> dict:
    """Run the full differential diagnosis pipeline.

    Pipeline:
        system_diagnostics + trajectory_geometry
        -> features -> scores -> ranked diagnoses -> explanations

    Parameters
    ----------
    result : dict
        Output of ``run_system_diagnostics`` (or a dict containing
        ``global_metrics`` and ``trajectory_geometry`` keys).
    refined_thresholds : dict, optional
        Refined thresholds from ``diagnosis_refinement.refine_diagnosis_rules``.
        If provided, adjusts scoring sensitivity.  If None, uses defaults.

    Returns
    -------
    dict
        Complete diagnosis with keys:
        ``features``, ``scores``, ``ranked``, ``primary_diagnosis``,
        ``diagnosis_confidence``, ``explanations``.
    """
    features = extract_diagnostic_features(result)

    if refined_thresholds is not None:
        from qec.analysis.diagnosis_refinement import apply_refined_thresholds
        scores = apply_refined_thresholds(features, refined_thresholds)
    else:
        scores = score_failure_modes(features)

    ranked = rank_diagnoses(scores)

    # Primary diagnosis is the top-ranked.
    primary_diagnosis = ranked[0][0] if ranked else "unknown"
    diagnosis_confidence = ranked[0][1] if ranked else 0.0

    # Generate explanations for all modes with nonzero scores.
    explanations: Dict[str, str] = {}
    for name, score in ranked:
        if score > 0.0:
            explanations[name] = explain_diagnosis(features, name)

    return {
        "features": features,
        "scores": scores,
        "ranked": ranked,
        "primary_diagnosis": primary_diagnosis,
        "diagnosis_confidence": round(diagnosis_confidence, ROUND_PRECISION),
        "explanations": explanations,
    }


# ---------------------------------------------------------------------------
# 6. Formatter
# ---------------------------------------------------------------------------


def format_differential_diagnosis(result: dict) -> str:
    """Format differential diagnosis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_differential_diagnosis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    ranked = result.get("ranked", [])
    primary = result.get("primary_diagnosis", "unknown")
    explanations = result.get("explanations", {})

    lines.append("")
    lines.append("=== Differential Diagnosis ===")
    lines.append("")

    for i, (name, score) in enumerate(ranked):
        if score > 0.0:
            lines.append(f"{i + 1}. {_format_mode_name(name)} ({score:.2f})")

    lines.append("")
    lines.append("Primary Diagnosis:")
    lines.append(f"  {_format_mode_name(primary)}")

    if primary in explanations:
        lines.append("")
        lines.append("Explanation:")
        for line in explanations[primary].split("\n"):
            lines.append(f"  {line}")

    return "\n".join(lines)


def _format_mode_name(mode: str) -> str:
    """Format a failure mode name for display."""
    return mode.replace("_", " ").title()
