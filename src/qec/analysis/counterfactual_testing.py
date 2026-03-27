"""v105.2.0 — Counterfactual and adversarial testing for law validation.

Implements adversarial law testing pipeline:
    laws → target selection → counterfactual generation → violation evaluation
    → stress testing → robustness update → diagnosis refinement

Uses Go-inspired region-aware targeting and Wu Wei minimal violation
attempts before escalation.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Law fragility threshold — laws below this stability are prioritized
_FRAGILITY_THRESHOLD = 0.6

# Minimum runs for stress test significance
_MIN_STRESS_RUNS = 3

# Violation severity levels
_SEVERITY_LOW = 0.3
_SEVERITY_HIGH = 0.7

# Confidence decay per violation
_CONFIDENCE_DECAY = 0.1

# Escalation strengths for adversarial probes
_ADVERSARIAL_STRENGTHS = (0.1, 0.2, 0.4)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Extract float from value, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Extract int from value, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Target law selection
# ---------------------------------------------------------------------------


def select_target_laws(
    registry: dict,
    laws: List[dict],
) -> List[str]:
    """Select laws to target for adversarial testing.

    Prioritizes:
    - fragile laws (low stability_score)
    - emerging laws (low count, recently promoted)
    - drifting invariants (high break_count relative to count)

    Parameters
    ----------
    registry : dict
        Invariant registry for historical context.
    laws : list of dict
        Current law set.

    Returns
    -------
    list of str
        Sorted list of law names/keys to target.
    """
    if not isinstance(laws, list):
        return []

    scored: List[Tuple[float, str]] = []

    for law in laws:
        if not isinstance(law, dict):
            continue
        name = str(law.get("name", law.get("key", "unknown")))
        stability_score = _safe_float(law.get("stability_score", 1.0))
        count = _safe_int(law.get("count", 0))

        # Compute targeting priority (higher = more targeted)
        priority = 0.0

        # Fragile laws
        if stability_score < _FRAGILITY_THRESHOLD:
            priority += (1.0 - stability_score) * 0.4

        # Emerging laws (recently promoted, low evidence)
        if count < 5:
            priority += (5 - count) * 0.1

        # Check registry for drift
        if isinstance(registry, dict):
            reg_entry = registry.get(name, {})
            if isinstance(reg_entry, dict):
                break_count = _safe_int(reg_entry.get("break_count", 0))
                total_count = max(_safe_int(reg_entry.get("count", 1)), 1)
                drift_ratio = break_count / total_count
                priority += drift_ratio * 0.3

        scored.append((_round(priority), name))

    # Sort by priority descending, then name for determinism
    scored.sort(key=lambda x: (-x[0], x[1]))

    return [name for _, name in scored if _ > 0.0]


# ---------------------------------------------------------------------------
# 2. Counterfactual generation (adversarial probes)
# ---------------------------------------------------------------------------


def generate_counterfactuals(
    law: dict,
    influence_map: dict,
) -> List[dict]:
    """Generate adversarial counterfactual probes for a law.

    Uses region-aware targeting (Go-style) and minimal violation
    attempts first (Wu Wei), escalating if the invariant holds.

    Parameters
    ----------
    law : dict
        Law to test.  Must contain ``conditions`` list.
    influence_map : dict
        Output of ``compute_influence_map`` for region awareness.

    Returns
    -------
    list of dict
        Counterfactual probes, each containing:
        - ``target_condition``: dict — the condition being tested
        - ``perturbation_strength``: float
        - ``perturbation_direction``: str — "violate" or "boundary"
        - ``target_region``: str — region type being targeted
    """
    conditions = law.get("conditions", [])
    if not isinstance(conditions, list):
        conditions = []

    # Identify target regions from influence map
    nodes = influence_map.get("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}

    # Determine which regions are most susceptible
    contested_nodes = []
    unstable_nodes = []
    for node_id in sorted(nodes.keys()):
        node = nodes[node_id]
        if not isinstance(node, dict):
            continue
        sensitivity = _safe_float(node.get("control_sensitivity", 0.0))
        pressure = _safe_float(node.get("instability_pressure", 0.0))
        if sensitivity >= 0.6 and pressure >= 0.5:
            contested_nodes.append(node_id)
        elif pressure >= 0.5:
            unstable_nodes.append(node_id)

    # Determine target region type
    if contested_nodes:
        target_region = "contested"
    elif unstable_nodes:
        target_region = "unstable"
    else:
        target_region = "neutral"

    counterfactuals: List[dict] = []

    for condition in conditions:
        if not isinstance(condition, dict):
            continue

        # Generate probes at escalating strengths (Wu Wei: start minimal)
        for strength in _ADVERSARIAL_STRENGTHS:
            counterfactuals.append({
                "target_condition": dict(condition),
                "perturbation_strength": _round(strength),
                "perturbation_direction": "violate",
                "target_region": target_region,
            })

        # Also test at boundary
        counterfactuals.append({
            "target_condition": dict(condition),
            "perturbation_strength": _round(_ADVERSARIAL_STRENGTHS[0]),
            "perturbation_direction": "boundary",
            "target_region": target_region,
        })

    return counterfactuals


# ---------------------------------------------------------------------------
# 3. Law violation evaluation
# ---------------------------------------------------------------------------


def evaluate_law_violation(
    before: dict,
    after: dict,
    law: dict,
) -> dict:
    """Evaluate whether a law was violated by a state transition.

    Parameters
    ----------
    before : dict
        System metrics before perturbation.
    after : dict
        System metrics after perturbation.
    law : dict
        The law being tested (must contain ``conditions``).

    Returns
    -------
    dict
        Evaluation with keys: ``violated`` (bool), ``severity`` (float),
        ``confidence_shift`` (float), ``violated_conditions`` (list).
    """
    conditions = law.get("conditions", [])
    if not isinstance(conditions, list):
        conditions = []

    violated_conditions: List[dict] = []
    total_severity = 0.0

    for condition in conditions:
        if not isinstance(condition, dict):
            continue

        metric = str(condition.get("metric", ""))
        operator = str(condition.get("operator", ""))
        threshold = _safe_float(condition.get("value", 0.0))

        # Get metric value from after-state
        after_value = _safe_float(after.get(metric, 0.0))

        # Evaluate condition
        holds = _evaluate_condition(after_value, operator, threshold)

        if not holds:
            # Compute severity based on distance from threshold
            distance = abs(after_value - threshold)
            severity = _round(_clamp(distance))
            violated_conditions.append({
                "metric": metric,
                "operator": operator,
                "threshold": _round(threshold),
                "actual_value": _round(after_value),
                "severity": severity,
            })
            total_severity += severity

    n_conditions = max(len(conditions), 1)
    avg_severity = _round(total_severity / n_conditions)
    violated = len(violated_conditions) > 0

    # Confidence shift: negative if violated, positive if held
    if violated:
        confidence_shift = _round(-_CONFIDENCE_DECAY * avg_severity)
    else:
        confidence_shift = _round(_CONFIDENCE_DECAY * 0.5)

    return {
        "violated": violated,
        "severity": avg_severity,
        "confidence_shift": confidence_shift,
        "violated_conditions": violated_conditions,
    }


def _evaluate_condition(value: float, operator: str, threshold: float) -> bool:
    """Evaluate a single condition. Deterministic."""
    if operator == "gt":
        return value > threshold
    elif operator == "gte":
        return value >= threshold
    elif operator == "lt":
        return value < threshold
    elif operator == "lte":
        return value <= threshold
    elif operator == "eq":
        return abs(value - threshold) < 1e-12
    elif operator == "neq":
        return abs(value - threshold) >= 1e-12
    return True  # Unknown operator → condition holds (conservative)


# ---------------------------------------------------------------------------
# 4. Stress testing pipeline
# ---------------------------------------------------------------------------


def run_law_stress_test(
    runs: List[dict],
    registry: dict,
    laws: List[dict],
) -> dict:
    """Run stress testing across multiple run results.

    Parameters
    ----------
    runs : list of dict
        List of run result dicts, each containing system metrics.
    registry : dict
        Invariant registry.
    laws : list of dict
        Laws to stress test.

    Returns
    -------
    dict
        Stress test results with keys:
        - ``law_results``: dict mapping law_name -> test results
        - ``summary``: aggregate stress test statistics
    """
    if not isinstance(runs, list) or not isinstance(laws, list):
        return {"law_results": {}, "summary": _empty_stress_summary()}

    law_results: Dict[str, Dict[str, Any]] = {}

    for law in laws:
        if not isinstance(law, dict):
            continue
        name = str(law.get("name", law.get("key", "unknown")))

        violations = 0
        total_severity = 0.0
        n_tested = 0

        for i, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            # Use consecutive run pairs for before/after
            if i == 0:
                continue
            before = runs[i - 1]
            after = run

            result = evaluate_law_violation(before, after, law)
            n_tested += 1
            if result.get("violated", False):
                violations += 1
                total_severity += _safe_float(result.get("severity", 0.0))

        violation_rate = _round(violations / max(n_tested, 1))
        avg_severity = _round(total_severity / max(violations, 1))

        law_results[name] = {
            "violations": violations,
            "tests": n_tested,
            "violation_rate": violation_rate,
            "avg_severity": avg_severity,
            "robust": violations == 0 and n_tested >= _MIN_STRESS_RUNS,
        }

    summary = _compute_stress_summary(law_results)

    return {
        "law_results": law_results,
        "summary": summary,
    }


def _empty_stress_summary() -> dict:
    """Return empty stress test summary."""
    return {
        "total_laws_tested": 0,
        "robust_laws": 0,
        "fragile_laws": 0,
        "law_violation_rate": 0.0,
    }


def _compute_stress_summary(law_results: Dict[str, Dict[str, Any]]) -> dict:
    """Compute aggregate stress test summary."""
    if not law_results:
        return _empty_stress_summary()

    total = len(law_results)
    robust = 0
    fragile = 0
    total_violation_rate = 0.0

    for name in sorted(law_results.keys()):
        result = law_results[name]
        if result.get("robust", False):
            robust += 1
        if _safe_float(result.get("violation_rate", 0.0)) > 0.0:
            fragile += 1
        total_violation_rate += _safe_float(result.get("violation_rate", 0.0))

    return {
        "total_laws_tested": total,
        "robust_laws": robust,
        "fragile_laws": fragile,
        "law_violation_rate": _round(total_violation_rate / max(total, 1)),
    }


# ---------------------------------------------------------------------------
# 5. Law robustness update
# ---------------------------------------------------------------------------


def update_law_robustness(
    laws: List[dict],
    stress_results: dict,
) -> List[dict]:
    """Update law robustness scores from stress test results.

    Returns new list of laws (no mutation of inputs).

    Parameters
    ----------
    laws : list of dict
        Current law set.
    stress_results : dict
        Output of ``run_law_stress_test``.

    Returns
    -------
    list of dict
        Updated laws with revised ``stability_score``, ``break_count``,
        and ``classification``.
    """
    law_results = stress_results.get("law_results", {})
    if not isinstance(law_results, dict):
        law_results = {}

    updated: List[dict] = []
    for law in laws:
        if not isinstance(law, dict):
            continue
        name = str(law.get("name", law.get("key", "unknown")))
        new_law = dict(law)

        result = law_results.get(name)
        if isinstance(result, dict):
            violation_rate = _safe_float(result.get("violation_rate", 0.0))
            new_stability = _round(_clamp(
                _safe_float(law.get("stability_score", 1.0)) * (1.0 - violation_rate)
            ))
            new_law["stability_score"] = new_stability
            new_law["break_count"] = (
                _safe_int(law.get("break_count", 0))
                + _safe_int(result.get("violations", 0))
            )

            # Reclassify
            if new_stability >= 0.8:
                new_law["classification"] = "robust"
            elif new_stability >= 0.5:
                new_law["classification"] = "moderate"
            else:
                new_law["classification"] = "fragile"

        updated.append(new_law)

    return updated


# ---------------------------------------------------------------------------
# 6. Diagnosis refinement from counterfactual results
# ---------------------------------------------------------------------------


def refine_diagnosis_from_counterfactuals(
    diagnosis: dict,
    results: List[dict],
) -> dict:
    """Refine a diagnosis based on counterfactual test results.

    If counterfactuals reveal violations, the diagnosis confidence
    is adjusted and new failure modes may be identified.

    Parameters
    ----------
    diagnosis : dict
        Current diagnosis with ``ranked_diagnoses``.
    results : list of dict
        List of counterfactual evaluation results (from
        ``evaluate_law_violation``).

    Returns
    -------
    dict
        Refined diagnosis with adjusted scores and optional
        ``counterfactual_insights``.
    """
    if not isinstance(diagnosis, dict):
        return {"ranked_diagnoses": [], "counterfactual_insights": []}

    ranked = diagnosis.get("ranked_diagnoses", [])
    if not isinstance(ranked, list):
        ranked = []

    # Compute aggregate counterfactual signal
    total_violations = 0
    total_severity = 0.0
    if isinstance(results, list):
        for r in results:
            if isinstance(r, dict) and r.get("violated", False):
                total_violations += 1
                total_severity += _safe_float(r.get("severity", 0.0))

    # Adjust diagnosis scores based on counterfactual evidence
    adjusted: List[dict] = []
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        new_entry = dict(entry)
        old_score = _safe_float(entry.get("score", 0.0))

        if total_violations > 0:
            # Laws being violated → increase instability-related diagnosis scores
            mode = str(entry.get("failure_mode", ""))
            if mode in ("oscillatory_trap", "basin_switch_instability"):
                new_entry["score"] = _round(_clamp(
                    old_score + total_severity * 0.1
                ))
            elif mode == "healthy_convergence":
                new_entry["score"] = _round(_clamp(
                    old_score - total_severity * 0.1
                ))
        adjusted.append(new_entry)

    # Re-sort by score descending, then failure_mode for determinism
    adjusted.sort(key=lambda x: (-_safe_float(x.get("score", 0.0)),
                                  str(x.get("failure_mode", ""))))

    insights: List[dict] = []
    if total_violations > 0:
        insights.append({
            "type": "law_violation_detected",
            "violation_count": total_violations,
            "total_severity": _round(total_severity),
            "recommendation": "escalate_intervention",
        })

    return {
        "ranked_diagnoses": adjusted,
        "counterfactual_insights": insights,
    }
