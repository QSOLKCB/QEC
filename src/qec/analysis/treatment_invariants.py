"""v104.5.0 — Treatment invariant discovery.

Identifies deterministic invariants and quasi-invariants that persist
across diagnosis, provocation, and treatment.

Invariant types:
    - sign: a metric always moves in one direction
    - ordering: a relationship between metrics holds throughout
    - topology: structural property preserved across interventions
    - diagnostic: diagnosis identity preserved
    - geometry: geometric metric monotonicity
    - control: control metric monotonicity

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
# Invariant types
# ---------------------------------------------------------------------------

INVARIANT_TYPES = [
    "sign",
    "ordering",
    "topology",
    "diagnostic",
    "geometry",
    "control",
]


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1. Invariant extraction
# ---------------------------------------------------------------------------


def extract_treatment_invariants(
    diagnosis_result: dict,
    provocation_result: dict,
    treatment_result: dict,
) -> dict:
    """Extract deterministic invariants across the treatment pipeline.

    Checks for invariants that hold across the analyzed sequence:
    diagnosis -> provocation -> treatment.

    Parameters
    ----------
    diagnosis_result : dict
        Output of ``run_differential_diagnosis``.
    provocation_result : dict
        Output of ``run_provocation_analysis``.
    treatment_result : dict
        Output of ``run_treatment_planning``.

    Returns
    -------
    dict
        Contains ``invariants`` (list of invariant dicts), each with
        ``name``, ``type``, ``description``, ``holds``, ``support``.
    """
    invariants: List[Dict[str, Any]] = []

    # --- Sign invariants ---
    invariants.extend(_check_sign_invariants(provocation_result, treatment_result))

    # --- Ordering invariants ---
    invariants.extend(_check_ordering_invariants(provocation_result, treatment_result))

    # --- Topology invariants ---
    invariants.extend(_check_topology_invariants(provocation_result, treatment_result))

    # --- Diagnostic invariants ---
    invariants.extend(
        _check_diagnostic_invariants(diagnosis_result, provocation_result, treatment_result)
    )

    # --- Geometry invariants ---
    invariants.extend(_check_geometry_invariants(provocation_result, treatment_result))

    # --- Control invariants ---
    invariants.extend(_check_control_invariants(provocation_result, treatment_result))

    # Filter to only invariants that hold.
    held = [inv for inv in invariants if inv.get("holds", False)]

    return {"invariants": held}


def _check_sign_invariants(
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check sign invariants: metrics that always move in one direction."""
    invariants: List[Dict[str, Any]] = []

    # Check if basin_switch_risk (transient weight) always decreases
    # under successful treatment.
    prov_response = provocation_result.get("response", {})
    prov_deltas = prov_response.get("deltas", {})
    prov_class = prov_response.get("classification", "unchanged")

    best = treatment_result.get("best_treatment", {})
    best_agg = best.get("aggregate_deltas", {})

    prov_d_trans = prov_deltas.get("transient_weight", 0.0)
    treat_d_trans = best_agg.get("delta_transient_weight", 0.0)

    # Basin-switch suppression: transient never increases under
    # successful interventions.
    support = 0
    total = 0

    if prov_class in ("improved", "unchanged"):
        total += 1
        if prov_d_trans <= 0.0:
            support += 1

    if best:
        total += 1
        if treat_d_trans <= 0.0:
            support += 1

    holds = support == total and total > 0

    invariants.append({
        "name": "basin_switch_suppression",
        "type": "sign",
        "description": (
            "Basin switch risk never increased under accepted treatments."
        ),
        "holds": holds,
        "support": support,
        "total": total,
    })

    return invariants


def _check_ordering_invariants(
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check ordering invariants: relationships between metrics."""
    invariants: List[Dict[str, Any]] = []

    # Check: attractor_weight > transient_weight in all improving states.
    best = treatment_result.get("best_treatment", {})
    post = best.get("post_metrics", {})

    prov_intervention = provocation_result.get("intervention_result", {})
    prov_after = prov_intervention.get("after", {})

    support = 0
    total = 0

    # Check in provocation after-state.
    for name in sorted(prov_after.keys()):
        sv = prov_after[name]
        m = sv.get("membership", {})
        attr = m.get("strong_attractor", 0.0) + m.get("weak_attractor", 0.0)
        trans = m.get("transient", 0.0)
        total += 1
        if attr >= trans:
            support += 1

    # Check in treatment post-metrics.
    if post:
        total += 1
        if post.get("attractor_weight", 0.0) >= post.get("transient_weight", 0.0):
            support += 1

    holds = support == total and total > 0

    invariants.append({
        "name": "attractor_dominance",
        "type": "ordering",
        "description": (
            "Attractor weight remains greater than or equal to transient "
            "weight across all improving states."
        ),
        "holds": holds,
        "support": support,
        "total": total,
    })

    return invariants


def _check_topology_invariants(
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check topology invariants: structural properties preserved."""
    invariants: List[Dict[str, Any]] = []

    # Check: response classification is never 'destabilized' across
    # both provocation and best treatment.
    prov_class = provocation_result.get("response", {}).get("classification", "unchanged")

    best = treatment_result.get("best_treatment", {})
    best_score = best.get("score", 0.0)

    support = 0
    total = 0

    total += 1
    if prov_class != "destabilized":
        support += 1

    # If best treatment has positive score, topology preserved.
    if best:
        total += 1
        if best_score > 0.0:
            support += 1

    holds = support == total and total > 0

    invariants.append({
        "name": "topology_preservation",
        "type": "topology",
        "description": (
            "System topology remains non-destabilized despite intervention."
        ),
        "holds": holds,
        "support": support,
        "total": total,
    })

    return invariants


def _check_diagnostic_invariants(
    diagnosis_result: dict,
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check diagnostic invariants: diagnosis identity preserved."""
    invariants: List[Dict[str, Any]] = []

    original_diag = diagnosis_result.get("primary_diagnosis", "unknown")
    revised = provocation_result.get("revision", {}).get(
        "revised_diagnosis", "unknown"
    )
    treatment_diag = treatment_result.get("revised_diagnosis", {}).get(
        "revised_diagnosis", "unknown"
    )

    support = 0
    total = 2

    if original_diag == revised:
        support += 1
    if original_diag == treatment_diag:
        support += 1

    holds = support == total

    invariants.append({
        "name": "diagnosis_persistence",
        "type": "diagnostic",
        "description": (
            f"Primary diagnosis remained {original_diag} across "
            "provocation and treatment."
        ),
        "holds": holds,
        "support": support,
        "total": total,
    })

    return invariants


def _check_geometry_invariants(
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check geometry invariants: geometric metric monotonicity."""
    invariants: List[Dict[str, Any]] = []

    # Check: stability never decreases across provocation and treatment.
    prov_deltas = provocation_result.get("response", {}).get("deltas", {})
    best = treatment_result.get("best_treatment", {})
    best_agg = best.get("aggregate_deltas", {})

    prov_d_stab = prov_deltas.get("stability", 0.0)
    treat_d_stab = best_agg.get("delta_stability", 0.0)

    support = 0
    total = 0

    if prov_deltas:
        total += 1
        if prov_d_stab >= 0.0:
            support += 1

    if best_agg:
        total += 1
        if treat_d_stab >= 0.0:
            support += 1

    holds = support == total and total > 0

    invariants.append({
        "name": "stability_monotonicity",
        "type": "geometry",
        "description": (
            "Stability increased at every successful intervention step."
        ),
        "holds": holds,
        "support": support,
        "total": total,
    })

    return invariants


def _check_control_invariants(
    provocation_result: dict,
    treatment_result: dict,
) -> list:
    """Check control invariants: score improvements correlate with stability."""
    invariants: List[Dict[str, Any]] = []

    best = treatment_result.get("best_treatment", {})
    best_score = best.get("score", 0.0)
    best_agg = best.get("aggregate_deltas", {})

    # Control invariant: when best treatment score is positive,
    # stability delta is non-negative.
    holds = False
    support = 0
    total = 0

    if best and best_score > 0.0:
        total += 1
        d_stab = best_agg.get("delta_stability", 0.0)
        if d_stab >= 0.0:
            support += 1
        holds = support == total
    elif not best:
        # No treatment = trivially holds.
        holds = True
        support = 0
        total = 0

    invariants.append({
        "name": "control_stability_correlation",
        "type": "control",
        "description": (
            "Stability never decreases when best treatment score improves."
        ),
        "holds": holds,
        "support": support,
        "total": max(total, 0),
    })

    return invariants


# ---------------------------------------------------------------------------
# 2. Invariant scoring
# ---------------------------------------------------------------------------


def score_invariants(invariants: dict) -> dict:
    """Score each invariant by confidence and support.

    Parameters
    ----------
    invariants : dict
        Output of ``extract_treatment_invariants``.

    Returns
    -------
    dict
        Contains ``scored_invariants`` (list of dicts), each with
        ``name``, ``type``, ``description``, ``strength``, ``support``.
    """
    inv_list = invariants.get("invariants", [])
    scored: List[Dict[str, Any]] = []

    for inv in inv_list:
        support = int(inv.get("support", 0))
        total = int(inv.get("total", 1))
        total = max(total, 1)

        strength = _round(_clamp(support / total))

        scored.append({
            "name": inv.get("name", "unknown"),
            "type": inv.get("type", "unknown"),
            "description": inv.get("description", ""),
            "strength": strength,
            "support": support,
            "total": total,
        })

    # Sort by strength DESC, name ASC for determinism.
    scored.sort(key=lambda x: (-x["strength"], x["name"]))

    return {"scored_invariants": scored}


# ---------------------------------------------------------------------------
# 3. Invariant formatter
# ---------------------------------------------------------------------------


def format_treatment_invariants(result: dict) -> str:
    """Format treatment invariants as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``score_invariants`` or a dict containing
        ``scored_invariants``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    scored = result.get("scored_invariants", [])

    lines.append("")
    lines.append("=== Treatment Invariants ===")

    if not scored:
        lines.append("")
        lines.append("No invariants discovered.")
        return "\n".join(lines)

    for i, inv in enumerate(scored):
        lines.append("")
        lines.append(f"{i + 1}. {_format_invariant_name(inv['name'])}")
        lines.append(f"   {inv['description']}")
        lines.append(f"   Strength: {inv['strength']:.2f}")

    return "\n".join(lines)


def _format_invariant_name(name: str) -> str:
    """Format an invariant name for display."""
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# 4. Full invariant pipeline
# ---------------------------------------------------------------------------


def run_treatment_invariant_analysis(runs: list) -> dict:
    """Run the full treatment invariant analysis pipeline.

    Pipeline:
        system diagnostics
        -> differential diagnosis
        -> provocation analysis
        -> treatment planning
        -> invariant extraction

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.

    Returns
    -------
    dict
        Contains ``diagnosis``, ``provocation``, ``treatment``,
        ``invariants``, ``scored_invariants``.
    """
    from qec.analysis.differential_diagnosis import run_differential_diagnosis
    from qec.analysis.provocation_analysis import run_provocation_analysis
    from qec.analysis.treatment_planning import run_treatment_planning
    from qec.analysis.system_diagnostics import run_system_diagnostics

    sys_diag = run_system_diagnostics(runs)

    diagnosis_input = {
        "global_metrics": sys_diag.get("global_metrics", {}),
        "trajectory_geometry": sys_diag.get("trajectory_geometry", {}),
    }
    diagnosis = run_differential_diagnosis(diagnosis_input)

    provocation = run_provocation_analysis(runs)
    treatment = run_treatment_planning(runs)

    raw_invariants = extract_treatment_invariants(
        diagnosis, provocation, treatment,
    )
    scored = score_invariants(raw_invariants)

    return {
        "diagnosis": diagnosis,
        "provocation": provocation,
        "treatment": treatment,
        "invariants": raw_invariants,
        "scored_invariants": scored.get("scored_invariants", []),
    }


__all__ = [
    "INVARIANT_TYPES",
    "ROUND_PRECISION",
    "extract_treatment_invariants",
    "format_treatment_invariants",
    "run_treatment_invariant_analysis",
    "score_invariants",
]
