"""v104.5.0 — Treatment planning for diagnosis-driven intervention selection.

Implements the treatment planning pipeline:
    differential diagnosis
    -> provocation analysis
    -> revised diagnosis
    -> treatment candidate generation
    -> treatment evaluation
    -> best treatment selection

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
# Constants
# ---------------------------------------------------------------------------

# Valid treatment actions (from control_layer).
VALID_ACTIONS = ("boost_stability", "reduce_escape", "force_transition")

# Treatment strength levels.
TREATMENT_STRENGTHS = (0.2, 0.4, 0.6)

# Diagnosis -> candidate action mapping.
_DIAGNOSIS_ACTIONS: Dict[str, List[str]] = {
    "oscillatory_trap": ["reduce_escape", "boost_stability"],
    "metastable_plateau": ["force_transition", "boost_stability"],
    "basin_switch_instability": ["boost_stability", "reduce_escape"],
    "control_overshoot": ["reduce_escape", "boost_stability"],
    "underconstrained_dynamics": ["boost_stability"],
    "slow_convergence": ["boost_stability", "force_transition"],
    "healthy_convergence": [],
}

# Scoring weights for treatment objective.
_SCORE_WEIGHTS = {
    "stability": 0.30,
    "attractor_weight": 0.30,
    "transient_weight": -0.20,  # penalize
    "basin_switch_risk": -0.10,  # penalize
    "angular_velocity": -0.10,  # penalize
}


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1. Treatment candidate generation
# ---------------------------------------------------------------------------


def generate_treatment_candidates(diagnosis: dict) -> list:
    """Generate deterministic treatment candidates from a diagnosis.

    Produces candidates by combining relevant actions with standard
    strength levels.  Candidate set is small and explainable.

    Parameters
    ----------
    diagnosis : dict
        Output of ``run_differential_diagnosis`` or a revised diagnosis
        dict.  Must contain ``primary_diagnosis`` or ``revised_diagnosis``.

    Returns
    -------
    list of dict
        Each dict has ``action``, ``strength``, and ``rationale`` keys.
    """
    primary = str(
        diagnosis.get("revised_diagnosis",
                       diagnosis.get("primary_diagnosis", "healthy_convergence"))
    )
    actions = _DIAGNOSIS_ACTIONS.get(primary, [])

    candidates: List[Dict[str, Any]] = []
    for action in actions:
        for strength in TREATMENT_STRENGTHS:
            candidates.append({
                "action": action,
                "strength": strength,
                "rationale": f"{action} at {strength} for {primary}",
            })

    return candidates


# ---------------------------------------------------------------------------
# 2. Treatment evaluation
# ---------------------------------------------------------------------------


def evaluate_treatments(
    runs: list,
    candidates: list,
) -> list:
    """Evaluate treatment candidates on the current system state.

    For each candidate, applies the intervention using the control layer,
    computes resulting state and deltas.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    candidates : list of dict
        Each must have ``action`` and ``strength``.

    Returns
    -------
    list of dict
        Each dict contains ``candidate``, ``deltas``, ``post_state``,
        and ``score``.
    """
    from qec.analysis.strategy_adapter import run_multistate_analysis
    from qec.analysis.control_layer import (
        apply_intervention,
        evaluate_intervention,
    )

    multistate_result = run_multistate_analysis(runs)
    before = multistate_result.get("multistate", {})

    results: List[Dict[str, Any]] = []

    for candidate in candidates:
        # Deep copy and apply intervention to all strategies.
        after: Dict[str, Dict[str, Any]] = {}
        for name in sorted(before.keys()):
            sv = before[name]
            state = {
                "ternary": dict(sv.get("ternary", {})),
                "membership": dict(sv.get("membership", {})),
            }
            rule = {
                "target": name,
                "action": str(candidate.get("action", "")),
                "strength": float(candidate.get("strength", 0.0)),
            }
            after[name] = apply_intervention(state, rule)

        deltas = evaluate_intervention(before, after)

        # Compute aggregate metrics for scoring.
        agg = _aggregate_deltas(deltas)
        post_metrics = _compute_post_metrics(before, after)

        sc = score_treatment({
            "aggregate_deltas": agg,
            "post_metrics": post_metrics,
        })

        results.append({
            "candidate": {
                "action": str(candidate.get("action", "")),
                "strength": float(candidate.get("strength", 0.0)),
                "rationale": str(candidate.get("rationale", "")),
            },
            "deltas": deltas,
            "post_metrics": post_metrics,
            "aggregate_deltas": agg,
            "score": sc,
        })

    return results


def _aggregate_deltas(deltas: dict) -> dict:
    """Aggregate per-strategy deltas into system-level averages."""
    if not deltas:
        return {
            "delta_stability": 0.0,
            "delta_attractor_weight": 0.0,
            "delta_transient_weight": 0.0,
        }

    total_stab = 0.0
    total_attr = 0.0
    total_trans = 0.0
    count = 0

    for name in sorted(deltas.keys()):
        d = deltas[name]
        total_stab += d.get("delta_stability", 0.0)
        total_attr += d.get("delta_attractor_weight", 0.0)
        total_trans += d.get("delta_transient_weight", 0.0)
        count += 1

    count = max(count, 1)
    return {
        "delta_stability": _round(total_stab / count),
        "delta_attractor_weight": _round(total_attr / count),
        "delta_transient_weight": _round(total_trans / count),
    }


def _compute_post_metrics(before: dict, after: dict) -> dict:
    """Compute aggregate post-intervention metrics."""
    if not after:
        return {
            "stability": 0.0,
            "attractor_weight": 0.0,
            "transient_weight": 0.0,
        }

    total_stab = 0.0
    total_attr = 0.0
    total_trans = 0.0
    count = 0

    for name in sorted(after.keys()):
        sv = after[name]
        t = sv.get("ternary", {})
        m = sv.get("membership", {})

        # Map stability_state from {-1, 0, +1} to {0.0, 0.5, 1.0}.
        total_stab += (t.get("stability_state", 0) + 1) / 2.0
        total_attr += m.get("strong_attractor", 0.0) + m.get("weak_attractor", 0.0)
        total_trans += m.get("transient", 0.0)
        count += 1

    count = max(count, 1)
    return {
        "stability": _round(total_stab / count),
        "attractor_weight": _round(total_attr / count),
        "transient_weight": _round(total_trans / count),
    }


# ---------------------------------------------------------------------------
# 3. Treatment scoring
# ---------------------------------------------------------------------------


def score_treatment(result: dict) -> float:
    """Score a treatment result using a deterministic objective.

    Rewards higher stability and attractor weight.
    Penalizes transient weight, basin-switch risk, angular velocity.

    Parameters
    ----------
    result : dict
        Must contain ``post_metrics`` with ``stability``,
        ``attractor_weight``, ``transient_weight``.
        May contain ``aggregate_deltas``.

    Returns
    -------
    float
        Score in [0, 1].  Higher is better.
    """
    post = result.get("post_metrics", {})

    stability = float(post.get("stability", 0.0))
    attractor = float(post.get("attractor_weight", 0.0))
    transient = float(post.get("transient_weight", 0.0))

    # Weighted sum with positive metrics contributing positively
    # and negative metrics (transient) penalizing.
    score = (
        0.35 * _clamp(stability)
        + 0.35 * _clamp(attractor)
        + 0.30 * _clamp(1.0 - transient)
    )

    return _round(_clamp(score))


# ---------------------------------------------------------------------------
# 4. Best treatment selection
# ---------------------------------------------------------------------------


def select_best_treatment(results: list) -> dict:
    """Select the best treatment from evaluated candidates.

    Sort by: score DESC -> intervention complexity ASC -> action name ASC.

    Complexity is measured as strength (lower = simpler).

    Parameters
    ----------
    results : list of dict
        Output of ``evaluate_treatments``.

    Returns
    -------
    dict
        The best treatment result dict, or empty dict if no candidates.
    """
    if not results:
        return {}

    # Build sortable entries.
    decorated: List[tuple] = []
    for i, r in enumerate(results):
        score = float(r.get("score", 0.0))
        candidate = r.get("candidate", {})
        strength = float(candidate.get("strength", 0.0))
        action = str(candidate.get("action", ""))
        # Sort key: (-score, strength, action, original_index)
        decorated.append((-score, strength, action, i))

    decorated.sort()
    best_idx = decorated[0][3]
    return results[best_idx]


# ---------------------------------------------------------------------------
# 5. Treatment explanation
# ---------------------------------------------------------------------------


def explain_treatment(best: dict, diagnosis: dict) -> str:
    """Generate a human-readable explanation for the selected treatment.

    Parameters
    ----------
    best : dict
        Output of ``select_best_treatment``.
    diagnosis : dict
        The diagnosis (original or revised) that drove treatment selection.

    Returns
    -------
    str
        Multi-line explanation string.
    """
    if not best:
        return "No treatment selected (healthy convergence or no candidates)."

    candidate = best.get("candidate", {})
    action = candidate.get("action", "unknown")
    strength = candidate.get("strength", 0.0)
    score = best.get("score", 0.0)
    post = best.get("post_metrics", {})
    agg = best.get("aggregate_deltas", {})

    primary = str(
        diagnosis.get("revised_diagnosis",
                       diagnosis.get("primary_diagnosis", "unknown"))
    )

    lines: List[str] = []
    lines.append(f"Selected {action}({strength}) due to:")

    if primary != "healthy_convergence":
        lines.append(f"- primary diagnosis: {primary}")

    d_stab = agg.get("delta_stability", 0.0)
    if d_stab > 0:
        lines.append(f"- stability improvement (+{d_stab:.2f})")
    elif d_stab < 0:
        lines.append(f"- stability trade-off ({d_stab:.2f})")

    post_stab = post.get("stability", 0.0)
    post_attr = post.get("attractor_weight", 0.0)
    if post_stab > 0.5:
        lines.append(f"- post-intervention stability: {post_stab:.2f}")
    if post_attr > 0.3:
        lines.append(f"- attractor weight: {post_attr:.2f}")

    lines.append(f"- treatment score: {score:.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Full treatment planning pipeline
# ---------------------------------------------------------------------------


def run_treatment_planning(runs: list) -> dict:
    """Run the full treatment planning pipeline.

    Pipeline:
        differential diagnosis
        -> provocation analysis
        -> revised diagnosis
        -> treatment candidate generation
        -> treatment evaluation
        -> best treatment selection

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.

    Returns
    -------
    dict
        Contains ``provocation``, ``revised_diagnosis``, ``candidates``,
        ``evaluations``, ``best_treatment``, ``explanation``.
    """
    from qec.analysis.provocation_analysis import run_provocation_analysis

    provocation = run_provocation_analysis(runs)

    revision = provocation.get("revision", {})
    diagnosis = provocation.get("diagnosis", {})

    # Build a combined diagnosis for candidate generation.
    combined_diagnosis = {
        "primary_diagnosis": diagnosis.get("primary_diagnosis", "unknown"),
        "diagnosis_confidence": diagnosis.get("diagnosis_confidence", 0.0),
        "revised_diagnosis": revision.get("revised_diagnosis",
                                           diagnosis.get("primary_diagnosis", "unknown")),
        "revised_confidence": revision.get("revised_confidence", 0.0),
        "ranked": diagnosis.get("ranked", []),
    }

    candidates = generate_treatment_candidates(combined_diagnosis)

    if not candidates:
        return {
            "provocation": provocation,
            "revised_diagnosis": combined_diagnosis,
            "candidates": [],
            "evaluations": [],
            "best_treatment": {},
            "explanation": "No treatment needed (healthy convergence).",
        }

    evaluations = evaluate_treatments(runs, candidates)
    best = select_best_treatment(evaluations)
    explanation = explain_treatment(best, combined_diagnosis)

    return {
        "provocation": provocation,
        "revised_diagnosis": combined_diagnosis,
        "candidates": candidates,
        "evaluations": evaluations,
        "best_treatment": best,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# 7. Formatter
# ---------------------------------------------------------------------------


def format_treatment_plan(result: dict) -> str:
    """Format treatment planning results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_treatment_planning``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []

    provocation = result.get("provocation", {})
    diagnosis = provocation.get("diagnosis", {})
    revision = result.get("revised_diagnosis", {})
    evaluations = result.get("evaluations", [])
    best = result.get("best_treatment", {})
    explanation = result.get("explanation", "")

    primary = diagnosis.get("primary_diagnosis", "unknown")
    revised = revision.get("revised_diagnosis", primary)

    lines.append("")
    lines.append("=== Treatment Plan ===")
    lines.append("")
    lines.append(f"Original Diagnosis: {primary}")
    lines.append(f"Revised Diagnosis: {revised}")
    lines.append(f"Candidates Evaluated: {len(evaluations)}")

    if best:
        candidate = best.get("candidate", {})
        lines.append("")
        lines.append("Best Treatment:")
        lines.append(f"  Action: {candidate.get('action', 'none')}")
        lines.append(f"  Strength: {candidate.get('strength', 0.0)}")
        lines.append(f"  Score: {best.get('score', 0.0):.2f}")
        lines.append("")
        lines.append("Explanation:")
        for line in explanation.split("\n"):
            lines.append(f"  {line}")
    else:
        lines.append("")
        lines.append("No treatment needed.")

    # Show top 3 candidates.
    if evaluations:
        sorted_evals = sorted(evaluations, key=lambda e: -e.get("score", 0.0))
        lines.append("")
        lines.append("Top Candidates:")
        for i, ev in enumerate(sorted_evals[:3]):
            c = ev.get("candidate", {})
            lines.append(
                f"  {i + 1}. {c.get('action', '?')}({c.get('strength', 0.0)}) "
                f"score={ev.get('score', 0.0):.2f}"
            )

    return "\n".join(lines)


__all__ = [
    "ROUND_PRECISION",
    "TREATMENT_STRENGTHS",
    "VALID_ACTIONS",
    "evaluate_treatments",
    "explain_treatment",
    "format_treatment_plan",
    "generate_treatment_candidates",
    "run_treatment_planning",
    "score_treatment",
    "select_best_treatment",
]
