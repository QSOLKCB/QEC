"""Deterministic self-adjustment loop (v94.0.0).

Upgrades the diagnostics pipeline from:
    diagnostics -> recommendations
to:
    diagnostics -> apply one recommendation -> re-measure -> accept/reject

Policy simulation layer only — does not rerun experiments.
Uses existing benchmark/diagnostic records as the adjustment surface.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No ML. No search.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    best_mode_per_system,
    classify_all_systems,
    normalize_results,
    run_self_diagnostics,
)


# ---------------------------------------------------------------------------
# PART 1 — ADJUSTMENT MODEL
# ---------------------------------------------------------------------------

# Deterministic mapping: recommendation -> candidate mode.
_RECOMMENDATION_TO_MODE: Dict[str, str] = {
    "prefer_d4_projection": "d4",
    "prefer_invariant_guided_modes": "d4+inv",
    "enable_invariant_guidance": "d4+inv",
    "reduce_projection_strength": "square",
    "disable_correction": "none",
    "switch_to_d4_or_invariant": "d4+inv",
    "increase_structure_guidance": "d4+inv",
    "skip_correction_or_expand_input": "none",
}

# Deterministic fallback chains per mode.
_FALLBACK_CHAINS: Dict[str, List[str]] = {
    "d4+inv": ["d4", "square", "none"],
    "d4": ["square", "none"],
    "square": ["none"],
    "none": [],
}


def resolve_recommendation_to_mode(
    recommendation: str,
    available_modes: List[str],
) -> Optional[str]:
    """Resolve a recommendation string to a concrete correction mode.

    Uses explicit mapping only. If preferred mode is unavailable,
    falls back deterministically through the fallback chain.
    Returns None only if no valid option exists.
    """
    preferred = _RECOMMENDATION_TO_MODE.get(recommendation)
    if preferred is None:
        return None

    if preferred in available_modes:
        return preferred

    # Walk fallback chain.
    for fallback in _FALLBACK_CHAINS.get(preferred, []):
        if fallback in available_modes:
            return fallback

    return None


# ---------------------------------------------------------------------------
# PART 2 — IMPROVEMENT CRITERION
# ---------------------------------------------------------------------------


def is_improvement(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
) -> Tuple[bool, str]:
    """Determine if after_metrics improves on before_metrics.

    Priority:
      1. higher stability_efficiency -> accept
      2. if tied, higher compression_efficiency -> accept
      3. if tied, higher stability_gain -> accept
      4. else reject

    Returns (accepted, reason).
    """
    b_stab = before_metrics.get("stability_efficiency", 0.0)
    a_stab = after_metrics.get("stability_efficiency", 0.0)
    b_comp = before_metrics.get("compression_efficiency", 0.0)
    a_comp = after_metrics.get("compression_efficiency", 0.0)
    b_gain = before_metrics.get("stability_gain", 0.0)
    a_gain = after_metrics.get("stability_gain", 0.0)

    if a_stab > b_stab:
        return True, "accepted_improved_stability"
    if a_stab < b_stab:
        return False, "rejected_no_improvement"

    # stability_efficiency tied — check compression.
    if a_comp > b_comp:
        return True, "accepted_improved_compression"
    if a_comp < b_comp:
        return False, "rejected_no_improvement"

    # compression also tied — check stability_gain.
    if a_gain > b_gain:
        return True, "accepted_improved_stability_gain"
    return False, "rejected_no_improvement"


# ---------------------------------------------------------------------------
# PART 3 — SINGLE-SYSTEM ADJUSTMENT
# ---------------------------------------------------------------------------


def _extract_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract the three comparison metrics from a record."""
    return {
        "stability_efficiency": float(record.get("stability_efficiency", 0.0)),
        "compression_efficiency": float(
            record.get("compression_efficiency", 0.0)
        ),
        "stability_gain": float(record.get("stability_gain", 0.0)),
    }


def adjust_system(
    system_records: List[Dict[str, Any]],
    recommendations: List[str],
) -> Dict[str, Any]:
    """Apply one-step adjustment to a single (dfa_type, n) system.

    Input:
      system_records: all mode records for one (dfa_type, n)
      recommendations: recommendation list for this system

    Output:
      dict with dfa_type, n, original_mode, candidate_mode, accepted,
      reason, before_metrics, after_metrics, applied_recommendation.
    """
    if not system_records:
        return {
            "dfa_type": "",
            "n": None,
            "original_mode": "",
            "candidate_mode": None,
            "accepted": False,
            "reason": "rejected_no_valid_candidate",
            "before_metrics": {},
            "after_metrics": {},
            "applied_recommendation": None,
        }

    dfa_type = system_records[0]["dfa_type"]
    n = system_records[0]["n"]

    # Find current best mode using same logic as diagnostics.
    best_record = max(
        system_records,
        key=lambda r: (
            r["stability_efficiency"],
            r["compression_efficiency"],
            [-ord(c) for c in r["mode"]],
        ),
    )
    original_mode = best_record["mode"]
    before_metrics = _extract_metrics(best_record)

    # No recommendations -> nothing to do.
    if not recommendations:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "original_mode": original_mode,
            "candidate_mode": None,
            "accepted": False,
            "reason": "rejected_no_valid_candidate",
            "before_metrics": before_metrics,
            "after_metrics": {},
            "applied_recommendation": None,
        }

    # Take first recommendation only.
    first_rec = recommendations[0]
    available_modes = sorted(set(r["mode"] for r in system_records))
    candidate_mode = resolve_recommendation_to_mode(first_rec, available_modes)

    if candidate_mode is None:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "original_mode": original_mode,
            "candidate_mode": None,
            "accepted": False,
            "reason": "rejected_no_valid_candidate",
            "before_metrics": before_metrics,
            "after_metrics": {},
            "applied_recommendation": first_rec,
        }

    # Same mode -> reject.
    if candidate_mode == original_mode:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "original_mode": original_mode,
            "candidate_mode": candidate_mode,
            "accepted": False,
            "reason": "rejected_same_mode",
            "before_metrics": before_metrics,
            "after_metrics": before_metrics,
            "applied_recommendation": first_rec,
        }

    # Find candidate record.
    candidate_record = None
    for r in system_records:
        if r["mode"] == candidate_mode:
            candidate_record = r
            break

    if candidate_record is None:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "original_mode": original_mode,
            "candidate_mode": candidate_mode,
            "accepted": False,
            "reason": "rejected_no_valid_candidate",
            "before_metrics": before_metrics,
            "after_metrics": {},
            "applied_recommendation": first_rec,
        }

    after_metrics = _extract_metrics(candidate_record)
    accepted, reason = is_improvement(before_metrics, after_metrics)

    return {
        "dfa_type": dfa_type,
        "n": n,
        "original_mode": original_mode,
        "candidate_mode": candidate_mode,
        "accepted": accepted,
        "reason": reason,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "applied_recommendation": first_rec,
    }


# ---------------------------------------------------------------------------
# PART 4 — FULL ADJUSTMENT PIPELINE
# ---------------------------------------------------------------------------


def _collect_system_recommendations(
    diagnostics_report: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], List[str]]:
    """Collect unique recommendations per system from diagnostics output."""
    rec_map: Dict[Tuple[str, Optional[int]], List[str]] = {}
    for r in diagnostics_report.get("recommendations", []):
        key = (r["dfa_type"], r["n"])
        if key not in rec_map:
            rec_map[key] = []
        for rec in r["recommendations"]:
            if rec not in rec_map[key]:
                rec_map[key].append(rec)
    return rec_map


def run_self_adjustment(data: Any) -> Dict[str, Any]:
    """Full self-adjustment pipeline.

    Accepts:
      - raw run_suite() results (list)
      - summarize() output (dict)
      - run_self_diagnostics() output (dict with 'metrics' key)

    Pipeline:
      1. normalize inputs
      2. run diagnostics if needed
      3. group records by (dfa_type, n)
      4. apply adjust_system() to each system
      5. return full report
    """
    # Detect if input is already a diagnostics report.
    if isinstance(data, dict) and "metrics" in data and "recommendations" in data:
        diagnostics = data
        agg = data["metrics"]
    else:
        diagnostics = run_self_diagnostics(data)
        agg = diagnostics["metrics"]

    # Build system class map.
    class_map: Dict[Tuple[str, Optional[int]], str] = {}
    for sc in diagnostics.get("system_classes", []):
        class_map[(sc["dfa_type"], sc["n"])] = sc["system_class"]

    # Collect recommendations per system.
    rec_map = _collect_system_recommendations(diagnostics)

    # Group aggregated records by system.
    system_groups: Dict[
        Tuple[str, Optional[int]], List[Dict[str, Any]]
    ] = {}
    for r in agg:
        key = (r["dfa_type"], r["n"])
        system_groups.setdefault(key, []).append(r)

    # Apply adjustment to each system in deterministic order.
    adjustments: List[Dict[str, Any]] = []
    for key in sorted(system_groups.keys(), key=lambda k: (k[0], str(k[1]))):
        records = system_groups[key]
        recs = rec_map.get(key, [])
        result = adjust_system(records, recs)
        result["system_class"] = class_map.get(key, "unknown")
        adjustments.append(result)

    return {
        "diagnostics": diagnostics,
        "adjustments": adjustments,
    }


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_adjustment_report(report: Dict[str, Any]) -> str:
    """Format adjustment report as human-readable text.

    Deterministic, sorted, text-only, diff-friendly.
    """
    lines: List[str] = []

    for adj in report.get("adjustments", []):
        n_label = "NA" if adj["n"] is None else str(adj["n"])
        lines.append(f"DFA: {adj['dfa_type']} (n={n_label})")
        lines.append(f"  class: {adj.get('system_class', 'unknown')}")
        lines.append(f"  best_mode: {adj['original_mode']}")

        rec = adj.get("applied_recommendation")
        lines.append(
            f"  recommendation: {rec if rec is not None else 'none'}"
        )
        lines.append(
            f"  candidate_mode: "
            f"{adj['candidate_mode'] if adj['candidate_mode'] is not None else 'none'}"
        )
        lines.append(f"  accepted: {adj['accepted']}")
        lines.append(f"  reason: {adj['reason']}")
        lines.append("")

    return "\n".join(lines)
