"""Invariant application layer (v95.1.0).

Upgrades from:
    accepted invariant -> evaluated
to:
    accepted invariant -> applied (overlay) -> re-evaluated -> compared

This layer creates a filtered/reweighted VIEW of existing records.
It does NOT modify base data, inject invariants into the correction engine,
or create feedback loops.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.invariant_synthesis import (
    _current_best_metrics,
    _extract_proxy_metrics,
    _is_improvement,
    run_invariant_synthesis,
)
from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    normalize_results,
)
from qec.analysis.structural_diagnostics import _group_by_system


# ---------------------------------------------------------------------------
# PART 1 — SELECT ACCEPTED INVARIANTS
# ---------------------------------------------------------------------------


def get_accepted_invariants(
    synthesis_report: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]]:
    """Extract accepted invariants from a synthesis report.

    Returns:
        Dict mapping (dfa_type, n) -> list of accepted invariant dicts.
        Only includes evaluations where accepted == True.
    """
    result: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}

    for evaluation in synthesis_report.get("accepted_invariants", []):
        if not evaluation.get("accepted", False):
            continue
        candidate = evaluation.get("candidate", {})
        dfa_type = candidate.get("dfa_type", "")
        n = candidate.get("n")
        key = (dfa_type, n)
        result.setdefault(key, []).append(candidate)

    # Sort keys deterministically.
    sorted_result: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for key in sorted(result.keys(), key=lambda k: (k[0], str(k[1]))):
        sorted_result[key] = result[key]

    return sorted_result


# ---------------------------------------------------------------------------
# PART 2 — OVERLAY APPLICATION MODEL
# ---------------------------------------------------------------------------


def _apply_local_stability_overlay(
    records: List[Dict[str, Any]],
    invariant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply local_stability_constraint overlay.

    Prefer records with higher stability_efficiency.
    Filter out bottom 25% stability.
    """
    if not records:
        return []

    stabilities = sorted(
        [r["stability_efficiency"] for r in records],
    )
    # Bottom 25% threshold.
    cutoff_idx = max(1, len(stabilities) // 4)
    threshold = stabilities[cutoff_idx - 1]

    # Filter out records at or below the bottom 25% threshold,
    # but keep at least one record.
    filtered = [
        r for r in records
        if r["stability_efficiency"] > threshold
    ]
    if not filtered:
        # Keep the best one.
        filtered = [
            max(records, key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
                r.get("stability_gain", 0),
            ))
        ]

    # Sort by stability descending for deterministic ordering.
    filtered.sort(
        key=lambda r: (
            -r["stability_efficiency"],
            -r["compression_efficiency"],
            -r.get("stability_gain", 0),
            r["mode"],
        )
    )
    return filtered


def _apply_equivalence_class_overlay(
    records: List[Dict[str, Any]],
    invariant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply equivalence_class_constraint overlay.

    Prefer higher compression_efficiency.
    Simulate collapse by selecting best compression mode.
    """
    if not records:
        return []

    # Sort by compression descending.
    sorted_records = sorted(
        records,
        key=lambda r: (
            -r["compression_efficiency"],
            -r["stability_efficiency"],
            -r.get("stability_gain", 0),
            r["mode"],
        ),
    )
    return sorted_records


def _apply_geometry_alignment_overlay(
    records: List[Dict[str, Any]],
    invariant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply geometry_alignment_constraint overlay.

    Force preferred mode: 'd4' or 'square'.
    """
    rule = invariant.get("rule", {})
    preferred = rule.get("align_projection", "")

    # Preferred modes in priority order.
    preferred_modes = []
    if preferred:
        preferred_modes.append(preferred)
    for m in ("d4", "square"):
        if m not in preferred_modes:
            preferred_modes.append(m)

    # Sort: preferred modes first, then by stability.
    def sort_key(r: Dict[str, Any]) -> tuple:
        mode = r["mode"]
        if mode in preferred_modes:
            priority = preferred_modes.index(mode)
        else:
            priority = len(preferred_modes)
        return (
            priority,
            -r["stability_efficiency"],
            -r["compression_efficiency"],
            -r.get("stability_gain", 0),
            mode,
        )

    return sorted(records, key=sort_key)


def _apply_explicit_allowed_state_overlay(
    records: List[Dict[str, Any]],
    invariant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply explicit_allowed_state_constraint overlay.

    Prefer invariant-guided modes: 'd4+inv' > 'd4' > others.
    """
    # Priority ordering for modes.
    mode_priority = {"d4+inv": 0, "d4": 1}

    def sort_key(r: Dict[str, Any]) -> tuple:
        mode = r["mode"]
        priority = mode_priority.get(mode, 2)
        return (
            priority,
            -r["stability_efficiency"],
            -r["compression_efficiency"],
            -r.get("stability_gain", 0),
            mode,
        )

    return sorted(records, key=sort_key)


def _apply_bounded_projection_overlay(
    records: List[Dict[str, Any]],
    invariant: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply bounded_projection_constraint overlay.

    Penalize overcorrection: remove modes with
    compression > 0.6 AND stability < 0.15.
    """
    if not records:
        return []

    filtered = [
        r for r in records
        if not (
            r["compression_efficiency"] > 0.6
            and r["stability_efficiency"] < 0.15
        )
    ]

    if not filtered:
        # Keep all if filtering removes everything.
        filtered = list(records)

    filtered.sort(
        key=lambda r: (
            -r["stability_efficiency"],
            -r["compression_efficiency"],
            -r.get("stability_gain", 0),
            r["mode"],
        )
    )
    return filtered


# Registry mapping invariant type to overlay function.
_OVERLAY_APPLICATORS = {
    "local_stability_constraint": _apply_local_stability_overlay,
    "equivalence_class_constraint": _apply_equivalence_class_overlay,
    "geometry_alignment_constraint": _apply_geometry_alignment_overlay,
    "explicit_allowed_state_constraint": _apply_explicit_allowed_state_overlay,
    "bounded_projection_constraint": _apply_bounded_projection_overlay,
}


def apply_invariant_overlay(
    system_records: List[Dict[str, Any]],
    invariants: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Apply invariant overlay to system records.

    Creates a filtered/reweighted view — does NOT modify base records.
    Applies each invariant's overlay sequentially (filtering/sorting only).

    Args:
        system_records: mode records for one (dfa_type, n).
        invariants: list of accepted invariant candidate dicts.

    Returns:
        New list of filtered/sorted records (copies, not originals).
    """
    if not system_records:
        return []

    # Deep copy to avoid any mutation.
    current = [dict(r) for r in system_records]

    for inv in invariants:
        inv_type = inv.get("type", "")
        applicator = _OVERLAY_APPLICATORS.get(inv_type)
        if applicator is not None:
            current = applicator(current, inv)

    return current


# ---------------------------------------------------------------------------
# PART 3 — RE-EVALUATION
# ---------------------------------------------------------------------------


def _best_mode_from_records(
    records: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, float]]:
    """Find the best mode and its metrics from a list of records.

    Uses the standard hierarchy: stability > compression > gain.

    Returns:
        (mode_name, metrics_dict)
    """
    if not records:
        return ("none", {
            "stability_efficiency": 0.0,
            "compression_efficiency": 0.0,
            "stability_gain": 0.0,
        })

    best = max(
        records,
        key=lambda r: (
            r["stability_efficiency"],
            r["compression_efficiency"],
            r.get("stability_gain", 0),
        ),
    )
    return (best["mode"], _extract_proxy_metrics(best))


def evaluate_overlay(
    system_records: List[Dict[str, Any]],
    overlay_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate overlay result vs original records.

    Compares best mode before and after overlay application.
    Uses same hierarchy: stability -> compression -> stability_gain.

    Args:
        system_records: original mode records.
        overlay_records: records after overlay filtering/sorting.

    Returns:
        Dict with before_mode, after_mode, improved, reason,
        before_metrics, after_metrics.
    """
    before_mode, before_metrics = _best_mode_from_records(system_records)
    after_mode, after_metrics = _best_mode_from_records(overlay_records)

    improved, reason = _is_improvement(before_metrics, after_metrics)

    # If the mode changed but metrics didn't improve, note the mode change.
    if not improved and before_mode != after_mode:
        reason = "mode_changed_no_improvement"
    elif not improved and before_mode == after_mode:
        reason = "no_change"

    return {
        "before_mode": before_mode,
        "after_mode": after_mode,
        "improved": improved,
        "reason": reason,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
    }


# ---------------------------------------------------------------------------
# PART 4 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_invariant_application(data: Any) -> Dict[str, Any]:
    """Full invariant application pipeline.

    Pipeline:
      1. Run invariant synthesis (or accept its output).
      2. Extract accepted invariants.
      3. For each system: apply overlay, evaluate result.

    Args:
        data: raw run_suite() results (list) or summarize() output (dict),
              or a pre-computed synthesis report (dict with
              'accepted_invariants' key).

    Returns:
        Dict with accepted_invariants, applications list.
    """
    # Detect if data is already a synthesis report.
    if (
        isinstance(data, dict)
        and "accepted_invariants" in data
        and "structural_diagnostics" in data
    ):
        synthesis_report = data
    else:
        synthesis_report = run_invariant_synthesis(data)

    # Extract accepted invariants.
    accepted = get_accepted_invariants(synthesis_report)

    # Build system records.
    if "structural_diagnostics" in synthesis_report:
        sd = synthesis_report["structural_diagnostics"]
        agg_records = sd.get("aggregated_records", [])
        if not agg_records:
            # Rebuild from data.
            if isinstance(data, dict) and "accepted_invariants" in data:
                agg_records = []
            else:
                records = normalize_results(data)
                agg_records = aggregate_metrics(records)
    else:
        records = normalize_results(data)
        agg_records = aggregate_metrics(records)

    system_records = _group_by_system(agg_records)

    # Apply overlay per system.
    applications: List[Dict[str, Any]] = []

    all_system_keys = set(system_records.keys()) | set(accepted.keys())
    for key in sorted(all_system_keys, key=lambda k: (k[0], str(k[1]))):
        sys_records = system_records.get(key, [])
        sys_invariants = accepted.get(key, [])

        if not sys_invariants:
            continue

        overlay_records = apply_invariant_overlay(sys_records, sys_invariants)
        evaluation = evaluate_overlay(sys_records, overlay_records)

        applications.append({
            "dfa_type": key[0],
            "n": key[1],
            "invariants": [inv.get("type", "") for inv in sys_invariants],
            "before_mode": evaluation["before_mode"],
            "after_mode": evaluation["after_mode"],
            "improved": evaluation["improved"],
            "reason": evaluation["reason"],
            "before_metrics": evaluation["before_metrics"],
            "after_metrics": evaluation["after_metrics"],
        })

    return {
        "accepted_invariants": {
            f"{k[0]}(n={k[1]})": [inv.get("type", "") for inv in v]
            for k, v in accepted.items()
        },
        "applications": applications,
    }


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_application_report(report: Dict[str, Any]) -> str:
    """Format invariant application report as human-readable text.

    Deterministic, sorted, text-only.
    """
    lines: List[str] = []
    lines.append("=== Invariant Application Report ===")
    lines.append("")

    applications = report.get("applications", [])

    if not applications:
        lines.append("No invariants applied.")
        lines.append("")
        return "\n".join(lines)

    for app in applications:
        dfa_type = app.get("dfa_type", "unknown")
        n = app.get("n")
        lines.append(f"DFA: {dfa_type} (n={n})")

        invariants = app.get("invariants", [])
        lines.append("invariants:")
        for inv in invariants:
            lines.append(f"  - {inv}")

        lines.append(f"before: {app.get('before_mode', 'unknown')}")
        lines.append(f"after: {app.get('after_mode', 'unknown')}")
        lines.append(f"improved: {app.get('improved', False)}")
        lines.append(f"reason: {app.get('reason', 'unknown')}")
        lines.append("---")

    # Summary.
    improved_count = sum(1 for a in applications if a.get("improved"))
    lines.append("")
    lines.append(f"Total applications: {len(applications)}")
    lines.append(f"Improved: {improved_count}")
    lines.append(f"Unchanged: {len(applications) - improved_count}")

    return "\n".join(lines)
