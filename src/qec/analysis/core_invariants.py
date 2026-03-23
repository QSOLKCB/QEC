"""Core invariant promotion and overlay evaluation (v96.0.0).

Upgrades from:
    invariant effectiveness analysis per class
to:
    cross-class invariant promotion + core invariant overlay + hierarchy evaluation

Identifies invariants that generalize across DFA system classes,
promotes them to core status, and evaluates their interaction with
hierarchical correction modes.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.invariant_analysis import (
    aggregate_by_class,
    normalize_application_data,
    run_invariant_analysis,
)
from qec.analysis.invariant_application import (
    _OVERLAY_APPLICATORS,
    apply_invariant_overlay,
)


# ---------------------------------------------------------------------------
# PART 1 — CORE INVARIANT IDENTIFICATION
# ---------------------------------------------------------------------------


def identify_core_invariants(
    invariant_report: Dict[str, Any],
) -> Dict[str, Any]:
    """Identify invariants that generalize across system classes.

    Promotion rule: an invariant becomes core if:
      - it appears in at least 2 distinct system classes
      - improved_ratio >= 0.5 in each of those classes
      - avg_score > 0 in each of those classes

    Args:
        invariant_report: output of run_invariant_analysis().

    Returns:
        Dict with 'core_invariants' list, each entry containing:
          invariant_type, classes, global_rank, evidence.
    """
    class_effectiveness = invariant_report.get("class_effectiveness", {})

    # Build per-invariant evidence: which classes it qualifies in.
    # class_effectiveness: {system_class: {invariant_type: metrics}}
    invariant_classes: Dict[str, List[Dict[str, Any]]] = {}

    for sys_class in sorted(class_effectiveness.keys()):
        inv_map = class_effectiveness[sys_class]
        for inv_type in sorted(inv_map.keys()):
            metrics = inv_map[inv_type]
            improved_ratio = metrics.get("improved_ratio", 0.0)
            avg_score = metrics.get("avg_score", 0.0)

            # Check promotion criteria per class.
            if improved_ratio >= 0.5 and avg_score > 0:
                entry = {
                    "system_class": sys_class,
                    "improved_ratio": improved_ratio,
                    "avg_score": avg_score,
                    "count": metrics.get("count", 0),
                }
                invariant_classes.setdefault(inv_type, []).append(entry)

    # Build global rank lookup from the invariant report.
    global_ranking = invariant_report.get("global_ranking", [])
    rank_map: Dict[str, int] = {}
    for entry in global_ranking:
        rank_map[entry["invariant_type"]] = entry["rank"]

    # Promote invariants appearing in >= 2 classes.
    core_list: List[Dict[str, Any]] = []
    for inv_type in sorted(invariant_classes.keys()):
        class_entries = invariant_classes[inv_type]
        if len(class_entries) < 2:
            continue

        classes = sorted([e["system_class"] for e in class_entries])
        evidence = {
            "per_class": {
                e["system_class"]: {
                    "improved_ratio": e["improved_ratio"],
                    "avg_score": e["avg_score"],
                    "count": e["count"],
                }
                for e in class_entries
            },
            "num_classes": len(classes),
            "avg_class_score": round(
                sum(e["avg_score"] for e in class_entries) / len(class_entries),
                6,
            ),
        }

        core_list.append({
            "invariant_type": inv_type,
            "classes": classes,
            "global_rank": rank_map.get(inv_type, 999),
            "evidence": evidence,
        })

    # Rank promoted invariants.
    core_list = _rank_core_invariants(core_list)

    return {"core_invariants": core_list}


def _rank_core_invariants(
    core_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank core invariants deterministically.

    Sort by:
      1. number of supported classes descending
      2. average class score descending
      3. invariant name ascending (tiebreak)

    Assigns global_rank based on final position.
    """
    core_list.sort(
        key=lambda x: (
            -x["evidence"]["num_classes"],
            -x["evidence"]["avg_class_score"],
            x["invariant_type"],
        )
    )

    for i, entry in enumerate(core_list):
        entry["global_rank"] = i + 1

    return core_list


# ---------------------------------------------------------------------------
# PART 2 — CORE INVARIANT OVERLAY
# ---------------------------------------------------------------------------


def apply_core_invariant_overlay(
    system_records: List[Dict[str, Any]],
    core_invariants: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply promoted core invariants as a shared overlay.

    Only applies overlay for invariant types that are in the core set.
    Reuses invariant application semantics from v95.1.
    Deterministic ordering. No metric mutation, filtering/prioritization only.

    Args:
        system_records: mode records for one system (list of dicts
            with mode, stability_efficiency, compression_efficiency, etc.).
        core_invariants: output of identify_core_invariants().

    Returns:
        New list of filtered/sorted records (copies, not originals).
    """
    if not system_records:
        return []

    core_list = core_invariants.get("core_invariants", [])
    if not core_list:
        return [dict(r) for r in system_records]

    # Build synthetic invariant dicts for the overlay applicator.
    # Only use core invariant types that have registered applicators.
    synthetic_invariants: List[Dict[str, Any]] = []
    for core in core_list:
        inv_type = core["invariant_type"]
        if inv_type in _OVERLAY_APPLICATORS:
            synthetic_invariants.append({
                "type": inv_type,
                "core": True,
                "rule": {},
            })

    if not synthetic_invariants:
        return [dict(r) for r in system_records]

    return apply_invariant_overlay(system_records, synthetic_invariants)


# ---------------------------------------------------------------------------
# PART 3 — HIERARCHY + CORE OVERLAY EVALUATION
# ---------------------------------------------------------------------------


def evaluate_core_overlay_with_hierarchy(
    system_records: List[Dict[str, Any]],
    core_invariants: Dict[str, Any],
    hierarchical_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare baseline, hierarchical, and hierarchical+core overlay.

    Args:
        system_records: original mode records for one system.
        core_invariants: output of identify_core_invariants().
        hierarchical_results: list of run_hierarchical_correction outputs
            for this system.

    Returns:
        Dict comparing baseline_mode, hierarchical_mode, core_overlay_mode,
        best_variant, and before/after metrics.
    """
    # 1. Baseline best: from original single-step records.
    baseline_mode, baseline_metrics = _best_from_records(system_records)

    # 2. Hierarchical best: from hierarchical results.
    hier_mode, hier_metrics = _best_from_hierarchical(hierarchical_results)

    # 3. Core overlay best: apply core invariant overlay to hierarchical
    #    records (converted to record-like format).
    hier_as_records = _hierarchical_to_records(hierarchical_results)
    overlay_records = apply_core_invariant_overlay(hier_as_records, core_invariants)
    core_mode, core_metrics = _best_from_records(overlay_records)

    # Determine winner.
    candidates = [
        ("baseline", baseline_mode, baseline_metrics),
        ("hierarchical", hier_mode, hier_metrics),
        ("core_overlay", core_mode, core_metrics),
    ]
    winner = _pick_winner(candidates)

    return {
        "baseline_mode": baseline_mode,
        "hierarchical_mode": hier_mode,
        "core_overlay_mode": core_mode,
        "best_variant": winner,
        "before_metrics": baseline_metrics,
        "after_metrics": core_metrics,
    }


def _best_from_records(
    records: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, float]]:
    """Find best mode from a list of record dicts."""
    if not records:
        return ("none", _empty_metrics())

    best = max(
        records,
        key=lambda r: (
            r.get("stability_efficiency", 0.0),
            r.get("compression_efficiency", 0.0),
            r.get("stability_gain", 0),
            "",  # ensure tuple length matches
        ),
    )
    return (
        best.get("mode", "none"),
        {
            "stability_efficiency": best.get("stability_efficiency", 0.0),
            "compression_efficiency": best.get("compression_efficiency", 0.0),
            "stability_gain": best.get("stability_gain", 0),
        },
    )


def _best_from_hierarchical(
    results: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, float]]:
    """Find best mode from hierarchical results."""
    if not results:
        return ("none", _empty_metrics())

    best = max(
        results,
        key=lambda r: (
            r.get("stability_efficiency", 0.0),
            r.get("compression_efficiency", 0.0),
            r.get("stability_gain", 0),
        ),
    )
    return (
        best.get("mode", "none"),
        {
            "stability_efficiency": best.get("stability_efficiency", 0.0),
            "compression_efficiency": best.get("compression_efficiency", 0.0),
            "stability_gain": best.get("stability_gain", 0),
        },
    )


def _hierarchical_to_records(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert hierarchical results to record-like dicts for overlay."""
    records: List[Dict[str, Any]] = []
    for r in results:
        records.append({
            "mode": r.get("mode", "none"),
            "stability_efficiency": r.get("stability_efficiency", 0.0),
            "compression_efficiency": r.get("compression_efficiency", 0.0),
            "stability_gain": r.get("stability_gain", 0),
        })
    return records


def _empty_metrics() -> Dict[str, float]:
    """Return empty metrics dict."""
    return {
        "stability_efficiency": 0.0,
        "compression_efficiency": 0.0,
        "stability_gain": 0,
    }


def _pick_winner(
    candidates: List[Tuple[str, str, Dict[str, float]]],
) -> str:
    """Pick the winning variant from candidates.

    Each candidate is (variant_label, mode_name, metrics_dict).
    Ranked by stability_efficiency > compression_efficiency > stability_gain,
    with variant label as tiebreak.
    """
    best = max(
        candidates,
        key=lambda c: (
            c[2].get("stability_efficiency", 0.0),
            c[2].get("compression_efficiency", 0.0),
            c[2].get("stability_gain", 0),
        ),
    )
    return best[0]


# ---------------------------------------------------------------------------
# PART 4 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_hierarchical_invariant_pipeline(
    data: Any,
) -> Dict[str, Any]:
    """Full hierarchical correction + core invariant promotion pipeline.

    Pipeline:
      1. Normalize input / existing benchmark data.
      2. Run invariant analysis if needed.
      3. Identify core invariants.
      4. Run hierarchical correction modes on all DFAs.
      5. Apply core invariant overlay.
      6. Compare: single-step best, hierarchical best, hierarchical + core.

    Args:
        data: raw run_suite() results (list), invariant_application output,
              or invariant_analysis report.

    Returns:
        Dict with core_invariants, hierarchical_results, comparisons,
        global_best_modes.
    """
    from qec.analysis.invariant_application import run_invariant_application
    from qec.analysis.self_diagnostics import (
        aggregate_metrics,
        normalize_results,
    )
    from qec.analysis.structural_diagnostics import _group_by_system
    from qec.experiments.dfa_benchmark import (
        DFA_REGISTRY,
        DFA_SIZES,
    )
    from qec.experiments.hierarchical_correction import (
        compare_hierarchical_modes,
        run_all_hierarchical_modes,
    )

    # Step 1: Run invariant analysis.
    if isinstance(data, dict) and "global_ranking" in data:
        invariant_report = data
    else:
        # Run full application pipeline first.
        if isinstance(data, dict) and "applications" in data:
            app_report = data
        else:
            app_report = run_invariant_application(data)
        invariant_report = run_invariant_analysis(app_report)

    # Step 2: Identify core invariants.
    core = identify_core_invariants(invariant_report)

    # Step 3: Run hierarchical correction on all DFAs.
    all_hierarchical: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    for dfa_name in sorted(DFA_REGISTRY.keys()):
        builder = DFA_REGISTRY[dfa_name]
        for n in DFA_SIZES[dfa_name]:
            dfa = builder(n)

            # Run all hierarchical modes.
            hier_results = run_all_hierarchical_modes(dfa)
            for r in hier_results:
                r["dfa_name"] = dfa_name
                r["n"] = n

            all_hierarchical.extend(hier_results)

            # Build system records from original data for comparison.
            system_records = _build_system_records_for_dfa(
                data, dfa_name, n
            )

            # Evaluate core overlay with hierarchy.
            evaluation = evaluate_core_overlay_with_hierarchy(
                system_records, core, hier_results
            )

            comparisons.append({
                "dfa_name": dfa_name,
                "n": n,
                **evaluation,
            })

    # Step 4: Determine global best modes.
    global_best: Dict[str, str] = {}

    # Count wins per variant.
    variant_wins: Dict[str, int] = {}
    for comp in comparisons:
        v = comp.get("best_variant", "unknown")
        variant_wins[v] = variant_wins.get(v, 0) + 1

    if variant_wins:
        best_variant = max(
            sorted(variant_wins.keys()),
            key=lambda v: (variant_wins[v], v),
        )
        global_best["dominant_variant"] = best_variant

    # Best hierarchical mode overall.
    if all_hierarchical:
        ranked = compare_hierarchical_modes(all_hierarchical)
        if ranked:
            global_best["best_hierarchical_mode"] = ranked[0]["mode"]

    return {
        "core_invariants": core,
        "hierarchical_results": all_hierarchical,
        "comparisons": comparisons,
        "global_best_modes": global_best,
    }


def _build_system_records_for_dfa(
    data: Any,
    dfa_name: str,
    n: Optional[int],
) -> List[Dict[str, Any]]:
    """Extract system records for a specific DFA from data.

    Returns list of record-like dicts for the given (dfa_name, n).
    """
    if isinstance(data, list):
        # Raw suite results.
        records = []
        for r in data:
            if r.get("dfa_name") == dfa_name and r.get("n") == n:
                metrics = r.get("metrics", {})
                records.append({
                    "mode": r.get("mode", "none"),
                    "stability_efficiency": metrics.get(
                        "stability_efficiency", 0.0
                    ),
                    "compression_efficiency": metrics.get(
                        "compression_efficiency", 0.0
                    ),
                    "stability_gain": metrics.get("stability_gain", 0),
                })
        return records

    # If data is a dict (analysis output), try to extract records.
    if isinstance(data, dict):
        applications = data.get("applications", [])
        for app in applications:
            if app.get("dfa_type") == dfa_name and app.get("n") == n:
                before = app.get("before_metrics", {})
                return [{
                    "mode": app.get("before_mode", "none"),
                    "stability_efficiency": before.get(
                        "stability_efficiency", 0.0
                    ),
                    "compression_efficiency": before.get(
                        "compression_efficiency", 0.0
                    ),
                    "stability_gain": before.get("stability_gain", 0),
                }]

    return []


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_core_invariant_report(core_report: Dict[str, Any]) -> str:
    """Format core invariant report as human-readable text.

    Deterministic, sorted, diff-friendly, no plotting.

    Args:
        core_report: output of identify_core_invariants().

    Returns:
        Formatted text string.
    """
    lines: List[str] = []
    lines.append("=== Core Invariants ===")

    core_list = core_report.get("core_invariants", [])
    if not core_list:
        lines.append("No core invariants identified.")
        return "\n".join(lines)

    for entry in core_list:
        rank = entry["global_rank"]
        inv_type = entry["invariant_type"]
        classes = entry["classes"]
        lines.append(f"{rank}. {inv_type}")
        lines.append(f"   classes: {', '.join(classes)}")

    return "\n".join(lines)
