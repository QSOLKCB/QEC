"""Self-diagnostics and deterministic recommendation layer (v93.1.0).

Converts benchmark + correction outputs into:
  metrics -> system class inference -> issue detection
  -> context-aware recommendations

Analysis only — no modifications to decoder, correction, or benchmark logic.
All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs.
"""

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_results(data: Any) -> List[Dict[str, Any]]:
    """Normalize benchmark data into flat records.

    Accepts:
      - run_suite() raw results: list of dicts with nested 'metrics'
      - summarize() output: dict keyed by (dfa_type, n) -> mode -> metrics

    Returns list of flat records with keys:
      dfa_type, n, mode, compression_efficiency, stability_efficiency,
      stability_gain, unique_before, unique_after
    """
    if isinstance(data, list):
        return _normalize_raw(data)
    elif isinstance(data, dict):
        return _normalize_summary(data)
    raise TypeError(
        "data must be a list (run_suite output) or dict (summarize output)"
    )


def _normalize_raw(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize run_suite() output to flat records."""
    records = []
    for r in results:
        m = r["metrics"]
        records.append({
            "dfa_type": r["dfa_name"],
            "n": r["n"],
            "mode": r["mode"],
            "compression_efficiency": float(m["compression_efficiency"]),
            "stability_efficiency": float(m["stability_efficiency"]),
            "stability_gain": int(m["stability_gain"]),
            "unique_before": int(m["unique_before"]),
            "unique_after": int(m["unique_after"]),
        })
    return sorted(records, key=_record_sort_key)


def _normalize_summary(
    summary: Dict[Tuple[str, Optional[int]], Dict[str, Dict[str, float]]],
) -> List[Dict[str, Any]]:
    """Normalize summarize() output to flat records.

    Summary format lacks stability_gain, unique_before, unique_after,
    so those default to 0.
    """
    records = []
    for (dfa_type, n), modes in summary.items():
        for mode, metrics in modes.items():
            records.append({
                "dfa_type": dfa_type,
                "n": n,
                "mode": mode,
                "compression_efficiency": float(
                    metrics["compression_efficiency"]
                ),
                "stability_efficiency": float(
                    metrics["stability_efficiency"]
                ),
                "stability_gain": 0,
                "unique_before": 0,
                "unique_after": 0,
            })
    return sorted(records, key=_record_sort_key)


def _record_sort_key(r: Dict[str, Any]) -> Tuple[str, str, str]:
    """Deterministic sort key for records."""
    return (r["dfa_type"], str(r["n"]), r["mode"])


# ---------------------------------------------------------------------------
# PART 2 — METRIC AGGREGATION
# ---------------------------------------------------------------------------


def aggregate_metrics(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate records by (dfa_type, n, mode).

    Computes mean of compression_efficiency, stability_efficiency,
    stability_gain across duplicate keys. Returns sorted list.
    """
    groups: Dict[
        Tuple[str, Optional[int], str], List[Dict[str, Any]]
    ] = {}
    for r in records:
        key = (r["dfa_type"], r["n"], r["mode"])
        groups.setdefault(key, []).append(r)

    agg = []
    for key in sorted(groups.keys(), key=lambda k: (k[0], str(k[1]), k[2])):
        items = groups[key]
        count = len(items)
        agg.append({
            "dfa_type": key[0],
            "n": key[1],
            "mode": key[2],
            "compression_efficiency": sum(
                r["compression_efficiency"] for r in items
            ) / count,
            "stability_efficiency": sum(
                r["stability_efficiency"] for r in items
            ) / count,
            "stability_gain": sum(
                r["stability_gain"] for r in items
            ) / count,
            "unique_before": items[0]["unique_before"],
            "unique_after": items[0]["unique_after"],
        })
    return agg


def best_mode_per_system(
    agg: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Find the best correction mode for each (dfa_type, n) system.

    Tie-break order:
      1. highest stability_efficiency
      2. highest compression_efficiency
      3. lexicographic mode name
    """
    systems: Dict[
        Tuple[str, Optional[int]], List[Dict[str, Any]]
    ] = {}
    for r in agg:
        key = (r["dfa_type"], r["n"])
        systems.setdefault(key, []).append(r)

    best = []
    for key in sorted(systems.keys(), key=lambda k: (k[0], str(k[1]))):
        candidates = systems[key]
        winner = max(
            candidates,
            key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
                # Reverse lexicographic so min name wins in max()
                [-ord(c) for c in r["mode"]],
            ),
        )
        best.append({
            "dfa_type": key[0],
            "n": key[1],
            "best_mode": winner["mode"],
            "stability_efficiency": winner["stability_efficiency"],
            "compression_efficiency": winner["compression_efficiency"],
        })
    return best


# ---------------------------------------------------------------------------
# PART 2b — SYSTEM CLASS INFERENCE (Taxonomy Layer)
# ---------------------------------------------------------------------------


def get_mode_metrics(
    records: List[Dict[str, Any]],
    mode_name: str,
) -> Dict[str, float]:
    """Extract aggregated metrics for a specific mode from system records.

    Returns dict with compression_efficiency, stability_efficiency.
    If the mode is not found, returns zeros.
    """
    for r in records:
        if r["mode"] == mode_name:
            return {
                "compression_efficiency": r["compression_efficiency"],
                "stability_efficiency": r["stability_efficiency"],
            }
    return {"compression_efficiency": 0.0, "stability_efficiency": 0.0}


def infer_system_class(
    records_for_system: List[Dict[str, Any]],
) -> str:
    """Classify a system based on aggregated metrics across modes.

    Input: all mode records for a single (dfa_type, n).
    Output: one of "chain_like", "cycle_like", "branching_like",
            "basin_like", "degenerate", "unknown".

    Fully deterministic, rule-based, no ML.
    """
    if not records_for_system:
        return "unknown"

    # Check degenerate first — any record with unique_before <= 2.
    max_unique_before = 0
    for r in records_for_system:
        ub = r.get("unique_before", 0)
        if ub > max_unique_before:
            max_unique_before = ub
    if max_unique_before <= 2:
        return "degenerate"

    # Extract per-mode metrics.
    d4_metrics = get_mode_metrics(records_for_system, "d4")
    square_metrics = get_mode_metrics(records_for_system, "square")
    inv_metrics = get_mode_metrics(records_for_system, "inv")
    d4_inv_metrics = get_mode_metrics(records_for_system, "d4+inv")

    d4_stab = d4_metrics["stability_efficiency"]
    square_stab = square_metrics["stability_efficiency"]
    inv_stab = inv_metrics["stability_efficiency"]
    d4_inv_stab = d4_inv_metrics["stability_efficiency"]
    inv_comp = inv_metrics["compression_efficiency"]
    d4_comp = d4_metrics["compression_efficiency"]

    # Rule: chain-like — d4 stability dominates and is meaningful.
    if d4_stab > square_stab and d4_stab > 0.4:
        return "chain_like"

    # Rule: cycle-like — d4 alone is weak, invariant-guided is stronger.
    if d4_stab < 0.2 and inv_stab > d4_stab:
        return "cycle_like"

    # Rule: branching-like — invariant compression dominates d4 compression.
    if inv_comp > d4_comp and inv_comp > 0.4:
        return "branching_like"

    # Rule: basin-like — strong correction across multiple modes.
    max_stab = max(
        d4_stab, square_stab, inv_stab, d4_inv_stab,
    )
    modes_above_threshold = 0
    for stab in (d4_stab, square_stab, inv_stab, d4_inv_stab):
        if stab > 0.3:
            modes_above_threshold += 1
    if max_stab > 0.5 and modes_above_threshold >= 2:
        return "basin_like"

    return "unknown"


def classify_all_systems(
    agg: List[Dict[str, Any]],
) -> Dict[Tuple[str, Optional[int]], str]:
    """Classify all systems from aggregated records.

    Returns dict mapping (dfa_type, n) -> system_class.
    Deterministic ordering via sorted keys.
    """
    # Group records by system.
    systems: Dict[
        Tuple[str, Optional[int]], List[Dict[str, Any]]
    ] = {}
    for r in agg:
        key = (r["dfa_type"], r["n"])
        systems.setdefault(key, []).append(r)

    result: Dict[Tuple[str, Optional[int]], str] = {}
    for key in sorted(systems.keys(), key=lambda k: (k[0], str(k[1]))):
        result[key] = infer_system_class(systems[key])
    return result


# ---------------------------------------------------------------------------
# PART 2c — TAXONOMY-AWARE ISSUE DETECTION
# ---------------------------------------------------------------------------


def detect_taxonomy_issues(
    system_class: str,
    best_mode: str,
    metrics: List[Dict[str, Any]],
) -> List[str]:
    """Detect taxonomy-aware issues for a system.

    Rules are deterministic and based on system class vs best mode.
    Returns list of taxonomy issue strings.
    """
    issues: List[str] = []

    # Cycle mismatch: cycle-like systems should prefer d4+inv.
    if system_class == "cycle_like" and best_mode != "d4+inv":
        issues.append("cycle_mismatch")

    # Chain underperformance: chain-like systems should prefer d4.
    if system_class == "chain_like" and best_mode != "d4":
        issues.append("chain_underperformance")

    # Branching underuse: branching-like systems need invariant guidance.
    if system_class == "branching_like" and "inv" not in best_mode:
        issues.append("missing_invariant_guidance")

    # Degenerate overprocessing: no correction needed for simple systems.
    if system_class == "degenerate" and best_mode != "none":
        issues.append("overprocessing_simple_system")

    return issues


# ---------------------------------------------------------------------------
# PART 3 — ISSUE DETECTION
# ---------------------------------------------------------------------------

# Deterministic issue-to-recommendation mapping.
_RECOMMENDATION_MAP = {
    "no_effect": "increase_structure_guidance",
    "over_correction": "reduce_projection_strength",
    "destabilizing": "switch_to_d4_or_invariant",
    "low_diversity": "skip_correction_or_expand_input",
    "globally_weak_correction": "enable_invariant_guidance",
    # Taxonomy-aware recommendations (v93.1.0).
    "cycle_mismatch": "prefer_invariant_guided_modes",
    "chain_underperformance": "prefer_d4_projection",
    "missing_invariant_guidance": "enable_invariant_guidance",
    "overprocessing_simple_system": "disable_correction",
}


def detect_issues(
    record: Dict[str, Any],
    best_mode_metrics: Dict[Tuple[str, Optional[int]], Dict[str, float]],
) -> Dict[str, Any]:
    """Detect issues for a single record.

    Rules:
      1. no_effect: comp < 0.05 and stab < 0.05
      2. over_correction: comp > 0.7 and stab < 0.1
      3. destabilizing: stability_gain < 0
      4. low_diversity: unique_before <= 2
      5. globally_weak_correction: best stability_eff < 0.2 for this system

    Returns dict with dfa_type, n, mode, issues.
    """
    comp = record["compression_efficiency"]
    stab = record["stability_efficiency"]
    gain = record["stability_gain"]
    unique_before = record["unique_before"]

    issues: List[str] = []

    if comp < 0.05 and stab < 0.05:
        issues.append("no_effect")
    if comp > 0.7 and stab < 0.1:
        issues.append("over_correction")
    if gain < 0:
        issues.append("destabilizing")
    if unique_before <= 2:
        issues.append("low_diversity")

    system_key = (record["dfa_type"], record["n"])
    best = best_mode_metrics.get(system_key, {})
    if best.get("stability_efficiency", 0.0) < 0.2:
        issues.append("globally_weak_correction")

    return {
        "dfa_type": record["dfa_type"],
        "n": record["n"],
        "mode": record["mode"],
        "issues": issues,
    }


def detect_all(
    records: List[Dict[str, Any]],
    best_modes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect issues for all records."""
    best_map: Dict[Tuple[str, Optional[int]], Dict[str, float]] = {}
    for b in best_modes:
        best_map[(b["dfa_type"], b["n"])] = {
            "stability_efficiency": b["stability_efficiency"],
            "compression_efficiency": b["compression_efficiency"],
        }
    return [detect_issues(r, best_map) for r in records]


# ---------------------------------------------------------------------------
# PART 4 — RECOMMENDATION ENGINE
# ---------------------------------------------------------------------------


def recommend(issues: List[str]) -> List[str]:
    """Map issues to deterministic recommendations.

    Each issue maps to exactly one recommendation via _RECOMMENDATION_MAP.
    Order follows issue order (stable, deterministic).
    """
    return [_RECOMMENDATION_MAP[issue] for issue in issues]


def recommend_all(
    issue_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate recommendations for all issue records."""
    recs = []
    for ir in issue_records:
        recs.append({
            "dfa_type": ir["dfa_type"],
            "n": ir["n"],
            "recommendations": recommend(ir["issues"]),
        })
    return recs


# ---------------------------------------------------------------------------
# PART 5 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_self_diagnostics(data: Any) -> Dict[str, Any]:
    """Full self-diagnostics pipeline.

    Accepts run_suite() or summarize() output.
    Returns dict with metrics, best_modes, system_classes,
    issues, recommendations.
    """
    records = normalize_results(data)
    agg = aggregate_metrics(records)
    best_modes = best_mode_per_system(agg)
    system_classes = classify_all_systems(agg)

    # Existing per-record issue detection.
    issues = detect_all(agg, best_modes)

    # Taxonomy-aware issues: append to each system's first record.
    best_mode_map: Dict[Tuple[str, Optional[int]], str] = {}
    for b in best_modes:
        best_mode_map[(b["dfa_type"], b["n"])] = b["best_mode"]

    # Group agg records by system for taxonomy issue detection.
    system_records: Dict[
        Tuple[str, Optional[int]], List[Dict[str, Any]]
    ] = {}
    for r in agg:
        key = (r["dfa_type"], r["n"])
        system_records.setdefault(key, []).append(r)

    # Build taxonomy issues per system.
    taxonomy_issues_map: Dict[
        Tuple[str, Optional[int]], List[str]
    ] = {}
    for sys_key in sorted(
        system_classes.keys(), key=lambda k: (k[0], str(k[1]))
    ):
        sc = system_classes[sys_key]
        bm = best_mode_map.get(sys_key, "none")
        recs_for_sys = system_records.get(sys_key, [])
        taxonomy_issues_map[sys_key] = detect_taxonomy_issues(
            sc, bm, recs_for_sys,
        )

    # Merge taxonomy issues into the first issue record per system.
    seen_systems: set = set()
    for ir in issues:
        sys_key = (ir["dfa_type"], ir["n"])
        skey = (ir["dfa_type"], str(ir["n"]))
        if skey not in seen_systems:
            seen_systems.add(skey)
            tax_issues = taxonomy_issues_map.get(sys_key, [])
            ir["issues"].extend(tax_issues)

    recs = recommend_all(issues)

    # Convert system_classes to serializable list format.
    sc_list = []
    for key in sorted(
        system_classes.keys(), key=lambda k: (k[0], str(k[1]))
    ):
        sc_list.append({
            "dfa_type": key[0],
            "n": key[1],
            "system_class": system_classes[key],
        })

    return {
        "metrics": agg,
        "best_modes": best_modes,
        "system_classes": sc_list,
        "issues": issues,
        "recommendations": recs,
    }


# ---------------------------------------------------------------------------
# PART 6 — PRINT DIAGNOSTICS
# ---------------------------------------------------------------------------


def print_diagnostics(report: Dict[str, Any]) -> str:
    """Format diagnostics report as readable text.

    Deterministic, sorted by (dfa_type, str(n)).
    """
    lines: List[str] = []

    best_map: Dict[Tuple[str, Optional[int]], str] = {}
    for b in report["best_modes"]:
        best_map[(b["dfa_type"], b["n"])] = b["best_mode"]

    class_map: Dict[Tuple[str, Optional[int]], str] = {}
    for sc in report.get("system_classes", []):
        class_map[(sc["dfa_type"], sc["n"])] = sc["system_class"]

    issue_map: Dict[Tuple[str, Optional[int], str], List[str]] = {}
    for ir in report["issues"]:
        issue_map[(ir["dfa_type"], ir["n"], ir["mode"])] = ir["issues"]

    rec_map: Dict[Tuple[str, Optional[int]], List[str]] = {}
    for r in report["recommendations"]:
        key = (r["dfa_type"], r["n"])
        rec_map.setdefault(key, []).extend(r["recommendations"])

    # Collect unique systems.
    systems: List[Tuple[str, Optional[int]]] = []
    seen = set()
    for m in report["metrics"]:
        key = (m["dfa_type"], m["n"])
        skey = (m["dfa_type"], str(m["n"]))
        if skey not in seen:
            seen.add(skey)
            systems.append(key)
    systems.sort(key=lambda k: (k[0], str(k[1])))

    for dfa_type, n in systems:
        n_label = "NA" if n is None else str(n)
        lines.append(f"DFA: {dfa_type} (n={n_label})")
        sys_class = class_map.get((dfa_type, n))
        if sys_class is not None:
            lines.append(f"  class: {sys_class}")
        lines.append(f"  best_mode: {best_map.get((dfa_type, n), 'N/A')}")

        # Collect all issues for this system across modes.
        sys_issues: List[str] = []
        for m in report["metrics"]:
            if m["dfa_type"] == dfa_type and m["n"] == n:
                key = (dfa_type, n, m["mode"])
                sys_issues.extend(issue_map.get(key, []))
        # Deduplicate while preserving order.
        unique_issues: List[str] = []
        issue_seen: set = set()
        for iss in sys_issues:
            if iss not in issue_seen:
                issue_seen.add(iss)
                unique_issues.append(iss)
        lines.append(f"  issues: {unique_issues}")

        sys_recs = rec_map.get((dfa_type, n), [])
        unique_recs: List[str] = []
        rec_seen: set = set()
        for r in sys_recs:
            if r not in rec_seen:
                rec_seen.add(r)
                unique_recs.append(r)
        lines.append(f"  recommendations: {unique_recs}")
        lines.append("")

    return "\n".join(lines)
