"""Self-diagnostics and deterministic recommendation layer (v93.0.0).

Converts benchmark + correction outputs into:
  metrics -> issue detection -> deterministic recommendations

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
# PART 3 — ISSUE DETECTION
# ---------------------------------------------------------------------------

# Deterministic issue-to-recommendation mapping.
_RECOMMENDATION_MAP = {
    "no_effect": "increase_structure_guidance",
    "over_correction": "reduce_projection_strength",
    "destabilizing": "switch_to_d4_or_invariant",
    "low_diversity": "skip_correction_or_expand_input",
    "globally_weak_correction": "enable_invariant_guidance",
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
    Returns dict with metrics, best_modes, issues, recommendations.
    """
    records = normalize_results(data)
    agg = aggregate_metrics(records)
    best_modes = best_mode_per_system(agg)
    issues = detect_all(agg, best_modes)
    recs = recommend_all(issues)
    return {
        "metrics": agg,
        "best_modes": best_modes,
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
