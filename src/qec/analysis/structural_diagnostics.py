"""Structural gap detection and invariant opportunity mapping (v94.1.0).

Extends the diagnostics pipeline from:
    diagnostics -> recommendations -> adjustment
to:
    diagnostics -> structural gap detection -> invariant opportunities

Analysis only — does not synthesize actual invariants or modify behavior.
All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No optimization loops.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    classify_all_systems,
    normalize_results,
    run_self_diagnostics,
)


# ---------------------------------------------------------------------------
# PART 1 — STRUCTURAL GAP DETECTION
# ---------------------------------------------------------------------------


def _group_by_system(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]]:
    """Group aggregated records by (dfa_type, n). Deterministic ordering."""
    systems: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["dfa_type"], r["n"])
        systems.setdefault(key, []).append(r)
    return systems


def _get_mode_value(
    records: List[Dict[str, Any]],
    mode: str,
    field: str,
) -> Optional[float]:
    """Extract a metric value for a specific mode. Returns None if absent."""
    for r in records:
        if r["mode"] == mode:
            return float(r[field])
    return None


def _detect_unstable_correction_region(
    dfa_type: str,
    n: Optional[int],
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect if stability_gain < 0 or oscillating behavior across modes."""
    gains = []
    for r in records:
        g = r.get("stability_gain", 0)
        gains.append((r["mode"], g))

    negative_gains = [(m, g) for m, g in gains if g < 0]
    if negative_gains:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "unstable_correction_region",
            "evidence": {
                "negative_gain_modes": [m for m, _ in negative_gains],
                "gains": {m: g for m, g in gains},
            },
        }

    # Oscillating: both positive and negative gains present (sign changes).
    positive = sum(1 for _, g in gains if g > 0)
    negative = sum(1 for _, g in gains if g < 0)
    if positive > 0 and negative > 0:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "unstable_correction_region",
            "evidence": {
                "oscillating": True,
                "gains": {m: g for m, g in gains},
            },
        }
    return None


def _detect_weak_compression_structure(
    dfa_type: str,
    n: Optional[int],
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect if compression_efficiency < 0.1 across all modes."""
    comps = {}
    for r in records:
        comps[r["mode"]] = r["compression_efficiency"]

    all_weak = all(c < 0.1 for c in comps.values())
    if all_weak and comps:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "weak_compression_structure",
            "evidence": {
                "compression_by_mode": dict(sorted(comps.items())),
                "max_compression": max(comps.values()),
            },
        }
    return None


def _detect_mode_disagreement(
    dfa_type: str,
    n: Optional[int],
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect large variance in stability_efficiency across modes."""
    stabs = {}
    for r in records:
        stabs[r["mode"]] = r["stability_efficiency"]

    values = list(stabs.values())
    if len(values) < 2:
        return None

    min_val = min(values)
    max_val = max(values)
    spread = max_val - min_val

    if spread > 0.4:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "mode_disagreement",
            "evidence": {
                "stability_by_mode": dict(sorted(stabs.items())),
                "spread": spread,
            },
        }
    return None


def _detect_invariant_dependency(
    dfa_type: str,
    n: Optional[int],
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect if d4+inv >> d4 (large improvement from invariant guidance)."""
    d4_stab = _get_mode_value(records, "d4", "stability_efficiency")
    d4_inv_stab = _get_mode_value(records, "d4+inv", "stability_efficiency")

    if d4_stab is None or d4_inv_stab is None:
        return None

    delta = d4_inv_stab - d4_stab
    if delta > 0.3:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "invariant_dependency",
            "evidence": {
                "d4_stability": d4_stab,
                "d4_inv_stability": d4_inv_stab,
                "delta": delta,
            },
        }
    return None


def _detect_overcorrection_pattern(
    dfa_type: str,
    n: Optional[int],
    records: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Detect high compression + low stability (overcorrection)."""
    overcorrected = []
    for r in records:
        comp = r["compression_efficiency"]
        stab = r["stability_efficiency"]
        if comp > 0.6 and stab < 0.15:
            overcorrected.append(r["mode"])

    if overcorrected:
        return {
            "dfa_type": dfa_type,
            "n": n,
            "gap_type": "overcorrection_pattern",
            "evidence": {
                "overcorrected_modes": sorted(overcorrected),
                "metrics": {
                    r["mode"]: {
                        "compression": r["compression_efficiency"],
                        "stability": r["stability_efficiency"],
                    }
                    for r in records
                    if r["mode"] in overcorrected
                },
            },
        }
    return None


_GAP_DETECTORS = [
    _detect_unstable_correction_region,
    _detect_weak_compression_structure,
    _detect_mode_disagreement,
    _detect_invariant_dependency,
    _detect_overcorrection_pattern,
]


def detect_structural_gaps(
    system_records: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]],
    diagnostics: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect structural gaps across all systems.

    Args:
        system_records: dict mapping (dfa_type, n) -> list of mode records.
        diagnostics: output from run_self_diagnostics().

    Returns:
        List of gap dicts, each with dfa_type, n, gap_type, evidence.
        Deterministically ordered by (dfa_type, n, gap_type).
    """
    gaps: List[Dict[str, Any]] = []

    for key in sorted(
        system_records.keys(), key=lambda k: (k[0], str(k[1]))
    ):
        dfa_type, n = key
        records = system_records[key]

        for detector in _GAP_DETECTORS:
            gap = detector(dfa_type, n, records)
            if gap is not None:
                gaps.append(gap)

    return gaps


# ---------------------------------------------------------------------------
# PART 2 — INVARIANT OPPORTUNITY MAPPING
# ---------------------------------------------------------------------------


_GAP_TO_INVARIANT: Dict[str, str] = {
    "unstable_correction_region": "local_stability_constraint",
    "weak_compression_structure": "equivalence_class_constraint",
    "mode_disagreement": "geometry_alignment_constraint",
    "invariant_dependency": "explicit_allowed_state_constraint",
    "overcorrection_pattern": "bounded_projection_constraint",
}

# Gaps with strong signal evidence get high confidence.
_HIGH_CONFIDENCE_GAPS = frozenset({
    "unstable_correction_region",
    "mode_disagreement",
    "invariant_dependency",
})


def map_gap_to_invariant(gap_type: str) -> Optional[Dict[str, str]]:
    """Map a gap type to a suggested invariant and confidence level.

    Returns None if gap_type is unrecognized.
    Confidence: "high" for strong metric signals, "medium" for borderline.
    """
    suggested = _GAP_TO_INVARIANT.get(gap_type)
    if suggested is None:
        return None

    confidence = (
        "high" if gap_type in _HIGH_CONFIDENCE_GAPS else "medium"
    )

    return {
        "gap_type": gap_type,
        "suggested_invariant": suggested,
        "confidence": confidence,
    }


def map_all_gaps_to_invariants(
    gaps: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map all detected gaps to invariant opportunities.

    Returns list of invariant opportunity dicts with:
        dfa_type, n, gap_type, suggested_invariant, confidence.
    Preserves input ordering (deterministic).
    """
    opportunities: List[Dict[str, Any]] = []
    for gap in gaps:
        mapping = map_gap_to_invariant(gap["gap_type"])
        if mapping is not None:
            opportunities.append({
                "dfa_type": gap["dfa_type"],
                "n": gap["n"],
                **mapping,
            })
    return opportunities


# ---------------------------------------------------------------------------
# PART 3 — SYSTEM-WIDE REPORT
# ---------------------------------------------------------------------------


def run_structural_diagnostics(data: Any) -> Dict[str, Any]:
    """Full structural diagnostics pipeline.

    Pipeline:
        1. normalize results
        2. run existing diagnostics (classification, issues, recommendations)
        3. detect structural gaps
        4. map gaps to invariant opportunities

    Accepts run_suite() or summarize() output.
    Returns dict with system_classes, gaps, invariant_opportunities.
    """
    # Step 1-2: reuse existing diagnostics pipeline.
    diag = run_self_diagnostics(data)

    # Build system records from aggregated metrics.
    records = normalize_results(data)
    agg = aggregate_metrics(records)
    system_records = _group_by_system(agg)

    # Step 3: detect gaps.
    gaps = detect_structural_gaps(system_records, diag)

    # Step 4: map to invariant opportunities.
    invariant_opportunities = map_all_gaps_to_invariants(gaps)

    return {
        "system_classes": diag["system_classes"],
        "gaps": gaps,
        "invariant_opportunities": invariant_opportunities,
    }


# ---------------------------------------------------------------------------
# PART 4 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_structural_report(report: Dict[str, Any]) -> str:
    """Format structural diagnostics report as human-readable text.

    Returns the formatted string (does not print to stdout).
    Deterministic output ordering.
    """
    lines: List[str] = []
    lines.append("=== Structural Diagnostics Report ===")
    lines.append("")

    # Group gaps and opportunities by (dfa_type, n).
    gap_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for gap in report.get("gaps", []):
        key = (gap["dfa_type"], str(gap["n"]))
        gap_map.setdefault(key, []).append(gap)

    opp_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for opp in report.get("invariant_opportunities", []):
        key = (opp["dfa_type"], str(opp["n"]))
        opp_map.setdefault(key, []).append(opp)

    # Build system class lookup.
    class_map: Dict[Tuple[str, str], str] = {}
    for sc in report.get("system_classes", []):
        key = (sc["dfa_type"], str(sc["n"]))
        class_map[key] = sc["system_class"]

    # Collect all system keys deterministically.
    all_keys: set = set()
    all_keys.update(gap_map.keys())
    all_keys.update(opp_map.keys())
    all_keys.update(class_map.keys())

    for key in sorted(all_keys):
        dfa_type, n_str = key
        sys_class = class_map.get(key, "unknown")
        lines.append(f"DFA: {dfa_type} (n={n_str})")
        lines.append(f"  class: {sys_class}")

        for gap in gap_map.get(key, []):
            lines.append(f"  gap: {gap['gap_type']}")

        for opp in opp_map.get(key, []):
            lines.append(
                f"    suggested_invariant: {opp['suggested_invariant']}"
            )
            lines.append(f"    confidence: {opp['confidence']}")

        lines.append("")

    if not all_keys:
        lines.append("No structural gaps detected.")
        lines.append("")

    return "\n".join(lines)
