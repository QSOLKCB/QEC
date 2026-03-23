"""Invariant synthesis and deterministic validation loop (v95.0.0).

Upgrades the diagnostics pipeline from:
    gap -> suggested invariant type
to:
    gap -> generate candidate invariant -> evaluate -> accept/reject

Proposal and validation only — does not modify correction engine,
benchmark logic, or decoder internals.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No ML. No symbolic algebra.
No search loops. Templates only. Max 2-3 candidates per gap.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    normalize_results,
    run_self_diagnostics,
)
from qec.analysis.structural_diagnostics import (
    _group_by_system,
    detect_structural_gaps,
    map_all_gaps_to_invariants,
    run_structural_diagnostics,
)


# ---------------------------------------------------------------------------
# PART 1 — INVARIANT TEMPLATES
# ---------------------------------------------------------------------------


def _generate_local_stability_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate candidates for unstable_correction_region gaps.

    Propose constraints that limit transitions through unstable modes.
    At most 2 candidates: one hard, one soft.
    """
    evidence = gap.get("evidence", {})
    negative_modes = evidence.get("negative_gain_modes", [])
    gains = evidence.get("gains", {})

    candidates: List[Dict[str, Any]] = []

    # Candidate 1: hard constraint — exclude negative-gain modes.
    if negative_modes:
        candidates.append({
            "type": "local_stability_constraint",
            "rule": {
                "limit_transition": sorted(negative_modes),
                "action": "exclude_negative_gain_modes",
            },
            "strength": "hard",
        })

    # Candidate 2: soft constraint — prefer modes with highest stability.
    stable_modes = sorted(
        [(m, g) for m, g in gains.items() if g >= 0],
        key=lambda x: (-x[1], x[0]),
    )
    if stable_modes:
        preferred = [m for m, _ in stable_modes]
        candidates.append({
            "type": "local_stability_constraint",
            "rule": {
                "prefer_stable_modes": preferred,
                "action": "prefer_positive_gain",
            },
            "strength": "soft",
        })

    return candidates[:3]


def _generate_equivalence_class_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate candidates for weak_compression_structure gaps.

    Propose equivalence groupings to improve compression.
    """
    evidence = gap.get("evidence", {})
    comp_by_mode = evidence.get("compression_by_mode", {})

    candidates: List[Dict[str, Any]] = []

    # Identify modes that can be grouped by similar compression values.
    mode_list = sorted(comp_by_mode.items(), key=lambda x: (x[1], x[0]))

    # Candidate 1: group all non-zero modes as equivalent.
    non_zero = [(m, c) for m, c in mode_list if c > 0.0]
    zero = [(m, c) for m, c in mode_list if c == 0.0]
    if non_zero and zero:
        candidates.append({
            "type": "equivalence_class_constraint",
            "rule": {
                "group_states": [
                    [m for m, _ in non_zero],
                    [m for m, _ in zero],
                ],
                "action": "group_by_compression_presence",
            },
            "strength": "soft",
        })

    # Candidate 2: prefer invariant-guided mode for better compression.
    inv_modes = [m for m, _ in mode_list if "inv" in m]
    if inv_modes:
        candidates.append({
            "type": "equivalence_class_constraint",
            "rule": {
                "group_states": [inv_modes],
                "action": "prefer_invariant_compression",
            },
            "strength": "soft",
        })

    # Candidate 3 (fallback): enable structure guidance.
    if not candidates:
        candidates.append({
            "type": "equivalence_class_constraint",
            "rule": {
                "group_states": [],
                "action": "enable_structure_guidance",
            },
            "strength": "soft",
        })

    return candidates[:3]


def _generate_geometry_alignment_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate candidates for mode_disagreement gaps.

    Propose geometry alignment to reduce mode variance.
    """
    evidence = gap.get("evidence", {})
    stab_by_mode = evidence.get("stability_by_mode", {})

    candidates: List[Dict[str, Any]] = []

    # Find best mode by stability.
    if stab_by_mode:
        best_mode = max(
            sorted(stab_by_mode.keys()),
            key=lambda m: stab_by_mode[m],
        )
        best_stab = stab_by_mode[best_mode]

        # Candidate 1: align to d4 if it is best or close to best.
        d4_stab = stab_by_mode.get("d4", 0.0)
        if d4_stab >= best_stab * 0.8:
            candidates.append({
                "type": "geometry_alignment_constraint",
                "rule": {
                    "align_projection": "d4",
                    "reference_stability": d4_stab,
                },
                "strength": "hard",
            })

        # Candidate 2: align to square.
        sq_stab = stab_by_mode.get("square", 0.0)
        if sq_stab >= best_stab * 0.8:
            candidates.append({
                "type": "geometry_alignment_constraint",
                "rule": {
                    "align_projection": "square",
                    "reference_stability": sq_stab,
                },
                "strength": "hard",
            })

        # Candidate 3: align to whatever is best.
        if not candidates:
            candidates.append({
                "type": "geometry_alignment_constraint",
                "rule": {
                    "align_projection": best_mode,
                    "reference_stability": best_stab,
                },
                "strength": "soft",
            })

    return candidates[:3]


def _generate_explicit_allowed_state_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate candidates for invariant_dependency gaps.

    Propose explicit allowed-state constraints from invariant-guided modes.
    """
    evidence = gap.get("evidence", {})
    d4_inv_stab = evidence.get("d4_inv_stability", 0.0)
    d4_stab = evidence.get("d4_stability", 0.0)

    candidates: List[Dict[str, Any]] = []

    # Candidate 1: require invariant guidance (d4+inv modes).
    candidates.append({
        "type": "explicit_allowed_state_constraint",
        "rule": {
            "allowed_states": ["d4+inv"],
            "action": "require_invariant_guidance",
        },
        "strength": "hard",
    })

    # Candidate 2: allow d4+inv and d4 as fallback.
    candidates.append({
        "type": "explicit_allowed_state_constraint",
        "rule": {
            "allowed_states": ["d4+inv", "d4"],
            "action": "prefer_invariant_with_d4_fallback",
        },
        "strength": "soft",
    })

    return candidates[:3]


def _generate_bounded_projection_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate candidates for overcorrection_pattern gaps.

    Propose bounded projection constraints to avoid overcorrection.
    """
    evidence = gap.get("evidence", {})
    metrics = evidence.get("metrics", {})

    candidates: List[Dict[str, Any]] = []

    # Compute max compression among overcorrected modes.
    max_comp = 0.0
    for mode_name in sorted(metrics.keys()):
        m = metrics[mode_name]
        comp = m.get("compression", 0.0)
        if comp > max_comp:
            max_comp = comp

    # Candidate 1: cap projection distance at 50% of max compression.
    if max_comp > 0.0:
        candidates.append({
            "type": "bounded_projection_constraint",
            "rule": {
                "max_projection_distance": round(max_comp * 0.5, 6),
                "action": "cap_compression",
            },
            "strength": "hard",
        })

    # Candidate 2: prefer modes that balance compression and stability.
    # Select modes from records where stability > compression.
    balanced = []
    for r in records:
        if r["stability_efficiency"] >= r["compression_efficiency"]:
            balanced.append(r["mode"])
    if balanced:
        candidates.append({
            "type": "bounded_projection_constraint",
            "rule": {
                "prefer_balanced_modes": sorted(set(balanced)),
                "action": "prefer_stability_over_compression",
            },
            "strength": "soft",
        })

    # Fallback if no candidates yet.
    if not candidates:
        candidates.append({
            "type": "bounded_projection_constraint",
            "rule": {
                "max_projection_distance": 0.5,
                "action": "default_cap",
            },
            "strength": "soft",
        })

    return candidates[:3]


# Template registry: gap_type -> generator function.
INVARIANT_TEMPLATES: Dict[str, Any] = {
    "local_stability_constraint": _generate_local_stability_candidates,
    "equivalence_class_constraint": _generate_equivalence_class_candidates,
    "geometry_alignment_constraint": _generate_geometry_alignment_candidates,
    "explicit_allowed_state_constraint": _generate_explicit_allowed_state_candidates,
    "bounded_projection_constraint": _generate_bounded_projection_candidates,
}

# Maps gap_type -> invariant template key.
_GAP_TO_TEMPLATE: Dict[str, str] = {
    "unstable_correction_region": "local_stability_constraint",
    "weak_compression_structure": "equivalence_class_constraint",
    "mode_disagreement": "geometry_alignment_constraint",
    "invariant_dependency": "explicit_allowed_state_constraint",
    "overcorrection_pattern": "bounded_projection_constraint",
}


# ---------------------------------------------------------------------------
# PART 2 — CANDIDATE GENERATION
# ---------------------------------------------------------------------------


def generate_candidates(
    gap: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate invariant candidates for a single gap.

    Args:
        gap: a gap dict with gap_type, dfa_type, n, evidence.
        records: mode records for the system (dfa_type, n).

    Returns:
        List of candidate dicts, max 3. Deterministic ordering.
        Each candidate has: type, rule, strength, gap_type, dfa_type, n.
    """
    gap_type = gap.get("gap_type", "")
    template_key = _GAP_TO_TEMPLATE.get(gap_type)
    if template_key is None:
        return []

    generator = INVARIANT_TEMPLATES.get(template_key)
    if generator is None:
        return []

    raw_candidates = generator(gap, records)

    # Annotate each candidate with source gap info.
    candidates: List[Dict[str, Any]] = []
    for c in raw_candidates[:3]:
        candidates.append({
            **c,
            "gap_type": gap_type,
            "dfa_type": gap.get("dfa_type", ""),
            "n": gap.get("n"),
        })

    return candidates


# ---------------------------------------------------------------------------
# PART 3 — SIMULATED APPLICATION (PROXY EVALUATION)
# ---------------------------------------------------------------------------


def apply_candidate_proxy(
    candidate: Dict[str, Any],
    system_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Simulate the effect of a candidate invariant on system records.

    Does NOT modify any records. Selects among existing mode results
    to estimate what metrics the candidate would produce.

    Returns proxy metrics: stability_efficiency, compression_efficiency,
    stability_gain.
    """
    ctype = candidate.get("type", "")
    rule = candidate.get("rule", {})

    if not system_records:
        return {
            "stability_efficiency": 0.0,
            "compression_efficiency": 0.0,
            "stability_gain": 0.0,
        }

    if ctype == "local_stability_constraint":
        # Prefer modes with higher stability_efficiency.
        exclude = set(rule.get("limit_transition", []))
        filtered = [r for r in system_records if r["mode"] not in exclude]
        if not filtered:
            filtered = list(system_records)
        best = max(
            filtered,
            key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
                r["stability_gain"],
            ),
        )
        return _extract_proxy_metrics(best)

    if ctype == "equivalence_class_constraint":
        # Simulate improved compression by selecting best compression mode.
        best = max(
            system_records,
            key=lambda r: (
                r["compression_efficiency"],
                r["stability_efficiency"],
                r["stability_gain"],
            ),
        )
        return _extract_proxy_metrics(best)

    if ctype == "geometry_alignment_constraint":
        # Force the aligned projection mode.
        align = rule.get("align_projection", "")
        for r in system_records:
            if r["mode"] == align:
                return _extract_proxy_metrics(r)
        # Fallback: best stability.
        best = max(
            system_records,
            key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
            ),
        )
        return _extract_proxy_metrics(best)

    if ctype == "explicit_allowed_state_constraint":
        # Prefer invariant-guided modes.
        allowed = set(rule.get("allowed_states", []))
        filtered = [r for r in system_records if r["mode"] in allowed]
        if not filtered:
            filtered = list(system_records)
        best = max(
            filtered,
            key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
                r["stability_gain"],
            ),
        )
        return _extract_proxy_metrics(best)

    if ctype == "bounded_projection_constraint":
        # Avoid overcorrection: prefer modes where stability >= compression.
        balanced = [
            r for r in system_records
            if r["stability_efficiency"] >= r["compression_efficiency"]
        ]
        if not balanced:
            balanced = list(system_records)
        best = max(
            balanced,
            key=lambda r: (
                r["stability_efficiency"],
                r["compression_efficiency"],
            ),
        )
        return _extract_proxy_metrics(best)

    # Unknown type: return best overall stability.
    best = max(
        system_records,
        key=lambda r: (
            r["stability_efficiency"],
            r["compression_efficiency"],
        ),
    )
    return _extract_proxy_metrics(best)


def _extract_proxy_metrics(record: Dict[str, Any]) -> Dict[str, float]:
    """Extract the three comparison metrics from a record."""
    return {
        "stability_efficiency": float(
            record.get("stability_efficiency", 0.0)
        ),
        "compression_efficiency": float(
            record.get("compression_efficiency", 0.0)
        ),
        "stability_gain": float(record.get("stability_gain", 0.0)),
    }


# ---------------------------------------------------------------------------
# PART 4 — VALIDATION (EVALUATE + ACCEPT/REJECT)
# ---------------------------------------------------------------------------


def _is_improvement(
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
        return True, "improved_stability"
    if a_stab < b_stab:
        return False, "no_improvement"

    if a_comp > b_comp:
        return True, "improved_compression"
    if a_comp < b_comp:
        return False, "no_improvement"

    if a_gain > b_gain:
        return True, "improved_stability_gain"
    return False, "no_improvement"


def evaluate_candidate(
    candidate: Dict[str, Any],
    current_best_metrics: Dict[str, float],
    system_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a single candidate against current best metrics.

    Args:
        candidate: candidate invariant dict.
        current_best_metrics: metrics of the current best mode.
        system_records: all mode records for this system.

    Returns:
        Evaluation dict with candidate, accepted, reason, improvement.
    """
    proxy_metrics = apply_candidate_proxy(candidate, system_records)
    accepted, reason = _is_improvement(current_best_metrics, proxy_metrics)

    return {
        "candidate": candidate,
        "accepted": accepted,
        "reason": reason,
        "improvement": {
            "before": dict(current_best_metrics),
            "after": dict(proxy_metrics),
        },
    }


# ---------------------------------------------------------------------------
# PART 5 — PER-SYSTEM SYNTHESIS
# ---------------------------------------------------------------------------


def _current_best_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Find the best mode metrics for a system using the standard hierarchy."""
    if not records:
        return {
            "stability_efficiency": 0.0,
            "compression_efficiency": 0.0,
            "stability_gain": 0.0,
        }
    best = max(
        records,
        key=lambda r: (
            r["stability_efficiency"],
            r["compression_efficiency"],
            r.get("stability_gain", 0),
        ),
    )
    return _extract_proxy_metrics(best)


def synthesize_for_system(
    system_records: List[Dict[str, Any]],
    gaps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run candidate generation and evaluation for one system.

    Args:
        system_records: mode records for one (dfa_type, n).
        gaps: gaps detected for this system.

    Returns:
        Dict with candidates, evaluations, accepted_invariants.
    """
    current_best = _current_best_metrics(system_records)

    all_candidates: List[Dict[str, Any]] = []
    all_evaluations: List[Dict[str, Any]] = []
    accepted_invariants: List[Dict[str, Any]] = []

    for gap in gaps:
        candidates = generate_candidates(gap, system_records)
        all_candidates.extend(candidates)

        # Evaluate each and keep only the best accepted one per gap.
        best_eval: Optional[Dict[str, Any]] = None
        for c in candidates:
            ev = evaluate_candidate(c, current_best, system_records)
            all_evaluations.append(ev)
            if ev["accepted"]:
                if best_eval is None:
                    best_eval = ev
                else:
                    # Compare proxy metrics: pick better one.
                    prev = best_eval["improvement"]["after"]
                    curr = ev["improvement"]["after"]
                    is_better, _ = _is_improvement(prev, curr)
                    if is_better:
                        best_eval = ev

        if best_eval is not None:
            accepted_invariants.append(best_eval)

    return {
        "candidates": all_candidates,
        "evaluations": all_evaluations,
        "accepted_invariants": accepted_invariants,
    }


# ---------------------------------------------------------------------------
# PART 6 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_invariant_synthesis(data: Any) -> Dict[str, Any]:
    """Full invariant synthesis pipeline.

    Accepts:
      - raw run_suite() results (list)
      - summarize() output (dict)

    Pipeline:
      1. normalize results
      2. run structural diagnostics (gaps + invariant opportunities)
      3. for each system: synthesize candidates, evaluate, accept/reject

    Returns:
      Dict with gaps, candidates, evaluations, accepted_invariants.
    """
    # Step 1-2: run structural diagnostics.
    structural = run_structural_diagnostics(data)

    # Build system records.
    records = normalize_results(data)
    agg = aggregate_metrics(records)
    system_records = _group_by_system(agg)

    # Index gaps by system.
    gaps_by_system: Dict[
        Tuple[str, Optional[int]], List[Dict[str, Any]]
    ] = {}
    for gap in structural.get("gaps", []):
        key = (gap["dfa_type"], gap["n"])
        gaps_by_system.setdefault(key, []).append(gap)

    # Step 3: synthesize per system.
    all_candidates: List[Dict[str, Any]] = []
    all_evaluations: List[Dict[str, Any]] = []
    all_accepted: List[Dict[str, Any]] = []

    for key in sorted(
        system_records.keys(), key=lambda k: (k[0], str(k[1]))
    ):
        sys_records = system_records[key]
        sys_gaps = gaps_by_system.get(key, [])

        if not sys_gaps:
            continue

        result = synthesize_for_system(sys_records, sys_gaps)
        all_candidates.extend(result["candidates"])
        all_evaluations.extend(result["evaluations"])
        all_accepted.extend(result["accepted_invariants"])

    return {
        "structural_diagnostics": structural,
        "gaps": structural.get("gaps", []),
        "candidates": all_candidates,
        "evaluations": all_evaluations,
        "accepted_invariants": all_accepted,
    }


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_invariant_report(report: Dict[str, Any]) -> str:
    """Format invariant synthesis report as human-readable text.

    Deterministic, sorted, text-only.
    """
    lines: List[str] = []
    lines.append("=== Invariant Synthesis Report ===")
    lines.append("")

    gaps = report.get("gaps", [])
    evaluations = report.get("evaluations", [])
    accepted = report.get("accepted_invariants", [])

    # Group evaluations by (dfa_type, n, gap_type).
    eval_by_gap: Dict[
        Tuple[str, str, str], List[Dict[str, Any]]
    ] = {}
    for ev in evaluations:
        c = ev["candidate"]
        key = (c.get("dfa_type", ""), str(c.get("n", "")), c.get("gap_type", ""))
        eval_by_gap.setdefault(key, []).append(ev)

    # Accepted candidate types for quick lookup.
    accepted_keys: set = set()
    for a in accepted:
        c = a["candidate"]
        accepted_keys.add(
            (c.get("dfa_type", ""), str(c.get("n", "")),
             c.get("gap_type", ""), c.get("type", ""))
        )

    # Print per system + gap.
    seen_systems: set = set()
    for gap in gaps:
        dfa_type = gap["dfa_type"]
        n = gap["n"]
        sys_key = (dfa_type, str(n))

        if sys_key not in seen_systems:
            if seen_systems:
                lines.append("")
            lines.append(f"DFA: {dfa_type} (n={n})")
            seen_systems.add(sys_key)

        gap_type = gap["gap_type"]
        lines.append(f"  gap: {gap_type}")

        ev_key = (dfa_type, str(n), gap_type)
        gap_evals = eval_by_gap.get(ev_key, [])
        for ev in gap_evals:
            c = ev["candidate"]
            ctype = c.get("type", "unknown")
            is_accepted = (
                dfa_type, str(n), gap_type, ctype
            ) in accepted_keys
            lines.append(f"    candidate: {ctype}")
            lines.append(f"      strength: {c.get('strength', 'unknown')}")
            lines.append(f"      accepted: {is_accepted}")
            lines.append(f"      reason: {ev['reason']}")

    if not gaps:
        lines.append("No gaps detected — no candidates generated.")

    lines.append("")

    # Summary.
    lines.append(f"Total gaps: {len(gaps)}")
    lines.append(f"Total candidates: {len(report.get('candidates', []))}")
    lines.append(f"Accepted invariants: {len(accepted)}")

    return "\n".join(lines)
