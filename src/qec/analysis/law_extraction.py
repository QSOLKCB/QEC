"""Deterministic law extraction and rulebook generation (v96.2.0).

Upgrades from:
    analysis -> ranking -> interaction patterns
to:
    analysis -> law extraction -> explicit rulebook

Extracts evidence-backed, template-based rules from upstream analysis
reports (invariant_analysis, core_invariants, structural_diagnostics).

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No probabilistic scoring.
No free-form prose generation.
"""

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LAW TEMPLATES
# ---------------------------------------------------------------------------

LAW_TEMPLATES = [
    "class_invariant_law",
    "class_hierarchy_law",
    "interaction_law",
    "gap_invariant_law",
    "core_invariant_law",
]


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_law_inputs(data: Any) -> List[Dict[str, Any]]:
    """Normalize one or more upstream reports to flat evidence records.

    Accepts:
      - invariant_analysis output (dict with 'class_effectiveness', etc.)
      - core_invariants / hierarchical pipeline output
      - structural_diagnostics output (dict with 'gaps', etc.)

    Returns list of flat evidence records:
        {
          "dfa_type": str,
          "n": Optional[int],
          "system_class": str,
          "invariant_type": Optional[str],
          "hierarchical_mode": Optional[str],
          "improved": bool,
          "avg_score": Optional[float],
          "interaction_type": Optional[str],
          "gap_type": Optional[str],
        }

    Deterministic ordering. No mutation of inputs.
    """
    if not isinstance(data, dict):
        return []

    records: List[Dict[str, Any]] = []

    # --- Path A: invariant_analysis class_effectiveness ---
    records.extend(_extract_from_class_effectiveness(data))

    # --- Path B: interactions ---
    records.extend(_extract_from_interactions(data))

    # --- Path C: structural_diagnostics gaps / invariant_opportunities ---
    records.extend(_extract_from_structural(data))

    # --- Path D: core_invariants pipeline ---
    records.extend(_extract_from_core_invariants(data))

    # --- Path E: hierarchical pipeline ---
    records.extend(_extract_from_hierarchical(data))

    # Deterministic sort.
    records.sort(key=_record_sort_key)
    return records


def _record_sort_key(r: Dict[str, Any]) -> Tuple:
    """Stable sort key for evidence records."""
    return (
        r.get("dfa_type", ""),
        str(r.get("n", "")),
        r.get("system_class", ""),
        r.get("invariant_type", "") or "",
        r.get("hierarchical_mode", "") or "",
        r.get("interaction_type", "") or "",
        r.get("gap_type", "") or "",
    )


def _make_evidence_record(
    dfa_type: str = "",
    n: Optional[int] = None,
    system_class: str = "",
    invariant_type: Optional[str] = None,
    hierarchical_mode: Optional[str] = None,
    improved: bool = False,
    avg_score: Optional[float] = None,
    interaction_type: Optional[str] = None,
    gap_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical evidence record."""
    return {
        "dfa_type": dfa_type,
        "n": n,
        "system_class": system_class,
        "invariant_type": invariant_type,
        "hierarchical_mode": hierarchical_mode,
        "improved": improved,
        "avg_score": avg_score,
        "interaction_type": interaction_type,
        "gap_type": gap_type,
    }


def _extract_from_class_effectiveness(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract evidence from invariant_analysis class_effectiveness."""
    records: List[Dict[str, Any]] = []
    ce = data.get("class_effectiveness", {})

    for sys_class in sorted(ce.keys()):
        inv_map = ce[sys_class]
        for inv_type in sorted(inv_map.keys()):
            metrics = inv_map[inv_type]
            improved_ratio = float(metrics.get("improved_ratio", 0.0))
            avg_score = float(metrics.get("avg_score", 0.0))
            count = int(metrics.get("count", 0))

            for _ in range(count):
                records.append(_make_evidence_record(
                    system_class=sys_class,
                    invariant_type=inv_type,
                    improved=improved_ratio >= 0.5,
                    avg_score=avg_score,
                ))

    return records


def _extract_from_interactions(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract evidence from invariant_analysis interactions."""
    records: List[Dict[str, Any]] = []
    interactions = data.get("interactions", [])

    for ix in interactions:
        pair = ix.get("pair", ())
        if len(pair) != 2:
            continue
        ix_type = ix.get("type", "")
        count = int(ix.get("evidence_count", 1))

        for _ in range(count):
            records.append(_make_evidence_record(
                invariant_type=",".join(sorted(pair)),
                interaction_type=ix_type,
            ))

    return records


def _extract_from_structural(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract evidence from structural_diagnostics output."""
    records: List[Dict[str, Any]] = []

    opportunities = data.get("invariant_opportunities", [])
    for opp in opportunities:
        records.append(_make_evidence_record(
            dfa_type=opp.get("dfa_type", ""),
            n=opp.get("n"),
            gap_type=opp.get("gap_type", ""),
            invariant_type=opp.get("suggested_invariant", ""),
            improved=opp.get("confidence", "") == "high",
        ))

    gaps = data.get("gaps", [])
    for gap in gaps:
        # Only add gaps that weren't already covered by opportunities.
        records.append(_make_evidence_record(
            dfa_type=gap.get("dfa_type", ""),
            n=gap.get("n"),
            gap_type=gap.get("gap_type", ""),
        ))

    return records


def _extract_from_core_invariants(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract evidence from core_invariants output."""
    records: List[Dict[str, Any]] = []
    core_list = data.get("core_invariants", [])

    for core in core_list:
        inv_type = core.get("invariant_type", "")
        classes = core.get("classes", [])
        evidence = core.get("evidence", {})
        per_class = evidence.get("per_class", {})

        for cls in sorted(classes):
            cls_evidence = per_class.get(cls, {})
            records.append(_make_evidence_record(
                system_class=cls,
                invariant_type=inv_type,
                improved=True,
                avg_score=float(cls_evidence.get("avg_score", 0.0)),
            ))

    return records


def _extract_from_hierarchical(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract evidence from hierarchical pipeline output."""
    records: List[Dict[str, Any]] = []
    global_best = data.get("global_best_modes", [])

    for entry in global_best:
        records.append(_make_evidence_record(
            dfa_type=entry.get("dfa_type", ""),
            n=entry.get("n"),
            system_class=entry.get("system_class", ""),
            hierarchical_mode=entry.get("best_mode", ""),
            improved=entry.get("improved_over_baseline", False),
            avg_score=entry.get("score", None),
        ))

    return records


# ---------------------------------------------------------------------------
# PART 3 — EVIDENCE AGGREGATION
# ---------------------------------------------------------------------------


def aggregate_class_invariant(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate class x invariant evidence.

    Returns:
        {(system_class, invariant_type): {
            "support_count": int,
            "improved_count": int,
            "improved_ratio": float,
            "mean_score": float,
        }}
    """
    accum: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in records:
        sys_class = r.get("system_class", "")
        inv_type = r.get("invariant_type")
        if not sys_class or not inv_type:
            continue
        if r.get("interaction_type") or r.get("gap_type"):
            continue

        key = (sys_class, inv_type)
        if key not in accum:
            accum[key] = {
                "support_count": 0,
                "improved_count": 0,
                "score_sum": 0.0,
            }

        bucket = accum[key]
        bucket["support_count"] += 1
        if r.get("improved", False):
            bucket["improved_count"] += 1
        score = r.get("avg_score")
        if score is not None:
            bucket["score_sum"] += float(score)

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(accum.keys()):
        bucket = accum[key]
        sc = bucket["support_count"]
        ic = bucket["improved_count"]
        result[key] = {
            "support_count": sc,
            "improved_count": ic,
            "improved_ratio": round(ic / sc, 6) if sc > 0 else 0.0,
            "mean_score": round(bucket["score_sum"] / sc, 6) if sc > 0 else 0.0,
        }

    return result


def aggregate_class_hierarchy(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate class x hierarchical mode evidence.

    Returns:
        {(system_class, hierarchical_mode): {
            "count": int,
            "wins_vs_baseline": int,
            "win_ratio": float,
        }}
    """
    accum: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in records:
        sys_class = r.get("system_class", "")
        h_mode = r.get("hierarchical_mode")
        if not sys_class or not h_mode:
            continue

        key = (sys_class, h_mode)
        if key not in accum:
            accum[key] = {"count": 0, "wins": 0}

        accum[key]["count"] += 1
        if r.get("improved", False):
            accum[key]["wins"] += 1

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(accum.keys()):
        bucket = accum[key]
        c = bucket["count"]
        w = bucket["wins"]
        result[key] = {
            "count": c,
            "wins_vs_baseline": w,
            "win_ratio": round(w / c, 6) if c > 0 else 0.0,
        }

    return result


def aggregate_interactions(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate interaction evidence.

    Returns:
        {(pair_str, interaction_type): {
            "pair": str,
            "interaction_type": str,
            "evidence_count": int,
        }}
    """
    accum: Dict[Tuple[str, str], int] = {}

    for r in records:
        ix_type = r.get("interaction_type")
        if not ix_type:
            continue
        inv = r.get("invariant_type", "")
        if not inv:
            continue

        key = (inv, ix_type)
        accum[key] = accum.get(key, 0) + 1

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(accum.keys()):
        result[key] = {
            "pair": key[0],
            "interaction_type": key[1],
            "evidence_count": accum[key],
        }

    return result


def aggregate_gap_invariant(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate gap x suggested invariant evidence.

    Returns:
        {(gap_type, suggested_invariant): {
            "gap_type": str,
            "suggested_invariant": str,
            "count": int,
        }}
    """
    accum: Dict[Tuple[str, str], int] = {}

    for r in records:
        gap = r.get("gap_type")
        inv = r.get("invariant_type")
        if not gap:
            continue
        # Include records with gap_type even if invariant_type is empty.
        inv_str = inv or ""
        key = (gap, inv_str)
        accum[key] = accum.get(key, 0) + 1

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(accum.keys()):
        result[key] = {
            "gap_type": key[0],
            "suggested_invariant": key[1],
            "count": accum[key],
        }

    return result


# ---------------------------------------------------------------------------
# PART 4 — LAW EXTRACTION RULES
# ---------------------------------------------------------------------------


def extract_class_invariant_laws(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract class-invariant laws from aggregated evidence.

    Promote a law if:
      - support_count >= 2
      - improved_ratio >= 0.5
      - mean_score > 0
    """
    laws: List[Dict[str, Any]] = []

    for (sys_class, inv_type), metrics in sorted(agg.items()):
        sc = metrics["support_count"]
        ir = metrics["improved_ratio"]
        ms = metrics["mean_score"]

        if sc >= 2 and ir >= 0.5 and ms > 0:
            laws.append({
                "law_type": "class_invariant_law",
                "condition": {
                    "system_class": sys_class,
                },
                "conclusion": {
                    "invariant_type": inv_type,
                    "effect": "tends to improve correction",
                },
                "evidence": {
                    "support_count": sc,
                    "improved_count": metrics["improved_count"],
                    "improved_ratio": ir,
                    "mean_score": ms,
                },
            })

    return laws


def extract_class_hierarchy_laws(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract class-hierarchy laws from aggregated evidence.

    Promote if:
      - count >= 2
      - win_ratio >= 0.5
    """
    laws: List[Dict[str, Any]] = []

    for (sys_class, h_mode), metrics in sorted(agg.items()):
        c = metrics["count"]
        wr = metrics["win_ratio"]

        if c >= 2 and wr >= 0.5:
            laws.append({
                "law_type": "class_hierarchy_law",
                "condition": {
                    "system_class": sys_class,
                    "hierarchical_mode": h_mode,
                },
                "conclusion": {
                    "effect": "tends to outperform baseline",
                },
                "evidence": {
                    "count": c,
                    "wins_vs_baseline": metrics["wins_vs_baseline"],
                    "win_ratio": wr,
                },
            })

    return laws


def extract_interaction_laws(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract interaction laws from aggregated evidence.

    Promote if:
      - evidence_count >= 2
    """
    laws: List[Dict[str, Any]] = []

    for (pair, ix_type), metrics in sorted(agg.items()):
        ec = metrics["evidence_count"]

        if ec >= 2:
            laws.append({
                "law_type": "interaction_law",
                "condition": {
                    "invariant_pair": pair,
                },
                "conclusion": {
                    "interaction_type": ix_type,
                    "effect": ix_type + " is likely",
                },
                "evidence": {
                    "evidence_count": ec,
                },
            })

    return laws


def extract_gap_invariant_laws(
    agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Extract gap-invariant laws from aggregated evidence.

    Promote if:
      - count >= 2
    """
    laws: List[Dict[str, Any]] = []

    for (gap, inv), metrics in sorted(agg.items()):
        c = metrics["count"]

        if c >= 2 and inv:
            laws.append({
                "law_type": "gap_invariant_law",
                "condition": {
                    "gap_type": gap,
                },
                "conclusion": {
                    "suggested_invariant": inv,
                    "effect": "is usually indicated",
                },
                "evidence": {
                    "count": c,
                },
            })

    return laws


def extract_core_invariant_laws(
    core_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Extract core invariant laws from core_invariants output.

    Each promoted core invariant becomes a reusable law record.
    """
    laws: List[Dict[str, Any]] = []
    core_list = core_report.get("core_invariants", [])

    for core in core_list:
        inv_type = core.get("invariant_type", "")
        classes = sorted(core.get("classes", []))
        evidence = core.get("evidence", {})

        if not inv_type:
            continue

        laws.append({
            "law_type": "core_invariant_law",
            "condition": {
                "invariant_type": inv_type,
                "min_classes": len(classes),
            },
            "conclusion": {
                "classes": classes,
                "effect": "reusable structural law across classes",
            },
            "evidence": {
                "num_classes": evidence.get("num_classes", len(classes)),
                "avg_class_score": evidence.get("avg_class_score", 0.0),
                "classes": classes,
            },
        })

    return laws


# ---------------------------------------------------------------------------
# PART 5 — LAW RANKING
# ---------------------------------------------------------------------------


def _law_evidence_strength(law: Dict[str, Any]) -> float:
    """Compute evidence strength for sorting (higher = stronger)."""
    ev = law.get("evidence", {})
    lt = law.get("law_type", "")

    if lt == "class_invariant_law":
        return float(ev.get("improved_ratio", 0.0))
    elif lt == "class_hierarchy_law":
        return float(ev.get("win_ratio", 0.0))
    elif lt == "interaction_law":
        return float(ev.get("evidence_count", 0))
    elif lt == "gap_invariant_law":
        return float(ev.get("count", 0))
    elif lt == "core_invariant_law":
        return float(ev.get("avg_class_score", 0.0))
    return 0.0


def _law_support_count(law: Dict[str, Any]) -> int:
    """Extract support count for sorting."""
    ev = law.get("evidence", {})
    for key in ("support_count", "count", "evidence_count", "num_classes"):
        if key in ev:
            return int(ev[key])
    return 0


def _law_sort_key(law: Dict[str, Any]) -> Tuple:
    """Deterministic sort key: evidence strength desc, support desc,
    law_type asc, condition/conclusion lexicographic asc."""
    strength = _law_evidence_strength(law)
    support = _law_support_count(law)
    lt = law.get("law_type", "")
    cond_str = _dict_to_sortable_str(law.get("condition", {}))
    concl_str = _dict_to_sortable_str(law.get("conclusion", {}))

    return (-strength, -support, lt, cond_str, concl_str)


def _dict_to_sortable_str(d: Dict[str, Any]) -> str:
    """Convert dict to deterministic sortable string."""
    parts = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, list):
            v = ",".join(str(x) for x in v)
        parts.append("{}={}".format(k, v))
    return ";".join(parts)


def _assign_confidence(law: Dict[str, Any]) -> str:
    """Assign deterministic confidence label.

    'strong' if support_count >= 3 and ratio/win_ratio >= 0.66
    'moderate' otherwise.
    """
    support = _law_support_count(law)
    strength = _law_evidence_strength(law)

    if support >= 3 and strength >= 0.66:
        return "strong"
    return "moderate"


def rank_laws(laws: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank laws deterministically and assign confidence labels.

    Returns new list sorted by evidence strength, support count,
    law_type, and lexicographic condition/conclusion.
    """
    ranked = []
    for law in laws:
        entry = {
            "law_type": law["law_type"],
            "condition": dict(law["condition"]),
            "conclusion": dict(law["conclusion"]),
            "evidence": dict(law["evidence"]),
            "confidence": _assign_confidence(law),
        }
        ranked.append(entry)

    ranked.sort(key=_law_sort_key)

    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1

    return ranked


# ---------------------------------------------------------------------------
# PART 6 — RULEBOOK GENERATION
# ---------------------------------------------------------------------------


def build_rulebook(
    laws: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build structured rulebook from ranked laws.

    Returns:
        {
          "laws": [...],
          "by_class": {system_class: [law indices]},
          "by_invariant": {invariant_type: [law indices]},
          "by_law_type": {law_type: [law indices]},
        }
    """
    by_class: Dict[str, List[int]] = {}
    by_invariant: Dict[str, List[int]] = {}
    by_law_type: Dict[str, List[int]] = {}

    for i, law in enumerate(laws):
        lt = law.get("law_type", "")
        by_law_type.setdefault(lt, []).append(i)

        cond = law.get("condition", {})
        concl = law.get("conclusion", {})

        # Index by system_class.
        sc = cond.get("system_class", "")
        if sc:
            by_class.setdefault(sc, []).append(i)

        # Index by invariant_type from condition or conclusion.
        inv = cond.get("invariant_type", "") or concl.get(
            "invariant_type", ""
        ) or concl.get("suggested_invariant", "")
        if inv:
            by_invariant.setdefault(inv, []).append(i)

        # Also index invariant pair as individual invariants.
        pair_str = cond.get("invariant_pair", "")
        if pair_str and "," in pair_str:
            for part in pair_str.split(","):
                part = part.strip()
                if part:
                    by_invariant.setdefault(part, []).append(i)

        # Index core invariant law classes.
        core_classes = concl.get("classes", [])
        for cls in core_classes:
            by_class.setdefault(cls, []).append(i)

    return {
        "laws": laws,
        "by_class": {k: v for k, v in sorted(by_class.items())},
        "by_invariant": {k: v for k, v in sorted(by_invariant.items())},
        "by_law_type": {k: v for k, v in sorted(by_law_type.items())},
    }


def render_law_text(law: Dict[str, Any]) -> str:
    """Render a single law as human-readable text.

    Template-based, deterministic, no free-form prose.
    """
    confidence = law.get("confidence", "moderate")
    lt = law.get("law_type", "")
    cond = law.get("condition", {})
    concl = law.get("conclusion", {})

    prefix = "[{}]".format(confidence)

    if lt == "class_invariant_law":
        return "{} IF system_class={} THEN invariant={} {}".format(
            prefix,
            cond.get("system_class", "?"),
            concl.get("invariant_type", "?"),
            concl.get("effect", ""),
        )

    if lt == "class_hierarchy_law":
        return "{} IF system_class={} THEN hierarchical_mode={} {}".format(
            prefix,
            cond.get("system_class", "?"),
            cond.get("hierarchical_mode", "?"),
            concl.get("effect", ""),
        )

    if lt == "interaction_law":
        return "{} IF invariant_pair=({}) THEN {} {}".format(
            prefix,
            cond.get("invariant_pair", "?"),
            concl.get("interaction_type", "?"),
            concl.get("effect", ""),
        )

    if lt == "gap_invariant_law":
        return "{} IF gap={} THEN {} {}".format(
            prefix,
            cond.get("gap_type", "?"),
            concl.get("suggested_invariant", "?"),
            concl.get("effect", ""),
        )

    if lt == "core_invariant_law":
        classes = ",".join(concl.get("classes", []))
        return "{} IF invariant={} spans >={} classes THEN {} [{}]".format(
            prefix,
            cond.get("invariant_type", "?"),
            cond.get("min_classes", "?"),
            concl.get("effect", ""),
            classes,
        )

    return "{} UNKNOWN LAW TYPE: {}".format(prefix, lt)


# ---------------------------------------------------------------------------
# PART 7 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_law_extraction(data: Any) -> Dict[str, Any]:
    """Run the full law extraction pipeline.

    Pipeline:
      1. Normalize inputs
      2. Aggregate evidence
      3. Extract laws by template
      4. Rank laws
      5. Build rulebook

    Returns:
        {
          "laws": [...],
          "rulebook": {...},
          "summary": {
            "law_count": int,
            "strong_count": int,
            "moderate_count": int,
            "law_types": {...},
            "classes_covered": [...],
            "invariants_covered": [...],
          }
        }
    """
    # 1. Normalize.
    records = normalize_law_inputs(data)

    # 2. Aggregate.
    ci_agg = aggregate_class_invariant(records)
    ch_agg = aggregate_class_hierarchy(records)
    ix_agg = aggregate_interactions(records)
    gi_agg = aggregate_gap_invariant(records)

    # 3. Extract laws.
    all_laws: List[Dict[str, Any]] = []
    all_laws.extend(extract_class_invariant_laws(ci_agg))
    all_laws.extend(extract_class_hierarchy_laws(ch_agg))
    all_laws.extend(extract_interaction_laws(ix_agg))
    all_laws.extend(extract_gap_invariant_laws(gi_agg))
    all_laws.extend(extract_core_invariant_laws(data))

    # 4. Rank.
    ranked = rank_laws(all_laws)

    # 5. Build rulebook.
    rulebook = build_rulebook(ranked)

    # 6. Summary.
    strong_count = sum(1 for l in ranked if l.get("confidence") == "strong")
    moderate_count = sum(
        1 for l in ranked if l.get("confidence") == "moderate"
    )

    type_counts: Dict[str, int] = {}
    classes_set: set = set()
    invariants_set: set = set()

    for law in ranked:
        lt = law.get("law_type", "")
        type_counts[lt] = type_counts.get(lt, 0) + 1

        cond = law.get("condition", {})
        concl = law.get("conclusion", {})

        sc = cond.get("system_class", "")
        if sc:
            classes_set.add(sc)

        inv = cond.get("invariant_type", "") or concl.get(
            "invariant_type", ""
        ) or concl.get("suggested_invariant", "")
        if inv:
            invariants_set.add(inv)

        for cls in concl.get("classes", []):
            classes_set.add(cls)

    summary = {
        "law_count": len(ranked),
        "strong_count": strong_count,
        "moderate_count": moderate_count,
        "law_types": {k: v for k, v in sorted(type_counts.items())},
        "classes_covered": sorted(classes_set),
        "invariants_covered": sorted(invariants_set),
    }

    return {
        "laws": ranked,
        "rulebook": rulebook,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# PART 8 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_rulebook(report: Dict[str, Any]) -> str:
    """Render the full rulebook as a deterministic, sorted text block.

    Returns the rendered string (also prints to stdout).
    """
    lines = ["=== Deterministic Rulebook ==="]

    laws = report.get("laws", [])
    for law in laws:
        lines.append(render_law_text(law))

    summary = report.get("summary", {})
    lines.append("")
    lines.append("--- Summary ---")
    lines.append("Total laws: {}".format(summary.get("law_count", 0)))
    lines.append("Strong: {}".format(summary.get("strong_count", 0)))
    lines.append("Moderate: {}".format(summary.get("moderate_count", 0)))

    type_counts = summary.get("law_types", {})
    if type_counts:
        lines.append("By type:")
        for lt in sorted(type_counts.keys()):
            lines.append("  {}: {}".format(lt, type_counts[lt]))

    classes = summary.get("classes_covered", [])
    if classes:
        lines.append("Classes covered: {}".format(", ".join(classes)))

    invariants = summary.get("invariants_covered", [])
    if invariants:
        lines.append("Invariants covered: {}".format(", ".join(invariants)))

    text = "\n".join(lines)
    print(text)
    return text
