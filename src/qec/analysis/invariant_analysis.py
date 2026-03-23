"""Invariant effectiveness mapping and conflict analysis (v95.2.0).

Upgrades from:
    invariant -> works / doesn't work
to:
    invariant -> where it works -> how well -> interactions -> ranking

Maps invariant success across DFA classes, detects conflicts and synergies,
and ranks invariant usefulness.

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No probabilistic scoring.
"""

from typing import Any, Dict, List, Optional, Tuple

from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    classify_all_systems,
    normalize_results,
)
from qec.analysis.structural_diagnostics import _group_by_system


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_application_data(
    data: Any,
) -> List[Dict[str, Any]]:
    """Normalize invariant application or synthesis output to flat records.

    Accepts:
      - invariant_application output (dict with 'applications' key)
      - invariant_synthesis output (dict with 'accepted_invariants' key)

    Returns list of flat records:
        {
          "dfa_type": str,
          "n": Optional[int],
          "system_class": str,
          "invariant_type": str,
          "improved": bool,
          "before_metrics": {...},
          "after_metrics": {...},
        }
    """
    if not isinstance(data, dict):
        return []

    records: List[Dict[str, Any]] = []

    # --- Path A: invariant_application output ---
    applications = data.get("applications", [])
    if applications:
        # Build system_class map from raw data if available.
        class_map = _build_class_map(data)

        for app in applications:
            dfa_type = app.get("dfa_type", "")
            n = app.get("n")
            sys_class = class_map.get((dfa_type, n), "unknown")
            invariants = app.get("invariants", [])
            improved = app.get("improved", False)
            before = app.get("before_metrics", {})
            after = app.get("after_metrics", {})

            for inv_type in invariants:
                records.append({
                    "dfa_type": dfa_type,
                    "n": n,
                    "system_class": sys_class,
                    "invariant_type": inv_type,
                    "improved": improved,
                    "before_metrics": dict(before),
                    "after_metrics": dict(after),
                })
        return records

    # --- Path B: invariant_synthesis output ---
    accepted = data.get("accepted_invariants", [])
    if accepted:
        class_map = _build_class_map(data)

        for evaluation in accepted:
            if not evaluation.get("accepted", False):
                continue
            candidate = evaluation.get("candidate", {})
            dfa_type = candidate.get("dfa_type", "")
            n = candidate.get("n")
            sys_class = class_map.get((dfa_type, n), "unknown")
            inv_type = candidate.get("type", "")
            improvement = evaluation.get("improvement", {})
            before = improvement.get("before", {})
            after = improvement.get("after", {})

            improved = evaluation.get("accepted", False)
            records.append({
                "dfa_type": dfa_type,
                "n": n,
                "system_class": sys_class,
                "invariant_type": inv_type,
                "improved": improved,
                "before_metrics": dict(before),
                "after_metrics": dict(after),
            })

    return records


def _build_class_map(
    data: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], str]:
    """Build (dfa_type, n) -> system_class map from data.

    Attempts to extract from structural_diagnostics -> self_diagnostics
    -> system_classes. Falls back to empty map.
    """
    result: Dict[Tuple[str, Optional[int]], str] = {}

    sd = data.get("structural_diagnostics", {})
    self_diag = sd.get("self_diagnostics", {})
    sc_list = self_diag.get("system_classes", [])

    for sc in sc_list:
        key = (sc.get("dfa_type", ""), sc.get("n"))
        result[key] = sc.get("system_class", "unknown")

    return result


# ---------------------------------------------------------------------------
# PART 2 — EFFECTIVENESS METRICS
# ---------------------------------------------------------------------------


def compute_improvement_score(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> int:
    """Compute integer improvement score from before/after metrics.

    Scoring rules (deterministic, integer only):
      +2 if stability_efficiency improved
      +1 if compression_efficiency improved
      +1 if stability_gain improved
       0 if no change
      -1 per metric that degraded

    Returns integer score.
    """
    score = 0

    b_stab = float(before.get("stability_efficiency", 0.0))
    a_stab = float(after.get("stability_efficiency", 0.0))
    b_comp = float(before.get("compression_efficiency", 0.0))
    a_comp = float(after.get("compression_efficiency", 0.0))
    b_gain = float(before.get("stability_gain", 0.0))
    a_gain = float(after.get("stability_gain", 0.0))

    if a_stab > b_stab:
        score += 2
    elif a_stab < b_stab:
        score -= 1

    if a_comp > b_comp:
        score += 1
    elif a_comp < b_comp:
        score -= 1

    if a_gain > b_gain:
        score += 1
    elif a_gain < b_gain:
        score -= 1

    return score


def aggregate_by_invariant(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate effectiveness metrics per invariant type.

    Returns:
        {
          invariant_type: {
            "count": int,
            "improved_count": int,
            "total_score": int,
            "avg_score": float,
          }
        }

    Deterministic: sorted by invariant_type.
    """
    accum: Dict[str, Dict[str, Any]] = {}

    for r in records:
        inv = r["invariant_type"]
        if inv not in accum:
            accum[inv] = {
                "count": 0,
                "improved_count": 0,
                "total_score": 0,
            }
        entry = accum[inv]
        entry["count"] += 1
        if r.get("improved", False):
            entry["improved_count"] += 1
        score = compute_improvement_score(
            r.get("before_metrics", {}),
            r.get("after_metrics", {}),
        )
        entry["total_score"] += score

    # Build result with avg_score, sorted deterministically.
    result: Dict[str, Dict[str, Any]] = {}
    for inv in sorted(accum.keys()):
        entry = accum[inv]
        count = entry["count"]
        avg = entry["total_score"] / count if count > 0 else 0.0
        # Round to 6 decimal places for determinism.
        avg = round(avg, 6)
        result[inv] = {
            "count": count,
            "improved_count": entry["improved_count"],
            "total_score": entry["total_score"],
            "avg_score": avg,
        }

    return result


# ---------------------------------------------------------------------------
# PART 3 — CLASS-LEVEL MAPPING
# ---------------------------------------------------------------------------


def aggregate_by_class(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate by (invariant_type, system_class).

    Returns:
        {
          (invariant_type, system_class): {
            "count": int,
            "improved_count": int,
            "improved_ratio": float,
            "total_score": int,
            "avg_score": float,
          }
        }

    Deterministic: sorted by (invariant_type, system_class).
    """
    accum: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in records:
        inv = r["invariant_type"]
        sc = r.get("system_class", "unknown")
        key = (inv, sc)
        if key not in accum:
            accum[key] = {
                "count": 0,
                "improved_count": 0,
                "total_score": 0,
            }
        entry = accum[key]
        entry["count"] += 1
        if r.get("improved", False):
            entry["improved_count"] += 1
        score = compute_improvement_score(
            r.get("before_metrics", {}),
            r.get("after_metrics", {}),
        )
        entry["total_score"] += score

    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key in sorted(accum.keys()):
        entry = accum[key]
        count = entry["count"]
        avg = entry["total_score"] / count if count > 0 else 0.0
        avg = round(avg, 6)
        ratio = entry["improved_count"] / count if count > 0 else 0.0
        ratio = round(ratio, 6)
        result[key] = {
            "count": count,
            "improved_count": entry["improved_count"],
            "improved_ratio": ratio,
            "total_score": entry["total_score"],
            "avg_score": avg,
        }

    return result


# ---------------------------------------------------------------------------
# PART 4 — CONFLICT & SYNERGY DETECTION
# ---------------------------------------------------------------------------


def _group_records_by_system(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]]:
    """Group records by (dfa_type, n)."""
    groups: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for r in records:
        key = (r["dfa_type"], r.get("n"))
        groups.setdefault(key, []).append(r)
    return groups


def detect_interactions(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect invariant conflicts and synergies.

    For each system (dfa_type, n):
      - collect all invariants applied and their outcomes
      - compare individual vs combined effectiveness

    Conflict: both A and B improve individually, but when both are applied
              to the same system, the combined outcome does NOT improve.
    Synergy: A and B individually score <= 0, but when both are present
             on the same system, the system improves.

    Returns list of interaction dicts, sorted deterministically.
    """
    system_groups = _group_records_by_system(records)

    # For each system, map invariant_type -> list of records.
    # Track which pairs co-occur and their outcomes.
    pair_evidence: Dict[
        Tuple[str, str], Dict[str, Any]
    ] = {}

    for sys_key in sorted(
        system_groups.keys(), key=lambda k: (k[0], str(k[1]))
    ):
        sys_records = system_groups[sys_key]

        # Map invariant -> records for this system.
        inv_records: Dict[str, List[Dict[str, Any]]] = {}
        for r in sys_records:
            inv = r["invariant_type"]
            inv_records.setdefault(inv, []).append(r)

        inv_types = sorted(inv_records.keys())
        if len(inv_types) < 2:
            continue

        # Check all pairs (ordered deterministically).
        for i in range(len(inv_types)):
            for j in range(i + 1, len(inv_types)):
                inv_a = inv_types[i]
                inv_b = inv_types[j]
                pair_key = (inv_a, inv_b)

                # Compute individual scores.
                a_scores = [
                    compute_improvement_score(
                        r.get("before_metrics", {}),
                        r.get("after_metrics", {}),
                    )
                    for r in inv_records[inv_a]
                ]
                b_scores = [
                    compute_improvement_score(
                        r.get("before_metrics", {}),
                        r.get("after_metrics", {}),
                    )
                    for r in inv_records[inv_b]
                ]

                a_improved = any(r.get("improved", False)
                                 for r in inv_records[inv_a])
                b_improved = any(r.get("improved", False)
                                 for r in inv_records[inv_b])
                a_max_score = max(a_scores) if a_scores else 0
                b_max_score = max(b_scores) if b_scores else 0

                # Combined: check if the system improved overall.
                all_sys_improved = any(
                    r.get("improved", False) for r in sys_records
                )
                all_sys_scores = [
                    compute_improvement_score(
                        r.get("before_metrics", {}),
                        r.get("after_metrics", {}),
                    )
                    for r in sys_records
                ]
                max_sys_score = max(all_sys_scores) if all_sys_scores else 0

                interaction_type = None

                # Conflict: both improve alone, combined doesn't.
                if a_improved and b_improved and not all_sys_improved:
                    interaction_type = "conflict"

                # Also conflict if both have positive scores individually
                # but combined max score is lower than either individual.
                if (interaction_type is None
                        and a_max_score > 0 and b_max_score > 0
                        and max_sys_score < min(a_max_score, b_max_score)):
                    interaction_type = "conflict"

                # Synergy: both weak individually, system improves.
                if (interaction_type is None
                        and a_max_score <= 0 and b_max_score <= 0
                        and all_sys_improved):
                    interaction_type = "synergy"

                # Also synergy if combined score exceeds sum of individuals.
                if (interaction_type is None
                        and a_max_score >= 0 and b_max_score >= 0
                        and max_sys_score > a_max_score + b_max_score):
                    interaction_type = "synergy"

                if interaction_type is not None:
                    if pair_key not in pair_evidence:
                        pair_evidence[pair_key] = {
                            "conflict_count": 0,
                            "synergy_count": 0,
                        }
                    if interaction_type == "conflict":
                        pair_evidence[pair_key]["conflict_count"] += 1
                    else:
                        pair_evidence[pair_key]["synergy_count"] += 1

    # Build output list.
    interactions: List[Dict[str, Any]] = []
    for pair_key in sorted(pair_evidence.keys()):
        ev = pair_evidence[pair_key]
        if ev["conflict_count"] > 0:
            interactions.append({
                "pair": pair_key,
                "type": "conflict",
                "evidence_count": ev["conflict_count"],
            })
        if ev["synergy_count"] > 0:
            interactions.append({
                "pair": pair_key,
                "type": "synergy",
                "evidence_count": ev["synergy_count"],
            })

    return interactions


# ---------------------------------------------------------------------------
# PART 5 — RANKING
# ---------------------------------------------------------------------------


def rank_invariants(
    agg: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank invariants globally.

    Sort by:
      1. avg_score descending
      2. improved_count descending
      3. invariant_type ascending (lexicographic tiebreak)

    Returns list of ranked dicts with rank, invariant_type, and metrics.
    """
    items = []
    for inv_type, metrics in agg.items():
        items.append({
            "invariant_type": inv_type,
            **metrics,
        })

    items.sort(key=lambda x: (
        -x["avg_score"],
        -x["improved_count"],
        x["invariant_type"],
    ))

    ranked: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        ranked.append({
            "rank": i + 1,
            **item,
        })

    return ranked


def rank_per_class(
    class_agg: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Rank invariants per system class.

    Returns:
        {
          system_class: [
            {"rank": 1, "invariant_type": ..., ...},
            ...
          ]
        }

    Deterministic: classes sorted lexicographically,
    within each class sorted by avg_score desc, improved_count desc, name asc.
    """
    # Group by system_class.
    by_class: Dict[str, List[Dict[str, Any]]] = {}
    for (inv_type, sys_class), metrics in class_agg.items():
        entry = {
            "invariant_type": inv_type,
            **metrics,
        }
        by_class.setdefault(sys_class, []).append(entry)

    result: Dict[str, List[Dict[str, Any]]] = {}
    for sc in sorted(by_class.keys()):
        items = by_class[sc]
        items.sort(key=lambda x: (
            -x["avg_score"],
            -x["improved_count"],
            x["invariant_type"],
        ))
        ranked = []
        for i, item in enumerate(items):
            ranked.append({"rank": i + 1, **item})
        result[sc] = ranked

    return result


# ---------------------------------------------------------------------------
# PART 6 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_invariant_analysis(
    data: Any,
) -> Dict[str, Any]:
    """Full invariant effectiveness analysis pipeline.

    Pipeline:
      1. normalize application/synthesis data
      2. compute improvement scores
      3. aggregate globally by invariant
      4. aggregate per (invariant, class)
      5. detect interactions (conflicts/synergies)
      6. rank invariants (global + per class)

    Args:
        data: invariant_application output or invariant_synthesis output.

    Returns:
        {
          "global_ranking": [...],
          "class_ranking": {...},
          "class_effectiveness": {...},
          "interactions": [...],
        }
    """
    records = normalize_application_data(data)

    global_agg = aggregate_by_invariant(records)
    class_agg = aggregate_by_class(records)
    interactions = detect_interactions(records)

    global_ranking = rank_invariants(global_agg)
    class_ranking = rank_per_class(class_agg)

    # Convert class_agg to serializable format.
    class_effectiveness: Dict[str, Any] = {}
    for (inv_type, sys_class), metrics in sorted(class_agg.items()):
        class_effectiveness.setdefault(sys_class, {})[inv_type] = metrics

    return {
        "global_ranking": global_ranking,
        "class_ranking": class_ranking,
        "class_effectiveness": class_effectiveness,
        "interactions": interactions,
    }


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_invariant_analysis(
    report: Dict[str, Any],
) -> str:
    """Format invariant analysis report as human-readable text.

    Deterministic, sorted, text-only.
    """
    lines: List[str] = []

    # --- Global Ranking ---
    lines.append("=== Global Ranking ===")
    global_ranking = report.get("global_ranking", [])
    if not global_ranking:
        lines.append("No invariants ranked.")
    else:
        for entry in global_ranking:
            rank = entry["rank"]
            inv = entry["invariant_type"]
            avg = entry["avg_score"]
            lines.append(f"{rank}. {inv} (avg_score={avg})")

    lines.append("")

    # --- By Class ---
    lines.append("=== By Class ===")
    class_ranking = report.get("class_ranking", {})
    if not class_ranking:
        lines.append("No class rankings available.")
    else:
        for sc in sorted(class_ranking.keys()):
            lines.append(f"{sc}:")
            for entry in class_ranking[sc]:
                rank = entry["rank"]
                inv = entry["invariant_type"]
                lines.append(f"  {rank}. {inv}")
            lines.append("")

    # --- Interactions ---
    lines.append("=== Interactions ===")
    interactions = report.get("interactions", [])
    if not interactions:
        lines.append("No interactions detected.")
    else:
        for ix in interactions:
            pair = ix["pair"]
            itype = ix["type"]
            count = ix["evidence_count"]
            symbol = "\u00d7"  # multiplication sign
            lines.append(
                f"({itype}) {pair[0]} {symbol} {pair[1]} (n={count})"
            )

    return "\n".join(lines)
