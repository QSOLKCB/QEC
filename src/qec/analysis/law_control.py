"""Law-guided scoring for friction-aware control (v97.0.0).

Extends v96.4 (friction-aware control) with a small deterministic layer
that combines efficiency, friction, and law_score into a final decision.

Uses outputs from:
  - v96.4 friction_control (candidates with efficiency)
  - v96.2 law_extraction (rulebook with laws)

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No weight tuning. No learning.
"""

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PART 1 — SIMPLE LAW MATCHING
# ---------------------------------------------------------------------------


def _system_class_from_record(record: Dict[str, Any]) -> str:
    """Extract system_class from a candidate record.

    Looks for explicit system_class, then falls back to dfa_type.
    """
    sc = record.get("system_class", "")
    if sc:
        return str(sc)
    return str(record.get("dfa_type", ""))


def match_laws(
    record: Dict[str, Any],
    laws: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Match a candidate record against a list of rulebook laws.

    A law matches if:
      - law condition 'system_class' matches the record's system_class
      - law condition 'hierarchical_mode' matches the record's mode
        (only checked if specified in the law condition)

    A law contradicts if:
      - system_class matches but effect contains 'not' or 'avoid'

    Returns:
        {"matched": [...], "contradicting": [...]}

    No mutation of inputs.
    """
    matched: List[Dict[str, Any]] = []
    contradicting: List[Dict[str, Any]] = []

    sys_class = _system_class_from_record(record)
    mode = str(record.get("mode", ""))

    for law in laws:
        cond = law.get("condition", {})
        concl = law.get("conclusion", {})

        # Check system_class condition.
        law_class = cond.get("system_class", "")
        if not law_class:
            continue
        if law_class != sys_class:
            continue

        # Check hierarchical_mode condition if specified.
        law_mode = cond.get("hierarchical_mode", "")
        if law_mode and law_mode != mode:
            continue

        # Determine if this is a contradiction.
        effect = str(concl.get("effect", "")).lower()
        if "not" in effect or "avoid" in effect:
            contradicting.append(law)
        else:
            matched.append(law)

    return {"matched": matched, "contradicting": contradicting}


# ---------------------------------------------------------------------------
# PART 2 — LAW SCORE
# ---------------------------------------------------------------------------


def compute_law_score(
    matched: List[Dict[str, Any]],
    contradicting: List[Dict[str, Any]],
) -> int:
    """Compute an integer law score from matched and contradicting laws.

    Rules:
      +2 per strong match (confidence == 'strong')
      +1 per moderate match (confidence != 'strong')
      -1 per contradiction

    Returns integer score.
    """
    score = 0

    for law in matched:
        confidence = law.get("confidence", "moderate")
        if confidence == "strong":
            score += 2
        else:
            score += 1

    for _law in contradicting:
        score -= 1

    return score


# ---------------------------------------------------------------------------
# PART 3 — LAW PRIOR
# ---------------------------------------------------------------------------


_LAW_WEIGHT = 0.1


def apply_law_prior(
    record: Dict[str, Any],
    law_score: int,
) -> Dict[str, Any]:
    """Apply law-guided prior to a candidate record.

    Computes:
        effective_score = record['efficiency'] + (law_score * 0.1)

    Returns a new dict with the original fields plus:
      - 'law_score': the integer law score
      - 'effective_score': the combined score

    Does NOT mutate the original record.
    """
    efficiency = float(record.get("efficiency", 0.0))
    effective_score = round(efficiency + (law_score * _LAW_WEIGHT), 6)

    result = dict(record)
    result["law_score"] = law_score
    result["effective_score"] = effective_score
    return result


# ---------------------------------------------------------------------------
# PART 4 — DECISION FUNCTION
# ---------------------------------------------------------------------------


def _decision_sort_key(record: Dict[str, Any]) -> Tuple:
    """Deterministic sort key for law-guided selection.

    Priority (descending):
      1. effective_score (desc)
      2. stability_efficiency (desc)
      3. compression_efficiency (desc)
      4. friction_score (asc)
      5. mode lexicographic (asc)
    """
    return (
        -record.get("effective_score", 0.0),
        -record.get("stability_efficiency", 0.0),
        -record.get("compression_efficiency", 0.0),
        record.get("friction_score", 0.0),
        record.get("mode", ""),
    )


def select_with_laws(
    records: List[Dict[str, Any]],
    laws: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the best candidate using law-guided scoring.

    For each record:
      1. match laws
      2. compute law_score
      3. apply law prior

    Then rank by:
      effective_score -> stability_efficiency -> compression_efficiency
      -> lower friction_score -> mode name

    Returns:
        {
          "best": <best record with law fields>,
          "trace": [<all scored records in ranked order>],
        }

    No mutation of input records.
    """
    if not records:
        return {"best": None, "trace": []}

    scored: List[Dict[str, Any]] = []

    for record in records:
        match_result = match_laws(record, laws)
        law_score = compute_law_score(
            match_result["matched"],
            match_result["contradicting"],
        )
        enriched = apply_law_prior(record, law_score)
        enriched["matched_laws"] = len(match_result["matched"])
        enriched["contradicting_laws"] = len(match_result["contradicting"])
        scored.append(enriched)

    scored.sort(key=_decision_sort_key)

    return {"best": scored[0], "trace": scored}


# ---------------------------------------------------------------------------
# PART 5 — PIPELINE
# ---------------------------------------------------------------------------


def _extract_laws_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the flat laws list from pipeline data.

    Accepts:
      - data with 'law_extraction' key containing run_law_extraction output
      - data with 'laws' key directly
      - data with 'rulebook' containing 'laws'
    """
    law_ext = data.get("law_extraction", {})
    if isinstance(law_ext, dict):
        laws = law_ext.get("laws", [])
        if laws:
            return list(laws)
        rb = law_ext.get("rulebook", {})
        if isinstance(rb, dict):
            laws = rb.get("laws", [])
            if laws:
                return list(laws)

    laws = data.get("laws", [])
    if laws:
        return list(laws)

    rb = data.get("rulebook", {})
    if isinstance(rb, dict):
        laws = rb.get("laws", [])
        if laws:
            return list(laws)

    return []


def _extract_candidates(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract friction-control candidates from pipeline data.

    Accepts:
      - data with 'friction_control' containing run_friction_control output
      - data with 'candidates' directly
    """
    fc = data.get("friction_control", {})
    if isinstance(fc, dict):
        cands = fc.get("candidates", [])
        if cands:
            return list(cands)

    cands = data.get("candidates", [])
    if cands:
        return list(cands)

    return []


def _group_key(record: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Deterministic group key: (dfa_type, n)."""
    return (str(record.get("dfa_type", "")), record.get("n"))


def run_law_control(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Run law-guided scoring on friction-control candidates.

    Uses:
      - candidates from v96.4 friction_control output
      - laws from v96.2 law_extraction output

    Groups candidates by (dfa_type, n) and selects best per group.

    Returns:
        {
          "groups": [
            {
              "dfa_type": str,
              "n": Optional[int],
              "best": <best record>,
              "trace": [<all scored records>],
            },
            ...
          ],
          "summary": {
            "total_groups": int,
            "total_candidates": int,
            "law_count": int,
          },
        }
    """
    if not isinstance(data, dict):
        return {"groups": [], "summary": _empty_summary()}

    candidates = _extract_candidates(data)
    laws = _extract_laws_list(data)

    if not candidates:
        return {"groups": [], "summary": _empty_summary()}

    # Ensure all candidates have efficiency computed.
    enriched_candidates: List[Dict[str, Any]] = []
    for c in candidates:
        entry = dict(c)
        if "efficiency" not in entry:
            stab = float(entry.get("stability_efficiency", 0.0))
            friction = float(entry.get("friction_score", 0.0))
            if stab > 0.0:
                entry["efficiency"] = round(stab / (1.0 + friction), 6)
            else:
                entry["efficiency"] = 0.0
        enriched_candidates.append(entry)

    # Group by (dfa_type, n).
    group_map: Dict[Tuple[str, Optional[int]], List[Dict[str, Any]]] = {}
    for c in enriched_candidates:
        key = _group_key(c)
        group_map.setdefault(key, []).append(c)

    # Select best per group.
    groups: List[Dict[str, Any]] = []
    for key in sorted(group_map.keys(), key=lambda k: (k[0], k[1] or 0)):
        members = group_map[key]
        result = select_with_laws(members, laws)
        groups.append({
            "dfa_type": key[0],
            "n": key[1],
            "best": result["best"],
            "trace": result["trace"],
        })

    summary = {
        "total_groups": len(groups),
        "total_candidates": len(enriched_candidates),
        "law_count": len(laws),
    }

    return {"groups": groups, "summary": summary}


def _empty_summary() -> Dict[str, int]:
    """Return empty summary dict."""
    return {
        "total_groups": 0,
        "total_candidates": 0,
        "law_count": 0,
    }


# ---------------------------------------------------------------------------
# PART 6 — PRINT LAYER
# ---------------------------------------------------------------------------


def _law_reason(best: Dict[str, Any]) -> str:
    """Generate deterministic reason string from law score."""
    law_score = best.get("law_score", 0)
    matched = best.get("matched_laws", 0)
    contradicting = best.get("contradicting_laws", 0)

    if law_score >= 2:
        return "strong law support ({} matched)".format(matched)
    elif law_score == 1:
        return "moderate law support ({} matched)".format(matched)
    elif law_score == 0:
        if contradicting > 0:
            return "neutral (matches and contradictions cancel)"
        return "no matching laws"
    else:
        return "law contradictions ({} contradicting)".format(contradicting)


def print_law_control(report: Dict[str, Any]) -> str:
    """Format law-control report as human-readable text.

    Returns deterministic string output.
    """
    lines: List[str] = []
    lines.append("=== Law-Guided Control Report ===")
    lines.append("")

    groups = report.get("groups", [])
    summary = report.get("summary", {})

    if not groups:
        lines.append("No candidates to evaluate.")
        return "\n".join(lines)

    for group in groups:
        dfa = group.get("dfa_type", "unknown")
        n = group.get("n")
        best = group.get("best")

        if n is not None:
            lines.append("DFA: {} (n={})".format(dfa, n))
        else:
            lines.append("DFA: {}".format(dfa))

        if best is None:
            lines.append("  best_mode: none")
            lines.append("")
            continue

        lines.append("  best_mode: {}".format(best.get("mode", "?")))
        lines.append("  efficiency: {:.2f}".format(
            best.get("efficiency", 0.0),
        ))
        law_score = best.get("law_score", 0)
        sign = "+" if law_score >= 0 else ""
        lines.append("  law_score: {}{}".format(sign, law_score))
        lines.append("  effective_score: {:.4f}".format(
            best.get("effective_score", 0.0),
        ))
        lines.append("  reason: {}".format(_law_reason(best)))
        lines.append("")

    lines.append("--- Summary ---")
    lines.append("groups: {}".format(summary.get("total_groups", 0)))
    lines.append("candidates: {}".format(summary.get("total_candidates", 0)))
    lines.append("laws: {}".format(summary.get("law_count", 0)))

    return "\n".join(lines)
