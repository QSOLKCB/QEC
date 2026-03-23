"""Deterministic conjecture engine for testable hypothesis generation (v97.1.0).

Extends v97.0 (law-guided control) with structured conjecture generation.
Takes system features (laws, dynamics, invariants) and produces testable
hypotheses with attached test procedures and confidence scores.

Uses outputs from:
  - v96.3 correction_dynamics (friction, oscillation, churn)
  - v96.2 law_extraction (rulebook with laws)
  - v96.0 core_invariants (cross-class invariants)
  - v97.0 law_control (law-guided decisions)

All algorithms are pure, deterministic, and use only stdlib.
No mutation of inputs. No randomness. No free-form speculation.
"""

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# PART 1 — FEATURE EXTRACTION
# ---------------------------------------------------------------------------

_MULTI_STAGE_MODES = frozenset({
    "d4>e8_like", "square>d4", "d4>square",
    "e8_like>d4", "square>e8_like", "e8_like>square",
})


def _extract_single_features(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract conjecture features from a single system record.

    Returns a flat feature dict. No mutation of input.
    """
    mode = str(record.get("mode", record.get("best_mode", "")))
    return {
        "system_class": str(
            record.get("system_class", record.get("dfa_type", ""))
        ),
        "best_mode": mode,
        "friction_score": float(record.get("friction_score", 0.0)),
        "oscillation_ratio": float(record.get("oscillation_ratio", 0.0)),
        "churn_score": float(record.get("churn_score", 0.0)),
        "core_invariants": list(record.get("core_invariants", [])),
        "law_matches": list(record.get("law_matches", [])),
        "is_multi_stage": mode in _MULTI_STAGE_MODES,
        "stability_efficiency": float(
            record.get("stability_efficiency", 0.0)
        ),
    }


def extract_features(data: Any) -> List[Dict[str, Any]]:
    """Extract conjecture features from upstream pipeline data.

    Accepts:
      - list of system records
      - dict with 'systems', 'candidates', or 'groups' key

    Returns sorted list of feature dicts. No mutation of inputs.
    """
    records: List[Dict[str, Any]] = []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("systems", "candidates", "groups"):
            val = data.get(key, [])
            if isinstance(val, list) and val:
                # groups may contain nested 'best' records
                if key == "groups":
                    for g in val:
                        if isinstance(g, dict):
                            best = g.get("best")
                            if isinstance(best, dict):
                                records.append(best)
                else:
                    records = val
                break

    features = []
    for rec in records:
        if isinstance(rec, dict):
            features.append(_extract_single_features(rec))

    features.sort(key=lambda f: (f["system_class"], f["best_mode"]))
    return features


# ---------------------------------------------------------------------------
# PART 2 — CONJECTURE RULES
# ---------------------------------------------------------------------------

_HIGH_OSCILLATION = 0.5
_HIGH_CHURN = 0.6
_HIGH_FRICTION = 2.5
_MIN_INVARIANT_CLASSES = 2


def _rule_a_oscillation(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule A: high oscillation suggests simpler modes reduce oscillation."""
    if feat["oscillation_ratio"] < _HIGH_OSCILLATION:
        return []
    return [{
        "statement": (
            "IF oscillation_ratio is high ({:.2f}) for class '{}' "
            "THEN simpler modes reduce oscillation"
        ).format(feat["oscillation_ratio"], feat["system_class"]),
        "type": "oscillation_reduction",
        "conditions": {
            "oscillation_ratio": feat["oscillation_ratio"],
            "system_class": feat["system_class"],
            "threshold": _HIGH_OSCILLATION,
        },
    }]


def _rule_b_churn(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule B: high churn suggests curvature constraint reduces churn."""
    if feat["churn_score"] < _HIGH_CHURN:
        return []
    return [{
        "statement": (
            "IF churn_score is high ({:.2f}) for class '{}' "
            "THEN curvature-like constraint (1,-2,1) reduces churn"
        ).format(feat["churn_score"], feat["system_class"]),
        "type": "churn_reduction",
        "conditions": {
            "churn_score": feat["churn_score"],
            "system_class": feat["system_class"],
            "threshold": _HIGH_CHURN,
        },
    }]


def _rule_c_friction(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule C: high friction suggests reducing invariant conflict."""
    if feat["friction_score"] < _HIGH_FRICTION:
        return []
    return [{
        "statement": (
            "IF friction_score is high ({:.2f}) for class '{}' "
            "THEN reducing invariant conflict improves efficiency"
        ).format(feat["friction_score"], feat["system_class"]),
        "type": "friction_reduction",
        "conditions": {
            "friction_score": feat["friction_score"],
            "system_class": feat["system_class"],
            "threshold": _HIGH_FRICTION,
        },
    }]


def _rule_d_invariants(
    all_features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rule D: invariants appearing across classes generalize."""
    # Collect invariants by name across classes.
    inv_classes: Dict[str, set] = {}
    for feat in all_features:
        sys_class = feat["system_class"]
        for inv in feat["core_invariants"]:
            inv_name = str(inv.get("name", inv) if isinstance(inv, dict)
                          else inv)
            inv_classes.setdefault(inv_name, set()).add(sys_class)

    conjectures = []
    for inv_name in sorted(inv_classes.keys()):
        classes = inv_classes[inv_name]
        if len(classes) >= _MIN_INVARIANT_CLASSES:
            conjectures.append({
                "statement": (
                    "IF invariant '{}' appears in {} classes ({}) "
                    "THEN it generalizes as a reusable correction rule"
                ).format(
                    inv_name,
                    len(classes),
                    ", ".join(sorted(classes)),
                ),
                "type": "invariant_generalization",
                "conditions": {
                    "invariant_name": inv_name,
                    "class_count": len(classes),
                    "classes": sorted(classes),
                    "threshold": _MIN_INVARIANT_CLASSES,
                },
            })
    return conjectures


def _rule_e_hierarchy(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rule E: multi-stage with high friction suggests partial hierarchy."""
    if not feat["is_multi_stage"]:
        return []
    if feat["friction_score"] < _HIGH_FRICTION:
        return []
    return [{
        "statement": (
            "IF multi-stage mode '{}' increases friction ({:.2f}) "
            "for class '{}' "
            "THEN optimal solution lies in partial hierarchy"
        ).format(
            feat["best_mode"],
            feat["friction_score"],
            feat["system_class"],
        ),
        "type": "hierarchy_optimization",
        "conditions": {
            "mode": feat["best_mode"],
            "friction_score": feat["friction_score"],
            "system_class": feat["system_class"],
            "is_multi_stage": True,
            "threshold": _HIGH_FRICTION,
        },
    }]


def generate_conjectures(
    features: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate conjectures from extracted features using fixed rules.

    Applies rules A-E deterministically. Returns sorted list.
    No mutation of inputs.
    """
    conjectures: List[Dict[str, Any]] = []

    for feat in features:
        conjectures.extend(_rule_a_oscillation(feat))
        conjectures.extend(_rule_b_churn(feat))
        conjectures.extend(_rule_c_friction(feat))
        conjectures.extend(_rule_e_hierarchy(feat))

    # Rule D operates across all features.
    conjectures.extend(_rule_d_invariants(features))

    # Deterministic sort by (type, statement).
    conjectures.sort(key=lambda c: (c["type"], c["statement"]))
    return conjectures


# ---------------------------------------------------------------------------
# PART 3 — TEST ATTACHMENT
# ---------------------------------------------------------------------------

_TEST_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "oscillation_reduction": {
        "method": "run DFA benchmark with simple vs complex modes",
        "compare": ["square", "d4", "multi-stage"],
        "metric": "oscillation_ratio",
        "expected": "decrease with simpler modes",
    },
    "churn_reduction": {
        "method": "run DFA benchmark with/without curvature constraint",
        "compare": ["baseline", "with_curvature_constraint"],
        "metric": "churn_score",
        "expected": "decrease with curvature constraint",
    },
    "friction_reduction": {
        "method": "run DFA benchmark with reduced invariant set",
        "compare": ["full_invariants", "reduced_invariants"],
        "metric": "friction_score",
        "expected": "decrease with fewer invariant conflicts",
    },
    "invariant_generalization": {
        "method": "apply invariant to unseen system classes",
        "compare": ["with_invariant", "without_invariant"],
        "metric": "stability_efficiency",
        "expected": "improvement in new classes",
    },
    "hierarchy_optimization": {
        "method": "run DFA benchmark comparing full vs partial hierarchy",
        "compare": ["full_multi_stage", "partial_hierarchy", "single_stage"],
        "metric": "friction_score",
        "expected": "partial hierarchy reduces friction vs full multi-stage",
    },
}


def attach_tests(conjecture: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a test procedure to a conjecture.

    Returns a new dict with the conjecture fields plus a 'test' key.
    No mutation of input.
    """
    ctype = conjecture.get("type", "")
    template = _TEST_TEMPLATES.get(ctype, {
        "method": "manual verification required",
        "compare": [],
        "metric": "unknown",
        "expected": "unknown",
    })

    result = dict(conjecture)
    result["test"] = dict(template)
    return result


# ---------------------------------------------------------------------------
# PART 4 — CONJECTURE SCORING
# ---------------------------------------------------------------------------


def score_conjecture(
    conjecture: Dict[str, Any],
    features: List[Dict[str, Any]],
) -> int:
    """Score a conjecture based on alignment with features.

    Rules:
      +2 -> aligns with existing laws
      +1 -> aligns with observed dynamics
       0 -> neutral
      -1 -> contradicts known law

    Returns integer score. No mutation of inputs.
    """
    score = 0
    ctype = conjecture.get("type", "")
    conditions = conjecture.get("conditions", {})

    # Check alignment with laws across features.
    law_names = set()
    for feat in features:
        for law in feat.get("law_matches", []):
            law_name = str(law.get("type", law) if isinstance(law, dict)
                          else law)
            law_names.add(law_name)

    # +2 if conjecture type aligns with a known law type.
    _LAW_ALIGNMENT = {
        "oscillation_reduction": {"oscillation_law", "oscillation_suppression"},
        "churn_reduction": {"churn_law", "churn_penalty"},
        "friction_reduction": {"friction_law", "conflict_avoidance"},
        "invariant_generalization": {
            "core_invariant_law", "class_invariant_law",
        },
        "hierarchy_optimization": {"hierarchy_law", "class_hierarchy_law"},
    }

    aligned_laws = _LAW_ALIGNMENT.get(ctype, set())
    if aligned_laws & law_names:
        score += 2

    # +1 if dynamics support the conjecture conditions.
    target_class = conditions.get("system_class", "")
    for feat in features:
        if target_class and feat["system_class"] != target_class:
            continue

        if ctype == "oscillation_reduction":
            if feat["oscillation_ratio"] >= _HIGH_OSCILLATION:
                score += 1
                break
        elif ctype == "churn_reduction":
            if feat["churn_score"] >= _HIGH_CHURN:
                score += 1
                break
        elif ctype == "friction_reduction":
            if feat["friction_score"] >= _HIGH_FRICTION:
                score += 1
                break
        elif ctype == "invariant_generalization":
            if len(feat["core_invariants"]) > 0:
                score += 1
                break
        elif ctype == "hierarchy_optimization":
            if feat["is_multi_stage"] and feat["friction_score"] >= _HIGH_FRICTION:
                score += 1
                break

    return score


# ---------------------------------------------------------------------------
# PART 5 — CONFIDENCE ASSIGNMENT
# ---------------------------------------------------------------------------


def _assign_confidence(score: int) -> str:
    """Assign confidence label from score. Deterministic."""
    if score >= 3:
        return "high"
    elif score >= 1:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# PART 6 — PIPELINE
# ---------------------------------------------------------------------------


def run_conjecture_engine(data: Any) -> Dict[str, Any]:
    """Run the full conjecture engine pipeline.

    Steps:
      1. Extract features
      2. Generate conjectures
      3. Attach tests
      4. Score and assign confidence

    Returns:
        {
          "conjectures": [
            {
              "statement": str,
              "type": str,
              "conditions": {...},
              "test": {...},
              "score": int,
              "confidence": "high"|"medium"|"low",
            },
            ...
          ],
          "summary": {
            "total": int,
            "high": int,
            "medium": int,
            "low": int,
          },
        }

    No mutation of inputs. Deterministic output.
    """
    features = extract_features(data)
    raw_conjectures = generate_conjectures(features)

    conjectures: List[Dict[str, Any]] = []
    counts = {"high": 0, "medium": 0, "low": 0}

    for conj in raw_conjectures:
        with_test = attach_tests(conj)
        score = score_conjecture(conj, features)
        confidence = _assign_confidence(score)

        entry = dict(with_test)
        entry["score"] = score
        entry["confidence"] = confidence
        conjectures.append(entry)
        counts[confidence] += 1

    # Sort by score desc, then type asc, then statement asc.
    conjectures.sort(key=lambda c: (-c["score"], c["type"], c["statement"]))

    summary = {
        "total": len(conjectures),
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
    }

    return {"conjectures": conjectures, "summary": summary}


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_conjectures(report: Dict[str, Any]) -> str:
    """Format conjecture report as human-readable text.

    Returns deterministic string output.
    """
    lines: List[str] = []
    lines.append("=== Conjectures ===")
    lines.append("")

    conjectures = report.get("conjectures", [])
    summary = report.get("summary", {})

    if not conjectures:
        lines.append("No conjectures generated.")
        return "\n".join(lines)

    for conj in conjectures:
        confidence = conj.get("confidence", "low")
        statement = conj.get("statement", "")
        score = conj.get("score", 0)
        test = conj.get("test", {})

        lines.append("[{}] (score={})".format(confidence, score))
        lines.append("  {}".format(statement))
        lines.append("  test:")
        lines.append("    method: {}".format(test.get("method", "?")))
        compare = test.get("compare", [])
        if compare:
            lines.append("    compare: {}".format(", ".join(str(c) for c in compare)))
        lines.append("    metric: {}".format(test.get("metric", "?")))
        lines.append("    expect: {}".format(test.get("expected", "?")))
        lines.append("")

    lines.append("--- Summary ---")
    lines.append("total: {}".format(summary.get("total", 0)))
    lines.append("high: {}".format(summary.get("high", 0)))
    lines.append("medium: {}".format(summary.get("medium", 0)))
    lines.append("low: {}".format(summary.get("low", 0)))

    return "\n".join(lines)
