"""Deterministic conjecture execution and validation engine (v97.2.0).

Extends v97.1 (conjecture generation) with:
  conjectures -> execute tests -> evaluate outcomes -> classify

Takes conjectures from conjecture_engine and system data, executes the
attached test procedures against actual metrics, validates outcomes,
and classifies each conjecture as confirmed/rejected/inconclusive.

Uses outputs from:
  - v97.1 conjecture_engine (conjectures with tests)
  - v96.3 correction_dynamics (friction, oscillation, churn)
  - v92+ DFA benchmark metrics

All algorithms are pure, deterministic, and use only stdlib.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------

_METRIC_KEYS = frozenset({
    "oscillation_ratio", "churn_score", "friction_score",
    "stability_efficiency",
})


def _extract_system_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a normalized system record. No mutation of input."""
    mode = str(record.get("mode", record.get("best_mode", "")))
    return {
        "system_class": str(
            record.get("system_class", record.get("dfa_type", ""))
        ),
        "best_mode": mode,
        "friction_score": float(record.get("friction_score", 0.0)),
        "oscillation_ratio": float(record.get("oscillation_ratio", 0.0)),
        "churn_score": float(record.get("churn_score", 0.0)),
        "stability_efficiency": float(
            record.get("stability_efficiency", 0.0)
        ),
        "core_invariants": list(record.get("core_invariants", [])),
        "law_matches": list(record.get("law_matches", [])),
    }


def _extract_systems(data: Any) -> List[Dict[str, Any]]:
    """Extract system records from upstream data."""
    records: List[Dict[str, Any]] = []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("systems", "candidates", "groups"):
            val = data.get(key, [])
            if isinstance(val, list) and val:
                if key == "groups":
                    for g in val:
                        if isinstance(g, dict):
                            best = g.get("best")
                            if isinstance(best, dict):
                                records.append(best)
                else:
                    records = val
                break

    systems = []
    for rec in records:
        if isinstance(rec, dict):
            systems.append(_extract_system_record(rec))

    systems.sort(key=lambda s: (s["system_class"], s["best_mode"]))
    return systems


def normalize_conjecture_inputs(
    data: Any,
) -> Dict[str, Any]:
    """Combine conjectures and system data into execution input.

    Accepts data dict with:
      - 'conjectures' key (from conjecture_engine output)
      - system records (from 'systems', 'candidates', or 'groups')

    Returns:
        {
          "conjecture": {...},  # first conjecture or from list
          "systems": [...]
        }

    For multiple conjectures, returns a list-aware structure.
    No mutation of inputs.
    """
    conjectures_raw: List[Dict[str, Any]] = []
    system_data = data

    if isinstance(data, dict):
        conj_val = data.get("conjectures", [])
        if isinstance(conj_val, list):
            conjectures_raw = [
                dict(c) for c in conj_val if isinstance(c, dict)
            ]
        # System data may be in same dict or nested
        system_data = data

    systems = _extract_systems(system_data)

    # Sort conjectures deterministically
    conjectures_raw.sort(
        key=lambda c: (c.get("type", ""), c.get("statement", ""))
    )

    return {
        "conjectures": conjectures_raw,
        "systems": systems,
    }


# ---------------------------------------------------------------------------
# PART 2 — CONJECTURE EXECUTION
# ---------------------------------------------------------------------------

# Metric comparison config per conjecture type
_METRIC_MAP: Dict[str, Dict[str, Any]] = {
    "oscillation_reduction": {
        "metric": "oscillation_ratio",
        "expected_direction": "decrease",
        "compare_modes": True,
    },
    "churn_reduction": {
        "metric": "churn_score",
        "expected_direction": "decrease",
        "compare_modes": False,
    },
    "friction_reduction": {
        "metric": "friction_score",
        "expected_direction": "decrease",
        "compare_modes": False,
    },
    "invariant_generalization": {
        "metric": "stability_efficiency",
        "expected_direction": "increase",
        "compare_modes": False,
    },
    "hierarchy_optimization": {
        "metric": "friction_score",
        "expected_direction": "decrease",
        "compare_modes": True,
    },
}

_MULTI_STAGE_MODES = frozenset({
    "d4>e8_like", "square>d4", "d4>square",
    "e8_like>d4", "square>e8_like", "e8_like>square",
})

_SIMPLE_MODES = frozenset({"square", "d4", "e8_like"})


def _compute_metric_change(
    systems: List[Dict[str, Any]],
    conjecture: Dict[str, Any],
    metric_config: Dict[str, Any],
) -> Tuple[float, str]:
    """Compute observed metric change for a conjecture.

    Returns (observed_change, direction).
    No mutation of inputs.
    """
    metric_name = metric_config["metric"]
    conditions = conjecture.get("conditions", {})
    target_class = conditions.get("system_class", "")

    # Filter systems to target class if specified
    relevant = [
        s for s in systems
        if not target_class or s["system_class"] == target_class
    ]

    if not relevant:
        return 0.0, "none"

    if metric_config.get("compare_modes") and len(relevant) >= 2:
        # Compare simple vs complex modes
        simple_values = []
        complex_values = []
        for s in relevant:
            val = s.get(metric_name, 0.0)
            if s["best_mode"] in _MULTI_STAGE_MODES:
                complex_values.append(val)
            elif s["best_mode"] in _SIMPLE_MODES:
                simple_values.append(val)

        if simple_values and complex_values:
            avg_simple = sum(simple_values) / len(simple_values)
            avg_complex = sum(complex_values) / len(complex_values)
            change = avg_simple - avg_complex
        else:
            # Fall back to spread across all relevant
            values = [s.get(metric_name, 0.0) for s in relevant]
            change = min(values) - max(values) if len(values) >= 2 else 0.0
    else:
        # Compare spread: best vs worst
        values = [s.get(metric_name, 0.0) for s in relevant]
        if len(values) >= 2:
            change = min(values) - max(values)
        elif len(values) == 1:
            # Single system: compare against threshold
            threshold = conditions.get("threshold", 0.0)
            change = values[0] - threshold
        else:
            change = 0.0

    # Round to avoid floating-point drift
    change = round(change, 6)

    if change < -1e-9:
        direction = "decrease"
    elif change > 1e-9:
        direction = "increase"
    else:
        direction = "none"

    return change, direction


def execute_conjecture(
    conjecture: Dict[str, Any],
    systems: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute a single conjecture test against system data.

    Maps conjecture type to metric computation and compares
    observed change against expected direction.

    Returns:
        {
          "observed_change": float,
          "direction": "increase"|"decrease"|"none",
        }

    No mutation of inputs.
    """
    ctype = conjecture.get("type", "")
    metric_config = _METRIC_MAP.get(ctype, {
        "metric": "stability_efficiency",
        "expected_direction": "increase",
        "compare_modes": False,
    })

    change, direction = _compute_metric_change(
        systems, conjecture, metric_config
    )

    return {
        "observed_change": change,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# PART 3 — VALIDATION
# ---------------------------------------------------------------------------


def validate_conjecture(
    conjecture: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate conjecture by comparing expected vs observed direction.

    Rules:
      - expected matches observed -> confirmed
      - opposite direction -> rejected
      - no change or unknown -> inconclusive

    Returns:
        {"status": "confirmed"|"rejected"|"inconclusive"}

    No mutation of inputs.
    """
    ctype = conjecture.get("type", "")
    metric_config = _METRIC_MAP.get(ctype, {})
    expected = metric_config.get("expected_direction", "")

    observed = result.get("direction", "none")

    if observed == "none" or not expected:
        return {"status": "inconclusive"}

    if expected == observed:
        return {"status": "confirmed"}

    # Opposite direction
    _OPPOSITES = {"increase": "decrease", "decrease": "increase"}
    if _OPPOSITES.get(expected) == observed:
        return {"status": "rejected"}

    return {"status": "inconclusive"}


# ---------------------------------------------------------------------------
# PART 4 — STRENGTH SCORING
# ---------------------------------------------------------------------------

_STRONG_EFFECT_THRESHOLD = 0.2


def score_validation(
    status: str,
    result: Dict[str, Any],
) -> int:
    """Score validation outcome by status and effect strength.

    Rules:
      confirmed + strong effect (|change| >= 0.2) -> +2
      confirmed + weak effect -> +1
      inconclusive -> 0
      rejected -> -1

    Returns integer score. No mutation of inputs.
    """
    if status == "confirmed":
        change = abs(result.get("observed_change", 0.0))
        if change >= _STRONG_EFFECT_THRESHOLD:
            return 2
        return 1
    elif status == "rejected":
        return -1
    return 0


# ---------------------------------------------------------------------------
# PART 5 — PIPELINE
# ---------------------------------------------------------------------------


def run_conjecture_execution(
    data: Any,
) -> Dict[str, Any]:
    """Run the full conjecture execution pipeline.

    Steps:
      1. Normalize inputs (load conjectures + systems)
      2. Execute tests for each conjecture
      3. Validate outcomes
      4. Score results

    Returns:
        {
          "results": [
            {
              "statement": str,
              "status": str,
              "score": int,
              "observed_change": float,
            },
            ...
          ],
          "summary": {
            "confirmed": int,
            "rejected": int,
            "inconclusive": int,
          },
        }

    No mutation of inputs. Deterministic output.
    """
    inputs = normalize_conjecture_inputs(data)
    conjectures = inputs["conjectures"]
    systems = inputs["systems"]

    results: List[Dict[str, Any]] = []
    counts = {"confirmed": 0, "rejected": 0, "inconclusive": 0}

    for conj in conjectures:
        exec_result = execute_conjecture(conj, systems)
        validation = validate_conjecture(conj, exec_result)
        status = validation["status"]
        score = score_validation(status, exec_result)

        results.append({
            "statement": conj.get("statement", ""),
            "status": status,
            "score": score,
            "observed_change": exec_result["observed_change"],
        })
        counts[status] += 1

    # Deterministic sort: status priority, then statement
    _STATUS_ORDER = {"confirmed": 0, "rejected": 1, "inconclusive": 2}
    results.sort(
        key=lambda r: (_STATUS_ORDER.get(r["status"], 3), r["statement"])
    )

    return {
        "results": results,
        "summary": dict(counts),
    }


# ---------------------------------------------------------------------------
# PART 6 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_conjecture_results(report: Dict[str, Any]) -> str:
    """Format conjecture execution results as human-readable text.

    Returns deterministic string output. No mutation of inputs.
    """
    lines: List[str] = []
    lines.append("=== Conjecture Results ===")
    lines.append("")

    results = report.get("results", [])
    summary = report.get("summary", {})

    if not results:
        lines.append("No conjecture results.")
        return "\n".join(lines)

    for r in results:
        status = r.get("status", "inconclusive").upper()
        statement = r.get("statement", "")
        change = r.get("observed_change", 0.0)
        score = r.get("score", 0)

        lines.append("[{}] {} (score={})".format(status, statement, score))
        lines.append("  observed: {:.2f}".format(change))
        lines.append("")

    lines.append("--- Summary ---")
    lines.append("confirmed: {}".format(summary.get("confirmed", 0)))
    lines.append("rejected: {}".format(summary.get("rejected", 0)))
    lines.append("inconclusive: {}".format(summary.get("inconclusive", 0)))

    return "\n".join(lines)
