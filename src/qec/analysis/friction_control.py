"""Friction-aware correction control (v96.4.0).

Upgrades from:
    measure friction
to:
    use friction to guide correction strategy

The system:
  - avoids high-friction correction paths
  - prefers low-dissipation strategies
  - chooses modes using performance (v96.0-v96.2) and efficiency (v96.3)

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No weight tuning. No learning.
"""

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PART 1 — INPUT NORMALIZATION
# ---------------------------------------------------------------------------


def normalize_control_inputs(data: Any) -> List[Dict[str, Any]]:
    """Combine performance and friction data into per-system candidates.

    Accepts a dict with keys from upstream modules:
      - "dynamics_results": output of run_correction_dynamics() (v96.3)
      - "hierarchical_results": from core_invariants pipeline (v96.0)
      - "invariant_analysis": from invariant_analysis (v95.2)
      - "law_extraction": from law_extraction (v96.2)

    Returns list of candidate dicts:
        {
          "dfa_type": str,
          "n": Optional[int],
          "mode": str,
          "stability_efficiency": float,
          "compression_efficiency": float,
          "friction_score": float,
          "regime": str,
          "core_invariants": [...],
        }

    Deterministic ordering. No mutation of inputs.
    """
    if not isinstance(data, dict):
        return []

    candidates: List[Dict[str, Any]] = []

    # Build friction lookup from dynamics results.
    friction_map = _build_friction_map(data)

    # Build core invariant lookup from law extraction.
    core_inv_map = _build_core_invariant_map(data)

    # Extract candidates from hierarchical results.
    candidates.extend(
        _extract_hierarchical_candidates(data, friction_map, core_inv_map)
    )

    # Extract candidates from dynamics results (standalone systems).
    candidates.extend(
        _extract_dynamics_candidates(data, friction_map, core_inv_map)
    )

    # Deduplicate by (dfa_type, n, mode).
    candidates = _deduplicate_candidates(candidates)

    # Deterministic sort.
    candidates.sort(key=_candidate_sort_key)
    return candidates


def _candidate_sort_key(c: Dict[str, Any]) -> Tuple:
    """Stable sort key for candidates."""
    return (
        c.get("dfa_type", ""),
        str(c.get("n", "")),
        c.get("mode", ""),
    )


def _build_friction_map(
    data: Dict[str, Any],
) -> Dict[Tuple[str, Optional[int]], Dict[str, Any]]:
    """Build (dfa_type, n) -> friction data lookup."""
    fmap: Dict[Tuple[str, Optional[int]], Dict[str, Any]] = {}

    dynamics = data.get("dynamics_results", {})
    results = dynamics.get("results", [])
    for r in results:
        key = (r.get("dfa_type", ""), r.get("n"))
        fmap[key] = {
            "friction_score": r.get("friction_score", 0.0),
            "regime": r.get("regime", "stable"),
            "components": dict(r.get("components", {})),
        }

    return fmap


def _build_core_invariant_map(
    data: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Build system_class -> [core invariant types] lookup."""
    inv_map: Dict[str, List[str]] = {}

    law_data = data.get("law_extraction", {})
    laws = law_data.get("laws", [])
    for law in laws:
        if law.get("law_type") != "core_invariant_law":
            continue
        concl = law.get("conclusion", {})
        inv_type = law.get("condition", {}).get("invariant_type", "")
        classes = concl.get("classes", [])
        for cls in classes:
            inv_map.setdefault(cls, [])
            if inv_type and inv_type not in inv_map[cls]:
                inv_map[cls].append(inv_type)

    # Sort each list for determinism.
    for cls in inv_map:
        inv_map[cls] = sorted(inv_map[cls])

    return inv_map


def _extract_hierarchical_candidates(
    data: Dict[str, Any],
    friction_map: Dict[Tuple[str, Optional[int]], Dict[str, Any]],
    core_inv_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Extract candidates from hierarchical results."""
    candidates: List[Dict[str, Any]] = []

    hier_results = data.get("hierarchical_results", [])
    for r in hier_results:
        dfa_type = r.get("dfa_name", r.get("dfa_type", ""))
        n = r.get("n")
        mode = r.get("mode", "none")
        stab = r.get("stability_efficiency", 0.0)
        comp = r.get("compression_efficiency", 0.0)

        fkey = (dfa_type, n)
        friction_data = friction_map.get(fkey, {})
        friction_score = friction_data.get("friction_score", 0.0)
        regime = friction_data.get("regime", "stable")

        sys_class = r.get("system_class", "")
        core_invs = sorted(core_inv_map.get(sys_class, []))

        candidates.append({
            "dfa_type": dfa_type,
            "n": n,
            "mode": mode,
            "stability_efficiency": float(stab),
            "compression_efficiency": float(comp),
            "friction_score": float(friction_score),
            "regime": regime,
            "core_invariants": list(core_invs),
        })

    return candidates


def _extract_dynamics_candidates(
    data: Dict[str, Any],
    friction_map: Dict[Tuple[str, Optional[int]], Dict[str, Any]],
    core_inv_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Extract candidates from dynamics results not in hierarchical."""
    candidates: List[Dict[str, Any]] = []

    # Find systems already covered by hierarchical.
    hier_systems: set = set()
    for r in data.get("hierarchical_results", []):
        dfa_type = r.get("dfa_name", r.get("dfa_type", ""))
        hier_systems.add((dfa_type, r.get("n")))

    dynamics = data.get("dynamics_results", {})
    for r in dynamics.get("results", []):
        dfa_type = r.get("dfa_type", "")
        n = r.get("n")
        key = (dfa_type, n)

        if key in hier_systems:
            continue

        friction_score = r.get("friction_score", 0.0)
        regime = r.get("regime", "stable")

        candidates.append({
            "dfa_type": dfa_type,
            "n": n,
            "mode": "baseline",
            "stability_efficiency": 0.0,
            "compression_efficiency": 0.0,
            "friction_score": float(friction_score),
            "regime": regime,
            "core_invariants": [],
        })

    return candidates


def _deduplicate_candidates(
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate (dfa_type, n, mode) entries, keeping first."""
    seen: set = set()
    result: List[Dict[str, Any]] = []

    for c in candidates:
        key = (c["dfa_type"], c.get("n"), c["mode"])
        if key not in seen:
            seen.add(key)
            result.append(c)

    return result


# ---------------------------------------------------------------------------
# PART 2 — EFFICIENCY METRIC
# ---------------------------------------------------------------------------


def compute_efficiency(record: Dict[str, Any]) -> float:
    """Compute friction-aware efficiency for a candidate.

    Rule:
        efficiency = stability_efficiency / (1 + friction_score)

    Properties:
      - high stability + low friction -> best
      - high friction penalized
      - bounded, deterministic
      - returns 0.0 if stability_efficiency <= 0

    No mutation of record.
    """
    stab = float(record.get("stability_efficiency", 0.0))
    friction = float(record.get("friction_score", 0.0))

    if stab <= 0.0:
        return 0.0

    return round(stab / (1.0 + friction), 6)


# ---------------------------------------------------------------------------
# PART 3 — MODE SELECTION (CORE)
# ---------------------------------------------------------------------------


def rank_control_strategies(
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank candidates by friction-aware efficiency.

    Sort order (descending priority):
      1. efficiency (desc)
      2. stability_efficiency (desc)
      3. compression_efficiency (desc)
      4. lower friction_score (asc)
      5. lexicographic mode (asc, tiebreak)

    Returns new list of dicts with 'efficiency' and 'rank' added.
    No mutation of input records.
    """
    enriched = []
    for r in records:
        entry = dict(r)
        entry["efficiency"] = compute_efficiency(r)
        enriched.append(entry)

    enriched.sort(key=lambda x: (
        -x["efficiency"],
        -x.get("stability_efficiency", 0.0),
        -x.get("compression_efficiency", 0.0),
        x.get("friction_score", 0.0),
        x.get("mode", ""),
    ))

    for i, entry in enumerate(enriched):
        entry["rank"] = i + 1

    return enriched


def select_best_strategy(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the best strategy from ranked candidates.

    Returns:
        {
          "mode": str,
          "efficiency": float,
          "friction_score": float,
          "reason": str,
        }

    Returns empty result if no records.
    """
    if not records:
        return {
            "mode": "none",
            "efficiency": 0.0,
            "friction_score": 0.0,
            "reason": "no candidates available",
        }

    ranked = rank_control_strategies(records)
    best = ranked[0]

    reason_parts = []
    if best.get("friction_score", 0.0) < 1.0:
        reason_parts.append("low friction path")
    elif best.get("friction_score", 0.0) <= 2.5:
        reason_parts.append("acceptable friction level")
    else:
        reason_parts.append("best available despite high friction")

    if best["efficiency"] > 0:
        reason_parts.append("highest efficiency score")

    if len(ranked) > 1:
        runner_up = ranked[1]
        if best["efficiency"] > runner_up["efficiency"]:
            reason_parts.append(
                "outperforms {} by {:.4f}".format(
                    runner_up.get("mode", "unknown"),
                    best["efficiency"] - runner_up["efficiency"],
                )
            )

    return {
        "mode": best.get("mode", "none"),
        "efficiency": best["efficiency"],
        "friction_score": best.get("friction_score", 0.0),
        "reason": "; ".join(reason_parts) if reason_parts else "default",
    }


# ---------------------------------------------------------------------------
# PART 4 — FRICTION-AWARE RULES
# ---------------------------------------------------------------------------


# Thresholds — fixed constants, not tunable.
_HIGH_FRICTION_THRESHOLD = 2.5
_OSCILLATION_THRESHOLD = 0.5
_CHURN_THRESHOLD = 0.6
_LOW_STABILITY_THRESHOLD = 0.3

# Multi-stage modes that should be avoided under high friction.
_MULTI_STAGE_MODES = frozenset([
    "d4>e8_like",
    "square>d4",
    "d4>square",
    "e8_like>d4",
    "square>e8_like",
    "e8_like>square",
])

# Simpler modes preferred under oscillation.
_SIMPLE_MODES = frozenset([
    "square",
    "d4",
    "baseline",
    "none",
])


def apply_friction_rules(
    record: Dict[str, Any],
    dynamics_components: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Apply deterministic friction-aware policy rules to a candidate.

    Rules:
      1. High friction override: IF friction_score > 2.5 AND mode is
         multi-stage THEN penalize.
      2. Oscillation suppression: IF oscillation_ratio high THEN prefer
         simpler modes.
      3. Churn penalty: IF churn high AND stability low THEN reject.
      4. Conflict avoidance: IF invariant conflicts present THEN
         deprioritize invariant-heavy modes.

    Args:
        record: candidate dict (not mutated).
        dynamics_components: optional friction component breakdown
            (oscillation, hysteresis, switching, churn, conflict).

    Returns:
        {
          "adjusted_efficiency": float,
          "rules_applied": [str],
          "rejected": bool,
        }
    """
    efficiency = compute_efficiency(record)
    friction = float(record.get("friction_score", 0.0))
    mode = record.get("mode", "")
    stab = float(record.get("stability_efficiency", 0.0))
    core_invs = record.get("core_invariants", [])

    components = dynamics_components or {}
    oscillation = float(components.get("oscillation", 0.0))
    churn = float(components.get("churn", 0.0))
    conflict = float(components.get("conflict", 0.0))

    rules_applied: List[str] = []
    rejected = False
    penalty = 0.0

    # Rule 1: High friction override.
    if friction > _HIGH_FRICTION_THRESHOLD and mode in _MULTI_STAGE_MODES:
        penalty += 0.3
        rules_applied.append(
            "high_friction_override: multi-stage mode penalized"
        )

    # Rule 2: Oscillation suppression.
    if oscillation > _OSCILLATION_THRESHOLD and mode not in _SIMPLE_MODES:
        penalty += 0.2
        rules_applied.append(
            "oscillation_suppression: non-simple mode penalized"
        )

    # Rule 3: Churn penalty.
    if churn > _CHURN_THRESHOLD and stab < _LOW_STABILITY_THRESHOLD:
        rejected = True
        rules_applied.append(
            "churn_rejection: high churn with low stability"
        )

    # Rule 4: Conflict avoidance.
    if conflict > 0 and len(core_invs) > 2:
        penalty += 0.1
        rules_applied.append(
            "conflict_avoidance: invariant-heavy mode deprioritized"
        )

    adjusted = max(round(efficiency - penalty, 6), 0.0)

    return {
        "adjusted_efficiency": adjusted,
        "rules_applied": rules_applied,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------------
# PART 5 — HYBRID CONTROL
# ---------------------------------------------------------------------------


def compute_control_decision(
    records: List[Dict[str, Any]],
    dynamics_components_map: Optional[
        Dict[Tuple[str, Optional[int]], Dict[str, float]]
    ] = None,
) -> Dict[str, Any]:
    """Combine ranking and friction rules into a control decision.

    Pipeline:
      1. Compute efficiency for each candidate.
      2. Apply friction rules.
      3. Re-rank with adjusted efficiencies.
      4. Select best non-rejected candidate.

    Args:
        records: list of candidate dicts.
        dynamics_components_map: optional mapping of (dfa_type, n) ->
            friction component breakdown.

    Returns:
        {
          "best_mode": str,
          "efficiency": float,
          "friction_score": float,
          "regime": str,
          "alternatives": [...],
          "decision_trace": [...],
        }
    """
    if not records:
        return {
            "best_mode": "none",
            "efficiency": 0.0,
            "friction_score": 0.0,
            "regime": "stable",
            "alternatives": [],
            "decision_trace": [],
        }

    comp_map = dynamics_components_map or {}

    # Step 1+2: compute efficiency and apply rules.
    evaluated: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []

    for r in records:
        key = (r.get("dfa_type", ""), r.get("n"))
        components = comp_map.get(key, {})
        rule_result = apply_friction_rules(r, components)

        entry = dict(r)
        entry["efficiency"] = compute_efficiency(r)
        entry["adjusted_efficiency"] = rule_result["adjusted_efficiency"]
        entry["rejected"] = rule_result["rejected"]
        entry["rules_applied"] = rule_result["rules_applied"]
        evaluated.append(entry)

        trace.append({
            "mode": r.get("mode", "none"),
            "dfa_type": r.get("dfa_type", ""),
            "n": r.get("n"),
            "efficiency": entry["efficiency"],
            "adjusted_efficiency": rule_result["adjusted_efficiency"],
            "rejected": rule_result["rejected"],
            "rules": list(rule_result["rules_applied"]),
        })

    # Step 3: Re-rank by adjusted efficiency.
    evaluated.sort(key=lambda x: (
        x["rejected"],  # non-rejected first
        -x["adjusted_efficiency"],
        -x.get("stability_efficiency", 0.0),
        -x.get("compression_efficiency", 0.0),
        x.get("friction_score", 0.0),
        x.get("mode", ""),
    ))

    # Step 4: Select best non-rejected.
    best = None
    alternatives: List[Dict[str, Any]] = []

    for i, entry in enumerate(evaluated):
        summary = {
            "mode": entry.get("mode", "none"),
            "efficiency": entry["efficiency"],
            "adjusted_efficiency": entry["adjusted_efficiency"],
            "friction_score": entry.get("friction_score", 0.0),
            "rejected": entry["rejected"],
        }

        if best is None and not entry["rejected"]:
            best = entry
        else:
            alternatives.append(summary)

    if best is None:
        # All rejected: pick least-bad.
        best = evaluated[0]
        alternatives = []
        for entry in evaluated[1:]:
            alternatives.append({
                "mode": entry.get("mode", "none"),
                "efficiency": entry["efficiency"],
                "adjusted_efficiency": entry["adjusted_efficiency"],
                "friction_score": entry.get("friction_score", 0.0),
                "rejected": entry["rejected"],
            })

    return {
        "best_mode": best.get("mode", "none"),
        "efficiency": best.get("adjusted_efficiency", 0.0),
        "friction_score": best.get("friction_score", 0.0),
        "regime": best.get("regime", "stable"),
        "alternatives": alternatives,
        "decision_trace": trace,
    }


# ---------------------------------------------------------------------------
# PART 6 — FULL PIPELINE
# ---------------------------------------------------------------------------


def run_friction_control(
    data: Any,
    dynamics_components_map: Optional[
        Dict[Tuple[str, Optional[int]], Dict[str, float]]
    ] = None,
) -> Dict[str, Any]:
    """Run the full friction-aware control pipeline.

    Steps:
      1. Normalize inputs.
      2. Compute efficiency.
      3. Apply rules.
      4. Select best strategy.

    Args:
        data: combined upstream data dict.
        dynamics_components_map: optional (dfa_type, n) -> components.

    Returns:
        {
          "candidates": [...],
          "decision": {...},
          "summary": {...},
        }
    """
    # Step 1: Normalize.
    candidates = normalize_control_inputs(data)

    # Step 2-4: Compute control decision.
    decision = compute_control_decision(candidates, dynamics_components_map)

    # Build summary.
    total = len(candidates)
    rejected_count = sum(
        1 for t in decision.get("decision_trace", [])
        if t.get("rejected", False)
    )
    regimes: Dict[str, int] = {}
    for c in candidates:
        regime = c.get("regime", "stable")
        regimes[regime] = regimes.get(regime, 0) + 1

    summary = {
        "total_candidates": total,
        "rejected_count": rejected_count,
        "accepted_count": total - rejected_count,
        "best_mode": decision["best_mode"],
        "best_efficiency": decision["efficiency"],
        "regimes": {k: v for k, v in sorted(regimes.items())},
    }

    return {
        "candidates": candidates,
        "decision": decision,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER
# ---------------------------------------------------------------------------


def print_control_report(report: Dict[str, Any]) -> str:
    """Format friction-aware control report as text.

    Returns formatted string (does not print directly).
    """
    lines = ["=== Friction-Aware Control ==="]

    decision = report.get("decision", {})
    summary = report.get("summary", {})

    best_mode = decision.get("best_mode", "none")
    efficiency = decision.get("efficiency", 0.0)
    friction = decision.get("friction_score", 0.0)
    regime = decision.get("regime", "stable")

    lines.append("best_mode: {}".format(best_mode))
    lines.append("efficiency: {:.6f}".format(efficiency))
    lines.append("friction_score: {:.1f}".format(friction))
    lines.append("regime: {}".format(regime))

    # Decision trace reasoning.
    trace = decision.get("decision_trace", [])
    reason_lines = []
    for t in trace:
        if t.get("mode") == best_mode and not t.get("rejected"):
            for rule in t.get("rules", []):
                reason_lines.append("  - {}".format(rule))
            break

    if reason_lines:
        lines.append("rules_applied:")
        lines.extend(reason_lines)

    # Alternatives.
    alternatives = decision.get("alternatives", [])
    if alternatives:
        lines.append("alternatives:")
        for i, alt in enumerate(alternatives[:5], 1):
            rejected_tag = " [REJECTED]" if alt.get("rejected") else ""
            lines.append(
                "  {}. {} (eff={:.6f}, friction={:.1f}){}".format(
                    i,
                    alt.get("mode", "?"),
                    alt.get("adjusted_efficiency", alt.get("efficiency", 0.0)),
                    alt.get("friction_score", 0.0),
                    rejected_tag,
                )
            )

    # Summary.
    lines.append("---")
    lines.append("total_candidates: {}".format(
        summary.get("total_candidates", 0)
    ))
    lines.append("accepted: {}".format(summary.get("accepted_count", 0)))
    lines.append("rejected: {}".format(summary.get("rejected_count", 0)))

    regimes = summary.get("regimes", {})
    if regimes:
        lines.append("regimes:")
        for regime_name in sorted(regimes.keys()):
            lines.append("  {}: {}".format(regime_name, regimes[regime_name]))

    return "\n".join(lines)
