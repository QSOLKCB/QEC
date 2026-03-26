"""v105.1.0 — Cross-run invariant registry with law stability and lifecycle.

Maintains a persistent registry of invariants discovered across multiple
runs, enabling cross-run aggregation, frequency tracking, emergent law
detection, law stability scoring, invariant lifecycle tracking, and
cross-run drift detection.

Registry structure per invariant key:
    count: int           — number of runs where invariant was observed
    avg_strength: float  — running average of invariant strength
    max_strength: float  — maximum observed strength
    first_seen: int      — earliest run_id where invariant appeared
    last_seen: int       — latest run_id where invariant appeared
    streak: int          — consecutive runs where invariant was observed
    break_count: int     — number of times invariant was absent after first seen
    last_observed: bool  — whether invariant was present in last update

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Emergent law detection thresholds
# ---------------------------------------------------------------------------

_MIN_COUNT_FOR_LAW = 3
_MIN_AVG_STRENGTH_FOR_LAW = 0.6
_MIN_CONFIDENCE_FOR_LAW = 0.5

# ---------------------------------------------------------------------------
# Law stability scoring weights (deterministic, sum to 1.0)
# ---------------------------------------------------------------------------

_W_FREQUENCY = 0.25
_W_STABILITY = 0.25
_W_ROBUSTNESS = 0.25
_W_STRENGTH = 0.25

# Law stability threshold for emergent law detection
_MIN_LAW_STABILITY = 0.3


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1. Registry initialization
# ---------------------------------------------------------------------------


def init_registry() -> dict:
    """Create an empty invariant registry.

    Returns
    -------
    dict
        Empty registry dict ready for ``update_registry`` calls.
    """
    return {}


# ---------------------------------------------------------------------------
# 2. Canonical invariant key
# ---------------------------------------------------------------------------


def canonicalize_invariant(inv: dict) -> str:
    """Produce a deterministic canonical string key for an invariant.

    The key encodes the invariant's type and name in a stable format
    suitable for use as a registry lookup key.

    Parameters
    ----------
    inv : dict
        An invariant dict with at least ``type`` and ``name`` keys.

    Returns
    -------
    str
        Canonical key of the form ``"type:name"``.
    """
    inv_type = str(inv.get("type", "unknown"))
    inv_name = str(inv.get("name", "unknown"))
    return f"{inv_type}:{inv_name}"


# ---------------------------------------------------------------------------
# 3. Registry update
# ---------------------------------------------------------------------------


def update_registry(
    registry: dict,
    invariants: dict,
    run_id: int,
) -> dict:
    """Update the registry with invariants from a single run.

    Creates a new registry dict (no mutation of inputs).

    Parameters
    ----------
    registry : dict
        Existing registry (or empty dict from ``init_registry``).
    invariants : dict
        Output of ``extract_treatment_invariants`` or ``score_invariants``.
        Must contain either ``scored_invariants`` or ``invariants`` key.
    run_id : int
        Monotonically increasing run identifier.

    Returns
    -------
    dict
        Updated registry with new counts, strengths, and timestamps.
    """
    # Deep copy to avoid mutation.
    new_registry: Dict[str, Dict[str, Any]] = {}
    for key in sorted(registry.keys()):
        entry = registry[key]
        new_registry[key] = {
            "count": int(entry.get("count", 0)),
            "avg_strength": float(entry.get("avg_strength", 0.0)),
            "max_strength": float(entry.get("max_strength", 0.0)),
            "first_seen": int(entry.get("first_seen", run_id)),
            "last_seen": int(entry.get("last_seen", run_id)),
            "streak": int(entry.get("streak", 0)),
            "break_count": int(entry.get("break_count", 0)),
            "last_observed": bool(entry.get("last_observed", False)),
        }

    # Extract invariant list from either format.
    inv_list: List[Dict[str, Any]] = []
    if "scored_invariants" in invariants:
        inv_list = list(invariants["scored_invariants"])
    elif "invariants" in invariants:
        inv_list = list(invariants["invariants"])

    for inv in inv_list:
        key = canonicalize_invariant(inv)
        strength = float(inv.get("strength", inv.get("support", 0)))

        # Normalize strength from support/total if no explicit strength.
        if "strength" not in inv and "total" in inv:
            total = max(int(inv.get("total", 1)), 1)
            strength = _clamp(float(inv.get("support", 0)) / total)

        if key in new_registry:
            entry = new_registry[key]
            old_count = entry["count"]
            old_avg = entry["avg_strength"]

            new_count = old_count + 1
            # Incremental average: new_avg = old_avg + (strength - old_avg) / new_count
            new_avg = _round(old_avg + (strength - old_avg) / new_count)
            new_max = max(entry["max_strength"], strength)

            new_registry[key] = {
                "count": new_count,
                "avg_strength": new_avg,
                "max_strength": _round(new_max),
                "first_seen": entry["first_seen"],
                "last_seen": run_id,
                "streak": entry["streak"] + 1,
                "break_count": entry["break_count"],
                "last_observed": True,
            }
        else:
            new_registry[key] = {
                "count": 1,
                "avg_strength": _round(strength),
                "max_strength": _round(strength),
                "first_seen": run_id,
                "last_seen": run_id,
                "streak": 1,
                "break_count": 0,
                "last_observed": True,
            }

    return new_registry


# ---------------------------------------------------------------------------
# 3b. Incremental registry update (with break tracking)
# ---------------------------------------------------------------------------


def update_registry_incremental(
    registry: dict,
    invariants: dict,
    run_id: int,
) -> dict:
    """Update the registry incrementally, tracking breaks for absent invariants.

    Unlike ``update_registry``, this also marks previously-seen invariants
    that are *absent* in the current run, incrementing their break_count
    and resetting their streak.

    Invariants are assumed true until contradicted.

    Parameters
    ----------
    registry : dict
        Existing registry (or empty dict from ``init_registry``).
    invariants : dict
        Output of ``extract_treatment_invariants`` or ``score_invariants``.
        Must contain either ``scored_invariants`` or ``invariants`` key.
    run_id : int
        Monotonically increasing run identifier.

    Returns
    -------
    dict
        Updated registry with streak/break tracking for all known invariants.
    """
    # First, apply the standard update for present invariants.
    updated = update_registry(registry, invariants, run_id)

    # Collect keys that were present in this update.
    inv_list: List[Dict[str, Any]] = []
    if "scored_invariants" in invariants:
        inv_list = list(invariants["scored_invariants"])
    elif "invariants" in invariants:
        inv_list = list(invariants["invariants"])

    present_keys: set = set()
    for inv in inv_list:
        present_keys.add(canonicalize_invariant(inv))

    # For previously-known invariants that are absent, mark the break.
    result: Dict[str, Dict[str, Any]] = {}
    for key in sorted(updated.keys()):
        entry = dict(updated[key])
        if key not in present_keys and key in registry:
            # Invariant was known but absent in this run.
            entry["streak"] = 0
            entry["break_count"] = int(entry.get("break_count", 0)) + 1
            entry["last_observed"] = False
        result[key] = entry

    return result


# ---------------------------------------------------------------------------
# 4. Emergent law detection
# ---------------------------------------------------------------------------


def detect_emergent_laws(registry: dict) -> list:
    """Detect emergent laws from the invariant registry.

    An invariant qualifies as an emergent law when it has:
    - high frequency (count >= threshold)
    - high average strength
    - high stability score (v105.1)
    - stability across runs

    Parameters
    ----------
    registry : dict
        Invariant registry from ``update_registry``.

    Returns
    -------
    list of dict
        Each dict has ``law``, ``support``, ``confidence``,
        ``stability_score``, ``classification`` keys.
        Sorted by confidence DESC, law ASC for determinism.
    """
    laws: List[Dict[str, Any]] = []

    for key in sorted(registry.keys()):
        entry = registry[key]
        count = int(entry.get("count", 0))
        avg_strength = float(entry.get("avg_strength", 0.0))
        max_strength = float(entry.get("max_strength", 0.0))
        first_seen = int(entry.get("first_seen", 0))
        last_seen = int(entry.get("last_seen", 0))

        if count < _MIN_COUNT_FOR_LAW:
            continue
        if avg_strength < _MIN_AVG_STRENGTH_FOR_LAW:
            continue

        # Confidence: combines frequency strength and consistency.
        # consistency = avg / max (how close average is to peak).
        consistency = avg_strength / max(max_strength, 1e-12)
        # span = how many distinct run positions observed.
        span = last_seen - first_seen + 1
        # frequency_ratio: count relative to span.
        freq_ratio = _clamp(count / max(span, 1))

        confidence = _round(_clamp(
            0.4 * avg_strength + 0.3 * consistency + 0.3 * freq_ratio
        ))

        if confidence < _MIN_CONFIDENCE_FOR_LAW:
            continue

        # v105.1: compute law stability and classification.
        law_stability = compute_law_stability(entry, span)
        classification = classify_law(entry)

        if law_stability < _MIN_LAW_STABILITY:
            continue

        laws.append({
            "law": key,
            "support": count,
            "confidence": confidence,
            "stability_score": law_stability,
            "classification": classification,
        })

    # Sort by confidence DESC, law ASC.
    laws.sort(key=lambda x: (-x["confidence"], x["law"]))

    return laws


# ---------------------------------------------------------------------------
# 4b. Law stability scoring
# ---------------------------------------------------------------------------


def compute_law_stability(entry: dict, total_runs: int = 0) -> float:
    """Compute a deterministic stability score for a registry entry.

    Combines four components with fixed weights:
    - frequency_score:  count / total_runs
    - stability_score:  streak / count
    - robustness_score: 1 - (break_count / count)
    - strength_score:   avg_strength

    Parameters
    ----------
    entry : dict
        A single registry entry.
    total_runs : int
        Total number of runs observed.  If 0, derived from entry span.

    Returns
    -------
    float
        Stability score in [0, 1].
    """
    count = max(int(entry.get("count", 0)), 1)
    streak = int(entry.get("streak", 0))
    break_count = int(entry.get("break_count", 0))
    avg_strength = float(entry.get("avg_strength", 0.0))
    first_seen = int(entry.get("first_seen", 0))
    last_seen = int(entry.get("last_seen", 0))

    if total_runs <= 0:
        total_runs = max(last_seen - first_seen + 1, 1)

    frequency_score = _clamp(count / max(total_runs, 1))
    stability_score = _clamp(streak / max(count, 1))
    robustness_score = _clamp(1.0 - (break_count / max(count, 1)))
    strength_score = _clamp(avg_strength)

    result = (
        _W_FREQUENCY * frequency_score
        + _W_STABILITY * stability_score
        + _W_ROBUSTNESS * robustness_score
        + _W_STRENGTH * strength_score
    )

    return _round(_clamp(result))


# ---------------------------------------------------------------------------
# 4c. Law classification
# ---------------------------------------------------------------------------


def classify_law(entry: dict) -> str:
    """Classify a registry entry into a law category.

    Categories:
    - ``stable_law``:   high stability, low breaks
    - ``fragile_law``:  frequent but breaks often
    - ``emerging_law``: recent high streak, not yet fully established
    - ``decaying_law``: breaks increasing, streak declining

    Parameters
    ----------
    entry : dict
        A single registry entry.

    Returns
    -------
    str
        One of ``stable_law``, ``fragile_law``, ``emerging_law``,
        ``decaying_law``.
    """
    count = max(int(entry.get("count", 0)), 1)
    streak = int(entry.get("streak", 0))
    break_count = int(entry.get("break_count", 0))

    break_ratio = break_count / max(count, 1)
    streak_ratio = streak / max(count, 1)

    # Emerging: not yet enough history to classify further.
    if count < _MIN_COUNT_FOR_LAW:
        return "emerging_law"

    # Stable: low break ratio and reasonable streak.
    if break_ratio <= 0.2 and streak_ratio >= 0.5:
        return "stable_law"

    # Decaying: high break ratio and low current streak.
    if break_ratio > 0.4 and streak_ratio < 0.3:
        return "decaying_law"

    # Fragile: has been observed frequently but breaks often.
    if break_ratio > 0.2:
        return "fragile_law"

    # Default: emerging pattern not yet classifiable.
    return "emerging_law"


# ---------------------------------------------------------------------------
# 4d. Invariant lifecycle tracking
# ---------------------------------------------------------------------------


def track_invariant_lifecycle(entry: dict) -> dict:
    """Determine the lifecycle phase and trend of an invariant.

    Phases:
    - ``emerging``:  recently appeared, building streak
    - ``stable``:    consistently observed, low breaks
    - ``unstable``:  mixed presence, some breaks
    - ``decaying``:  breaks increasing, streak declining

    Trends:
    - ``strengthening``: streak growing relative to count
    - ``weakening``:     breaks growing relative to count
    - ``oscillating``:   mixed pattern

    Parameters
    ----------
    entry : dict
        A single registry entry.

    Returns
    -------
    dict
        ``{"phase": str, "trend": str}``.
    """
    count = max(int(entry.get("count", 0)), 1)
    streak = int(entry.get("streak", 0))
    break_count = int(entry.get("break_count", 0))
    last_observed = bool(entry.get("last_observed", False))

    break_ratio = break_count / max(count, 1)
    streak_ratio = streak / max(count, 1)

    # Determine phase (check emerging first — not yet established).
    if count < _MIN_COUNT_FOR_LAW:
        phase = "emerging"
    elif break_ratio <= 0.1 and streak_ratio >= 0.5:
        phase = "stable"
    elif break_ratio > 0.4:
        phase = "decaying"
    else:
        phase = "unstable"

    # Determine trend.
    if last_observed and streak_ratio >= 0.5:
        trend = "strengthening"
    elif not last_observed or break_ratio > 0.3:
        trend = "weakening"
    else:
        trend = "oscillating"

    return {"phase": phase, "trend": trend}


# ---------------------------------------------------------------------------
# 4e. Cross-run drift detection
# ---------------------------------------------------------------------------


def detect_invariant_drift(registry: dict) -> dict:
    """Detect drift patterns across the invariant registry.

    Categorizes invariants into:
    - ``drifting_invariants``:  losing strength or increasing breaks
    - ``stable_invariants``:   consistently observed with low breaks
    - ``new_invariants``:      recently appeared (count < threshold)
    - ``lost_invariants``:     not observed recently, high break count

    Parameters
    ----------
    registry : dict
        Invariant registry from ``update_registry_incremental``.

    Returns
    -------
    dict
        Contains ``drifting_invariants``, ``stable_invariants``,
        ``new_invariants``, ``lost_invariants`` — each a list of
        ``{"key": str, "entry": dict, "lifecycle": dict}`` dicts.
        All lists sorted by key for determinism.
    """
    drifting: List[Dict[str, Any]] = []
    stable: List[Dict[str, Any]] = []
    new: List[Dict[str, Any]] = []
    lost: List[Dict[str, Any]] = []

    for key in sorted(registry.keys()):
        entry = registry[key]
        count = int(entry.get("count", 0))
        break_count = int(entry.get("break_count", 0))
        last_observed = bool(entry.get("last_observed", False))
        lifecycle = track_invariant_lifecycle(entry)

        item = {"key": key, "entry": dict(entry), "lifecycle": lifecycle}

        # New: not yet established.
        if count < _MIN_COUNT_FOR_LAW:
            new.append(item)
            continue

        # Lost: not recently observed and many breaks.
        if not last_observed and break_count > count * 0.5:
            lost.append(item)
            continue

        # Drifting: lifecycle shows decaying or weakening.
        if lifecycle["phase"] == "decaying" or lifecycle["trend"] == "weakening":
            drifting.append(item)
            continue

        # Stable: everything else that's established.
        stable.append(item)

    return {
        "drifting_invariants": drifting,
        "stable_invariants": stable,
        "new_invariants": new,
        "lost_invariants": lost,
    }


# ---------------------------------------------------------------------------
# 5. Full registry analysis pipeline
# ---------------------------------------------------------------------------


def run_invariant_registry_analysis(
    run_results: list,
) -> dict:
    """Run invariant registry analysis across multiple run results.

    Each run result should contain ``scored_invariants`` or ``invariants``.

    Parameters
    ----------
    run_results : list of dict
        List of per-run results, each containing invariant data.

    Returns
    -------
    dict
        Contains ``registry``, ``emergent_laws``, ``top_invariants``.
    """
    registry = init_registry()

    for run_id, result in enumerate(run_results):
        # Accept either scored_invariants or invariants key.
        inv_data: Dict[str, Any] = {}
        if "scored_invariants" in result:
            inv_data["scored_invariants"] = result["scored_invariants"]
        elif "invariants" in result:
            inv_data["invariants"] = result.get("invariants", {}).get(
                "invariants", result.get("invariants", [])
            )
            if isinstance(inv_data["invariants"], dict):
                inv_data["invariants"] = inv_data["invariants"].get("invariants", [])

        registry = update_registry(registry, inv_data, run_id)

    laws = detect_emergent_laws(registry)

    # Extract top invariants by count.
    top: List[Dict[str, Any]] = []
    for key in sorted(registry.keys()):
        entry = registry[key]
        top.append({
            "invariant": key,
            "count": entry["count"],
            "avg_strength": entry["avg_strength"],
            "max_strength": entry["max_strength"],
        })
    top.sort(key=lambda x: (-x["count"], -x["avg_strength"], x["invariant"]))

    return {
        "registry": registry,
        "emergent_laws": laws,
        "top_invariants": top[:10],
    }


# ---------------------------------------------------------------------------
# 6. Formatter
# ---------------------------------------------------------------------------


def format_invariant_registry(registry: dict, laws: list) -> str:
    """Format invariant registry and emergent laws as human-readable text.

    Parameters
    ----------
    registry : dict
        Invariant registry from ``update_registry``.
    laws : list of dict
        Emergent laws from ``detect_emergent_laws``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []

    lines.append("")
    lines.append("=== Invariant Registry ===")

    if not registry:
        lines.append("")
        lines.append("No invariants registered.")
        return "\n".join(lines)

    lines.append("")
    lines.append("Registered Invariants:")

    # Sort by count DESC, key ASC.
    sorted_keys = sorted(registry.keys(), key=lambda k: (-registry[k]["count"], k))
    for i, key in enumerate(sorted_keys):
        entry = registry[key]
        lines.append(
            f"  {i + 1}. {key}  "
            f"count={entry['count']}  "
            f"avg={entry['avg_strength']:.2f}  "
            f"max={entry['max_strength']:.2f}  "
            f"runs={entry['first_seen']}-{entry['last_seen']}"
        )

    if laws:
        lines.append("")
        lines.append("Emergent Laws:")
        for i, law in enumerate(laws):
            lines.append(
                f"  {i + 1}. {law['law']}  "
                f"support={law['support']}  "
                f"confidence={law['confidence']:.2f}"
            )
    else:
        lines.append("")
        lines.append("No emergent laws detected.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Incremental analysis pipeline
# ---------------------------------------------------------------------------


def run_incremental_invariant_analysis(
    run_results: list,
) -> dict:
    """Run incremental invariant analysis with lifecycle and drift tracking.

    Uses ``update_registry_incremental`` to track absent invariants,
    then computes law stability, lifecycle, and drift information.

    Parameters
    ----------
    run_results : list of dict
        List of per-run results, each containing invariant data.

    Returns
    -------
    dict
        Contains ``registry``, ``emergent_laws``, ``top_invariants``,
        ``drift``, ``law_summary``.
    """
    registry = init_registry()

    for run_id, result in enumerate(run_results):
        inv_data: Dict[str, Any] = {}
        if "scored_invariants" in result:
            inv_data["scored_invariants"] = result["scored_invariants"]
        elif "invariants" in result:
            inv_data["invariants"] = result.get("invariants", {}).get(
                "invariants", result.get("invariants", [])
            )
            if isinstance(inv_data["invariants"], dict):
                inv_data["invariants"] = inv_data["invariants"].get("invariants", [])

        registry = update_registry_incremental(registry, inv_data, run_id)

    laws = detect_emergent_laws(registry)
    drift = detect_invariant_drift(registry)

    # Top invariants by count.
    top: List[Dict[str, Any]] = []
    total_runs = len(run_results)
    for key in sorted(registry.keys()):
        entry = registry[key]
        top.append({
            "invariant": key,
            "count": entry["count"],
            "avg_strength": entry["avg_strength"],
            "max_strength": entry["max_strength"],
            "stability_score": compute_law_stability(entry, total_runs),
            "classification": classify_law(entry),
            "lifecycle": track_invariant_lifecycle(entry),
        })
    top.sort(key=lambda x: (-x["count"], -x["avg_strength"], x["invariant"]))

    # Law summary statistics.
    law_summary = _compute_law_summary(registry, laws, drift, total_runs)

    return {
        "registry": registry,
        "emergent_laws": laws,
        "top_invariants": top[:10],
        "drift": drift,
        "law_summary": law_summary,
    }


def _compute_law_summary(
    registry: dict,
    laws: list,
    drift: dict,
    total_runs: int,
) -> dict:
    """Compute summary statistics for law stability diagnostics.

    Parameters
    ----------
    registry : dict
        Invariant registry.
    laws : list
        Emergent laws.
    drift : dict
        Drift detection results.
    total_runs : int
        Total number of runs analyzed.

    Returns
    -------
    dict
        Summary statistics.
    """
    # Count laws by classification.
    stable_count = 0
    emerging_count = 0
    fragile_count = 0
    decaying_count = 0
    for law in laws:
        cls = law.get("classification", "")
        if cls == "stable_law":
            stable_count += 1
        elif cls == "emerging_law":
            emerging_count += 1
        elif cls == "fragile_law":
            fragile_count += 1
        elif cls == "decaying_law":
            decaying_count += 1

    # Compute average law stability across registry.
    stability_scores: List[float] = []
    for key in sorted(registry.keys()):
        entry = registry[key]
        stability_scores.append(compute_law_stability(entry, total_runs))

    avg_stability = 0.0
    if stability_scores:
        avg_stability = _round(sum(stability_scores) / len(stability_scores))

    return {
        "law_stability_score": avg_stability,
        "stable_law_count": stable_count,
        "emerging_law_count": emerging_count,
        "fragile_law_count": fragile_count,
        "decaying_law_count": decaying_count,
        "drifting_invariant_count": len(drift.get("drifting_invariants", [])),
        "lost_invariant_count": len(drift.get("lost_invariants", [])),
        "new_invariant_count": len(drift.get("new_invariants", [])),
        "total_invariants": len(registry),
        "total_runs": total_runs,
    }


# ---------------------------------------------------------------------------
# 8. Law stability formatter
# ---------------------------------------------------------------------------


def format_law_stability_summary(analysis: dict) -> str:
    """Format law stability analysis as human-readable text.

    Parameters
    ----------
    analysis : dict
        Output of ``run_incremental_invariant_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []

    summary = analysis.get("law_summary", {})
    laws = analysis.get("emergent_laws", [])
    drift = analysis.get("drift", {})
    registry = analysis.get("registry", {})

    lines.append("")
    lines.append("=== Law Stability Summary ===")
    lines.append("")
    lines.append(f"  Total invariants: {summary.get('total_invariants', 0)}")
    lines.append(f"  Total runs: {summary.get('total_runs', 0)}")
    lines.append(f"  Avg stability score: {summary.get('law_stability_score', 0.0):.4f}")
    lines.append("")
    lines.append(f"  Stable laws: {summary.get('stable_law_count', 0)}")
    lines.append(f"  Emerging laws: {summary.get('emerging_law_count', 0)}")
    lines.append(f"  Fragile laws: {summary.get('fragile_law_count', 0)}")
    lines.append(f"  Decaying laws: {summary.get('decaying_law_count', 0)}")
    lines.append("")
    lines.append(f"  Drifting invariants: {summary.get('drifting_invariant_count', 0)}")
    lines.append(f"  Lost invariants: {summary.get('lost_invariant_count', 0)}")
    lines.append(f"  New invariants: {summary.get('new_invariant_count', 0)}")

    if laws:
        lines.append("")
        lines.append("Emergent Laws (with stability):")
        for i, law in enumerate(laws):
            cls = law.get("classification", "unknown")
            stab = law.get("stability_score", 0.0)
            lines.append(
                f"  {i + 1}. {law['law']}  "
                f"confidence={law['confidence']:.2f}  "
                f"stability={stab:.2f}  "
                f"class={cls}"
            )

    # Lifecycle for top invariants.
    top = analysis.get("top_invariants", [])
    if top:
        lines.append("")
        lines.append("Top Invariant Lifecycles:")
        for i, item in enumerate(top[:5]):
            lc = item.get("lifecycle", {})
            lines.append(
                f"  {i + 1}. {item['invariant']}  "
                f"phase={lc.get('phase', '?')}  "
                f"trend={lc.get('trend', '?')}  "
                f"class={item.get('classification', '?')}"
            )

    # Drift summary.
    drifting = drift.get("drifting_invariants", [])
    lost = drift.get("lost_invariants", [])
    if drifting or lost:
        lines.append("")
        lines.append("Drift Alerts:")
        for item in drifting:
            lines.append(f"  DRIFTING: {item['key']}")
        for item in lost:
            lines.append(f"  LOST: {item['key']}")

    return "\n".join(lines)


__all__ = [
    "ROUND_PRECISION",
    "canonicalize_invariant",
    "classify_law",
    "compute_law_stability",
    "detect_emergent_laws",
    "detect_invariant_drift",
    "format_invariant_registry",
    "format_law_stability_summary",
    "init_registry",
    "run_incremental_invariant_analysis",
    "run_invariant_registry_analysis",
    "track_invariant_lifecycle",
    "update_registry",
    "update_registry_incremental",
]
