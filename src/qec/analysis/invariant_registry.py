"""v105.0.0 — Cross-run invariant registry and emergent law detection.

Maintains a persistent registry of invariants discovered across multiple
runs, enabling cross-run aggregation, frequency tracking, and emergent
law detection.

Registry structure per invariant key:
    count: int           — number of runs where invariant was observed
    avg_strength: float  — running average of invariant strength
    max_strength: float  — maximum observed strength
    first_seen: int      — earliest run_id where invariant appeared
    last_seen: int       — latest run_id where invariant appeared

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
            }
        else:
            new_registry[key] = {
                "count": 1,
                "avg_strength": _round(strength),
                "max_strength": _round(strength),
                "first_seen": run_id,
                "last_seen": run_id,
            }

    return new_registry


# ---------------------------------------------------------------------------
# 4. Emergent law detection
# ---------------------------------------------------------------------------


def detect_emergent_laws(registry: dict) -> list:
    """Detect emergent laws from the invariant registry.

    An invariant qualifies as an emergent law when it has:
    - high frequency (count >= threshold)
    - high average strength
    - stability across runs

    Parameters
    ----------
    registry : dict
        Invariant registry from ``update_registry``.

    Returns
    -------
    list of dict
        Each dict has ``law``, ``support``, ``confidence`` keys.
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

        laws.append({
            "law": key,
            "support": count,
            "confidence": confidence,
        })

    # Sort by confidence DESC, law ASC.
    laws.sort(key=lambda x: (-x["confidence"], x["law"]))

    return laws


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


__all__ = [
    "ROUND_PRECISION",
    "canonicalize_invariant",
    "detect_emergent_laws",
    "format_invariant_registry",
    "init_registry",
    "run_invariant_registry_analysis",
    "update_registry",
]
