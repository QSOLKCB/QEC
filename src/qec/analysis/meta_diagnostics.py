"""v105.0.0 — Cross-run meta-diagnostics.

Compares multiple run results to identify universal vs contextual
invariants, shared diagnoses, and topological patterns across runs.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------

_UNIVERSAL_RATIO = 0.8     # present in >= 80% of runs
_RARE_RATIO = 0.2          # present in <= 20% of runs


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# 1. Compare runs
# ---------------------------------------------------------------------------


def compare_runs(run_results: list) -> dict:
    """Compare invariants, diagnoses, and topologies across runs.

    Parameters
    ----------
    run_results : list of dict
        List of per-run system diagnostics results.  Each should contain
        ``global_metrics``, ``scored_invariants``, and optionally
        ``trajectory_geometry``.

    Returns
    -------
    dict
        Contains ``invariant_overlap``, ``invariant_divergence``,
        ``shared_diagnoses``, ``shared_topologies``, ``run_count``.
    """
    if not run_results:
        return {
            "invariant_overlap": [],
            "invariant_divergence": [],
            "shared_diagnoses": [],
            "shared_topologies": [],
            "run_count": 0,
        }

    n_runs = len(run_results)

    # Collect invariant keys per run.
    inv_per_run: List[Set[str]] = []
    for result in run_results:
        keys: Set[str] = set()
        scored = result.get("scored_invariants", [])
        if isinstance(scored, list):
            for inv in scored:
                inv_type = str(inv.get("type", "unknown"))
                inv_name = str(inv.get("name", "unknown"))
                keys.add(f"{inv_type}:{inv_name}")
        inv_per_run.append(keys)

    # Invariant overlap: keys present in ALL runs.
    if inv_per_run:
        overlap = sorted(set.intersection(*inv_per_run)) if inv_per_run else []
    else:
        overlap = []

    # Invariant divergence: keys present in exactly one run.
    all_keys: Dict[str, int] = {}
    for keys in inv_per_run:
        for k in keys:
            all_keys[k] = all_keys.get(k, 0) + 1
    divergence = sorted(k for k, v in all_keys.items() if v == 1)

    # Shared diagnoses.
    diagnosis_counts: Dict[str, int] = {}
    for result in run_results:
        gm = result.get("global_metrics", {})
        diag = str(gm.get("primary_diagnosis", "unknown"))
        diagnosis_counts[diag] = diagnosis_counts.get(diag, 0) + 1

    shared_diagnoses: List[Dict[str, Any]] = []
    for diag in sorted(diagnosis_counts.keys()):
        count = diagnosis_counts[diag]
        shared_diagnoses.append({
            "diagnosis": diag,
            "count": count,
            "ratio": _round(count / max(n_runs, 1)),
        })
    shared_diagnoses.sort(key=lambda x: (-x["count"], x["diagnosis"]))

    # Shared topologies.
    topology_counts: Dict[str, int] = {}
    for result in run_results:
        gm = result.get("global_metrics", {})
        topo = str(gm.get("topology_type", "unknown"))
        topology_counts[topo] = topology_counts.get(topo, 0) + 1

    shared_topologies: List[Dict[str, Any]] = []
    for topo in sorted(topology_counts.keys()):
        count = topology_counts[topo]
        shared_topologies.append({
            "topology": topo,
            "count": count,
            "ratio": _round(count / max(n_runs, 1)),
        })
    shared_topologies.sort(key=lambda x: (-x["count"], x["topology"]))

    return {
        "invariant_overlap": overlap,
        "invariant_divergence": divergence,
        "shared_diagnoses": shared_diagnoses,
        "shared_topologies": shared_topologies,
        "run_count": n_runs,
    }


# ---------------------------------------------------------------------------
# 2. Classify invariants: universal vs contextual vs rare
# ---------------------------------------------------------------------------


def classify_invariants(registry: dict) -> dict:
    """Classify invariants as universal, contextual, or rare.

    Uses the invariant registry (from ``update_registry``) to determine
    how broadly each invariant appears relative to total run count.

    Parameters
    ----------
    registry : dict
        Invariant registry.  Each value must have ``count`` and the
        registry should be populated from a known number of runs.

    Returns
    -------
    dict
        Contains ``universal``, ``contextual``, ``rare`` lists.
        Each entry is a dict with ``invariant``, ``count``,
        ``avg_strength``.
    """
    if not registry:
        return {"universal": [], "contextual": [], "rare": []}

    # Infer total runs from max last_seen + 1.
    max_run = 0
    for key in registry:
        last = int(registry[key].get("last_seen", 0))
        if last > max_run:
            max_run = last
    total_runs = max(max_run + 1, 1)

    universal: List[Dict[str, Any]] = []
    contextual: List[Dict[str, Any]] = []
    rare: List[Dict[str, Any]] = []

    for key in sorted(registry.keys()):
        entry = registry[key]
        count = int(entry.get("count", 0))
        avg_str = float(entry.get("avg_strength", 0.0))
        ratio = count / total_runs

        item = {
            "invariant": key,
            "count": count,
            "avg_strength": _round(avg_str),
            "ratio": _round(ratio),
        }

        if ratio >= _UNIVERSAL_RATIO:
            universal.append(item)
        elif ratio <= _RARE_RATIO:
            rare.append(item)
        else:
            contextual.append(item)

    # Sort each category by count DESC, key ASC.
    for lst in (universal, contextual, rare):
        lst.sort(key=lambda x: (-x["count"], x["invariant"]))

    return {
        "universal": universal,
        "contextual": contextual,
        "rare": rare,
    }


# ---------------------------------------------------------------------------
# 3. Full meta-diagnostics pipeline
# ---------------------------------------------------------------------------


def run_meta_diagnostics_analysis(
    run_results: list,
    registry: dict | None = None,
) -> dict:
    """Run cross-run meta-diagnostics analysis.

    Parameters
    ----------
    run_results : list of dict
        Per-run system diagnostics results.
    registry : dict, optional
        Pre-computed invariant registry.  If None, a registry is built
        from the run results.

    Returns
    -------
    dict
        Contains ``comparison``, ``classification``, ``registry``.
    """
    from qec.analysis.invariant_registry import (
        init_registry,
        update_registry,
    )

    comparison = compare_runs(run_results)

    # Build registry if not provided.
    if registry is None:
        registry = init_registry()
        for run_id, result in enumerate(run_results):
            inv_data: Dict[str, Any] = {}
            if "scored_invariants" in result:
                inv_data["scored_invariants"] = result["scored_invariants"]
            elif "invariants" in result:
                raw = result.get("invariants", {})
                if isinstance(raw, dict):
                    inv_data["invariants"] = raw.get("invariants", [])
                else:
                    inv_data["invariants"] = raw
            registry = update_registry(registry, inv_data, run_id)

    classification = classify_invariants(registry)

    return {
        "comparison": comparison,
        "classification": classification,
        "registry": registry,
    }


# ---------------------------------------------------------------------------
# 4. Formatter
# ---------------------------------------------------------------------------


def format_meta_diagnostics(result: dict) -> str:
    """Format meta-diagnostics results as human-readable text.

    Parameters
    ----------
    result : dict
        Output of ``run_meta_diagnostics_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []

    lines.append("")
    lines.append("=== Meta-Diagnostics (Cross-Run) ===")

    comparison = result.get("comparison", {})
    classification = result.get("classification", {})

    n_runs = comparison.get("run_count", 0)
    lines.append(f"")
    lines.append(f"Runs Compared: {n_runs}")

    # Overlap.
    overlap = comparison.get("invariant_overlap", [])
    lines.append(f"Shared Invariants (all runs): {len(overlap)}")
    for key in overlap:
        lines.append(f"  - {key}")

    # Divergence.
    divergence = comparison.get("invariant_divergence", [])
    lines.append(f"Unique Invariants (single run): {len(divergence)}")
    for key in divergence[:5]:
        lines.append(f"  - {key}")

    # Shared diagnoses.
    shared_diag = comparison.get("shared_diagnoses", [])
    if shared_diag:
        lines.append("")
        lines.append("Diagnosis Distribution:")
        for entry in shared_diag:
            lines.append(
                f"  {entry['diagnosis']}: {entry['count']} runs "
                f"({entry['ratio']:.0%})"
            )

    # Shared topologies.
    shared_topo = comparison.get("shared_topologies", [])
    if shared_topo:
        lines.append("")
        lines.append("Topology Distribution:")
        for entry in shared_topo:
            lines.append(
                f"  {entry['topology']}: {entry['count']} runs "
                f"({entry['ratio']:.0%})"
            )

    # Classification.
    universal = classification.get("universal", [])
    contextual = classification.get("contextual", [])
    rare = classification.get("rare", [])

    lines.append("")
    lines.append(f"Universal Invariants: {len(universal)}")
    for item in universal:
        lines.append(f"  - {item['invariant']} (avg={item['avg_strength']:.2f})")

    lines.append(f"Contextual Invariants: {len(contextual)}")
    for item in contextual:
        lines.append(f"  - {item['invariant']} (avg={item['avg_strength']:.2f})")

    lines.append(f"Rare Invariants: {len(rare)}")
    for item in rare:
        lines.append(f"  - {item['invariant']} (avg={item['avg_strength']:.2f})")

    return "\n".join(lines)


__all__ = [
    "ROUND_PRECISION",
    "classify_invariants",
    "compare_runs",
    "format_meta_diagnostics",
    "run_meta_diagnostics_analysis",
]
