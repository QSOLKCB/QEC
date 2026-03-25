"""v102.0.0 — Representation comparison (ternary vs quaternary).

Splits strategies by state_system and computes summary statistics
for each representation.  Enables side-by-side comparison of
ternary and quaternary strategy populations.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical computation

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def compare_representations(strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare ternary vs quaternary strategy populations.

    Splits strategies by ``state_system`` field, computes count,
    average design_score, and best strategy per system.

    Parameters
    ----------
    strategies : list of dict
        Strategy dicts with ``"state_system"`` and ``"metrics"`` keys.

    Returns
    -------
    dict
        Contains ``"ternary"`` and ``"quaternary"`` sub-dicts, each with
        ``"count"``, ``"avg_design_score"``, and ``"best"`` keys.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {
        "ternary": [],
        "quaternary": [],
    }

    for s in strategies:
        system = s.get("state_system", "ternary")
        if system in groups:
            groups[system].append(s)

    result = {}
    for system_name in ("ternary", "quaternary"):
        group = groups[system_name]

        if not group:
            result[system_name] = {
                "count": 0,
                "avg_design_score": 0.0,
                "best": None,
            }
            continue

        # Sort deterministically by name for stable best selection
        sorted_group = sorted(group, key=lambda s: s.get("name", ""))

        scores = []
        best_strategy = None
        best_score = -1.0

        for s in sorted_group:
            ds = float(s.get("metrics", {}).get("design_score", 0.0))
            scores.append(ds)
            if ds > best_score or (ds == best_score and best_strategy is None):
                best_score = ds
                best_strategy = s

        avg_ds = round(sum(scores) / len(scores), 12) if scores else 0.0

        result[system_name] = {
            "count": len(group),
            "avg_design_score": avg_ds,
            "best": best_strategy.get("name", "") if best_strategy else None,
        }

    return result


__all__ = [
    "compare_representations",
]
