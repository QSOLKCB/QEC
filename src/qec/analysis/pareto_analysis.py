"""v102.0.0 — Pareto front extraction for strategy analysis.

Extracts the non-dominated (Pareto-optimal) strategies using the
existing dominance logic.  This module provides an analysis-oriented
interface that delegates to the core dominance_pruning module.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical filtering

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List

from qec.analysis.dominance_pruning import pareto_prune


def compute_pareto_front(
    strategies: List[Dict[str, Any]],
    *,
    structure_aware: bool = False,
) -> List[Dict[str, Any]]:
    """Extract the Pareto front from a set of strategies.

    Uses existing dominance logic from ``dominance_pruning.pareto_prune``.
    Returns only non-dominated strategies, sorted deterministically by
    descending design_score, then by name (ascending).

    Parameters
    ----------
    strategies : list of dict
        Strategy dicts with a ``"metrics"`` sub-dict.
    structure_aware : bool
        If True, apply structure-aware dominance conditions.

    Returns
    -------
    list of dict
        Non-dominated strategies sorted by descending design_score,
        then ascending name.
    """
    if not strategies:
        return []

    front = pareto_prune(strategies, structure_aware=structure_aware)

    # Sort by descending design_score, then ascending name
    front.sort(key=lambda s: (
        -float(s.get("metrics", {}).get("design_score", 0.0)),
        s.get("name", ""),
    ))

    return front


__all__ = [
    "compute_pareto_front",
]
