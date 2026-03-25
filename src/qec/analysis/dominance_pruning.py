"""v101.9.0 — Structure-aware dominance pruning (Pareto frontier).

Removes dominated strategies from a candidate set, retaining only
Pareto-optimal (non-dominated) strategies.

A strategy A dominates B if A is at least as good as B on all metric
keys and strictly better on at least one.

v101.9.0 adds structure-aware refinement: dominance additionally
requires that A's consistency_gap <= B's consistency_gap, and when
both strategies have revival data, prefers higher revival_strength.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical filtering (no heuristics)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Metric keys — fixed order, no dict iteration
# ---------------------------------------------------------------------------

DOMINANCE_KEYS: List[str] = [
    "design_score",
    "confidence_efficiency",
    "temporal_stability",
    "trust_modulation",
]


# ---------------------------------------------------------------------------
# Dominance check
# ---------------------------------------------------------------------------


def dominates(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    structure_aware: bool = False,
) -> bool:
    """Return True if strategy *a* dominates strategy *b*.

    A dominates B iff:
    - for every key in DOMINANCE_KEYS: a_metrics[key] >= b_metrics[key]
    - for at least one key: a_metrics[key] > b_metrics[key]

    When *structure_aware* is True, two additional conditions must hold:
    - a's consistency_gap <= b's consistency_gap
    - if both have revival data, a's revival_strength >= b's revival_strength

    Missing keys default to 0.0.

    Parameters
    ----------
    a, b : dict
        Strategy dicts with a ``"metrics"`` sub-dict.
    structure_aware : bool
        If True, apply consistency gap and revival constraints.

    Returns
    -------
    bool
    """
    a_metrics = a.get("metrics", {})
    b_metrics = b.get("metrics", {})

    all_geq = True
    any_gt = False

    for key in DOMINANCE_KEYS:
        a_val = float(a_metrics.get(key, 0.0))
        b_val = float(b_metrics.get(key, 0.0))

        if a_val < b_val:
            all_geq = False
            break
        if a_val > b_val:
            any_gt = True

    if not (all_geq and any_gt):
        return False

    if structure_aware:
        # Consistency gap constraint: a must be at least as consistent
        a_gap = float(a_metrics.get("consistency_gap", 0.0))
        b_gap = float(b_metrics.get("consistency_gap", 0.0))
        if a_gap > b_gap:
            return False

        # Revival constraint: if both have revival, prefer higher strength
        a_has = a_metrics.get("has_revival", False)
        b_has = b_metrics.get("has_revival", False)
        if a_has and b_has:
            a_strength = float(a_metrics.get("revival_strength", 0.0))
            b_strength = float(b_metrics.get("revival_strength", 0.0))
            if a_strength < b_strength:
                return False

    return True


# ---------------------------------------------------------------------------
# Pareto pruning
# ---------------------------------------------------------------------------


def pareto_prune(
    strategies: List[Dict[str, Any]],
    *,
    structure_aware: bool = False,
) -> List[Dict[str, Any]]:
    """Return only non-dominated (Pareto-optimal) strategies.

    Uses O(n²) pairwise comparison — acceptable for ≤54 strategies.
    Does not mutate inputs.

    When *structure_aware* is True, dominance checks additionally
    require consistency_gap and revival constraints.

    Parameters
    ----------
    strategies : list of dict
        Candidate strategies, each with ``"name"`` and ``"metrics"``.
    structure_aware : bool
        If True, use structure-aware dominance conditions.

    Returns
    -------
    list of dict
        Non-dominated strategies, sorted by name (deterministic).
    """
    if not strategies:
        return []

    n = len(strategies)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if dominates(strategies[j], strategies[i],
                         structure_aware=structure_aware):
                dominated[i] = True
                break

    result = [strategies[i] for i in range(n) if not dominated[i]]

    # Deterministic ordering by name
    result.sort(key=lambda s: s.get("name", ""))

    return result


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def pruning_stats(
    original: List[Dict[str, Any]],
    pruned: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute simple statistics about dominance pruning.

    Parameters
    ----------
    original : list of dict
        Full candidate set before pruning.
    pruned : list of dict
        Non-dominated set after pruning.

    Returns
    -------
    dict
        Contains ``pruned_count``, ``retained_count``, ``dominance_ratio``.
    """
    n_original = len(original)
    n_retained = len(pruned)
    n_pruned = n_original - n_retained

    dominance_ratio = round(n_pruned / n_original, 12) if n_original > 0 else 0.0

    return {
        "pruned_count": n_pruned,
        "retained_count": n_retained,
        "dominance_ratio": dominance_ratio,
    }


__all__ = [
    "DOMINANCE_KEYS",
    "dominates",
    "pareto_prune",
    "pruning_stats",
]
