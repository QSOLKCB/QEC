"""v102.0.0 — Deterministic strategy clustering by correlation.

Groups strategies where pairwise correlation meets or exceeds a
threshold.  Uses the existing ``compute_strategy_correlation`` function.
Clustering is deterministic: strategies are processed in sorted name
order and the first element of each group becomes the representative.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- pure mathematical grouping

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List

from qec.analysis.strategy_correlation import compute_strategy_correlation


def cluster_strategies(
    strategies: List[Dict[str, Any]],
    threshold: float = 0.9,
) -> Dict[str, Any]:
    """Cluster strategies by pairwise correlation.

    Groups strategies where correlation >= threshold.  Processing order
    is deterministic (sorted by name).  The first element assigned to
    each cluster becomes its representative.

    Parameters
    ----------
    strategies : list of dict
        Strategy dicts with a ``"metrics"`` sub-dict.
    threshold : float
        Correlation threshold for cluster membership (default 0.9).

    Returns
    -------
    dict
        Contains ``"clusters"`` list, each with ``"representative"``,
        ``"members"``, and ``"size"`` keys.  Clusters are sorted by
        representative name.
    """
    if not strategies:
        return {"clusters": []}

    # Sort by name for deterministic processing
    sorted_strats = sorted(strategies, key=lambda s: s.get("name", ""))

    n = len(sorted_strats)
    assigned = [False] * n
    clusters: List[Dict[str, Any]] = []

    for i in range(n):
        if assigned[i]:
            continue

        # Start new cluster with strategy i as representative
        assigned[i] = True
        rep_name = sorted_strats[i].get("name", "")
        members = [rep_name]

        for j in range(i + 1, n):
            if assigned[j]:
                continue

            corr = compute_strategy_correlation(sorted_strats[i], sorted_strats[j])
            if corr >= threshold:
                assigned[j] = True
                members.append(sorted_strats[j].get("name", ""))

        # Members already in sorted order (i < j guarantees name order)
        clusters.append({
            "representative": rep_name,
            "members": members,
            "size": len(members),
        })

    # Clusters sorted by representative name (already in order due to
    # processing sorted_strats sequentially)
    return {"clusters": clusters}


__all__ = [
    "cluster_strategies",
]
