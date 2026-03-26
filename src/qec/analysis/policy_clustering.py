"""v103.8.0 — Policy clustering and archetype extraction.

Provides:
- deterministic policy distance metric
- agglomerative clustering of similar policies
- archetype construction from clusters
- archetype extraction from policy memory
- archetype ranking by historical performance

Groups similar policies into clusters, builds representative
archetypes, and compresses policy memory into reusable strategy
templates.

Pipeline::

    policy_memory -> policies -> clustering -> archetypes -> ranking

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- rule-based only (no stochastic clustering, no ML)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CLUSTER_THRESHOLD = 0.2
MODE_DIFFERENCE_PENALTY = 0.5
PRIORITY_DIFFERENCE_PENALTY = 0.5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Public API — Policy Distance Metric
# ---------------------------------------------------------------------------


def compute_policy_distance(p1: Any, p2: Any) -> float:
    """Compute deterministic distance between two policies.

    Distance components:

    - **Threshold distance**: sum of absolute differences across all
      threshold keys present in either policy.
    - **Mode penalty**: +0.5 if modes differ.
    - **Priority penalty**: +0.5 if priorities differ.

    Parameters
    ----------
    p1 : Policy
        First policy.  Must have ``thresholds``, ``mode``, and
        ``priority`` attributes.
    p2 : Policy
        Second policy.

    Returns
    -------
    float
        Non-negative distance value.
    """
    # Threshold distance: sum of absolute differences.
    all_keys = sorted(set(list(p1.thresholds.keys()) + list(p2.thresholds.keys())))
    threshold_dist = 0.0
    for key in all_keys:
        v1 = float(p1.thresholds.get(key, 0.0))
        v2 = float(p2.thresholds.get(key, 0.0))
        threshold_dist += abs(v1 - v2)

    # Mode difference penalty.
    mode_penalty = MODE_DIFFERENCE_PENALTY if p1.mode != p2.mode else 0.0

    # Priority difference penalty.
    priority_penalty = (
        PRIORITY_DIFFERENCE_PENALTY if p1.priority != p2.priority else 0.0
    )

    return _round(threshold_dist + mode_penalty + priority_penalty)


# ---------------------------------------------------------------------------
# Public API — Deterministic Agglomerative Clustering
# ---------------------------------------------------------------------------


def cluster_policies(
    policies: List[Any],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> List[List[Any]]:
    """Cluster policies using deterministic agglomerative clustering.

    Algorithm:
    1. Start with each policy as its own cluster.
    2. Find the pair of clusters with minimum distance.
    3. If that distance is below *threshold*, merge them.
    4. Repeat until no more merges are possible.

    Cluster distance is single-linkage (minimum distance between
    any two members of distinct clusters).

    Ties are broken by comparing cluster indices (lower index first),
    ensuring deterministic merging order.

    Parameters
    ----------
    policies : list of Policy
        Policies to cluster.  Order is preserved within clusters.
    threshold : float
        Maximum distance for merging clusters.

    Returns
    -------
    list of list of Policy
        Clusters sorted by size (descending), then by the name of
        the first member (ascending) for deterministic ordering.
    """
    if not policies:
        return []

    if len(policies) == 1:
        return [[policies[0]]]

    # Sort policies by name for deterministic input ordering.
    sorted_policies = sorted(policies, key=lambda p: p.name)

    # Initialize: each policy is its own cluster.
    clusters: List[List[Any]] = [[p] for p in sorted_policies]

    # Precompute pairwise distances.
    n = len(sorted_policies)
    dist: Dict[tuple, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_policy_distance(sorted_policies[i], sorted_policies[j])
            dist[(i, j)] = d

    # Track which original indices belong to which cluster.
    cluster_members: List[List[int]] = [[i] for i in range(n)]

    while len(cluster_members) > 1:
        # Find minimum distance pair (single-linkage).
        best_dist = float("inf")
        best_pair = (-1, -1)

        for ci in range(len(cluster_members)):
            for cj in range(ci + 1, len(cluster_members)):
                # Single-linkage: minimum distance between members.
                min_d = float("inf")
                for mi in cluster_members[ci]:
                    for mj in cluster_members[cj]:
                        key = (min(mi, mj), max(mi, mj))
                        d = dist[key]
                        if d < min_d:
                            min_d = d

                if min_d < best_dist or (
                    min_d == best_dist and (ci, cj) < best_pair
                ):
                    best_dist = min_d
                    best_pair = (ci, cj)

        # Stop if best distance exceeds threshold.
        if best_dist >= threshold:
            break

        # Merge clusters.
        ci, cj = best_pair
        merged = cluster_members[ci] + cluster_members[cj]
        # Remove in reverse order to preserve indices.
        new_cluster_members = []
        for idx in range(len(cluster_members)):
            if idx == ci:
                new_cluster_members.append(merged)
            elif idx != cj:
                new_cluster_members.append(cluster_members[idx])
        cluster_members = new_cluster_members

    # Build result clusters from member indices.
    result = []
    for members in cluster_members:
        cluster = [sorted_policies[i] for i in sorted(members)]
        result.append(cluster)

    # Sort clusters: by size descending, then first member name ascending.
    result.sort(key=lambda c: (-len(c), c[0].name))

    return result


# ---------------------------------------------------------------------------
# Public API — Archetype Construction
# ---------------------------------------------------------------------------


def build_archetype(cluster: List[Any], archetype_id: int = 0) -> Any:
    """Build a representative archetype policy from a cluster.

    Rules:
    - **Thresholds**: mean of all member thresholds.
    - **Mode**: majority vote (ties broken alphabetically).
    - **Priority**: majority vote (ties broken alphabetically).
    - **Name**: ``"archetype_{archetype_id}"``.

    Parameters
    ----------
    cluster : list of Policy
        Non-empty list of policies in the cluster.
    archetype_id : int
        Numeric identifier for the archetype name.

    Returns
    -------
    Policy
        Representative archetype policy.

    Raises
    ------
    ValueError
        If *cluster* is empty.
    """
    from qec.analysis.policy import Policy

    if not cluster:
        raise ValueError("Cannot build archetype from empty cluster")

    if len(cluster) == 1:
        p = cluster[0]
        return Policy(
            name=f"archetype_{archetype_id}",
            mode=p.mode,
            priority=p.priority,
            thresholds=dict(p.thresholds),
        )

    # Thresholds: mean across all members.
    all_keys = sorted(set(k for p in cluster for k in p.thresholds))
    thresholds: Dict[str, float] = {}
    for key in all_keys:
        values = [float(p.thresholds[key]) for p in cluster if key in p.thresholds]
        thresholds[key] = _round(sum(values) / len(values))

    # Mode: majority vote, ties broken alphabetically.
    mode = _majority_vote([p.mode for p in cluster])

    # Priority: majority vote, ties broken alphabetically.
    priority = _majority_vote([p.priority for p in cluster])

    return Policy(
        name=f"archetype_{archetype_id}",
        mode=mode,
        priority=priority,
        thresholds=thresholds,
    )


def _majority_vote(values: List[str]) -> str:
    """Deterministic majority vote with alphabetical tiebreak."""
    counts: Dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    max_count = max(counts.values())
    candidates = sorted(v for v, c in counts.items() if c == max_count)
    return candidates[0]


# ---------------------------------------------------------------------------
# Public API — Archetype Extraction from Memory
# ---------------------------------------------------------------------------


def extract_policy_archetypes(
    memory: Dict[str, Any],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> List[Any]:
    """Extract archetype policies from policy memory.

    Pipeline: memory -> policies -> clustering -> archetypes.

    Parameters
    ----------
    memory : dict
        Policy memory (from ``init_policy_memory`` or prior calls).
    threshold : float
        Clustering distance threshold.

    Returns
    -------
    list of Policy
        Archetype policies extracted from memory clusters.
    """
    from qec.analysis.policy import Policy

    policies_data = memory.get("policies", {})
    if not policies_data:
        return []

    # Reconstruct Policy objects from memory.
    policies = []
    for name in sorted(policies_data.keys()):
        entry = policies_data[name]
        p = Policy.from_dict(name, entry["policy_dict"])
        policies.append(p)

    if not policies:
        return []

    # Cluster and build archetypes.
    clusters = cluster_policies(policies, threshold=threshold)
    archetypes = []
    for idx, cluster in enumerate(clusters):
        archetype = build_archetype(cluster, archetype_id=idx)
        archetypes.append(archetype)

    return archetypes


# ---------------------------------------------------------------------------
# Public API — Archetype Ranking
# ---------------------------------------------------------------------------


def rank_archetypes(
    archetypes: List[Any],
    memory: Dict[str, Any],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> List[Any]:
    """Rank archetypes by average score of their cluster members.

    For each archetype, finds the cluster of memory policies it
    represents and computes the mean of their ``avg_score`` values.

    Parameters
    ----------
    archetypes : list of Policy
        Archetype policies (from ``extract_policy_archetypes``).
    memory : dict
        Policy memory with score history.
    threshold : float
        Clustering distance threshold (must match the threshold
        used to create the archetypes).

    Returns
    -------
    list of Policy
        Archetypes sorted by descending cluster score, with
        ties broken by archetype name (ascending).
    """
    from qec.analysis.policy import Policy

    if not archetypes:
        return []

    policies_data = memory.get("policies", {})
    if not policies_data:
        return list(archetypes)

    # Reconstruct policies and cluster them.
    policies = []
    for name in sorted(policies_data.keys()):
        entry = policies_data[name]
        policies.append(Policy.from_dict(name, entry["policy_dict"]))

    clusters = cluster_policies(policies, threshold=threshold)

    # Compute average score per cluster.
    cluster_scores: List[float] = []
    for cluster in clusters:
        member_scores = []
        for p in cluster:
            entry = policies_data.get(p.name, {})
            avg = float(entry.get("avg_score", 0.0))
            member_scores.append(avg)
        if member_scores:
            cluster_scores.append(
                _round(sum(member_scores) / len(member_scores))
            )
        else:
            cluster_scores.append(0.0)

    # Pair archetypes with cluster scores.
    # archetypes[i] corresponds to clusters[i].
    n = min(len(archetypes), len(cluster_scores))
    scored = []
    for i in range(n):
        scored.append((archetypes[i], cluster_scores[i]))

    # Add any extra archetypes with score 0.
    for i in range(n, len(archetypes)):
        scored.append((archetypes[i], 0.0))

    # Sort: descending score, ascending name for ties.
    scored.sort(key=lambda x: (-x[1], x[0].name))

    return [a for a, _ in scored]


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_policy_clusters_summary(
    clusters: List[List[Any]],
    archetypes: Optional[List[Any]] = None,
) -> str:
    """Format clustering results as a human-readable summary.

    Parameters
    ----------
    clusters : list of list of Policy
        Policy clusters from ``cluster_policies``.
    archetypes : list of Policy, optional
        Archetype policies from ``build_archetype``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Policy Clusters ===")
    lines.append("")

    if not clusters:
        lines.append("No clusters found.")
        return "\n".join(lines)

    for i, cluster in enumerate(clusters):
        lines.append(f"Cluster {i + 1}:")
        for p in cluster:
            lines.append(f"  {p.name}")
        lines.append("")

    if archetypes:
        lines.append("=== Archetypes ===")
        lines.append("")
        for archetype in archetypes:
            lines.append(f"{archetype.name}:")
            for key in sorted(archetype.thresholds.keys()):
                val = archetype.thresholds[key]
                lines.append(f"  {key}={val:.4f}")
            lines.append(f"  mode={archetype.mode}")
            lines.append(f"  priority={archetype.priority}")
            lines.append("")

    return "\n".join(lines)


__all__ = [
    "DEFAULT_CLUSTER_THRESHOLD",
    "MODE_DIFFERENCE_PENALTY",
    "PRIORITY_DIFFERENCE_PENALTY",
    "ROUND_PRECISION",
    "build_archetype",
    "cluster_policies",
    "compute_policy_distance",
    "extract_policy_archetypes",
    "format_policy_clusters_summary",
    "rank_archetypes",
]
