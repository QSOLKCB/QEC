"""Basin-aware trajectory clustering.

Clusters trajectories by their basin visitation patterns using
deterministic L1 histogram distance and greedy threshold assignment.
Pure post-processing — no randomness, no mutation of inputs.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

# Default clustering threshold (L1 histogram distance).
DEFAULT_THRESHOLD = 0.3


# -- B1: extract basin sequence -------------------------------------------


def extract_basin_sequence(
    trajectory_states: List[Tuple[int, ...]],
    basin_mapping: Dict[Tuple[int, ...], int],
) -> List[int]:
    """Map each trajectory state to its basin ID."""
    return [basin_mapping[s] for s in trajectory_states]


# -- B2: basin histogram --------------------------------------------------


def basin_histogram(sequence: List[int]) -> Dict[int, float]:
    """Normalised basin visitation histogram from a basin sequence."""
    counts: Dict[int, int] = {}
    for b in sequence:
        counts[b] = counts.get(b, 0) + 1
    total = float(len(sequence))
    return {k: v / total for k, v in counts.items()}


# -- B3: deterministic distance -------------------------------------------


def histogram_distance(h1: Dict[int, float], h2: Dict[int, float]) -> float:
    """L1 distance over the union of keys (deterministic key order)."""
    keys = sorted(set(h1) | set(h2))
    return sum(abs(h1.get(k, 0.0) - h2.get(k, 0.0)) for k in keys)


# -- B4: pairwise distance matrix -----------------------------------------


def compute_distance_matrix(
    histograms: Dict[int, Dict[int, float]],
) -> Tuple[np.ndarray, List[int]]:
    """Return (N×N distance matrix, sorted trajectory IDs).

    The matrix is symmetric with zero diagonal.  Row/column order
    follows ``sorted(histograms.keys())``.
    """
    ids = sorted(histograms.keys())
    n = len(ids)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = histogram_distance(histograms[ids[i]], histograms[ids[j]])
            matrix[i, j] = d
            matrix[j, i] = d
    return matrix, ids


# -- B5: greedy threshold clustering --------------------------------------


def cluster_trajectories(
    distance_matrix: np.ndarray,
    threshold: float,
) -> List[List[int]]:
    """Greedy deterministic clustering by distance threshold.

    Iterates trajectories in index order.  Each trajectory is assigned to
    the first existing cluster whose representative (first member) is
    within *threshold*; otherwise a new cluster is created.

    Returns a list of clusters, each a list of trajectory indices into
    the distance matrix.
    """
    n = distance_matrix.shape[0]
    clusters: List[List[int]] = []
    representatives: List[int] = []

    for i in range(n):
        assigned = False
        for ci, rep in enumerate(representatives):
            if distance_matrix[i, rep] <= threshold:
                clusters[ci].append(i)
                assigned = True
                break
        if not assigned:
            representatives.append(i)
            clusters.append([i])

    return clusters


# -- B6: pipeline entry point ---------------------------------------------


def run_trajectory_clustering(
    trajectories: Dict[int, List[Tuple[int, ...]]],
    basin_mapping: Dict[Tuple[int, ...], int],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Cluster trajectories by basin visitation similarity.

    Parameters
    ----------
    trajectories:
        ``{trajectory_id: [state, state, ...]}`` — ordered state sequences.
    basin_mapping:
        ``{state: basin_id}`` from basin analysis.
    threshold:
        Maximum L1 histogram distance for same-cluster membership.

    Returns
    -------
    dict with ``clusters`` list and ``n_clusters`` count.
    """
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")

    # N = 0: no trajectories.
    if not trajectories:
        return {"clusters": [], "n_clusters": 0}

    # Build per-trajectory histograms.
    histograms: Dict[int, Dict[int, float]] = {}
    for tid in sorted(trajectories.keys()):
        seq = extract_basin_sequence(trajectories[tid], basin_mapping)
        histograms[tid] = basin_histogram(seq)

    # N = 1: single trajectory, skip distance computation.
    if len(histograms) == 1:
        tid = next(iter(histograms))
        return {
            "clusters": [{"id": 0, "members": [tid], "centroid_histogram": histograms[tid]}],
            "n_clusters": 1,
        }

    dist_matrix, ordered_ids = compute_distance_matrix(histograms)
    raw_clusters = cluster_trajectories(dist_matrix, threshold)

    # Build output with centroid histograms.
    result_clusters: List[Dict[str, Any]] = []
    for members_idx in raw_clusters:
        member_ids = sorted(ordered_ids[idx] for idx in members_idx)
        # Centroid = element-wise mean of member histograms.
        all_keys: set = set()
        for mid in member_ids:
            all_keys |= set(histograms[mid].keys())
        sorted_keys = sorted(all_keys)
        centroid: Dict[int, float] = {}
        for k in sorted_keys:
            centroid[k] = sum(histograms[mid].get(k, 0.0) for mid in member_ids) / len(member_ids)
        result_clusters.append({
            "members": member_ids,
            "centroid_histogram": centroid,
        })

    # Canonical cluster ordering: sort by (min_member, cluster_size).
    result_clusters.sort(key=lambda c: (min(c["members"]), len(c["members"])))
    for ci, cluster in enumerate(result_clusters):
        cluster["id"] = ci

    return {
        "clusters": result_clusters,
        "n_clusters": len(result_clusters),
    }
