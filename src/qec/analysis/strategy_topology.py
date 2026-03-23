"""Strategy topology layer.

Understands relationships between strategies:
- similarity / distance
- clustering
- dominant strategy identification

All functions are pure, deterministic, and never mutate inputs.
Dependencies: numpy only.
"""

from typing import Any, Dict, List, Sequence, Tuple


def strategy_signature(strategy: Any) -> Tuple[str, Tuple[Tuple[str, Any], ...]]:
    """Compute a deterministic, hashable signature for a strategy.

    Parameters
    ----------
    strategy : object
        Must have ``action_type`` (str) and ``params`` (dict) attributes.

    Returns
    -------
    tuple
        (action_type, sorted_params_tuple)
    """
    action_type = str(strategy.action_type)
    params = sorted(strategy.params.items())
    return (action_type, tuple(params))


def strategy_distance(s1: Any, s2: Any) -> float:
    """Compute distance between two strategies.

    Distance is in [0, 1]:
    - 0.0 = identical
    - 1.0 = completely different

    Components:
    - action_type match (0 or 0.5)
    - parameter difference (0 to 0.5)

    Parameters
    ----------
    s1, s2 : object
        Must have ``action_type`` (str) and ``params`` (dict) attributes.

    Returns
    -------
    float
        Distance in [0, 1].
    """
    # Action type component
    if s1.action_type != s2.action_type:
        type_dist = 0.5
    else:
        type_dist = 0.0

    # Parameter component
    all_keys = sorted(set(list(s1.params.keys()) + list(s2.params.keys())))
    if not all_keys:
        return type_dist

    diffs = 0.0
    for key in all_keys:
        v1 = s1.params.get(key)
        v2 = s2.params.get(key)
        if v1 is None or v2 is None:
            diffs += 1.0
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = abs(float(v1) - float(v2))
            diffs += diff / (diff + 1.0)
        elif v1 != v2:
            diffs += 1.0
        # else equal -> 0

    param_dist = (diffs / len(all_keys)) * 0.5
    return min(1.0, type_dist + param_dist)


def build_topology(
    strategies: Dict[str, Any],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise distance matrix for strategies.

    Parameters
    ----------
    strategies : dict
        {strategy_id: strategy_object}

    Returns
    -------
    dict
        {(id_i, id_j): distance} for all ordered pairs where i < j.
    """
    ids = sorted(strategies.keys())
    topology: Dict[Tuple[str, str], float] = {}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = strategy_distance(strategies[ids[i]], strategies[ids[j]])
            topology[(ids[i], ids[j])] = d
    return topology


def cluster_strategies(
    strategies: Dict[str, Any],
    threshold: float = 0.3,
) -> List[List[str]]:
    """Group strategies into clusters by distance threshold.

    Uses single-linkage: two strategies are in the same cluster if
    their distance < threshold. Deterministic via sorted ordering.

    Parameters
    ----------
    strategies : dict
        {strategy_id: strategy_object}
    threshold : float
        Maximum distance to be in the same cluster.

    Returns
    -------
    list of list of str
        Clusters of strategy IDs, sorted deterministically.
    """
    ids = sorted(strategies.keys())
    if not ids:
        return []

    # Union-find with path compression
    parent: Dict[str, str] = {sid: sid for sid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: smaller id becomes root
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    # Merge close strategies
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = strategy_distance(strategies[ids[i]], strategies[ids[j]])
            if d < threshold:
                union(ids[i], ids[j])

    # Collect clusters
    clusters_map: Dict[str, List[str]] = {}
    for sid in ids:
        root = find(sid)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(sid)

    # Sort deterministically
    result = [sorted(c) for c in clusters_map.values()]
    result.sort(key=lambda c: c[0])
    return result


def find_dominant_strategy(
    strategies: Dict[str, Any],
    topology: Dict[Tuple[str, str], float],
) -> str:
    """Find the strategy with lowest average distance to all others.

    Parameters
    ----------
    strategies : dict
        {strategy_id: strategy_object}
    topology : dict
        Pairwise distances from build_topology().

    Returns
    -------
    str
        ID of the dominant strategy. Empty string if no strategies.
    """
    ids = sorted(strategies.keys())
    if not ids:
        return ""
    if len(ids) == 1:
        return ids[0]

    best_id = ids[0]
    best_avg = float("inf")

    for sid in ids:
        total = 0.0
        count = 0
        for other in ids:
            if other == sid:
                continue
            key = (min(sid, other), max(sid, other))
            total += topology.get(key, 1.0)
            count += 1
        avg = total / count if count > 0 else 0.0
        if avg < best_avg or (avg == best_avg and sid < best_id):
            best_avg = avg
            best_id = sid

    return best_id


def compute_strategy_topology(
    strategies: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute full strategy topology summary.

    Parameters
    ----------
    strategies : dict
        {strategy_id: strategy_object}

    Returns
    -------
    dict
        Keys: "topology", "clusters", "dominant".
    """
    if not strategies:
        return {"topology": {}, "clusters": [], "dominant": ""}

    topology = build_topology(strategies)
    clusters = cluster_strategies(strategies)
    dominant = find_dominant_strategy(strategies, topology)

    # Serialize topology keys for JSON compatibility
    topo_serialized = {
        f"{k[0]}|{k[1]}": v for k, v in sorted(topology.items())
    }

    return {
        "topology": topo_serialized,
        "clusters": clusters,
        "dominant": dominant,
    }
