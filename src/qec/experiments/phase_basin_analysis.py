"""Attractor basin detection via strongly connected components.

Partitions the state-transition graph into basins of attraction using
Tarjan's SCC algorithm.  Pure post-processing over ``state_graph`` and
``attractor_field`` structures produced by earlier pipeline stages.
"""

from typing import Any, Dict, List, Tuple


# -- Step 1: adjacency ---------------------------------------------------


def build_adjacency(
    state_graph: Dict[str, Any],
) -> Dict[Tuple[int, ...], List[Tuple[int, ...]]]:
    """Return deterministic adjacency list from *state_graph*.

    Nodes and neighbour lists are sorted for reproducibility.
    """
    adj: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
    for node in sorted(state_graph["nodes"]):
        adj[node] = []
    for edge in state_graph["edges"]:
        src = edge["from"]
        dst = edge["to"]
        if src not in adj:
            adj[src] = []
        adj[src].append(dst)
    # Sort neighbours for determinism.
    for node in adj:
        adj[node] = sorted(adj[node])
    return adj


# -- Step 2: Tarjan SCC --------------------------------------------------


def compute_scc(
    adjacency: Dict[Tuple[int, ...], List[Tuple[int, ...]]],
) -> Dict[str, Any]:
    """Deterministic Tarjan SCC over *adjacency*.

    Nodes are processed in sorted order.  Returns components sorted by
    their smallest node so that output is fully deterministic.
    """
    index_counter = [0]
    stack: List[Tuple[int, ...]] = []
    on_stack: Dict[Tuple[int, ...], bool] = {}
    index_map: Dict[Tuple[int, ...], int] = {}
    lowlink: Dict[Tuple[int, ...], int] = {}
    components: List[List[Tuple[int, ...]]] = []

    def strongconnect(v: Tuple[int, ...]) -> None:
        index_map[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in adjacency.get(v, []):
            if w not in index_map:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif on_stack.get(w, False):
                lowlink[v] = min(lowlink[v], index_map[w])

        if lowlink[v] == index_map[v]:
            component: List[Tuple[int, ...]] = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            components.append(sorted(component))

    for node in sorted(adjacency.keys()):
        if node not in index_map:
            strongconnect(node)

    # Sort components by (canonical_node, size) for deterministic IDs.
    components.sort(key=lambda c: (min(c), len(c)))

    return {
        "components": [
            {"id": i, "nodes": c} for i, c in enumerate(components)
        ],
        "n_components": len(components),
    }


# -- Step 3: basin metrics -----------------------------------------------


def compute_basin_metrics(
    components: List[Dict[str, Any]],
    state_graph: Dict[str, Any],
    attractor_field: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute size, mass, and coherence for each component."""
    # Build score lookup from attractor_field.
    score_map: Dict[Tuple[int, ...], float] = {}
    for node_info in attractor_field.get("nodes", []):
        score_map[node_info["state"]] = node_info["score"]

    # Index edges by source for quick lookup.
    edge_index: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
    for edge in state_graph["edges"]:
        src = edge["from"]
        if src not in edge_index:
            edge_index[src] = []
        edge_index[src].append(edge)

    basins: List[Dict[str, Any]] = []
    for comp in components:
        cid = comp["id"]
        nodes = comp["nodes"]
        node_set = set(map(tuple, nodes))
        size = len(nodes)
        mass = float(sum(score_map.get(tuple(n), 0.0) for n in nodes))

        internal = 0
        outgoing = 0
        for n in nodes:
            for edge in edge_index.get(tuple(n), []):
                if tuple(edge["to"]) in node_set:
                    internal += 1
                else:
                    outgoing += 1

        total = internal + outgoing
        coherence = float(internal / total) if total > 0 else 0.0

        basins.append({
            "id": cid,
            "nodes": nodes,
            "size": size,
            "mass": mass,
            "coherence": coherence,
            "internal_edges": internal,
            "outgoing_edges": outgoing,
        })

    return basins


# -- Step 4: classification ----------------------------------------------


def has_self_loop(
    node: Tuple[int, ...],
    state_graph: Dict[str, Any],
) -> bool:
    """Return True if *node* has an explicit self-loop in *state_graph*."""
    return any(
        e["from"] == node and e["to"] == node
        for e in state_graph["edges"]
    )


def classify_basins(
    basins: List[Dict[str, Any]],
    state_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Classify each basin by structure.

    Rules (applied in order):
    * size == 1 **and** self-loop present → ``"fixed_point"``
    * size == 2 → ``"oscillatory"``
    * coherence >= 0.8 → ``"stable_basin"``
    * otherwise → ``"transient_basin"``
    """
    classifications: List[Dict[str, Any]] = []
    for basin in basins:
        size = basin["size"]
        coherence = basin["coherence"]
        nodes = basin["nodes"]

        if size == 1 and has_self_loop(tuple(nodes[0]), state_graph):
            btype = "fixed_point"
            confidence = 1.0
        elif size == 2:
            btype = "oscillatory"
            confidence = float(coherence)
        elif coherence >= 0.8:
            btype = "stable_basin"
            confidence = float(coherence)
        else:
            btype = "transient_basin"
            confidence = float(1.0 - coherence)

        classifications.append({
            "id": basin["id"],
            "type": btype,
            "confidence": confidence,
        })

    return classifications


# -- Step 5: state → basin mapping ----------------------------------------


def map_states_to_basins(
    components: List[Dict[str, Any]],
) -> Dict[Tuple[int, ...], int]:
    """Return ``{state: basin_id}`` for every state in every component."""
    mapping: Dict[Tuple[int, ...], int] = {}
    for comp in components:
        for node in comp["nodes"]:
            mapping[tuple(node)] = comp["id"]
    return mapping


# -- Step 6: pipeline -----------------------------------------------------


def run_basin_analysis(
    state_graph: Dict[str, Any],
    attractor_field: Dict[str, Any],
) -> Dict[str, Any]:
    """Full basin detection pipeline.

    Parameters
    ----------
    state_graph:
        ``{"nodes": [...], "edges": [...]}`` from motif-graph stage.
    attractor_field:
        ``{"nodes": [{"state": ..., "score": ...}], ...}`` from
        resonance analysis.

    Returns
    -------
    dict with ``basins``, ``classifications``, ``mapping``, ``n_basins``.
    """
    adjacency = build_adjacency(state_graph)
    scc = compute_scc(adjacency)
    components = scc["components"]
    basins = compute_basin_metrics(components, state_graph, attractor_field)
    classifications = classify_basins(basins, state_graph)
    mapping = map_states_to_basins(components)

    return {
        "basins": basins,
        "classifications": classifications,
        "mapping": mapping,
        "n_basins": scc["n_components"],
    }
