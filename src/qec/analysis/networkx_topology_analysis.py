"""v126.0.0 Network topology invariant discovery using NetworkX.

This module is fully additive and intentionally isolated from decoder paths.
All returned sequences use deterministic lexicographic ordering.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    """Return a deterministic canonical rotation for a cycle."""
    if not cycle:
        return ()
    n = len(cycle)
    forward_rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(n)]
    reversed_cycle = list(reversed(cycle))
    reversed_rotations = [tuple(reversed_cycle[i:] + reversed_cycle[:i]) for i in range(n)]
    return min(forward_rotations + reversed_rotations)


def build_nx_graph(state_graph: dict[str, list[str]]) -> nx.DiGraph:
    """Convert deterministic adjacency mapping into a directed NetworkX graph."""
    graph = nx.DiGraph()
    all_nodes = set(state_graph.keys())
    for neighbors in state_graph.values():
        all_nodes.update(neighbors)

    for node in sorted(all_nodes):
        graph.add_node(node)

    for source in sorted(state_graph.keys()):
        for target in sorted(state_graph[source]):
            graph.add_edge(source, target)

    return graph


def compute_topology_metrics(graph: nx.DiGraph) -> dict[str, int]:
    """Compute deterministic topology metrics for a directed graph."""
    scc_count = sum(1 for _ in nx.strongly_connected_components(graph))
    cycle_count = sum(1 for _ in nx.simple_cycles(graph))

    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]

    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "scc_count": scc_count,
        "cycle_count": cycle_count,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
    }


def discover_structural_invariants(graph: nx.DiGraph) -> dict[str, Any]:
    """Discover deterministic structural invariants from directed topology."""
    undirected_graph = graph.to_undirected(as_view=False)

    articulation_nodes = tuple(sorted(nx.articulation_points(undirected_graph)))

    bridge_pairs = []
    for u, v in nx.bridges(undirected_graph):
        pair = (u, v) if u <= v else (v, u)
        bridge_pairs.append(pair)
    critical_bridges = tuple(sorted(set(bridge_pairs)))

    cycle_set: set[tuple[str, ...]] = set()
    for cycle in nx.simple_cycles(graph):
        cycle_set.add(_canonical_cycle(cycle))
    unsafe_cycles = tuple(sorted(cycle_set))

    articulation_set = set(articulation_nodes)
    path_participation: dict[str, int] = defaultdict(int)
    nodes = sorted(graph.nodes())
    for source in nodes:
        for target in nodes:
            if source == target:
                continue
            if not nx.has_path(graph, source, target):
                continue
            path = nx.shortest_path(graph, source=source, target=target)
            internal_nodes = path[1:-1]
            if any(node in articulation_set for node in internal_nodes):
                continue
            for node in path:
                path_participation[node] += 1

    if path_participation:
        max_score = max(path_participation.values())
        dominant_nodes = tuple(sorted(node for node, score in path_participation.items() if score == max_score))
    else:
        dominant_nodes = ()

    return {
        "articulation_nodes": articulation_nodes,
        "critical_bridges": critical_bridges,
        "unsafe_cycles": unsafe_cycles,
        "dominant_nodes": dominant_nodes,
    }


def classify_topology_risk(metrics: dict[str, int], invariants: dict[str, Any]) -> str:
    """Classify topology risk label from metrics and invariants."""
    articulation_nodes = invariants.get("articulation_nodes", ())
    unsafe_cycles = invariants.get("unsafe_cycles", ())

    if len(articulation_nodes) >= 2 or len(unsafe_cycles) > 0:
        return "critical"
    if metrics.get("scc_count", 0) > 3 or metrics.get("cycle_count", 0) > 1:
        return "warning"
    return "safe"


def run_network_topology_analysis(state_graph: dict[str, list[str]]) -> dict[str, Any]:
    """Run full NetworkX-backed topology analysis with stable schema."""
    graph = build_nx_graph(state_graph)
    metrics = compute_topology_metrics(graph)
    invariants = discover_structural_invariants(graph)
    risk = classify_topology_risk(metrics=metrics, invariants=invariants)

    return {
        "topology_metrics": metrics,
        "structural_invariants": invariants,
        "topology_risk": risk,
        "networkx_enabled": True,
    }
