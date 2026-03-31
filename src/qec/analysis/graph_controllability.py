"""v123.0.0 — Deterministic graph controllability analysis.

Pure-Python directed graph utilities for:
- deterministic state graph construction
- Tarjan strongly connected component decomposition
- SCC condensation DAG construction
- shortest deterministic escape-route search
- SCC risk classification

All outputs are deterministic and stable for identical inputs.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple


Graph = Dict[str, List[str]]
SCCs = Tuple[Tuple[str, ...], ...]


def build_state_graph(edges: Iterable[Tuple[str, str]]) -> Graph:
    """Build a deterministic adjacency map from directed edges."""
    adjacency: Dict[str, Set[str]] = {}

    for src, dst in edges:
        if src not in adjacency:
            adjacency[src] = set()
        if dst not in adjacency:
            adjacency[dst] = set()
        adjacency[src].add(dst)

    graph: Graph = {}
    for node in sorted(adjacency.keys()):
        graph[node] = sorted(adjacency[node])

    return graph


def tarjan_scc(graph: Graph) -> SCCs:
    """Compute SCCs with deterministic Tarjan traversal and ordering."""
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    components: List[Tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1

        stack.append(node)
        on_stack.add(node)

        for neighbor in sorted(graph.get(node, [])):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component: List[str] = []
            while stack:
                candidate = stack.pop()
                on_stack.remove(candidate)
                component.append(candidate)
                if candidate == node:
                    break
            components.append(tuple(sorted(component)))

    for node in sorted(graph.keys()):
        if node not in indices:
            strongconnect(node)

    return tuple(sorted(components))


def build_condensation_dag(graph: Graph, sccs: SCCs) -> Dict[int, List[int]]:
    """Build deterministic SCC condensation DAG."""
    node_to_scc: Dict[str, int] = {}
    for scc_index, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_index

    dag_sets: Dict[int, Set[int]] = {idx: set() for idx in range(len(sccs))}

    for src in sorted(graph.keys()):
        src_scc = node_to_scc[src]
        for dst in sorted(graph.get(src, [])):
            dst_scc = node_to_scc[dst]
            if src_scc != dst_scc:
                dag_sets[src_scc].add(dst_scc)

    dag: Dict[int, List[int]] = {}
    for idx in range(len(sccs)):
        dag[idx] = sorted(dag_sets[idx])

    return dag


def find_escape_path(
    graph: Graph,
    start: str,
    safe_nodes: Set[str],
) -> Tuple[str, ...]:
    """Find shortest deterministic path from ``start`` to any safe node."""
    if start in safe_nodes:
        return (start,)

    queue: deque[Tuple[str, ...]] = deque([(start,)])
    visited: Set[str] = {start}

    while queue:
        path = queue.popleft()
        node = path[-1]

        for neighbor in sorted(graph.get(node, [])):
            if neighbor in visited:
                continue

            next_path = path + (neighbor,)
            if neighbor in safe_nodes:
                return next_path

            visited.add(neighbor)
            queue.append(next_path)

    return ()


def classify_scc_risk(
    graph: Graph,
    sccs: SCCs,
    safe_nodes: Set[str],
) -> Dict[int, str]:
    """Classify SCC risk as ``safe``, ``warning``, or ``critical``."""
    node_to_scc: Dict[str, int] = {}
    for scc_index, scc in enumerate(sccs):
        for node in scc:
            node_to_scc[node] = scc_index

    risk: Dict[int, str] = {}

    for scc_index, scc in enumerate(sccs):
        if any(node in safe_nodes for node in scc):
            risk[scc_index] = "safe"
            continue

        has_external_edge = False
        for node in scc:
            for neighbor in sorted(graph.get(node, [])):
                if node_to_scc.get(neighbor) != scc_index:
                    has_external_edge = True
                    break
            if has_external_edge:
                break

        if has_external_edge:
            risk[scc_index] = "warning"
        else:
            risk[scc_index] = "critical"

    return risk


def run_graph_controllability(
    edges: Iterable[Tuple[str, str]],
    start: str,
    safe_nodes: Set[str],
) -> Dict[str, object]:
    """Run deterministic graph controllability analysis pipeline."""
    graph = build_state_graph(edges)

    # Keep ``start`` observable even with no incident edges.
    if start not in graph:
        graph[start] = []

    sccs = tarjan_scc(graph)
    condensation_dag = build_condensation_dag(graph, sccs)
    escape_path = find_escape_path(graph, start, set(safe_nodes))
    risk_by_scc = classify_scc_risk(graph, sccs, set(safe_nodes))

    critical_sccs = tuple(sccs[idx] for idx in range(len(sccs)) if risk_by_scc[idx] == "critical")
    safe_sccs = tuple(sccs[idx] for idx in range(len(sccs)) if risk_by_scc[idx] == "safe")

    return {
        "sccs": sccs,
        "condensation_dag": condensation_dag,
        "escape_path": escape_path,
        "escape_possible": bool(escape_path),
        "critical_sccs": critical_sccs,
        "safe_sccs": safe_sccs,
    }
