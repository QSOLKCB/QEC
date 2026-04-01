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
from typing import Dict, Iterable, List, Optional, Set, Tuple


Graph = Dict[str, List[str]]
SCCs = Tuple[Tuple[str, ...], ...]


def build_state_graph(edges: Iterable[Tuple[str, str]]) -> Graph:
    """Build a deterministic adjacency map from directed edges."""
    adjacency: Dict[str, Set[str]] = {}

    for src, dst in edges:
        adjacency.setdefault(src, set())
        adjacency.setdefault(dst, set())
        adjacency[src].add(dst)

    graph: Graph = {}
    for node in sorted(adjacency.keys()):
        graph[node] = sorted(adjacency[node])

    return graph


def tarjan_scc(graph: Graph) -> SCCs:
    """Compute SCCs with deterministic iterative Kosaraju traversal/order.

    Note: adjacency lists are expected to be pre-sorted by ``build_state_graph``.
    """
    nodes = sorted(graph.keys())
    reverse_sets: Dict[str, Set[str]] = {node: set() for node in nodes}
    for src in nodes:
        for dst in graph.get(src, []):
            reverse_sets[dst].add(src)
    reverse_graph: Dict[str, List[str]] = {
        node: sorted(reverse_sets[node]) for node in nodes
    }

    visited: Set[str] = set()
    finish_order: List[str] = []

    for root in nodes:
        if root in visited:
            continue

        stack: List[Tuple[str, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            if node in visited:
                continue

            visited.add(node)
            stack.append((node, True))
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append((neighbor, False))

    assigned: Set[str] = set()
    components: List[Tuple[str, ...]] = []

    for root in reversed(finish_order):
        if root in assigned:
            continue

        stack = [root]
        component: List[str] = []
        assigned.add(root)

        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reverse_graph.get(node, []):
                if neighbor not in assigned:
                    assigned.add(neighbor)
                    stack.append(neighbor)

        components.append(tuple(sorted(component)))

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

    queue: deque[str] = deque([start])
    predecessors: Dict[str, Optional[str]] = {start: None}

    while queue:
        node = queue.popleft()

        for neighbor in sorted(graph.get(node, [])):
            if neighbor in predecessors:
                continue

            predecessors[neighbor] = node
            if neighbor in safe_nodes:
                path: List[str] = []
                cursor: Optional[str] = neighbor
                while cursor is not None:
                    path.append(cursor)
                    cursor = predecessors[cursor]
                path.reverse()
                return tuple(path)

            queue.append(neighbor)

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
    safe_nodes_set = set(safe_nodes)

    # Keep ``start`` observable even with no incident edges.
    if start not in graph:
        graph[start] = []

    sccs = tarjan_scc(graph)
    condensation_dag = build_condensation_dag(graph, sccs)
    escape_path = find_escape_path(graph, start, safe_nodes_set)
    risk_by_scc = classify_scc_risk(graph, sccs, safe_nodes_set)

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
