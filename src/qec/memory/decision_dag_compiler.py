"""v137.16.1 — Decision DAG Compiler.

Extends v137.16.0 — Deterministic Memory Graph Kernel.

This module compiles a deterministic Directed Acyclic Graph (DAG) from the
memory graph substrate. It is the second layer of the Codebase Memory +
Decision Compiler arc.

Determinism law:

    same input
    → same bytes
    → same dag hash
    → same traversal order
    → same traversal hash

The compiler is additive: it does not mutate the kernel, does not touch the
decoder, and does not introduce upward layer leakage. All dataclasses are
frozen, all ordering is canonical, and all hashes are SHA-256 over canonical
JSON.

Cycle detection is fail-fast: any cycle is rejected at normalization time
via Kahn's algorithm. Deterministic topological ordering is produced via a
priority-queue Kahn's walk keyed on ``(creation_epoch, node_kind, node_id)``
so a given input always yields the same linear extension.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import heapq
import json
from typing import Any, Dict, List, Mapping, Tuple, Union

from qec.memory.deterministic_memory_graph_kernel import (
    GraphValidationError,
    MemoryGraphEdge,
    MemoryGraphNode,
    normalize_deterministic_memory_graph,
)


# ---------------------------------------------------------------------------
# Structured validation error hierarchy.
#
# Compiler-specific errors carry a stable symbolic ``code`` tag. Codes from
# the kernel layer are preserved when re-raised so downstream consumers can
# branch on a single unified taxonomy.
# ---------------------------------------------------------------------------


class DecisionDAGError(ValueError):
    """Structured DAG compiler error carrying a stable symbolic code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Stable error codes — additive to the kernel's code surface.
ERR_DAG_TYPE_INVALID: str = "dag_type_invalid"
ERR_DAG_FIELDS_MISSING: str = "dag_fields_missing"
ERR_DAG_ID_INVALID: str = "dag_id_invalid"
ERR_DAG_CYCLE: str = "dag_cycle_detected"
ERR_TRAVERSAL_MODE_INVALID: str = "traversal_mode_invalid"
ERR_TRAVERSAL_START_INVALID: str = "traversal_start_invalid"
ERR_TRAVERSAL_DAG_TYPE_INVALID: str = "traversal_dag_type_invalid"
ERR_TRAVERSAL_CYCLE_STATE_INVALID: str = "traversal_cycle_state_invalid"


_REQUIRED_DAG_FIELDS: Tuple[str, ...] = ("dag_id", "nodes", "edges")

_ALLOWED_TRAVERSAL_MODES: Tuple[str, ...] = (
    "topological",
    "lineage",
    "dependency",
    "critical_path",
)

_LINEAGE_EDGE_KINDS: Tuple[str, ...] = ("derived_from", "supersedes")
_DEPENDENCY_EDGE_KINDS: Tuple[str, ...] = ("depends_on",)


DAGInputLike = Union["CompiledDecisionDAG", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Frozen dataclasses.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionDAGNode:
    node_id: str
    node_kind: str
    node_payload: str
    lineage_hash: str
    creation_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "node_payload": json.loads(self.node_payload),
            "lineage_hash": self.lineage_hash,
            "creation_epoch": self.creation_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DecisionDAGEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_kind: str
    edge_weight: float
    creation_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_kind": self.edge_kind,
            "edge_weight": self.edge_weight,
            "creation_epoch": self.creation_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CompiledDecisionDAG:
    dag_id: str
    nodes: Tuple[DecisionDAGNode, ...]
    edges: Tuple[DecisionDAGEdge, ...]
    topological_order: Tuple[str, ...]
    dag_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "topological_order": list(self.topological_order),
            "dag_hash": self.dag_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DecisionDAGValidationReport:
    dag_id: str
    is_valid: bool
    node_count: int
    edge_count: int
    structure_ok: bool
    uniqueness_ok: bool
    cycle_free_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dag_id": self.dag_id,
            "is_valid": self.is_valid,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "structure_ok": self.structure_ok,
            "uniqueness_ok": self.uniqueness_ok,
            "cycle_free_ok": self.cycle_free_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class DecisionDAGExecutionReceipt:
    receipt_id: str
    dag_id: str
    dag_hash: str
    traversal_mode: str
    start_node_id: str
    visited_nodes: Tuple[str, ...]
    visited_edges: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "dag_id": self.dag_id,
            "dag_hash": self.dag_hash,
            "traversal_mode": self.traversal_mode,
            "start_node_id": self.start_node_id,
            "visited_nodes": list(self.visited_nodes),
            "visited_edges": list(self.visited_edges),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _from_memory_node(node: MemoryGraphNode) -> DecisionDAGNode:
    return DecisionDAGNode(
        node_id=node.node_id,
        node_kind=node.node_kind,
        node_payload=node.node_payload,
        lineage_hash=node.lineage_hash,
        creation_epoch=node.creation_epoch,
    )


def _from_memory_edge(edge: MemoryGraphEdge) -> DecisionDAGEdge:
    return DecisionDAGEdge(
        edge_id=edge.edge_id,
        source_node_id=edge.source_node_id,
        target_node_id=edge.target_node_id,
        edge_kind=edge.edge_kind,
        edge_weight=edge.edge_weight,
        creation_epoch=edge.creation_epoch,
    )


def _unpack_dag_input(value: DAGInputLike) -> Tuple[str, Any, Any]:
    if isinstance(value, CompiledDecisionDAG):
        return (
            value.dag_id,
            [n.to_dict() for n in value.nodes],
            [e.to_dict() for e in value.edges],
        )
    if not isinstance(value, Mapping):
        raise DecisionDAGError(
            ERR_DAG_TYPE_INVALID,
            "dag input must be mapping or CompiledDecisionDAG",
        )
    missing = [f for f in _REQUIRED_DAG_FIELDS if f not in value]
    if missing:
        raise DecisionDAGError(
            ERR_DAG_FIELDS_MISSING,
            f"missing required dag fields: {missing}",
        )
    raw_id = value["dag_id"]
    if not isinstance(raw_id, str) or not raw_id.strip():
        raise DecisionDAGError(ERR_DAG_ID_INVALID, "invalid dag id")
    return raw_id.strip(), value["nodes"], value["edges"]


def _kernel_normalize(
    dag_id: str,
    raw_nodes: Any,
    raw_edges: Any,
) -> Tuple[Tuple[DecisionDAGNode, ...], Tuple[DecisionDAGEdge, ...]]:
    try:
        _, mem_nodes, mem_edges = normalize_deterministic_memory_graph(
            {
                "graph_id": dag_id,
                "nodes": raw_nodes,
                "edges": raw_edges,
            }
        )
    except GraphValidationError as exc:
        # Preserve the structured code so downstream callers can branch.
        raise DecisionDAGError(exc.code, str(exc)) from exc
    dag_nodes = tuple(_from_memory_node(n) for n in mem_nodes)
    dag_edges = tuple(_from_memory_edge(e) for e in mem_edges)
    return dag_nodes, dag_edges


def _topological_order(
    nodes: Tuple[DecisionDAGNode, ...],
    edges: Tuple[DecisionDAGEdge, ...],
) -> Union[Tuple[str, ...], None]:
    """Deterministic Kahn's topological walk.

    Returns ``None`` if a cycle is detected so the caller can decide
    whether to raise fail-fast or emit a validation report entry.
    """
    in_degree: Dict[str, int] = {n.node_id: 0 for n in nodes}
    out_edges: Dict[str, List[DecisionDAGEdge]] = {
        n.node_id: [] for n in nodes
    }
    for edge in edges:
        out_edges[edge.source_node_id].append(edge)
        in_degree[edge.target_node_id] += 1
    for node_id in list(out_edges.keys()):
        out_edges[node_id] = sorted(
            out_edges[node_id],
            key=lambda e: (e.creation_epoch, e.edge_kind, e.edge_id),
        )
    id_to_node: Dict[str, DecisionDAGNode] = {n.node_id: n for n in nodes}

    heap: List[Tuple[int, str, str]] = []
    for node in nodes:
        if in_degree[node.node_id] == 0:
            heapq.heappush(
                heap, (node.creation_epoch, node.node_kind, node.node_id)
            )

    topo: List[str] = []
    while heap:
        _, _, node_id = heapq.heappop(heap)
        topo.append(node_id)
        for edge in out_edges[node_id]:
            in_degree[edge.target_node_id] -= 1
            if in_degree[edge.target_node_id] == 0:
                tgt = id_to_node[edge.target_node_id]
                heapq.heappush(
                    heap,
                    (tgt.creation_epoch, tgt.node_kind, tgt.node_id),
                )

    if len(topo) != len(nodes):
        return None
    return tuple(topo)


def _compute_dag_hash(
    dag_id: str,
    nodes: Tuple[DecisionDAGNode, ...],
    edges: Tuple[DecisionDAGEdge, ...],
    topological_order: Tuple[str, ...],
) -> str:
    payload = {
        "dag_id": dag_id,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
        "topological_order": list(topological_order),
    }
    return _sha256_hex(_canonical_bytes(payload))


def _normalize_and_toposort(
    value: DAGInputLike,
) -> Tuple[
    str,
    Tuple[DecisionDAGNode, ...],
    Tuple[DecisionDAGEdge, ...],
    Tuple[str, ...],
]:
    dag_id, raw_nodes, raw_edges = _unpack_dag_input(value)
    dag_nodes, dag_edges = _kernel_normalize(dag_id, raw_nodes, raw_edges)
    topo = _topological_order(dag_nodes, dag_edges)
    if topo is None:
        raise DecisionDAGError(
            ERR_DAG_CYCLE,
            f"cycle detected in decision dag {dag_id}",
        )
    return dag_id, dag_nodes, dag_edges, topo


def _assert_compiled_topology_consistent(dag: "CompiledDecisionDAG") -> None:
    """Assert that a compiled DAG's topological order is consistent with its nodes and edges.

    Raises :class:`DecisionDAGError` with ``ERR_TRAVERSAL_CYCLE_STATE_INVALID``
    for any inconsistency: mismatched node sets, duplicate order entries, edge
    endpoints that reference unknown node ids, or edges that violate the
    topological ordering.
    """
    node_ids = {n.node_id for n in dag.nodes}
    topo_ids = set(dag.topological_order)
    if topo_ids != node_ids or len(dag.topological_order) != len(dag.nodes):
        raise DecisionDAGError(
            ERR_TRAVERSAL_CYCLE_STATE_INVALID,
            "compiled dag topological order is inconsistent with nodes",
        )
    for e in dag.edges:
        if e.source_node_id not in node_ids or e.target_node_id not in node_ids:
            raise DecisionDAGError(
                ERR_TRAVERSAL_CYCLE_STATE_INVALID,
                "compiled dag edge references unknown node id",
            )
    topo_pos = {nid: idx for idx, nid in enumerate(dag.topological_order)}
    if any(
        topo_pos[e.source_node_id] >= topo_pos[e.target_node_id]
        for e in dag.edges
    ):
        raise DecisionDAGError(
            ERR_TRAVERSAL_CYCLE_STATE_INVALID,
            "compiled dag topological order violates edge ordering",
        )


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def normalize_decision_dag_input(
    value: DAGInputLike,
) -> Tuple[str, Tuple[DecisionDAGNode, ...], Tuple[DecisionDAGEdge, ...]]:
    """Deterministically normalize a decision DAG input.

    Accepts a mapping ``{"dag_id": ..., "nodes": [...], "edges": [...]}`` or
    a :class:`CompiledDecisionDAG`. Structural invariants are delegated to
    the memory graph kernel; cycle detection is performed here via Kahn's
    algorithm.

    Rejects (fail fast):

    * non-mapping / missing fields / invalid dag id
    * duplicate node ids
    * duplicate edge ids
    * missing edge references
    * invalid node / edge kinds
    * malformed payloads
    * negative weights
    * malformed lineage hashes
    * cycles
    """
    dag_id, dag_nodes, dag_edges, _ = _normalize_and_toposort(value)
    return dag_id, dag_nodes, dag_edges


def compile_decision_dag(value: DAGInputLike) -> CompiledDecisionDAG:
    """Deterministically compile a canonical decision DAG.

    Normalizes the input, computes a deterministic topological ordering,
    stores nodes in topological order, and seals the result with a
    SHA-256 ``dag_hash`` over the canonical JSON form. Repeated compilation
    over identical input yields byte-identical DAGs and byte-identical
    hashes.
    """
    dag_id, dag_nodes, dag_edges, topo = _normalize_and_toposort(value)
    id_to_node = {n.node_id: n for n in dag_nodes}
    ordered_nodes = tuple(id_to_node[nid] for nid in topo)
    dag_hash = _compute_dag_hash(dag_id, ordered_nodes, dag_edges, topo)
    compiled = CompiledDecisionDAG(
        dag_id=dag_id,
        nodes=ordered_nodes,
        edges=dag_edges,
        topological_order=topo,
        dag_hash=dag_hash,
    )
    _assert_compiled_topology_consistent(compiled)
    return compiled


def validate_decision_dag(
    value: DAGInputLike,
) -> DecisionDAGValidationReport:
    """Produce a deterministic validation report for a decision DAG input.

    Top-level unpack failures raise fast (for example, non-mapping input,
    missing top-level fields, or an invalid dag id). Once ``dag_id``,
    ``nodes``, and ``edges`` are unpacked, normalization and graph-structure
    problems are returned as a deterministic invalid report rather than
    raised.
    """
    dag_id, raw_nodes, raw_edges = _unpack_dag_input(value)

    violations: List[str] = []
    structure_ok = True
    uniqueness_ok = True
    cycle_free_ok = True
    node_count = 0
    edge_count = 0

    try:
        dag_nodes, dag_edges = _kernel_normalize(dag_id, raw_nodes, raw_edges)
    except DecisionDAGError as exc:
        if exc.code in ("duplicate_node", "duplicate_edge"):
            uniqueness_ok = False
        else:
            structure_ok = False
        violations.append(f"{exc.code}: {exc}")
        return DecisionDAGValidationReport(
            dag_id=dag_id,
            is_valid=False,
            node_count=0,
            edge_count=0,
            structure_ok=structure_ok,
            uniqueness_ok=uniqueness_ok,
            cycle_free_ok=cycle_free_ok,
            violations=tuple(violations),
        )

    node_count = len(dag_nodes)
    edge_count = len(dag_edges)
    topo = _topological_order(dag_nodes, dag_edges)
    if topo is None:
        cycle_free_ok = False
        violations.append(
            f"{ERR_DAG_CYCLE}: cycle detected in decision dag {dag_id}"
        )

    is_valid = structure_ok and uniqueness_ok and cycle_free_ok
    return DecisionDAGValidationReport(
        dag_id=dag_id,
        is_valid=is_valid,
        node_count=node_count,
        edge_count=edge_count,
        structure_ok=structure_ok,
        uniqueness_ok=uniqueness_ok,
        cycle_free_ok=cycle_free_ok,
        violations=tuple(violations),
    )


# ---------------------------------------------------------------------------
# Traversal.
# ---------------------------------------------------------------------------


def _build_out_adjacency(
    dag: CompiledDecisionDAG,
) -> Dict[str, Tuple[DecisionDAGEdge, ...]]:
    buckets: Dict[str, List[DecisionDAGEdge]] = {
        n.node_id: [] for n in dag.nodes
    }
    for edge in dag.edges:
        buckets[edge.source_node_id].append(edge)
    adjacency: Dict[str, Tuple[DecisionDAGEdge, ...]] = {}
    for node_id, bucket in buckets.items():
        adjacency[node_id] = tuple(
            sorted(
                bucket,
                key=lambda e: (e.creation_epoch, e.edge_kind, e.edge_id),
            )
        )
    return adjacency


def _traverse_topological(
    dag: CompiledDecisionDAG,
    start: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    adjacency = _build_out_adjacency(dag)
    reachable: set = {start}
    queue: "deque[str]" = deque([start])
    while queue:
        current = queue.popleft()
        for edge in adjacency.get(current, ()):
            if edge.target_node_id in reachable:
                continue
            reachable.add(edge.target_node_id)
            queue.append(edge.target_node_id)
    visited_nodes = tuple(
        nid for nid in dag.topological_order if nid in reachable
    )
    visited_edges = tuple(
        e.edge_id
        for e in dag.edges
        if e.source_node_id in reachable and e.target_node_id in reachable
    )
    return visited_nodes, visited_edges


def _traverse_filtered_bfs(
    dag: CompiledDecisionDAG,
    start: str,
    allowed_kinds: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    adjacency = _build_out_adjacency(dag)
    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    seen: set = {start}
    queue: "deque[str]" = deque([start])
    while queue:
        current = queue.popleft()
        visited_nodes.append(current)
        for edge in adjacency.get(current, ()):
            if edge.edge_kind not in allowed_kinds:
                continue
            if edge.target_node_id in seen:
                continue
            seen.add(edge.target_node_id)
            visited_edges.append(edge.edge_id)
            queue.append(edge.target_node_id)
    return tuple(visited_nodes), tuple(visited_edges)


def _traverse_critical_path(
    dag: CompiledDecisionDAG,
    start: str,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    adjacency = _build_out_adjacency(dag)
    neg_inf = float("-inf")
    dist: Dict[str, float] = {n.node_id: neg_inf for n in dag.nodes}
    prev_edge: Dict[str, str] = {}
    prev_node: Dict[str, str] = {}
    dist[start] = 0.0
    for nid in dag.topological_order:
        if dist[nid] == neg_inf:
            continue
        for edge in adjacency.get(nid, ()):
            candidate = dist[nid] + edge.edge_weight
            target = edge.target_node_id
            if candidate > dist[target]:
                dist[target] = candidate
                prev_edge[target] = edge.edge_id
                prev_node[target] = nid
    # Pick farthest reachable node. Ties broken deterministically by
    # ascending node_id; iteration uses topological order for a stable
    # base traversal.
    reachable_pairs: List[Tuple[float, str]] = [
        (dist[nid], nid)
        for nid in dag.topological_order
        if dist[nid] != neg_inf
    ]
    reachable_pairs.sort(key=lambda t: (-t[0], t[1]))
    best_nid = reachable_pairs[0][1]

    path_nodes_rev: List[str] = []
    path_edges_rev: List[str] = []
    cursor: str = best_nid
    while True:
        path_nodes_rev.append(cursor)
        if cursor not in prev_node:
            break
        path_edges_rev.append(prev_edge[cursor])
        cursor = prev_node[cursor]
    path_nodes_rev.reverse()
    path_edges_rev.reverse()
    return tuple(path_nodes_rev), tuple(path_edges_rev)


def _compute_traversal_hash(
    dag_hash: str,
    traversal_mode: str,
    start_node_id: str,
    visited_nodes: Tuple[str, ...],
    visited_edges: Tuple[str, ...],
) -> str:
    payload = {
        "dag_hash": dag_hash,
        "traversal_mode": traversal_mode,
        "start_node_id": start_node_id,
        "visited_nodes": list(visited_nodes),
        "visited_edges": list(visited_edges),
    }
    return _sha256_hex(_canonical_bytes(payload))


def traverse_decision_dag(
    dag: CompiledDecisionDAG,
    start_node_id: str,
    traversal_mode: str,
) -> DecisionDAGExecutionReceipt:
    """Deterministically traverse a compiled decision DAG.

    Supported traversal modes:

    * ``topological`` — emit all nodes reachable from ``start_node_id`` in
      the DAG's canonical topological order, plus all induced edges.
    * ``lineage`` — BFS restricted to ``derived_from`` and ``supersedes``
      edges, following the same canonical child ordering as the kernel.
    * ``dependency`` — BFS restricted to ``depends_on`` edges.
    * ``critical_path`` — longest-weighted-path from ``start_node_id``,
      reconstructed deterministically.

    The result is a :class:`DecisionDAGExecutionReceipt` containing the
    deterministic node trace, deterministic edge trace, and a SHA-256
    ``traversal_hash``. Repeated traversal yields byte-identical output.
    """
    if not isinstance(dag, CompiledDecisionDAG):
        raise DecisionDAGError(
            ERR_TRAVERSAL_DAG_TYPE_INVALID,
            "dag must be CompiledDecisionDAG",
        )
    if not isinstance(start_node_id, str) or not start_node_id.strip():
        raise DecisionDAGError(
            ERR_TRAVERSAL_START_INVALID,
            "invalid start node id",
        )
    start = start_node_id.strip()
    if traversal_mode not in _ALLOWED_TRAVERSAL_MODES:
        raise DecisionDAGError(
            ERR_TRAVERSAL_MODE_INVALID,
            f"unsupported traversal mode: {traversal_mode}",
        )

    _assert_compiled_topology_consistent(dag)
    if start not in {n.node_id for n in dag.nodes}:
        raise DecisionDAGError(
            ERR_TRAVERSAL_START_INVALID,
            f"start node not in dag: {start}",
        )

    if traversal_mode == "topological":
        visited_nodes, visited_edges = _traverse_topological(dag, start)
    elif traversal_mode == "lineage":
        visited_nodes, visited_edges = _traverse_filtered_bfs(
            dag, start, _LINEAGE_EDGE_KINDS
        )
    elif traversal_mode == "dependency":
        visited_nodes, visited_edges = _traverse_filtered_bfs(
            dag, start, _DEPENDENCY_EDGE_KINDS
        )
    else:  # critical_path
        visited_nodes, visited_edges = _traverse_critical_path(dag, start)

    traversal_hash = _compute_traversal_hash(
        dag.dag_hash,
        traversal_mode,
        start,
        visited_nodes,
        visited_edges,
    )
    receipt_id = f"dag-receipt-{traversal_hash[:32]}"
    return DecisionDAGExecutionReceipt(
        receipt_id=receipt_id,
        dag_id=dag.dag_id,
        dag_hash=dag.dag_hash,
        traversal_mode=traversal_mode,
        start_node_id=start,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        traversal_hash=traversal_hash,
    )


__all__ = (
    "DecisionDAGNode",
    "DecisionDAGEdge",
    "CompiledDecisionDAG",
    "DecisionDAGValidationReport",
    "DecisionDAGExecutionReceipt",
    "DecisionDAGError",
    "ERR_DAG_TYPE_INVALID",
    "ERR_DAG_FIELDS_MISSING",
    "ERR_DAG_ID_INVALID",
    "ERR_DAG_CYCLE",
    "ERR_TRAVERSAL_MODE_INVALID",
    "ERR_TRAVERSAL_START_INVALID",
    "ERR_TRAVERSAL_DAG_TYPE_INVALID",
    "ERR_TRAVERSAL_CYCLE_STATE_INVALID",
    "normalize_decision_dag_input",
    "compile_decision_dag",
    "validate_decision_dag",
    "traverse_decision_dag",
)
