"""v137.16.0 — Deterministic Memory Graph Kernel.

First release of the Codebase Memory + Decision Compiler arc.

This module defines a deterministic directed memory graph substrate for:

- codebase topology
- release lineage
- decision dependencies
- replay-safe reasoning traces

It is the graph kernel only. It does not compile decision DAGs, index
topology, or implement a reasoning engine. Higher layers in the memory /
compiler arc consume this substrate.

Determinism law:
    same input = same bytes = same graph hash = same traversal order.

The kernel exposes frozen dataclasses for nodes, edges, graphs, validation
reports, and traversal execution receipts. All ordering is canonical and
all hashes are SHA-256 over canonical JSON.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Set, Tuple, Union


_ALLOWED_NODE_KINDS: Tuple[str, ...] = (
    "release",
    "module",
    "decision",
    "artifact",
    "proof",
    "test",
)

_ALLOWED_EDGE_KINDS: Tuple[str, ...] = (
    "depends_on",
    "derived_from",
    "verified_by",
    "tests",
    "supersedes",
)

_ALLOWED_TRAVERSAL_MODES: Tuple[str, ...] = (
    "bfs",
    "dfs",
    "lineage",
    "dependency",
)

_LINEAGE_TRAVERSAL_KINDS: Tuple[str, ...] = ("derived_from", "supersedes")
_DEPENDENCY_TRAVERSAL_KINDS: Tuple[str, ...] = ("depends_on",)

_REQUIRED_NODE_FIELDS: Tuple[str, ...] = (
    "node_id",
    "node_kind",
    "node_payload",
    "lineage_hash",
    "creation_epoch",
)

_REQUIRED_EDGE_FIELDS: Tuple[str, ...] = (
    "edge_id",
    "source_node_id",
    "target_node_id",
    "edge_kind",
    "edge_weight",
    "creation_epoch",
)

_REQUIRED_GRAPH_FIELDS: Tuple[str, ...] = ("graph_id", "nodes", "edges")

_LINEAGE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

NodeLike = Union["MemoryGraphNode", Mapping[str, Any]]
EdgeLike = Union["MemoryGraphEdge", Mapping[str, Any]]
GraphLike = Union["DeterministicMemoryGraph", Mapping[str, Any]]


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


def _canonical_payload_value(value: Any, label: str) -> Any:
    # Determinism contract: node payloads must be JSON-safe and composed of
    # primitives, lists/tuples, and string-keyed mappings only. Unsupported
    # types are rejected so serialization order and bytes cannot drift.
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"malformed payload in {label}")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for key in value.keys():
            if not isinstance(key, str):
                raise ValueError(f"malformed payload in {label}")
            out[key] = _canonical_payload_value(value[key], label)
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical_payload_value(v, label) for v in value]
    raise ValueError(f"malformed payload in {label}")


def _normalize_payload(raw: Any, label: str) -> str:
    """Normalize a node payload mapping to a canonical JSON string.

    The canonical string form makes the payload hashable (so the enclosing
    dataclass can remain frozen) while preserving deterministic byte output
    on replay.
    """
    if not isinstance(raw, Mapping):
        raise ValueError(f"malformed payload in {label}")
    canonical_value = _canonical_payload_value(raw, label)
    return _canonical_json(canonical_value)


@dataclass(frozen=True)
class MemoryGraphNode:
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
class MemoryGraphEdge:
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
class DeterministicMemoryGraph:
    graph_id: str
    nodes: Tuple[MemoryGraphNode, ...]
    edges: Tuple[MemoryGraphEdge, ...]
    graph_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "graph_hash": self.graph_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class MemoryGraphValidationReport:
    graph_id: str
    is_valid: bool
    node_count: int
    edge_count: int
    uniqueness_ok: bool
    node_validity_ok: bool
    edge_validity_ok: bool
    payload_validity_ok: bool
    lineage_validity_ok: bool
    weight_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "is_valid": self.is_valid,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "uniqueness_ok": self.uniqueness_ok,
            "node_validity_ok": self.node_validity_ok,
            "edge_validity_ok": self.edge_validity_ok,
            "payload_validity_ok": self.payload_validity_ok,
            "lineage_validity_ok": self.lineage_validity_ok,
            "weight_validity_ok": self.weight_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class MemoryGraphExecutionReceipt:
    receipt_id: str
    graph_id: str
    graph_hash: str
    traversal_mode: str
    start_node_id: str
    visited_nodes: Tuple[str, ...]
    visited_edges: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "graph_id": self.graph_id,
            "graph_hash": self.graph_hash,
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


def _normalize_node(value: NodeLike) -> MemoryGraphNode:
    if isinstance(value, MemoryGraphNode):
        data: Mapping[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        data = value
    else:
        raise ValueError("node must be mapping or MemoryGraphNode")

    missing = [f for f in _REQUIRED_NODE_FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing required node fields: {missing}")

    raw_id = data["node_id"]
    if not isinstance(raw_id, str):
        raise ValueError("invalid node id")
    node_id = raw_id.strip()
    if not node_id:
        raise ValueError("invalid node id")

    raw_kind = data["node_kind"]
    if not isinstance(raw_kind, str):
        raise ValueError(f"unsupported node kind: {raw_kind!r}")
    node_kind = raw_kind
    if node_kind not in _ALLOWED_NODE_KINDS:
        raise ValueError(f"unsupported node kind: {node_kind}")

    node_payload = _normalize_payload(
        data["node_payload"], f"node {node_id}"
    )

    raw_lineage = data["lineage_hash"]
    if not isinstance(raw_lineage, str):
        raise ValueError(f"malformed lineage hash on node {node_id}")
    lineage_hash = raw_lineage
    if not _LINEAGE_HASH_RE.match(lineage_hash):
        raise ValueError(f"malformed lineage hash on node {node_id}")

    raw_epoch = data["creation_epoch"]
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int):
        raise ValueError(f"invalid creation epoch on node {node_id}")
    if raw_epoch < 0:
        raise ValueError(f"invalid creation epoch on node {node_id}")

    return MemoryGraphNode(
        node_id=node_id,
        node_kind=node_kind,
        node_payload=node_payload,
        lineage_hash=lineage_hash,
        creation_epoch=raw_epoch,
    )


def _normalize_edge(value: EdgeLike) -> MemoryGraphEdge:
    if isinstance(value, MemoryGraphEdge):
        data: Mapping[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        data = value
    else:
        raise ValueError("edge must be mapping or MemoryGraphEdge")

    missing = [f for f in _REQUIRED_EDGE_FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing required edge fields: {missing}")

    raw_id = data["edge_id"]
    if not isinstance(raw_id, str):
        raise ValueError("invalid edge id")
    edge_id = raw_id.strip()
    if not edge_id:
        raise ValueError("invalid edge id")

    raw_source = data["source_node_id"]
    raw_target = data["target_node_id"]
    if not isinstance(raw_source, str) or not isinstance(raw_target, str):
        raise ValueError(f"invalid edge endpoints on edge {edge_id}")
    source_node_id = raw_source.strip()
    target_node_id = raw_target.strip()
    if not source_node_id or not target_node_id:
        raise ValueError(f"invalid edge endpoints on edge {edge_id}")

    raw_kind = data["edge_kind"]
    if not isinstance(raw_kind, str):
        raise ValueError(f"unsupported edge kind: {raw_kind!r}")
    edge_kind = raw_kind
    if edge_kind not in _ALLOWED_EDGE_KINDS:
        raise ValueError(f"unsupported edge kind: {edge_kind}")

    raw_weight = data["edge_weight"]
    if isinstance(raw_weight, bool):
        raise ValueError(f"invalid edge weight on edge {edge_id}")
    if not isinstance(raw_weight, (int, float)):
        raise ValueError(f"invalid edge weight on edge {edge_id}")
    edge_weight = float(raw_weight)
    if edge_weight != edge_weight or edge_weight in (
        float("inf"),
        float("-inf"),
    ):
        raise ValueError(f"invalid edge weight on edge {edge_id}")
    if edge_weight < 0.0:
        raise ValueError(f"negative edge weight on edge {edge_id}")

    raw_epoch = data["creation_epoch"]
    if isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int):
        raise ValueError(f"invalid creation epoch on edge {edge_id}")
    if raw_epoch < 0:
        raise ValueError(f"invalid creation epoch on edge {edge_id}")

    return MemoryGraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_kind=edge_kind,
        edge_weight=edge_weight,
        creation_epoch=raw_epoch,
    )


def _unpack_graph_input(
    value: GraphLike,
) -> Tuple[str, Any, Any]:
    if isinstance(value, DeterministicMemoryGraph):
        data: Mapping[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        data = value
    else:
        raise ValueError("graph must be mapping or DeterministicMemoryGraph")

    missing = [f for f in _REQUIRED_GRAPH_FIELDS if f not in data]
    if missing:
        raise ValueError(f"missing required graph fields: {missing}")

    raw_graph_id = data["graph_id"]
    if not isinstance(raw_graph_id, str) or not raw_graph_id.strip():
        raise ValueError("invalid graph id")

    raw_nodes = data["nodes"]
    raw_edges = data["edges"]
    if not isinstance(raw_nodes, (list, tuple)):
        raise ValueError("nodes must be list or tuple")
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError("edges must be list or tuple")

    return raw_graph_id.strip(), raw_nodes, raw_edges


def normalize_deterministic_memory_graph(
    value: GraphLike,
) -> Tuple[str, Tuple[MemoryGraphNode, ...], Tuple[MemoryGraphEdge, ...]]:
    """Deterministically normalize a memory graph input.

    Accepts either a mapping ``{"graph_id": ..., "nodes": [...],
    "edges": [...]}`` or a ``DeterministicMemoryGraph`` instance. Returns
    the canonical graph id and fully ordered node / edge tuples.

    Rejects (fail fast):

    * duplicate node ids
    * duplicate edge ids
    * missing edge references
    * unsupported node kinds
    * unsupported edge kinds
    * malformed payloads
    * negative weights
    * malformed lineage hashes
    """
    graph_id, raw_nodes, raw_edges = _unpack_graph_input(value)

    normalized_nodes: List[MemoryGraphNode] = []
    seen_node_ids: Set[str] = set()
    for raw in raw_nodes:
        node = _normalize_node(raw)
        if node.node_id in seen_node_ids:
            raise ValueError(f"duplicate node id: {node.node_id}")
        seen_node_ids.add(node.node_id)
        normalized_nodes.append(node)

    normalized_edges: List[MemoryGraphEdge] = []
    seen_edge_ids: Set[str] = set()
    for raw in raw_edges:
        edge = _normalize_edge(raw)
        if edge.edge_id in seen_edge_ids:
            raise ValueError(f"duplicate edge id: {edge.edge_id}")
        if edge.source_node_id not in seen_node_ids:
            raise ValueError(
                f"edge {edge.edge_id} references missing source node "
                f"{edge.source_node_id}"
            )
        if edge.target_node_id not in seen_node_ids:
            raise ValueError(
                f"edge {edge.edge_id} references missing target node "
                f"{edge.target_node_id}"
            )
        seen_edge_ids.add(edge.edge_id)
        normalized_edges.append(edge)

    ordered_nodes = tuple(
        sorted(
            normalized_nodes,
            key=lambda n: (n.creation_epoch, n.node_kind, n.node_id),
        )
    )
    ordered_edges = tuple(
        sorted(
            normalized_edges,
            key=lambda e: (e.creation_epoch, e.edge_kind, e.edge_id),
        )
    )
    return graph_id, ordered_nodes, ordered_edges


def _classify_violation(
    msg: str,
) -> Tuple[str, ...]:
    # Map an exception message to (uniqueness, node, edge, payload, lineage,
    # weight) channels. The returned tuple lists flag names that must be
    # flipped to False.
    if msg.startswith("duplicate node id") or msg.startswith(
        "duplicate edge id"
    ):
        return ("uniqueness_ok",)
    if "payload" in msg:
        return ("payload_validity_ok",)
    if "lineage" in msg:
        return ("lineage_validity_ok",)
    if "weight" in msg:
        return ("weight_validity_ok",)
    if msg.startswith("edge ") or msg.startswith("invalid edge") or (
        "edge kind" in msg
    ) or msg.startswith("unsupported edge kind"):
        return ("edge_validity_ok",)
    return ("node_validity_ok",)


def validate_deterministic_memory_graph(
    value: GraphLike,
) -> MemoryGraphValidationReport:
    """Produce a deterministic validation report for a memory graph input.

    Unlike :func:`normalize_deterministic_memory_graph`, this function does
    not raise on invariant violations — instead it returns a byte-stable
    report enumerating every failure. Structural input errors (non-mapping
    graph value, missing top-level fields) still raise fast since the
    report cannot be constructed without a graph id and iterable nodes /
    edges.
    """
    graph_id, raw_nodes, raw_edges = _unpack_graph_input(value)

    violations: List[str] = []
    flags: Dict[str, bool] = {
        "uniqueness_ok": True,
        "node_validity_ok": True,
        "edge_validity_ok": True,
        "payload_validity_ok": True,
        "lineage_validity_ok": True,
        "weight_validity_ok": True,
    }

    normalized_nodes: List[MemoryGraphNode] = []
    seen_node_ids: Set[str] = set()
    for index, raw in enumerate(raw_nodes):
        try:
            node = _normalize_node(raw)
        except ValueError as exc:
            for flag in _classify_violation(str(exc)):
                flags[flag] = False
            violations.append(f"node[{index}]: {exc}")
            continue
        if node.node_id in seen_node_ids:
            flags["uniqueness_ok"] = False
            violations.append(f"duplicate node id: {node.node_id}")
            continue
        seen_node_ids.add(node.node_id)
        normalized_nodes.append(node)

    seen_edge_ids: Set[str] = set()
    for index, raw in enumerate(raw_edges):
        try:
            edge = _normalize_edge(raw)
        except ValueError as exc:
            for flag in _classify_violation(str(exc)):
                flags[flag] = False
            violations.append(f"edge[{index}]: {exc}")
            continue
        if edge.edge_id in seen_edge_ids:
            flags["uniqueness_ok"] = False
            violations.append(f"duplicate edge id: {edge.edge_id}")
            continue
        if edge.source_node_id not in seen_node_ids:
            flags["edge_validity_ok"] = False
            violations.append(
                f"edge {edge.edge_id} references missing source node "
                f"{edge.source_node_id}"
            )
            continue
        if edge.target_node_id not in seen_node_ids:
            flags["edge_validity_ok"] = False
            violations.append(
                f"edge {edge.edge_id} references missing target node "
                f"{edge.target_node_id}"
            )
            continue
        seen_edge_ids.add(edge.edge_id)

    is_valid = all(flags.values())

    return MemoryGraphValidationReport(
        graph_id=graph_id,
        is_valid=is_valid,
        node_count=len(seen_node_ids),
        edge_count=len(seen_edge_ids),
        uniqueness_ok=flags["uniqueness_ok"],
        node_validity_ok=flags["node_validity_ok"],
        edge_validity_ok=flags["edge_validity_ok"],
        payload_validity_ok=flags["payload_validity_ok"],
        lineage_validity_ok=flags["lineage_validity_ok"],
        weight_validity_ok=flags["weight_validity_ok"],
        violations=tuple(violations),
    )


def _compute_graph_hash(
    graph_id: str,
    nodes: Tuple[MemoryGraphNode, ...],
    edges: Tuple[MemoryGraphEdge, ...],
) -> str:
    payload = {
        "graph_id": graph_id,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }
    return _sha256_hex(_canonical_bytes(payload))


def build_deterministic_memory_graph(
    value: GraphLike,
) -> DeterministicMemoryGraph:
    """Deterministically construct a canonical memory graph and its hash.

    Invokes :func:`normalize_deterministic_memory_graph` to enforce
    invariants, then computes the canonical SHA-256 ``graph_hash``.
    Repeated construction over identical input yields byte-identical
    graphs and byte-identical hashes.
    """
    graph_id, ordered_nodes, ordered_edges = (
        normalize_deterministic_memory_graph(value)
    )
    graph_hash = _compute_graph_hash(graph_id, ordered_nodes, ordered_edges)
    return DeterministicMemoryGraph(
        graph_id=graph_id,
        nodes=ordered_nodes,
        edges=ordered_edges,
        graph_hash=graph_hash,
    )


def _build_adjacency(
    graph: DeterministicMemoryGraph,
) -> Dict[str, Tuple[MemoryGraphEdge, ...]]:
    buckets: Dict[str, List[MemoryGraphEdge]] = {
        n.node_id: [] for n in graph.nodes
    }
    for edge in graph.edges:
        buckets[edge.source_node_id].append(edge)
    adjacency: Dict[str, Tuple[MemoryGraphEdge, ...]] = {}
    for node_id, bucket in buckets.items():
        adjacency[node_id] = tuple(
            sorted(
                bucket,
                key=lambda e: (e.creation_epoch, e.edge_kind, e.edge_id),
            )
        )
    return adjacency


def _allowed_edge_kinds_for_mode(traversal_mode: str) -> Tuple[str, ...]:
    if traversal_mode == "lineage":
        return _LINEAGE_TRAVERSAL_KINDS
    if traversal_mode == "dependency":
        return _DEPENDENCY_TRAVERSAL_KINDS
    return _ALLOWED_EDGE_KINDS


def _bfs_traverse(
    start: str,
    adjacency: Dict[str, Tuple[MemoryGraphEdge, ...]],
    allowed_kinds: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    seen: Set[str] = {start}
    queue: "deque[str]" = deque()
    queue.append(start)
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


def _dfs_traverse(
    start: str,
    adjacency: Dict[str, Tuple[MemoryGraphEdge, ...]],
    allowed_kinds: Tuple[str, ...],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    seen: Set[str] = {start}
    stack: List[str] = [start]
    while stack:
        current = stack.pop()
        visited_nodes.append(current)
        frontier: List[MemoryGraphEdge] = []
        for edge in adjacency.get(current, ()):
            if edge.edge_kind not in allowed_kinds:
                continue
            if edge.target_node_id in seen:
                continue
            frontier.append(edge)
        # Push in reverse so the canonically-first child is popped next.
        for edge in reversed(frontier):
            if edge.target_node_id in seen:
                continue
            seen.add(edge.target_node_id)
            visited_edges.append(edge.edge_id)
            stack.append(edge.target_node_id)
    return tuple(visited_nodes), tuple(visited_edges)


def _compute_traversal_hash(
    graph_hash: str,
    traversal_mode: str,
    start_node_id: str,
    visited_nodes: Tuple[str, ...],
    visited_edges: Tuple[str, ...],
) -> str:
    payload = {
        "graph_hash": graph_hash,
        "traversal_mode": traversal_mode,
        "start_node_id": start_node_id,
        "visited_nodes": list(visited_nodes),
        "visited_edges": list(visited_edges),
    }
    return _sha256_hex(_canonical_bytes(payload))


def traverse_deterministic_memory_graph(
    graph: DeterministicMemoryGraph,
    start_node_id: str,
    traversal_mode: str,
) -> MemoryGraphExecutionReceipt:
    """Deterministically traverse a canonical memory graph.

    Supported traversal modes:

    * ``bfs`` — breadth-first traversal over all edge kinds.
    * ``dfs`` — depth-first traversal over all edge kinds.
    * ``lineage`` — breadth-first restricted to ``derived_from`` and
      ``supersedes`` edges.
    * ``dependency`` — breadth-first restricted to ``depends_on`` edges.

    The result is a :class:`MemoryGraphExecutionReceipt` containing the
    deterministic node trace, deterministic edge trace, and a SHA-256
    traversal hash. Repeated traversal yields byte-identical output.
    """
    if not isinstance(graph, DeterministicMemoryGraph):
        raise ValueError("graph must be DeterministicMemoryGraph")
    if not isinstance(start_node_id, str) or not start_node_id.strip():
        raise ValueError("invalid start node id")
    start = start_node_id.strip()
    if traversal_mode not in _ALLOWED_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {traversal_mode}")

    node_ids = {n.node_id for n in graph.nodes}
    if start not in node_ids:
        raise ValueError(f"start node not in graph: {start}")

    adjacency = _build_adjacency(graph)
    allowed_kinds = _allowed_edge_kinds_for_mode(traversal_mode)

    if traversal_mode == "dfs":
        visited_nodes, visited_edges = _dfs_traverse(
            start, adjacency, allowed_kinds
        )
    else:
        visited_nodes, visited_edges = _bfs_traverse(
            start, adjacency, allowed_kinds
        )

    traversal_hash = _compute_traversal_hash(
        graph.graph_hash,
        traversal_mode,
        start,
        visited_nodes,
        visited_edges,
    )
    receipt_id = f"receipt-{traversal_hash[:32]}"
    return MemoryGraphExecutionReceipt(
        receipt_id=receipt_id,
        graph_id=graph.graph_id,
        graph_hash=graph.graph_hash,
        traversal_mode=traversal_mode,
        start_node_id=start,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        traversal_hash=traversal_hash,
    )


__all__ = (
    "MemoryGraphNode",
    "MemoryGraphEdge",
    "DeterministicMemoryGraph",
    "MemoryGraphValidationReport",
    "MemoryGraphExecutionReceipt",
    "normalize_deterministic_memory_graph",
    "validate_deterministic_memory_graph",
    "build_deterministic_memory_graph",
    "traverse_deterministic_memory_graph",
)
