"""v137.16.2 — Codebase Topology Indexer.

Extends v137.16.0 (memory graph) and v137.16.1 (decision DAG compiler)
with a deterministic topology index over codebase structure + lineage.

Determinism law:
    same input = same bytes = same index hash = same traversal hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple, Union

from qec.memory.decision_dag_compiler import CompiledDecisionDAG, DecisionDAGNode
from qec.memory.deterministic_memory_graph_kernel import DeterministicMemoryGraph


_ALLOWED_NODE_KINDS: Tuple[str, ...] = (
    "module",
    "package",
    "release",
    "proof",
    "test",
    "artifact",
)

_ALLOWED_RELATIONSHIP_KINDS: Tuple[str, ...] = (
    "imports",
    "depends_on",
    "tested_by",
    "verified_by",
    "supersedes",
    "belongs_to",
)

_ALLOWED_TRAVERSAL_MODES: Tuple[str, ...] = (
    "hierarchy",
    "dependency",
    "lineage",
    "coverage",
)

_RELATIONSHIPS_BY_TRAVERSAL: Dict[str, Tuple[str, ...]] = {
    "hierarchy": ("belongs_to",),
    "dependency": ("imports", "depends_on"),
    "lineage": ("supersedes",),
    "coverage": ("tested_by", "verified_by"),
}

_LINEAGE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class TopologyIndexError(ValueError):
    """Structured topology-indexing error carrying stable code tags."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


NodeLike = Union["TopologyIndexNode", Mapping[str, Any]]
EdgeLike = Union["TopologyIndexEdge", Mapping[str, Any]]


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


@dataclass(frozen=True)
class TopologyIndexNode:
    node_id: str
    node_kind: str
    module_path: str
    lineage_hash: str
    index_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "module_path": self.module_path,
            "lineage_hash": self.lineage_hash,
            "index_epoch": self.index_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class TopologyIndexEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_kind: str
    edge_weight: float
    index_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "relationship_kind": self.relationship_kind,
            "edge_weight": self.edge_weight,
            "index_epoch": self.index_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class CodebaseTopologyIndex:
    index_id: str
    nodes: Tuple[TopologyIndexNode, ...]
    edges: Tuple[TopologyIndexEdge, ...]
    index_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_id": self.index_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "index_hash": self.index_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class TopologyIndexValidationReport:
    index_id: str
    is_valid: bool
    node_count: int
    edge_count: int
    uniqueness_ok: bool
    node_validity_ok: bool
    edge_validity_ok: bool
    lineage_validity_ok: bool
    hierarchy_validity_ok: bool
    weight_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index_id": self.index_id,
            "is_valid": self.is_valid,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "uniqueness_ok": self.uniqueness_ok,
            "node_validity_ok": self.node_validity_ok,
            "edge_validity_ok": self.edge_validity_ok,
            "lineage_validity_ok": self.lineage_validity_ok,
            "hierarchy_validity_ok": self.hierarchy_validity_ok,
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
class TopologyIndexExecutionReceipt:
    receipt_id: str
    index_id: str
    index_hash: str
    traversal_mode: str
    visited_nodes: Tuple[str, ...]
    visited_edges: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "index_id": self.index_id,
            "index_hash": self.index_hash,
            "traversal_mode": self.traversal_mode,
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


def _node_sort_key(node: TopologyIndexNode) -> Tuple[int, str, str]:
    return (node.index_epoch, node.node_kind, node.node_id)


def _edge_sort_key(edge: TopologyIndexEdge) -> Tuple[int, str, str]:
    return (edge.index_epoch, edge.relationship_kind, edge.edge_id)


def _normalize_node(raw: NodeLike) -> TopologyIndexNode:
    if isinstance(raw, TopologyIndexNode):
        node = raw
    else:
        node = TopologyIndexNode(
            node_id=str(raw.get("node_id", "")).strip(),
            node_kind=str(raw.get("node_kind", "")).strip(),
            module_path=str(raw.get("module_path", "")).strip(),
            lineage_hash=str(raw.get("lineage_hash", "")).strip(),
            index_epoch=int(raw.get("index_epoch", 0)),
        )
    if not node.node_id:
        raise TopologyIndexError("invalid_node_id", "invalid topology node id")
    if node.node_kind not in _ALLOWED_NODE_KINDS:
        raise TopologyIndexError("invalid_node_kind", "unsupported topology node kind")
    if not node.module_path:
        raise TopologyIndexError("invalid_module_path", "invalid module path")
    if not _LINEAGE_HASH_RE.fullmatch(node.lineage_hash):
        raise TopologyIndexError("invalid_lineage_hash", "malformed lineage hash")
    if node.index_epoch < 0:
        raise TopologyIndexError("invalid_index_epoch", "index epoch must be non-negative")
    return node


def _normalize_edge(raw: EdgeLike) -> TopologyIndexEdge:
    if isinstance(raw, TopologyIndexEdge):
        edge = raw
    else:
        edge = TopologyIndexEdge(
            edge_id=str(raw.get("edge_id", "")).strip(),
            source_node_id=str(raw.get("source_node_id", "")).strip(),
            target_node_id=str(raw.get("target_node_id", "")).strip(),
            relationship_kind=str(raw.get("relationship_kind", "")).strip(),
            edge_weight=float(raw.get("edge_weight", 0.0)),
            index_epoch=int(raw.get("index_epoch", 0)),
        )
    if not edge.edge_id:
        raise TopologyIndexError("invalid_edge_id", "invalid topology edge id")
    if not edge.source_node_id or not edge.target_node_id:
        raise TopologyIndexError("invalid_edge_reference", "invalid edge node reference")
    if edge.relationship_kind not in _ALLOWED_RELATIONSHIP_KINDS:
        raise TopologyIndexError("invalid_relationship_kind", "unsupported relationship kind")
    if edge.edge_weight < 0.0:
        raise TopologyIndexError("negative_edge_weight", "negative edge weight")
    if edge.index_epoch < 0:
        raise TopologyIndexError("invalid_index_epoch", "index epoch must be non-negative")
    return edge


def _coerce_inputs(*args: Any, **kwargs: Any) -> Tuple[str, Any, Any]:
    if len(args) == 1 and isinstance(args[0], Mapping):
        payload = args[0]
        return (
            str(payload.get("index_id", "codebase_topology_index")).strip(),
            payload.get("nodes", ()),
            payload.get("edges", ()),
        )
    if len(args) >= 2:
        index_id = str(kwargs.get("index_id", "codebase_topology_index")).strip()
        return index_id, args[0], args[1]
    raise TopologyIndexError("invalid_topology_input", "invalid topology input")


def _validate_raw_nodes(raw_nodes: Any) -> None:
    """Validate raw node input for deterministic normalization errors."""
    if not isinstance(raw_nodes, (list, tuple)):
        raise TopologyIndexError("invalid_nodes", "nodes must be a list or tuple")
    for index, node in enumerate(raw_nodes):
        if not isinstance(node, (TopologyIndexNode, Mapping)):
            raise TopologyIndexError(
                "invalid_node",
                f"node at index {index} must be a mapping or TopologyIndexNode",
            )


def _validate_raw_edges(raw_edges: Any) -> None:
    """Validate raw edge input for deterministic normalization errors."""
    if not isinstance(raw_edges, (list, tuple)):
        raise TopologyIndexError("invalid_edges", "edges must be a list or tuple")
    for index, edge in enumerate(raw_edges):
        if not isinstance(edge, (TopologyIndexEdge, Mapping)):
            raise TopologyIndexError(
                "invalid_edge",
                f"edge at index {index} must be a mapping or TopologyIndexEdge",
            )


def normalize_codebase_topology_input(*args: Any, **kwargs: Any) -> Tuple[str, Tuple[TopologyIndexNode, ...], Tuple[TopologyIndexEdge, ...]]:
    index_id, raw_nodes, raw_edges = _coerce_inputs(*args, **kwargs)

    if not index_id:
        raise TopologyIndexError("invalid_index_id", "invalid topology index id")

    _validate_raw_nodes(raw_nodes)
    _validate_raw_edges(raw_edges)
    nodes = tuple(sorted((_normalize_node(n) for n in raw_nodes), key=_node_sort_key))
    edges = tuple(sorted((_normalize_edge(e) for e in raw_edges), key=_edge_sort_key))

    node_ids: Set[str] = set()
    for node in nodes:
        if node.node_id in node_ids:
            raise TopologyIndexError("duplicate_node", f"duplicate topology node id: {node.node_id}")
        node_ids.add(node.node_id)

    edge_ids: Set[str] = set()
    for edge in edges:
        if edge.edge_id in edge_ids:
            raise TopologyIndexError("duplicate_edge", f"duplicate topology edge id: {edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise TopologyIndexError("invalid_edge_reference", "edge references unknown node")

    for edge in edges:
        if edge.relationship_kind == "belongs_to":
            source_kind = next(n.node_kind for n in nodes if n.node_id == edge.source_node_id)
            target_kind = next(n.node_kind for n in nodes if n.node_id == edge.target_node_id)
            if target_kind != "package" or source_kind == "release":
                raise TopologyIndexError("invalid_hierarchy_reference", "invalid hierarchy relationship")

    return index_id, nodes, edges


def build_codebase_topology_index(*args: Any, **kwargs: Any) -> CodebaseTopologyIndex:
    index_id, nodes, edges = normalize_codebase_topology_input(*args, **kwargs)
    payload = {
        "index_id": index_id,
        "nodes": [n.to_dict() for n in nodes],
        "edges": [e.to_dict() for e in edges],
    }
    index_hash = _sha256_hex(_canonical_bytes(payload))
    return CodebaseTopologyIndex(index_id=index_id, nodes=nodes, edges=edges, index_hash=index_hash)


def validate_codebase_topology_index(value: Union[CodebaseTopologyIndex, Mapping[str, Any]]) -> TopologyIndexValidationReport:
    violations: List[str] = []
    try:
        if isinstance(value, CodebaseTopologyIndex):
            index_id, nodes, edges = value.index_id, value.nodes, value.edges
        else:
            index_id, nodes, edges = normalize_codebase_topology_input(value)
        normalize_codebase_topology_input({"index_id": index_id, "nodes": nodes, "edges": edges})
    except TopologyIndexError as exc:
        index_id = ""
        if isinstance(value, Mapping):
            index_id = str(value.get("index_id", ""))
        violations.append(exc.code)
        nodes = ()
        edges = ()

    uniqueness_ok = "duplicate_node" not in violations and "duplicate_edge" not in violations
    node_validity_ok = not any(v.startswith("invalid_node") for v in violations)
    edge_validity_ok = not any(v in {"invalid_edge_reference", "invalid_edge_id", "invalid_relationship_kind"} for v in violations)
    lineage_validity_ok = "invalid_lineage_hash" not in violations
    hierarchy_validity_ok = "invalid_hierarchy_reference" not in violations
    weight_validity_ok = "negative_edge_weight" not in violations
    is_valid = not violations

    return TopologyIndexValidationReport(
        index_id=index_id,
        is_valid=is_valid,
        node_count=len(nodes),
        edge_count=len(edges),
        uniqueness_ok=uniqueness_ok,
        node_validity_ok=node_validity_ok,
        edge_validity_ok=edge_validity_ok,
        lineage_validity_ok=lineage_validity_ok,
        hierarchy_validity_ok=hierarchy_validity_ok,
        weight_validity_ok=weight_validity_ok,
        violations=tuple(sorted(set(violations))),
    )


def traverse_codebase_topology_index(
    value: Union[CodebaseTopologyIndex, Mapping[str, Any]],
    traversal_mode: str,
) -> TopologyIndexExecutionReceipt:
    if traversal_mode not in _ALLOWED_TRAVERSAL_MODES:
        raise TopologyIndexError("invalid_traversal_mode", "unsupported traversal mode")

    if isinstance(value, CodebaseTopologyIndex):
        index_obj = value
    else:
        index_obj = build_codebase_topology_index(value)

    allowed_relationships = _RELATIONSHIPS_BY_TRAVERSAL[traversal_mode]

    adjacency: Dict[str, List[Tuple[str, str]]] = {n.node_id: [] for n in index_obj.nodes}
    candidate_edges = [e for e in index_obj.edges if e.relationship_kind in allowed_relationships]
    for edge in sorted(candidate_edges, key=_edge_sort_key):
        adjacency[edge.source_node_id].append((edge.target_node_id, edge.edge_id))

    visited_nodes: List[str] = []
    visited_edges: List[str] = []
    seen_nodes: Set[str] = set()

    for root in sorted(index_obj.nodes, key=_node_sort_key):
        if root.node_id in seen_nodes:
            continue
        stack: List[str] = [root.node_id]
        while stack:
            current = stack.pop(0)
            if current in seen_nodes:
                continue
            seen_nodes.add(current)
            visited_nodes.append(current)
            outgoing = adjacency.get(current, [])
            for target_id, edge_id in outgoing:
                if edge_id not in visited_edges:
                    visited_edges.append(edge_id)
                if target_id not in seen_nodes and target_id not in stack:
                    stack.append(target_id)

    receipt_seed = {
        "index_id": index_obj.index_id,
        "index_hash": index_obj.index_hash,
        "traversal_mode": traversal_mode,
        "visited_nodes": visited_nodes,
        "visited_edges": visited_edges,
    }
    traversal_hash = _sha256_hex(_canonical_bytes(receipt_seed))
    receipt_id = _sha256_hex(_canonical_bytes({"receipt": receipt_seed, "traversal_hash": traversal_hash}))

    return TopologyIndexExecutionReceipt(
        receipt_id=receipt_id,
        index_id=index_obj.index_id,
        index_hash=index_obj.index_hash,
        traversal_mode=traversal_mode,
        visited_nodes=tuple(visited_nodes),
        visited_edges=tuple(visited_edges),
        traversal_hash=traversal_hash,
    )

