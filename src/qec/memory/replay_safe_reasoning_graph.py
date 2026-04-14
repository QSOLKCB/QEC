"""v137.16.3 — Replay-Safe Reasoning Graph.

Deterministic reasoning-graph substrate over memory lineage, decision
dependencies, topology relationships, and release/proof/test/artifact traces.

Determinism law:
    same input = same bytes = same reasoning hash = same traversal hash.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union


_ALLOWED_REASONING_KINDS: Tuple[str, ...] = (
    "memory",
    "decision",
    "topology",
    "proof",
    "test",
    "release",
    "artifact",
)

_ALLOWED_RELATIONS: Tuple[str, ...] = (
    "supports",
    "depends_on",
    "verifies",
    "indexes",
    "replays",
    "derives_from",
)

_ALLOWED_TRAVERSAL_MODES: Tuple[str, ...] = (
    "reasoning",
    "dependency",
    "verification",
    "replay",
)

_RELATIONS_BY_TRAVERSAL: Dict[str, Tuple[str, ...]] = {
    "reasoning": _ALLOWED_RELATIONS,
    "dependency": ("depends_on", "supports", "derives_from"),
    "verification": ("verifies", "indexes"),
    "replay": ("replays", "derives_from"),
}

_LINEAGE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ReasoningGraphError(ValueError):
    """Structured reasoning-graph validation error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


NodeLike = Union["ReasoningGraphNode", Mapping[str, Any]]
EdgeLike = Union["ReasoningGraphEdge", Mapping[str, Any]]
GraphLike = Union["ReplaySafeReasoningGraph", Mapping[str, Any]]


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
class ReasoningGraphNode:
    node_id: str
    reasoning_kind: str
    source_ref: str
    lineage_hash: str
    reasoning_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "reasoning_kind": self.reasoning_kind,
            "source_ref": self.source_ref,
            "lineage_hash": self.lineage_hash,
            "reasoning_epoch": self.reasoning_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ReasoningGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    reasoning_relation: str
    edge_weight: float
    reasoning_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "reasoning_relation": self.reasoning_relation,
            "edge_weight": self.edge_weight,
            "reasoning_epoch": self.reasoning_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ReplaySafeReasoningGraph:
    graph_id: str
    nodes: Tuple[ReasoningGraphNode, ...]
    edges: Tuple[ReasoningGraphEdge, ...]
    reasoning_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "reasoning_hash": self.reasoning_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ReasoningGraphValidationReport:
    graph_id: str
    is_valid: bool
    node_count: int
    edge_count: int
    uniqueness_ok: bool
    node_validity_ok: bool
    edge_validity_ok: bool
    lineage_validity_ok: bool
    cross_artifact_reference_validity_ok: bool
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
            "lineage_validity_ok": self.lineage_validity_ok,
            "cross_artifact_reference_validity_ok": self.cross_artifact_reference_validity_ok,
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
class ReasoningGraphExecutionReceipt:
    receipt_id: str
    graph_id: str
    reasoning_hash: str
    traversal_mode: str
    visited_nodes: Tuple[str, ...]
    visited_edges: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "graph_id": self.graph_id,
            "reasoning_hash": self.reasoning_hash,
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


def _node_sort_key(node: ReasoningGraphNode) -> Tuple[int, str, str]:
    return (node.reasoning_epoch, node.reasoning_kind, node.node_id)


def _edge_sort_key(edge: ReasoningGraphEdge) -> Tuple[int, str, str]:
    return (edge.reasoning_epoch, edge.reasoning_relation, edge.edge_id)


def _collect_valid_source_refs(payload: Mapping[str, Any]) -> Tuple[str, ...]:
    refs: List[str] = []
    memory_graph = payload.get("memory_graph")
    if isinstance(memory_graph, Mapping):
        graph_id = memory_graph.get("graph_id")
        if isinstance(graph_id, str) and graph_id.strip():
            refs.append(f"memory_graph:{graph_id.strip()}")

    decision_dag = payload.get("decision_dag")
    if isinstance(decision_dag, Mapping):
        dag_id = decision_dag.get("dag_id")
        if isinstance(dag_id, str) and dag_id.strip():
            refs.append(f"decision_dag:{dag_id.strip()}")

    topology_index = payload.get("topology_index")
    if isinstance(topology_index, Mapping):
        index_id = topology_index.get("index_id")
        if isinstance(index_id, str) and index_id.strip():
            refs.append(f"topology_index:{index_id.strip()}")

    for trace_key, prefix in (
        ("release_traces", "release"),
        ("proof_traces", "proof"),
        ("test_traces", "test"),
        ("artifact_traces", "artifact"),
    ):
        traces = payload.get(trace_key)
        if isinstance(traces, Sequence):
            for item in traces:
                if isinstance(item, Mapping):
                    trace_id = item.get("trace_id")
                    if isinstance(trace_id, str) and trace_id.strip():
                        refs.append(f"{prefix}:{trace_id.strip()}")

    return tuple(sorted(set(refs)))


def _normalize_node(raw: NodeLike, valid_refs: Tuple[str, ...]) -> ReasoningGraphNode:
    if isinstance(raw, ReasoningGraphNode):
        node = raw
    else:
        if not isinstance(raw, Mapping):
            raise ReasoningGraphError("invalid_node", "node must be a mapping or ReasoningGraphNode")
        raw_epoch = raw.get("reasoning_epoch", 0)
        if isinstance(raw_epoch, bool):
            raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be an integer")
        try:
            reasoning_epoch = int(raw_epoch)
        except (TypeError, ValueError):
            raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be an integer")

        node = ReasoningGraphNode(
            node_id=str(raw.get("node_id", "")).strip(),
            reasoning_kind=str(raw.get("reasoning_kind", "")).strip(),
            source_ref=str(raw.get("source_ref", "")).strip(),
            lineage_hash=str(raw.get("lineage_hash", "")).strip(),
            reasoning_epoch=reasoning_epoch,
        )

    if not node.node_id:
        raise ReasoningGraphError("invalid_node_id", "invalid reasoning node id")
    if node.reasoning_kind not in _ALLOWED_REASONING_KINDS:
        raise ReasoningGraphError("invalid_reasoning_kind", "unsupported reasoning kind")
    if not _LINEAGE_HASH_RE.fullmatch(node.lineage_hash):
        raise ReasoningGraphError("invalid_lineage_hash", "malformed lineage hash")
    if node.reasoning_epoch < 0:
        raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be non-negative")
    if node.source_ref not in valid_refs:
        raise ReasoningGraphError("invalid_cross_artifact_reference", "invalid cross-artifact reference")
    return node


def _normalize_edge(raw: EdgeLike) -> ReasoningGraphEdge:
    if isinstance(raw, ReasoningGraphEdge):
        edge = raw
    else:
        if not isinstance(raw, Mapping):
            raise ReasoningGraphError("invalid_edge", "edge must be a mapping or ReasoningGraphEdge")
        raw_weight = raw.get("edge_weight", 0.0)
        if isinstance(raw_weight, bool):
            raise ReasoningGraphError("invalid_edge_weight", "edge_weight must be numeric")
        try:
            edge_weight = float(raw_weight)
        except (TypeError, ValueError):
            raise ReasoningGraphError("invalid_edge_weight", "edge_weight must be numeric")
        if not math.isfinite(edge_weight):
            raise ReasoningGraphError("invalid_edge_weight", "edge_weight must be finite")

        raw_epoch = raw.get("reasoning_epoch", 0)
        if isinstance(raw_epoch, bool):
            raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be an integer")
        try:
            reasoning_epoch = int(raw_epoch)
        except (TypeError, ValueError):
            raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be an integer")

        edge = ReasoningGraphEdge(
            edge_id=str(raw.get("edge_id", "")).strip(),
            source_node_id=str(raw.get("source_node_id", "")).strip(),
            target_node_id=str(raw.get("target_node_id", "")).strip(),
            reasoning_relation=str(raw.get("reasoning_relation", "")).strip(),
            edge_weight=edge_weight,
            reasoning_epoch=reasoning_epoch,
        )

    if not edge.edge_id:
        raise ReasoningGraphError("invalid_edge_id", "invalid reasoning edge id")
    if not edge.source_node_id or not edge.target_node_id:
        raise ReasoningGraphError("invalid_edge_reference", "invalid edge node reference")
    if edge.reasoning_relation not in _ALLOWED_RELATIONS:
        raise ReasoningGraphError("invalid_relation", "unsupported relation")
    if edge.edge_weight < 0.0:
        raise ReasoningGraphError("negative_edge_weight", "negative weights are not allowed")
    if edge.reasoning_epoch < 0:
        raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be non-negative")
    return edge


def _validate_built_graph_object(graph: ReplaySafeReasoningGraph) -> None:
    graph_id = str(graph.graph_id).strip()
    if not graph_id:
        raise ReasoningGraphError("invalid_graph_id", "invalid graph id")

    normalized_nodes: List[ReasoningGraphNode] = []
    seen_node_ids = set()
    for raw_node in graph.nodes:
        node = raw_node
        if not node.source_ref:
            raise ReasoningGraphError("invalid_node", "source_ref must be non-empty")
        if not node.node_id:
            raise ReasoningGraphError("invalid_node_id", "invalid reasoning node id")
        if node.reasoning_kind not in _ALLOWED_REASONING_KINDS:
            raise ReasoningGraphError("invalid_reasoning_kind", "unsupported reasoning kind")
        if not _LINEAGE_HASH_RE.fullmatch(node.lineage_hash):
            raise ReasoningGraphError("invalid_lineage_hash", "malformed lineage hash")
        if node.reasoning_epoch < 0:
            raise ReasoningGraphError("invalid_reasoning_epoch", "reasoning_epoch must be non-negative")
        if node.node_id in seen_node_ids:
            raise ReasoningGraphError("duplicate_node", f"duplicate reasoning node id: {node.node_id}")
        seen_node_ids.add(node.node_id)
        normalized_nodes.append(node)

    normalized_edges: List[ReasoningGraphEdge] = []
    seen_edge_ids = set()
    node_ids = {node.node_id for node in normalized_nodes}
    for raw_edge in graph.edges:
        edge = _normalize_edge(raw_edge)
        if edge.edge_id in seen_edge_ids:
            raise ReasoningGraphError("duplicate_edge", f"duplicate reasoning edge id: {edge.edge_id}")
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise ReasoningGraphError("invalid_edge_reference", "invalid edge node reference")
        seen_edge_ids.add(edge.edge_id)
        normalized_edges.append(edge)

    sorted_nodes = tuple(sorted(normalized_nodes, key=_node_sort_key))
    sorted_edges = tuple(sorted(normalized_edges, key=_edge_sort_key))
    material = {
        "graph_id": graph_id,
        "nodes": [node.to_dict() for node in sorted_nodes],
        "edges": [edge.to_dict() for edge in sorted_edges],
    }
    expected_reasoning_hash = _sha256_hex(_canonical_bytes(material))
    if graph.reasoning_hash != expected_reasoning_hash:
        raise ReasoningGraphError("invalid_reasoning_hash", "reasoning hash mismatch")

    for mode in _ALLOWED_TRAVERSAL_MODES:
        first = traverse_replay_safe_reasoning_graph(graph, mode)
        second = traverse_replay_safe_reasoning_graph(graph, mode)
        if first.to_canonical_bytes() != second.to_canonical_bytes():
            raise ReasoningGraphError("invalid_traversal_invariant", "non-deterministic traversal receipt")


def normalize_replay_safe_reasoning_input(payload: Mapping[str, Any]) -> Tuple[str, Tuple[ReasoningGraphNode, ...], Tuple[ReasoningGraphEdge, ...]]:
    if not isinstance(payload, Mapping):
        raise ReasoningGraphError("invalid_graph", "reasoning graph input must be a mapping")

    graph_id = str(payload.get("graph_id", "")).strip()
    if not graph_id:
        raise ReasoningGraphError("invalid_graph_id", "invalid graph id")

    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, (str, bytes, bytearray)):
        raise ReasoningGraphError("invalid_nodes", "nodes must be a sequence")
    if not isinstance(raw_edges, Sequence) or isinstance(raw_edges, (str, bytes, bytearray)):
        raise ReasoningGraphError("invalid_edges", "edges must be a sequence")

    valid_refs = _collect_valid_source_refs(payload)

    nodes: List[ReasoningGraphNode] = []
    seen_node_ids = set()
    for raw_node in raw_nodes:
        node = _normalize_node(raw_node, valid_refs)
        if node.node_id in seen_node_ids:
            raise ReasoningGraphError("duplicate_node", f"duplicate reasoning node id: {node.node_id}")
        seen_node_ids.add(node.node_id)
        nodes.append(node)

    edges: List[ReasoningGraphEdge] = []
    seen_edge_ids = set()
    node_ids = {n.node_id for n in nodes}
    for raw_edge in raw_edges:
        edge = _normalize_edge(raw_edge)
        if edge.edge_id in seen_edge_ids:
            raise ReasoningGraphError("duplicate_edge", f"duplicate reasoning edge id: {edge.edge_id}")
        if edge.source_node_id not in node_ids or edge.target_node_id not in node_ids:
            raise ReasoningGraphError(
                "invalid_edge_reference",
                (
                    f"edge references missing node: {edge.edge_id} "
                    f"({edge.source_node_id} -> {edge.target_node_id})"
                ),
            )
        seen_edge_ids.add(edge.edge_id)
        edges.append(edge)

    nodes_sorted = tuple(sorted(nodes, key=_node_sort_key))
    edges_sorted = tuple(sorted(edges, key=_edge_sort_key))
    return graph_id, nodes_sorted, edges_sorted


def build_replay_safe_reasoning_graph(payload: GraphLike) -> ReplaySafeReasoningGraph:
    if isinstance(payload, ReplaySafeReasoningGraph):
        graph_id = payload.graph_id
        nodes = tuple(sorted(payload.nodes, key=_node_sort_key))
        edges = tuple(sorted(payload.edges, key=_edge_sort_key))
    else:
        graph_id, nodes, edges = normalize_replay_safe_reasoning_input(payload)

    material = {
        "graph_id": graph_id,
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
    }
    reasoning_hash = _sha256_hex(_canonical_bytes(material))
    return ReplaySafeReasoningGraph(
        graph_id=graph_id,
        nodes=nodes,
        edges=edges,
        reasoning_hash=reasoning_hash,
    )


def validate_replay_safe_reasoning_graph(payload: GraphLike) -> ReasoningGraphValidationReport:
    graph_id = ""
    node_count = 0
    edge_count = 0
    violations: List[str] = []

    uniqueness_ok = True
    node_validity_ok = True
    edge_validity_ok = True
    lineage_validity_ok = True
    cross_artifact_reference_validity_ok = True
    weight_validity_ok = True

    try:
        if isinstance(payload, ReplaySafeReasoningGraph):
            graph_id = payload.graph_id
            node_count = len(payload.nodes)
            edge_count = len(payload.edges)
            _validate_built_graph_object(payload)
        else:
            graph_id = str(payload.get("graph_id", "")).strip() if isinstance(payload, Mapping) else ""
            raw_nodes = payload.get("nodes", ()) if isinstance(payload, Mapping) else ()
            raw_edges = payload.get("edges", ()) if isinstance(payload, Mapping) else ()
            node_count = len(raw_nodes) if isinstance(raw_nodes, Sequence) and not isinstance(raw_nodes, (str, bytes, bytearray)) else 0
            edge_count = len(raw_edges) if isinstance(raw_edges, Sequence) and not isinstance(raw_edges, (str, bytes, bytearray)) else 0
            normalize_replay_safe_reasoning_input(payload)
    except ReasoningGraphError as exc:
        violations.append(str(exc))
        if exc.code in ("duplicate_node", "duplicate_edge"):
            uniqueness_ok = False
        if exc.code in ("invalid_node", "invalid_node_id", "invalid_reasoning_kind", "invalid_reasoning_epoch"):
            node_validity_ok = False
        if exc.code in ("invalid_edge", "invalid_edge_id", "invalid_edge_reference", "invalid_relation", "invalid_reasoning_epoch", "invalid_traversal_invariant"):
            edge_validity_ok = False
        if exc.code == "invalid_lineage_hash":
            lineage_validity_ok = False
        if exc.code == "invalid_cross_artifact_reference":
            cross_artifact_reference_validity_ok = False
        if exc.code in ("negative_edge_weight", "invalid_edge_weight"):
            weight_validity_ok = False

    is_valid = (
        uniqueness_ok
        and node_validity_ok
        and edge_validity_ok
        and lineage_validity_ok
        and cross_artifact_reference_validity_ok
        and weight_validity_ok
        and not violations
    )

    return ReasoningGraphValidationReport(
        graph_id=graph_id,
        is_valid=is_valid,
        node_count=node_count,
        edge_count=edge_count,
        uniqueness_ok=uniqueness_ok,
        node_validity_ok=node_validity_ok,
        edge_validity_ok=edge_validity_ok,
        lineage_validity_ok=lineage_validity_ok,
        cross_artifact_reference_validity_ok=cross_artifact_reference_validity_ok,
        weight_validity_ok=weight_validity_ok,
        violations=tuple(violations),
    )


def traverse_replay_safe_reasoning_graph(
    payload: GraphLike,
    traversal_mode: str = "reasoning",
) -> ReasoningGraphExecutionReceipt:
    graph = build_replay_safe_reasoning_graph(payload)
    mode = str(traversal_mode).strip()
    if mode not in _ALLOWED_TRAVERSAL_MODES:
        raise ReasoningGraphError("invalid_traversal_mode", "unsupported traversal mode")

    allowed_relations = set(_RELATIONS_BY_TRAVERSAL[mode])
    mode_edges = tuple(edge for edge in graph.edges if edge.reasoning_relation in allowed_relations)
    visited_edges = tuple(edge.edge_id for edge in mode_edges)

    if mode == "reasoning":
        visited_nodes = tuple(node.node_id for node in graph.nodes)
    else:
        edge_node_ids = {
            edge.source_node_id for edge in mode_edges
        } | {
            edge.target_node_id for edge in mode_edges
        }
        visited_nodes = tuple(
            node.node_id for node in graph.nodes if node.node_id in edge_node_ids
        )

    traversal_payload = {
        "graph_id": graph.graph_id,
        "reasoning_hash": graph.reasoning_hash,
        "traversal_mode": mode,
        "visited_nodes": list(visited_nodes),
        "visited_edges": list(visited_edges),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(traversal_payload))
    receipt_id = _sha256_hex(
        _canonical_bytes(
            {
                "graph_id": graph.graph_id,
                "traversal_mode": mode,
                "traversal_hash": traversal_hash,
            }
        )
    )

    return ReasoningGraphExecutionReceipt(
        receipt_id=receipt_id,
        graph_id=graph.graph_id,
        reasoning_hash=graph.reasoning_hash,
        traversal_mode=mode,
        visited_nodes=visited_nodes,
        visited_edges=visited_edges,
        traversal_hash=traversal_hash,
    )
