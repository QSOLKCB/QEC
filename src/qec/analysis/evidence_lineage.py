"""v137.10.2 — Evidence Lineage Engine.

Deterministic Layer-4 evidence lineage and provenance artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

EVIDENCE_LINEAGE_SCHEMA_VERSION = 1

_ALLOWED_NODE_TYPES: tuple[str, ...] = (
    "claim",
    "experiment",
    "measurement",
    "criterion",
    "result",
    "receipt",
)

_ALLOWED_EDGE_TYPES: tuple[str, ...] = (
    "supports",
    "derives_from",
    "validates",
    "produced_by",
    "linked_to",
)



def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("metadata keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")



def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )



def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")



def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class EvidenceNode:
    node_id: str
    node_type: str
    source_hash: str
    source_kind: str
    metadata: Mapping[str, Any]
    linked_experiment_hash: str
    linked_measurement_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "source_hash": self.source_hash,
            "source_kind": self.source_kind,
            "metadata": _canonicalize_json(self.metadata),
            "linked_experiment_hash": self.linked_experiment_hash,
            "linked_measurement_ids": tuple(self.linked_measurement_ids),
        }


@dataclass(frozen=True)
class EvidenceEdge:
    from_node_id: str
    to_node_id: str
    relation_type: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "relation_type": self.relation_type,
        }


@dataclass(frozen=True)
class EvidenceLineageGraph:
    graph_id: str
    claim_id: str
    experiment_hash: str
    nodes: tuple[EvidenceNode, ...]
    edges: tuple[EvidenceEdge, ...]
    schema_version: int = EVIDENCE_LINEAGE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "graph_id": self.graph_id,
            "claim_id": self.claim_id,
            "experiment_hash": self.experiment_hash,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class EvidenceLineageReceipt:
    graph_hash: str
    node_count: int
    edge_count: int
    claim_id: str
    experiment_hash: str
    byte_length: int
    validation_passed: bool

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "graph_hash": self.graph_hash,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "claim_id": self.claim_id,
            "experiment_hash": self.experiment_hash,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")



def normalize_evidence_graph(raw_graph: Mapping[str, Any]) -> EvidenceLineageGraph:
    if not isinstance(raw_graph, Mapping):
        raise ValueError("raw_graph must be a mapping")

    claim_id = str(raw_graph.get("claim_id", ""))
    experiment_hash = str(raw_graph.get("experiment_hash", ""))
    graph_id = str(raw_graph.get("graph_id", ""))

    raw_nodes = raw_graph.get("nodes", ())
    if not isinstance(raw_nodes, (tuple, list)):
        raise ValueError("nodes must be a tuple/list")

    nodes: list[EvidenceNode] = []
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise ValueError("each node must be a mapping")
        linked_measurement_ids = raw.get("linked_measurement_ids", ())
        if not isinstance(linked_measurement_ids, (tuple, list)):
            raise ValueError("linked_measurement_ids must be a tuple/list")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        nodes.append(
            EvidenceNode(
                node_id=str(raw.get("node_id", "")),
                node_type=str(raw.get("node_type", "")),
                source_hash=str(raw.get("source_hash", "")),
                source_kind=str(raw.get("source_kind", "")),
                metadata=dict(_canonicalize_json(metadata)),
                linked_experiment_hash=str(raw.get("linked_experiment_hash", "")),
                linked_measurement_ids=tuple(str(v) for v in linked_measurement_ids),
            )
        )

    raw_edges = raw_graph.get("edges", ())
    if not isinstance(raw_edges, (tuple, list)):
        raise ValueError("edges must be a tuple/list")

    edges: list[EvidenceEdge] = []
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            raise ValueError("each edge must be a mapping")
        edges.append(
            EvidenceEdge(
                from_node_id=str(raw.get("from_node_id", "")),
                to_node_id=str(raw.get("to_node_id", "")),
                relation_type=str(raw.get("relation_type", "")),
            )
        )

    nodes_sorted = tuple(sorted(nodes, key=lambda n: (n.node_type, n.node_id)))
    edges_sorted = tuple(sorted(edges, key=lambda e: (e.from_node_id, e.to_node_id, e.relation_type)))

    if not graph_id:
        graph_id = _sha256_hex(
            {
                "claim_id": claim_id,
                "experiment_hash": experiment_hash,
                "nodes": tuple(node.to_dict() for node in nodes_sorted),
                "edges": tuple(edge.to_dict() for edge in edges_sorted),
                "schema_version": EVIDENCE_LINEAGE_SCHEMA_VERSION,
            }
        )

    return EvidenceLineageGraph(
        graph_id=graph_id,
        claim_id=claim_id,
        experiment_hash=experiment_hash,
        nodes=nodes_sorted,
        edges=edges_sorted,
        schema_version=int(raw_graph.get("schema_version", EVIDENCE_LINEAGE_SCHEMA_VERSION)),
    )



def validate_evidence_graph(graph: EvidenceLineageGraph) -> None:
    if not graph.claim_id:
        raise ValueError("claim_id must be non-empty")
    if not graph.experiment_hash:
        raise ValueError("experiment_hash must be non-empty")

    node_ids = tuple(node.node_id for node in graph.nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node IDs are not allowed")

    node_id_set = set(node_ids)
    measurement_ids = {node.node_id for node in graph.nodes if node.node_type == "measurement"}

    for node in graph.nodes:
        if node.node_type not in _ALLOWED_NODE_TYPES:
            raise ValueError(f"unsupported node type: {node.node_type}")
        if node.node_type == "measurement":
            if node.linked_experiment_hash != graph.experiment_hash:
                raise ValueError("measurement node linked_experiment_hash must match experiment_hash")
            for measurement_id in node.linked_measurement_ids:
                if measurement_id not in measurement_ids:
                    raise ValueError("measurement node linked_measurement_ids must reference existing measurement IDs")

    edge_keys = tuple((e.from_node_id, e.to_node_id, e.relation_type) for e in graph.edges)
    if len(set(edge_keys)) != len(edge_keys):
        raise ValueError("duplicate edges are not allowed")

    for edge in graph.edges:
        if edge.relation_type not in _ALLOWED_EDGE_TYPES:
            raise ValueError(f"unsupported edge type: {edge.relation_type}")
        if edge.from_node_id not in node_id_set or edge.to_node_id not in node_id_set:
            raise ValueError("edge references unknown node IDs")
        if edge.from_node_id == edge.to_node_id:
            from_node = next(node for node in graph.nodes if node.node_id == edge.from_node_id)
            to_node = next(node for node in graph.nodes if node.node_id == edge.to_node_id)
            if not (from_node.node_type == "receipt" and to_node.node_type == "receipt"):
                raise ValueError("self-loop edges are only allowed for receipt-linked nodes")



def _stable_evidence_hash_payload(graph: EvidenceLineageGraph) -> dict[str, Any]:
    """Return the deterministic payload used for stable evidence hashing."""
    payload = graph.to_dict().copy()
    payload.pop("graph_id", None)
    return payload


def stable_evidence_hash(graph: EvidenceLineageGraph) -> str:
    return _sha256_hex(_stable_evidence_hash_payload(graph))
def build_evidence_receipt(graph: EvidenceLineageGraph) -> EvidenceLineageReceipt:
    validate_evidence_graph(graph)
    canonical_bytes = graph.to_canonical_bytes()
    return EvidenceLineageReceipt(
        graph_hash=stable_evidence_hash(graph),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        claim_id=graph.claim_id,
        experiment_hash=graph.experiment_hash,
        byte_length=len(canonical_bytes),
        validation_passed=True,
    )



def compile_evidence_graph(raw_graph: Mapping[str, Any]) -> tuple[EvidenceLineageGraph, EvidenceLineageReceipt]:
    graph = normalize_evidence_graph(raw_graph)
    receipt = build_evidence_receipt(graph)
    return graph, receipt
