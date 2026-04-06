"""v137.8.0 — Topological Graph Kernel.

Deterministic Layer-4 graph substrate for replay-safe topology artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.fragmentation_recovery_engine import FragmentationRecoveryArtifact
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, CompressionRecord

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

TOPOLOGICAL_GRAPH_KERNEL_LAW = "TOPOLOGICAL_GRAPH_KERNEL_LAW"
DETERMINISTIC_NODE_ORDERING_RULE = "DETERMINISTIC_NODE_ORDERING_RULE"
DETERMINISTIC_EDGE_ORDERING_RULE = "DETERMINISTIC_EDGE_ORDERING_RULE"
REPLAY_SAFE_GRAPH_IDENTITY_RULE = "REPLAY_SAFE_GRAPH_IDENTITY_RULE"


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
            raise ValueError("payload keys must be strings")
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


def _clamp01(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError("score must be float")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, value))


def _safe_fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return _clamp01(float(numerator / denominator))


def _validate_source_artifact(artifact: CompressedMemoryArtifact) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("source_artifact must be a CompressedMemoryArtifact")
    if artifact.compressed_record_count != len(artifact.records):
        raise ValueError("source_artifact compressed_record_count must match records length")
    if artifact.source_theme_count != len(artifact.records):
        raise ValueError("source_artifact source_theme_count must match records length")
    records = artifact.records
    if not records:
        raise ValueError("source_artifact must contain at least one CompressionRecord")
    if artifact.compression_chain_head != records[-1].source_replay_identity_hash:
        raise ValueError("source_artifact compression_chain_head must match last replay identity")
    for idx, record in enumerate(records):
        if record.theme_index != idx:
            raise ValueError("source_artifact records must be contiguous in theme_index")
        if idx > 0 and record.source_parent_theme_hash != records[idx - 1].source_replay_identity_hash:
            raise ValueError("source_artifact contains invalid lineage structure")
    return records


def _validate_recovery_artifact(
    artifact: FragmentationRecoveryArtifact,
    *,
    source_artifact: CompressedMemoryArtifact,
    source_records: tuple[CompressionRecord, ...],
) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, FragmentationRecoveryArtifact):
        raise ValueError("recovery_artifact must be a FragmentationRecoveryArtifact")
    if artifact.source_compression_hash != source_artifact.compression_hash:
        raise ValueError("recovery_artifact source_compression_hash must match source_artifact")
    if artifact.source_replay_identity_hash != source_artifact.replay_identity_hash:
        raise ValueError("recovery_artifact source_replay_identity_hash must match source_artifact")
    if len(artifact.repaired_records) != len(source_records):
        raise ValueError("recovery_artifact repaired_records must match source record count")
    if artifact.repaired_chain_head != artifact.repaired_records[-1].source_replay_identity_hash:
        raise ValueError("recovery_artifact repaired_chain_head must match final repaired record")
    return artifact.repaired_records


def _node_id(record: CompressionRecord) -> str:
    return _sha256_hex(
        {
            "theme_index": record.theme_index,
            "theme_hash": record.source_theme_hash,
            "replay_identity_hash": record.source_replay_identity_hash,
            "node_type": "memory_theme",
        }
    )


@dataclass(frozen=True)
class TopologicalGraphNode:
    node_id: str
    theme_index: int
    theme_hash: str
    replay_identity_hash: str
    node_type: str
    lineage_parent_id: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "theme_index": self.theme_index,
            "theme_hash": self.theme_hash,
            "replay_identity_hash": self.replay_identity_hash,
            "node_type": self.node_type,
            "lineage_parent_id": self.lineage_parent_id,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TopologicalGraphEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    continuity_weight: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type,
            "continuity_weight": self.continuity_weight,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TopologicalGraphKernelResult:
    schema_version: int
    source_compression_hash: str
    source_replay_identity_hash: str
    recovered_chain_head: str
    node_count: int
    edge_count: int
    nodes: tuple[TopologicalGraphNode, ...]
    edges: tuple[TopologicalGraphEdge, ...]
    connectivity_score: float
    continuity_graph_score: float
    lineage_integrity_score: float
    overall_topology_score: float
    law_invariants: tuple[str, ...]
    graph_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "recovered_chain_head": self.recovered_chain_head,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "edges": tuple(edge.to_dict() for edge in self.edges),
            "connectivity_score": self.connectivity_score,
            "continuity_graph_score": self.continuity_graph_score,
            "lineage_integrity_score": self.lineage_integrity_score,
            "overall_topology_score": self.overall_topology_score,
            "law_invariants": self.law_invariants,
            "graph_hash": self.graph_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("graph_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class TopologicalGraphReceipt:
    schema_version: int
    source_compression_hash: str
    source_replay_identity_hash: str
    graph_hash: str
    graph_chain_head: str
    node_count: int
    edge_count: int
    overall_topology_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "graph_hash": self.graph_hash,
            "graph_chain_head": self.graph_chain_head,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "overall_topology_score": self.overall_topology_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def build_topological_graph_kernel(
    source_artifact: CompressedMemoryArtifact,
    recovery_artifact: FragmentationRecoveryArtifact,
) -> TopologicalGraphKernelResult:
    source_records = _validate_source_artifact(source_artifact)
    repaired_records = _validate_recovery_artifact(
        recovery_artifact,
        source_artifact=source_artifact,
        source_records=source_records,
    )

    nodes_list: list[TopologicalGraphNode] = []
    for idx, record in enumerate(source_records):
        parent_id = ""
        if idx > 0:
            parent_id = _node_id(source_records[idx - 1])
        node = TopologicalGraphNode(
            node_id=_node_id(record),
            theme_index=record.theme_index,
            theme_hash=record.source_theme_hash,
            replay_identity_hash=record.source_replay_identity_hash,
            node_type="memory_theme",
            lineage_parent_id=parent_id,
        )
        nodes_list.append(node)

    nodes = tuple(sorted(nodes_list, key=lambda n: (n.theme_index, n.node_id)))

    edges_list: list[TopologicalGraphEdge] = []
    continuity_hits = 0.0
    observed_continuity = _clamp01(float(recovery_artifact.continuity_score))
    for idx in range(1, len(source_records)):
        src_record = source_records[idx - 1]
        tgt_record = source_records[idx]
        repaired_src = repaired_records[idx - 1]
        repaired_tgt = repaired_records[idx]
        contiguous = (
            repaired_tgt.source_parent_theme_hash == repaired_src.source_replay_identity_hash
            and repaired_tgt.source_replay_identity_hash == tgt_record.source_replay_identity_hash
            and repaired_src.source_replay_identity_hash == src_record.source_replay_identity_hash
        )
        weight = observed_continuity if contiguous else 0.0
        continuity_hits += weight
        edge_payload = {
            "source_node_id": _node_id(src_record),
            "target_node_id": _node_id(tgt_record),
            "edge_type": "continuity",
            "continuity_weight": weight,
        }
        edge = TopologicalGraphEdge(
            edge_id=_sha256_hex(edge_payload),
            source_node_id=edge_payload["source_node_id"],
            target_node_id=edge_payload["target_node_id"],
            edge_type="continuity",
            continuity_weight=weight,
        )
        edges_list.append(edge)

    edges = tuple(
        sorted(
            edges_list,
            key=lambda e: (e.source_node_id, e.target_node_id, e.edge_type, e.edge_id),
        )
    )

    expected_edge_count = max(0, len(source_records) - 1)
    connectivity_score = _safe_fraction(len(edges), expected_edge_count)
    continuity_graph_score = _clamp01(
        1.0 if expected_edge_count <= 0 else float(continuity_hits / expected_edge_count)
    )

    lineage_pass = 0
    for idx, node in enumerate(nodes):
        if idx == 0 and node.lineage_parent_id == "":
            lineage_pass += 1
        elif idx > 0 and node.lineage_parent_id == nodes[idx - 1].node_id:
            lineage_pass += 1
    lineage_integrity_score = _safe_fraction(lineage_pass, len(nodes))

    overall_topology_score = _clamp01(
        float(connectivity_score * 0.25 + continuity_graph_score * 0.5 + lineage_integrity_score * 0.25)
    )

    result = TopologicalGraphKernelResult(
        schema_version=_SCHEMA_VERSION,
        source_compression_hash=source_artifact.compression_hash,
        source_replay_identity_hash=source_artifact.replay_identity_hash,
        recovered_chain_head=recovery_artifact.repaired_chain_head,
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
        connectivity_score=connectivity_score,
        continuity_graph_score=continuity_graph_score,
        lineage_integrity_score=lineage_integrity_score,
        overall_topology_score=overall_topology_score,
        law_invariants=(
            TOPOLOGICAL_GRAPH_KERNEL_LAW,
            DETERMINISTIC_NODE_ORDERING_RULE,
            DETERMINISTIC_EDGE_ORDERING_RULE,
            REPLAY_SAFE_GRAPH_IDENTITY_RULE,
        ),
        graph_hash="",
    )
    return replace(result, graph_hash=result.stable_hash())


def export_topological_graph_bytes(artifact: TopologicalGraphKernelResult) -> bytes:
    if not isinstance(artifact, TopologicalGraphKernelResult):
        raise ValueError("artifact must be a TopologicalGraphKernelResult")
    return artifact.to_canonical_bytes()


def generate_topological_graph_receipt(
    artifact: TopologicalGraphKernelResult,
) -> TopologicalGraphReceipt:
    if not isinstance(artifact, TopologicalGraphKernelResult):
        raise ValueError("artifact must be a TopologicalGraphKernelResult")
    receipt = TopologicalGraphReceipt(
        schema_version=artifact.schema_version,
        source_compression_hash=artifact.source_compression_hash,
        source_replay_identity_hash=artifact.source_replay_identity_hash,
        graph_hash=artifact.graph_hash,
        graph_chain_head=artifact.recovered_chain_head,
        node_count=artifact.node_count,
        edge_count=artifact.edge_count,
        overall_topology_score=artifact.overall_topology_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "DETERMINISTIC_EDGE_ORDERING_RULE",
    "DETERMINISTIC_NODE_ORDERING_RULE",
    "REPLAY_SAFE_GRAPH_IDENTITY_RULE",
    "TOPOLOGICAL_GRAPH_KERNEL_LAW",
    "TopologicalGraphEdge",
    "TopologicalGraphKernelResult",
    "TopologicalGraphNode",
    "TopologicalGraphReceipt",
    "build_topological_graph_kernel",
    "export_topological_graph_bytes",
    "generate_topological_graph_receipt",
]
