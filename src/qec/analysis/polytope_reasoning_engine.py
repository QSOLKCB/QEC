"""v137.8.1 — Polytope Reasoning Engine.

Deterministic Layer-4 consumer of topological graph kernel artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.topological_graph_kernel import TopologicalGraphKernelResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

POLYTOPE_REASONING_LAW = "POLYTOPE_REASONING_LAW"
DETERMINISTIC_VERTEX_ORDERING_RULE = "DETERMINISTIC_VERTEX_ORDERING_RULE"
DETERMINISTIC_FACE_ORDERING_RULE = "DETERMINISTIC_FACE_ORDERING_RULE"
REPLAY_SAFE_POLYTOPE_IDENTITY_RULE = "REPLAY_SAFE_POLYTOPE_IDENTITY_RULE"


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


def _validate_graph_artifact(graph: TopologicalGraphKernelResult) -> None:
    if not isinstance(graph, TopologicalGraphKernelResult):
        raise ValueError("graph_artifact must be a TopologicalGraphKernelResult")
    if graph.node_count != len(graph.nodes):
        raise ValueError("graph_artifact node_count must match nodes length")
    if graph.edge_count != len(graph.edges):
        raise ValueError("graph_artifact edge_count must match edges length")
    if graph.stable_hash() != graph.graph_hash:
        raise ValueError("graph_artifact graph_hash must match stable_hash")
    node_ids = tuple(node.node_id for node in graph.nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("graph_artifact nodes must contain unique node_id values")
    node_set = frozenset(node_ids)
    for edge in graph.edges:
        if edge.source_node_id not in node_set or edge.target_node_id not in node_set:
            raise ValueError("graph_artifact edges must reference existing nodes")
        if edge.continuity_weight < 0.0 or edge.continuity_weight > 1.0:
            raise ValueError("graph_artifact edge continuity_weight must be in [0.0, 1.0]")


@dataclass(frozen=True)
class PolytopeVertex:
    vertex_id: str
    source_node_id: str
    theme_index: int
    adjacent_edge_count: int
    local_connectivity_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "vertex_id": self.vertex_id,
            "source_node_id": self.source_node_id,
            "theme_index": self.theme_index,
            "adjacent_edge_count": self.adjacent_edge_count,
            "local_connectivity_score": self.local_connectivity_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PolytopeFace:
    face_id: str
    face_index: int
    vertex_ids: tuple[str, ...]
    source_edge_ids: tuple[str, ...]
    face_dimension: int
    continuity_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "face_id": self.face_id,
            "face_index": self.face_index,
            "vertex_ids": self.vertex_ids,
            "source_edge_ids": self.source_edge_ids,
            "face_dimension": self.face_dimension,
            "continuity_score": self.continuity_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class PolytopeReasoningResult:
    schema_version: int
    source_graph_hash: str
    source_replay_identity_hash: str
    vertex_count: int
    face_count: int
    vertices: tuple[PolytopeVertex, ...]
    faces: tuple[PolytopeFace, ...]
    vertex_connectivity_score: float
    face_continuity_score: float
    dimensional_consistency_score: float
    polytope_integrity_score: float
    overall_polytope_score: float
    law_invariants: tuple[str, ...]
    polytope_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "vertices": tuple(vertex.to_dict() for vertex in self.vertices),
            "faces": tuple(face.to_dict() for face in self.faces),
            "vertex_connectivity_score": self.vertex_connectivity_score,
            "face_continuity_score": self.face_continuity_score,
            "dimensional_consistency_score": self.dimensional_consistency_score,
            "polytope_integrity_score": self.polytope_integrity_score,
            "overall_polytope_score": self.overall_polytope_score,
            "law_invariants": self.law_invariants,
            "polytope_hash": self.polytope_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("polytope_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class PolytopeReasoningReceipt:
    schema_version: int
    source_graph_hash: str
    source_replay_identity_hash: str
    polytope_hash: str
    vertex_count: int
    face_count: int
    overall_polytope_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "polytope_hash": self.polytope_hash,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "overall_polytope_score": self.overall_polytope_score,
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


def build_polytope_reasoning_engine(graph_artifact: TopologicalGraphKernelResult) -> PolytopeReasoningResult:
    _validate_graph_artifact(graph_artifact)

    degree_by_node_id = {node.node_id: 0 for node in graph_artifact.nodes}
    for edge in graph_artifact.edges:
        degree_by_node_id[edge.source_node_id] += 1
        degree_by_node_id[edge.target_node_id] += 1

    vertices_list: list[PolytopeVertex] = []
    for node in graph_artifact.nodes:
        degree = degree_by_node_id[node.node_id]
        local_connectivity = _clamp01(float(degree / 2.0))
        vertex_payload = {
            "source_graph_hash": graph_artifact.graph_hash,
            "source_node_id": node.node_id,
            "theme_index": node.theme_index,
            "adjacent_edge_count": degree,
        }
        vertices_list.append(
            PolytopeVertex(
                vertex_id=_sha256_hex(vertex_payload),
                source_node_id=node.node_id,
                theme_index=node.theme_index,
                adjacent_edge_count=degree,
                local_connectivity_score=local_connectivity,
            )
        )

    vertices = tuple(sorted(vertices_list, key=lambda v: (v.theme_index, v.source_node_id, v.vertex_id)))

    vertex_id_by_node_id = {vertex.source_node_id: vertex.vertex_id for vertex in vertices}

    faces_list: list[PolytopeFace] = []
    for idx, edge in enumerate(graph_artifact.edges):
        src_vertex_id = vertex_id_by_node_id[edge.source_node_id]
        tgt_vertex_id = vertex_id_by_node_id[edge.target_node_id]
        dimension = 2
        vertex_ids = tuple(sorted((src_vertex_id, tgt_vertex_id)))
        source_edge_ids = (edge.edge_id,)
        continuity = _clamp01(float(edge.continuity_weight))
        if idx > 0:
            prev_edge = graph_artifact.edges[idx - 1]
            if prev_edge.target_node_id == edge.source_node_id:
                dimension = 3
                prev_vertex_id = vertex_id_by_node_id[prev_edge.source_node_id]
                vertex_ids = tuple(sorted((prev_vertex_id, src_vertex_id, tgt_vertex_id)))
                source_edge_ids = tuple(sorted((prev_edge.edge_id, edge.edge_id)))
                continuity = _clamp01(float((prev_edge.continuity_weight + edge.continuity_weight) / 2.0))

        face_payload = {
            "source_graph_hash": graph_artifact.graph_hash,
            "face_index": idx,
            "vertex_ids": vertex_ids,
            "source_edge_ids": source_edge_ids,
            "face_dimension": dimension,
            "continuity_score": continuity,
        }
        faces_list.append(
            PolytopeFace(
                face_id=_sha256_hex(face_payload),
                face_index=idx,
                vertex_ids=vertex_ids,
                source_edge_ids=source_edge_ids,
                face_dimension=dimension,
                continuity_score=continuity,
            )
        )

    faces = tuple(sorted(faces_list, key=lambda f: (f.face_index, f.face_id, f.vertex_ids, f.source_edge_ids)))

    connected_vertices = sum(1 for vertex in vertices if vertex.adjacent_edge_count > 0)
    vertex_connectivity_score = _safe_fraction(connected_vertices, len(vertices))

    continuity_hits = sum(face.continuity_score for face in faces)
    face_continuity_score = _clamp01(float(continuity_hits / len(faces))) if faces else 1.0

    expected_face_count = graph_artifact.edge_count
    face_count_match = 1 if len(faces) == expected_face_count else 0
    expected_dimensions = sum(1 for face in faces if face.face_dimension in (2, 3))
    dimensional_consistency_score = _clamp01(
        float(0.5 * _safe_fraction(expected_dimensions, len(faces)) + 0.5 * float(face_count_match))
    )

    source_alignment = 1.0 if graph_artifact.source_replay_identity_hash else 0.0
    polytope_integrity_score = _clamp01(
        float(0.4 * source_alignment + 0.3 * graph_artifact.overall_topology_score + 0.3 * dimensional_consistency_score)
    )

    overall_polytope_score = _clamp01(
        float(
            0.25 * vertex_connectivity_score
            + 0.35 * face_continuity_score
            + 0.2 * dimensional_consistency_score
            + 0.2 * polytope_integrity_score
        )
    )

    result = PolytopeReasoningResult(
        schema_version=_SCHEMA_VERSION,
        source_graph_hash=graph_artifact.graph_hash,
        source_replay_identity_hash=graph_artifact.source_replay_identity_hash,
        vertex_count=len(vertices),
        face_count=len(faces),
        vertices=vertices,
        faces=faces,
        vertex_connectivity_score=vertex_connectivity_score,
        face_continuity_score=face_continuity_score,
        dimensional_consistency_score=dimensional_consistency_score,
        polytope_integrity_score=polytope_integrity_score,
        overall_polytope_score=overall_polytope_score,
        law_invariants=(
            POLYTOPE_REASONING_LAW,
            DETERMINISTIC_VERTEX_ORDERING_RULE,
            DETERMINISTIC_FACE_ORDERING_RULE,
            REPLAY_SAFE_POLYTOPE_IDENTITY_RULE,
        ),
        polytope_hash="",
    )
    return replace(result, polytope_hash=result.stable_hash())


def export_polytope_bytes(artifact: PolytopeReasoningResult) -> bytes:
    if not isinstance(artifact, PolytopeReasoningResult):
        raise ValueError("artifact must be a PolytopeReasoningResult")
    return artifact.to_canonical_bytes()


def generate_polytope_receipt(artifact: PolytopeReasoningResult) -> PolytopeReasoningReceipt:
    if not isinstance(artifact, PolytopeReasoningResult):
        raise ValueError("artifact must be a PolytopeReasoningResult")
    receipt = PolytopeReasoningReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_replay_identity_hash=artifact.source_replay_identity_hash,
        polytope_hash=artifact.polytope_hash,
        vertex_count=artifact.vertex_count,
        face_count=artifact.face_count,
        overall_polytope_score=artifact.overall_polytope_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "DETERMINISTIC_FACE_ORDERING_RULE",
    "DETERMINISTIC_VERTEX_ORDERING_RULE",
    "POLYTOPE_REASONING_LAW",
    "REPLAY_SAFE_POLYTOPE_IDENTITY_RULE",
    "PolytopeFace",
    "PolytopeReasoningReceipt",
    "PolytopeReasoningResult",
    "PolytopeVertex",
    "build_polytope_reasoning_engine",
    "export_polytope_bytes",
    "generate_polytope_receipt",
]
