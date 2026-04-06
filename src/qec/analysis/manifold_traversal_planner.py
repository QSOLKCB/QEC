"""v137.8.3 — Manifold Traversal Planner.

Deterministic Layer-4 consumer of E8 symmetry artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.e8_symmetry_projection_layer import E8SymmetryResult
from qec.analysis.polytope_reasoning_engine import PolytopeReasoningResult
from qec.analysis.topological_graph_kernel import TopologicalGraphKernelResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

MANIFOLD_TRAVERSAL_LAW = "MANIFOLD_TRAVERSAL_LAW"
DETERMINISTIC_PATH_ORDERING_RULE = "DETERMINISTIC_PATH_ORDERING_RULE"
REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE = "REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE"
BOUNDED_TRAVERSAL_SCORE_RULE = "BOUNDED_TRAVERSAL_SCORE_RULE"


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


def _safe_mean(values: tuple[float, ...]) -> float:
    if not values:
        return 1.0
    return _clamp01(float(sum(values) / len(values)))


def _validate_optional_lineage(
    *,
    symmetry_artifact: E8SymmetryResult,
    polytope_artifact: PolytopeReasoningResult | None,
    graph_artifact: TopologicalGraphKernelResult | None,
) -> None:
    if polytope_artifact is not None:
        if not isinstance(polytope_artifact, PolytopeReasoningResult):
            raise ValueError("polytope_artifact must be a PolytopeReasoningResult")
        if polytope_artifact.stable_hash() != polytope_artifact.polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match stable_hash")
        if polytope_artifact.polytope_hash != symmetry_artifact.source_polytope_hash:
            raise ValueError("polytope_artifact polytope_hash must match symmetry_artifact.source_polytope_hash")
        if polytope_artifact.source_graph_hash != symmetry_artifact.source_graph_hash:
            raise ValueError("polytope_artifact source_graph_hash must match symmetry_artifact.source_graph_hash")

    if graph_artifact is not None:
        if not isinstance(graph_artifact, TopologicalGraphKernelResult):
            raise ValueError("graph_artifact must be a TopologicalGraphKernelResult")
        if graph_artifact.stable_hash() != graph_artifact.graph_hash:
            raise ValueError("graph_artifact graph_hash must match stable_hash")
        if graph_artifact.graph_hash != symmetry_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match symmetry_artifact.source_graph_hash")
        if polytope_artifact is not None and graph_artifact.graph_hash != polytope_artifact.source_graph_hash:
            raise ValueError("graph_artifact graph_hash must match polytope_artifact.source_graph_hash")


def _validate_symmetry_artifact(symmetry_artifact: E8SymmetryResult) -> None:
    if not isinstance(symmetry_artifact, E8SymmetryResult):
        raise ValueError("symmetry_artifact must be an E8SymmetryResult")
    if symmetry_artifact.projection.vector_count != len(symmetry_artifact.projection.vectors):
        raise ValueError("symmetry_artifact projection vector_count must match vectors length")
    if symmetry_artifact.projection.coordinate_dimension <= 0:
        raise ValueError("symmetry_artifact projection coordinate_dimension must be positive")
    if symmetry_artifact.projection.stable_hash() != symmetry_artifact.projection.symmetry_hash:
        raise ValueError("symmetry_artifact projection symmetry_hash must match stable_hash")
    if symmetry_artifact.stable_hash() != symmetry_artifact.symmetry_hash:
        raise ValueError("symmetry_artifact symmetry_hash must match stable_hash")

    for score in (
        symmetry_artifact.projection.symmetry_alignment_score,
        symmetry_artifact.projection.basis_consistency_score,
        symmetry_artifact.projection.projection_integrity_score,
        symmetry_artifact.projection.lattice_continuity_score,
        symmetry_artifact.projection.overall_symmetry_score,
    ):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("symmetry_artifact scores must be finite numbers in [0.0, 1.0]")
        score_value = float(score)
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            raise ValueError("symmetry_artifact scores must be finite numbers in [0.0, 1.0]")

    expected_dim = symmetry_artifact.projection.coordinate_dimension
    for vector in symmetry_artifact.projection.vectors:
        if len(vector.coordinate_order) != expected_dim:
            raise ValueError("symmetry_artifact vector coordinate_order must match coordinate_dimension")
        if len(vector.normalized_coordinates) != expected_dim:
            raise ValueError("symmetry_artifact vector normalized_coordinates must match coordinate_dimension")


@dataclass(frozen=True)
class ManifoldNode:
    node_id: str
    node_index: int
    source_basis_id: str
    source_basis_index: int
    coordinate_order: tuple[str, ...]
    manifold_coordinates: tuple[float, ...]
    continuity_weight: float
    alignment_weight: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "node_index": self.node_index,
            "source_basis_id": self.source_basis_id,
            "source_basis_index": self.source_basis_index,
            "coordinate_order": self.coordinate_order,
            "manifold_coordinates": self.manifold_coordinates,
            "continuity_weight": self.continuity_weight,
            "alignment_weight": self.alignment_weight,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class TraversalPath:
    path_id: str
    path_index: int
    node_ids: tuple[str, ...]
    path_length: int
    path_continuity_score: float
    path_alignment_score: float
    route_integrity_score: float
    traversal_efficiency_score: float
    path_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "path_id": self.path_id,
            "path_index": self.path_index,
            "node_ids": self.node_ids,
            "path_length": self.path_length,
            "path_continuity_score": self.path_continuity_score,
            "path_alignment_score": self.path_alignment_score,
            "route_integrity_score": self.route_integrity_score,
            "traversal_efficiency_score": self.traversal_efficiency_score,
            "path_score": self.path_score,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ManifoldTraversalResult:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    source_replay_identity_hash: str
    node_count: int
    path_count: int
    nodes: tuple[ManifoldNode, ...]
    paths: tuple[TraversalPath, ...]
    path_continuity_score: float
    manifold_alignment_score: float
    symmetry_route_integrity_score: float
    traversal_efficiency_score: float
    overall_traversal_score: float
    law_invariants: tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "node_count": self.node_count,
            "path_count": self.path_count,
            "nodes": tuple(node.to_dict() for node in self.nodes),
            "paths": tuple(path.to_dict() for path in self.paths),
            "path_continuity_score": self.path_continuity_score,
            "manifold_alignment_score": self.manifold_alignment_score,
            "symmetry_route_integrity_score": self.symmetry_route_integrity_score,
            "traversal_efficiency_score": self.traversal_efficiency_score,
            "overall_traversal_score": self.overall_traversal_score,
            "law_invariants": self.law_invariants,
            "traversal_hash": self.traversal_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("traversal_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ManifoldTraversalReceipt:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_symmetry_hash: str
    traversal_hash: str
    overall_traversal_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_symmetry_hash": self.source_symmetry_hash,
            "traversal_hash": self.traversal_hash,
            "overall_traversal_score": self.overall_traversal_score,
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


def _build_manifold_nodes(symmetry_artifact: E8SymmetryResult) -> tuple[ManifoldNode, ...]:
    vectors = tuple(
        sorted(
            symmetry_artifact.projection.vectors,
            key=lambda vector: (vector.basis_index, vector.basis_id),
        )
    )

    nodes: list[ManifoldNode] = []
    for node_index, vector in enumerate(vectors):
        payload = {
            "source_symmetry_hash": symmetry_artifact.symmetry_hash,
            "source_basis_hash": vector.stable_hash(),
            "node_index": node_index,
        }
        nodes.append(
            ManifoldNode(
                node_id=_sha256_hex(payload),
                node_index=node_index,
                source_basis_id=vector.basis_id,
                source_basis_index=vector.basis_index,
                coordinate_order=vector.coordinate_order,
                manifold_coordinates=vector.normalized_coordinates,
                continuity_weight=_clamp01(float(vector.continuity_component)),
                alignment_weight=_clamp01(float(vector.projection_weight)),
            )
        )

    return tuple(sorted(nodes, key=lambda node: (node.node_index, node.source_basis_index, node.node_id)))


def _pair_similarity(left: ManifoldNode, right: ManifoldNode) -> float:
    if len(left.manifold_coordinates) != len(right.manifold_coordinates):
        return 0.0
    delta = tuple(abs(a - b) for a, b in zip(left.manifold_coordinates, right.manifold_coordinates))
    return _clamp01(float(1.0 - (sum(delta) / len(delta))))


def _build_traversal_paths(
    *,
    nodes: tuple[ManifoldNode, ...],
    symmetry_artifact: E8SymmetryResult,
) -> tuple[TraversalPath, ...]:
    if not nodes:
        return ()

    if len(nodes) == 1:
        node = nodes[0]
        singleton = TraversalPath(
            path_id=_sha256_hex(
                {
                    "source_symmetry_hash": symmetry_artifact.symmetry_hash,
                    "path_index": 0,
                    "node_ids": (node.node_id,),
                }
            ),
            path_index=0,
            node_ids=(node.node_id,),
            path_length=1,
            path_continuity_score=node.continuity_weight,
            path_alignment_score=node.alignment_weight,
            route_integrity_score=1.0,
            traversal_efficiency_score=1.0,
            path_score=_clamp01(float(0.5 * node.continuity_weight + 0.5 * node.alignment_weight)),
        )
        return (singleton,)

    paths: list[TraversalPath] = []
    for path_index, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
        pair_similarity = _pair_similarity(left, right)
        path_continuity_score = _clamp01(float((left.continuity_weight + right.continuity_weight) / 2.0))
        path_alignment_score = _clamp01(float((left.alignment_weight + right.alignment_weight) / 2.0))
        route_integrity_score = _clamp01(float(0.6 * pair_similarity + 0.4 * path_continuity_score))
        traversal_efficiency_score = _clamp01(float(0.7 * pair_similarity + 0.3 * path_alignment_score))
        path_score = _clamp01(
            float(
                0.35 * path_continuity_score
                + 0.2 * path_alignment_score
                + 0.25 * route_integrity_score
                + 0.2 * traversal_efficiency_score
            )
        )

        path_payload = {
            "source_symmetry_hash": symmetry_artifact.symmetry_hash,
            "path_index": path_index,
            "left_node_id": left.node_id,
            "right_node_id": right.node_id,
            "path_score": path_score,
        }
        paths.append(
            TraversalPath(
                path_id=_sha256_hex(path_payload),
                path_index=path_index,
                node_ids=(left.node_id, right.node_id),
                path_length=2,
                path_continuity_score=path_continuity_score,
                path_alignment_score=path_alignment_score,
                route_integrity_score=route_integrity_score,
                traversal_efficiency_score=traversal_efficiency_score,
                path_score=path_score,
            )
        )

    return tuple(sorted(paths, key=lambda path: (path.path_index, path.path_id)))


def build_manifold_traversal_plan(
    symmetry_artifact: E8SymmetryResult,
    polytope_artifact: PolytopeReasoningResult | None = None,
    graph_artifact: TopologicalGraphKernelResult | None = None,
) -> ManifoldTraversalResult:
    _validate_symmetry_artifact(symmetry_artifact)
    _validate_optional_lineage(
        symmetry_artifact=symmetry_artifact,
        polytope_artifact=polytope_artifact,
        graph_artifact=graph_artifact,
    )

    nodes = _build_manifold_nodes(symmetry_artifact)
    paths = _build_traversal_paths(nodes=nodes, symmetry_artifact=symmetry_artifact)

    raw_path_continuity = _safe_mean(tuple(path.path_continuity_score for path in paths))
    path_continuity_score = _clamp01(
        float(0.6 * raw_path_continuity + 0.4 * symmetry_artifact.projection.lattice_continuity_score)
    )
    manifold_alignment_score = _safe_mean(tuple(node.alignment_weight for node in nodes))
    symmetry_route_integrity_score = _clamp01(
        float(
            0.4 * _safe_mean(tuple(path.route_integrity_score for path in paths))
            + 0.3 * symmetry_artifact.projection.projection_integrity_score
            + 0.3 * symmetry_artifact.projection.overall_symmetry_score
        )
    )

    node_coverage = _clamp01(
        float(len({node_id for path in paths for node_id in path.node_ids}) / len(nodes))
        if nodes
        else 1.0
    )
    traversal_efficiency_score = _clamp01(
        float(
            0.55 * _safe_mean(tuple(path.traversal_efficiency_score for path in paths))
            + 0.25 * node_coverage
            + 0.2 * symmetry_artifact.projection.lattice_continuity_score
        )
    )

    overall_traversal_score = _clamp01(
        float(
            0.2 * path_continuity_score
            + 0.15 * manifold_alignment_score
            + 0.25 * symmetry_route_integrity_score
            + 0.2 * traversal_efficiency_score
            + 0.2 * symmetry_artifact.projection.overall_symmetry_score
        )
    )

    result = ManifoldTraversalResult(
        schema_version=_SCHEMA_VERSION,
        source_graph_hash=symmetry_artifact.source_graph_hash,
        source_polytope_hash=symmetry_artifact.source_polytope_hash,
        source_symmetry_hash=symmetry_artifact.symmetry_hash,
        source_replay_identity_hash=symmetry_artifact.source_replay_identity_hash,
        node_count=len(nodes),
        path_count=len(paths),
        nodes=nodes,
        paths=paths,
        path_continuity_score=path_continuity_score,
        manifold_alignment_score=manifold_alignment_score,
        symmetry_route_integrity_score=symmetry_route_integrity_score,
        traversal_efficiency_score=traversal_efficiency_score,
        overall_traversal_score=overall_traversal_score,
        law_invariants=(
            MANIFOLD_TRAVERSAL_LAW,
            DETERMINISTIC_PATH_ORDERING_RULE,
            REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE,
            BOUNDED_TRAVERSAL_SCORE_RULE,
        ),
        traversal_hash="",
    )
    return replace(result, traversal_hash=result.stable_hash())


def export_manifold_traversal_bytes(artifact: ManifoldTraversalResult) -> bytes:
    if not isinstance(artifact, ManifoldTraversalResult):
        raise ValueError("artifact must be a ManifoldTraversalResult")
    return artifact.to_canonical_bytes()


def generate_manifold_traversal_receipt(artifact: ManifoldTraversalResult) -> ManifoldTraversalReceipt:
    if not isinstance(artifact, ManifoldTraversalResult):
        raise ValueError("artifact must be a ManifoldTraversalResult")
    receipt = ManifoldTraversalReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_polytope_hash=artifact.source_polytope_hash,
        source_symmetry_hash=artifact.source_symmetry_hash,
        traversal_hash=artifact.traversal_hash,
        overall_traversal_score=artifact.overall_traversal_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_TRAVERSAL_SCORE_RULE",
    "DETERMINISTIC_PATH_ORDERING_RULE",
    "MANIFOLD_TRAVERSAL_LAW",
    "REPLAY_SAFE_TRAVERSAL_IDENTITY_RULE",
    "ManifoldNode",
    "TraversalPath",
    "ManifoldTraversalResult",
    "ManifoldTraversalReceipt",
    "build_manifold_traversal_plan",
    "export_manifold_traversal_bytes",
    "generate_manifold_traversal_receipt",
]
