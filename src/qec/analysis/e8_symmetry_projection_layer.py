"""v137.8.2 — E8 Symmetry Projection Layer.

Deterministic Layer-4 consumer of polytope topology artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.polytope_reasoning_engine import PolytopeReasoningResult
from qec.analysis.topological_graph_kernel import TopologicalGraphKernelResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1
_COORDINATE_ORDER = tuple(f"x{i}" for i in range(8))

E8_SYMMETRY_PROJECTION_LAW = "E8_SYMMETRY_PROJECTION_LAW"
DETERMINISTIC_BASIS_ORDERING_RULE = "DETERMINISTIC_BASIS_ORDERING_RULE"
REPLAY_SAFE_SYMMETRY_IDENTITY_RULE = "REPLAY_SAFE_SYMMETRY_IDENTITY_RULE"
BOUNDED_PROJECTION_SCORE_RULE = "BOUNDED_PROJECTION_SCORE_RULE"


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


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return _clamp01(float(numerator / denominator))


def _hash_unit_interval(*, source_polytope_hash: str, basis_index: int, coordinate_index: int) -> float:
    payload = {
        "source_polytope_hash": source_polytope_hash,
        "basis_index": basis_index,
        "coordinate_index": coordinate_index,
    }
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    unit = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64 - 1)
    return _clamp01(float(unit))


def _validate_polytope_artifact(polytope_artifact: PolytopeReasoningResult) -> None:
    if not isinstance(polytope_artifact, PolytopeReasoningResult):
        raise ValueError("polytope_artifact must be a PolytopeReasoningResult")
    if polytope_artifact.vertex_count != len(polytope_artifact.vertices):
        raise ValueError("polytope_artifact vertex_count must match vertices length")
    if polytope_artifact.face_count != len(polytope_artifact.faces):
        raise ValueError("polytope_artifact face_count must match faces length")
    if polytope_artifact.stable_hash() != polytope_artifact.polytope_hash:
        raise ValueError("polytope_artifact polytope_hash must match stable_hash")
    for score in (
        polytope_artifact.vertex_connectivity_score,
        polytope_artifact.face_continuity_score,
        polytope_artifact.dimensional_consistency_score,
        polytope_artifact.polytope_integrity_score,
        polytope_artifact.overall_polytope_score,
    ):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("polytope_artifact scores must be finite numbers in [0.0, 1.0]")
        score_value = float(score)
        if not math.isfinite(score_value) or not 0.0 <= score_value <= 1.0:
            raise ValueError("polytope_artifact scores must be finite numbers in [0.0, 1.0]")


def _validate_optional_graph_lineage(
    *,
    graph_artifact: TopologicalGraphKernelResult | None,
    polytope_artifact: PolytopeReasoningResult,
) -> None:
    if graph_artifact is None:
        return
    if not isinstance(graph_artifact, TopologicalGraphKernelResult):
        raise ValueError("graph_artifact must be a TopologicalGraphKernelResult")
    if graph_artifact.stable_hash() != graph_artifact.graph_hash:
        raise ValueError("graph_artifact graph_hash must match stable_hash")
    if graph_artifact.graph_hash != polytope_artifact.source_graph_hash:
        raise ValueError("graph_artifact graph_hash must match polytope_artifact.source_graph_hash")


@dataclass(frozen=True)
class E8SymmetryVector:
    basis_id: str
    basis_index: int
    coordinate_order: tuple[str, ...]
    normalized_coordinates: tuple[float, ...]
    projection_weight: float
    continuity_component: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "basis_id": self.basis_id,
            "basis_index": self.basis_index,
            "coordinate_order": self.coordinate_order,
            "normalized_coordinates": self.normalized_coordinates,
            "projection_weight": self.projection_weight,
            "continuity_component": self.continuity_component,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class E8SymmetryProjection:
    projection_id: str
    source_graph_hash: str
    source_polytope_hash: str
    vector_count: int
    coordinate_dimension: int
    vectors: tuple[E8SymmetryVector, ...]
    symmetry_alignment_score: float
    basis_consistency_score: float
    projection_integrity_score: float
    lattice_continuity_score: float
    overall_symmetry_score: float
    law_invariants: tuple[str, ...]
    symmetry_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "projection_id": self.projection_id,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "vector_count": self.vector_count,
            "coordinate_dimension": self.coordinate_dimension,
            "vectors": tuple(vector.to_dict() for vector in self.vectors),
            "symmetry_alignment_score": self.symmetry_alignment_score,
            "basis_consistency_score": self.basis_consistency_score,
            "projection_integrity_score": self.projection_integrity_score,
            "lattice_continuity_score": self.lattice_continuity_score,
            "overall_symmetry_score": self.overall_symmetry_score,
            "law_invariants": self.law_invariants,
            "symmetry_hash": self.symmetry_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("symmetry_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class E8SymmetryResult:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    source_replay_identity_hash: str
    projection: E8SymmetryProjection
    symmetry_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "projection": self.projection.to_dict(),
            "symmetry_hash": self.symmetry_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("symmetry_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class E8SymmetryReceipt:
    schema_version: int
    source_graph_hash: str
    source_polytope_hash: str
    symmetry_hash: str
    overall_symmetry_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_graph_hash": self.source_graph_hash,
            "source_polytope_hash": self.source_polytope_hash,
            "symmetry_hash": self.symmetry_hash,
            "overall_symmetry_score": self.overall_symmetry_score,
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


def _deterministic_basis_vectors(polytope_artifact: PolytopeReasoningResult) -> tuple[E8SymmetryVector, ...]:
    face_vertex_cardinality_mean = 1.0
    if polytope_artifact.face_count > 0:
        face_vertex_cardinality_mean = float(
            sum(len(face.vertex_ids) for face in polytope_artifact.faces) / polytope_artifact.face_count
        )

    unique_dimensions = len({face.face_dimension for face in polytope_artifact.faces})
    structural_components: tuple[float, ...] = (
        _clamp01(float(polytope_artifact.vertex_connectivity_score)),
        _clamp01(float(polytope_artifact.face_continuity_score)),
        _clamp01(float(polytope_artifact.dimensional_consistency_score)),
        _clamp01(float(polytope_artifact.polytope_integrity_score)),
        _clamp01(float(polytope_artifact.overall_polytope_score)),
        _safe_ratio(polytope_artifact.vertex_count, polytope_artifact.vertex_count + polytope_artifact.face_count),
        _clamp01(float(unique_dimensions / 8.0)),
        _clamp01(float(face_vertex_cardinality_mean / 8.0)),
    )

    vectors: list[E8SymmetryVector] = []
    for basis_index in range(8):
        raw_coordinates = []
        for coordinate_index in range(8):
            structural = structural_components[(basis_index + coordinate_index) % 8]
            hash_component = _hash_unit_interval(
                source_polytope_hash=polytope_artifact.polytope_hash,
                basis_index=basis_index,
                coordinate_index=coordinate_index,
            )
            raw_coordinates.append(_clamp01(float(0.75 * structural + 0.25 * hash_component)))

        scale = max(raw_coordinates) if raw_coordinates else 1.0
        if scale <= 0.0:
            normalized = tuple(0.0 for _ in raw_coordinates)
        else:
            normalized = tuple(_clamp01(float(c / scale)) for c in raw_coordinates)

        projection_weight = _clamp01(float(sum(normalized) / len(normalized)))
        continuity_component = _clamp01(
            float(0.5 * polytope_artifact.face_continuity_score + 0.5 * projection_weight)
        )

        basis_payload = {
            "source_polytope_hash": polytope_artifact.polytope_hash,
            "basis_index": basis_index,
            "coordinate_order": _COORDINATE_ORDER,
            "normalized_coordinates": normalized,
            "projection_weight": projection_weight,
            "continuity_component": continuity_component,
        }
        vectors.append(
            E8SymmetryVector(
                basis_id=_sha256_hex(basis_payload),
                basis_index=basis_index,
                coordinate_order=_COORDINATE_ORDER,
                normalized_coordinates=normalized,
                projection_weight=projection_weight,
                continuity_component=continuity_component,
            )
        )

    return tuple(sorted(vectors, key=lambda vector: (vector.basis_index, vector.basis_id)))


def build_e8_symmetry_projection(
    polytope_artifact: PolytopeReasoningResult,
    graph_artifact: TopologicalGraphKernelResult | None = None,
) -> E8SymmetryResult:
    _validate_polytope_artifact(polytope_artifact)
    _validate_optional_graph_lineage(graph_artifact=graph_artifact, polytope_artifact=polytope_artifact)

    vectors = _deterministic_basis_vectors(polytope_artifact)

    projection_weight_mean = _clamp01(float(sum(v.projection_weight for v in vectors) / len(vectors)))
    basis_ordered = tuple(v.basis_index for v in vectors) == tuple(range(len(vectors)))
    basis_ids_unique = len({v.basis_id for v in vectors}) == len(vectors)
    coordinates_valid = all(
        v.coordinate_order == _COORDINATE_ORDER and len(v.normalized_coordinates) == len(_COORDINATE_ORDER)
        for v in vectors
    )

    symmetry_alignment_score = _clamp01(
        float(0.65 * polytope_artifact.overall_polytope_score + 0.35 * projection_weight_mean)
    )
    basis_consistency_score = _clamp01(
        float(0.7 * (1.0 if basis_ordered else 0.0) + 0.3 * (1.0 if coordinates_valid else 0.0))
    )
    projection_integrity_score = _clamp01(
        float(0.6 * (1.0 if basis_ids_unique else 0.0) + 0.4 * (1.0 if coordinates_valid else 0.0))
    )
    lattice_continuity_score = _clamp01(
        float(0.7 * polytope_artifact.face_continuity_score + 0.3 * polytope_artifact.vertex_connectivity_score)
    )
    overall_symmetry_score = _clamp01(
        float(
            0.3 * symmetry_alignment_score
            + 0.2 * basis_consistency_score
            + 0.2 * projection_integrity_score
            + 0.3 * lattice_continuity_score
        )
    )

    projection = E8SymmetryProjection(
        projection_id=_sha256_hex(
            {
                "source_graph_hash": polytope_artifact.source_graph_hash,
                "source_polytope_hash": polytope_artifact.polytope_hash,
                "vector_hashes": tuple(vector.stable_hash() for vector in vectors),
            }
        ),
        source_graph_hash=polytope_artifact.source_graph_hash,
        source_polytope_hash=polytope_artifact.polytope_hash,
        vector_count=len(vectors),
        coordinate_dimension=len(_COORDINATE_ORDER),
        vectors=vectors,
        symmetry_alignment_score=symmetry_alignment_score,
        basis_consistency_score=basis_consistency_score,
        projection_integrity_score=projection_integrity_score,
        lattice_continuity_score=lattice_continuity_score,
        overall_symmetry_score=overall_symmetry_score,
        law_invariants=(
            E8_SYMMETRY_PROJECTION_LAW,
            DETERMINISTIC_BASIS_ORDERING_RULE,
            REPLAY_SAFE_SYMMETRY_IDENTITY_RULE,
            BOUNDED_PROJECTION_SCORE_RULE,
        ),
        symmetry_hash="",
    )
    projection = replace(projection, symmetry_hash=projection.stable_hash())

    result = E8SymmetryResult(
        schema_version=_SCHEMA_VERSION,
        source_graph_hash=polytope_artifact.source_graph_hash,
        source_polytope_hash=polytope_artifact.polytope_hash,
        source_replay_identity_hash=polytope_artifact.source_replay_identity_hash,
        projection=projection,
        symmetry_hash="",
    )
    return replace(result, symmetry_hash=result.stable_hash())


def export_e8_projection_bytes(artifact: E8SymmetryResult) -> bytes:
    if not isinstance(artifact, E8SymmetryResult):
        raise ValueError("artifact must be an E8SymmetryResult")
    return artifact.to_canonical_bytes()


def generate_e8_projection_receipt(artifact: E8SymmetryResult) -> E8SymmetryReceipt:
    if not isinstance(artifact, E8SymmetryResult):
        raise ValueError("artifact must be an E8SymmetryResult")
    receipt = E8SymmetryReceipt(
        schema_version=artifact.schema_version,
        source_graph_hash=artifact.source_graph_hash,
        source_polytope_hash=artifact.source_polytope_hash,
        symmetry_hash=artifact.symmetry_hash,
        overall_symmetry_score=artifact.projection.overall_symmetry_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_PROJECTION_SCORE_RULE",
    "DETERMINISTIC_BASIS_ORDERING_RULE",
    "E8_SYMMETRY_PROJECTION_LAW",
    "REPLAY_SAFE_SYMMETRY_IDENTITY_RULE",
    "E8SymmetryProjection",
    "E8SymmetryReceipt",
    "E8SymmetryResult",
    "E8SymmetryVector",
    "build_e8_symmetry_projection",
    "export_e8_projection_bytes",
    "generate_e8_projection_receipt",
]
