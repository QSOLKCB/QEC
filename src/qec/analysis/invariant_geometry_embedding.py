"""v143.0 — Invariant Geometry Embedding Kernel (SPHAERA Phase 1).

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.convergence_engine import ConvergenceReceipt
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt, InvariantPattern
from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionTrace

INVARIANT_GEOMETRY_EMBEDDING_VERSION = "v143.0"
_CONTROL_MODE = "invariant_geometry_embedding_advisory"
_SUPPORTED_INVARIANT_TYPES: tuple[str, ...] = ("fixed_point", "plateau", "oscillation")
_EMBEDDING_DIMENSION = 4


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return output


@dataclass(frozen=True)
class InvariantClass:
    class_id: str
    member_state_ids: tuple[str, ...]
    invariant_type: str
    embedding_vector: tuple[float, ...]
    invariant_signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id:
            raise ValueError("class_id must be non-empty str")
        if self.invariant_type not in _SUPPORTED_INVARIANT_TYPES:
            raise ValueError("invalid invariant_type")
        if not isinstance(self.member_state_ids, tuple):
            raise ValueError("member_state_ids must be tuple[str, ...]")
        for state_id in self.member_state_ids:
            if not isinstance(state_id, str) or not state_id:
                raise ValueError("member_state_ids must be tuple[str, ...]")
        if tuple(sorted(self.member_state_ids)) != self.member_state_ids:
            raise ValueError("member_state_ids must be sorted")
        if len(set(self.member_state_ids)) != len(self.member_state_ids):
            raise ValueError("member_state_ids must be unique")
        if not isinstance(self.embedding_vector, tuple):
            raise ValueError("embedding_vector must be tuple[float, ...]")
        if len(self.embedding_vector) != _EMBEDDING_DIMENSION:
            raise ValueError("invalid embedding dimension")
        for coordinate in self.embedding_vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError("embedding_vector entries must be numeric")
            coordinate_value = float(coordinate)
            if not math.isfinite(coordinate_value):
                raise ValueError("embedding_vector entries must be finite")
            if coordinate_value < 0.0 or coordinate_value > 1.0:
                raise ValueError("embedding_vector entries must be in [0,1]")
        if not isinstance(self.invariant_signature, str) or len(self.invariant_signature) != 64:
            raise ValueError("invariant_signature must be 64-char sha256 hex")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "member_state_ids": self.member_state_ids,
            "invariant_type": self.invariant_type,
            "embedding_vector": self.embedding_vector,
            "invariant_signature": self.invariant_signature,
        }


@dataclass(frozen=True)
class InvariantGeometryReceipt:
    invariant_classes: tuple[InvariantClass, ...]
    embedding_dimension: int
    class_count: int
    geometric_consistency_score: float
    embedding_stability_score: float
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.invariant_classes, tuple):
            raise ValueError("invariant_classes must be tuple[InvariantClass, ...]")
        for invariant_class in self.invariant_classes:
            if not isinstance(invariant_class, InvariantClass):
                raise ValueError("invariant_classes must be tuple[InvariantClass, ...]")
        if isinstance(self.embedding_dimension, bool) or not isinstance(self.embedding_dimension, int):
            raise ValueError("embedding_dimension must be int")
        if self.embedding_dimension < 1:
            raise ValueError("embedding_dimension must be >= 1")
        if self.embedding_dimension != _EMBEDDING_DIMENSION:
            raise ValueError("embedding_dimension mismatch")
        if isinstance(self.class_count, bool) or not isinstance(self.class_count, int):
            raise ValueError("class_count must be int")
        if self.class_count != len(self.invariant_classes):
            raise ValueError("class_count mismatch")
        object.__setattr__(self, "geometric_consistency_score", _bounded(self.geometric_consistency_score, "geometric_consistency_score"))
        object.__setattr__(self, "embedding_stability_score", _bounded(self.embedding_stability_score, "embedding_stability_score"))

        signatures = tuple(invariant_class.invariant_signature for invariant_class in self.invariant_classes)
        if tuple(sorted(signatures)) != signatures:
            raise ValueError("invariant_classes must be sorted by invariant_signature")

        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "invariant_classes": tuple(invariant_class.to_dict() for invariant_class in self.invariant_classes),
            "embedding_dimension": self.embedding_dimension,
            "class_count": self.class_count,
            "geometric_consistency_score": self.geometric_consistency_score,
            "embedding_stability_score": self.embedding_stability_score,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _parse_member_state_ids(pattern: InvariantPattern) -> tuple[str, ...]:
    if pattern.pattern_type == "fixed_point":
        return (pattern.key,)
    if pattern.pattern_type == "plateau":
        return (pattern.key,)
    if pattern.pattern_type == "oscillation":
        if "<->" in pattern.key:
            left, right = pattern.key.split("<->", 1)
            if left and right:
                return tuple(sorted((left, right)))
        return (pattern.key,)
    raise ValueError("unsupported invariant pattern type")


def _signature(invariant_type: str, member_state_ids: tuple[str, ...]) -> str:
    return sha256_hex(
        {
            "invariant_type": invariant_type,
            "member_state_ids": member_state_ids,
        }
    )


def _embedding_from_signature(invariant_signature: str, dimension: int) -> tuple[float, ...]:
    if dimension < 1:
        raise ValueError("dimension must be >= 1")
    max_uint64 = float((1 << 64) - 1)
    output: list[float] = []
    for index in range(dimension):
        chunk = sha256_hex({"invariant_signature": invariant_signature, "index": index})[:16]
        as_uint64 = int(chunk, 16)
        output.append(as_uint64 / max_uint64)
    return tuple(output)


def _geometric_consistency_score(invariant_classes: tuple[InvariantClass, ...]) -> float:
    """Score how tightly grouped distinct embedding vectors are.

    The caller may deduplicate invariant classes by signature before invoking this
    function, so a within-signature variance metric would collapse to `1.0` for any
    non-empty input. Measure normalized dispersion across the distinct embedding
    vectors instead so the score can vary for deduplicated outputs.
    """
    if not invariant_classes:
        return 1.0

    vectors = [invariant_class.embedding_vector for invariant_class in invariant_classes]
    if len(vectors) == 1:
        return 1.0

    dimension = len(vectors[0])
    if dimension < 1:
        raise ValueError("embedding vectors must have dimension >= 1")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("all embedding vectors must have the same dimension")

    means = [0.0] * dimension
    for vector in vectors:
        for idx, value in enumerate(vector):
            means[idx] += value

    inv_count = 1.0 / float(len(vectors))
    for idx in range(dimension):
        means[idx] *= inv_count

    mean_squared_distance = 0.0
    for vector in vectors:
        squared_distance = 0.0
        for idx, value in enumerate(vector):
            delta = value - means[idx]
            squared_distance += delta * delta
        mean_squared_distance += squared_distance
    mean_squared_distance *= inv_count

    max_mean_squared_distance = float(dimension) * 0.25
    if max_mean_squared_distance <= 0.0:
        return 1.0

    normalized_dispersion = min(1.0, mean_squared_distance / max_mean_squared_distance)
    return 1.0 - normalized_dispersion
def _embedding_stability_score(invariant_classes: tuple[InvariantClass, ...]) -> float:
    if not invariant_classes:
        return 1.0
    stable_count = 0
    for invariant_class in invariant_classes:
        replay = _embedding_from_signature(invariant_class.invariant_signature, len(invariant_class.embedding_vector))
        if replay == invariant_class.embedding_vector:
            stable_count += 1
    return float(stable_count) / float(len(invariant_classes))


def evaluate_invariant_geometry_embedding(
    invariant_receipt: InvariantDetectionReceipt,
    convergence_receipt: ConvergenceReceipt,
    execution_trace: IterativeExecutionTrace | None = None,
    *,
    version: str = INVARIANT_GEOMETRY_EMBEDDING_VERSION,
) -> InvariantGeometryReceipt:
    if not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if not isinstance(convergence_receipt, ConvergenceReceipt):
        raise ValueError("invalid input type")
    if execution_trace is not None and not isinstance(execution_trace, IterativeExecutionTrace):
        raise ValueError("invalid input type")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")

    dedup: dict[tuple[str, tuple[str, ...]], InvariantClass] = {}
    for pattern in invariant_receipt.patterns:
        if pattern.pattern_type not in _SUPPORTED_INVARIANT_TYPES:
            continue
        member_state_ids = _parse_member_state_ids(pattern)
        signature = _signature(pattern.pattern_type, member_state_ids)
        class_payload = (pattern.pattern_type, member_state_ids)
        if class_payload in dedup:
            continue
        dedup[class_payload] = InvariantClass(
            class_id=f"{pattern.pattern_type}:{signature[:12]}",
            member_state_ids=member_state_ids,
            invariant_type=pattern.pattern_type,
            embedding_vector=_embedding_from_signature(signature, _EMBEDDING_DIMENSION),
            invariant_signature=signature,
        )

    invariant_classes = tuple(sorted(dedup.values(), key=lambda item: (item.invariant_signature, item.class_id)))

    return InvariantGeometryReceipt(
        invariant_classes=invariant_classes,
        embedding_dimension=_EMBEDDING_DIMENSION,
        class_count=len(invariant_classes),
        geometric_consistency_score=_geometric_consistency_score(invariant_classes),
        embedding_stability_score=_embedding_stability_score(invariant_classes),
    )


__all__ = [
    "INVARIANT_GEOMETRY_EMBEDDING_VERSION",
    "InvariantClass",
    "InvariantGeometryReceipt",
    "evaluate_invariant_geometry_embedding",
]
