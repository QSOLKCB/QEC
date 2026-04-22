"""v143.1 — Ensemble Consistency Engine (SPHAERA Phase 2).

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
from qec.analysis.generalized_invariant_detector import InvariantDetectionReceipt
from qec.analysis.invariant_geometry_embedding import InvariantGeometryReceipt
from qec.analysis.iterative_system_abstraction_layer import IterativeExecutionTrace

ENSEMBLE_CONSISTENCY_ENGINE_VERSION = "v143.1"
_CONTROL_MODE = "ensemble_consistency_engine_advisory"
_FULLY_CONSISTENT = "fully_consistent"
_CONSISTENT = "consistent"
_INCONSISTENT = "inconsistent"
_ALLOWED_LABELS: tuple[str, ...] = (_FULLY_CONSISTENT, _CONSISTENT, _INCONSISTENT)
_DEFAULT_EPSILON = 1e-3
_ROUND_DIGITS = 12


def _bounded(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    if output < 0.0 or output > 1.0:
        raise ValueError(f"{name} must be in [0,1]")
    return output


def _round_stable(value: float) -> float:
    return round(float(value), _ROUND_DIGITS)


@dataclass(frozen=True)
class EnsembleClass:
    class_id: str
    member_state_ids: tuple[str, ...]
    centroid_vector: tuple[float, ...]
    max_deviation: float
    mean_deviation: float
    consistency_label: str

    def __post_init__(self) -> None:
        if not isinstance(self.class_id, str) or not self.class_id:
            raise ValueError("class_id must be non-empty str")
        if not isinstance(self.member_state_ids, tuple):
            raise ValueError("member_state_ids must be tuple[str, ...]")
        for state_id in self.member_state_ids:
            if not isinstance(state_id, str) or not state_id:
                raise ValueError("member_state_ids must be tuple[str, ...]")
        if tuple(sorted(self.member_state_ids)) != self.member_state_ids:
            raise ValueError("member_state_ids must be sorted")
        if len(set(self.member_state_ids)) != len(self.member_state_ids):
            raise ValueError("member_state_ids must be unique")
        if not isinstance(self.centroid_vector, tuple) or len(self.centroid_vector) < 1:
            raise ValueError("centroid_vector must be non-empty tuple[float, ...]")
        for coordinate in self.centroid_vector:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError("centroid_vector entries must be numeric")
            coordinate_value = float(coordinate)
            if not math.isfinite(coordinate_value):
                raise ValueError("centroid_vector entries must be finite")
            if coordinate_value < 0.0 or coordinate_value > 1.0:
                raise ValueError("centroid_vector entries must be in [0,1]")

        object.__setattr__(self, "max_deviation", _bounded(self.max_deviation, "max_deviation"))
        object.__setattr__(self, "mean_deviation", _bounded(self.mean_deviation, "mean_deviation"))
        if self.consistency_label not in _ALLOWED_LABELS:
            raise ValueError("invalid consistency_label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "member_state_ids": self.member_state_ids,
            "centroid_vector": self.centroid_vector,
            "max_deviation": self.max_deviation,
            "mean_deviation": self.mean_deviation,
            "consistency_label": self.consistency_label,
        }


@dataclass(frozen=True)
class EnsembleConsistencyReceipt:
    ensembles: tuple[EnsembleClass, ...]
    ensemble_count: int
    global_consistency_score: float
    inconsistent_count: int
    invariant_receipt_stable_hash: str
    version: str
    control_mode: str
    observatory_only: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ensembles, tuple):
            raise ValueError("ensembles must be tuple[EnsembleClass, ...]")
        for ensemble in self.ensembles:
            if not isinstance(ensemble, EnsembleClass):
                raise ValueError("ensembles must be tuple[EnsembleClass, ...]")
        if isinstance(self.ensemble_count, bool) or not isinstance(self.ensemble_count, int):
            raise ValueError("ensemble_count must be int")
        if self.ensemble_count != len(self.ensembles):
            raise ValueError("ensemble_count mismatch")
        object.__setattr__(self, "global_consistency_score", _bounded(self.global_consistency_score, "global_consistency_score"))
        if isinstance(self.inconsistent_count, bool) or not isinstance(self.inconsistent_count, int):
            raise ValueError("inconsistent_count must be int")
        if self.inconsistent_count < 0:
            raise ValueError("inconsistent_count must be >= 0")
        if self.inconsistent_count > len(self.ensembles):
            raise ValueError("inconsistent_count out of range")
        if (
            not isinstance(self.invariant_receipt_stable_hash, str)
            or len(self.invariant_receipt_stable_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.invariant_receipt_stable_hash)
        ):
            raise ValueError("invariant_receipt_stable_hash must be 64-char lowercase sha256 hex")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be non-empty str")
        if self.control_mode != _CONTROL_MODE:
            raise ValueError(f"control_mode must be {_CONTROL_MODE!r}")
        if self.observatory_only is not True:
            raise ValueError("observatory_only must be True")

        signatures = tuple(ensemble.class_id for ensemble in self.ensembles)
        if tuple(sorted(signatures)) != signatures:
            raise ValueError("ensembles must be sorted by class_id")

        object.__setattr__(self, "stable_hash", sha256_hex(self._payload_without_hash()))

    def _payload_without_hash(self) -> dict[str, Any]:
        return {
            "ensembles": tuple(ensemble.to_dict() for ensemble in self.ensembles),
            "ensemble_count": self.ensemble_count,
            "global_consistency_score": self.global_consistency_score,
            "inconsistent_count": self.inconsistent_count,
            "invariant_receipt_stable_hash": self.invariant_receipt_stable_hash,
            "version": self.version,
            "control_mode": self.control_mode,
            "observatory_only": self.observatory_only,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_without_hash()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())


def _normalized_distance(vector: tuple[float, ...], centroid: tuple[float, ...]) -> float:
    if len(vector) != len(centroid):
        raise ValueError("embedding dimension mismatch")
    dimension = len(vector)
    if dimension < 1:
        raise ValueError("embedding vector must be non-empty")
    squared_distance = 0.0
    for coordinate, center in zip(vector, centroid):
        delta = coordinate - center
        squared_distance += delta * delta
    euclidean_distance = math.sqrt(squared_distance)
    max_distance = math.sqrt(float(dimension))
    if max_distance <= 0.0:
        return 0.0
    return _round_stable(min(1.0, euclidean_distance / max_distance))


def _centroid(vectors: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    if not vectors:
        raise ValueError("vectors must be non-empty")
    dimension = len(vectors[0])
    if dimension < 1:
        raise ValueError("embedding vector must be non-empty")
    if any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding dimension mismatch")
    sums = [0.0] * dimension
    for vector in vectors:
        for idx, coordinate in enumerate(vector):
            sums[idx] += coordinate
    count = float(len(vectors))
    return tuple(_round_stable(sums[idx] / count) for idx in range(dimension))


def _member_vectors(
    state_ids: tuple[str, ...],
    fallback_vector: tuple[float, ...],
    execution_trace: IterativeExecutionTrace | None,
) -> tuple[tuple[float, ...], ...]:
    if execution_trace is None:
        return tuple(fallback_vector for _ in state_ids)

    trace_vectors: dict[str, tuple[float, ...]] = {}
    target_member_ids = set(state_ids)
    for snapshot in execution_trace.snapshots:
        if snapshot.state_id not in target_member_ids:
            continue
        payload = snapshot.state_payload
        if not isinstance(payload, dict):
            continue
        embedding = payload.get("embedding_vector")
        if not isinstance(embedding, (tuple, list)):
            continue
        if len(embedding) != len(fallback_vector):
            raise ValueError("embedding dimension mismatch")
        vector: list[float] = []
        for coordinate in embedding:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError("embedding_vector entries must be numeric")
            coordinate_value = float(coordinate)
            if not math.isfinite(coordinate_value):
                raise ValueError("embedding_vector entries must be finite")
            if coordinate_value < 0.0 or coordinate_value > 1.0:
                raise ValueError("embedding_vector entries must be in [0,1]")
            vector.append(_round_stable(coordinate_value))
        trace_vectors[snapshot.state_id] = tuple(vector)

    return tuple(trace_vectors.get(state_id, fallback_vector) for state_id in state_ids)


def _parse_pattern_member_state_ids(pattern_key: str, pattern_type: str) -> tuple[str, ...]:
    if pattern_type == "fixed_point":
        return (pattern_key,)
    if pattern_type == "plateau":
        return (pattern_key,)
    if pattern_type == "oscillation":
        if "<->" in pattern_key:
            left, right = pattern_key.split("<->", 1)
            if left and right:
                return tuple(sorted((left, right)))
        return (pattern_key,)
    raise ValueError("unsupported invariant pattern type")


def _signature(invariant_type: str, member_state_ids: tuple[str, ...]) -> str:
    return sha256_hex({"invariant_type": invariant_type, "member_state_ids": member_state_ids})


def _validate_receipt_consistency(
    geometry_receipt: InvariantGeometryReceipt,
    invariant_receipt: InvariantDetectionReceipt,
) -> None:
    supported_types = {"fixed_point", "plateau", "oscillation"}
    expected_signatures: set[str] = set()
    for pattern in invariant_receipt.patterns:
        if pattern.pattern_type not in supported_types:
            continue
        member_state_ids = _parse_pattern_member_state_ids(pattern.key, pattern.pattern_type)
        expected_signatures.add(_signature(pattern.pattern_type, member_state_ids))
    geometry_signatures = {invariant_class.invariant_signature for invariant_class in geometry_receipt.invariant_classes}
    if geometry_signatures != expected_signatures:
        raise ValueError("geometry and invariant receipts are inconsistent")


def evaluate_ensemble_consistency_engine(
    geometry_receipt: InvariantGeometryReceipt,
    invariant_receipt: InvariantDetectionReceipt,
    execution_trace: IterativeExecutionTrace | None = None,
    *,
    epsilon: float = _DEFAULT_EPSILON,
    version: str = ENSEMBLE_CONSISTENCY_ENGINE_VERSION,
) -> EnsembleConsistencyReceipt:
    if not isinstance(geometry_receipt, InvariantGeometryReceipt):
        raise ValueError("invalid input type")
    if not isinstance(invariant_receipt, InvariantDetectionReceipt):
        raise ValueError("invalid input type")
    if execution_trace is not None and not isinstance(execution_trace, IterativeExecutionTrace):
        raise ValueError("invalid input type")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be numeric")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value < 0.0 or epsilon_value > 1.0:
        raise ValueError("epsilon must be in [0,1]")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty str")
    _validate_receipt_consistency(geometry_receipt, invariant_receipt)

    ensembles: list[EnsembleClass] = []
    mean_deviations: list[float] = []

    for invariant_class in geometry_receipt.invariant_classes:
        member_vectors = _member_vectors(
            invariant_class.member_state_ids,
            invariant_class.embedding_vector,
            execution_trace,
        )
        centroid_vector = _centroid(member_vectors)
        deviations = tuple(_normalized_distance(vector, centroid_vector) for vector in member_vectors)
        max_deviation = _round_stable(max(deviations) if deviations else 0.0)
        mean_deviation = _round_stable(sum(deviations) / float(len(deviations)) if deviations else 0.0)
        if max_deviation == 0.0:
            label = _FULLY_CONSISTENT
        elif max_deviation < epsilon_value:
            label = _CONSISTENT
        else:
            label = _INCONSISTENT

        ensembles.append(
            EnsembleClass(
                class_id=invariant_class.invariant_signature,
                member_state_ids=invariant_class.member_state_ids,
                centroid_vector=centroid_vector,
                max_deviation=max_deviation,
                mean_deviation=mean_deviation,
                consistency_label=label,
            )
        )
        mean_deviations.append(mean_deviation)

    ensembles_sorted = tuple(sorted(ensembles, key=lambda item: item.class_id))
    inconsistent_count = sum(1 for item in ensembles_sorted if item.consistency_label == _INCONSISTENT)
    mean_of_means = _round_stable(sum(mean_deviations) / float(len(mean_deviations)) if mean_deviations else 0.0)
    global_score = _round_stable(max(0.0, min(1.0, 1.0 - mean_of_means)))

    return EnsembleConsistencyReceipt(
        ensembles=ensembles_sorted,
        ensemble_count=len(ensembles_sorted),
        global_consistency_score=global_score,
        inconsistent_count=inconsistent_count,
        invariant_receipt_stable_hash=invariant_receipt.stable_hash,
        version=version,
        control_mode=_CONTROL_MODE,
        observatory_only=True,
    )


__all__ = [
    "ENSEMBLE_CONSISTENCY_ENGINE_VERSION",
    "EnsembleClass",
    "EnsembleConsistencyReceipt",
    "evaluate_ensemble_consistency_engine",
]
