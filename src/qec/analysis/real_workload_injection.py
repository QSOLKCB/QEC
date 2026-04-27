"""v148.8 — Deterministic real workload injection analysis."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Sequence

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

REAL_WORKLOAD_INJECTION_SCHEMA_VERSION = "v148.8"
REAL_WORKLOAD_INJECTION_MODULE_VERSION = "v148.8"

_ALLOWED_WORKLOAD_TYPES = frozenset({"TRANSFORMER", "DIFFUSION", "SCHEDULING", "DECODING"})
_ALLOWED_WORKLOAD_STATUS = frozenset({"EVALUATED", "INVALID_INPUT"})
_ALLOWED_CLASSIFICATIONS = frozenset({"HIGH_GAIN", "MODERATE_GAIN", "LOW_GAIN", "NO_GAIN"})

_COUNT_FIELDS = (
    "operation_count",
    "redundant_operation_count",
    "invariant_count",
    "decision_count",
    "stable_decision_count",
    "repair_action_count",
    "validated_repair_count",
)


def _is_sha256_hex(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _validate_canonical_string(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")


def _validate_non_negative_int(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _round_metric(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("metric must be finite")
    rounded = round(float(value), 12)
    if rounded < 0.0 or rounded > 1.0:
        raise ValueError("metric must be within [0.0, 1.0]")
    return rounded


@dataclasses.dataclass(frozen=True)
class WorkloadDescriptor:
    workload_id: str
    workload_type: str
    operation_count: int
    redundant_operation_count: int
    invariant_count: int
    decision_count: int
    stable_decision_count: int
    repair_action_count: int
    validated_repair_count: int
    metadata_hash: str
    stable_hash: str

    def __post_init__(self) -> None:
        _validate_canonical_string(self.workload_id, name="workload_id")
        if self.workload_type not in _ALLOWED_WORKLOAD_TYPES:
            raise ValueError("workload_type must be one of TRANSFORMER, DIFFUSION, SCHEDULING, DECODING")

        for field_name in _COUNT_FIELDS:
            _validate_non_negative_int(getattr(self, field_name), name=field_name)

        if self.redundant_operation_count > self.operation_count:
            raise ValueError("redundant_operation_count must be <= operation_count")
        if self.invariant_count > self.operation_count:
            raise ValueError("invariant_count must be <= operation_count")
        if self.stable_decision_count > self.decision_count:
            raise ValueError("stable_decision_count must be <= decision_count")
        if self.validated_repair_count > self.repair_action_count:
            raise ValueError("validated_repair_count must be <= repair_action_count")

        if not _is_sha256_hex(self.metadata_hash):
            raise ValueError("metadata_hash must be a valid SHA-256 hex string")
        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be a valid SHA-256 hex string")

        expected_stable_hash = self._compute_stable_hash()
        if self.stable_hash != expected_stable_hash:
            raise ValueError("stable_hash mismatch for WorkloadDescriptor")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "workload_type": self.workload_type,
            "operation_count": self.operation_count,
            "redundant_operation_count": self.redundant_operation_count,
            "invariant_count": self.invariant_count,
            "decision_count": self.decision_count,
            "stable_decision_count": self.stable_decision_count,
            "repair_action_count": self.repair_action_count,
            "validated_repair_count": self.validated_repair_count,
            "metadata_hash": self.metadata_hash,
        }

    def _compute_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclasses.dataclass(frozen=True)
class WorkloadMetricSet:
    compute_eliminated: float
    redundancy_collapsed: float
    decision_stability: float
    repair_consistency: float
    overall_deterministic_gain: float
    stable_hash: str

    def __post_init__(self) -> None:
        for metric_name in (
            "compute_eliminated",
            "redundancy_collapsed",
            "decision_stability",
            "repair_consistency",
            "overall_deterministic_gain",
        ):
            metric = getattr(self, metric_name)
            if not isinstance(metric, float):
                raise ValueError(f"{metric_name} must be a float")
            _round_metric(metric)

        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be a valid SHA-256 hex string")
        expected_stable_hash = self._compute_stable_hash()
        if self.stable_hash != expected_stable_hash:
            raise ValueError("stable_hash mismatch for WorkloadMetricSet")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "compute_eliminated": self.compute_eliminated,
            "redundancy_collapsed": self.redundancy_collapsed,
            "decision_stability": self.decision_stability,
            "repair_consistency": self.repair_consistency,
            "overall_deterministic_gain": self.overall_deterministic_gain,
        }

    def _compute_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclasses.dataclass(frozen=True)
class DeterministicWorkloadReceipt:
    schema_version: str
    module_version: str
    workload_status: str
    workload: WorkloadDescriptor
    metrics: WorkloadMetricSet
    classification: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.workload_status not in _ALLOWED_WORKLOAD_STATUS:
            raise ValueError("workload_status must be EVALUATED or INVALID_INPUT")
        if self.classification not in _ALLOWED_CLASSIFICATIONS:
            raise ValueError("classification must be one of HIGH_GAIN, MODERATE_GAIN, LOW_GAIN, NO_GAIN")
        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be a valid SHA-256 hex string")

        expected_stable_hash = self._compute_stable_hash()
        if self.stable_hash != expected_stable_hash:
            raise ValueError("stable_hash mismatch for DeterministicWorkloadReceipt")

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "workload_status": self.workload_status,
            "workload": self.workload.to_dict(),
            "metrics": self.metrics.to_dict(),
            "classification": self.classification,
        }

    def _compute_stable_hash(self) -> str:
        return sha256_hex(self._hash_payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _classification_from_gain(overall_deterministic_gain: float) -> str:
    if overall_deterministic_gain >= 0.75:
        return "HIGH_GAIN"
    if overall_deterministic_gain >= 0.50:
        return "MODERATE_GAIN"
    if overall_deterministic_gain > 0.0:
        return "LOW_GAIN"
    return "NO_GAIN"


def _metric_ratio(numerator: int, denominator: int, *, zero_denominator_value: float) -> float:
    if denominator == 0:
        return zero_denominator_value
    return float(numerator) / float(denominator)


def _build_metric_set(workload: WorkloadDescriptor) -> WorkloadMetricSet:
    compute_eliminated = _round_metric(
        _metric_ratio(
            workload.redundant_operation_count,
            workload.operation_count,
            zero_denominator_value=0.0,
        )
    )
    redundancy_collapsed = _round_metric(
        _metric_ratio(
            workload.invariant_count,
            workload.operation_count,
            zero_denominator_value=0.0,
        )
    )
    decision_stability = _round_metric(
        _metric_ratio(
            workload.stable_decision_count,
            workload.decision_count,
            zero_denominator_value=1.0,
        )
    )
    repair_consistency = _round_metric(
        _metric_ratio(
            workload.validated_repair_count,
            workload.repair_action_count,
            zero_denominator_value=1.0,
        )
    )

    overall_deterministic_gain = _round_metric(
        (compute_eliminated + redundancy_collapsed + decision_stability + repair_consistency) / 4.0
    )

    payload = {
        "compute_eliminated": compute_eliminated,
        "redundancy_collapsed": redundancy_collapsed,
        "decision_stability": decision_stability,
        "repair_consistency": repair_consistency,
        "overall_deterministic_gain": overall_deterministic_gain,
    }

    return WorkloadMetricSet(
        compute_eliminated=compute_eliminated,
        redundancy_collapsed=redundancy_collapsed,
        decision_stability=decision_stability,
        repair_consistency=repair_consistency,
        overall_deterministic_gain=overall_deterministic_gain,
        stable_hash=sha256_hex(payload),
    )


def evaluate_deterministic_workload(workload: WorkloadDescriptor) -> DeterministicWorkloadReceipt:
    if not isinstance(workload, WorkloadDescriptor):
        raise ValueError("workload must be a WorkloadDescriptor")

    metrics = _build_metric_set(workload)
    classification = _classification_from_gain(metrics.overall_deterministic_gain)

    payload = {
        "schema_version": REAL_WORKLOAD_INJECTION_SCHEMA_VERSION,
        "module_version": REAL_WORKLOAD_INJECTION_MODULE_VERSION,
        "workload_status": "EVALUATED",
        "workload": workload.to_dict(),
        "metrics": metrics.to_dict(),
        "classification": classification,
    }

    return DeterministicWorkloadReceipt(
        schema_version=REAL_WORKLOAD_INJECTION_SCHEMA_VERSION,
        module_version=REAL_WORKLOAD_INJECTION_MODULE_VERSION,
        workload_status="EVALUATED",
        workload=workload,
        metrics=metrics,
        classification=classification,
        stable_hash=sha256_hex(payload),
    )


def _validate_workloads_sequence(
    workloads: Sequence[WorkloadDescriptor],
) -> tuple[WorkloadDescriptor, ...]:
    if not isinstance(workloads, Sequence):
        raise ValueError("workloads must be a Sequence of WorkloadDescriptor")

    validated_workloads = []
    for index, workload in enumerate(workloads):
        if not isinstance(workload, WorkloadDescriptor):
            raise ValueError(
                f"workloads[{index}] must be a WorkloadDescriptor"
            )
        validated_workloads.append(workload)

    return tuple(validated_workloads)


def evaluate_deterministic_workloads(
    workloads: Sequence[WorkloadDescriptor],
) -> tuple[DeterministicWorkloadReceipt, ...]:
    validated_workloads = _validate_workloads_sequence(workloads)
    ordered = tuple(
        sorted(
            validated_workloads,
            key=lambda item: (item.workload_type, item.workload_id, item.stable_hash),
        )
    )
    return tuple(evaluate_deterministic_workload(workload) for workload in ordered)


__all__ = [
    "REAL_WORKLOAD_INJECTION_SCHEMA_VERSION",
    "REAL_WORKLOAD_INJECTION_MODULE_VERSION",
    "WorkloadDescriptor",
    "WorkloadMetricSet",
    "DeterministicWorkloadReceipt",
    "evaluate_deterministic_workload",
    "evaluate_deterministic_workloads",
]
