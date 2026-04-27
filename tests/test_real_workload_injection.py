from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.real_workload_injection import (
    DeterministicWorkloadReceipt,
    WorkloadDescriptor,
    evaluate_deterministic_workload,
    evaluate_deterministic_workloads,
)


def _descriptor(
    *,
    workload_id: str,
    workload_type: str,
    operation_count: int,
    redundant_operation_count: int,
    invariant_count: int,
    decision_count: int,
    stable_decision_count: int,
    repair_action_count: int,
    validated_repair_count: int,
    metadata_hash: str | None = None,
) -> WorkloadDescriptor:
    metadata_hash_value = metadata_hash or sha256_hex({"workload_id": workload_id, "kind": workload_type})
    payload = {
        "workload_id": workload_id,
        "workload_type": workload_type,
        "operation_count": operation_count,
        "redundant_operation_count": redundant_operation_count,
        "invariant_count": invariant_count,
        "decision_count": decision_count,
        "stable_decision_count": stable_decision_count,
        "repair_action_count": repair_action_count,
        "validated_repair_count": validated_repair_count,
        "metadata_hash": metadata_hash_value,
    }
    return WorkloadDescriptor(
        workload_id=workload_id,
        workload_type=workload_type,
        operation_count=operation_count,
        redundant_operation_count=redundant_operation_count,
        invariant_count=invariant_count,
        decision_count=decision_count,
        stable_decision_count=stable_decision_count,
        repair_action_count=repair_action_count,
        validated_repair_count=validated_repair_count,
        metadata_hash=metadata_hash_value,
        stable_hash=sha256_hex(payload),
    )


@pytest.mark.parametrize(
    ("workload_type", "workload_id"),
    (
        ("TRANSFORMER", "transformer-main"),
        ("DIFFUSION", "diffusion-main"),
        ("SCHEDULING", "scheduling-main"),
        ("DECODING", "decoding-main"),
    ),
)
def test_supported_workload_types_evaluate(workload_type: str, workload_id: str) -> None:
    workload = _descriptor(
        workload_id=workload_id,
        workload_type=workload_type,
        operation_count=10,
        redundant_operation_count=4,
        invariant_count=5,
        decision_count=8,
        stable_decision_count=6,
        repair_action_count=7,
        validated_repair_count=6,
    )
    receipt = evaluate_deterministic_workload(workload)
    assert isinstance(receipt, DeterministicWorkloadReceipt)
    assert receipt.workload_status == "EVALUATED"


def test_metric_bounds() -> None:
    workload = _descriptor(
        workload_id="bounds",
        workload_type="TRANSFORMER",
        operation_count=50,
        redundant_operation_count=5,
        invariant_count=10,
        decision_count=4,
        stable_decision_count=3,
        repair_action_count=8,
        validated_repair_count=4,
    )
    metrics = evaluate_deterministic_workload(workload).metrics
    assert 0.0 <= metrics.compute_eliminated <= 1.0
    assert 0.0 <= metrics.redundancy_collapsed <= 1.0
    assert 0.0 <= metrics.decision_stability <= 1.0
    assert 0.0 <= metrics.repair_consistency <= 1.0
    assert 0.0 <= metrics.overall_deterministic_gain <= 1.0


def test_zero_count_behavior() -> None:
    workload = _descriptor(
        workload_id="zeros",
        workload_type="DIFFUSION",
        operation_count=0,
        redundant_operation_count=0,
        invariant_count=0,
        decision_count=0,
        stable_decision_count=0,
        repair_action_count=0,
        validated_repair_count=0,
    )
    metrics = evaluate_deterministic_workload(workload).metrics
    assert metrics.compute_eliminated == 0.0
    assert metrics.redundancy_collapsed == 0.0
    assert metrics.decision_stability == 1.0
    assert metrics.repair_consistency == 1.0
    assert metrics.overall_deterministic_gain == 0.5


def test_classification_thresholds() -> None:
    high = _descriptor(
        workload_id="high",
        workload_type="TRANSFORMER",
        operation_count=4,
        redundant_operation_count=4,
        invariant_count=4,
        decision_count=4,
        stable_decision_count=4,
        repair_action_count=4,
        validated_repair_count=4,
    )
    moderate = _descriptor(
        workload_id="moderate",
        workload_type="DIFFUSION",
        operation_count=4,
        redundant_operation_count=2,
        invariant_count=2,
        decision_count=4,
        stable_decision_count=2,
        repair_action_count=4,
        validated_repair_count=2,
    )
    low = _descriptor(
        workload_id="low",
        workload_type="SCHEDULING",
        operation_count=4,
        redundant_operation_count=0,
        invariant_count=0,
        decision_count=4,
        stable_decision_count=0,
        repair_action_count=4,
        validated_repair_count=1,
    )
    no = _descriptor(
        workload_id="no",
        workload_type="DECODING",
        operation_count=4,
        redundant_operation_count=0,
        invariant_count=0,
        decision_count=4,
        stable_decision_count=0,
        repair_action_count=4,
        validated_repair_count=0,
    )

    assert evaluate_deterministic_workload(high).classification == "HIGH_GAIN"
    assert evaluate_deterministic_workload(moderate).classification == "MODERATE_GAIN"
    assert evaluate_deterministic_workload(low).classification == "LOW_GAIN"
    assert evaluate_deterministic_workload(no).classification == "NO_GAIN"


def test_invalid_workload_type_rejected() -> None:
    with pytest.raises(ValueError, match="workload_type"):
        _descriptor(
            workload_id="invalid-type",
            workload_type="GRAPHICS",
            operation_count=1,
            redundant_operation_count=0,
            invariant_count=0,
            decision_count=1,
            stable_decision_count=1,
            repair_action_count=1,
            validated_repair_count=1,
        )


def test_invalid_hash_rejected() -> None:
    with pytest.raises(ValueError, match="metadata_hash"):
        _descriptor(
            workload_id="bad-hash",
            workload_type="TRANSFORMER",
            operation_count=1,
            redundant_operation_count=0,
            invariant_count=0,
            decision_count=1,
            stable_decision_count=1,
            repair_action_count=1,
            validated_repair_count=1,
            metadata_hash="z" * 64,
        )


def test_child_count_exceeding_parent_rejected() -> None:
    with pytest.raises(ValueError, match="stable_decision_count"):
        _descriptor(
            workload_id="bad-child",
            workload_type="TRANSFORMER",
            operation_count=1,
            redundant_operation_count=0,
            invariant_count=0,
            decision_count=1,
            stable_decision_count=2,
            repair_action_count=1,
            validated_repair_count=1,
        )


def test_frozen_dataclass_immutability() -> None:
    workload = _descriptor(
        workload_id="frozen",
        workload_type="DECODING",
        operation_count=2,
        redundant_operation_count=1,
        invariant_count=1,
        decision_count=2,
        stable_decision_count=2,
        repair_action_count=2,
        validated_repair_count=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        workload.workload_id = "mutated"  # type: ignore[misc]


def test_canonical_json_hash_stability() -> None:
    workload = _descriptor(
        workload_id="canon",
        workload_type="SCHEDULING",
        operation_count=8,
        redundant_operation_count=4,
        invariant_count=3,
        decision_count=8,
        stable_decision_count=7,
        repair_action_count=8,
        validated_repair_count=6,
    )
    receipt = evaluate_deterministic_workload(workload)
    rebuilt = evaluate_deterministic_workload(workload)
    assert receipt.to_canonical_json() == rebuilt.to_canonical_json()
    assert receipt.stable_hash == rebuilt.stable_hash


def test_replay_stability_repeated_evaluation() -> None:
    workload = _descriptor(
        workload_id="replay",
        workload_type="TRANSFORMER",
        operation_count=32,
        redundant_operation_count=10,
        invariant_count=11,
        decision_count=9,
        stable_decision_count=9,
        repair_action_count=3,
        validated_repair_count=2,
    )
    a = evaluate_deterministic_workload(workload)
    b = evaluate_deterministic_workload(workload)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_batch_ordering_stability() -> None:
    a = _descriptor(
        workload_id="z-id",
        workload_type="DECODING",
        operation_count=2,
        redundant_operation_count=1,
        invariant_count=1,
        decision_count=2,
        stable_decision_count=1,
        repair_action_count=2,
        validated_repair_count=1,
    )
    b = _descriptor(
        workload_id="a-id",
        workload_type="TRANSFORMER",
        operation_count=2,
        redundant_operation_count=1,
        invariant_count=1,
        decision_count=2,
        stable_decision_count=1,
        repair_action_count=2,
        validated_repair_count=1,
    )

    receipts = evaluate_deterministic_workloads((a, b))
    assert tuple(r.workload.workload_id for r in receipts) == ("z-id", "a-id")
