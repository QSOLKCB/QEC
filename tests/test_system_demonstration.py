from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.system_demonstration import (
    SYSTEM_DEMONSTRATION_MODULE_VERSION,
    SystemDemonstrationReceipt,
    SystemStackEvidence,
    WorkloadDemonstrationDescriptor,
    demonstrate_full_system,
)


def _hash(seed: str) -> str:
    return sha256_hex({"seed": seed})


def _workload(
    *,
    workload_id: str,
    baseline_cost: int = 100,
    optimized_cost: int = 50,
    failure_count: int = 0,
    repair_count: int = 0,
    traces: tuple[str, ...] | None = None,
) -> WorkloadDemonstrationDescriptor:
    return WorkloadDemonstrationDescriptor(
        workload_id=workload_id,
        workload_type="decode",
        input_hash=_hash(f"input::{workload_id}"),
        baseline_cost=baseline_cost,
        optimized_cost=optimized_cost,
        failure_count=failure_count,
        repair_count=repair_count,
        governance_trace_hashes=traces if traces is not None else (_hash(f"trace::{workload_id}"),),
    )


def _evidence(
    *,
    compression_plan_status: str = "COMPRESSED",
    compression_ratio: float = 0.8,
    total_canonical_size_bytes: int = 1000,
    total_compressed_size_bytes: int = 800,
) -> SystemStackEvidence:
    return SystemStackEvidence(
        multi_agent_receipt_hash=_hash("multi"),
        hierarchical_memory_receipt_hash=_hash("memory"),
        hardware_alignment_receipt_hash=_hash("hardware"),
        execution_bridge_receipt_hash=_hash("bridge"),
        compression_storage_receipt_hash=_hash("compression"),
        compression_plan_status=compression_plan_status,
        compression_ratio=compression_ratio,
        total_canonical_size_bytes=total_canonical_size_bytes,
        total_compressed_size_bytes=total_compressed_size_bytes,
    )


def _metric_map(receipt: SystemDemonstrationReceipt) -> dict[str, float]:
    return {metric.metric_name: metric.metric_value for metric in receipt.metrics}


def test_fully_valid_stack_promoted() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=200, optimized_cost=120, failure_count=0, repair_count=1),
        _workload(workload_id="w-2", baseline_cost=150, optimized_cost=90, failure_count=0, repair_count=1),
    ), _evidence(compression_plan_status="COMPRESSED", compression_ratio=0.8, total_canonical_size_bytes=1000, total_compressed_size_bytes=800))
    assert receipt.module_version == SYSTEM_DEMONSTRATION_MODULE_VERSION
    assert receipt.promotion_status == "PROMOTED"
    assert receipt.deterministic_integrity_preserved is True
    assert receipt.measurable_gain_exists is True


def test_compression_driven_gain_only_promoted() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=100, failure_count=0, repair_count=1),
    ), _evidence(compression_plan_status="COMPRESSED", compression_ratio=0.75, total_canonical_size_bytes=400, total_compressed_size_bytes=300))
    metrics = _metric_map(receipt)
    assert metrics["cost_reduction"] == 0.0
    assert metrics["compression_effectiveness"] == 0.25
    assert receipt.promotion_status == "PROMOTED"
    assert receipt.measurable_gain_exists is True


def test_cost_only_gain_promoted_on_no_gain_storage_plan() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=40, failure_count=0, repair_count=1),
        _workload(workload_id="w-2", baseline_cost=50, optimized_cost=20, failure_count=0, repair_count=1),
    ), _evidence(compression_plan_status="NO_GAIN", compression_ratio=1.0, total_canonical_size_bytes=100, total_compressed_size_bytes=100))
    metrics = _metric_map(receipt)
    assert metrics["cost_reduction"] > 0.0
    assert metrics["compression_effectiveness"] == 0.0
    assert receipt.promotion_status == "PROMOTED"


def test_no_gain_at_all_is_blocked() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=100, failure_count=1, repair_count=1),
    ), _evidence(compression_plan_status="NO_GAIN", compression_ratio=1.0, total_canonical_size_bytes=10, total_compressed_size_bytes=10))
    assert receipt.promotion_status == "PROMOTION_BLOCKED"
    assert receipt.measurable_gain_exists is False


def test_compression_inconsistency_raises() -> None:
    with pytest.raises(ValueError, match="compression_ratio must match"):
        demonstrate_full_system((
            _workload(workload_id="w-1"),
        ), _evidence(compression_plan_status="COMPRESSED", compression_ratio=0.9, total_canonical_size_bytes=100, total_compressed_size_bytes=80))


def test_repair_convergence_zero_blocks_promotion() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=50, failure_count=3, repair_count=0),
    ), _evidence())
    statuses = {metric.metric_name: metric.metric_status for metric in receipt.metrics}
    assert statuses["repair_convergence"] == "FAIL"
    assert receipt.promotion_status == "PROMOTION_BLOCKED"


def test_missing_trace_rejected() -> None:
    with pytest.raises(ValueError, match="governance_trace_hashes must be non-empty"):
        _workload(workload_id="w-1", traces=tuple())


def test_deterministic_ordering_invariance() -> None:
    workloads = (
        _workload(workload_id="w-2", baseline_cost=100, optimized_cost=70),
        _workload(workload_id="w-1", baseline_cost=80, optimized_cost=40),
    )
    evidence = _evidence()
    left = demonstrate_full_system(workloads, evidence)
    right = demonstrate_full_system(tuple(reversed(workloads)), evidence)
    assert left.to_dict() == right.to_dict()
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_workload_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate workload_id"):
        demonstrate_full_system((_workload(workload_id="dup"), _workload(workload_id="dup")), _evidence())


def test_invalid_evidence_rejected() -> None:
    with pytest.raises(ValueError, match="compression_plan_status"):
        _evidence(compression_plan_status="INVALID")


def test_optimized_cost_greater_than_baseline_rejected() -> None:
    with pytest.raises(ValueError, match="optimized_cost must be <= baseline_cost"):
        _workload(workload_id="w-1", baseline_cost=10, optimized_cost=11)


def test_zero_baseline_edge_case() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=0, optimized_cost=0, failure_count=0, repair_count=0),
    ), _evidence(compression_plan_status="NO_GAIN", compression_ratio=0.0, total_canonical_size_bytes=0, total_compressed_size_bytes=0))
    metrics = _metric_map(receipt)
    assert metrics["cost_reduction"] == 0.0
    assert metrics["bounded_failure_rate"] == 1.0


def test_compression_no_gain_consistency() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=50),
    ), _evidence(compression_plan_status="NO_GAIN", compression_ratio=1.0, total_canonical_size_bytes=30, total_compressed_size_bytes=30))
    metrics = _metric_map(receipt)
    assert metrics["compression_effectiveness"] == 0.0
    assert metrics["storage_efficiency_gain"] == 0.0


def test_research_question_mapping_correctness() -> None:
    receipt = demonstrate_full_system((
        _workload(workload_id="w-1", baseline_cost=100, optimized_cost=50, failure_count=1, repair_count=1),
    ), _evidence())
    assert receipt.research_questions_supported == (
        "repair_reasoning_without_probabilistic_search",
        "cross_environment_convergence_supported",
        "repair_necessity_counterfactual_supported",
        "failure_fix_validation_chain_deterministic",
        "long_term_governance_repair_convergence_supported",
    )


def test_metric_ordering_and_uniqueness() -> None:
    receipt = demonstrate_full_system((_workload(workload_id="w-1"),), _evidence())
    names = tuple(metric.metric_name for metric in receipt.metrics)
    assert names == tuple(sorted(names))
    assert len(set(names)) == len(names)


def test_frozen_immutability() -> None:
    receipt = demonstrate_full_system((_workload(workload_id="w-1"),), _evidence())
    with pytest.raises(FrozenInstanceError):
        receipt.workload_count = 99


def test_canonical_json_stability() -> None:
    receipt_a = demonstrate_full_system((_workload(workload_id="w-1"),), _evidence())
    receipt_b = SystemDemonstrationReceipt(
        module_version=receipt_a.module_version,
        workload_count=receipt_a.workload_count,
        evidence=receipt_a.evidence,
        metrics=receipt_a.metrics,
        promotion_status=receipt_a.promotion_status,
        deterministic_integrity_preserved=receipt_a.deterministic_integrity_preserved,
        measurable_gain_exists=receipt_a.measurable_gain_exists,
        research_questions_supported=receipt_a.research_questions_supported,
        stable_hash_input=receipt_a.stable_hash(),
    )
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_stable_hash_replay_stability() -> None:
    receipt_a = demonstrate_full_system((_workload(workload_id="w-1"),), _evidence())
    receipt_b = demonstrate_full_system((_workload(workload_id="w-1"),), _evidence())
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
