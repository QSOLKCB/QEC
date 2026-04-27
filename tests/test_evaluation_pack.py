from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.adversarial_determinism_battery import run_adversarial_determinism_battery
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.counterfactual_replay_kernel import run_counterfactual_replay
from qec.analysis.cross_environment_replay_kernel import EnvironmentReplayArtifact, compare_cross_environment_replay
from qec.analysis.evaluation_pack import ALLOWED_ITEM_TYPES, build_evaluation_pack
from qec.analysis.failure_ledger import build_failure_ledger
from qec.analysis.fix_proposal_kernel import generate_fix_proposals
from qec.analysis.fix_validation_kernel import validate_fix_proposals
from qec.analysis.governance_validation_kernel import (
    GOVERNANCE_VALIDATION_MODULE_VERSION,
    GOVERNANCE_VALIDATION_SCHEMA_VERSION,
    GovernanceValidationReceipt,
)
from qec.analysis.issue_normalization_kernel import normalize_review_issues
from qec.analysis.real_workload_injection import WorkloadDescriptor, evaluate_deterministic_workload


def _issue(summary: str, *, severity: str = "LOW", category: str = "VALIDATION") -> dict[str, str]:
    return {
        "summary": summary,
        "body": summary,
        "source": "MANUAL",
        "severity": severity,
        "category": category,
        "target_path": "src/qec/analysis/evaluation_pack.py",
        "invariant": "FAIL_FAST_VALIDATION",
    }


def _cross_env_receipt(*, mismatch: bool) -> object:
    base = {
        "workload_id": "wl-1",
        "artifact_hash": sha256_hex({"artifact": 1}),
        "canonical_payload_hash": sha256_hex({"payload": 1}),
        "platform_label": "linux",
        "python_label": "3.12",
        "metadata_hash": sha256_hex({"meta": 1}),
    }
    first = EnvironmentReplayArtifact(environment_id="env-a", receipt_hash=sha256_hex({"r": 1}), **base)
    second = EnvironmentReplayArtifact(
        environment_id="env-b",
        receipt_hash=sha256_hex({"r": 2}) if mismatch else sha256_hex({"r": 1}),
        **base,
    )
    return compare_cross_environment_replay((first, second))


def _governance_receipt() -> GovernanceValidationReceipt:
    payload = {
        "schema_version": GOVERNANCE_VALIDATION_SCHEMA_VERSION,
        "module_version": GOVERNANCE_VALIDATION_MODULE_VERSION,
        "validation_status": "VALIDATED",
        "expected_recommendation": "MAINTAIN_POLICY",
        "recomputed_recommendation": "MAINTAIN_POLICY",
        "recommendation_stable": True,
        "memory_hash": sha256_hex({"memory": 1}),
        "expected_governance_hash": sha256_hex({"expected": 1}),
        "recomputed_governance_hash": sha256_hex({"recomputed": 1}),
        "validation_hash": sha256_hex({"validation": 1}),
        "validation_score": 1.0,
        "hash_match": True,
    }
    return GovernanceValidationReceipt(**payload, stable_hash=sha256_hex(payload))


def _workload_receipt() -> object:
    descriptor_payload = {
        "workload_id": "workload-eval-pack",
        "workload_type": "SCHEDULING",
        "operation_count": 10,
        "redundant_operation_count": 3,
        "invariant_count": 2,
        "decision_count": 5,
        "stable_decision_count": 5,
        "repair_action_count": 4,
        "validated_repair_count": 4,
        "metadata_hash": sha256_hex({"meta": "w"}),
    }
    descriptor = WorkloadDescriptor(**descriptor_payload, stable_hash=sha256_hex(descriptor_payload))
    return evaluate_deterministic_workload(descriptor)


def _bundle(*, with_failures: bool) -> tuple[object, ...]:
    issues = (_issue("stable check", severity="LOW"),)
    issue_receipt = normalize_review_issues(issues)
    proposal_receipt = generate_fix_proposals(issue_receipt)
    validation_receipt = validate_fix_proposals(proposal_receipt)
    counterfactual_receipt = run_counterfactual_replay(validation_receipt)
    adversarial_receipt = run_adversarial_determinism_battery({"issues": issues})
    cross_env_receipt = _cross_env_receipt(mismatch=with_failures)
    failure_ledger_receipt = build_failure_ledger(
        issue_receipt,
        proposal_receipt,
        validation_receipt,
        counterfactual_receipt,
        adversarial_receipt,
        cross_env_receipt,
    )
    return (
        _governance_receipt(),
        issue_receipt,
        proposal_receipt,
        validation_receipt,
        counterfactual_receipt,
        adversarial_receipt,
        cross_env_receipt,
        failure_ledger_receipt,
        _workload_receipt(),
    )


def test_complete_bundle_produces_complete() -> None:
    receipt = build_evaluation_pack(_bundle(with_failures=False))
    assert receipt.pack_status == "COMPLETE"
    assert receipt.summary.bundle_complete is True
    assert receipt.summary.failure_count == 0


def test_partial_bundle_produces_partial() -> None:
    full = _bundle(with_failures=False)
    receipt = build_evaluation_pack(full[:-1])
    assert receipt.pack_status == "PARTIAL"
    assert receipt.summary.item_count == 8
    assert receipt.summary.bundle_complete is False


def test_empty_bundle_produces_empty() -> None:
    receipt = build_evaluation_pack(())
    assert receipt.pack_status == "EMPTY"
    assert receipt.summary.item_count == 0


def test_failure_indicators_produce_has_failures() -> None:
    receipt = build_evaluation_pack(_bundle(with_failures=True))
    assert receipt.pack_status == "HAS_FAILURES"
    assert receipt.summary.failure_count > 0
    assert receipt.summary.determinism_preserved is False


def test_duplicate_receipt_hash_rejected() -> None:
    full = _bundle(with_failures=False)
    with pytest.raises(ValueError, match="duplicate receipt hash"):
        build_evaluation_pack((full[0], full[0]))


def test_input_order_stability() -> None:
    full = _bundle(with_failures=False)
    first = build_evaluation_pack(full)
    second = build_evaluation_pack(tuple(reversed(full)))

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash() == second.stable_hash()


def test_type_counts_include_all_item_types() -> None:
    receipt = build_evaluation_pack((_governance_receipt(),))
    assert tuple(receipt.summary.type_counts.keys()) == ALLOWED_ITEM_TYPES
    assert set(receipt.summary.type_counts) == set(ALLOWED_ITEM_TYPES)


def test_status_counts_deterministic() -> None:
    receipt = build_evaluation_pack(_bundle(with_failures=False))
    keys = tuple(receipt.summary.status_counts.keys())
    assert keys == tuple(sorted(keys))


def test_stable_hash_and_canonical_replay_stability() -> None:
    bundle = _bundle(with_failures=False)
    first = build_evaluation_pack(bundle)
    second = build_evaluation_pack(bundle)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.stable_hash() == second.stable_hash()


def test_invalid_receipt_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported receipt type"):
        build_evaluation_pack(({"not": "a receipt"},))



def test_frozen_dataclass_immutability() -> None:
    receipt = build_evaluation_pack((_governance_receipt(),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.pack_status = "EMPTY"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.summary.item_count = 2  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.items[0].status = "X"  # type: ignore[misc]


def test_bundle_completeness_logic() -> None:
    full = build_evaluation_pack(_bundle(with_failures=False))
    partial = build_evaluation_pack((_governance_receipt(), _workload_receipt()))

    assert full.summary.bundle_complete is True
    assert partial.summary.bundle_complete is False
