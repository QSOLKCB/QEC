from __future__ import annotations

import dataclasses
import json

import pytest

from qec.analysis.adversarial_determinism_battery import run_adversarial_determinism_battery
from qec.analysis.counterfactual_replay_kernel import run_counterfactual_replay
from qec.analysis.cross_environment_replay_kernel import EnvironmentReplayArtifact, compare_cross_environment_replay
from qec.analysis.evaluation_pack import EvaluationPackReceipt, build_evaluation_pack
from qec.analysis.failure_ledger import build_failure_ledger
from qec.analysis.fix_proposal_kernel import generate_fix_proposals
from qec.analysis.fix_validation_kernel import validate_fix_proposals
from qec.analysis.counterfactual_replay_kernel import CounterfactualReplayReceipt
from qec.analysis.failure_ledger import FailureLedgerReceipt
from qec.analysis.governance_validation_kernel import (
    GOVERNANCE_VALIDATION_MODULE_VERSION,
    GOVERNANCE_VALIDATION_SCHEMA_VERSION,
    GovernanceValidationReceipt,
)
from qec.analysis.issue_normalization_kernel import normalize_review_issues
from qec.analysis.promotion_gate import (
    CANONICAL_CHECK_ORDER,
    MODULE_VERSION,
    SCHEMA_VERSION,
    PromotionGateReceipt,
    evaluate_promotion_gate,
)
from qec.analysis.real_workload_injection import (
    DeterministicWorkloadReceipt,
    WorkloadDescriptor,
    evaluate_deterministic_workload,
)


def _issue(summary: str, *, severity: str = "LOW", category: str = "VALIDATION") -> dict[str, str]:
    return {
        "summary": summary,
        "body": summary,
        "source": "MANUAL",
        "severity": severity,
        "category": category,
        "target_path": "src/qec/analysis/promotion_gate.py",
        "invariant": "FAIL_FAST_VALIDATION",
    }


def _cross_env_receipt(*, mismatch: bool) -> object:
    from qec.analysis.canonical_hashing import sha256_hex

    base = {
        "workload_id": "wl-promotion-gate",
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


def _governance_receipt(*, status: str = "VALIDATED") -> GovernanceValidationReceipt:
    from qec.analysis.canonical_hashing import sha256_hex

    payload = {
        "schema_version": GOVERNANCE_VALIDATION_SCHEMA_VERSION,
        "module_version": GOVERNANCE_VALIDATION_MODULE_VERSION,
        "validation_status": status,
        "expected_recommendation": "MAINTAIN_POLICY",
        "recomputed_recommendation": "MAINTAIN_POLICY",
        "recommendation_stable": True,
        "memory_hash": sha256_hex({"memory": 1}),
        "expected_governance_hash": sha256_hex({"expected": 1}),
        "recomputed_governance_hash": sha256_hex({"recomputed": 1}),
        "validation_hash": sha256_hex({"validation": status}),
        "validation_score": 1.0,
        "hash_match": True,
    }
    return GovernanceValidationReceipt(**payload, stable_hash=sha256_hex(payload))


def _workload_receipt() -> object:
    from qec.analysis.canonical_hashing import sha256_hex

    descriptor_payload = {
        "workload_id": "workload-promotion-gate",
        "workload_type": "SCHEDULING",
        "operation_count": 8,
        "redundant_operation_count": 2,
        "invariant_count": 2,
        "decision_count": 4,
        "stable_decision_count": 4,
        "repair_action_count": 3,
        "validated_repair_count": 3,
        "metadata_hash": sha256_hex({"meta": "w"}),
    }
    descriptor = WorkloadDescriptor(**descriptor_payload, stable_hash=sha256_hex(descriptor_payload))
    return evaluate_deterministic_workload(descriptor)


def _bundle(*, with_failures: bool = False, governance_status: str = "VALIDATED") -> tuple[object, ...]:
    issues = (_issue("promotion gate stable check", severity="LOW"),)
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
        _governance_receipt(status=governance_status),
        issue_receipt,
        proposal_receipt,
        validation_receipt,
        counterfactual_receipt,
        adversarial_receipt,
        cross_env_receipt,
        failure_ledger_receipt,
        _workload_receipt(),
    )


def _pack(**kwargs: object) -> EvaluationPackReceipt:
    return build_evaluation_pack(_bundle(**kwargs))


def test_fully_complete_clean_pack_promotes() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=False))
    assert gate.promotion_status == "PROMOTE"
    assert gate.stop_reason == "NONE"
    assert gate.fail_count == 0
    assert gate.pass_count == len(CANONICAL_CHECK_ORDER)


def test_determinism_failure_stops() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=True))
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "DETERMINISTIC_INTEGRITY"


def test_missing_governance_validation_stops() -> None:
    pack = build_evaluation_pack(_bundle(with_failures=False)[1:])
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "GOVERNANCE_STABILITY"


def test_repair_inconsistency_stops() -> None:
    pack = build_evaluation_pack(
        tuple(item for item in _bundle(with_failures=False) if not isinstance(item, CounterfactualReplayReceipt))
    )
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "REPAIR_REASONING_CONSISTENCY"


def test_failure_count_gt_zero_stops() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=True))
    check = next(item for item in gate.checks if item.check_name == "FAILURES_BOUNDED_CLASSIFIED")
    assert check.check_status == "FAIL"


def test_missing_failure_ledger_stops() -> None:
    pack = build_evaluation_pack(
        tuple(item for item in _bundle(with_failures=False) if not isinstance(item, FailureLedgerReceipt))
    )
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "FAILURES_BOUNDED_CLASSIFIED"


def test_missing_deterministic_workload_stops() -> None:
    pack = build_evaluation_pack(
        tuple(item for item in _bundle(with_failures=False) if not isinstance(item, DeterministicWorkloadReceipt))
    )
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "MEASURABLE_BENEFIT"


def test_stop_reason_uses_first_failing_canonical_check() -> None:
    pack = build_evaluation_pack(_bundle(with_failures=True)[1:])
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "DETERMINISTIC_INTEGRITY"


def test_check_ordering_stability() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=False))
    assert tuple(item.check_name for item in gate.checks) == CANONICAL_CHECK_ORDER


def test_replay_stability_stable_hash_equality() -> None:
    pack = _pack(with_failures=False)
    first = evaluate_promotion_gate(pack)
    second = evaluate_promotion_gate(pack)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
    assert first.stable_hash() == second.stable_hash()


def test_frozen_dataclass_immutability() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=False))
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.stop_reason = "X"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        gate.checks[0].reason = "Y"  # type: ignore[misc]


def test_canonical_json_ordering() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=False))
    parsed = json.loads(gate.to_canonical_json())
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_invalid_input_rejected() -> None:
    with pytest.raises(ValueError, match="EvaluationPackReceipt"):
        evaluate_promotion_gate({"not": "a pack"})  # type: ignore[arg-type]


def test_invalid_workload_status_stops_measurable_benefit() -> None:
    from qec.analysis.canonical_hashing import sha256_hex

    workload = _workload_receipt()
    invalid_workload = dataclasses.replace(
        workload,
        workload_status="INVALID_INPUT",
        stable_hash=sha256_hex(
            {
                "schema_version": workload.schema_version,
                "module_version": workload.module_version,
                "workload_status": "INVALID_INPUT",
                "workload": workload.workload.to_dict(),
                "metrics": workload.metrics.to_dict(),
                "classification": workload.classification,
            }
        ),
    )
    pack = build_evaluation_pack((*_bundle(with_failures=False), invalid_workload))
    gate = evaluate_promotion_gate(pack)
    assert gate.promotion_status == "STOP"
    assert gate.stop_reason == "MEASURABLE_BENEFIT"


def test_receipt_version_invariants_rejected_when_mismatched() -> None:
    gate = evaluate_promotion_gate(_pack(with_failures=False))
    with pytest.raises(ValueError, match="schema_version"):
        PromotionGateReceipt(
            schema_version="v0",
            module_version=MODULE_VERSION,
            promotion_status=gate.promotion_status,
            input_pack_hash=gate.input_pack_hash,
            checks=gate.checks,
            pass_count=gate.pass_count,
            fail_count=gate.fail_count,
            stop_reason=gate.stop_reason,
        )
    with pytest.raises(ValueError, match="module_version"):
        PromotionGateReceipt(
            schema_version=SCHEMA_VERSION,
            module_version="v0",
            promotion_status=gate.promotion_status,
            input_pack_hash=gate.input_pack_hash,
            checks=gate.checks,
            pass_count=gate.pass_count,
            fail_count=gate.fail_count,
            stop_reason=gate.stop_reason,
        )
