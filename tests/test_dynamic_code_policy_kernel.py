from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.dynamic_code_policy_kernel import (
    CandidatePolicyInput,
    DynamicCodePolicy,
    MigrationPolicyInput,
    OrchestrationPolicyInput,
    PolicyDecisionReceipt,
    RuntimeCodeState,
    decide_dynamic_code_policy,
)


def _base_runtime() -> RuntimeCodeState:
    return RuntimeCodeState(
        current_code_id="code_current",
        current_code_family="surface",
        current_logical_stability=0.8,
        current_projected_loss=0.2,
        current_hardware_alignment=0.7,
        current_execution_efficiency=0.75,
        current_migration_overhead=0.2,
        current_orchestration_depth=1,
    )


def _base_candidate() -> CandidatePolicyInput:
    return CandidatePolicyInput(
        candidate_code_id="code_candidate",
        candidate_code_family="color",
        selection_confidence=0.85,
        candidate_logical_stability=0.85,
        candidate_projected_loss=0.15,
        candidate_hardware_alignment=0.8,
        candidate_execution_efficiency=0.8,
    )


def _base_migration() -> MigrationPolicyInput:
    return MigrationPolicyInput(
        migration_target_family="color",
        migration_compatibility=0.9,
        migration_projected_loss=0.2,
        migration_distance_retention=0.9,
        migration_observable_overlap=0.8,
        migration_hardware_fit=0.85,
        migration_confidence=0.85,
        migration_admissible=True,
    )


def _base_orchestration() -> OrchestrationPolicyInput:
    return OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.8,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.2,
        cross_family_winner=True,
        benchmark_admissible=True,
    )


def _base_policy() -> DynamicCodePolicy:
    return DynamicCodePolicy(
        minimum_selection_confidence=0.7,
        minimum_migration_confidence=0.8,
        minimum_benchmark_utility=0.75,
        minimum_improvement_margin=0.1,
        maximum_projected_loss=0.4,
        maximum_migration_overhead=0.5,
        require_cross_family_benefit=False,
        require_migration_admissibility=True,
        prefer_stability_gain=True,
        prefer_hardware_alignment=True,
    )


def test_orchestrate_decision_when_benchmark_justifies() -> None:
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    assert receipt.decision.selected_action == "orchestrate"
    assert receipt.decision.recommend_orchestration is True


def test_migrate_decision_when_orchestration_not_admissible() -> None:
    orchestration = OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.8,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.2,
        cross_family_winner=True,
        benchmark_admissible=False,
    )
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), orchestration, _base_policy())
    assert receipt.decision.selected_action == "migrate"
    assert receipt.decision.approve_migration is True


def test_switch_decision_when_migration_not_approved() -> None:
    migration = MigrationPolicyInput(
        migration_target_family="color",
        migration_compatibility=0.9,
        migration_projected_loss=0.2,
        migration_distance_retention=0.9,
        migration_observable_overlap=0.8,
        migration_hardware_fit=0.85,
        migration_confidence=0.6,
        migration_admissible=True,
    )
    orchestration = OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.8,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.2,
        cross_family_winner=True,
        benchmark_admissible=False,
    )
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), migration, orchestration, _base_policy())
    assert receipt.decision.selected_action == "switch"


def test_stay_decision_when_current_remains_best() -> None:
    candidate = CandidatePolicyInput(
        candidate_code_id="code_current",
        candidate_code_family="surface",
        selection_confidence=0.6,
        candidate_logical_stability=0.7,
        candidate_projected_loss=0.25,
        candidate_hardware_alignment=0.7,
        candidate_execution_efficiency=0.7,
    )
    receipt = decide_dynamic_code_policy(_base_runtime(), candidate, _base_migration(), _base_orchestration(), _base_policy())
    assert receipt.decision.selected_action == "stay"
    assert receipt.decision.stay_on_current_code is True


def test_defer_decision_when_promising_but_thresholds_not_met() -> None:
    runtime = _base_runtime()
    orchestration = OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.72,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.05,
        cross_family_winner=True,
        benchmark_admissible=False,
    )
    migration = MigrationPolicyInput(
        migration_target_family="color",
        migration_compatibility=0.9,
        migration_projected_loss=0.2,
        migration_distance_retention=0.9,
        migration_observable_overlap=0.8,
        migration_hardware_fit=0.85,
        migration_confidence=0.75,
        migration_admissible=True,
    )
    receipt = decide_dynamic_code_policy(runtime, _base_candidate(), migration, orchestration, _base_policy())
    assert receipt.decision.selected_action == "defer"
    assert receipt.decision.stay_on_current_code is True
    assert receipt.decision.target_code_id == runtime.current_code_id
    assert receipt.decision.target_code_family == runtime.current_code_family


def test_reject_decision_when_projected_loss_exceeds_policy() -> None:
    runtime = _base_runtime()
    candidate = CandidatePolicyInput(
        candidate_code_id="code_candidate",
        candidate_code_family="color",
        selection_confidence=0.95,
        candidate_logical_stability=0.9,
        candidate_projected_loss=0.9,
        candidate_hardware_alignment=0.9,
        candidate_execution_efficiency=0.9,
    )
    receipt = decide_dynamic_code_policy(runtime, candidate, _base_migration(), _base_orchestration(), _base_policy())
    assert receipt.decision.selected_action == "reject"
    assert receipt.decision.stay_on_current_code is True
    assert receipt.decision.target_code_id == runtime.current_code_id
    assert receipt.decision.target_code_family == runtime.current_code_family


def test_require_cross_family_benefit_enforced() -> None:
    policy_payload = _base_policy().to_dict()
    policy_payload["require_cross_family_benefit"] = True
    policy = DynamicCodePolicy(**policy_payload)
    orchestration = OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.8,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.2,
        cross_family_winner=False,
        benchmark_admissible=True,
    )
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), orchestration, policy)
    assert receipt.decision.selected_action in ("defer", "stay")
    assert "cross-family benefit required but not demonstrated" in receipt.decision.rationale


def test_require_migration_admissibility_enforced() -> None:
    migration = MigrationPolicyInput(
        migration_target_family="color",
        migration_compatibility=0.9,
        migration_projected_loss=0.2,
        migration_distance_retention=0.9,
        migration_observable_overlap=0.8,
        migration_hardware_fit=0.85,
        migration_confidence=0.9,
        migration_admissible=False,
    )
    orchestration = OrchestrationPolicyInput(
        benchmark_best_candidate_id="code_candidate",
        benchmark_best_family="color",
        benchmark_best_utility=0.8,
        benchmark_baseline_utility=0.7,
        benchmark_improvement_margin=0.2,
        cross_family_winner=True,
        benchmark_admissible=False,
    )
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), migration, orchestration, _base_policy())
    assert receipt.decision.selected_action == "switch"
    assert "migration admissibility required but not satisfied" in receipt.decision.rationale


def test_rationale_order_is_deterministic() -> None:
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    assert receipt.decision.rationale[0] == "selection confidence threshold satisfied"
    assert receipt.decision.rationale[-1] == "selected action: orchestrate"


def test_canonical_json_and_hash_determinism() -> None:
    receipt_a = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    receipt_b = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash == receipt_b.stable_hash
    assert receipt_a.replay_identity == receipt_b.replay_identity


def test_frozen_dataclass_immutability() -> None:
    runtime = _base_runtime()
    with pytest.raises(FrozenInstanceError):
        runtime.current_code_id = "mutated"


def test_action_priority_prefers_orchestrate_over_migrate() -> None:
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    assert receipt.decision.selected_action == "orchestrate"


def test_invalid_numeric_bool_rejected() -> None:
    with pytest.raises(ValueError):
        CandidatePolicyInput(
            candidate_code_id="code_candidate",
            candidate_code_family="color",
            selection_confidence=True,
            candidate_logical_stability=0.85,
            candidate_projected_loss=0.15,
            candidate_hardware_alignment=0.8,
            candidate_execution_efficiency=0.8,
        )


def test_receipt_rejects_non_hex_hash_fields() -> None:
    receipt = decide_dynamic_code_policy(_base_runtime(), _base_candidate(), _base_migration(), _base_orchestration(), _base_policy())
    with pytest.raises(ValueError, match="replay_identity must be a 64-character SHA-256 hex string"):
        PolicyDecisionReceipt(
            runtime_state=receipt.runtime_state,
            candidate_input=receipt.candidate_input,
            migration_input=receipt.migration_input,
            orchestration_input=receipt.orchestration_input,
            policy_snapshot=receipt.policy_snapshot,
            decision=receipt.decision,
            schema_version=receipt.schema_version,
            replay_identity="z" * 64,
            stable_hash=receipt.stable_hash,
        )
