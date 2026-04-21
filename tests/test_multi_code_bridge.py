from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.multi_code_bridge import (
    BRIDGE_STEP_NAMES,
    BridgeBenchmarkInput,
    BridgeMigrationInput,
    BridgePolicyInput,
    BridgeSelectionInput,
    build_multi_code_bridge,
)


def _hex(ch: str) -> str:
    return ch * 64


def _base_selection() -> BridgeSelectionInput:
    return BridgeSelectionInput(
        selected_code_id="surface_a",
        selected_code_family="surface",
        selection_confidence=0.8,
        ranking_order=("surface_a", "qldpc_b"),
        selection_stable_hash=_hex("a"),
    )


def _base_migration() -> BridgeMigrationInput:
    return BridgeMigrationInput(
        source_code_id="surface_a",
        source_code_family="surface",
        target_code_family="surface",
        migration_admissible=True,
        migration_confidence=0.7,
        migration_projected_loss=0.2,
        migration_distance_retention=0.8,
        migration_stable_hash=_hex("b"),
    )


def _base_benchmark() -> BridgeBenchmarkInput:
    return BridgeBenchmarkInput(
        best_candidate_id="surface_a",
        best_candidate_family="surface",
        best_utility=0.9,
        baseline_utility=0.8,
        improvement_margin=0.1,
        cross_family_winner=False,
        benchmark_admissible=True,
        benchmark_stable_hash=_hex("c"),
    )


def _base_policy(action: str = "stay") -> BridgePolicyInput:
    return BridgePolicyInput(
        selected_action=action,
        target_code_id="surface_a",
        target_code_family="surface",
        stay_on_current_code=(action in ("stay", "defer", "reject")),
        approve_migration=(action in ("migrate", "orchestrate")),
        recommend_orchestration=(action == "orchestrate"),
        policy_confidence=0.75,
        improvement_score=0.6,
        risk_score=0.2,
        escalation_level="none",
        policy_stable_hash=_hex("d"),
    )


def test_stay_handoff_success() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("stay"))
    assert receipt.readiness.bridge_ready is True
    assert "bridge readiness satisfied" in receipt.readiness.rationale


def test_migrate_handoff_success() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("migrate"))
    assert receipt.readiness.bridge_ready is True
    assert receipt.readiness.migration_policy_aligned is True


def test_orchestrate_handoff_success() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("orchestrate"))
    assert receipt.readiness.bridge_ready is True
    assert receipt.readiness.benchmark_policy_aligned is True


def test_defer_handoff_not_ready_with_hold_guidance() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("defer"))
    assert receipt.readiness.bridge_ready is False
    assert "defer action emits hold-state handoff" in receipt.readiness.rationale


def test_reject_handoff_blocked_semantics() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("reject"))
    assert receipt.readiness.bridge_ready is False
    assert "reject action emits blocked handoff" in receipt.readiness.rationale


def test_contradictory_inputs_return_receipt_not_value_error() -> None:
    policy = BridgePolicyInput(
        selected_action="migrate",
        target_code_id="qldpc_b",
        target_code_family="qldpc",
        stay_on_current_code=False,
        approve_migration=False,
        recommend_orchestration=False,
        policy_confidence=0.4,
        improvement_score=0.4,
        risk_score=0.8,
        escalation_level="review",
        policy_stable_hash=_hex("e"),
    )
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), policy)
    assert receipt.readiness.bridge_ready is False
    assert "migration approval required but not satisfied" in receipt.readiness.rationale


def test_malformed_hash_rejected() -> None:
    with pytest.raises(ValueError, match="selection_stable_hash"):
        BridgeSelectionInput(
            selected_code_id="surface_a",
            selected_code_family="surface",
            selection_confidence=0.8,
            ranking_order=("surface_a",),
            selection_stable_hash="BAD_HASH",
        )


def test_duplicate_ranking_order_rejected() -> None:
    with pytest.raises(ValueError, match="ranking_order"):
        BridgeSelectionInput(
            selected_code_id="surface_a",
            selected_code_family="surface",
            selection_confidence=0.8,
            ranking_order=("surface_a", "surface_a"),
            selection_stable_hash=_hex("a"),
        )


def test_canonical_json_stability() -> None:
    args = (_base_selection(), _base_migration(), _base_benchmark(), _base_policy("stay"))
    r1 = build_multi_code_bridge(*args)
    r2 = build_multi_code_bridge(*args)
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.to_canonical_bytes() == r2.to_canonical_bytes()


def test_stable_hash_and_replay_identity_determinism() -> None:
    args = (_base_selection(), _base_migration(), _base_benchmark(), _base_policy("migrate"))
    r1 = build_multi_code_bridge(*args)
    r2 = build_multi_code_bridge(*args)
    assert r1.stable_hash == r2.stable_hash
    assert r1.replay_identity == r2.replay_identity


def test_frozen_dataclass_immutability() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("stay"))
    with pytest.raises(FrozenInstanceError):
        receipt.schema_version = "mutated"


def test_deterministic_bridge_step_ordering() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("stay"))
    assert tuple(step.step_name for step in receipt.bridge_steps) == BRIDGE_STEP_NAMES
    assert tuple(step.step_index for step in receipt.bridge_steps) == tuple(range(len(BRIDGE_STEP_NAMES)))


def test_deterministic_rationale_ordering() -> None:
    receipt = build_multi_code_bridge(_base_selection(), _base_migration(), _base_benchmark(), _base_policy("migrate"))
    expected = (
        "selection and policy targets align",
        "migration target aligns with policy",
        "bridge readiness satisfied",
    )
    assert receipt.readiness.rationale == expected
