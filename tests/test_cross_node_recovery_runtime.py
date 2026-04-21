from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib

import pytest

from qec.analysis.cross_node_recovery_runtime import (
    CrossNodeRecoveryReceipt,
    RecoveryAction,
    RecoveryNodeInput,
    RecoveryNodeStatus,
    RecoveryPolicy,
    export_cross_node_recovery_runtime_bytes,
    run_cross_node_recovery_runtime,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _node(
    node_id: str,
    *,
    role: str = "validator",
    epoch: int = 7,
    sync_admissible: bool = True,
    replay_admissible: bool = True,
    proof_admissible: bool = True,
    sync_confidence: float = 0.9,
    replay_confidence: float = 0.9,
    proof_confidence: float = 0.9,
    sync_risk: float = 0.1,
    replay_risk: float = 0.1,
    proof_risk: float = 0.1,
) -> RecoveryNodeInput:
    return RecoveryNodeInput(
        node_id=node_id,
        node_role=role,
        epoch_index=epoch,
        state_hash=_h(f"state:{node_id}:{epoch}:{role}"),
        replay_identity=_h(f"replay:{node_id}:{epoch}:{role}"),
        log_hash=_h(f"log:{node_id}:{epoch}:{role}"),
        proof_bundle_hash=_h(f"proof:{node_id}:{epoch}:{role}"),
        sync_admissible=sync_admissible,
        replay_admissible=replay_admissible,
        proof_admissible=proof_admissible,
        sync_confidence=sync_confidence,
        replay_confidence=replay_confidence,
        proof_confidence=proof_confidence,
        sync_risk=sync_risk,
        replay_risk=replay_risk,
        proof_risk=proof_risk,
    )


def _policy(**overrides: object) -> RecoveryPolicy:
    base = dict(
        minimum_sync_confidence=0.5,
        minimum_replay_confidence=0.5,
        minimum_proof_confidence=0.5,
        maximum_sync_risk=0.5,
        maximum_replay_risk=0.5,
        maximum_proof_risk=0.5,
        require_sync_admissibility=True,
        require_replay_admissibility=True,
        require_proof_admissibility=True,
        require_epoch_alignment=True,
        allow_role_mixing=False,
        allow_partial_cluster_recovery=False,
    )
    base.update(overrides)
    return RecoveryPolicy(**base)


def _status(receipt: CrossNodeRecoveryReceipt, node_id: str) -> RecoveryNodeStatus:
    return next(status for status in receipt.node_statuses if status.node_id == node_id)


def test_fully_recoverable_cluster_ready_true() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2"), _node("n3")), _policy())
    assert receipt.recovery_ready is True
    assert receipt.structurally_consistent is True


def test_replay_only_recovery_path() -> None:
    receipt = run_cross_node_recovery_runtime(
        (_node("n1"), _node("n2", replay_confidence=0.2), _node("n3")),
        _policy(allow_partial_cluster_recovery=True),
    )
    node2 = _status(receipt, "n2")
    assert node2.requires_replay_recovery is True
    assert node2.requires_proof_recovery is False


def test_proof_only_recovery_path() -> None:
    receipt = run_cross_node_recovery_runtime(
        (_node("n1"), _node("n2", proof_confidence=0.2), _node("n3")),
        _policy(allow_partial_cluster_recovery=True),
    )
    node2 = _status(receipt, "n2")
    assert node2.requires_proof_recovery is True
    assert node2.requires_replay_recovery is False


def test_full_rejoin_classification() -> None:
    receipt = run_cross_node_recovery_runtime(
        (_node("n1"), _node("n2", sync_confidence=0.2), _node("n3")),
        _policy(allow_partial_cluster_recovery=True),
    )
    status = _status(receipt, "n2")
    assert status.requires_full_rejoin is True
    assert any(
        action.action_type == "rejoin_node" and action.target_node_id == "n2"
        for action in receipt.recovery_actions
    )


def test_role_mixing_allowed_vs_blocked() -> None:
    nodes = (_node("n1", role="validator"), _node("n2", role="observer"), _node("n3", role="validator"))
    blocked = run_cross_node_recovery_runtime(nodes, _policy(allow_role_mixing=False))
    allowed = run_cross_node_recovery_runtime(nodes, _policy(allow_role_mixing=True))
    assert blocked.recovery_ready is False
    assert allowed.recovery_ready is True


def test_partial_cluster_recovery_allowed_vs_blocked() -> None:
    nodes = (_node("n1"), _node("n2", replay_confidence=0.2), _node("n3"))
    blocked = run_cross_node_recovery_runtime(nodes, _policy(allow_partial_cluster_recovery=False))
    allowed = run_cross_node_recovery_runtime(nodes, _policy(allow_partial_cluster_recovery=True))
    assert blocked.recovery_ready is False
    assert allowed.recovery_ready is True


def test_epoch_mismatch_allowed_vs_blocked() -> None:
    nodes = (_node("n1", epoch=7), _node("n2", epoch=8), _node("n3", epoch=7))
    blocked = run_cross_node_recovery_runtime(nodes, _policy(require_epoch_alignment=True, allow_partial_cluster_recovery=True))
    allowed = run_cross_node_recovery_runtime(nodes, _policy(require_epoch_alignment=False, allow_partial_cluster_recovery=True))
    assert blocked.recovery_ready is False
    assert allowed.recovery_ready is True


def test_duplicate_node_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        run_cross_node_recovery_runtime((_node("n1"), _node("n1")), _policy())


def test_malformed_hash_rejection() -> None:
    with pytest.raises(ValueError, match="state_hash"):
        _node("n1").__class__(
            node_id="n1",
            node_role="validator",
            epoch_index=1,
            state_hash="bad",
            replay_identity=_h("r"),
            log_hash=_h("l"),
            proof_bundle_hash=_h("p"),
            sync_admissible=True,
            replay_admissible=True,
            proof_admissible=True,
            sync_confidence=0.9,
            replay_confidence=0.9,
            proof_confidence=0.9,
            sync_risk=0.1,
            replay_risk=0.1,
            proof_risk=0.1,
        )


def test_deterministic_reference_node_selection() -> None:
    receipt = run_cross_node_recovery_runtime(
        (
            _node("n2", sync_confidence=1.0, replay_confidence=1.0, proof_confidence=1.0),
            _node("n1", sync_confidence=0.8, replay_confidence=0.8, proof_confidence=0.8),
        ),
        _policy(),
    )
    assert receipt.reference_node_id == "n2"


def test_deterministic_action_ordering() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n2"), _node("n1"), _node("n3")), _policy())
    assert tuple(a.action_index for a in receipt.recovery_actions) == tuple(range(len(receipt.recovery_actions)))
    assert receipt.recovery_actions[-1].action_type == "emit_recovery_view"


def test_deterministic_rationale_ordering() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    assert receipt.rationale == (
        "reference recovery node selected deterministically",
        "at least one node reports sync ok",
        "cross-node recovery ready",
    )


def test_canonical_json_stability() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    assert receipt.to_canonical_bytes() == receipt.to_canonical_json().encode("utf-8")


def test_replay_identity_and_stable_hash_determinism() -> None:
    nodes = (_node("n1"), _node("n2"), _node("n3"))
    a = run_cross_node_recovery_runtime(nodes, _policy())
    b = run_cross_node_recovery_runtime(nodes, _policy())
    assert a.replay_identity == b.replay_identity
    assert a.stable_hash == b.stable_hash
    assert a.stable_hash_value() == a.stable_hash
    assert export_cross_node_recovery_runtime_bytes(a) == export_cross_node_recovery_runtime_bytes(b)


def test_metadata_free_deterministic_behavior() -> None:
    first = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    second = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    assert first.to_canonical_json() == second.to_canonical_json()


def test_frozen_dataclass_immutability() -> None:
    node = _node("n1")
    with pytest.raises(FrozenInstanceError):
        node.node_id = "x"  # type: ignore[misc]


def test_explicit_bool_validation_for_status_and_action_fields() -> None:
    with pytest.raises(ValueError, match="blocking must be bool"):
        RecoveryAction(
            action_index=0,
            action_type="compare_sync_state",
            source_node_id="n1",
            target_node_id="n2",
            blocking=1,  # type: ignore[arg-type]
            ready=True,
            detail="x",
        )

    with pytest.raises(ValueError, match="admissible must be bool"):
        RecoveryNodeStatus(
            node_id="n1",
            admissible=1,  # type: ignore[arg-type]
            epoch_aligned=True,
            role_aligned=True,
            sync_ok=True,
            replay_ok=True,
            proof_ok=True,
            recoverable=True,
            requires_hold=False,
            requires_replay_recovery=False,
            requires_proof_recovery=False,
            requires_full_rejoin=False,
            recovery_confidence=1.0,
            recovery_risk=0.0,
            reasons=("ok",),
        )


def test_receipt_stable_hash_must_match_payload() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    with pytest.raises(ValueError, match="stable_hash must match"):
        replace(receipt, stable_hash=_h("tampered"))


def test_replay_identity_mismatch_rejected() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    with pytest.raises(ValueError, match="replay_identity mismatch"):
        replace(receipt, replay_identity=_h("tampered"))


def test_receipt_type_is_explicit() -> None:
    receipt = run_cross_node_recovery_runtime((_node("n1"), _node("n2")), _policy())
    assert isinstance(receipt, CrossNodeRecoveryReceipt)
