from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.distributed_state_synchronization_bus import (
    DISTRIBUTED_SYNC_BUS_SCHEMA_VERSION,
    NodeStateSnapshot,
    SyncBusPolicy,
    build_distributed_sync_receipt,
)


def _h(ch: str) -> str:
    return ch * 64


def _node(
    node_id: str,
    *,
    role: str = "worker",
    epoch: int = 3,
    state_hash: str | None = None,
    replay_identity: str | None = None,
    stability: float = 0.9,
    projected_loss: float = 0.1,
    hardware_alignment: float = 0.9,
) -> NodeStateSnapshot:
    return NodeStateSnapshot(
        node_id=node_id,
        node_role=role,
        epoch_index=epoch,
        state_hash=state_hash or _h("a"),
        replay_identity=replay_identity or _h("b"),
        logical_stability=stability,
        projected_loss=projected_loss,
        hardware_alignment=hardware_alignment,
        execution_efficiency=0.85,
        orchestration_depth=2,
        metadata={"site": "lab", "rack": "r1"},
    )


def _policy(**overrides) -> SyncBusPolicy:
    base = {
        "require_matching_epoch": True,
        "require_matching_state_hash": True,
        "maximum_projected_loss_delta": 0.2,
        "minimum_logical_stability": 0.7,
        "minimum_hardware_alignment": 0.7,
        "allow_role_mixing": True,
    }
    base.update(overrides)
    return SyncBusPolicy(**base)


def test_fully_aligned_cluster_ready_success():
    receipt = build_distributed_sync_receipt((_node("n2"), _node("n1")), _policy())
    assert receipt.cluster_ready is True
    assert receipt.structurally_consistent is True
    assert receipt.schema_version == DISTRIBUTED_SYNC_BUS_SCHEMA_VERSION


def test_epoch_mismatch_cluster_not_ready_without_error():
    receipt = build_distributed_sync_receipt((_node("n1", epoch=2), _node("n2", epoch=3)), _policy())
    assert receipt.cluster_ready is False
    assert "epoch mismatch blocks cluster readiness" in receipt.rationale


def test_state_hash_mismatch_cluster_not_ready_without_error():
    receipt = build_distributed_sync_receipt((_node("n1", state_hash=_h("a")), _node("n2", state_hash=_h("c"))), _policy())
    assert receipt.cluster_ready is False
    assert "state hash mismatch blocks cluster readiness" in receipt.rationale


def test_epoch_mismatch_allowed_by_policy():
    receipt = build_distributed_sync_receipt(
        (_node("n1", epoch=2), _node("n2", epoch=3)),
        _policy(require_matching_epoch=False),
    )
    assert receipt.cluster_ready is True
    assert "epoch mismatch noted but allowed by policy" in receipt.rationale


def test_epoch_mismatch_blocked_by_policy():
    receipt = build_distributed_sync_receipt(
        (_node("n1", epoch=2), _node("n2", epoch=3)),
        _policy(require_matching_epoch=True),
    )
    assert receipt.cluster_ready is False
    assert "epoch mismatch blocks cluster readiness" in receipt.rationale


def test_reference_node_epoch_alignment_when_required():
    receipt = build_distributed_sync_receipt(
        (
            _node("n1", epoch=4, stability=0.99, hardware_alignment=0.99, state_hash=_h("c")),
            _node("n2", epoch=3, stability=0.98, hardware_alignment=0.98, state_hash=_h("a")),
            _node("n3", epoch=3, stability=0.97, hardware_alignment=0.97, state_hash=_h("b")),
        ),
        _policy(require_matching_epoch=True, require_matching_state_hash=False),
    )

    emit_action = next(action for action in receipt.sync_actions if action.action_type == "emit_cluster_view")
    assert emit_action.source_node_id == "n2"
    assert receipt.cluster_epoch == 3


def test_duplicate_node_rejected():
    with pytest.raises(ValueError, match="duplicate node_id"):
        build_distributed_sync_receipt((_node("n1"), _node("n1")), _policy())


def test_malformed_hash_rejected():
    with pytest.raises(ValueError, match="state_hash"):
        _node("n1", state_hash="xyz")


def test_deterministic_reference_node_selection():
    receipt = build_distributed_sync_receipt(
        (
            _node("n2", stability=0.95, hardware_alignment=0.8, projected_loss=0.1, state_hash=_h("d")),
            _node("n1", stability=0.95, hardware_alignment=0.8, projected_loss=0.1, state_hash=_h("e")),
            _node("n3", stability=0.92, hardware_alignment=0.99, projected_loss=0.01, state_hash=_h("f")),
        ),
        _policy(require_matching_state_hash=False),
    )
    assert receipt.cluster_state_hash == _h("e")


def test_deterministic_action_ordering():
    receipt = build_distributed_sync_receipt((_node("n3"), _node("n1"), _node("n2")), _policy())
    indices = [action.action_index for action in receipt.sync_actions]
    assert indices == list(range(len(receipt.sync_actions)))
    first_eight = [action.action_type for action in receipt.sync_actions[:8]]
    assert first_eight == [
        "compare_state",
        "compare_replay",
        "align_epoch",
        "align_hash",
        "compare_state",
        "compare_replay",
        "align_epoch",
        "align_hash",
    ]


def test_deterministic_rationale_ordering():
    receipt = build_distributed_sync_receipt((_node("n1"), _node("n2", epoch=9)), _policy())
    assert receipt.rationale[0] == "reference node selected deterministically"
    assert receipt.rationale[-1] == "cluster synchronization not ready"


def test_canonical_json_stability_and_hash_determinism():
    receipt_a = build_distributed_sync_receipt((_node("n1"), _node("n2")), _policy())
    receipt_b = build_distributed_sync_receipt((_node("n1"), _node("n2")), _policy())
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.replay_identity == receipt_b.replay_identity
    assert receipt_a.stable_hash == receipt_b.stable_hash
    assert receipt_a.stable_hash_value() == receipt_a.stable_hash


def test_metadata_immutable_after_init():
    snap = _node("n1")
    assert snap.metadata is not None
    with pytest.raises(TypeError):
        snap.metadata["new"] = "value"  # type: ignore[index]


def test_frozen_dataclass_immutability():
    snap = _node("n1")
    with pytest.raises(FrozenInstanceError):
        snap.node_id = "x"  # type: ignore[misc]
