from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from qec.analysis.replay_log_consensus_layer import (
    ALLOWED_CONSENSUS_ACTION_TYPES,
    NodeReplayLog,
    ReplayConsensusPolicy,
    ReplayLogEntry,
    build_replay_log_consensus_receipt,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _entry(index: int, suffix: str) -> ReplayLogEntry:
    return ReplayLogEntry(
        sequence_index=index,
        event_type=f"event-{suffix}",
        event_hash=_h(f"event-{suffix}"),
        replay_identity=_h(f"replay-{suffix}"),
        payload_hash=_h(f"payload-{suffix}"),
        deterministic_order_key=f"key-{suffix}",
    )


def _log(node_id: str, epoch: int, entry_suffixes: tuple[str, ...], role: str = "worker", metadata: dict[str, str] | None = None) -> NodeReplayLog:
    entries = tuple(_entry(i, suffix) for i, suffix in enumerate(entry_suffixes))
    return NodeReplayLog(
        node_id=node_id,
        node_role=role,
        epoch_index=epoch,
        entries=entries,
        log_hash=_h("|".join(entry_suffixes)),
        metadata=metadata,
    )


def _policy(**overrides: object) -> ReplayConsensusPolicy:
    base: dict[str, object] = {
        "require_matching_epoch": True,
        "require_full_prefix_agreement": True,
        "allow_length_skew": False,
        "maximum_length_delta": 0,
        "require_matching_log_hash": True,
        "minimum_consensus_fraction": 1.0,
        "allow_role_mixing": True,
    }
    base.update(overrides)
    return ReplayConsensusPolicy(**base)


def test_fully_aligned_logs_consensus_ready() -> None:
    logs = (
        _log("node-b", 4, ("a", "b")),
        _log("node-a", 4, ("a", "b")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy())
    assert receipt.consensus_ready is True
    assert receipt.structurally_consistent is True
    assert receipt.reference_node_id == "node-a"


def test_epoch_mismatch_allowed_by_policy() -> None:
    logs = (
        _log("node-a", 2, ("a", "b")),
        _log("node-b", 3, ("a", "b")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_epoch=False, require_matching_log_hash=False))
    assert receipt.consensus_ready is True
    assert any("epoch mismatch tolerated" in reason for reason in receipt.rationale)


def test_epoch_mismatch_blocked_by_policy() -> None:
    logs = (
        _log("node-a", 2, ("a", "b")),
        _log("node-b", 3, ("a", "b")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    assert receipt.consensus_ready is False
    assert "epoch mismatch blocks consensus readiness" in receipt.rationale


def test_log_hash_mismatch_allowed_vs_blocked() -> None:
    node_a = _log("node-a", 1, ("a", "b"))
    node_b = NodeReplayLog(
        node_id="node-b",
        node_role=node_a.node_role,
        epoch_index=node_a.epoch_index,
        entries=node_a.entries,
        log_hash=_h("different-log-hash"),
        metadata=None,
    )
    logs = (node_a, node_b)
    blocked = build_replay_log_consensus_receipt(logs, _policy())
    allowed = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    assert blocked.consensus_ready is False
    assert allowed.consensus_ready is True


def test_prefix_divergence_returns_receipt_not_value_error() -> None:
    logs = (
        _log("node-a", 1, ("a", "b", "c")),
        _log("node-b", 1, ("a", "x", "c")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    assert receipt.consensus_ready is False
    assert "prefix divergence blocks consensus readiness" in receipt.rationale


def test_duplicate_node_rejected() -> None:
    logs = (
        _log("node-a", 1, ("a",)),
        _log("node-a", 1, ("a",)),
    )
    with pytest.raises(ValueError, match="duplicate node_id"):
        build_replay_log_consensus_receipt(logs, _policy())


def test_malformed_hash_rejected() -> None:
    with pytest.raises(ValueError, match="event_hash"):
        ReplayLogEntry(
            sequence_index=0,
            event_type="x",
            event_hash="bad",
            replay_identity=_h("ok"),
            payload_hash=_h("ok2"),
            deterministic_order_key="k",
        )


def test_non_monotonic_entry_ordering_rejected() -> None:
    a = _entry(1, "a")
    b = _entry(0, "b")
    with pytest.raises(ValueError, match="ordered by sequence_index"):
        NodeReplayLog(node_id="n", node_role="r", epoch_index=0, entries=(a, b), log_hash=_h("l"))


def test_deterministic_reference_log_selection() -> None:
    logs = (
        _log("node-z", 7, ("a", "b")),
        _log("node-a", 7, ("a", "b", "c")),
        _log("node-b", 7, ("a", "b", "c")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False, allow_length_skew=True, maximum_length_delta=4, minimum_consensus_fraction=0.66))
    assert receipt.reference_node_id == "node-a"


def test_deterministic_action_ordering() -> None:
    logs = (
        _log("node-b", 1, ("a", "b")),
        _log("node-a", 1, ("a", "b")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    action_types = tuple(action.action_type for action in receipt.consensus_actions)
    assert action_types[:6] == (
        "compare_prefix",
        "compare_log_hash",
        "align_epoch",
        "hold_node",
        "admit_log",
        "flag_divergence",
    )
    assert action_types[-1] == "emit_consensus_view"
    assert set(action_types).issubset(set(ALLOWED_CONSENSUS_ACTION_TYPES))


def test_deterministic_rationale_ordering() -> None:
    logs = (
        _log("node-a", 1, ("a", "b")),
        _log("node-b", 2, ("a", "z")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    assert receipt.rationale[0] == "reference log selected deterministically"
    assert receipt.rationale[-1] in {"structurally consistent", "structural consistency violated"}


def test_canonical_json_stability() -> None:
    logs = (
        _log("node-a", 1, ("a", "b")),
        _log("node-b", 1, ("a", "b")),
    )
    receipt = build_replay_log_consensus_receipt(logs, _policy(require_matching_log_hash=False))
    assert receipt.to_canonical_json() == receipt.to_canonical_bytes().decode("utf-8")


def test_replay_identity_and_stable_hash_determinism() -> None:
    logs = (
        _log("node-a", 1, ("a", "b")),
        _log("node-b", 1, ("a", "b")),
    )
    policy = _policy(require_matching_log_hash=False)
    first = build_replay_log_consensus_receipt(logs, policy)
    second = build_replay_log_consensus_receipt(logs, policy)
    assert first.replay_identity == second.replay_identity
    assert first.stable_hash == second.stable_hash
    assert first.stable_hash == first.stable_hash_value()


def test_metadata_immutability() -> None:
    log = _log("node-a", 1, ("a",), metadata={"z": "2", "a": "1"})
    assert tuple(log.metadata.keys()) == ("a", "z")
    with pytest.raises(TypeError):
        log.metadata["new"] = "x"  # type: ignore[index]


def test_frozen_dataclass_immutability() -> None:
    log = _log("node-a", 1, ("a",))
    with pytest.raises(FrozenInstanceError):
        log.node_id = "mutated"  # type: ignore[misc]
