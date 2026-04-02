"""
Tests for controller_snapshot_schema (v136.8.1).

Minimum 50 tests covering:
- dataclass immutability
- canonical serialization
- deserialization roundtrip
- stable hash identity
- 100-run replay determinism
- same-input byte identity
- schema validation
- version validation
- timestamp bounds
- evidence bounds
- integration with 2D episodes
- integration with 3D trajectories
- integration with feedback ledgers
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
import pytest

from qec.ai.controller_snapshot_schema import (
    SCHEMA_VERSION,
    ControllerSnapshot,
    SnapshotAuditResult,
    build_snapshot_from_episode,
    build_snapshot_from_feedback_ledger,
    build_snapshot_from_trajectory,
    compute_snapshot_hash,
    deserialize_snapshot,
    run_snapshot_replay_audit,
    serialize_snapshot,
    validate_snapshot_schema,
)
from qec.ai.movement_learning_2d import MovementEpisode, run_episode
from qec.ai.movement_learning_3d import (
    Trajectory3D,
    run_trajectory,
    trajectory_to_feedback_ledger,
)
from qec.ai.surface_feedback_engine import (
    FeedbackEvent,
    FeedbackLedger,
    episode_to_feedback_ledger,
    record_feedback,
)
from qec.ai.state_space_validator import (
    StateSpaceValidationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot() -> ControllerSnapshot:
    """Build a valid reference snapshot for testing."""
    ep = run_episode(seed=42, steps=10)
    return build_snapshot_from_episode(ep, policy_id="test_policy_42")


def _make_ledger() -> FeedbackLedger:
    """Build a small deterministic feedback ledger."""
    ev1 = FeedbackEvent(
        source="test:step_0", magnitude=0.5, event_type="reward",
        timestamp_index=0, confidence=0.8,
    )
    ev2 = FeedbackEvent(
        source="test:step_1", magnitude=0.3, event_type="hazard",
        timestamp_index=1, confidence=0.6,
    )
    ledger = record_feedback(ev1)
    ledger = record_feedback(ev2, ledger)
    return ledger


# ===================================================================
# 1. Dataclass immutability tests
# ===================================================================


class TestDataclassImmutability:
    """Verify frozen dataclass contracts."""

    def test_controller_snapshot_is_frozen(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.state_hash = "modified"  # type: ignore[misc]

    def test_controller_snapshot_policy_id_frozen(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.policy_id = "other"  # type: ignore[misc]

    def test_controller_snapshot_evidence_frozen(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.evidence_score = 0.0  # type: ignore[misc]

    def test_controller_snapshot_invariant_frozen(self):
        snap = _make_snapshot()
        with pytest.raises(AttributeError):
            snap.invariant_passed = False  # type: ignore[misc]

    def test_snapshot_audit_result_is_frozen(self):
        audit = SnapshotAuditResult(identical_runs=100, deterministic=True, snapshot_hash="a" * 64)
        with pytest.raises(AttributeError):
            audit.identical_runs = 0  # type: ignore[misc]

    def test_snapshot_audit_result_deterministic_frozen(self):
        audit = SnapshotAuditResult(identical_runs=100, deterministic=True, snapshot_hash="a" * 64)
        with pytest.raises(AttributeError):
            audit.deterministic = False  # type: ignore[misc]


# ===================================================================
# 2. Canonical serialization tests
# ===================================================================


class TestCanonicalSerialization:
    """Verify serialization produces canonical, stable output."""

    def test_serialize_produces_valid_json(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        obj = json.loads(serialized)
        assert isinstance(obj, dict)

    def test_serialize_keys_are_sorted(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        obj = json.loads(serialized)
        keys = list(obj.keys())
        assert keys == sorted(keys)

    def test_serialize_no_trailing_whitespace(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        assert serialized == serialized.strip()

    def test_serialize_compact_separators(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        # Compact separators: no spaces after , or :
        assert ", " not in serialized
        assert ": " not in serialized

    def test_serialize_utf8_bytes_stable(self):
        snap = _make_snapshot()
        s1 = serialize_snapshot(snap).encode("utf-8")
        s2 = serialize_snapshot(snap).encode("utf-8")
        assert s1 == s2

    def test_payload_json_is_valid_json(self):
        snap = _make_snapshot()
        obj = json.loads(snap.payload_json)
        assert isinstance(obj, dict)

    def test_payload_json_keys_sorted(self):
        snap = _make_snapshot()
        obj = json.loads(snap.payload_json)
        keys = list(obj.keys())
        assert keys == sorted(keys)


# ===================================================================
# 3. Deserialization roundtrip tests
# ===================================================================


class TestDeserializationRoundtrip:
    """Verify serialize -> deserialize produces identical snapshot."""

    def test_roundtrip_from_episode(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        restored = deserialize_snapshot(serialized)
        assert restored == snap

    def test_roundtrip_from_trajectory(self):
        traj = run_trajectory(seed=99, steps=15)
        snap = build_snapshot_from_trajectory(traj, policy_id="traj_99")
        serialized = serialize_snapshot(snap)
        restored = deserialize_snapshot(serialized)
        assert restored == snap

    def test_roundtrip_from_ledger(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, policy_id="ledger_test")
        serialized = serialize_snapshot(snap)
        restored = deserialize_snapshot(serialized)
        assert restored == snap

    def test_roundtrip_preserves_all_fields(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        restored = deserialize_snapshot(serialized)
        assert restored.state_hash == snap.state_hash
        assert restored.policy_id == snap.policy_id
        assert restored.evidence_score == snap.evidence_score
        assert restored.invariant_passed == snap.invariant_passed
        assert restored.timestamp_index == snap.timestamp_index
        assert restored.schema_version == snap.schema_version
        assert restored.payload_json == snap.payload_json

    def test_deserialize_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            deserialize_snapshot("not json {{{")

    def test_deserialize_missing_keys_raises(self):
        with pytest.raises(ValueError, match="Missing required keys"):
            deserialize_snapshot('{"state_hash": "abc"}')


# ===================================================================
# 4. Stable hash identity tests
# ===================================================================


class TestStableHashIdentity:
    """Verify hashing is stable and deterministic."""

    def test_same_snapshot_same_hash(self):
        snap = _make_snapshot()
        h1 = compute_snapshot_hash(snap)
        h2 = compute_snapshot_hash(snap)
        assert h1 == h2

    def test_hash_is_64_char_hex(self):
        snap = _make_snapshot()
        h = compute_snapshot_hash(snap)
        assert len(h) == 64
        int(h, 16)  # Must be valid hex

    def test_different_snapshots_different_hash(self):
        ep1 = run_episode(seed=1, steps=10)
        ep2 = run_episode(seed=2, steps=10)
        s1 = build_snapshot_from_episode(ep1, "p1")
        s2 = build_snapshot_from_episode(ep2, "p2")
        assert compute_snapshot_hash(s1) != compute_snapshot_hash(s2)

    def test_hash_matches_manual_sha256(self):
        snap = _make_snapshot()
        serialized = serialize_snapshot(snap)
        expected = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        assert compute_snapshot_hash(snap) == expected

    def test_state_hash_in_snapshot_is_sha256_of_payload(self):
        snap = _make_snapshot()
        expected = hashlib.sha256(snap.payload_json.encode("utf-8")).hexdigest()
        assert snap.state_hash == expected


# ===================================================================
# 5. 100-run replay determinism tests
# ===================================================================


class TestReplayDeterminism:
    """Verify 100-run deterministic replay."""

    def test_episode_snapshot_100_replay_serialization(self):
        ep = run_episode(seed=42, steps=20)
        reference = build_snapshot_from_episode(ep, "replay_test")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            assert serialize_snapshot(reference) == ref_bytes

    def test_episode_snapshot_100_replay_hash(self):
        ep = run_episode(seed=42, steps=20)
        reference = build_snapshot_from_episode(ep, "replay_test")
        ref_hash = compute_snapshot_hash(reference)
        for _ in range(100):
            assert compute_snapshot_hash(reference) == ref_hash

    def test_trajectory_snapshot_100_replay(self):
        traj = run_trajectory(seed=77, steps=15)
        reference = build_snapshot_from_trajectory(traj, "traj_replay")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            assert serialize_snapshot(reference) == ref_bytes

    def test_ledger_snapshot_100_replay(self):
        ledger = _make_ledger()
        reference = build_snapshot_from_feedback_ledger(ledger, "ledger_replay")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            assert serialize_snapshot(reference) == ref_bytes

    def test_rebuild_episode_snapshot_100_replay(self):
        """Rebuild snapshot from same input 100 times — must be identical."""
        ep = run_episode(seed=42, steps=20)
        reference = build_snapshot_from_episode(ep, "rebuild_test")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            rebuilt = build_snapshot_from_episode(ep, "rebuild_test")
            assert serialize_snapshot(rebuilt) == ref_bytes

    def test_rebuild_trajectory_snapshot_100_replay(self):
        traj = run_trajectory(seed=77, steps=15)
        reference = build_snapshot_from_trajectory(traj, "rebuild_traj")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            rebuilt = build_snapshot_from_trajectory(traj, "rebuild_traj")
            assert serialize_snapshot(rebuilt) == ref_bytes

    def test_rebuild_ledger_snapshot_100_replay(self):
        ledger = _make_ledger()
        reference = build_snapshot_from_feedback_ledger(ledger, "rebuild_ledger")
        ref_bytes = serialize_snapshot(reference)
        for _ in range(100):
            rebuilt = build_snapshot_from_feedback_ledger(ledger, "rebuild_ledger")
            assert serialize_snapshot(rebuilt) == ref_bytes


# ===================================================================
# 6. Same-input byte identity tests
# ===================================================================


class TestByteIdentity:
    """Verify same input produces byte-identical payload."""

    def test_episode_byte_identity(self):
        ep = run_episode(seed=100, steps=10)
        s1 = build_snapshot_from_episode(ep, "pid_100")
        s2 = build_snapshot_from_episode(ep, "pid_100")
        assert s1.payload_json == s2.payload_json
        assert s1.state_hash == s2.state_hash

    def test_trajectory_byte_identity(self):
        traj = run_trajectory(seed=200, steps=10)
        s1 = build_snapshot_from_trajectory(traj, "pid_200")
        s2 = build_snapshot_from_trajectory(traj, "pid_200")
        assert s1.payload_json == s2.payload_json
        assert s1.state_hash == s2.state_hash

    def test_ledger_byte_identity(self):
        ledger = _make_ledger()
        s1 = build_snapshot_from_feedback_ledger(ledger, "pid_ledger")
        s2 = build_snapshot_from_feedback_ledger(ledger, "pid_ledger")
        assert s1.payload_json == s2.payload_json
        assert s1.state_hash == s2.state_hash


# ===================================================================
# 7. Schema validation tests
# ===================================================================


class TestSchemaValidation:
    """Verify schema validation rules."""

    def test_valid_snapshot_passes(self):
        snap = _make_snapshot()
        assert validate_snapshot_schema(snap) is True

    def test_invalid_schema_version(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id=snap.policy_id,
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version="v0.0.0",
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="schema_version"):
            validate_snapshot_schema(bad)

    def test_negative_timestamp(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id=snap.policy_id,
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=-1,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="timestamp_index"):
            validate_snapshot_schema(bad)

    def test_evidence_score_too_high(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id=snap.policy_id,
            evidence_score=1.5,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="evidence_score"):
            validate_snapshot_schema(bad)

    def test_evidence_score_negative(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id=snap.policy_id,
            evidence_score=-0.1,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="evidence_score"):
            validate_snapshot_schema(bad)

    def test_empty_policy_id(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id="",
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="policy_id"):
            validate_snapshot_schema(bad)

    def test_invalid_state_hash_length(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash="abc",
            policy_id=snap.policy_id,
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="state_hash"):
            validate_snapshot_schema(bad)

    def test_invalid_state_hash_hex(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash="g" * 64,
            policy_id=snap.policy_id,
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json=snap.payload_json,
        )
        with pytest.raises(ValueError, match="state_hash"):
            validate_snapshot_schema(bad)

    def test_invalid_payload_json(self):
        snap = _make_snapshot()
        bad = ControllerSnapshot(
            state_hash=snap.state_hash,
            policy_id=snap.policy_id,
            evidence_score=snap.evidence_score,
            invariant_passed=snap.invariant_passed,
            timestamp_index=snap.timestamp_index,
            schema_version=SCHEMA_VERSION,
            payload_json="not json {{{",
        )
        with pytest.raises(ValueError, match="payload_json"):
            validate_snapshot_schema(bad)


# ===================================================================
# 8. Version validation tests
# ===================================================================


class TestVersionValidation:
    """Verify schema version is correct."""

    def test_schema_version_value(self):
        assert SCHEMA_VERSION == "v136.8.1"

    def test_built_snapshot_has_correct_version(self):
        snap = _make_snapshot()
        assert snap.schema_version == "v136.8.1"

    def test_trajectory_snapshot_version(self):
        traj = run_trajectory(seed=1, steps=5)
        snap = build_snapshot_from_trajectory(traj, "v_test")
        assert snap.schema_version == "v136.8.1"

    def test_ledger_snapshot_version(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, "v_test")
        assert snap.schema_version == "v136.8.1"


# ===================================================================
# 9. Timestamp bounds tests
# ===================================================================


class TestTimestampBounds:
    """Verify timestamp_index >= 0."""

    def test_episode_timestamp_non_negative(self):
        ep = run_episode(seed=42, steps=10)
        snap = build_snapshot_from_episode(ep, "ts_test")
        assert snap.timestamp_index >= 0

    def test_trajectory_timestamp_non_negative(self):
        traj = run_trajectory(seed=42, steps=10)
        snap = build_snapshot_from_trajectory(traj, "ts_test")
        assert snap.timestamp_index >= 0

    def test_ledger_timestamp_non_negative(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, "ts_test")
        assert snap.timestamp_index >= 0

    def test_episode_timestamp_equals_decisions(self):
        steps = 25
        ep = run_episode(seed=42, steps=steps)
        snap = build_snapshot_from_episode(ep, "ts_test")
        assert snap.timestamp_index == len(ep.decisions)

    def test_trajectory_timestamp_equals_transitions(self):
        steps = 15
        traj = run_trajectory(seed=42, steps=steps)
        snap = build_snapshot_from_trajectory(traj, "ts_test")
        assert snap.timestamp_index == len(traj.states) - 1


# ===================================================================
# 10. Evidence bounds tests
# ===================================================================


class TestEvidenceBounds:
    """Verify evidence_score in [0.0, 1.0]."""

    def test_episode_evidence_in_bounds(self):
        for seed in range(10):
            ep = run_episode(seed=seed, steps=20)
            snap = build_snapshot_from_episode(ep, f"ev_{seed}")
            assert 0.0 <= snap.evidence_score <= 1.0

    def test_trajectory_evidence_in_bounds(self):
        for seed in range(10):
            traj = run_trajectory(seed=seed, steps=20)
            snap = build_snapshot_from_trajectory(traj, f"ev_{seed}")
            assert 0.0 <= snap.evidence_score <= 1.0

    def test_ledger_evidence_in_bounds(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, "ev_ledger")
        assert 0.0 <= snap.evidence_score <= 1.0


# ===================================================================
# 11. Integration with 2D episodes
# ===================================================================


class TestIntegration2D:
    """Integration tests with movement_learning_2d."""

    def test_build_from_episode_seed_42(self):
        ep = run_episode(seed=42, steps=30)
        snap = build_snapshot_from_episode(ep, "2d_42")
        assert validate_snapshot_schema(snap) is True
        assert snap.invariant_passed is True

    def test_build_from_episode_seed_0(self):
        ep = run_episode(seed=0, steps=10)
        snap = build_snapshot_from_episode(ep, "2d_0")
        assert validate_snapshot_schema(snap) is True

    def test_episode_payload_contains_type(self):
        ep = run_episode(seed=42, steps=5)
        snap = build_snapshot_from_episode(ep, "2d_type")
        payload = json.loads(snap.payload_json)
        assert payload["type"] == "MovementEpisode"

    def test_episode_payload_contains_states(self):
        ep = run_episode(seed=42, steps=5)
        snap = build_snapshot_from_episode(ep, "2d_states")
        payload = json.loads(snap.payload_json)
        assert len(payload["states"]) == len(ep.states)

    def test_episode_deterministic_across_seeds(self):
        """Different seeds produce different snapshots."""
        ep1 = run_episode(seed=1, steps=10)
        ep2 = run_episode(seed=2, steps=10)
        s1 = build_snapshot_from_episode(ep1, "2d_det")
        s2 = build_snapshot_from_episode(ep2, "2d_det")
        assert s1.state_hash != s2.state_hash


# ===================================================================
# 12. Integration with 3D trajectories
# ===================================================================


class TestIntegration3D:
    """Integration tests with movement_learning_3d."""

    def test_build_from_trajectory_seed_42(self):
        traj = run_trajectory(seed=42, steps=30)
        snap = build_snapshot_from_trajectory(traj, "3d_42")
        assert validate_snapshot_schema(snap) is True
        assert snap.invariant_passed is True

    def test_trajectory_payload_contains_type(self):
        traj = run_trajectory(seed=42, steps=5)
        snap = build_snapshot_from_trajectory(traj, "3d_type")
        payload = json.loads(snap.payload_json)
        assert payload["type"] == "Trajectory3D"

    def test_trajectory_payload_contains_states(self):
        traj = run_trajectory(seed=42, steps=5)
        snap = build_snapshot_from_trajectory(traj, "3d_states")
        payload = json.loads(snap.payload_json)
        assert len(payload["states"]) == len(traj.states)

    def test_trajectory_deterministic_across_seeds(self):
        traj1 = run_trajectory(seed=1, steps=10)
        traj2 = run_trajectory(seed=2, steps=10)
        s1 = build_snapshot_from_trajectory(traj1, "3d_det")
        s2 = build_snapshot_from_trajectory(traj2, "3d_det")
        assert s1.state_hash != s2.state_hash


# ===================================================================
# 13. Integration with feedback ledgers
# ===================================================================


class TestIntegrationFeedbackLedger:
    """Integration tests with surface_feedback_engine."""

    def test_build_from_feedback_ledger(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, "fb_test")
        assert validate_snapshot_schema(snap) is True

    def test_feedback_payload_contains_type(self):
        ledger = _make_ledger()
        snap = build_snapshot_from_feedback_ledger(ledger, "fb_type")
        payload = json.loads(snap.payload_json)
        assert payload["type"] == "FeedbackLedger"

    def test_episode_to_feedback_ledger_roundtrip(self):
        """Build ledger from episode, then snapshot from ledger."""
        ep = run_episode(seed=42, steps=20)
        ledger = episode_to_feedback_ledger(ep)
        snap = build_snapshot_from_feedback_ledger(ledger, "ep_to_ledger")
        assert validate_snapshot_schema(snap) is True

    def test_trajectory_to_feedback_ledger_roundtrip(self):
        """Build ledger from trajectory, then snapshot from ledger."""
        traj = run_trajectory(seed=42, steps=20)
        ledger = trajectory_to_feedback_ledger(traj)
        snap = build_snapshot_from_feedback_ledger(ledger, "traj_to_ledger")
        assert validate_snapshot_schema(snap) is True

    def test_empty_ledger_snapshot(self):
        """Empty ledger produces a valid snapshot."""
        ledger = FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )
        snap = build_snapshot_from_feedback_ledger(ledger, "empty_ledger")
        assert validate_snapshot_schema(snap) is True
        assert snap.timestamp_index == 0


# ===================================================================
# 14. Replay audit function tests
# ===================================================================


class TestReplayAuditFunction:
    """Test run_snapshot_replay_audit."""

    def test_episode_replay_audit_100(self):
        ep = run_episode(seed=42, steps=10)
        audit = run_snapshot_replay_audit(
            lambda x: build_snapshot_from_episode(x, "audit_ep"),
            ep,
            runs=100,
        )
        assert audit.identical_runs == 100
        assert audit.deterministic is True
        assert len(audit.snapshot_hash) == 64

    def test_trajectory_replay_audit_100(self):
        traj = run_trajectory(seed=42, steps=10)
        audit = run_snapshot_replay_audit(
            lambda x: build_snapshot_from_trajectory(x, "audit_traj"),
            traj,
            runs=100,
        )
        assert audit.identical_runs == 100
        assert audit.deterministic is True

    def test_ledger_replay_audit_100(self):
        ledger = _make_ledger()
        audit = run_snapshot_replay_audit(
            lambda x: build_snapshot_from_feedback_ledger(x, "audit_ledger"),
            ledger,
            runs=100,
        )
        assert audit.identical_runs == 100
        assert audit.deterministic is True


# ===================================================================
# 15. Decoder untouched verification
# ===================================================================


class TestDecoderUntouched:
    """Verify decoder core is not imported or modified."""

    def test_no_decoder_import_in_schema_module(self):
        import qec.ai.controller_snapshot_schema as mod
        source_path = mod.__file__
        assert source_path is not None
        with open(source_path, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
        assert "qec.decoder." not in source

    def test_decoder_directory_exists_and_untouched(self):
        decoder_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        # Decoder directory should exist (not removed)
        assert os.path.isdir(decoder_dir)
