from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.decay_checkpoint_contract import (
    DecayCheckpoint,
    DecayCheckpointSet,
    build_decay_checkpoint,
    build_decay_checkpoint_set,
    validate_decay_checkpoint_set,
)


def _h(c: str) -> str:
    return c * 64


def test_decay_checkpoint_determinism() -> None:
    checkpoints = [build_decay_checkpoint("p0", _h("a"), _h("b")) for _ in range(10)]
    hashes = {cp.checkpoint_hash for cp in checkpoints}
    assert len(hashes) == 1


def test_decay_checkpoint_drifted_false_when_hashes_match() -> None:
    cp = build_decay_checkpoint("p1", _h("a"), _h("a"))
    assert cp.drifted is False


def test_decay_checkpoint_drifted_true_when_hashes_differ() -> None:
    cp = build_decay_checkpoint("p1", _h("a"), _h("b"))
    assert cp.drifted is True


def test_invalid_hash_format_rejected() -> None:
    invalid_hashes = ["a" * 63, "a" * 65, "A" * 64, "g" * 64, 123]
    for bad in invalid_hashes:
        with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
            build_decay_checkpoint("p1", bad, _h("a"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
            build_decay_checkpoint("p1", _h("a"), bad)  # type: ignore[arg-type]


def test_partial_hash_matching_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_decay_checkpoint("p1", "a" * 63, _h("b"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_decay_checkpoint("p1", "a" * 65, _h("b"))


def test_artifact_position_id_required() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_checkpoint("", _h("a"), _h("a"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_decay_checkpoint(10, _h("a"), _h("a"))  # type: ignore[arg-type]


def test_checkpoint_set_sorting_determinism() -> None:
    a = build_decay_checkpoint("b", _h("a"), _h("a"))
    b = build_decay_checkpoint("a", _h("a"), _h("b"))
    set1 = build_decay_checkpoint_set([a, b])
    set2 = build_decay_checkpoint_set([b, a])
    assert tuple(cp.artifact_position_id for cp in set1.checkpoints) == ("a", "b")
    assert set1.checkpoint_set_hash == set2.checkpoint_set_hash


def test_duplicate_artifact_position_rejected() -> None:
    a = build_decay_checkpoint("x", _h("a"), _h("a"))
    b = build_decay_checkpoint("x", _h("a"), _h("b"))
    with pytest.raises(ValueError, match="DUPLICATE_ARTIFACT_POSITION"):
        build_decay_checkpoint_set([a, b])


def test_decay_score_exact_count() -> None:
    s = build_decay_checkpoint_set(
        [
            build_decay_checkpoint("a", _h("a"), _h("a")),
            build_decay_checkpoint("b", _h("a"), _h("b")),
            build_decay_checkpoint("c", _h("a"), _h("c")),
        ]
    )
    assert s.decay_score == 2


def test_decay_score_mismatch_detected() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    expected_hash = build_decay_checkpoint_set([cp]).checkpoint_set_hash
    with pytest.raises(ValueError, match="DECAY_SCORE_MISMATCH"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=0, checkpoint_set_hash=expected_hash)


def test_checkpoint_hash_tamper_detected() -> None:
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        DecayCheckpoint(
            artifact_position_id="a",
            expected_hash=_h("a"),
            observed_hash=_h("b"),
            drifted=True,
            checkpoint_hash="0" * 64,
        )


def test_checkpoint_set_hash_tamper_detected() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=1, checkpoint_set_hash="0" * 64)


def test_drifted_field_cannot_be_spoofed() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DecayCheckpoint(
            artifact_position_id="a",
            expected_hash=_h("a"),
            observed_hash=_h("a"),
            drifted=True,
            checkpoint_hash=build_decay_checkpoint("a", _h("a"), _h("a")).checkpoint_hash,
        )


def test_no_float_or_probability_semantics() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    expected_hash = build_decay_checkpoint_set([cp]).checkpoint_set_hash
    with pytest.raises(ValueError, match="DECAY_SCORE_MISMATCH"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=1.0, checkpoint_set_hash=expected_hash)
    with pytest.raises(ValueError, match="DECAY_SCORE_MISMATCH"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=True, checkpoint_set_hash=expected_hash)
    artifact = build_decay_checkpoint_set([cp]).to_dict()
    assert "probability" not in artifact
    assert "threshold" not in artifact
    assert "decay_class" not in artifact


def test_artifacts_are_frozen() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    with pytest.raises(FrozenInstanceError):
        cp.drifted = False  # type: ignore[misc]


def test_cross_run_stability() -> None:
    cp1 = build_decay_checkpoint("a", _h("a"), _h("b"))
    cp2 = build_decay_checkpoint("a", _h("a"), _h("b"))
    s1 = build_decay_checkpoint_set([cp1])
    s2 = build_decay_checkpoint_set([cp2])

    assert cp1.to_canonical_json() == cp2.to_canonical_json()
    assert cp1.to_canonical_bytes() == cp2.to_canonical_bytes()
    assert cp1.checkpoint_hash == cp2.checkpoint_hash
    assert s1.to_canonical_json() == s2.to_canonical_json()
    assert s1.to_canonical_bytes() == s2.to_canonical_bytes()
    assert s1.checkpoint_set_hash == s2.checkpoint_set_hash
    assert validate_decay_checkpoint_set(s1) is True


def test_child_checkpoint_tamper_detected_inside_set() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    object.__setattr__(cp, "observed_hash", _h("c"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        build_decay_checkpoint_set([cp])


def test_malformed_child_checkpoint_hash_inside_set() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    object.__setattr__(cp, "checkpoint_hash", "bad")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        build_decay_checkpoint_set([cp])


def test_malformed_decay_checkpoint_self_hash_rejected() -> None:
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DecayCheckpoint(
            artifact_position_id="a",
            expected_hash=_h("a"),
            observed_hash=_h("b"),
            drifted=True,
            checkpoint_hash="bad",
        )


def test_malformed_decay_checkpoint_set_self_hash_rejected() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("b"))
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=1, checkpoint_set_hash="bad")


def test_duplicate_unsorted_direct_construction_precedence() -> None:
    cp_b = build_decay_checkpoint("b", _h("a"), _h("a"))
    cp_a_1 = build_decay_checkpoint("a", _h("a"), _h("b"))
    cp_a_2 = build_decay_checkpoint("a", _h("a"), _h("c"))
    with pytest.raises(ValueError, match="DUPLICATE_ARTIFACT_POSITION"):
        DecayCheckpointSet(checkpoints=(cp_b, cp_a_1, cp_a_2), decay_score=2, checkpoint_set_hash="0" * 64)


def test_direct_unsorted_non_duplicate_construction_rejected() -> None:
    cp_b = build_decay_checkpoint("b", _h("a"), _h("a"))
    cp_a = build_decay_checkpoint("a", _h("a"), _h("b"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DecayCheckpointSet(checkpoints=(cp_b, cp_a), decay_score=1, checkpoint_set_hash="0" * 64)


def test_validate_decay_checkpoint_set_rejects_tampered_set() -> None:
    s = build_decay_checkpoint_set([build_decay_checkpoint("a", _h("a"), _h("b"))])
    object.__setattr__(s, "checkpoint_set_hash", "0" * 64)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_decay_checkpoint_set(s)


def test_bool_as_int_decay_score_rejected_when_expected_zero() -> None:
    cp = build_decay_checkpoint("a", _h("a"), _h("a"))
    valid = build_decay_checkpoint_set([cp])
    with pytest.raises(ValueError, match="DECAY_SCORE_MISMATCH"):
        DecayCheckpointSet(checkpoints=(cp,), decay_score=True, checkpoint_set_hash=valid.checkpoint_set_hash)
