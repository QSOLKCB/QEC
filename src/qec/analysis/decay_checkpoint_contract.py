from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

# ---------------------------------------------------------------------------
# Error code constants – centralised so all raises and tests reference the
# same literal values and refactoring only needs to touch one place.
# ---------------------------------------------------------------------------
_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_DUPLICATE_ARTIFACT_POSITION = "DUPLICATE_ARTIFACT_POSITION"
_ERR_DECAY_SCORE_MISMATCH = "DECAY_SCORE_MISMATCH"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash_string(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return value


def _validate_artifact_position_id(value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(_ERR_INVALID_INPUT)
    return value


def _checkpoint_payload(
    artifact_position_id: str,
    expected_hash: str,
    observed_hash: str,
    drifted: bool,
) -> dict[str, Any]:
    return {
        "artifact_position_id": artifact_position_id,
        "expected_hash": expected_hash,
        "observed_hash": observed_hash,
        "drifted": drifted,
    }


@dataclass(frozen=True)
class DecayCheckpoint:
    artifact_position_id: str
    expected_hash: str
    observed_hash: str
    drifted: bool
    checkpoint_hash: str

    def __post_init__(self) -> None:
        _validate_checkpoint_integrity(self)

    def _hash_payload(self) -> dict[str, Any]:
        return _checkpoint_payload(
            artifact_position_id=self.artifact_position_id,
            expected_hash=self.expected_hash,
            observed_hash=self.observed_hash,
            drifted=self.drifted,
        )

    def _stable_hash(self) -> str:
        return _recompute_checkpoint_hash(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_position_id": self.artifact_position_id,
            "expected_hash": self.expected_hash,
            "observed_hash": self.observed_hash,
            "drifted": self.drifted,
            "checkpoint_hash": self.checkpoint_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON; not the self-hash preimage."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes; not the self-hash preimage."""
        return canonical_bytes(self.to_dict())


def _recompute_checkpoint_hash(checkpoint: DecayCheckpoint) -> str:
    return sha256_hex(
        _checkpoint_payload(
            artifact_position_id=checkpoint.artifact_position_id,
            expected_hash=checkpoint.expected_hash,
            observed_hash=checkpoint.observed_hash,
            drifted=checkpoint.drifted,
        )
    )


def _validate_checkpoint_integrity(checkpoint: DecayCheckpoint) -> None:
    _validate_artifact_position_id(checkpoint.artifact_position_id)
    _validate_hash_string(checkpoint.expected_hash)
    _validate_hash_string(checkpoint.observed_hash)
    if checkpoint.drifted is not (checkpoint.expected_hash != checkpoint.observed_hash):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(checkpoint.checkpoint_hash)
    if checkpoint.checkpoint_hash != _recompute_checkpoint_hash(checkpoint):
        raise ValueError(_ERR_HASH_MISMATCH)


def _recompute_checkpoint_set_hash(checkpoints: tuple[DecayCheckpoint, ...], decay_score: int) -> str:
    return sha256_hex(_checkpoint_set_payload(checkpoints, decay_score))


def _checkpoint_with_hash_payload(checkpoint: DecayCheckpoint) -> dict[str, Any]:
    payload = _checkpoint_payload(
        artifact_position_id=checkpoint.artifact_position_id,
        expected_hash=checkpoint.expected_hash,
        observed_hash=checkpoint.observed_hash,
        drifted=checkpoint.drifted,
    )
    payload["checkpoint_hash"] = checkpoint.checkpoint_hash
    return payload


def _checkpoint_set_payload(checkpoints: tuple[DecayCheckpoint, ...], decay_score: int) -> dict[str, Any]:
    return {
        "checkpoints": [_checkpoint_with_hash_payload(checkpoint) for checkpoint in checkpoints],
        "decay_score": decay_score,
    }


def _validate_checkpoint_set_integrity(
    checkpoints: tuple[DecayCheckpoint, ...],
    decay_score: int,
    checkpoint_set_hash: str,
) -> None:
    """Shared invariant checks used by both ``DecayCheckpointSet.__post_init__``
    and ``validate_decay_checkpoint_set`` so both paths stay in sync."""
    if not isinstance(checkpoints, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, DecayCheckpoint):
            raise ValueError(_ERR_INVALID_INPUT)
        _validate_checkpoint_integrity(checkpoint)

    position_ids = tuple(cp.artifact_position_id for cp in checkpoints)
    if len(set(position_ids)) != len(position_ids):
        raise ValueError(_ERR_DUPLICATE_ARTIFACT_POSITION)
    if position_ids != tuple(sorted(position_ids)):
        raise ValueError(_ERR_INVALID_INPUT)

    if not isinstance(decay_score, int) or isinstance(decay_score, bool):
        raise ValueError(_ERR_DECAY_SCORE_MISMATCH)
    expected_decay_score = sum(checkpoint.drifted is True for checkpoint in checkpoints)
    if decay_score != expected_decay_score:
        raise ValueError(_ERR_DECAY_SCORE_MISMATCH)

    _validate_hash_string(checkpoint_set_hash)
    if checkpoint_set_hash != _recompute_checkpoint_set_hash(checkpoints, decay_score):
        raise ValueError(_ERR_HASH_MISMATCH)


@dataclass(frozen=True)
class DecayCheckpointSet:
    checkpoints: tuple[DecayCheckpoint, ...]
    decay_score: int
    checkpoint_set_hash: str

    def __post_init__(self) -> None:
        _validate_checkpoint_set_integrity(self.checkpoints, self.decay_score, self.checkpoint_set_hash)

    def _hash_payload(self) -> dict[str, Any]:
        return _checkpoint_set_payload(self.checkpoints, self.decay_score)

    def _stable_hash(self) -> str:
        return _recompute_checkpoint_set_hash(self.checkpoints, self.decay_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
            "decay_score": self.decay_score,
            "checkpoint_set_hash": self.checkpoint_set_hash,
        }

    def to_canonical_json(self) -> str:
        """Export full canonical artifact JSON; not the self-hash preimage."""
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        """Export full canonical artifact bytes; not the self-hash preimage."""
        return canonical_bytes(self.to_dict())


def build_decay_checkpoint(
    artifact_position_id: str,
    expected_hash: str,
    observed_hash: str,
) -> DecayCheckpoint:
    _validate_artifact_position_id(artifact_position_id)
    _validate_hash_string(expected_hash)
    _validate_hash_string(observed_hash)
    drifted = expected_hash != observed_hash
    checkpoint_hash = sha256_hex(
        _checkpoint_payload(
            artifact_position_id=artifact_position_id,
            expected_hash=expected_hash,
            observed_hash=observed_hash,
            drifted=drifted,
        )
    )
    return DecayCheckpoint(
        artifact_position_id=artifact_position_id,
        expected_hash=expected_hash,
        observed_hash=observed_hash,
        drifted=drifted,
        checkpoint_hash=checkpoint_hash,
    )


def build_decay_checkpoint_set(checkpoints: list[DecayCheckpoint]) -> DecayCheckpointSet:
    if not isinstance(checkpoints, list):
        raise ValueError(_ERR_INVALID_INPUT)
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, DecayCheckpoint):
            raise ValueError(_ERR_INVALID_INPUT)

    sorted_checkpoints = tuple(sorted(checkpoints, key=lambda cp: cp.artifact_position_id))
    position_ids = tuple(cp.artifact_position_id for cp in sorted_checkpoints)
    if len(set(position_ids)) != len(position_ids):
        raise ValueError(_ERR_DUPLICATE_ARTIFACT_POSITION)
    decay_score = sum(checkpoint.drifted is True for checkpoint in sorted_checkpoints)
    checkpoint_set_hash = _recompute_checkpoint_set_hash(sorted_checkpoints, decay_score)
    return DecayCheckpointSet(
        checkpoints=sorted_checkpoints,
        decay_score=decay_score,
        checkpoint_set_hash=checkpoint_set_hash,
    )


def validate_decay_checkpoint_set(s: DecayCheckpointSet) -> bool:
    if not isinstance(s, DecayCheckpointSet):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_checkpoint_set_integrity(s.checkpoints, s.decay_score, s.checkpoint_set_hash)
    return True
