from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.decay_checkpoint_contract import (
    DecayCheckpointSet,
    validate_decay_checkpoint_set,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_DECAY_CLASS = "INVALID_DECAY_CLASS"
_ERR_INVALID_THRESHOLD = "INVALID_THRESHOLD"
_ERR_INVALID_THRESHOLD_ORDER = "INVALID_THRESHOLD_ORDER"

_ALLOWED_DECAY_CLASSES = ("CLEAN", "DEGRADED", "CORRUPTED", "ADVERSARIAL")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash_string(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return value


def _validate_threshold_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(_ERR_INVALID_THRESHOLD)
    return value


def _threshold_payload(degraded_threshold: int, corrupted_threshold: int) -> dict[str, Any]:
    return {
        "degraded_threshold": degraded_threshold,
        "corrupted_threshold": corrupted_threshold,
    }


def _signature_payload(
    checkpoint_set_hash: str,
    threshold_contract_hash: str,
    decay_class: str,
    decay_score: int,
    adversarial_positions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "checkpoint_set_hash": checkpoint_set_hash,
        "threshold_contract_hash": threshold_contract_hash,
        "decay_class": decay_class,
        "decay_score": decay_score,
        "adversarial_positions": list(adversarial_positions),
    }


def _recompute_threshold_contract_hash(contract: DecayThresholdContract) -> str:
    return sha256_hex(_threshold_payload(contract.degraded_threshold, contract.corrupted_threshold))


def _recompute_digital_decay_signature_hash(sig: DigitalDecaySignature) -> str:
    return sha256_hex(
        _signature_payload(
            checkpoint_set_hash=sig.checkpoint_set_hash,
            threshold_contract_hash=sig.threshold_contract_hash,
            decay_class=sig.decay_class,
            decay_score=sig.decay_score,
            adversarial_positions=sig.adversarial_positions,
        )
    )


def _classify_decay_score(decay_score: int, degraded_threshold: int, corrupted_threshold: int) -> str:
    if decay_score == 0:
        return "CLEAN"
    if 1 <= decay_score <= degraded_threshold:
        return "DEGRADED"
    if degraded_threshold < decay_score <= corrupted_threshold:
        return "CORRUPTED"
    return "ADVERSARIAL"


def _validate_adversarial_positions(
    checkpoint_set: DecayCheckpointSet,
    adversarial_positions: list[str],
) -> tuple[str, ...]:
    if not isinstance(adversarial_positions, list):
        raise ValueError(_ERR_INVALID_INPUT)
    for position in adversarial_positions:
        if not isinstance(position, str) or position == "":
            raise ValueError(_ERR_INVALID_INPUT)
    if len(set(adversarial_positions)) != len(adversarial_positions):
        raise ValueError(_ERR_INVALID_INPUT)
    sorted_positions = tuple(sorted(adversarial_positions))
    checkpoint_by_id = {cp.artifact_position_id: cp for cp in checkpoint_set.checkpoints}
    for position in sorted_positions:
        if position not in checkpoint_by_id:
            raise ValueError(_ERR_INVALID_INPUT)
        if checkpoint_by_id[position].drifted is not True:
            raise ValueError(_ERR_INVALID_INPUT)
    return sorted_positions


def _validate_threshold_contract_integrity(contract: DecayThresholdContract) -> None:
    _validate_threshold_int(contract.degraded_threshold)
    _validate_threshold_int(contract.corrupted_threshold)
    if contract.corrupted_threshold <= contract.degraded_threshold:
        raise ValueError(_ERR_INVALID_THRESHOLD_ORDER)
    _validate_hash_string(contract.threshold_contract_hash)
    if contract.threshold_contract_hash != _recompute_threshold_contract_hash(contract):
        raise ValueError(_ERR_HASH_MISMATCH)


def _validate_signature_integrity(sig: DigitalDecaySignature) -> None:
    _validate_hash_string(sig.checkpoint_set_hash)
    _validate_hash_string(sig.threshold_contract_hash)
    if sig.decay_class not in _ALLOWED_DECAY_CLASSES:
        raise ValueError(_ERR_INVALID_DECAY_CLASS)
    if not isinstance(sig.decay_score, int) or isinstance(sig.decay_score, bool):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(sig.adversarial_positions, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    for position in sig.adversarial_positions:
        if not isinstance(position, str) or position == "":
            raise ValueError(_ERR_INVALID_INPUT)
    if tuple(sorted(sig.adversarial_positions)) != sig.adversarial_positions:
        raise ValueError(_ERR_INVALID_INPUT)
    if len(set(sig.adversarial_positions)) != len(sig.adversarial_positions):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_hash_string(sig.digital_decay_signature_hash)
    if sig.digital_decay_signature_hash != _recompute_digital_decay_signature_hash(sig):
        raise ValueError(_ERR_HASH_MISMATCH)


@dataclass(frozen=True)
class DecayThresholdContract:
    degraded_threshold: int
    corrupted_threshold: int
    threshold_contract_hash: str

    def __post_init__(self) -> None:
        _validate_threshold_contract_integrity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded_threshold": self.degraded_threshold,
            "corrupted_threshold": self.corrupted_threshold,
            "threshold_contract_hash": self.threshold_contract_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class DigitalDecaySignature:
    checkpoint_set_hash: str
    threshold_contract_hash: str
    decay_class: str
    decay_score: int
    adversarial_positions: tuple[str, ...]
    digital_decay_signature_hash: str

    def __post_init__(self) -> None:
        _validate_signature_integrity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_set_hash": self.checkpoint_set_hash,
            "threshold_contract_hash": self.threshold_contract_hash,
            "decay_class": self.decay_class,
            "decay_score": self.decay_score,
            "adversarial_positions": list(self.adversarial_positions),
            "digital_decay_signature_hash": self.digital_decay_signature_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_decay_threshold_contract(degraded_threshold: int, corrupted_threshold: int) -> DecayThresholdContract:
    _validate_threshold_int(degraded_threshold)
    _validate_threshold_int(corrupted_threshold)
    if corrupted_threshold <= degraded_threshold:
        raise ValueError(_ERR_INVALID_THRESHOLD_ORDER)
    threshold_contract_hash = sha256_hex(_threshold_payload(degraded_threshold, corrupted_threshold))
    return DecayThresholdContract(
        degraded_threshold=degraded_threshold,
        corrupted_threshold=corrupted_threshold,
        threshold_contract_hash=threshold_contract_hash,
    )


def build_digital_decay_signature(
    checkpoint_set: DecayCheckpointSet,
    threshold_contract: DecayThresholdContract,
    adversarial_positions: list[str],
) -> DigitalDecaySignature:
    validate_decay_checkpoint_set(checkpoint_set)
    _validate_threshold_contract_integrity(threshold_contract)
    sorted_positions = _validate_adversarial_positions(checkpoint_set, adversarial_positions)
    decay_class = _classify_decay_score(
        checkpoint_set.decay_score,
        threshold_contract.degraded_threshold,
        threshold_contract.corrupted_threshold,
    )
    payload = _signature_payload(
        checkpoint_set_hash=checkpoint_set.checkpoint_set_hash,
        threshold_contract_hash=threshold_contract.threshold_contract_hash,
        decay_class=decay_class,
        decay_score=checkpoint_set.decay_score,
        adversarial_positions=sorted_positions,
    )
    return DigitalDecaySignature(
        checkpoint_set_hash=checkpoint_set.checkpoint_set_hash,
        threshold_contract_hash=threshold_contract.threshold_contract_hash,
        decay_class=decay_class,
        decay_score=checkpoint_set.decay_score,
        adversarial_positions=sorted_positions,
        digital_decay_signature_hash=sha256_hex(payload),
    )


def validate_digital_decay_signature(sig: DigitalDecaySignature) -> bool:
    if not isinstance(sig, DigitalDecaySignature):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_signature_integrity(sig)
    return True
