from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import re
from typing import Any

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.digital_decay_signature import (
    DigitalDecaySignature,
    validate_digital_decay_signature,
)
from qec.analysis.entropy_drift_receipt import (
    EntropyDriftReceipt,
    validate_entropy_drift_receipt,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_DECAY_RESISTANCE_IMPOSSIBLE = "DECAY_RESISTANCE_IMPOSSIBLE"
_ERR_ADVERSARIAL_POSITIONS_PRESENT = "ADVERSARIAL_POSITIONS_PRESENT"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SENTINEL = object()
_PLACEHOLDER_HASH = "0" * 64


def _validate_hash_string(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return value


def _decay_resistance_payload(
    entropy_drift_receipt_hash: str,
    replay_proof_hash: str,
    decay_class_at_proof: str,
    adversarial_positions_at_proof: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "entropy_drift_receipt_hash": entropy_drift_receipt_hash,
        "replay_proof_hash": replay_proof_hash,
        "decay_class_at_proof": decay_class_at_proof,
        "adversarial_positions_at_proof": list(adversarial_positions_at_proof),
    }


def _recompute_decay_resistance_proof_hash(proof: DecayResistanceProof) -> str:
    return sha256_hex(
        _decay_resistance_payload(
            proof.entropy_drift_receipt_hash,
            proof.replay_proof_hash,
            proof.decay_class_at_proof,
            proof.adversarial_positions_at_proof,
        )
    )


def _referenced_signature_hashes_from_entropy_receipt(
    entropy_drift_receipt: EntropyDriftReceipt,
) -> tuple[str, ...]:
    return (
        *(r.decay_signature_hash for r in entropy_drift_receipt.layer_decay_receipts),
        *(r.decay_signature_hash for r in entropy_drift_receipt.router_decay_receipts),
        *(r.decay_signature_hash for r in entropy_drift_receipt.mask_collision_decay_receipts),
        *(r.decay_signature_hash for r in entropy_drift_receipt.shift_decay_receipts),
        *(r.decay_signature_hash for r in entropy_drift_receipt.readout_projection_decay_receipts),
    )


def _validate_signature_registry_clean(
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...],
) -> dict[str, DigitalDecaySignature]:
    if not isinstance(decay_signatures, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    registry: dict[str, DigitalDecaySignature] = {}
    for signature in decay_signatures:
        if not isinstance(signature, DigitalDecaySignature):
            raise ValueError(_ERR_INVALID_INPUT)
        validate_digital_decay_signature(signature)
        if signature.digital_decay_signature_hash in registry:
            raise ValueError(_ERR_INVALID_INPUT)
        if signature.adversarial_positions != ():
            raise ValueError(_ERR_ADVERSARIAL_POSITIONS_PRESENT)
        if signature.decay_class != "CLEAN" or signature.decay_score != 0:
            raise ValueError(_ERR_DECAY_RESISTANCE_IMPOSSIBLE)
        registry[signature.digital_decay_signature_hash] = signature
    return registry


def _validate_entropy_receipt_clean(entropy_drift_receipt: EntropyDriftReceipt) -> None:
    for attr in ("adversarial_positions", "adversarial_positions_at_proof"):
        value = getattr(entropy_drift_receipt, attr, _SENTINEL)
        if value is not _SENTINEL and value != ():
            raise ValueError(_ERR_ADVERSARIAL_POSITIONS_PRESENT)
    if entropy_drift_receipt.aggregate_decay_score != 0:
        raise ValueError(_ERR_DECAY_RESISTANCE_IMPOSSIBLE)


def _extract_replay_proof_hash(replay_proof: object) -> str:
    lattice = getattr(replay_proof, "lattice_replay_proof_hash", _SENTINEL)
    replay = getattr(replay_proof, "replay_proof_hash", _SENTINEL)

    if lattice is not _SENTINEL and replay is not _SENTINEL:
        if lattice != replay:
            raise ValueError(_ERR_INVALID_INPUT)
        value = lattice
    elif lattice is not _SENTINEL:
        value = lattice
    elif replay is not _SENTINEL:
        value = replay
    else:
        raise ValueError(_ERR_INVALID_INPUT)

    return _validate_hash_string(value)


def _validate_replay_proof_artifact(replay_proof: object) -> None:
    """Require replay_proof to be a frozen dataclass; re-run __post_init__ if present."""
    if not dataclasses.is_dataclass(replay_proof) or isinstance(replay_proof, type):
        raise ValueError(_ERR_INVALID_INPUT)
    params = getattr(type(replay_proof), "__dataclass_params__", None)
    if params is None or not params.frozen:
        raise ValueError(_ERR_INVALID_INPUT)
    post_init = getattr(type(replay_proof), "__post_init__", None)
    if post_init is not None:
        try:
            post_init(replay_proof)
        except (ValueError, TypeError):
            raise ValueError(_ERR_INVALID_INPUT)


def _validate_decay_resistance_proof_integrity(proof: DecayResistanceProof) -> None:
    _validate_hash_string(proof.entropy_drift_receipt_hash)
    _validate_hash_string(proof.replay_proof_hash)
    if proof.decay_class_at_proof != "CLEAN":
        raise ValueError(_ERR_DECAY_RESISTANCE_IMPOSSIBLE)
    if not isinstance(proof.adversarial_positions_at_proof, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if proof.adversarial_positions_at_proof != ():
        raise ValueError(_ERR_ADVERSARIAL_POSITIONS_PRESENT)
    _validate_hash_string(proof.decay_resistance_proof_hash)
    if proof.decay_resistance_proof_hash != _recompute_decay_resistance_proof_hash(proof):
        raise ValueError(_ERR_HASH_MISMATCH)


@dataclass(frozen=True)
class DecayResistanceProof:
    entropy_drift_receipt_hash: str
    replay_proof_hash: str
    decay_class_at_proof: str
    adversarial_positions_at_proof: tuple[str, ...]
    decay_resistance_proof_hash: str

    def __post_init__(self) -> None:
        _validate_decay_resistance_proof_integrity(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entropy_drift_receipt_hash": self.entropy_drift_receipt_hash,
            "replay_proof_hash": self.replay_proof_hash,
            "decay_class_at_proof": self.decay_class_at_proof,
            "adversarial_positions_at_proof": list(self.adversarial_positions_at_proof),
            "decay_resistance_proof_hash": self.decay_resistance_proof_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_decay_resistance_proof(
    entropy_drift_receipt: EntropyDriftReceipt,
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...],
    replay_proof: object,
) -> DecayResistanceProof:
    if not isinstance(entropy_drift_receipt, EntropyDriftReceipt):
        raise ValueError(_ERR_INVALID_INPUT)

    registry = _validate_signature_registry_clean(decay_signatures)
    validate_entropy_drift_receipt(entropy_drift_receipt, decay_signatures)

    for signature_hash in _referenced_signature_hashes_from_entropy_receipt(entropy_drift_receipt):
        if signature_hash not in registry:
            raise ValueError(_ERR_INVALID_INPUT)

    _validate_entropy_receipt_clean(entropy_drift_receipt)
    _validate_replay_proof_artifact(replay_proof)
    replay_proof_hash = _extract_replay_proof_hash(replay_proof)

    temp = object.__new__(DecayResistanceProof)
    object.__setattr__(temp, "entropy_drift_receipt_hash", entropy_drift_receipt.entropy_drift_receipt_hash)
    object.__setattr__(temp, "replay_proof_hash", replay_proof_hash)
    object.__setattr__(temp, "decay_class_at_proof", "CLEAN")
    object.__setattr__(temp, "adversarial_positions_at_proof", ())
    object.__setattr__(temp, "decay_resistance_proof_hash", _PLACEHOLDER_HASH)
    proof_hash = _recompute_decay_resistance_proof_hash(temp)

    return DecayResistanceProof(
        entropy_drift_receipt_hash=entropy_drift_receipt.entropy_drift_receipt_hash,
        replay_proof_hash=replay_proof_hash,
        decay_class_at_proof="CLEAN",
        adversarial_positions_at_proof=(),
        decay_resistance_proof_hash=proof_hash,
    )


def validate_decay_resistance_proof(
    proof: DecayResistanceProof,
    *,
    entropy_drift_receipt: EntropyDriftReceipt | None = None,
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...] | None = None,
) -> bool:
    if not isinstance(proof, DecayResistanceProof):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_decay_resistance_proof_integrity(proof)
    if entropy_drift_receipt is not None or decay_signatures is not None:
        if entropy_drift_receipt is None or decay_signatures is None:
            raise ValueError(_ERR_INVALID_INPUT)
        if not isinstance(entropy_drift_receipt, EntropyDriftReceipt):
            raise ValueError(_ERR_INVALID_INPUT)
        registry = _validate_signature_registry_clean(decay_signatures)
        validate_entropy_drift_receipt(entropy_drift_receipt, decay_signatures)
        for sig_hash in _referenced_signature_hashes_from_entropy_receipt(entropy_drift_receipt):
            if sig_hash not in registry:
                raise ValueError(_ERR_INVALID_INPUT)
        _validate_entropy_receipt_clean(entropy_drift_receipt)
        if proof.entropy_drift_receipt_hash != entropy_drift_receipt.entropy_drift_receipt_hash:
            raise ValueError(_ERR_HASH_MISMATCH)
    return True
