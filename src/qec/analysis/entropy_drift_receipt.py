from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from qec.analysis.digital_decay_signature import (
    DigitalDecaySignature,
    validate_digital_decay_signature,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_MISSING_SUBSYSTEM_DECAY_RECEIPTS = "MISSING_SUBSYSTEM_DECAY_RECEIPTS"
_ERR_AGGREGATE_SCORE_MISMATCH = "AGGREGATE_SCORE_MISMATCH"

_ALLOWED_COLLISION_TYPES = (
    "NO_COLLISION",
    "KNOWN_EQUIVALENT_COLLISION",
    "INVALID_COLLISION",
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_hash_string(value: Any) -> str:
    if not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return value


def _validate_exact_non_negative_int(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(_ERR_AGGREGATE_SCORE_MISMATCH)
    return value


def _layer_decay_receipt_payload(receipt: LayerDecayReceipt) -> dict[str, Any]:
    return {
        "layered_hash": receipt.layered_hash,
        "expected_layered_hash": receipt.expected_layered_hash,
        "decay_signature_hash": receipt.decay_signature_hash,
    }


def _router_decay_receipt_payload(receipt: RouterDecayReceipt) -> dict[str, Any]:
    return {
        "router_lattice_path_receipt_hash": receipt.router_lattice_path_receipt_hash,
        "expected_router_hash": receipt.expected_router_hash,
        "decay_signature_hash": receipt.decay_signature_hash,
    }


def _mask_collision_decay_receipt_payload(
    receipt: MaskCollisionDecayReceipt,
) -> dict[str, Any]:
    return {
        "mask_reduction_receipt_hash": receipt.mask_reduction_receipt_hash,
        "expected_mask_hash": receipt.expected_mask_hash,
        "collision_type": receipt.collision_type,
        "decay_signature_hash": receipt.decay_signature_hash,
    }


def _shift_decay_receipt_payload(receipt: ShiftDecayReceipt) -> dict[str, Any]:
    return {
        "shift_projection_receipt_hash": receipt.shift_projection_receipt_hash,
        "expected_shift_hash": receipt.expected_shift_hash,
        "decay_signature_hash": receipt.decay_signature_hash,
    }


def _readout_projection_decay_receipt_payload(
    receipt: ReadoutProjectionDecayReceipt,
) -> dict[str, Any]:
    return {
        "readout_projection_receipt_hash": receipt.readout_projection_receipt_hash,
        "expected_readout_hash": receipt.expected_readout_hash,
        "decay_signature_hash": receipt.decay_signature_hash,
    }


def _layer_decay_receipt_with_hash_payload(receipt: LayerDecayReceipt) -> dict[str, Any]:
    p = _layer_decay_receipt_payload(receipt)
    p["layer_decay_receipt_hash"] = receipt.layer_decay_receipt_hash
    return p


def _router_decay_receipt_with_hash_payload(
    receipt: RouterDecayReceipt,
) -> dict[str, Any]:
    p = _router_decay_receipt_payload(receipt)
    p["router_decay_receipt_hash"] = receipt.router_decay_receipt_hash
    return p


def _mask_collision_decay_receipt_with_hash_payload(
    receipt: MaskCollisionDecayReceipt,
) -> dict[str, Any]:
    p = _mask_collision_decay_receipt_payload(receipt)
    p["mask_collision_decay_receipt_hash"] = receipt.mask_collision_decay_receipt_hash
    return p


def _shift_decay_receipt_with_hash_payload(receipt: ShiftDecayReceipt) -> dict[str, Any]:
    p = _shift_decay_receipt_payload(receipt)
    p["shift_decay_receipt_hash"] = receipt.shift_decay_receipt_hash
    return p


def _readout_projection_decay_receipt_with_hash_payload(
    receipt: ReadoutProjectionDecayReceipt,
) -> dict[str, Any]:
    p = _readout_projection_decay_receipt_payload(receipt)
    p["readout_projection_decay_receipt_hash"] = receipt.readout_projection_decay_receipt_hash
    return p


def _recompute_layer_decay_receipt_hash(receipt: LayerDecayReceipt) -> str:
    return sha256_hex(_layer_decay_receipt_payload(receipt))


def _recompute_router_decay_receipt_hash(receipt: RouterDecayReceipt) -> str:
    return sha256_hex(_router_decay_receipt_payload(receipt))


def _recompute_mask_collision_decay_receipt_hash(
    receipt: MaskCollisionDecayReceipt,
) -> str:
    return sha256_hex(_mask_collision_decay_receipt_payload(receipt))


def _recompute_shift_decay_receipt_hash(receipt: ShiftDecayReceipt) -> str:
    return sha256_hex(_shift_decay_receipt_payload(receipt))


def _recompute_readout_projection_decay_receipt_hash(
    receipt: ReadoutProjectionDecayReceipt,
) -> str:
    return sha256_hex(_readout_projection_decay_receipt_payload(receipt))


def _entropy_drift_receipt_payload(receipt: EntropyDriftReceipt) -> dict[str, Any]:
    return {
        "layer_decay_receipts": [
            _layer_decay_receipt_with_hash_payload(r) for r in receipt.layer_decay_receipts
        ],
        "router_decay_receipts": [
            _router_decay_receipt_with_hash_payload(r)
            for r in receipt.router_decay_receipts
        ],
        "mask_collision_decay_receipts": [
            _mask_collision_decay_receipt_with_hash_payload(r)
            for r in receipt.mask_collision_decay_receipts
        ],
        "shift_decay_receipts": [
            _shift_decay_receipt_with_hash_payload(r) for r in receipt.shift_decay_receipts
        ],
        "readout_projection_decay_receipts": [
            _readout_projection_decay_receipt_with_hash_payload(r)
            for r in receipt.readout_projection_decay_receipts
        ],
        "aggregate_decay_score": receipt.aggregate_decay_score,
    }


def _recompute_entropy_drift_receipt_hash(receipt: EntropyDriftReceipt) -> str:
    return sha256_hex(_entropy_drift_receipt_payload(receipt))


def _validate_signature_registry(
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...],
) -> dict[str, DigitalDecaySignature]:
    if not isinstance(decay_signatures, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    by_hash: dict[str, DigitalDecaySignature] = {}
    for sig in decay_signatures:
        if not isinstance(sig, DigitalDecaySignature):
            raise ValueError(_ERR_INVALID_INPUT)
        validate_digital_decay_signature(sig)
        if sig.digital_decay_signature_hash in by_hash:
            raise ValueError(_ERR_INVALID_INPUT)
        by_hash[sig.digital_decay_signature_hash] = sig
    return by_hash


def _signature_score_for_hash(
    decay_signature_hash: str,
    by_hash: dict[str, DigitalDecaySignature],
) -> int:
    sig = by_hash.get(decay_signature_hash)
    if sig is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return sig.decay_score


def _validate_subsystem_receipt_sequence(
    seq: Any,
    expected_type: type,
    hash_attr: str,
    validator: Callable[[Any], bool],
    missing_error: bool,
    require_sorted: bool,
) -> tuple[Any, ...]:
    if seq is None or not isinstance(seq, (list, tuple)):
        raise ValueError(
            _ERR_MISSING_SUBSYSTEM_DECAY_RECEIPTS if missing_error else _ERR_INVALID_INPUT
        )
    if len(seq) == 0:
        raise ValueError(
            _ERR_MISSING_SUBSYSTEM_DECAY_RECEIPTS if missing_error else _ERR_INVALID_INPUT
        )
    items = tuple(seq)
    for item in items:
        if not isinstance(item, expected_type):
            raise ValueError(_ERR_INVALID_INPUT)
    for item in items:
        validator(item)

    hashes: list[str] = []
    for item in items:
        hash_value = getattr(item, hash_attr)
        _validate_hash_string(hash_value)
        hashes.append(hash_value)

    if len(set(hashes)) != len(hashes):
        raise ValueError(_ERR_INVALID_INPUT)
    if require_sorted and tuple(sorted(hashes)) != tuple(hashes):
        raise ValueError(_ERR_INVALID_INPUT)
    return items


@dataclass(frozen=True)
class LayerDecayReceipt:
    layered_hash: str
    expected_layered_hash: str
    decay_signature_hash: str
    layer_decay_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_hash_string(self.layered_hash)
        _validate_hash_string(self.expected_layered_hash)
        _validate_hash_string(self.decay_signature_hash)
        _validate_hash_string(self.layer_decay_receipt_hash)
        if self.layer_decay_receipt_hash != _recompute_layer_decay_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return _layer_decay_receipt_with_hash_payload(self)

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class RouterDecayReceipt:
    router_lattice_path_receipt_hash: str
    expected_router_hash: str
    decay_signature_hash: str
    router_decay_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_hash_string(self.router_lattice_path_receipt_hash)
        _validate_hash_string(self.expected_router_hash)
        _validate_hash_string(self.decay_signature_hash)
        _validate_hash_string(self.router_decay_receipt_hash)
        if self.router_decay_receipt_hash != _recompute_router_decay_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return _router_decay_receipt_with_hash_payload(self)

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class MaskCollisionDecayReceipt:
    mask_reduction_receipt_hash: str
    expected_mask_hash: str
    collision_type: str
    decay_signature_hash: str
    mask_collision_decay_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_hash_string(self.mask_reduction_receipt_hash)
        _validate_hash_string(self.expected_mask_hash)
        _validate_hash_string(self.decay_signature_hash)
        _validate_hash_string(self.mask_collision_decay_receipt_hash)
        if self.collision_type not in _ALLOWED_COLLISION_TYPES:
            raise ValueError(_ERR_INVALID_INPUT)
        if self.mask_collision_decay_receipt_hash != _recompute_mask_collision_decay_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return _mask_collision_decay_receipt_with_hash_payload(self)

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ShiftDecayReceipt:
    shift_projection_receipt_hash: str
    expected_shift_hash: str
    decay_signature_hash: str
    shift_decay_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_hash_string(self.shift_projection_receipt_hash)
        _validate_hash_string(self.expected_shift_hash)
        _validate_hash_string(self.decay_signature_hash)
        _validate_hash_string(self.shift_decay_receipt_hash)
        if self.shift_decay_receipt_hash != _recompute_shift_decay_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return _shift_decay_receipt_with_hash_payload(self)

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class ReadoutProjectionDecayReceipt:
    readout_projection_receipt_hash: str
    expected_readout_hash: str
    decay_signature_hash: str
    readout_projection_decay_receipt_hash: str

    def __post_init__(self) -> None:
        _validate_hash_string(self.readout_projection_receipt_hash)
        _validate_hash_string(self.expected_readout_hash)
        _validate_hash_string(self.decay_signature_hash)
        _validate_hash_string(self.readout_projection_decay_receipt_hash)
        if self.readout_projection_decay_receipt_hash != _recompute_readout_projection_decay_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        return _readout_projection_decay_receipt_with_hash_payload(self)

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class EntropyDriftReceipt:
    layer_decay_receipts: tuple[LayerDecayReceipt, ...]
    router_decay_receipts: tuple[RouterDecayReceipt, ...]
    mask_collision_decay_receipts: tuple[MaskCollisionDecayReceipt, ...]
    shift_decay_receipts: tuple[ShiftDecayReceipt, ...]
    readout_projection_decay_receipts: tuple[ReadoutProjectionDecayReceipt, ...]
    aggregate_decay_score: int
    entropy_drift_receipt_hash: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, tuple)
            for v in (
                self.layer_decay_receipts,
                self.router_decay_receipts,
                self.mask_collision_decay_receipts,
                self.shift_decay_receipts,
                self.readout_projection_decay_receipts,
            )
        ):
            raise ValueError(_ERR_MISSING_SUBSYSTEM_DECAY_RECEIPTS)

        _validate_subsystem_receipt_sequence(
            self.layer_decay_receipts,
            LayerDecayReceipt,
            "layer_decay_receipt_hash",
            validate_layer_decay_receipt,
            True,
            True,
        )
        _validate_subsystem_receipt_sequence(
            self.router_decay_receipts,
            RouterDecayReceipt,
            "router_decay_receipt_hash",
            validate_router_decay_receipt,
            True,
            True,
        )
        _validate_subsystem_receipt_sequence(
            self.mask_collision_decay_receipts,
            MaskCollisionDecayReceipt,
            "mask_collision_decay_receipt_hash",
            validate_mask_collision_decay_receipt,
            True,
            True,
        )
        _validate_subsystem_receipt_sequence(
            self.shift_decay_receipts,
            ShiftDecayReceipt,
            "shift_decay_receipt_hash",
            validate_shift_decay_receipt,
            True,
            True,
        )
        _validate_subsystem_receipt_sequence(
            self.readout_projection_decay_receipts,
            ReadoutProjectionDecayReceipt,
            "readout_projection_decay_receipt_hash",
            validate_readout_projection_decay_receipt,
            True,
            True,
        )
        _validate_exact_non_negative_int(self.aggregate_decay_score)
        _validate_hash_string(self.entropy_drift_receipt_hash)
        if self.entropy_drift_receipt_hash != _recompute_entropy_drift_receipt_hash(self):
            raise ValueError(_ERR_HASH_MISMATCH)

    def to_dict(self) -> dict[str, Any]:
        payload = _entropy_drift_receipt_payload(self)
        payload["entropy_drift_receipt_hash"] = self.entropy_drift_receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_layer_decay_receipt(
    layered_hash: str,
    expected_layered_hash: str,
    decay_signature: DigitalDecaySignature,
) -> LayerDecayReceipt:
    validate_digital_decay_signature(decay_signature)
    temp = object.__new__(LayerDecayReceipt)
    object.__setattr__(temp, "layered_hash", layered_hash)
    object.__setattr__(temp, "expected_layered_hash", expected_layered_hash)
    object.__setattr__(temp, "decay_signature_hash", decay_signature.digital_decay_signature_hash)
    object.__setattr__(temp, "layer_decay_receipt_hash", "0" * 64)
    return LayerDecayReceipt(
        layered_hash=layered_hash,
        expected_layered_hash=expected_layered_hash,
        decay_signature_hash=decay_signature.digital_decay_signature_hash,
        layer_decay_receipt_hash=_recompute_layer_decay_receipt_hash(temp),
    )


def build_router_decay_receipt(
    router_lattice_path_receipt_hash: str,
    expected_router_hash: str,
    decay_signature: DigitalDecaySignature,
) -> RouterDecayReceipt:
    validate_digital_decay_signature(decay_signature)
    temp = object.__new__(RouterDecayReceipt)
    object.__setattr__(temp, "router_lattice_path_receipt_hash", router_lattice_path_receipt_hash)
    object.__setattr__(temp, "expected_router_hash", expected_router_hash)
    object.__setattr__(temp, "decay_signature_hash", decay_signature.digital_decay_signature_hash)
    object.__setattr__(temp, "router_decay_receipt_hash", "0" * 64)
    return RouterDecayReceipt(
        router_lattice_path_receipt_hash=router_lattice_path_receipt_hash,
        expected_router_hash=expected_router_hash,
        decay_signature_hash=decay_signature.digital_decay_signature_hash,
        router_decay_receipt_hash=_recompute_router_decay_receipt_hash(temp),
    )


def build_mask_collision_decay_receipt(
    mask_reduction_receipt_hash: str,
    expected_mask_hash: str,
    collision_type: str,
    decay_signature: DigitalDecaySignature,
) -> MaskCollisionDecayReceipt:
    validate_digital_decay_signature(decay_signature)
    if collision_type not in _ALLOWED_COLLISION_TYPES:
        raise ValueError(_ERR_INVALID_INPUT)
    temp = object.__new__(MaskCollisionDecayReceipt)
    object.__setattr__(temp, "mask_reduction_receipt_hash", mask_reduction_receipt_hash)
    object.__setattr__(temp, "expected_mask_hash", expected_mask_hash)
    object.__setattr__(temp, "collision_type", collision_type)
    object.__setattr__(temp, "decay_signature_hash", decay_signature.digital_decay_signature_hash)
    object.__setattr__(temp, "mask_collision_decay_receipt_hash", "0" * 64)
    return MaskCollisionDecayReceipt(
        mask_reduction_receipt_hash=mask_reduction_receipt_hash,
        expected_mask_hash=expected_mask_hash,
        collision_type=collision_type,
        decay_signature_hash=decay_signature.digital_decay_signature_hash,
        mask_collision_decay_receipt_hash=_recompute_mask_collision_decay_receipt_hash(temp),
    )


def build_shift_decay_receipt(
    shift_projection_receipt_hash: str,
    expected_shift_hash: str,
    decay_signature: DigitalDecaySignature,
) -> ShiftDecayReceipt:
    validate_digital_decay_signature(decay_signature)
    temp = object.__new__(ShiftDecayReceipt)
    object.__setattr__(temp, "shift_projection_receipt_hash", shift_projection_receipt_hash)
    object.__setattr__(temp, "expected_shift_hash", expected_shift_hash)
    object.__setattr__(temp, "decay_signature_hash", decay_signature.digital_decay_signature_hash)
    object.__setattr__(temp, "shift_decay_receipt_hash", "0" * 64)
    return ShiftDecayReceipt(
        shift_projection_receipt_hash=shift_projection_receipt_hash,
        expected_shift_hash=expected_shift_hash,
        decay_signature_hash=decay_signature.digital_decay_signature_hash,
        shift_decay_receipt_hash=_recompute_shift_decay_receipt_hash(temp),
    )


def build_readout_projection_decay_receipt(
    readout_projection_receipt_hash: str,
    expected_readout_hash: str,
    decay_signature: DigitalDecaySignature,
) -> ReadoutProjectionDecayReceipt:
    validate_digital_decay_signature(decay_signature)
    temp = object.__new__(ReadoutProjectionDecayReceipt)
    object.__setattr__(temp, "readout_projection_receipt_hash", readout_projection_receipt_hash)
    object.__setattr__(temp, "expected_readout_hash", expected_readout_hash)
    object.__setattr__(temp, "decay_signature_hash", decay_signature.digital_decay_signature_hash)
    object.__setattr__(temp, "readout_projection_decay_receipt_hash", "0" * 64)
    return ReadoutProjectionDecayReceipt(
        readout_projection_receipt_hash=readout_projection_receipt_hash,
        expected_readout_hash=expected_readout_hash,
        decay_signature_hash=decay_signature.digital_decay_signature_hash,
        readout_projection_decay_receipt_hash=_recompute_readout_projection_decay_receipt_hash(temp),
    )


def build_entropy_drift_receipt(
    layer_decay_receipts: list[LayerDecayReceipt] | tuple[LayerDecayReceipt, ...],
    router_decay_receipts: list[RouterDecayReceipt] | tuple[RouterDecayReceipt, ...],
    mask_collision_decay_receipts: list[MaskCollisionDecayReceipt]
    | tuple[MaskCollisionDecayReceipt, ...],
    shift_decay_receipts: list[ShiftDecayReceipt] | tuple[ShiftDecayReceipt, ...],
    readout_projection_decay_receipts: list[ReadoutProjectionDecayReceipt]
    | tuple[ReadoutProjectionDecayReceipt, ...],
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...],
) -> EntropyDriftReceipt:
    layers = _validate_subsystem_receipt_sequence(
        layer_decay_receipts,
        LayerDecayReceipt,
        "layer_decay_receipt_hash",
        validate_layer_decay_receipt,
        True,
        False,
    )
    routers = _validate_subsystem_receipt_sequence(
        router_decay_receipts,
        RouterDecayReceipt,
        "router_decay_receipt_hash",
        validate_router_decay_receipt,
        True,
        False,
    )
    masks = _validate_subsystem_receipt_sequence(
        mask_collision_decay_receipts,
        MaskCollisionDecayReceipt,
        "mask_collision_decay_receipt_hash",
        validate_mask_collision_decay_receipt,
        True,
        False,
    )
    shifts = _validate_subsystem_receipt_sequence(
        shift_decay_receipts,
        ShiftDecayReceipt,
        "shift_decay_receipt_hash",
        validate_shift_decay_receipt,
        True,
        False,
    )
    reads = _validate_subsystem_receipt_sequence(
        readout_projection_decay_receipts,
        ReadoutProjectionDecayReceipt,
        "readout_projection_decay_receipt_hash",
        validate_readout_projection_decay_receipt,
        True,
        False,
    )
    registry = _validate_signature_registry(decay_signatures)
    layers = tuple(sorted(layers, key=lambda x: x.layer_decay_receipt_hash))
    routers = tuple(sorted(routers, key=lambda x: x.router_decay_receipt_hash))
    masks = tuple(sorted(masks, key=lambda x: x.mask_collision_decay_receipt_hash))
    shifts = tuple(sorted(shifts, key=lambda x: x.shift_decay_receipt_hash))
    reads = tuple(sorted(reads, key=lambda x: x.readout_projection_decay_receipt_hash))

    aggregate_decay_score = 0
    for receipt in (*layers, *routers, *masks, *shifts, *reads):
        aggregate_decay_score += _signature_score_for_hash(
            receipt.decay_signature_hash,
            registry,
        )

    entropy_hash = sha256_hex(
        {
            "layer_decay_receipts": [
                _layer_decay_receipt_with_hash_payload(r) for r in layers
            ],
            "router_decay_receipts": [
                _router_decay_receipt_with_hash_payload(r) for r in routers
            ],
            "mask_collision_decay_receipts": [
                _mask_collision_decay_receipt_with_hash_payload(r) for r in masks
            ],
            "shift_decay_receipts": [
                _shift_decay_receipt_with_hash_payload(r) for r in shifts
            ],
            "readout_projection_decay_receipts": [
                _readout_projection_decay_receipt_with_hash_payload(r) for r in reads
            ],
            "aggregate_decay_score": aggregate_decay_score,
        }
    )

    return EntropyDriftReceipt(
        layer_decay_receipts=layers,
        router_decay_receipts=routers,
        mask_collision_decay_receipts=masks,
        shift_decay_receipts=shifts,
        readout_projection_decay_receipts=reads,
        aggregate_decay_score=aggregate_decay_score,
        entropy_drift_receipt_hash=entropy_hash,
    )


def validate_layer_decay_receipt(receipt: LayerDecayReceipt) -> bool:
    if not isinstance(receipt, LayerDecayReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    receipt.__post_init__()
    return True


def validate_router_decay_receipt(receipt: RouterDecayReceipt) -> bool:
    if not isinstance(receipt, RouterDecayReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    receipt.__post_init__()
    return True


def validate_mask_collision_decay_receipt(receipt: MaskCollisionDecayReceipt) -> bool:
    if not isinstance(receipt, MaskCollisionDecayReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    receipt.__post_init__()
    return True


def validate_shift_decay_receipt(receipt: ShiftDecayReceipt) -> bool:
    if not isinstance(receipt, ShiftDecayReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    receipt.__post_init__()
    return True


def validate_readout_projection_decay_receipt(
    receipt: ReadoutProjectionDecayReceipt,
) -> bool:
    if not isinstance(receipt, ReadoutProjectionDecayReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    receipt.__post_init__()
    return True


def validate_entropy_drift_receipt(
    receipt: EntropyDriftReceipt,
    decay_signatures: list[DigitalDecaySignature] | tuple[DigitalDecaySignature, ...],
) -> bool:
    if not isinstance(receipt, EntropyDriftReceipt):
        raise ValueError(_ERR_INVALID_INPUT)

    registry = _validate_signature_registry(decay_signatures)
    receipt.__post_init__()

    aggregate_decay_score = 0
    for child in (
        *receipt.layer_decay_receipts,
        *receipt.router_decay_receipts,
        *receipt.mask_collision_decay_receipts,
        *receipt.shift_decay_receipts,
        *receipt.readout_projection_decay_receipts,
    ):
        aggregate_decay_score += _signature_score_for_hash(child.decay_signature_hash, registry)
    if aggregate_decay_score != receipt.aggregate_decay_score:
        raise ValueError(_ERR_AGGREGATE_SCORE_MISMATCH)
    return True
