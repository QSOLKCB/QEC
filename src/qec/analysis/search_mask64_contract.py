from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from qec.analysis.canonical_hashing import canonical_json, sha256_hex
from qec.analysis.layer_spec_contract import _deep_freeze, _ensure_json_safe

SEARCH_MASK64_VERSION = "v153.5"
UINT64_MIN = 0
UINT64_MAX = 18446744073709551615
MAX_MASK_INPUT_FIELDS = 128
MAX_COLLISION_RECORDS = 128
MAX_COMPATIBILITY_RECORDS = 128

_ALLOWED_SOURCE_TYPES = {"ROUTER_PATH", "READOUT_FIELD", "LAYERED_PROJECTION", "QAM_FEATURE", "GENERIC"}
_ALLOWED_REDUCTION_ALGORITHMS = {"SHA256_FIRST_64_BITS"}
_ALLOWED_BYTE_ORDERS = {"BIG_ENDIAN"}
_ALLOWED_COLLISION_STATUS = {"NO_COLLISION", "KNOWN_EQUIVALENT_COLLISION", "INVALID_COLLISION"}
_ALLOWED_COLLISION_REASON = {"SINGLE_PARTICIPANT", "EQUIVALENT_IDENTITY_PROVIDED", "MISSING_EQUIVALENCE_PROOF"}
_COLLISION_STATUS_REASON_MAP = {
    "NO_COLLISION": "SINGLE_PARTICIPANT",
    "KNOWN_EQUIVALENT_COLLISION": "EQUIVALENT_IDENTITY_PROVIDED",
    "INVALID_COLLISION": "MISSING_EQUIVALENCE_PROOF",
}
_ALLOWED_COMPATIBILITY_STATUS = {"MASK_COMPATIBLE", "MASK_COMPATIBLE_WITH_KNOWN_COLLISIONS", "MASK_INCOMPATIBLE"}
_ALLOWED_COMPATIBILITY_REASON = {"NO_COLLISIONS_DETECTED", "KNOWN_EQUIVALENT_COLLISIONS_PRESENT", "INVALID_COLLISIONS_PRESENT"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256_hex(value: str) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))


def _freeze_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({k: _deep_freeze(mapping[k]) for k in sorted(mapping)})


def _validate_canonical_input(canonical_input: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(canonical_input, Mapping):
        raise ValueError("INVALID_INPUT")
    if len(canonical_input) > MAX_MASK_INPUT_FIELDS:
        raise ValueError("INVALID_INPUT")
    if not all(isinstance(k, str) and k for k in canonical_input):
        raise ValueError("INVALID_INPUT")
    frozen = _freeze_mapping(dict(canonical_input))
    _ensure_json_safe(dict(frozen))
    return frozen


def _mask_value_from_payload(payload: Mapping[str, Any]) -> int:
    digest = hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class SearchMask64:
    mask_id: str
    mask_version: str
    source_type: str
    source_id: str
    canonical_input_hash: str
    reduction_algorithm: str
    byte_order: str
    mask_value: int
    mask_hex: str
    mask_hash: str

    def __post_init__(self) -> None:
        if not self.mask_id or not self.mask_version or not self.source_id:
            raise ValueError("INVALID_INPUT")
        if self.source_type not in _ALLOWED_SOURCE_TYPES or self.reduction_algorithm not in _ALLOWED_REDUCTION_ALGORITHMS or self.byte_order not in _ALLOWED_BYTE_ORDERS:
            raise ValueError("INVALID_INPUT")
        if not _is_sha256_hex(self.canonical_input_hash):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.mask_value, int) or isinstance(self.mask_value, bool) or not (UINT64_MIN <= self.mask_value <= UINT64_MAX):
            raise ValueError("INVALID_INPUT")
        expected_hex = format(self.mask_value, "016x")
        if self.mask_hex != expected_hex:
            raise ValueError("INVALID_INPUT")
        if self.mask_hash and self.mask_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _canonical_payload(self) -> dict[str, Any]:
        p = {
            "mask_id": self.mask_id, "mask_version": self.mask_version, "source_type": self.source_type,
            "source_id": self.source_id, "canonical_input_hash": self.canonical_input_hash,
            "reduction_algorithm": self.reduction_algorithm, "byte_order": self.byte_order,
            "mask_value": self.mask_value, "mask_hex": self.mask_hex,
        }
        _ensure_json_safe(p)
        return p

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), mask_hash=self.mask_hash)


@dataclass(frozen=True)
class MaskReductionReceipt:
    reduction_id: str
    mask_id: str
    mask_version: str
    source_type: str
    source_id: str
    canonical_input_hash: str
    canonical_input_payload_hash: str
    reduction_algorithm: str
    byte_order: str
    mask_value: int
    mask_hex: str
    search_mask_hash: str
    reduction_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        if not self.reduction_id or not self.mask_id or not self.mask_version or not self.source_id:
            raise ValueError("INVALID_INPUT")
        if self.source_type not in _ALLOWED_SOURCE_TYPES or self.reduction_algorithm not in _ALLOWED_REDUCTION_ALGORITHMS or self.byte_order not in _ALLOWED_BYTE_ORDERS:
            raise ValueError("INVALID_INPUT")
        for hv in (self.canonical_input_hash, self.canonical_input_payload_hash, self.search_mask_hash):
            if not _is_sha256_hex(hv):
                raise ValueError("INVALID_INPUT")
        if not isinstance(self.mask_value, int) or isinstance(self.mask_value, bool) or not (UINT64_MIN <= self.mask_value <= UINT64_MAX):
            raise ValueError("INVALID_INPUT")
        if self.mask_hex != format(self.mask_value, "016x"):
            raise ValueError("INVALID_INPUT")
        if self.reduction_hash and self.reduction_hash != self._reduction_stable_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _reduction_payload(self) -> dict[str, Any]:
        p = {k: v for k, v in self._canonical_payload().items() if k != "reduction_hash"}
        p.pop("receipt_hash", None)
        return p

    def _canonical_payload(self) -> dict[str, Any]:
        p = {
            "reduction_id": self.reduction_id, "mask_id": self.mask_id, "mask_version": self.mask_version,
            "source_type": self.source_type, "source_id": self.source_id,
            "canonical_input_hash": self.canonical_input_hash,
            "canonical_input_payload_hash": self.canonical_input_payload_hash,
            "reduction_algorithm": self.reduction_algorithm, "byte_order": self.byte_order,
            "mask_value": self.mask_value, "mask_hex": self.mask_hex, "search_mask_hash": self.search_mask_hash,
            "reduction_hash": self.reduction_hash,
        }
        _ensure_json_safe(p)
        return p

    def _reduction_stable_hash(self) -> str:
        return sha256_hex(self._reduction_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class MaskCollisionReceipt:
    collision_id: str
    mask_version: str
    mask_value: int
    mask_hex: str
    participant_mask_hashes: tuple[str, ...]
    participant_source_ids: tuple[str, ...]
    collision_status: str
    collision_reason: str
    equivalent_identity_hash: str | None
    collision_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "participant_mask_hashes", tuple(self.participant_mask_hashes))
        object.__setattr__(self, "participant_source_ids", tuple(self.participant_source_ids))
        if not self.collision_id or not self.mask_version:
            raise ValueError("INVALID_INPUT")
        if self.collision_status not in _ALLOWED_COLLISION_STATUS:
            raise ValueError("INVALID_INPUT")
        if self.collision_reason not in _ALLOWED_COLLISION_REASON:
            raise ValueError("INVALID_INPUT")
        if self.collision_reason != _COLLISION_STATUS_REASON_MAP[self.collision_status]:
            raise ValueError("INVALID_INPUT")
        if len(self.participant_mask_hashes) != len(self.participant_source_ids):
            raise ValueError("INVALID_INPUT")
        if len(self.participant_mask_hashes) == 0 or len(self.participant_mask_hashes) > MAX_COLLISION_RECORDS:
            raise ValueError("INVALID_INPUT")
        expected_pairs = tuple(sorted(zip(self.participant_mask_hashes, self.participant_source_ids)))
        if tuple(p[0] for p in expected_pairs) != self.participant_mask_hashes or tuple(p[1] for p in expected_pairs) != self.participant_source_ids:
            raise ValueError("INVALID_INPUT")
        if len(set(self.participant_mask_hashes)) != len(self.participant_mask_hashes) or len(set(self.participant_source_ids)) != len(self.participant_source_ids):
            raise ValueError("INVALID_INPUT")
        if not all(_is_sha256_hex(h) for h in self.participant_mask_hashes):
            raise ValueError("INVALID_INPUT")
        if not isinstance(self.mask_value, int) or isinstance(self.mask_value, bool) or not (UINT64_MIN <= self.mask_value <= UINT64_MAX):
            raise ValueError("INVALID_INPUT")
        if self.mask_hex != format(self.mask_value, "016x"):
            raise ValueError("INVALID_INPUT")
        if self.equivalent_identity_hash is not None and not _is_sha256_hex(self.equivalent_identity_hash):
            raise ValueError("INVALID_INPUT")
        if self.collision_status == "NO_COLLISION" and (len(self.participant_mask_hashes) != 1 or self.equivalent_identity_hash is not None):
            raise ValueError("INVALID_INPUT")
        if self.collision_status == "KNOWN_EQUIVALENT_COLLISION" and self.equivalent_identity_hash is None:
            raise ValueError("INVALID_INPUT")
        if self.collision_status == "INVALID_COLLISION" and self.equivalent_identity_hash is not None:
            raise ValueError("INVALID_INPUT")
        if self.collision_hash and self.collision_hash != self._collision_stable_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _collision_payload(self) -> dict[str, Any]:
        return {
            "collision_id": self.collision_id, "mask_version": self.mask_version, "mask_value": self.mask_value,
            "mask_hex": self.mask_hex, "participant_mask_hashes": list(self.participant_mask_hashes),
            "participant_source_ids": list(self.participant_source_ids), "collision_status": self.collision_status,
            "collision_reason": self.collision_reason, "equivalent_identity_hash": self.equivalent_identity_hash,
        }

    def _canonical_payload(self) -> dict[str, Any]:
        p = dict(self._collision_payload(), collision_hash=self.collision_hash)
        _ensure_json_safe(p)
        return p

    def _collision_stable_hash(self) -> str:
        return sha256_hex(self._collision_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


@dataclass(frozen=True)
class MaskCompatibilityReceipt:
    compatibility_id: str
    mask_version: str
    mask_reduction_receipt_hashes: tuple[str, ...]
    collision_receipt_hashes: tuple[str, ...]
    mask_count: int
    collision_count: int
    compatibility_status: str
    compatibility_reason: str
    compatibility_hash: str
    receipt_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "mask_reduction_receipt_hashes", tuple(self.mask_reduction_receipt_hashes))
        object.__setattr__(self, "collision_receipt_hashes", tuple(self.collision_receipt_hashes))
        if not self.compatibility_id or not self.mask_version:
            raise ValueError("INVALID_INPUT")
        if self.compatibility_status not in _ALLOWED_COMPATIBILITY_STATUS or self.compatibility_reason not in _ALLOWED_COMPATIBILITY_REASON:
            raise ValueError("INVALID_INPUT")
        if len(self.mask_reduction_receipt_hashes) > MAX_COMPATIBILITY_RECORDS or len(self.collision_receipt_hashes) > MAX_COMPATIBILITY_RECORDS:
            raise ValueError("INVALID_INPUT")
        if tuple(sorted(self.mask_reduction_receipt_hashes)) != self.mask_reduction_receipt_hashes or tuple(sorted(self.collision_receipt_hashes)) != self.collision_receipt_hashes:
            raise ValueError("INVALID_INPUT")
        if len(set(self.mask_reduction_receipt_hashes)) != len(self.mask_reduction_receipt_hashes) or len(set(self.collision_receipt_hashes)) != len(self.collision_receipt_hashes):
            raise ValueError("INVALID_INPUT")
        if self.mask_count != len(self.mask_reduction_receipt_hashes) or self.collision_count != len(self.collision_receipt_hashes):
            raise ValueError("INVALID_INPUT")
        if not all(_is_sha256_hex(x) for x in self.mask_reduction_receipt_hashes + self.collision_receipt_hashes):
            raise ValueError("INVALID_INPUT")
        if self.compatibility_hash and self.compatibility_hash != self._compatibility_stable_hash():
            raise ValueError("INVALID_INPUT")
        if self.receipt_hash and self.receipt_hash != self.stable_hash():
            raise ValueError("INVALID_INPUT")

    def _compatibility_payload(self) -> dict[str, Any]:
        return {
            "compatibility_id": self.compatibility_id, "mask_version": self.mask_version,
            "mask_reduction_receipt_hashes": list(self.mask_reduction_receipt_hashes),
            "collision_receipt_hashes": list(self.collision_receipt_hashes), "mask_count": self.mask_count,
            "collision_count": self.collision_count, "compatibility_status": self.compatibility_status,
            "compatibility_reason": self.compatibility_reason,
        }

    def _canonical_payload(self) -> dict[str, Any]:
        p = dict(self._compatibility_payload(), compatibility_hash=self.compatibility_hash)
        _ensure_json_safe(p)
        return p

    def _compatibility_stable_hash(self) -> str:
        return sha256_hex(self._compatibility_payload())

    def stable_hash(self) -> str:
        return sha256_hex(self._canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return dict(self._canonical_payload(), receipt_hash=self.receipt_hash)


def build_search_mask64(mask_id: str, source_type: str, source_id: str, canonical_input: Mapping[str, Any]) -> SearchMask64:
    payload = _validate_canonical_input(canonical_input)
    canonical_input_hash = sha256_hex(dict(payload))
    mask_value = _mask_value_from_payload(payload)
    mask_hex = format(mask_value, "016x")
    mask_payload = {
        "mask_id": mask_id,
        "mask_version": SEARCH_MASK64_VERSION,
        "source_type": source_type,
        "source_id": source_id,
        "canonical_input_hash": canonical_input_hash,
        "reduction_algorithm": "SHA256_FIRST_64_BITS",
        "byte_order": "BIG_ENDIAN",
        "mask_value": mask_value,
        "mask_hex": mask_hex,
    }
    _ensure_json_safe(mask_payload)
    mask_hash = sha256_hex(mask_payload)
    return SearchMask64(**dict(mask_payload, mask_hash=mask_hash))


def build_mask_reduction_receipt(reduction_id: str, search_mask: SearchMask64, canonical_input: Mapping[str, Any]) -> MaskReductionReceipt:
    _ = SearchMask64(**search_mask.__dict__)
    payload = _validate_canonical_input(canonical_input)
    payload_hash = sha256_hex(dict(payload))
    if payload_hash != search_mask.canonical_input_hash:
        raise ValueError("INVALID_INPUT")
    if search_mask.mask_hash != search_mask.stable_hash():
        raise ValueError("INVALID_INPUT")
    reduction_payload = {
        "reduction_id": reduction_id,
        "mask_id": search_mask.mask_id,
        "mask_version": search_mask.mask_version,
        "source_type": search_mask.source_type,
        "source_id": search_mask.source_id,
        "canonical_input_hash": search_mask.canonical_input_hash,
        "canonical_input_payload_hash": payload_hash,
        "reduction_algorithm": search_mask.reduction_algorithm,
        "byte_order": search_mask.byte_order,
        "mask_value": search_mask.mask_value,
        "mask_hex": search_mask.mask_hex,
        "search_mask_hash": search_mask.mask_hash,
    }
    _ensure_json_safe(reduction_payload)
    reduction_hash = sha256_hex(reduction_payload)
    receipt_payload = dict(reduction_payload, reduction_hash=reduction_hash)
    receipt_hash = sha256_hex(receipt_payload)
    return MaskReductionReceipt(**dict(receipt_payload, receipt_hash=receipt_hash))


def build_mask_collision_receipt(collision_id: str, search_masks: Sequence[SearchMask64], equivalent_identity_hash: str | None = None) -> MaskCollisionReceipt:
    masks = tuple(search_masks)
    if len(masks) == 0:
        raise ValueError("INVALID_INPUT")
    for m in masks:
        SearchMask64(**m.__dict__)
        if m.mask_hash != m.stable_hash():
            raise ValueError("INVALID_INPUT")
    mask_values = {m.mask_value for m in masks}
    mask_hexes = {m.mask_hex for m in masks}
    if len(mask_values) != 1 or len(mask_hexes) != 1:
        raise ValueError("INVALID_INPUT")
    if len({m.mask_version for m in masks}) != 1:
        raise ValueError("INVALID_INPUT")
    if len(masks) == 1:
        status, reason, eq = "NO_COLLISION", "SINGLE_PARTICIPANT", None
    elif equivalent_identity_hash is not None:
        if not _is_sha256_hex(equivalent_identity_hash):
            raise ValueError("INVALID_INPUT")
        status, reason, eq = "KNOWN_EQUIVALENT_COLLISION", "EQUIVALENT_IDENTITY_PROVIDED", equivalent_identity_hash
    else:
        status, reason, eq = "INVALID_COLLISION", "MISSING_EQUIVALENCE_PROOF", None
    pairs = tuple(sorted((m.mask_hash, m.source_id) for m in masks))
    pmh = tuple(p[0] for p in pairs)
    psi = tuple(p[1] for p in pairs)
    collision_payload = {
        "collision_id": collision_id,
        "mask_version": masks[0].mask_version,
        "mask_value": masks[0].mask_value,
        "mask_hex": masks[0].mask_hex,
        "participant_mask_hashes": list(pmh),
        "participant_source_ids": list(psi),
        "collision_status": status,
        "collision_reason": reason,
        "equivalent_identity_hash": eq,
    }
    _ensure_json_safe(collision_payload)
    collision_hash = sha256_hex(collision_payload)
    receipt_payload = dict(collision_payload, collision_hash=collision_hash)
    receipt_hash = sha256_hex(receipt_payload)
    return MaskCollisionReceipt(**dict(receipt_payload, receipt_hash=receipt_hash))


def build_mask_compatibility_receipt(compatibility_id: str, reduction_receipts: Sequence[MaskReductionReceipt], collision_receipts: Sequence[MaskCollisionReceipt]) -> MaskCompatibilityReceipt:
    reductions = tuple(reduction_receipts)
    collisions = tuple(collision_receipts)
    for r in reductions:
        MaskReductionReceipt(**r.__dict__)
        if r.reduction_hash != r._reduction_stable_hash() or r.receipt_hash != r.stable_hash():
            raise ValueError("INVALID_INPUT")
    for c in collisions:
        MaskCollisionReceipt(**c.__dict__)
        if c.collision_hash != c._collision_stable_hash() or c.receipt_hash != c.stable_hash():
            raise ValueError("INVALID_INPUT")
    if reductions and len({r.mask_version for r in reductions}) != 1:
        raise ValueError("INVALID_INPUT")
    if collisions and len({c.mask_version for c in collisions}) != 1:
        raise ValueError("INVALID_INPUT")
    if reductions and collisions and reductions[0].mask_version != collisions[0].mask_version:
        raise ValueError("INVALID_INPUT")
    mv = reductions[0].mask_version if reductions else (collisions[0].mask_version if collisions else SEARCH_MASK64_VERSION)
    statuses = {c.collision_status for c in collisions}
    if "INVALID_COLLISION" in statuses:
        st, rs = "MASK_INCOMPATIBLE", "INVALID_COLLISIONS_PRESENT"
    elif "KNOWN_EQUIVALENT_COLLISION" in statuses:
        st, rs = "MASK_COMPATIBLE_WITH_KNOWN_COLLISIONS", "KNOWN_EQUIVALENT_COLLISIONS_PRESENT"
    else:
        st, rs = "MASK_COMPATIBLE", "NO_COLLISIONS_DETECTED"
    compatibility_payload = {
        "compatibility_id": compatibility_id,
        "mask_version": mv,
        "mask_reduction_receipt_hashes": list(sorted(r.receipt_hash for r in reductions)),
        "collision_receipt_hashes": list(sorted(c.receipt_hash for c in collisions)),
        "mask_count": len(reductions),
        "collision_count": len(collisions),
        "compatibility_status": st,
        "compatibility_reason": rs,
    }
    _ensure_json_safe(compatibility_payload)
    compatibility_hash = sha256_hex(compatibility_payload)
    receipt_payload = dict(compatibility_payload, compatibility_hash=compatibility_hash)
    receipt_hash = sha256_hex(receipt_payload)
    receipt = MaskCompatibilityReceipt(**dict(receipt_payload, receipt_hash=receipt_hash))
    validate_mask_compatibility_receipt(receipt, reductions, collisions)
    return receipt


def validate_mask_compatibility_receipt(receipt: MaskCompatibilityReceipt, reduction_receipts: Sequence[MaskReductionReceipt], collision_receipts: Sequence[MaskCollisionReceipt]) -> None:
    MaskCompatibilityReceipt(**receipt.__dict__)
    if receipt.compatibility_hash != receipt._compatibility_stable_hash() or receipt.receipt_hash != receipt.stable_hash():
        raise ValueError("INVALID_INPUT")
    reductions = tuple(reduction_receipts)
    collisions = tuple(collision_receipts)
    reduction_hashes = tuple(sorted(r.receipt_hash for r in reductions))
    collision_hashes = tuple(sorted(c.receipt_hash for c in collisions))
    if receipt.mask_reduction_receipt_hashes != reduction_hashes or receipt.collision_receipt_hashes != collision_hashes:
        raise ValueError("INVALID_INPUT")
    if reductions and len({r.mask_version for r in reductions}) != 1:
        raise ValueError("INVALID_INPUT")
    if collisions and len({c.mask_version for c in collisions}) != 1:
        raise ValueError("INVALID_INPUT")
    if reductions and collisions and reductions[0].mask_version != collisions[0].mask_version:
        raise ValueError("INVALID_INPUT")
    mv = reductions[0].mask_version if reductions else (collisions[0].mask_version if collisions else SEARCH_MASK64_VERSION)
    statuses = {c.collision_status for c in collisions}
    if "INVALID_COLLISION" in statuses:
        expected_st, expected_rs = "MASK_INCOMPATIBLE", "INVALID_COLLISIONS_PRESENT"
    elif "KNOWN_EQUIVALENT_COLLISION" in statuses:
        expected_st, expected_rs = "MASK_COMPATIBLE_WITH_KNOWN_COLLISIONS", "KNOWN_EQUIVALENT_COLLISIONS_PRESENT"
    else:
        expected_st, expected_rs = "MASK_COMPATIBLE", "NO_COLLISIONS_DETECTED"
    if receipt.mask_version != mv or receipt.compatibility_status != expected_st or receipt.compatibility_reason != expected_rs:
        raise ValueError("INVALID_INPUT")


for _n in ("apply", "execute", "run", "traverse", "pathfind", "resolve", "project", "readout", "shift", "hilber", "hilbert", "shell", "matrix", "markov"):
    for _cls in (SearchMask64, MaskReductionReceipt, MaskCollisionReceipt, MaskCompatibilityReceipt):
        if hasattr(_cls, _n):
            raise RuntimeError("INVALID_STATE")


__all__ = [
    "SEARCH_MASK64_VERSION", "UINT64_MIN", "UINT64_MAX", "MAX_MASK_INPUT_FIELDS", "MAX_COLLISION_RECORDS", "MAX_COMPATIBILITY_RECORDS",
    "SearchMask64", "MaskReductionReceipt", "MaskCollisionReceipt", "MaskCompatibilityReceipt",
    "build_search_mask64", "build_mask_reduction_receipt", "build_mask_collision_receipt", "build_mask_compatibility_receipt", "validate_mask_compatibility_receipt",
]
