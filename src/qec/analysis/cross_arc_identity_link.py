from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .reality_loop_composition_spec import (
    CompositionSlot,
    RealityLoopCompositionSpec,
    validate_composition_slot,
    validate_reality_loop_composition_spec,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_LINK_KIND = "INVALID_LINK_KIND"
_ERR_INVALID_LINK_INDEX = "INVALID_LINK_INDEX"
_ERR_INVALID_LINK_LABEL = "INVALID_LINK_LABEL"
_ERR_INVALID_SLOT_INDEX = "INVALID_SLOT_INDEX"
_ERR_LINK_DEFINITION_MISMATCH = "LINK_DEFINITION_MISMATCH"
_ERR_SLOT_LINK_MISMATCH = "SLOT_LINK_MISMATCH"
_ERR_DUPLICATE_CROSS_ARC_LINK = "DUPLICATE_CROSS_ARC_LINK"
_ERR_MISSING_CROSS_ARC_LINK = "MISSING_CROSS_ARC_LINK"
_ERR_CROSS_ARC_LINK_ORDER_MISMATCH = "CROSS_ARC_LINK_ORDER_MISMATCH"
_ERR_CROSS_ARC_LINK_COUNT_MISMATCH = "CROSS_ARC_LINK_COUNT_MISMATCH"
_ERR_CROSS_ARC_IDENTITY_LINK_RECEIPT_MISMATCH = "CROSS_ARC_IDENTITY_LINK_RECEIPT_MISMATCH"

_REQUIRED_LINK_COUNT = 18
_MAX_LINK_INDEX = 17
_MAX_SLOT_INDEX = 18
_MAX_LINK_LABEL_LENGTH = 32
_MAX_ARC_LABEL_LENGTH = 16
_MAX_RECEIPT_FIELD_NAME_LENGTH = 96

_LINK_KIND_SEQUENTIAL_COMPOSITION_LINK = "SEQUENTIAL_COMPOSITION_LINK"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARC_LABEL_RE = re.compile(r"^v[0-9]{3}$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

_REALITY_LOOP_LINK_DEFINITIONS: tuple[tuple[int, int, str], ...] = tuple(
    (i, i + 1, f"LINK_{i:03d}_{i+1:03d}") for i in range(_REQUIRED_LINK_COUNT)
)


def get_reality_loop_link_definitions() -> tuple[tuple[int, int, str], ...]:
    return _REALITY_LOOP_LINK_DEFINITIONS


def get_allowed_cross_arc_link_kinds() -> frozenset[str]:
    return frozenset({_LINK_KIND_SEQUENTIAL_COMPOSITION_LINK})


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_link_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > _MAX_LINK_INDEX:
        raise ValueError(_ERR_INVALID_LINK_INDEX)
    return v


def _validate_slot_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > _MAX_SLOT_INDEX:
        raise ValueError(_ERR_INVALID_SLOT_INDEX)
    return v


def _validate_link_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_LINK_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_LINK_LABEL)
    return v


def _validate_link_kind(v: object) -> str:
    if not isinstance(v, str) or v not in get_allowed_cross_arc_link_kinds():
        raise ValueError(_ERR_INVALID_LINK_KIND)
    return v


def _validate_arc_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_ARC_LABEL_LENGTH or _ARC_LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _validate_receipt_field_name(v: object) -> str:
    if (
        not isinstance(v, str)
        or not v
        or len(v) > _MAX_RECEIPT_FIELD_NAME_LENGTH
        or _FIELD_NAME_RE.fullmatch(v) is None
    ):
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _cross_arc_identity_link_payload(
    composition_spec_hash: str,
    link_index: int,
    link_label: str,
    link_kind: str,
    source_slot_index: int,
    target_slot_index: int,
    source_arc_label: str,
    target_arc_label: str,
    source_receipt_field_name: str,
    target_receipt_field_name: str,
    source_receipt_hash: str,
    target_receipt_hash: str,
    source_composition_slot_hash: str,
    target_composition_slot_hash: str,
) -> dict[str, Any]:
    return {
        "composition_spec_hash": composition_spec_hash,
        "link_index": link_index,
        "link_label": link_label,
        "link_kind": link_kind,
        "source_slot_index": source_slot_index,
        "target_slot_index": target_slot_index,
        "source_arc_label": source_arc_label,
        "target_arc_label": target_arc_label,
        "source_receipt_field_name": source_receipt_field_name,
        "target_receipt_field_name": target_receipt_field_name,
        "source_receipt_hash": source_receipt_hash,
        "target_receipt_hash": target_receipt_hash,
        "source_composition_slot_hash": source_composition_slot_hash,
        "target_composition_slot_hash": target_composition_slot_hash,
    }


@dataclass(frozen=True)
class CrossArcIdentityLink:
    composition_spec_hash: str
    link_index: int
    link_label: str
    link_kind: str
    source_slot_index: int
    target_slot_index: int
    source_arc_label: str
    target_arc_label: str
    source_receipt_field_name: str
    target_receipt_field_name: str
    source_receipt_hash: str
    target_receipt_hash: str
    source_composition_slot_hash: str
    target_composition_slot_hash: str
    cross_arc_identity_link_hash: str

    def __post_init__(self) -> None:
        validate_cross_arc_identity_link(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _cross_arc_identity_link_payload(
            self.composition_spec_hash,
            self.link_index,
            self.link_label,
            self.link_kind,
            self.source_slot_index,
            self.target_slot_index,
            self.source_arc_label,
            self.target_arc_label,
            self.source_receipt_field_name,
            self.target_receipt_field_name,
            self.source_receipt_hash,
            self.target_receipt_hash,
            self.source_composition_slot_hash,
            self.target_composition_slot_hash,
        )
        payload["cross_arc_identity_link_hash"] = self.cross_arc_identity_link_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _cross_arc_identity_link_receipt_payload(
    composition_spec_hash: str,
    composition_mode: str,
    cross_arc_identity_links: tuple[CrossArcIdentityLink, ...],
    link_count: int,
    required_link_count: int,
    first_composition_slot_hash: str,
    final_composition_slot_hash: str,
) -> dict[str, Any]:
    return {
        "composition_spec_hash": composition_spec_hash,
        "composition_mode": composition_mode,
        "cross_arc_identity_links": [link.to_dict() for link in cross_arc_identity_links],
        "link_count": link_count,
        "required_link_count": required_link_count,
        "first_composition_slot_hash": first_composition_slot_hash,
        "final_composition_slot_hash": final_composition_slot_hash,
    }


@dataclass(frozen=True)
class CrossArcIdentityLinkReceipt:
    composition_spec_hash: str
    composition_mode: str
    cross_arc_identity_links: tuple[CrossArcIdentityLink, ...]
    link_count: int
    required_link_count: int
    first_composition_slot_hash: str
    final_composition_slot_hash: str
    cross_arc_identity_link_receipt_hash: str

    def __post_init__(self) -> None:
        validate_cross_arc_identity_link_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _cross_arc_identity_link_receipt_payload(
            self.composition_spec_hash,
            self.composition_mode,
            self.cross_arc_identity_links,
            self.link_count,
            self.required_link_count,
            self.first_composition_slot_hash,
            self.final_composition_slot_hash,
        )
        payload["cross_arc_identity_link_receipt_hash"] = self.cross_arc_identity_link_receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _validate_link_definition(link_index: int, source_slot_index: int, target_slot_index: int, link_label: str) -> None:
    defn = _REALITY_LOOP_LINK_DEFINITIONS[link_index]
    if defn[0] != source_slot_index or defn[1] != target_slot_index or defn[2] != link_label:
        raise ValueError(_ERR_LINK_DEFINITION_MISMATCH)


def build_cross_arc_identity_link(composition_spec: RealityLoopCompositionSpec, link_index: int) -> CrossArcIdentityLink:
    validate_reality_loop_composition_spec(composition_spec)
    idx = _validate_link_index(link_index)
    source_slot_index, target_slot_index, link_label = _REALITY_LOOP_LINK_DEFINITIONS[idx]
    source_slot = composition_spec.composition_slots[source_slot_index]
    target_slot = composition_spec.composition_slots[target_slot_index]
    validate_composition_slot(source_slot)
    validate_composition_slot(target_slot)
    if source_slot.slot_index != source_slot_index or target_slot.slot_index != target_slot_index:
        raise ValueError(_ERR_SLOT_LINK_MISMATCH)
    payload = _cross_arc_identity_link_payload(
        composition_spec_hash=composition_spec.composition_spec_hash,
        link_index=idx,
        link_label=link_label,
        link_kind=_LINK_KIND_SEQUENTIAL_COMPOSITION_LINK,
        source_slot_index=source_slot_index,
        target_slot_index=target_slot_index,
        source_arc_label=source_slot.arc_label,
        target_arc_label=target_slot.arc_label,
        source_receipt_field_name=source_slot.receipt_field_name,
        target_receipt_field_name=target_slot.receipt_field_name,
        source_receipt_hash=source_slot.receipt_hash,
        target_receipt_hash=target_slot.receipt_hash,
        source_composition_slot_hash=source_slot.composition_slot_hash,
        target_composition_slot_hash=target_slot.composition_slot_hash,
    )
    return CrossArcIdentityLink(**payload, cross_arc_identity_link_hash=sha256_hex(payload))


def build_cross_arc_identity_link_receipt(composition_spec: RealityLoopCompositionSpec) -> CrossArcIdentityLinkReceipt:
    validate_reality_loop_composition_spec(composition_spec)
    links = tuple(build_cross_arc_identity_link(composition_spec, i) for i in range(_REQUIRED_LINK_COUNT))
    payload = _cross_arc_identity_link_receipt_payload(
        composition_spec_hash=composition_spec.composition_spec_hash,
        composition_mode=composition_spec.composition_mode,
        cross_arc_identity_links=links,
        link_count=_REQUIRED_LINK_COUNT,
        required_link_count=_REQUIRED_LINK_COUNT,
        first_composition_slot_hash=composition_spec.composition_slots[0].composition_slot_hash,
        final_composition_slot_hash=composition_spec.composition_slots[_MAX_SLOT_INDEX].composition_slot_hash,
    )
    return CrossArcIdentityLinkReceipt(
        composition_spec_hash=composition_spec.composition_spec_hash,
        composition_mode=composition_spec.composition_mode,
        cross_arc_identity_links=links,
        link_count=_REQUIRED_LINK_COUNT,
        required_link_count=_REQUIRED_LINK_COUNT,
        first_composition_slot_hash=composition_spec.composition_slots[0].composition_slot_hash,
        final_composition_slot_hash=composition_spec.composition_slots[_MAX_SLOT_INDEX].composition_slot_hash,
        cross_arc_identity_link_receipt_hash=sha256_hex(payload),
    )


def validate_cross_arc_identity_link(link: CrossArcIdentityLink) -> bool:
    if not isinstance(link, CrossArcIdentityLink):
        raise ValueError(_ERR_INVALID_INPUT)
    composition_spec_hash = _validate_sha(link.composition_spec_hash)
    idx = _validate_link_index(link.link_index)
    link_label = _validate_link_label(link.link_label)
    link_kind = _validate_link_kind(link.link_kind)
    source_slot_index = _validate_slot_index(link.source_slot_index)
    target_slot_index = _validate_slot_index(link.target_slot_index)
    if target_slot_index != source_slot_index + 1:
        raise ValueError(_ERR_SLOT_LINK_MISMATCH)
    _validate_link_definition(idx, source_slot_index, target_slot_index, link_label)
    source_arc_label = _validate_arc_label(link.source_arc_label)
    target_arc_label = _validate_arc_label(link.target_arc_label)
    source_receipt_field_name = _validate_receipt_field_name(link.source_receipt_field_name)
    target_receipt_field_name = _validate_receipt_field_name(link.target_receipt_field_name)
    source_receipt_hash = _validate_sha(link.source_receipt_hash)
    target_receipt_hash = _validate_sha(link.target_receipt_hash)
    source_composition_slot_hash = _validate_sha(link.source_composition_slot_hash)
    target_composition_slot_hash = _validate_sha(link.target_composition_slot_hash)
    stored_hash = _validate_sha(link.cross_arc_identity_link_hash)
    expected_hash = sha256_hex(
        _cross_arc_identity_link_payload(
            composition_spec_hash,
            idx,
            link_label,
            link_kind,
            source_slot_index,
            target_slot_index,
            source_arc_label,
            target_arc_label,
            source_receipt_field_name,
            target_receipt_field_name,
            source_receipt_hash,
            target_receipt_hash,
            source_composition_slot_hash,
            target_composition_slot_hash,
        )
    )
    if stored_hash != expected_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_cross_arc_identity_link_receipt(receipt: CrossArcIdentityLinkReceipt) -> bool:
    if not isinstance(receipt, CrossArcIdentityLinkReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    composition_spec_hash = _validate_sha(receipt.composition_spec_hash)
    if not isinstance(receipt.cross_arc_identity_links, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(receipt.link_count, int) or isinstance(receipt.link_count, bool):
        raise ValueError(_ERR_CROSS_ARC_LINK_COUNT_MISMATCH)
    if not isinstance(receipt.required_link_count, int) or isinstance(receipt.required_link_count, bool):
        raise ValueError(_ERR_CROSS_ARC_LINK_COUNT_MISMATCH)
    if receipt.link_count != _REQUIRED_LINK_COUNT or receipt.required_link_count != _REQUIRED_LINK_COUNT:
        raise ValueError(_ERR_CROSS_ARC_LINK_COUNT_MISMATCH)
    if len(receipt.cross_arc_identity_links) != _REQUIRED_LINK_COUNT:
        raise ValueError(_ERR_CROSS_ARC_LINK_COUNT_MISMATCH)

    seen_indexes: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for pos, link in enumerate(receipt.cross_arc_identity_links):
        validate_cross_arc_identity_link(link)
        if link.link_index in seen_indexes or (link.source_slot_index, link.target_slot_index) in seen_pairs:
            raise ValueError(_ERR_DUPLICATE_CROSS_ARC_LINK)
        seen_indexes.add(link.link_index)
        seen_pairs.add((link.source_slot_index, link.target_slot_index))
        defn = _REALITY_LOOP_LINK_DEFINITIONS[pos]
        if (link.source_slot_index, link.target_slot_index, link.link_label) != defn:
            raise ValueError(_ERR_CROSS_ARC_LINK_ORDER_MISMATCH)
        if link.link_index != pos:
            raise ValueError(_ERR_CROSS_ARC_LINK_ORDER_MISMATCH)

    if seen_indexes != set(range(_REQUIRED_LINK_COUNT)):
        raise ValueError(_ERR_MISSING_CROSS_ARC_LINK)

    first_hash = _validate_sha(receipt.first_composition_slot_hash)
    final_hash = _validate_sha(receipt.final_composition_slot_hash)
    if receipt.cross_arc_identity_links[0].source_composition_slot_hash != first_hash:
        raise ValueError(_ERR_SLOT_LINK_MISMATCH)
    if receipt.cross_arc_identity_links[-1].target_composition_slot_hash != final_hash:
        raise ValueError(_ERR_SLOT_LINK_MISMATCH)

    stored_hash = _validate_sha(receipt.cross_arc_identity_link_receipt_hash)
    expected_hash = sha256_hex(
        _cross_arc_identity_link_receipt_payload(
            composition_spec_hash,
            receipt.composition_mode,
            receipt.cross_arc_identity_links,
            receipt.link_count,
            receipt.required_link_count,
            first_hash,
            final_hash,
        )
    )
    if stored_hash != expected_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_cross_arc_identity_link_with_composition_spec(
    link: CrossArcIdentityLink,
    composition_spec: RealityLoopCompositionSpec,
) -> bool:
    validate_cross_arc_identity_link(link)
    validate_reality_loop_composition_spec(composition_spec)
    expected = build_cross_arc_identity_link(composition_spec, link.link_index)
    if link.to_dict() != expected.to_dict():
        raise ValueError(_ERR_SLOT_LINK_MISMATCH)
    return True


def validate_cross_arc_identity_link_receipt_with_composition_spec(
    receipt: CrossArcIdentityLinkReceipt,
    composition_spec: RealityLoopCompositionSpec,
) -> bool:
    validate_cross_arc_identity_link_receipt(receipt)
    validate_reality_loop_composition_spec(composition_spec)
    expected = build_cross_arc_identity_link_receipt(composition_spec)
    if receipt.to_dict() != expected.to_dict():
        raise ValueError(_ERR_CROSS_ARC_IDENTITY_LINK_RECEIPT_MISMATCH)
    return True
