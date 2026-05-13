from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .cross_arc_identity_link import (
    CrossArcIdentityLink,
    CrossArcIdentityLinkReceipt,
    get_reality_loop_link_definitions,
    validate_cross_arc_identity_link,
    validate_cross_arc_identity_link_receipt,
    validate_cross_arc_identity_link_receipt_with_composition_spec,
)
from .reality_loop_composition_spec import (
    COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION,
    CompositionSlot,
    RealityLoopCompositionSpec,
    get_reality_loop_slot_definitions,
    validate_composition_slot,
    validate_reality_loop_composition_spec,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_PROOF_MODE = "INVALID_PROOF_MODE"
_ERR_INVALID_REALITY_LOOP_PROOF_CLASS = "INVALID_REALITY_LOOP_PROOF_CLASS"
_ERR_SLOT_COUNT_MISMATCH = "SLOT_COUNT_MISMATCH"
_ERR_LINK_COUNT_MISMATCH = "LINK_COUNT_MISMATCH"
_ERR_DUPLICATE_SLOT_HASH = "DUPLICATE_SLOT_HASH"
_ERR_DUPLICATE_LINK_HASH = "DUPLICATE_LINK_HASH"
_ERR_SLOT_LINK_TOPOLOGY_MISMATCH = "SLOT_LINK_TOPOLOGY_MISMATCH"
_ERR_REALITY_LOOP_COMPLETION_MISMATCH = "REALITY_LOOP_COMPLETION_MISMATCH"
_ERR_REALITY_LOOP_PROOF_CLASS_MISMATCH = "REALITY_LOOP_PROOF_CLASS_MISMATCH"
_ERR_REALITY_LOOP_PROOF_RECEIPT_MISMATCH = "REALITY_LOOP_PROOF_RECEIPT_MISMATCH"

_REQUIRED_SLOT_COUNT = 19
_REQUIRED_LINK_COUNT = 18
_MAX_SLOT_HASHES = 19
_MAX_LINK_HASHES = 18

_PROOF_MODE_FIXED_19_SLOT_18_LINK_REALITY_LOOP_PROOF = "FIXED_19_SLOT_18_LINK_REALITY_LOOP_PROOF"

_REALITY_LOOP_PROOF_CLASS_COMPLETE = "REALITY_LOOP_PROOF_COMPLETE"
_REALITY_LOOP_PROOF_CLASS_INCOMPLETE = "REALITY_LOOP_PROOF_INCOMPLETE"
_REALITY_LOOP_PROOF_CLASS_INVALID = "REALITY_LOOP_PROOF_INVALID"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def get_allowed_reality_loop_proof_modes() -> frozenset[str]:
    return frozenset({_PROOF_MODE_FIXED_19_SLOT_18_LINK_REALITY_LOOP_PROOF})


def get_allowed_reality_loop_proof_classes() -> frozenset[str]:
    return frozenset(
        {
            _REALITY_LOOP_PROOF_CLASS_COMPLETE,
            _REALITY_LOOP_PROOF_CLASS_INCOMPLETE,
            _REALITY_LOOP_PROOF_CLASS_INVALID,
        }
    )


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_count(v: object, err: str) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(err)
    return v


def _derive_completion_and_class(
    slot_count: int,
    required_slot_count: int,
    link_count: int,
    required_link_count: int,
    slots_complete: bool,
    links_complete: bool,
    slot_link_topology_aligned: bool,
) -> tuple[bool, str]:
    if (
        slots_complete
        and links_complete
        and slot_link_topology_aligned
        and slot_count == required_slot_count == _REQUIRED_SLOT_COUNT
        and link_count == required_link_count == _REQUIRED_LINK_COUNT
    ):
        return True, _REALITY_LOOP_PROOF_CLASS_COMPLETE
    if not slots_complete or not links_complete:
        return False, _REALITY_LOOP_PROOF_CLASS_INCOMPLETE
    return False, _REALITY_LOOP_PROOF_CLASS_INVALID


def _reality_loop_proof_receipt_payload(
    composition_spec_hash: str,
    cross_arc_identity_link_receipt_hash: str,
    composition_mode: str,
    proof_mode: str,
    slot_count: int,
    required_slot_count: int,
    link_count: int,
    required_link_count: int,
    first_composition_slot_hash: str,
    final_composition_slot_hash: str,
    composition_slot_hashes: tuple[str, ...],
    cross_arc_identity_link_hashes: tuple[str, ...],
    slots_complete: bool,
    links_complete: bool,
    slot_link_topology_aligned: bool,
    reality_loop_complete: bool,
    reality_loop_proof_class: str,
) -> dict[str, Any]:
    return {
        "composition_spec_hash": composition_spec_hash,
        "cross_arc_identity_link_receipt_hash": cross_arc_identity_link_receipt_hash,
        "composition_mode": composition_mode,
        "proof_mode": proof_mode,
        "slot_count": slot_count,
        "required_slot_count": required_slot_count,
        "link_count": link_count,
        "required_link_count": required_link_count,
        "first_composition_slot_hash": first_composition_slot_hash,
        "final_composition_slot_hash": final_composition_slot_hash,
        "composition_slot_hashes": list(composition_slot_hashes),
        "cross_arc_identity_link_hashes": list(cross_arc_identity_link_hashes),
        "slots_complete": slots_complete,
        "links_complete": links_complete,
        "slot_link_topology_aligned": slot_link_topology_aligned,
        "reality_loop_complete": reality_loop_complete,
        "reality_loop_proof_class": reality_loop_proof_class,
    }


@dataclass(frozen=True)
class RealityLoopProofReceipt:
    composition_spec_hash: str
    cross_arc_identity_link_receipt_hash: str
    composition_mode: str
    proof_mode: str
    slot_count: int
    required_slot_count: int
    link_count: int
    required_link_count: int
    first_composition_slot_hash: str
    final_composition_slot_hash: str
    composition_slot_hashes: tuple[str, ...]
    cross_arc_identity_link_hashes: tuple[str, ...]
    slots_complete: bool
    links_complete: bool
    slot_link_topology_aligned: bool
    reality_loop_complete: bool
    reality_loop_proof_class: str
    reality_loop_proof_receipt_hash: str

    def __post_init__(self) -> None:
        validate_reality_loop_proof_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _reality_loop_proof_receipt_payload(
            self.composition_spec_hash,
            self.cross_arc_identity_link_receipt_hash,
            self.composition_mode,
            self.proof_mode,
            self.slot_count,
            self.required_slot_count,
            self.link_count,
            self.required_link_count,
            self.first_composition_slot_hash,
            self.final_composition_slot_hash,
            self.composition_slot_hashes,
            self.cross_arc_identity_link_hashes,
            self.slots_complete,
            self.links_complete,
            self.slot_link_topology_aligned,
            self.reality_loop_complete,
            self.reality_loop_proof_class,
        )
        payload["reality_loop_proof_receipt_hash"] = self.reality_loop_proof_receipt_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_reality_loop_proof_receipt(
    composition_spec: RealityLoopCompositionSpec,
    cross_arc_identity_link_receipt: CrossArcIdentityLinkReceipt,
) -> RealityLoopProofReceipt:
    """Build a deterministic reality loop proof receipt from validated upstream artifacts.

    The slot_link_topology_aligned flag is set to True when upstream validators pass.
    Topology validation is delegated entirely to upstream validators:
    - validate_reality_loop_composition_spec validates slot structure
    - validate_cross_arc_identity_link_receipt validates link structure
    - validate_cross_arc_identity_link_receipt_with_composition_spec validates
      that links correctly reference composition slots
    - composition_spec_hash equality check ensures artifacts are from the same spec

    This builder does not perform independent topology validation beyond what
    upstream validators provide.
    """
    validate_reality_loop_composition_spec(composition_spec)
    validate_cross_arc_identity_link_receipt(cross_arc_identity_link_receipt)
    validate_cross_arc_identity_link_receipt_with_composition_spec(cross_arc_identity_link_receipt, composition_spec)
    if composition_spec.composition_spec_hash != cross_arc_identity_link_receipt.composition_spec_hash:
        raise ValueError(_ERR_SLOT_LINK_TOPOLOGY_MISMATCH)

    composition_slots = composition_spec.composition_slots
    cross_arc_links = cross_arc_identity_link_receipt.cross_arc_identity_links
    if len(composition_slots) != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_SLOT_COUNT_MISMATCH)
    if len(cross_arc_links) != _REQUIRED_LINK_COUNT:
        raise ValueError(_ERR_LINK_COUNT_MISMATCH)

    composition_slot_hashes = tuple(slot.composition_slot_hash for slot in composition_slots)
    cross_arc_identity_link_hashes = tuple(link.cross_arc_identity_link_hash for link in cross_arc_links)
    first_hash = composition_slot_hashes[0]
    final_hash = composition_slot_hashes[-1]

    slots_complete = len(composition_slot_hashes) == _REQUIRED_SLOT_COUNT
    links_complete = len(cross_arc_identity_link_hashes) == _REQUIRED_LINK_COUNT
    # Topology alignment is True when upstream validators pass (see docstring)
    slot_link_topology_aligned = True
    reality_loop_complete, proof_class = _derive_completion_and_class(
        len(composition_slot_hashes), _REQUIRED_SLOT_COUNT, len(cross_arc_identity_link_hashes), _REQUIRED_LINK_COUNT,
        slots_complete, links_complete, slot_link_topology_aligned
    )

    payload = _reality_loop_proof_receipt_payload(
        composition_spec_hash=composition_spec.composition_spec_hash,
        cross_arc_identity_link_receipt_hash=cross_arc_identity_link_receipt.cross_arc_identity_link_receipt_hash,
        composition_mode=composition_spec.composition_mode,
        proof_mode=_PROOF_MODE_FIXED_19_SLOT_18_LINK_REALITY_LOOP_PROOF,
        slot_count=len(composition_slot_hashes),
        required_slot_count=_REQUIRED_SLOT_COUNT,
        link_count=len(cross_arc_identity_link_hashes),
        required_link_count=_REQUIRED_LINK_COUNT,
        first_composition_slot_hash=first_hash,
        final_composition_slot_hash=final_hash,
        composition_slot_hashes=composition_slot_hashes,
        cross_arc_identity_link_hashes=cross_arc_identity_link_hashes,
        slots_complete=slots_complete,
        links_complete=links_complete,
        slot_link_topology_aligned=slot_link_topology_aligned,
        reality_loop_complete=reality_loop_complete,
        reality_loop_proof_class=proof_class,
    )
    return RealityLoopProofReceipt(
        composition_spec_hash=composition_spec.composition_spec_hash,
        cross_arc_identity_link_receipt_hash=cross_arc_identity_link_receipt.cross_arc_identity_link_receipt_hash,
        composition_mode=composition_spec.composition_mode,
        proof_mode=_PROOF_MODE_FIXED_19_SLOT_18_LINK_REALITY_LOOP_PROOF,
        slot_count=len(composition_slot_hashes),
        required_slot_count=_REQUIRED_SLOT_COUNT,
        link_count=len(cross_arc_identity_link_hashes),
        required_link_count=_REQUIRED_LINK_COUNT,
        first_composition_slot_hash=first_hash,
        final_composition_slot_hash=final_hash,
        composition_slot_hashes=composition_slot_hashes,
        cross_arc_identity_link_hashes=cross_arc_identity_link_hashes,
        slots_complete=slots_complete,
        links_complete=links_complete,
        slot_link_topology_aligned=slot_link_topology_aligned,
        reality_loop_complete=reality_loop_complete,
        reality_loop_proof_class=proof_class,
        reality_loop_proof_receipt_hash=sha256_hex(payload),
    )


def validate_reality_loop_proof_receipt(receipt: RealityLoopProofReceipt) -> bool:
    if not isinstance(receipt, RealityLoopProofReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    composition_spec_hash = _validate_sha(receipt.composition_spec_hash)
    cross_hash = _validate_sha(receipt.cross_arc_identity_link_receipt_hash)
    if receipt.composition_mode != COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION:
        raise ValueError("INVALID_COMPOSITION_MODE")
    if receipt.proof_mode not in get_allowed_reality_loop_proof_modes():
        raise ValueError(_ERR_INVALID_PROOF_MODE)

    slot_count = _validate_count(receipt.slot_count, _ERR_SLOT_COUNT_MISMATCH)
    required_slot_count = _validate_count(receipt.required_slot_count, _ERR_SLOT_COUNT_MISMATCH)
    link_count = _validate_count(receipt.link_count, _ERR_LINK_COUNT_MISMATCH)
    required_link_count = _validate_count(receipt.required_link_count, _ERR_LINK_COUNT_MISMATCH)

    if not isinstance(receipt.composition_slot_hashes, tuple):
        raise ValueError(_ERR_SLOT_COUNT_MISMATCH)
    if not isinstance(receipt.cross_arc_identity_link_hashes, tuple):
        raise ValueError(_ERR_LINK_COUNT_MISMATCH)
    if len(receipt.composition_slot_hashes) > _MAX_SLOT_HASHES:
        raise ValueError(_ERR_SLOT_COUNT_MISMATCH)
    if len(receipt.cross_arc_identity_link_hashes) > _MAX_LINK_HASHES:
        raise ValueError(_ERR_LINK_COUNT_MISMATCH)

    for v in receipt.composition_slot_hashes:
        _validate_sha(v)
    for v in receipt.cross_arc_identity_link_hashes:
        _validate_sha(v)

    if len(set(receipt.composition_slot_hashes)) != len(receipt.composition_slot_hashes):
        raise ValueError(_ERR_DUPLICATE_SLOT_HASH)
    if len(set(receipt.cross_arc_identity_link_hashes)) != len(receipt.cross_arc_identity_link_hashes):
        raise ValueError(_ERR_DUPLICATE_LINK_HASH)

    if slot_count != len(receipt.composition_slot_hashes) or required_slot_count != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_SLOT_COUNT_MISMATCH)
    if link_count != len(receipt.cross_arc_identity_link_hashes) or required_link_count != _REQUIRED_LINK_COUNT:
        raise ValueError(_ERR_LINK_COUNT_MISMATCH)

    first_hash = _validate_sha(receipt.first_composition_slot_hash)
    final_hash = _validate_sha(receipt.final_composition_slot_hash)
    if not receipt.composition_slot_hashes or first_hash != receipt.composition_slot_hashes[0] or final_hash != receipt.composition_slot_hashes[-1]:
        raise ValueError(_ERR_SLOT_LINK_TOPOLOGY_MISMATCH)

    if not isinstance(receipt.slots_complete, bool) or not isinstance(receipt.links_complete, bool) or not isinstance(receipt.slot_link_topology_aligned, bool) or not isinstance(receipt.reality_loop_complete, bool):
        raise ValueError(_ERR_INVALID_INPUT)

    derived_slots_complete = slot_count == required_slot_count
    derived_links_complete = link_count == required_link_count
    if receipt.slots_complete != derived_slots_complete or receipt.links_complete != derived_links_complete:
        raise ValueError(_ERR_INVALID_INPUT)

    expected_complete, expected_class = _derive_completion_and_class(
        slot_count,
        required_slot_count,
        link_count,
        required_link_count,
        derived_slots_complete,
        derived_links_complete,
        receipt.slot_link_topology_aligned,
    )
    if receipt.reality_loop_complete != expected_complete:
        raise ValueError(_ERR_REALITY_LOOP_COMPLETION_MISMATCH)
    if receipt.reality_loop_proof_class not in get_allowed_reality_loop_proof_classes():
        raise ValueError(_ERR_INVALID_REALITY_LOOP_PROOF_CLASS)
    if receipt.reality_loop_proof_class != expected_class:
        raise ValueError(_ERR_REALITY_LOOP_PROOF_CLASS_MISMATCH)

    payload = _reality_loop_proof_receipt_payload(
        composition_spec_hash=composition_spec_hash,
        cross_arc_identity_link_receipt_hash=cross_hash,
        composition_mode=receipt.composition_mode,
        proof_mode=receipt.proof_mode,
        slot_count=slot_count,
        required_slot_count=required_slot_count,
        link_count=link_count,
        required_link_count=required_link_count,
        first_composition_slot_hash=first_hash,
        final_composition_slot_hash=final_hash,
        composition_slot_hashes=receipt.composition_slot_hashes,
        cross_arc_identity_link_hashes=receipt.cross_arc_identity_link_hashes,
        slots_complete=receipt.slots_complete,
        links_complete=receipt.links_complete,
        slot_link_topology_aligned=receipt.slot_link_topology_aligned,
        reality_loop_complete=receipt.reality_loop_complete,
        reality_loop_proof_class=receipt.reality_loop_proof_class,
    )
    stored = _validate_sha(receipt.reality_loop_proof_receipt_hash)
    if stored != sha256_hex(payload):
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_reality_loop_proof_receipt_with_artifacts(
    receipt: RealityLoopProofReceipt,
    composition_spec: RealityLoopCompositionSpec,
    cross_arc_identity_link_receipt: CrossArcIdentityLinkReceipt,
) -> bool:
    validate_reality_loop_composition_spec(composition_spec)
    # Validate cross-arc receipt generically first, then with composition spec
    validate_cross_arc_identity_link_receipt(cross_arc_identity_link_receipt)
    validate_cross_arc_identity_link_receipt_with_composition_spec(cross_arc_identity_link_receipt, composition_spec)
    validate_reality_loop_proof_receipt(receipt)
    rebuilt = build_reality_loop_proof_receipt(composition_spec, cross_arc_identity_link_receipt)
    if receipt.reality_loop_proof_receipt_hash != rebuilt.reality_loop_proof_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    if receipt.to_dict() != rebuilt.to_dict():
        raise ValueError(_ERR_REALITY_LOOP_PROOF_RECEIPT_MISMATCH)
    return True
