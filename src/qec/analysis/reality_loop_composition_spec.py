from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_COMPOSITION_MODE = "INVALID_COMPOSITION_MODE"
_ERR_INVALID_SLOT_INDEX = "INVALID_SLOT_INDEX"
_ERR_INVALID_ARC_LABEL = "INVALID_ARC_LABEL"
_ERR_INVALID_RECEIPT_FIELD_NAME = "INVALID_RECEIPT_FIELD_NAME"
_ERR_SLOT_DEFINITION_MISMATCH = "SLOT_DEFINITION_MISMATCH"
_ERR_DUPLICATE_COMPOSITION_SLOT = "DUPLICATE_COMPOSITION_SLOT"
_ERR_MISSING_COMPOSITION_SLOT = "MISSING_COMPOSITION_SLOT"
_ERR_COMPOSITION_SLOT_ORDER_MISMATCH = "COMPOSITION_SLOT_ORDER_MISMATCH"
_ERR_COMPOSITION_SLOT_COUNT_MISMATCH = "COMPOSITION_SLOT_COUNT_MISMATCH"
_ERR_COMPOSITION_SPEC_MISMATCH = "COMPOSITION_SPEC_MISMATCH"

_REQUIRED_SLOT_COUNT = 19
_MAX_SLOT_INDEX = 18
_MAX_ARC_LABEL_LENGTH = 16
_MAX_RECEIPT_FIELD_NAME_LENGTH = 96

COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION = "FIXED_19_SLOT_REALITY_LOOP_COMPOSITION"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARC_LABEL_RE = re.compile(r"^v[0-9]{3}$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_REALITY_LOOP_SLOT_DEFINITIONS: tuple[tuple[int, str, str], ...] = (
    (0, "v151", "canonical_hash"),
    (1, "v151", "semantic_field_hash"),
    (2, "v151", "resonance_hash"),
    (3, "v152", "layered_hash"),
    (4, "v152", "fractal_equivalence_receipt_hash"),
    (5, "v153", "lattice_graph_hash"),
    (6, "v153", "router_lattice_path_receipt_hash"),
    (7, "v153", "readout_projection_receipt_hash"),
    (8, "v153", "lattice_replay_proof_hash"),
    (9, "v154", "multi_scale_invariant_receipt_hash"),
    (10, "v154", "sierpinski_compression_receipt_hash"),
    (11, "v155", "digital_decay_signature_hash"),
    (12, "v155", "entropy_drift_receipt_hash"),
    (13, "v156", "game_world_interaction_report_hash"),
    (14, "v157", "energy_matrix_receipt_hash"),
    (15, "v157", "perturbation_stability_proof_hash"),
    (16, "v158", "substrate_state_receipt_hash"),
    (17, "v159", "recursive_proof_receipt_hash"),
    (18, "v159", "loop_termination_proof_hash"),
)


def get_reality_loop_slot_definitions() -> tuple[tuple[int, str, str], ...]:
    return _REALITY_LOOP_SLOT_DEFINITIONS


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_slot_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(_ERR_INVALID_SLOT_INDEX)
    if v < 0 or v > _MAX_SLOT_INDEX:
        raise ValueError(_ERR_INVALID_SLOT_INDEX)
    return v


def _validate_arc_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_ARC_LABEL_LENGTH or _ARC_LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_ARC_LABEL)
    return v


def _validate_receipt_field_name(v: object) -> str:
    if (
        not isinstance(v, str)
        or not v
        or len(v) > _MAX_RECEIPT_FIELD_NAME_LENGTH
        or _FIELD_NAME_RE.fullmatch(v) is None
    ):
        raise ValueError(_ERR_INVALID_RECEIPT_FIELD_NAME)
    return v


def _composition_slot_payload(slot_index: int, arc_label: str, receipt_field_name: str, receipt_hash: str) -> dict[str, Any]:
    return {
        "slot_index": slot_index,
        "arc_label": arc_label,
        "receipt_field_name": receipt_field_name,
        "receipt_hash": receipt_hash,
    }


@dataclass(frozen=True)
class CompositionSlot:
    slot_index: int
    arc_label: str
    receipt_field_name: str
    receipt_hash: str
    composition_slot_hash: str

    def __post_init__(self) -> None:
        validate_composition_slot(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _composition_slot_payload(
            slot_index=self.slot_index,
            arc_label=self.arc_label,
            receipt_field_name=self.receipt_field_name,
            receipt_hash=self.receipt_hash,
        )
        payload["composition_slot_hash"] = self.composition_slot_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_composition_slot(slot_index: int, receipt_hash: str) -> CompositionSlot:
    idx = _validate_slot_index(slot_index)
    _validate_sha(receipt_hash)
    expected_idx, arc_label, receipt_field_name = _REALITY_LOOP_SLOT_DEFINITIONS[idx]
    if expected_idx != idx:
        raise ValueError(_ERR_SLOT_DEFINITION_MISMATCH)
    payload = _composition_slot_payload(idx, arc_label, receipt_field_name, receipt_hash)
    return CompositionSlot(
        slot_index=idx,
        arc_label=arc_label,
        receipt_field_name=receipt_field_name,
        receipt_hash=receipt_hash,
        composition_slot_hash=sha256_hex(payload),
    )


def validate_composition_slot(slot: CompositionSlot) -> bool:
    if not isinstance(slot, CompositionSlot):
        raise ValueError(_ERR_INVALID_INPUT)
    idx = _validate_slot_index(slot.slot_index)
    arc_label = _validate_arc_label(slot.arc_label)
    receipt_field_name = _validate_receipt_field_name(slot.receipt_field_name)
    receipt_hash = _validate_sha(slot.receipt_hash)
    stored_hash = _validate_sha(slot.composition_slot_hash)

    expected_idx, expected_arc, expected_field = _REALITY_LOOP_SLOT_DEFINITIONS[idx]
    if expected_idx != idx or arc_label != expected_arc or receipt_field_name != expected_field:
        raise ValueError(_ERR_SLOT_DEFINITION_MISMATCH)

    expected_hash = sha256_hex(
            _composition_slot_payload(
                slot_index=idx,
                arc_label=arc_label,
                receipt_field_name=receipt_field_name,
                receipt_hash=receipt_hash,
            )
    )
    if stored_hash != expected_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def _reality_loop_composition_spec_payload(
    composition_mode: str,
    composition_slots: tuple[CompositionSlot, ...],
) -> dict[str, Any]:
    return {
        "composition_mode": composition_mode,
        "composition_slots": [slot.to_dict() for slot in composition_slots],
        "slot_count": len(composition_slots),
        "required_slot_count": _REQUIRED_SLOT_COUNT,
    }


@dataclass(frozen=True)
class RealityLoopCompositionSpec:
    composition_mode: str
    composition_slots: tuple[CompositionSlot, ...]
    slot_count: int
    required_slot_count: int
    composition_spec_hash: str

    def __post_init__(self) -> None:
        validate_reality_loop_composition_spec(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _reality_loop_composition_spec_payload(
            composition_mode=self.composition_mode,
            composition_slots=self.composition_slots,
        )
        payload["composition_spec_hash"] = self.composition_spec_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_reality_loop_composition_spec(receipt_hashes_by_field: dict[str, str]) -> RealityLoopCompositionSpec:
    if not isinstance(receipt_hashes_by_field, dict):
        raise ValueError(_ERR_INVALID_INPUT)

    required_fields = tuple(field for _, _, field in _REALITY_LOOP_SLOT_DEFINITIONS)
    supplied_fields = set()
    for key, value in receipt_hashes_by_field.items():
        _validate_receipt_field_name(key)
        _validate_sha(value)
        supplied_fields.add(key)

    required_set = set(required_fields)
    missing = required_set - supplied_fields
    if missing:
        raise ValueError(_ERR_MISSING_COMPOSITION_SLOT)
    extras = supplied_fields - required_set
    if extras:
        raise ValueError(_ERR_SLOT_DEFINITION_MISMATCH)

    slots = tuple(
        build_composition_slot(slot_index=slot_index, receipt_hash=receipt_hashes_by_field[field_name])
        for slot_index, _, field_name in _REALITY_LOOP_SLOT_DEFINITIONS
    )
    payload = _reality_loop_composition_spec_payload(
        composition_mode=COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION,
        composition_slots=slots,
    )
    return RealityLoopCompositionSpec(
        composition_mode=COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION,
        composition_slots=slots,
        slot_count=_REQUIRED_SLOT_COUNT,
        required_slot_count=_REQUIRED_SLOT_COUNT,
        composition_spec_hash=sha256_hex(payload),
    )


def build_reality_loop_composition_spec_from_ordered_hashes(
    receipt_hashes: list[str] | tuple[str, ...],
) -> RealityLoopCompositionSpec:
    if not isinstance(receipt_hashes, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(receipt_hashes) != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_COMPOSITION_SLOT_COUNT_MISMATCH)
    mapping = {
        field_name: receipt_hashes[idx]
        for idx, (_, _, field_name) in enumerate(_REALITY_LOOP_SLOT_DEFINITIONS)
    }
    return build_reality_loop_composition_spec(mapping)


def validate_reality_loop_composition_spec(spec: RealityLoopCompositionSpec) -> bool:
    if not isinstance(spec, RealityLoopCompositionSpec):
        raise ValueError(_ERR_INVALID_INPUT)
    if spec.composition_mode != COMPOSITION_MODE_FIXED_19_SLOT_REALITY_LOOP_COMPOSITION:
        raise ValueError(_ERR_INVALID_COMPOSITION_MODE)
    if not isinstance(spec.composition_slots, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(spec.slot_count, int) or isinstance(spec.slot_count, bool):
        raise ValueError(_ERR_COMPOSITION_SLOT_COUNT_MISMATCH)
    if not isinstance(spec.required_slot_count, int) or isinstance(spec.required_slot_count, bool):
        raise ValueError(_ERR_COMPOSITION_SLOT_COUNT_MISMATCH)
    if spec.slot_count != _REQUIRED_SLOT_COUNT or spec.required_slot_count != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_COMPOSITION_SLOT_COUNT_MISMATCH)
    if len(spec.composition_slots) != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_COMPOSITION_SLOT_COUNT_MISMATCH)

    seen_idx: set[int] = set()
    seen_field: set[str] = set()
    for pos, slot in enumerate(spec.composition_slots):
        validate_composition_slot(slot)
        expected_idx, _, expected_field = _REALITY_LOOP_SLOT_DEFINITIONS[pos]
        if slot.slot_index != expected_idx:
            raise ValueError(_ERR_COMPOSITION_SLOT_ORDER_MISMATCH)
        if slot.slot_index in seen_idx or slot.receipt_field_name in seen_field:
            raise ValueError(_ERR_DUPLICATE_COMPOSITION_SLOT)
        seen_idx.add(slot.slot_index)
        seen_field.add(slot.receipt_field_name)
        if slot.receipt_field_name != expected_field:
            raise ValueError(_ERR_COMPOSITION_SLOT_ORDER_MISMATCH)

    if len(seen_idx) != _REQUIRED_SLOT_COUNT or len(seen_field) != _REQUIRED_SLOT_COUNT:
        raise ValueError(_ERR_MISSING_COMPOSITION_SLOT)

    stored_hash = _validate_sha(spec.composition_spec_hash)
    expected_hash = sha256_hex(
            _reality_loop_composition_spec_payload(
                composition_mode=spec.composition_mode,
                composition_slots=spec.composition_slots,
            )
    )
    if stored_hash != expected_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_reality_loop_composition_spec_matches_hashes(
    spec: RealityLoopCompositionSpec,
    receipt_hashes_by_field: dict[str, str],
) -> bool:
    validate_reality_loop_composition_spec(spec)
    expected = build_reality_loop_composition_spec(receipt_hashes_by_field)
    if spec.composition_spec_hash != expected.composition_spec_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    if spec.to_dict() != expected.to_dict():
        raise ValueError(_ERR_COMPOSITION_SPEC_MISMATCH)
    return True
