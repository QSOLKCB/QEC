from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_INDEX_MODE = "INVALID_INDEX_MODE"
_ERR_INVALID_ENTRY_INDEX = "INVALID_ENTRY_INDEX"
_ERR_INVALID_ARC_LABEL = "INVALID_ARC_LABEL"
_ERR_INVALID_RECEIPT_FIELD_NAME = "INVALID_RECEIPT_FIELD_NAME"
_ERR_ENTRY_DEFINITION_MISMATCH = "ENTRY_DEFINITION_MISMATCH"
_ERR_DUPLICATE_GLOBAL_VALIDATION_ENTRY = "DUPLICATE_GLOBAL_VALIDATION_ENTRY"
_ERR_MISSING_GLOBAL_VALIDATION_ENTRY = "MISSING_GLOBAL_VALIDATION_ENTRY"
_ERR_GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH = "GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH"
_ERR_GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH = "GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH"
_ERR_REALITY_LOOP_PROOF_HASH_MISMATCH = "REALITY_LOOP_PROOF_HASH_MISMATCH"
_ERR_GLOBAL_VALIDATION_INDEX_MISMATCH = "GLOBAL_VALIDATION_INDEX_MISMATCH"

_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT = 48
_MAX_GLOBAL_VALIDATION_ENTRY_INDEX = 47
_MAX_ARC_LABEL_LENGTH = 16
_MAX_RECEIPT_FIELD_NAME_LENGTH = 128
_INDEX_MODE_FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX = "FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARC_LABEL_RE = re.compile(r"^v[0-9]{3}$")
_FIELD_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_GLOBAL_VALIDATION_ENTRY_DEFINITIONS: tuple[tuple[int, str, str], ...] = (
    (0, "v151", "canonical_hash"), (1, "v151", "semantic_field_hash"), (2, "v151", "resonance_hash"),
    (3, "v152", "layered_hash"), (4, "v152", "fractal_equivalence_receipt_hash"), (5, "v153", "lattice_graph_hash"),
    (6, "v153", "router_lattice_path_receipt_hash"), (7, "v153", "readout_projection_receipt_hash"),
    (8, "v153", "layered_lattice_projection_hash"), (9, "v153", "qam_spec_hash"),
    (10, "v153", "mask_reduction_hash"), (11, "v153", "shift_projection_hash"),
    (12, "v153", "kernel_composition_hash"), (13, "v153", "readout_matrix_hash"),
    (14, "v153", "lattice_replay_proof_hash"), (15, "v154", "subgraph_invariant_pattern_hash"),
    (16, "v154", "multi_scale_invariant_receipt_hash"), (17, "v154", "sierpinski_compression_receipt_hash"),
    (18, "v154", "router_scale_receipt_hash"), (19, "v154", "readout_scale_projection_receipt_hash"),
    (20, "v154", "governance_compression_receipt_hash"), (21, "v154", "semantic_compression_receipt_hash"),
    (22, "v155", "decay_checkpoint_hash"), (23, "v155", "decay_checkpoint_set_hash"),
    (24, "v155", "decay_threshold_contract_hash"), (25, "v155", "digital_decay_signature_hash"),
    (26, "v155", "entropy_drift_receipt_hash"), (27, "v155", "decay_resistance_proof_hash"),
    (28, "v156", "game_world_archive_manifest_hash"), (29, "v156", "game_world_corpus_manifest_hash"),
    (30, "v156", "game_world_intake_receipt_hash"), (31, "v156", "action_atom_hash"),
    (32, "v156", "action_alphabet_hash"), (33, "v156", "world_adapter_spec_hash"),
    (34, "v156", "world_adapter_contract_receipt_hash"), (35, "v156", "observation_snapshot_receipt_hash"),
    (36, "v156", "episode_trace_receipt_hash"), (37, "v156", "strategy_probe_receipt_hash"),
    (38, "v156", "chaos_replay_verdict_hash"), (39, "v156", "game_world_interaction_report_hash"),
    (40, "v157", "perturbation_contract_hash"), (41, "v157", "energy_matrix_receipt_hash"),
    (42, "v157", "perturbation_stability_proof_hash"), (43, "v158", "substrate_contract_hash"),
    (44, "v158", "substrate_state_receipt_hash"), (45, "v159", "recursive_proof_receipt_hash"),
    (46, "v159", "loop_termination_proof_hash"), (47, "v160", "reality_loop_proof_receipt_hash"),
)


def get_global_validation_entry_definitions() -> tuple[tuple[int, str, str], ...]:
    return _GLOBAL_VALIDATION_ENTRY_DEFINITIONS


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_entry_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > _MAX_GLOBAL_VALIDATION_ENTRY_INDEX:
        raise ValueError(_ERR_INVALID_ENTRY_INDEX)
    return v


def _validate_count(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH)
    return v


def _global_validation_entry_payload(entry_index: int, arc_label: str, receipt_field_name: str, receipt_hash: str) -> dict[str, Any]:
    return {
        "entry_index": entry_index,
        "arc_label": arc_label,
        "receipt_field_name": receipt_field_name,
        "receipt_hash": receipt_hash,
    }


def _global_validation_index_payload(index_mode: str, global_validation_entries: tuple[GlobalValidationEntry, ...], entry_count: int, required_entry_count: int, first_global_validation_entry_hash: str, final_global_validation_entry_hash: str, reality_loop_proof_receipt_hash: str) -> dict[str, Any]:
    return {
        "index_mode": index_mode,
        "global_validation_entries": [e.to_dict() for e in global_validation_entries],
        "entry_count": entry_count,
        "required_entry_count": required_entry_count,
        "first_global_validation_entry_hash": first_global_validation_entry_hash,
        "final_global_validation_entry_hash": final_global_validation_entry_hash,
        "reality_loop_proof_receipt_hash": reality_loop_proof_receipt_hash,
    }


@dataclass(frozen=True)
class GlobalValidationEntry:
    entry_index: int
    arc_label: str
    receipt_field_name: str
    receipt_hash: str
    global_validation_entry_hash: str

    def __post_init__(self) -> None:
        validate_global_validation_entry(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _global_validation_entry_payload(self.entry_index, self.arc_label, self.receipt_field_name, self.receipt_hash)
        payload["global_validation_entry_hash"] = self.global_validation_entry_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class GlobalValidationIndex:
    index_mode: str
    global_validation_entries: tuple[GlobalValidationEntry, ...]
    entry_count: int
    required_entry_count: int
    first_global_validation_entry_hash: str
    final_global_validation_entry_hash: str
    reality_loop_proof_receipt_hash: str
    global_validation_index_hash: str

    def __post_init__(self) -> None:
        validate_global_validation_index(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _global_validation_index_payload(
            self.index_mode, self.global_validation_entries, self.entry_count, self.required_entry_count,
            self.first_global_validation_entry_hash, self.final_global_validation_entry_hash,
            self.reality_loop_proof_receipt_hash,
        )
        payload["global_validation_index_hash"] = self.global_validation_index_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_global_validation_entry(entry_index: int, receipt_hash: str) -> GlobalValidationEntry:
    idx = _validate_entry_index(entry_index)
    _, arc_label, field_name = _GLOBAL_VALIDATION_ENTRY_DEFINITIONS[idx]
    payload = _global_validation_entry_payload(idx, arc_label, field_name, _validate_sha(receipt_hash))
    return GlobalValidationEntry(idx, arc_label, field_name, receipt_hash, sha256_hex(canonical_json(payload)))


def build_global_validation_index(receipt_hashes_by_field: dict[str, str]) -> GlobalValidationIndex:
    if not isinstance(receipt_hashes_by_field, dict):
        raise ValueError(_ERR_INVALID_INPUT)
    forbidden = {
        "global_" + "threshold_contract_hash",
        "global_" + "truth_receipt_hash",
        "replay_" + "record_hash",
        "global_" + "replay_proof_hash",
    }
    defs = _GLOBAL_VALIDATION_ENTRY_DEFINITIONS
    required_fields = {d[2] for d in defs}
    for k, v in receipt_hashes_by_field.items():
        if not isinstance(k, str):
            raise ValueError(_ERR_INVALID_RECEIPT_FIELD_NAME)
        if k in forbidden:
            raise ValueError(_ERR_ENTRY_DEFINITION_MISMATCH)
        if _FIELD_NAME_RE.fullmatch(k) is None or len(k) > _MAX_RECEIPT_FIELD_NAME_LENGTH:
            raise ValueError(_ERR_INVALID_RECEIPT_FIELD_NAME)
        _validate_sha(v)
        if k not in required_fields:
            raise ValueError(_ERR_ENTRY_DEFINITION_MISMATCH)
    if set(receipt_hashes_by_field.keys()) != required_fields:
        raise ValueError(_ERR_MISSING_GLOBAL_VALIDATION_ENTRY)
    entries = tuple(build_global_validation_entry(i, receipt_hashes_by_field[f]) for i, _, f in defs)
    payload = _global_validation_index_payload(_INDEX_MODE_FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX, entries, 48, 48, entries[0].global_validation_entry_hash, entries[-1].global_validation_entry_hash, entries[-1].receipt_hash)
    return GlobalValidationIndex(_INDEX_MODE_FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX, entries, 48, 48, entries[0].global_validation_entry_hash, entries[-1].global_validation_entry_hash, entries[-1].receipt_hash, sha256_hex(canonical_json(payload)))


def build_global_validation_index_from_ordered_hashes(receipt_hashes: list[str] | tuple[str, ...]) -> GlobalValidationIndex:
    if not isinstance(receipt_hashes, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if len(receipt_hashes) != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT:
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH)
    return build_global_validation_index({field: h for _, _, field, h in ((d[0], d[1], d[2], receipt_hashes[d[0]]) for d in _GLOBAL_VALIDATION_ENTRY_DEFINITIONS)})


def validate_global_validation_entry(entry: GlobalValidationEntry) -> bool:
    if not isinstance(entry, GlobalValidationEntry):
        raise ValueError(_ERR_INVALID_INPUT)
    idx = _validate_entry_index(entry.entry_index)
    if not isinstance(entry.arc_label, str) or len(entry.arc_label) == 0 or len(entry.arc_label) > _MAX_ARC_LABEL_LENGTH or _ARC_LABEL_RE.fullmatch(entry.arc_label) is None:
        raise ValueError(_ERR_INVALID_ARC_LABEL)
    if not isinstance(entry.receipt_field_name, str) or len(entry.receipt_field_name) == 0 or len(entry.receipt_field_name) > _MAX_RECEIPT_FIELD_NAME_LENGTH or _FIELD_NAME_RE.fullmatch(entry.receipt_field_name) is None:
        raise ValueError(_ERR_INVALID_RECEIPT_FIELD_NAME)
    _validate_sha(entry.receipt_hash)
    _validate_sha(entry.global_validation_entry_hash)
    expected = _GLOBAL_VALIDATION_ENTRY_DEFINITIONS[idx]
    if (idx, entry.arc_label, entry.receipt_field_name) != expected:
        raise ValueError(_ERR_ENTRY_DEFINITION_MISMATCH)
    payload = _global_validation_entry_payload(entry.entry_index, entry.arc_label, entry.receipt_field_name, entry.receipt_hash)
    if sha256_hex(canonical_json(payload)) != entry.global_validation_entry_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_global_validation_index(index: GlobalValidationIndex) -> bool:
    if not isinstance(index, GlobalValidationIndex):
        raise ValueError(_ERR_INVALID_INPUT)
    if index.index_mode != _INDEX_MODE_FIXED_V151_TO_V160_GLOBAL_VALIDATION_INDEX:
        raise ValueError(_ERR_INVALID_INDEX_MODE)
    if not isinstance(index.global_validation_entries, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(index.first_global_validation_entry_hash)
    _validate_sha(index.final_global_validation_entry_hash)
    _validate_sha(index.reality_loop_proof_receipt_hash)
    _validate_sha(index.global_validation_index_hash)
    ec = _validate_count(index.entry_count)
    rc = _validate_count(index.required_entry_count)
    if ec != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT or rc != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT or ec != rc:
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH)
    if len(index.global_validation_entries) != _REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT:
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_COUNT_MISMATCH)
    seen_idx: set[int] = set(); seen_field: set[str] = set()
    for i, entry in enumerate(index.global_validation_entries):
        validate_global_validation_entry(entry)
        if entry.entry_index in seen_idx or entry.receipt_field_name in seen_field:
            raise ValueError(_ERR_DUPLICATE_GLOBAL_VALIDATION_ENTRY)
        seen_idx.add(entry.entry_index); seen_field.add(entry.receipt_field_name)
        if entry.entry_index != i:
            raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH)
    if seen_idx != set(range(_REQUIRED_GLOBAL_VALIDATION_ENTRY_COUNT)):
        raise ValueError(_ERR_MISSING_GLOBAL_VALIDATION_ENTRY)
    if index.first_global_validation_entry_hash != index.global_validation_entries[0].global_validation_entry_hash:
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH)
    if index.final_global_validation_entry_hash != index.global_validation_entries[-1].global_validation_entry_hash:
        raise ValueError(_ERR_GLOBAL_VALIDATION_ENTRY_ORDER_MISMATCH)
    if index.reality_loop_proof_receipt_hash != index.global_validation_entries[-1].receipt_hash:
        raise ValueError(_ERR_REALITY_LOOP_PROOF_HASH_MISMATCH)
    payload = _global_validation_index_payload(index.index_mode, index.global_validation_entries, index.entry_count, index.required_entry_count, index.first_global_validation_entry_hash, index.final_global_validation_entry_hash, index.reality_loop_proof_receipt_hash)
    if sha256_hex(canonical_json(payload)) != index.global_validation_index_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_global_validation_index_matches_hashes(index: GlobalValidationIndex, receipt_hashes_by_field: dict[str, str]) -> bool:
    validate_global_validation_index(index)
    expected = build_global_validation_index(receipt_hashes_by_field)
    if expected.to_dict() != index.to_dict():
        raise ValueError(_ERR_GLOBAL_VALIDATION_INDEX_MISMATCH)
    return True
