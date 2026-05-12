from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Callable

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .perturbation_contract import (
    PerturbationContract,
    PerturbationResult,
    get_allowed_perturbation_operation_types,
    validate_perturbation_contract,
    validate_perturbation_result,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_MATRIX_MODE = "INVALID_MATRIX_MODE"
_ERR_INVALID_MATRIX_LABEL = "INVALID_MATRIX_LABEL"
_ERR_INVALID_ENTRY_LABEL = "INVALID_ENTRY_LABEL"
_ERR_MATRIX_INDEX_OUT_OF_BOUNDS = "MATRIX_INDEX_OUT_OF_BOUNDS"
_ERR_MATRIX_DIMENSION_MISMATCH = "MATRIX_DIMENSION_MISMATCH"
_ERR_MATRIX_COUNT_MISMATCH = "MATRIX_COUNT_MISMATCH"
_ERR_DUPLICATE_MATRIX_ENTRY = "DUPLICATE_MATRIX_ENTRY"
_ERR_MATRIX_ORDER_MISMATCH = "MATRIX_ORDER_MISMATCH"
_ERR_ENTRY_CONTRACT_RESULT_MISMATCH = "ENTRY_CONTRACT_RESULT_MISMATCH"
_ERR_ENTRY_TARGET_MISMATCH = "ENTRY_TARGET_MISMATCH"
_ERR_ENTRY_OPERATION_MISMATCH = "ENTRY_OPERATION_MISMATCH"
_ERR_ENTRY_CHANGED_MISMATCH = "ENTRY_CHANGED_MISMATCH"
_ERR_IMPACT_SCORE_MISMATCH = "IMPACT_SCORE_MISMATCH"
_ERR_IMPACT_SCORE_OUT_OF_BOUNDS = "IMPACT_SCORE_OUT_OF_BOUNDS"
_ERR_ENERGY_RECEIPT_MISMATCH = "ENERGY_RECEIPT_MISMATCH"

_MAX_MATRIX_ENTRIES = 10_000
_MAX_MATRIX_ROWS = 1_000
_MAX_MATRIX_COLUMNS = 1_000
_MAX_MATRIX_INDEX = 999
_MAX_LABEL_LENGTH = 96
_MAX_ABS_INTEGER_IMPACT_SCORE = 1_000_000_000
_MAX_ABS_TOTAL_IMPACT_SCORE = 1_000_000_000_000

_MATRIX_MODE = "CANONICAL_PERTURBATION_MATRIX"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MAX_TARGET_ARTIFACT_TYPE_LENGTH = 96


def _validate_sha256_hex(value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)


def _validate_label(value: object, err: str) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_LABEL_LENGTH:
        raise ValueError(err)
    if _LABEL_RE.fullmatch(value) is None:
        raise ValueError(err)
    return value


def _validate_index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(_ERR_MATRIX_INDEX_OUT_OF_BOUNDS)
    if value < 0 or value > _MAX_MATRIX_INDEX:
        raise ValueError(_ERR_MATRIX_INDEX_OUT_OF_BOUNDS)
    return value


def _validate_int_score(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    if abs(value) > _MAX_ABS_INTEGER_IMPACT_SCORE:
        raise ValueError(_ERR_IMPACT_SCORE_OUT_OF_BOUNDS)
    return value


def _validate_target_artifact_type(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > _MAX_TARGET_ARTIFACT_TYPE_LENGTH:
        raise ValueError(_ERR_INVALID_INPUT)
    if _ARTIFACT_TYPE_RE.fullmatch(value) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return value


def _derive_integer_impact_score(changed: bool) -> int:
    return 1 if changed else 0


def _matrix_entry_order_key(entry: PerturbationMatrixEntry) -> tuple[int, int, str, str]:
    return (entry.row_index, entry.column_index, entry.entry_label, entry.perturbation_matrix_entry_hash)


def _validate_non_bool_int(value: object, *, minimum: int | None = None, maximum: int | None = None, err: str = _ERR_INVALID_INPUT) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(err)
    if minimum is not None and value < minimum:
        raise ValueError(err)
    if maximum is not None and value > maximum:
        raise ValueError(err)
    return value


def _validate_type_counts(
    value: object,
    key_validator: Callable[[object], None],
    max_length: int | None = None,
) -> tuple[tuple[str, int], ...]:
    """Generic validator for type count tuples.

    Validates that value is a tuple of (key, count) pairs where:
    - Each item is a tuple of exactly 2 elements
    - Keys pass the provided key_validator
    - Counts are positive integers (non-bool)
    - Keys are unique and sorted in ascending order
    - Total length does not exceed max_length (if provided)
    """
    if not isinstance(value, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if max_length is not None and len(value) > max_length:
        raise ValueError(_ERR_INVALID_INPUT)
    out: list[tuple[str, int]] = []
    last_key: str | None = None
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(_ERR_INVALID_INPUT)
        key, count = item
        key_validator(key)
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(_ERR_INVALID_INPUT)
        if key in seen:
            raise ValueError(_ERR_INVALID_INPUT)
        if last_key is not None and key < last_key:
            raise ValueError(_ERR_INVALID_INPUT)
        seen.add(key)
        last_key = key
        out.append((key, count))
    return tuple(out)


def _validate_operation_type_key(key: object) -> None:
    """Validate that key is an allowed operation type."""
    if not isinstance(key, str) or key not in get_allowed_perturbation_operation_types():
        raise ValueError(_ERR_INVALID_INPUT)


def _validate_operation_type_counts(value: object, max_length: int | None = None) -> tuple[tuple[str, int], ...]:
    return _validate_type_counts(value, _validate_operation_type_key, max_length)


def _validate_target_artifact_type_counts(value: object, max_length: int | None = None) -> tuple[tuple[str, int], ...]:
    return _validate_type_counts(value, _validate_target_artifact_type, max_length)


def _canonicalize_counts(counter: Counter) -> tuple[tuple[str, int], ...]:
    """Produce a canonicalized count tuple from a Counter, sorted by key."""
    return tuple(sorted(counter.items(), key=lambda x: x[0]))


def _perturbation_matrix_entry_payload(row_index: int, column_index: int, entry_label: str, perturbation_contract_hash: str, perturbation_result_hash: str, target_artifact_type: str, target_artifact_hash: str, operation_type: str, changed: bool, integer_impact_score: int) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "column_index": column_index,
        "entry_label": entry_label,
        "perturbation_contract_hash": perturbation_contract_hash,
        "perturbation_result_hash": perturbation_result_hash,
        "target_artifact_type": target_artifact_type,
        "target_artifact_hash": target_artifact_hash,
        "operation_type": operation_type,
        "changed": changed,
        "integer_impact_score": integer_impact_score,
    }


def _perturbation_matrix_payload(matrix_label: str, matrix_mode: str, entries: tuple['PerturbationMatrixEntry', ...], row_count: int, column_count: int, entry_count: int, changed_entry_count: int, unchanged_entry_count: int) -> dict[str, Any]:
    return {
        "matrix_label": matrix_label,
        "matrix_mode": matrix_mode,
        "entries": [e.to_dict() for e in entries],
        "row_count": row_count,
        "column_count": column_count,
        "entry_count": entry_count,
        "changed_entry_count": changed_entry_count,
        "unchanged_entry_count": unchanged_entry_count,
    }


def _energy_matrix_receipt_payload(perturbation_matrix_hash: str, matrix_label: str, matrix_mode: str, total_integer_impact_score: int, changed_entry_count: int, unchanged_entry_count: int, operation_type_counts: tuple[tuple[str, int], ...], target_artifact_type_counts: tuple[tuple[str, int], ...], perturbation_matrix: 'PerturbationMatrix') -> dict[str, Any]:
    return {
        "perturbation_matrix_hash": perturbation_matrix_hash,
        "matrix_label": matrix_label,
        "matrix_mode": matrix_mode,
        "total_integer_impact_score": total_integer_impact_score,
        "changed_entry_count": changed_entry_count,
        "unchanged_entry_count": unchanged_entry_count,
        "operation_type_counts": [[k, v] for k, v in operation_type_counts],
        "target_artifact_type_counts": [[k, v] for k, v in target_artifact_type_counts],
        "perturbation_matrix": perturbation_matrix.to_dict(),
    }

@dataclass(frozen=True)
class PerturbationMatrixEntry:
    row_index: int
    column_index: int
    entry_label: str
    perturbation_contract_hash: str
    perturbation_result_hash: str
    target_artifact_type: str
    target_artifact_hash: str
    operation_type: str
    changed: bool
    integer_impact_score: int
    perturbation_matrix_entry_hash: str
    def __post_init__(self) -> None:
        validate_perturbation_matrix_entry(self)
    def _hash_payload(self) -> dict[str, Any]:
        return _perturbation_matrix_entry_payload(self.row_index, self.column_index, self.entry_label, self.perturbation_contract_hash, self.perturbation_result_hash, self.target_artifact_type, self.target_artifact_hash, self.operation_type, self.changed, self.integer_impact_score)
    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "perturbation_matrix_entry_hash": self.perturbation_matrix_entry_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class PerturbationMatrix:
    matrix_label: str
    matrix_mode: str
    entries: tuple[PerturbationMatrixEntry, ...]
    row_count: int
    column_count: int
    entry_count: int
    changed_entry_count: int
    unchanged_entry_count: int
    perturbation_matrix_hash: str
    def __post_init__(self) -> None:
        validate_perturbation_matrix(self)
    def _hash_payload(self) -> dict[str, Any]:
        return _perturbation_matrix_payload(self.matrix_label, self.matrix_mode, self.entries, self.row_count, self.column_count, self.entry_count, self.changed_entry_count, self.unchanged_entry_count)
    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "perturbation_matrix_hash": self.perturbation_matrix_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

@dataclass(frozen=True)
class EnergyMatrixReceipt:
    perturbation_matrix_hash: str
    matrix_label: str
    matrix_mode: str
    total_integer_impact_score: int
    changed_entry_count: int
    unchanged_entry_count: int
    operation_type_counts: tuple[tuple[str, int], ...]
    target_artifact_type_counts: tuple[tuple[str, int], ...]
    perturbation_matrix: PerturbationMatrix
    energy_matrix_receipt_hash: str
    def __post_init__(self) -> None:
        validate_energy_matrix_receipt(self)
    def _hash_payload(self) -> dict[str, Any]:
        return _energy_matrix_receipt_payload(self.perturbation_matrix_hash, self.matrix_label, self.matrix_mode, self.total_integer_impact_score, self.changed_entry_count, self.unchanged_entry_count, self.operation_type_counts, self.target_artifact_type_counts, self.perturbation_matrix)
    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "energy_matrix_receipt_hash": self.energy_matrix_receipt_hash}
    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())
    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_perturbation_matrix_entry(row_index: int, column_index: int, entry_label: str, perturbation_contract: PerturbationContract, perturbation_result: PerturbationResult) -> PerturbationMatrixEntry:
    _validate_index(row_index)
    _validate_index(column_index)
    _validate_label(entry_label, _ERR_INVALID_ENTRY_LABEL)
    validate_perturbation_contract(perturbation_contract)
    validate_perturbation_result(perturbation_result)
    if perturbation_result.perturbation_contract_hash != perturbation_contract.perturbation_contract_hash:
        raise ValueError(_ERR_ENTRY_CONTRACT_RESULT_MISMATCH)
    if perturbation_result.target_artifact_type != perturbation_contract.target_artifact_type:
        raise ValueError(_ERR_ENTRY_TARGET_MISMATCH)
    if perturbation_result.target_artifact_hash != perturbation_contract.target_artifact_hash:
        raise ValueError(_ERR_ENTRY_TARGET_MISMATCH)
    if perturbation_result.target_field_path != perturbation_contract.target_field_path:
        raise ValueError(_ERR_ENTRY_TARGET_MISMATCH)
    if perturbation_result.operation_type != perturbation_contract.operation_type:
        raise ValueError(_ERR_ENTRY_OPERATION_MISMATCH)
    integer_impact_score = _derive_integer_impact_score(perturbation_result.changed)
    payload = _perturbation_matrix_entry_payload(row_index, column_index, entry_label, perturbation_contract.perturbation_contract_hash, perturbation_result.perturbation_result_hash, perturbation_result.target_artifact_type, perturbation_result.target_artifact_hash, perturbation_result.operation_type, perturbation_result.changed, integer_impact_score)
    return PerturbationMatrixEntry(**payload, perturbation_matrix_entry_hash=sha256_hex(payload))


def build_perturbation_matrix(matrix_label: str, entries: list[PerturbationMatrixEntry] | tuple[PerturbationMatrixEntry, ...]) -> PerturbationMatrix:
    _validate_label(matrix_label, _ERR_INVALID_MATRIX_LABEL)
    if not isinstance(entries, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    entries_tuple = tuple(entries)
    if not entries_tuple:
        raise ValueError(_ERR_INVALID_INPUT)
    if len(entries_tuple) > _MAX_MATRIX_ENTRIES:
        raise ValueError(_ERR_MATRIX_COUNT_MISMATCH)
    for entry in entries_tuple:
        validate_perturbation_matrix_entry(entry)
        _validate_index(entry.row_index)
        _validate_index(entry.column_index)
    seen_coords: set[tuple[int, int]] = set()
    seen_labels: set[str] = set()
    for entry in entries_tuple:
        coord = (entry.row_index, entry.column_index)
        if coord in seen_coords or entry.entry_label in seen_labels:
            raise ValueError(_ERR_DUPLICATE_MATRIX_ENTRY)
        seen_coords.add(coord)
        seen_labels.add(entry.entry_label)
    sorted_entries = tuple(sorted(entries_tuple, key=_matrix_entry_order_key))
    row_count = max(e.row_index for e in sorted_entries) + 1
    column_count = max(e.column_index for e in sorted_entries) + 1
    if row_count > _MAX_MATRIX_ROWS or column_count > _MAX_MATRIX_COLUMNS:
        raise ValueError(_ERR_MATRIX_DIMENSION_MISMATCH)
    changed_entry_count = sum(1 for e in sorted_entries if e.changed)
    unchanged_entry_count = len(sorted_entries) - changed_entry_count
    hash_payload = _perturbation_matrix_payload(matrix_label, _MATRIX_MODE, sorted_entries, row_count, column_count, len(sorted_entries), changed_entry_count, unchanged_entry_count)
    return PerturbationMatrix(
        matrix_label=matrix_label,
        matrix_mode=_MATRIX_MODE,
        entries=sorted_entries,
        row_count=row_count,
        column_count=column_count,
        entry_count=len(sorted_entries),
        changed_entry_count=changed_entry_count,
        unchanged_entry_count=unchanged_entry_count,
        perturbation_matrix_hash=sha256_hex(hash_payload),
    )


def build_energy_matrix_receipt(perturbation_matrix: PerturbationMatrix) -> EnergyMatrixReceipt:
    validate_perturbation_matrix(perturbation_matrix)
    total_integer_impact_score = sum(e.integer_impact_score for e in perturbation_matrix.entries)
    if abs(total_integer_impact_score) > _MAX_ABS_TOTAL_IMPACT_SCORE:
        raise ValueError(_ERR_IMPACT_SCORE_OUT_OF_BOUNDS)
    changed_count = sum(1 for e in perturbation_matrix.entries if e.changed)
    unchanged_count = len(perturbation_matrix.entries) - changed_count
    if changed_count != perturbation_matrix.changed_entry_count or unchanged_count != perturbation_matrix.unchanged_entry_count:
        raise ValueError(_ERR_ENERGY_RECEIPT_MISMATCH)
    op_counts = _canonicalize_counts(Counter(e.operation_type for e in perturbation_matrix.entries))
    tt_counts = _canonicalize_counts(Counter(e.target_artifact_type for e in perturbation_matrix.entries))
    hash_payload = _energy_matrix_receipt_payload(perturbation_matrix.perturbation_matrix_hash, perturbation_matrix.matrix_label, perturbation_matrix.matrix_mode, total_integer_impact_score, changed_count, unchanged_count, op_counts, tt_counts, perturbation_matrix)
    return EnergyMatrixReceipt(
        perturbation_matrix_hash=perturbation_matrix.perturbation_matrix_hash,
        matrix_label=perturbation_matrix.matrix_label,
        matrix_mode=perturbation_matrix.matrix_mode,
        total_integer_impact_score=total_integer_impact_score,
        changed_entry_count=changed_count,
        unchanged_entry_count=unchanged_count,
        operation_type_counts=op_counts,
        target_artifact_type_counts=tt_counts,
        perturbation_matrix=perturbation_matrix,
        energy_matrix_receipt_hash=sha256_hex(hash_payload),
    )


def validate_perturbation_matrix_entry(entry: PerturbationMatrixEntry) -> bool:
    if not isinstance(entry, PerturbationMatrixEntry):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_index(entry.row_index)
    _validate_index(entry.column_index)
    _validate_label(entry.entry_label, _ERR_INVALID_ENTRY_LABEL)
    _validate_sha256_hex(entry.perturbation_contract_hash)
    _validate_sha256_hex(entry.perturbation_result_hash)
    _validate_sha256_hex(entry.target_artifact_hash)
    _validate_target_artifact_type(entry.target_artifact_type)
    if not isinstance(entry.operation_type, str) or entry.operation_type not in get_allowed_perturbation_operation_types():
        raise ValueError(_ERR_ENTRY_OPERATION_MISMATCH)
    if not isinstance(entry.changed, bool):
        raise ValueError(_ERR_ENTRY_CHANGED_MISMATCH)
    _validate_int_score(entry.integer_impact_score)
    if entry.integer_impact_score != _derive_integer_impact_score(entry.changed):
        raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    _validate_sha256_hex(entry.perturbation_matrix_entry_hash)
    expected = sha256_hex(entry._hash_payload())
    if expected != entry.perturbation_matrix_entry_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_perturbation_matrix(matrix: PerturbationMatrix) -> bool:
    if not isinstance(matrix, PerturbationMatrix):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_label(matrix.matrix_label, _ERR_INVALID_MATRIX_LABEL)
    if matrix.matrix_mode != _MATRIX_MODE:
        raise ValueError(_ERR_INVALID_MATRIX_MODE)
    if not isinstance(matrix.entries, tuple) or not matrix.entries:
        raise ValueError(_ERR_INVALID_INPUT)
    if len(matrix.entries) > _MAX_MATRIX_ENTRIES:
        raise ValueError(_ERR_MATRIX_COUNT_MISMATCH)
    _validate_non_bool_int(matrix.row_count, minimum=1, maximum=_MAX_MATRIX_ROWS, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(matrix.column_count, minimum=1, maximum=_MAX_MATRIX_COLUMNS, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(matrix.entry_count, minimum=1, maximum=_MAX_MATRIX_ENTRIES, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(matrix.changed_entry_count, minimum=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(matrix.unchanged_entry_count, minimum=0, err=_ERR_INVALID_INPUT)
    for entry in matrix.entries:
        if not isinstance(entry, PerturbationMatrixEntry):
            raise ValueError(_ERR_INVALID_INPUT)
        validate_perturbation_matrix_entry(entry)
    if tuple(sorted(matrix.entries, key=_matrix_entry_order_key)) != matrix.entries:
        raise ValueError(_ERR_MATRIX_ORDER_MISMATCH)
    seen_coords: set[tuple[int, int]] = set()
    seen_labels: set[str] = set()
    for entry in matrix.entries:
        coord = (entry.row_index, entry.column_index)
        if coord in seen_coords or entry.entry_label in seen_labels:
            raise ValueError(_ERR_DUPLICATE_MATRIX_ENTRY)
        seen_coords.add(coord)
        seen_labels.add(entry.entry_label)
    row_count = max(e.row_index for e in matrix.entries) + 1
    column_count = max(e.column_index for e in matrix.entries) + 1
    if row_count > _MAX_MATRIX_ROWS or column_count > _MAX_MATRIX_COLUMNS:
        raise ValueError(_ERR_MATRIX_DIMENSION_MISMATCH)
    if matrix.row_count != row_count or matrix.column_count != column_count:
        raise ValueError(_ERR_MATRIX_DIMENSION_MISMATCH)
    if matrix.entry_count != len(matrix.entries):
        raise ValueError(_ERR_MATRIX_COUNT_MISMATCH)
    changed = sum(1 for e in matrix.entries if e.changed)
    unchanged = len(matrix.entries) - changed
    if matrix.changed_entry_count != changed or matrix.unchanged_entry_count != unchanged:
        raise ValueError(_ERR_MATRIX_COUNT_MISMATCH)
    _validate_sha256_hex(matrix.perturbation_matrix_hash)
    if sha256_hex(matrix._hash_payload()) != matrix.perturbation_matrix_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_energy_matrix_receipt(receipt: EnergyMatrixReceipt) -> bool:
    if not isinstance(receipt, EnergyMatrixReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    validate_perturbation_matrix(receipt.perturbation_matrix)
    _validate_sha256_hex(receipt.perturbation_matrix_hash)
    if receipt.perturbation_matrix_hash != receipt.perturbation_matrix.perturbation_matrix_hash:
        raise ValueError(_ERR_ENERGY_RECEIPT_MISMATCH)
    _validate_label(receipt.matrix_label, _ERR_INVALID_MATRIX_LABEL)
    if receipt.matrix_mode != _MATRIX_MODE:
        raise ValueError(_ERR_INVALID_MATRIX_MODE)
    if receipt.matrix_label != receipt.perturbation_matrix.matrix_label or receipt.matrix_mode != receipt.perturbation_matrix.matrix_mode:
        raise ValueError(_ERR_ENERGY_RECEIPT_MISMATCH)
    _validate_non_bool_int(receipt.total_integer_impact_score, minimum=-_MAX_ABS_TOTAL_IMPACT_SCORE, maximum=_MAX_ABS_TOTAL_IMPACT_SCORE, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(receipt.changed_entry_count, minimum=0, err=_ERR_INVALID_INPUT)
    _validate_non_bool_int(receipt.unchanged_entry_count, minimum=0, err=_ERR_INVALID_INPUT)
    # O(1) length caps before iterating to prevent DoS from huge count lists
    max_operation_types = len(get_allowed_perturbation_operation_types())
    max_target_types = receipt.perturbation_matrix.entry_count
    normalized_operation_counts = _validate_operation_type_counts(receipt.operation_type_counts, max_length=max_operation_types)
    normalized_target_counts = _validate_target_artifact_type_counts(receipt.target_artifact_type_counts, max_length=max_target_types)
    total = sum(e.integer_impact_score for e in receipt.perturbation_matrix.entries)
    if receipt.total_integer_impact_score != total:
        raise ValueError(_ERR_IMPACT_SCORE_MISMATCH)
    changed = sum(1 for e in receipt.perturbation_matrix.entries if e.changed)
    unchanged = len(receipt.perturbation_matrix.entries) - changed
    if receipt.changed_entry_count != changed or receipt.unchanged_entry_count != unchanged:
        raise ValueError(_ERR_ENERGY_RECEIPT_MISMATCH)
    op_counts = _canonicalize_counts(Counter(e.operation_type for e in receipt.perturbation_matrix.entries))
    tt_counts = _canonicalize_counts(Counter(e.target_artifact_type for e in receipt.perturbation_matrix.entries))
    if normalized_operation_counts != op_counts or normalized_target_counts != tt_counts:
        raise ValueError(_ERR_ENERGY_RECEIPT_MISMATCH)
    _validate_sha256_hex(receipt.energy_matrix_receipt_hash)
    if sha256_hex(receipt._hash_payload()) != receipt.energy_matrix_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_perturbation_matrix_entry_with_contract_result(entry: PerturbationMatrixEntry, perturbation_contract: PerturbationContract, perturbation_result: PerturbationResult) -> bool:
    validate_perturbation_matrix_entry(entry)
    validate_perturbation_contract(perturbation_contract)
    validate_perturbation_result(perturbation_result)
    rebuilt = build_perturbation_matrix_entry(entry.row_index, entry.column_index, entry.entry_label, perturbation_contract, perturbation_result)
    if rebuilt.to_dict() != entry.to_dict():
        raise ValueError(_ERR_ENTRY_CONTRACT_RESULT_MISMATCH)
    return True
