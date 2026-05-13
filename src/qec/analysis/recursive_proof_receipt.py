from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .canonical_hashing import canonical_bytes, canonical_json, sha256_hex
from .loop_termination_contract import (
    LoopTerminationContract,
    get_allowed_loop_modes,
    get_allowed_loop_termination_policies,
    validate_loop_termination_contract,
)

_ERR_INVALID_INPUT = "INVALID_INPUT"
_ERR_INVALID_HASH_FORMAT = "INVALID_HASH_FORMAT"
_ERR_HASH_MISMATCH = "HASH_MISMATCH"
_ERR_INVALID_ITERATION_STATUS = "INVALID_ITERATION_STATUS"
_ERR_ITERATION_INDEX_OUT_OF_BOUNDS = "ITERATION_INDEX_OUT_OF_BOUNDS"
_ERR_DUPLICATE_ITERATION = "DUPLICATE_ITERATION"
_ERR_ITERATION_ORDER_MISMATCH = "ITERATION_ORDER_MISMATCH"
_ERR_ITERATION_COUNT_MISMATCH = "ITERATION_COUNT_MISMATCH"
_ERR_MAX_DEPTH_EXCEEDED = "MAX_DEPTH_EXCEEDED"
_ERR_LOOP_CONTRACT_MISMATCH = "LOOP_CONTRACT_MISMATCH"
_ERR_RECEIPT_HASH_FIELD_MISMATCH = "RECEIPT_HASH_FIELD_MISMATCH"
_ERR_CHANGED_FLAG_MISMATCH = "CHANGED_FLAG_MISMATCH"
_ERR_TERMINAL_ITERATION_MISMATCH = "TERMINAL_ITERATION_MISMATCH"
_ERR_FINAL_OUTPUT_HASH_MISMATCH = "FINAL_OUTPUT_HASH_MISMATCH"
_ERR_RECURSIVE_PROOF_RECEIPT_MISMATCH = "RECURSIVE_PROOF_RECEIPT_MISMATCH"
_ERR_HASH_CHAIN_CONTINUITY_BROKEN = "HASH_CHAIN_CONTINUITY_BROKEN"

_MAX_ITERATION_RECORDS = 10_000
_MAX_ITERATION_INDEX = 9_999
_MAX_LOOP_LABEL_LENGTH = 96
_MAX_RECEIPT_HASH_FIELD_LENGTH = 96

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ALLOWED_LOOP_ITERATION_STATUSES = frozenset({
    "ITERATION_CONTINUED",
    "ITERATION_FIXED_POINT",
    "ITERATION_TARGET_REACHED",
    "ITERATION_STATUS_TERMINAL",
    "ITERATION_DIVERGENCE_LIMIT",
    "ITERATION_MAX_DEPTH_REACHED",
})
_TERMINAL_ITERATION_STATUSES = frozenset({
    "ITERATION_FIXED_POINT",
    "ITERATION_TARGET_REACHED",
    "ITERATION_STATUS_TERMINAL",
    "ITERATION_DIVERGENCE_LIMIT",
    "ITERATION_MAX_DEPTH_REACHED",
})


def get_allowed_loop_iteration_statuses() -> frozenset[str]:
    return _ALLOWED_LOOP_ITERATION_STATUSES


def _validate_sha(v: object) -> str:
    if not isinstance(v, str) or _SHA256_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_HASH_FORMAT)
    return v


def _validate_loop_label(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_LOOP_LABEL_LENGTH or _LABEL_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _validate_receipt_hash_field(v: object) -> str:
    if not isinstance(v, str) or not v or len(v) > _MAX_RECEIPT_HASH_FIELD_LENGTH or _FIELD_NAME_RE.fullmatch(v) is None:
        raise ValueError(_ERR_INVALID_INPUT)
    return v


def _validate_iteration_index(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or not (0 <= v <= _MAX_ITERATION_INDEX):
        raise ValueError(_ERR_ITERATION_INDEX_OUT_OF_BOUNDS)
    return v


def _validate_iteration_status(v: object) -> str:
    if not isinstance(v, str) or v not in _ALLOWED_LOOP_ITERATION_STATUSES:
        raise ValueError(_ERR_INVALID_ITERATION_STATUS)
    return v


def _validate_count(v: object) -> int:
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    return v


def _loop_iteration_record_payload(
    loop_termination_contract_hash: str,
    loop_label: str,
    iteration_index: int,
    input_receipt_hash_field: str,
    input_receipt_hash: str,
    output_receipt_hash_field: str,
    output_receipt_hash: str,
    iteration_status: str,
    changed: bool,
) -> dict[str, Any]:
    return {
        "loop_termination_contract_hash": loop_termination_contract_hash,
        "loop_label": loop_label,
        "iteration_index": iteration_index,
        "input_receipt_hash_field": input_receipt_hash_field,
        "input_receipt_hash": input_receipt_hash,
        "output_receipt_hash_field": output_receipt_hash_field,
        "output_receipt_hash": output_receipt_hash,
        "iteration_status": iteration_status,
        "changed": changed,
    }


@dataclass(frozen=True)
class LoopIterationRecord:
    loop_termination_contract_hash: str
    loop_label: str
    iteration_index: int
    input_receipt_hash_field: str
    input_receipt_hash: str
    output_receipt_hash_field: str
    output_receipt_hash: str
    iteration_status: str
    changed: bool
    loop_iteration_record_hash: str

    def __post_init__(self) -> None:
        validate_loop_iteration_record(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _loop_iteration_record_payload(
            self.loop_termination_contract_hash,
            self.loop_label,
            self.iteration_index,
            self.input_receipt_hash_field,
            self.input_receipt_hash,
            self.output_receipt_hash_field,
            self.output_receipt_hash,
            self.iteration_status,
            self.changed,
        )
        return {**payload, "loop_iteration_record_hash": self.loop_iteration_record_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def build_loop_iteration_record(loop_termination_contract: LoopTerminationContract, iteration_index: int, input_receipt_hash: str, output_receipt_hash: str, iteration_status: str) -> LoopIterationRecord:
    validate_loop_termination_contract(loop_termination_contract)
    idx = _validate_iteration_index(iteration_index)
    if idx >= loop_termination_contract.max_depth:
        raise ValueError(_ERR_ITERATION_INDEX_OUT_OF_BOUNDS)
    _validate_sha(input_receipt_hash)
    _validate_sha(output_receipt_hash)
    status = _validate_iteration_status(iteration_status)
    changed = input_receipt_hash != output_receipt_hash
    payload = _loop_iteration_record_payload(
        loop_termination_contract.loop_termination_contract_hash,
        loop_termination_contract.loop_label,
        idx,
        loop_termination_contract.input_receipt_hash_field,
        input_receipt_hash,
        loop_termination_contract.output_receipt_hash_field,
        output_receipt_hash,
        status,
        changed,
    )
    return LoopIterationRecord(**payload, loop_iteration_record_hash=sha256_hex(payload))


def validate_loop_iteration_record(record: LoopIterationRecord) -> bool:
    if not isinstance(record, LoopIterationRecord):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(record.loop_termination_contract_hash)
    _validate_loop_label(record.loop_label)
    _validate_iteration_index(record.iteration_index)
    _validate_receipt_hash_field(record.input_receipt_hash_field)
    _validate_sha(record.input_receipt_hash)
    _validate_receipt_hash_field(record.output_receipt_hash_field)
    _validate_sha(record.output_receipt_hash)
    if record.input_receipt_hash_field == record.output_receipt_hash_field:
        raise ValueError(_ERR_RECEIPT_HASH_FIELD_MISMATCH)
    _validate_iteration_status(record.iteration_status)
    if not isinstance(record.changed, bool):
        raise ValueError(_ERR_CHANGED_FLAG_MISMATCH)
    if record.changed != (record.input_receipt_hash != record.output_receipt_hash):
        raise ValueError(_ERR_CHANGED_FLAG_MISMATCH)
    _validate_sha(record.loop_iteration_record_hash)
    payload = _loop_iteration_record_payload(
        record.loop_termination_contract_hash,
        record.loop_label,
        record.iteration_index,
        record.input_receipt_hash_field,
        record.input_receipt_hash,
        record.output_receipt_hash_field,
        record.output_receipt_hash,
        record.iteration_status,
        record.changed,
    )
    if sha256_hex(payload) != record.loop_iteration_record_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


@dataclass(frozen=True)
class RecursiveProofReceipt:
    loop_termination_contract_hash: str
    source_artifact_type: str
    source_artifact_hash: str
    loop_label: str
    loop_mode: str
    max_depth: int
    input_receipt_hash_field: str
    output_receipt_hash_field: str
    iteration_records: tuple[LoopIterationRecord, ...]
    iteration_count: int
    terminal_iteration_index: int
    final_output_receipt_hash: str
    recursive_proof_receipt_hash: str

    def __post_init__(self) -> None:
        validate_recursive_proof_receipt(self)

    def to_dict(self) -> dict[str, Any]:
        payload = _recursive_proof_receipt_payload(
            self.loop_termination_contract_hash,
            self.source_artifact_type,
            self.source_artifact_hash,
            self.loop_label,
            self.loop_mode,
            self.max_depth,
            self.input_receipt_hash_field,
            self.output_receipt_hash_field,
            self.iteration_records,
            self.iteration_count,
            self.terminal_iteration_index,
            self.final_output_receipt_hash,
        )
        return {**payload, "recursive_proof_receipt_hash": self.recursive_proof_receipt_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _recursive_proof_receipt_payload(
    loop_termination_contract_hash: str,
    source_artifact_type: str,
    source_artifact_hash: str,
    loop_label: str,
    loop_mode: str,
    max_depth: int,
    input_receipt_hash_field: str,
    output_receipt_hash_field: str,
    iteration_records: tuple[LoopIterationRecord, ...],
    iteration_count: int,
    terminal_iteration_index: int,
    final_output_receipt_hash: str,
) -> dict[str, Any]:
    return {
        "loop_termination_contract_hash": loop_termination_contract_hash,
        "source_artifact_type": source_artifact_type,
        "source_artifact_hash": source_artifact_hash,
        "loop_label": loop_label,
        "loop_mode": loop_mode,
        "max_depth": max_depth,
        "input_receipt_hash_field": input_receipt_hash_field,
        "output_receipt_hash_field": output_receipt_hash_field,
        "iteration_records": [r.to_dict() for r in iteration_records],
        "iteration_count": iteration_count,
        "terminal_iteration_index": terminal_iteration_index,
        "final_output_receipt_hash": final_output_receipt_hash,
    }


def _validate_terminal_rules(records: tuple[LoopIterationRecord, ...]) -> None:
    terminal_positions = [i for i, r in enumerate(records) if r.iteration_status in _TERMINAL_ITERATION_STATUSES]
    if len(terminal_positions) != 1 or terminal_positions[0] != len(records) - 1:
        raise ValueError(_ERR_TERMINAL_ITERATION_MISMATCH)


def build_recursive_proof_receipt(loop_termination_contract: LoopTerminationContract, iteration_records: list[LoopIterationRecord] | tuple[LoopIterationRecord, ...]) -> RecursiveProofReceipt:
    validate_loop_termination_contract(loop_termination_contract)
    if not isinstance(iteration_records, (list, tuple)):
        raise ValueError(_ERR_INVALID_INPUT)
    if not (1 <= len(iteration_records) <= _MAX_ITERATION_RECORDS):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    seen: set[int] = set()
    validated: list[LoopIterationRecord] = []
    for rec in iteration_records:
        validate_loop_iteration_record(rec)
        if rec.loop_termination_contract_hash != loop_termination_contract.loop_termination_contract_hash:
            raise ValueError(_ERR_LOOP_CONTRACT_MISMATCH)
        if rec.iteration_index >= loop_termination_contract.max_depth:
            raise ValueError(_ERR_ITERATION_INDEX_OUT_OF_BOUNDS)
        if rec.iteration_index in seen:
            raise ValueError(_ERR_DUPLICATE_ITERATION)
        seen.add(rec.iteration_index)
        validated.append(rec)
    ordered = tuple(sorted(validated, key=lambda r: r.iteration_index))
    for i, rec in enumerate(ordered):
        if rec.iteration_index != i:
            raise ValueError(_ERR_ITERATION_ORDER_MISMATCH)
    # P1: Enforce hash-chain continuity
    for i in range(1, len(ordered)):
        if ordered[i].input_receipt_hash != ordered[i - 1].output_receipt_hash:
            raise ValueError(_ERR_HASH_CHAIN_CONTINUITY_BROKEN)
    _validate_terminal_rules(ordered)
    count = len(ordered)
    if count > loop_termination_contract.max_depth:
        raise ValueError(_ERR_MAX_DEPTH_EXCEEDED)
    terminal = ordered[-1].iteration_index
    final_output = ordered[-1].output_receipt_hash
    payload = _recursive_proof_receipt_payload(
        loop_termination_contract.loop_termination_contract_hash,
        loop_termination_contract.source_artifact_type,
        loop_termination_contract.source_artifact_hash,
        loop_termination_contract.loop_label,
        loop_termination_contract.loop_mode,
        loop_termination_contract.max_depth,
        loop_termination_contract.input_receipt_hash_field,
        loop_termination_contract.output_receipt_hash_field,
        ordered,
        count,
        terminal,
        final_output,
    )
    return RecursiveProofReceipt(
        loop_termination_contract_hash=loop_termination_contract.loop_termination_contract_hash,
        source_artifact_type=loop_termination_contract.source_artifact_type,
        source_artifact_hash=loop_termination_contract.source_artifact_hash,
        loop_label=loop_termination_contract.loop_label,
        loop_mode=loop_termination_contract.loop_mode,
        max_depth=loop_termination_contract.max_depth,
        input_receipt_hash_field=loop_termination_contract.input_receipt_hash_field,
        output_receipt_hash_field=loop_termination_contract.output_receipt_hash_field,
        iteration_records=ordered,
        iteration_count=count,
        terminal_iteration_index=terminal,
        final_output_receipt_hash=final_output,
        recursive_proof_receipt_hash=sha256_hex(payload),
    )


def validate_recursive_proof_receipt(receipt: RecursiveProofReceipt) -> bool:
    if not isinstance(receipt, RecursiveProofReceipt):
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(receipt.loop_termination_contract_hash)
    if not isinstance(receipt.source_artifact_type, str) or not receipt.source_artifact_type:
        raise ValueError(_ERR_INVALID_INPUT)
    _validate_sha(receipt.source_artifact_hash)
    _validate_loop_label(receipt.loop_label)
    if receipt.loop_mode not in get_allowed_loop_modes():
        raise ValueError(_ERR_INVALID_INPUT)
    if not isinstance(receipt.max_depth, int) or isinstance(receipt.max_depth, bool) or receipt.max_depth < 1:
        raise ValueError(_ERR_MAX_DEPTH_EXCEEDED)
    _validate_receipt_hash_field(receipt.input_receipt_hash_field)
    _validate_receipt_hash_field(receipt.output_receipt_hash_field)
    if receipt.input_receipt_hash_field == receipt.output_receipt_hash_field:
        raise ValueError(_ERR_RECEIPT_HASH_FIELD_MISMATCH)
    if not isinstance(receipt.iteration_records, tuple):
        raise ValueError(_ERR_INVALID_INPUT)
    if not (1 <= len(receipt.iteration_records) <= _MAX_ITERATION_RECORDS):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    for rec in receipt.iteration_records:
        validate_loop_iteration_record(rec)
        # P2: Verify per-record contract hash matches receipt header
        if rec.loop_termination_contract_hash != receipt.loop_termination_contract_hash:
            raise ValueError(_ERR_LOOP_CONTRACT_MISMATCH)
    indices = [r.iteration_index for r in receipt.iteration_records]
    if len(set(indices)) != len(indices):
        raise ValueError(_ERR_DUPLICATE_ITERATION)
    if indices != sorted(indices):
        raise ValueError(_ERR_ITERATION_ORDER_MISMATCH)
    for i, idx in enumerate(indices):
        if idx != i:
            raise ValueError(_ERR_ITERATION_ORDER_MISMATCH)
    # P1: Enforce hash-chain continuity
    for i in range(1, len(receipt.iteration_records)):
        if receipt.iteration_records[i].input_receipt_hash != receipt.iteration_records[i - 1].output_receipt_hash:
            raise ValueError(_ERR_HASH_CHAIN_CONTINUITY_BROKEN)
    _validate_terminal_rules(receipt.iteration_records)
    count = _validate_count(receipt.iteration_count)
    if count != len(receipt.iteration_records):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    if count > receipt.max_depth:
        raise ValueError(_ERR_MAX_DEPTH_EXCEEDED)
    if not isinstance(receipt.terminal_iteration_index, int) or isinstance(receipt.terminal_iteration_index, bool):
        raise ValueError(_ERR_ITERATION_COUNT_MISMATCH)
    if receipt.terminal_iteration_index != receipt.iteration_records[-1].iteration_index:
        raise ValueError(_ERR_TERMINAL_ITERATION_MISMATCH)
    _validate_sha(receipt.final_output_receipt_hash)
    if receipt.final_output_receipt_hash != receipt.iteration_records[-1].output_receipt_hash:
        raise ValueError(_ERR_FINAL_OUTPUT_HASH_MISMATCH)
    _validate_sha(receipt.recursive_proof_receipt_hash)
    payload = _recursive_proof_receipt_payload(
        receipt.loop_termination_contract_hash,
        receipt.source_artifact_type,
        receipt.source_artifact_hash,
        receipt.loop_label,
        receipt.loop_mode,
        receipt.max_depth,
        receipt.input_receipt_hash_field,
        receipt.output_receipt_hash_field,
        receipt.iteration_records,
        receipt.iteration_count,
        receipt.terminal_iteration_index,
        receipt.final_output_receipt_hash,
    )
    if sha256_hex(payload) != receipt.recursive_proof_receipt_hash:
        raise ValueError(_ERR_HASH_MISMATCH)
    return True


def validate_loop_iteration_record_with_contract(record: LoopIterationRecord, loop_termination_contract: LoopTerminationContract) -> bool:
    validate_loop_termination_contract(loop_termination_contract)
    validate_loop_iteration_record(record)
    if record.iteration_index >= loop_termination_contract.max_depth:
        raise ValueError(_ERR_ITERATION_INDEX_OUT_OF_BOUNDS)
    expected = build_loop_iteration_record(
        loop_termination_contract,
        record.iteration_index,
        record.input_receipt_hash,
        record.output_receipt_hash,
        record.iteration_status,
    )
    if expected.to_dict() != record.to_dict():
        raise ValueError(_ERR_LOOP_CONTRACT_MISMATCH)
    return True


def validate_recursive_proof_receipt_with_contract(receipt: RecursiveProofReceipt, loop_termination_contract: LoopTerminationContract) -> bool:
    validate_loop_termination_contract(loop_termination_contract)
    validate_recursive_proof_receipt(receipt)
    if receipt.loop_mode not in get_allowed_loop_modes() or loop_termination_contract.termination_policy not in get_allowed_loop_termination_policies():
        raise ValueError(_ERR_INVALID_INPUT)
    expected = build_recursive_proof_receipt(loop_termination_contract, list(receipt.iteration_records))
    if expected.to_dict() != receipt.to_dict():
        raise ValueError(_ERR_RECURSIVE_PROOF_RECEIPT_MISMATCH)
    return True
