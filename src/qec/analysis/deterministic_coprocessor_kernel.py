"""v137.11.0 — Deterministic Co-Processor Kernel.

Layer-4 deterministic contract substrate:
cpu -> descriptor -> co-processor -> receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1
_SUPPORTED_OPERATIONS = (
    "integer_matrix",
    "bitfield_transform",
    "lookup_table",
    "fixed_point_mac",
    "identity_pass",
)
# Explicit, versioned weight mapping — stable across reorderings of _SUPPORTED_OPERATIONS.
_OP_CYCLE_WEIGHTS: dict[str, int] = {
    "integer_matrix": 1,
    "bitfield_transform": 2,
    "lookup_table": 3,
    "fixed_point_mac": 4,
    "identity_pass": 5,
}
_SUPPORTED_LANES = ("cpu_sidecar", "fixed_function_0", "fixed_function_1")
_LOOKUP_TABLE: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if callable(value):
        raise ValueError("callable leakage in payload is not allowed")
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise ValueError("payload keys must be strings")
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


@dataclass(frozen=True)
class CoProcessorDescriptor:
    descriptor_id: str
    operation_type: str
    input_hash: str
    payload: _JSONValue
    lane_id: str
    epoch_id: str
    scratchpad_size: int
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor_id": self.descriptor_id,
            "operation_type": self.operation_type,
            "input_hash": self.input_hash,
            "payload": self.payload,
            "lane_id": self.lane_id,
            "epoch_id": self.epoch_id,
            "scratchpad_size": self.scratchpad_size,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CoProcessorResult:
    descriptor_id: str
    output_payload: _JSONValue
    output_hash: str
    cycle_count: int
    epoch_id: str
    lane_id: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor_id": self.descriptor_id,
            "output_payload": self.output_payload,
            "output_hash": self.output_hash,
            "cycle_count": self.cycle_count,
            "epoch_id": self.epoch_id,
            "lane_id": self.lane_id,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CoProcessorReceipt:
    receipt_hash: str
    descriptor_id: str
    output_hash: str
    epoch_id: str
    lane_id: str
    cycle_count: int
    validation_passed: bool
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "descriptor_id": self.descriptor_id,
            "output_hash": self.output_hash,
            "epoch_id": self.epoch_id,
            "lane_id": self.lane_id,
            "cycle_count": self.cycle_count,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CoProcessorKernelReport:
    descriptor: CoProcessorDescriptor
    result: CoProcessorResult
    receipt: CoProcessorReceipt
    stable_hash: str
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "result": self.result.to_dict(),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "result": self.result.to_dict(),
            "receipt": self.receipt.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_coprocessor_descriptor(raw_input: Mapping[str, Any] | CoProcessorDescriptor) -> CoProcessorDescriptor:
    if isinstance(raw_input, CoProcessorDescriptor):
        return raw_input
    data = _require_mapping(raw_input, "raw_input")

    if "descriptors" in data:
        batch = data["descriptors"]
        if not isinstance(batch, (tuple, list)):
            raise ValueError("descriptors must be list or tuple when present")
        ids: list[str] = []
        for item in batch:
            entry = _require_mapping(item, "descriptors[]")
            if "descriptor_id" not in entry:
                raise ValueError("descriptor_id missing in batch item")
            ids.append(str(entry["descriptor_id"]))
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate descriptor IDs are not allowed")
        raise ValueError("batch descriptor mode is unsupported in this release")

    payload = _canonicalize_json(data.get("payload", {}))
    descriptor = CoProcessorDescriptor(
        descriptor_id=str(data.get("descriptor_id", "")).strip(),
        operation_type=str(data.get("operation_type", "")),
        input_hash=str(data.get("input_hash", "")),
        payload=payload,
        lane_id=str(data.get("lane_id", "")),
        epoch_id=str(data.get("epoch_id", "")).strip(),
        scratchpad_size=_require_int(data.get("scratchpad_size", 0), "scratchpad_size"),
        schema_version=_require_int(data.get("schema_version", _SCHEMA_VERSION), "schema_version"),
    )
    if descriptor.input_hash == "":
        descriptor_hash = _sha256_hex(descriptor.payload)
        descriptor = CoProcessorDescriptor(
            descriptor_id=descriptor.descriptor_id,
            operation_type=descriptor.operation_type,
            input_hash=descriptor_hash,
            payload=descriptor.payload,
            lane_id=descriptor.lane_id,
            epoch_id=descriptor.epoch_id,
            scratchpad_size=descriptor.scratchpad_size,
            schema_version=descriptor.schema_version,
        )
    return descriptor


def validate_coprocessor_descriptor(descriptor: CoProcessorDescriptor) -> CoProcessorDescriptor:
    if descriptor.descriptor_id == "":
        raise ValueError("descriptor_id must be non-empty")
    if descriptor.operation_type not in _SUPPORTED_OPERATIONS:
        raise ValueError("unsupported operation_type")
    if descriptor.lane_id not in _SUPPORTED_LANES:
        raise ValueError("unsupported lane_id")
    if descriptor.epoch_id == "":
        raise ValueError("epoch_id must be non-empty")
    if descriptor.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if descriptor.scratchpad_size < 0:
        raise ValueError("scratchpad_size must be non-negative")
    if not isinstance(descriptor.payload, (dict, tuple, str, int, float, bool)) and descriptor.payload is not None:
        raise ValueError("malformed payload")
    if descriptor.input_hash != _sha256_hex(descriptor.payload):
        raise ValueError("input_hash must match canonical payload hash")
    return descriptor


def _op_identity_pass(payload: _JSONValue) -> _JSONValue:
    return payload


def _op_lookup_table(payload: _JSONValue) -> _JSONValue:
    data = _require_mapping(payload, "payload")
    symbols = data.get("symbols")
    if not isinstance(symbols, tuple):
        raise ValueError("lookup_table payload.symbols must be a tuple")
    mapped = tuple(_LOOKUP_TABLE.get(str(symbol), -1) for symbol in symbols)
    return {"symbols": symbols, "values": mapped}


def _op_bitfield_transform(payload: _JSONValue) -> _JSONValue:
    data = _require_mapping(payload, "payload")
    value = _require_int(data.get("value"), "payload.value")
    mask = _require_int(data.get("mask", 0xFFFFFFFF), "payload.mask")
    xor = _require_int(data.get("xor", 0), "payload.xor")
    shift = _require_int(data.get("shift", 0), "payload.shift")
    if shift < 0 or shift > 31:
        raise ValueError("payload.shift must be between 0 and 31")
    transformed = (((value & mask) ^ xor) << shift) & 0xFFFFFFFF
    return {"value": value, "mask": mask, "xor": xor, "shift": shift, "transformed": transformed}


def _op_integer_matrix(payload: _JSONValue) -> _JSONValue:
    data = _require_mapping(payload, "payload")
    matrix_a = data.get("matrix_a")
    matrix_b = data.get("matrix_b")
    if not isinstance(matrix_a, tuple) or not isinstance(matrix_b, tuple):
        raise ValueError("integer_matrix payload must include tuple matrices")
    if len(matrix_a) == 0 or len(matrix_b) == 0:
        raise ValueError("integer_matrix matrices must be non-empty")
    for i, row in enumerate(matrix_a):
        if not isinstance(row, tuple):
            raise ValueError(f"matrix_a row {i} must be a tuple, got {type(row).__name__}")
    for i, row in enumerate(matrix_b):
        if not isinstance(row, tuple):
            raise ValueError(f"matrix_b row {i} must be a tuple, got {type(row).__name__}")
    a_rows = tuple(tuple(_require_int(v, "matrix_a value") for v in row) for row in matrix_a)
    b_rows = tuple(tuple(_require_int(v, "matrix_b value") for v in row) for row in matrix_b)
    if any(len(row) == 0 for row in a_rows):
        raise ValueError("matrix_a rows must be non-empty")
    if any(len(row) == 0 for row in b_rows):
        raise ValueError("matrix_b rows must be non-empty")
    width = len(a_rows[0])
    b_width = len(b_rows[0])
    if any(len(row) != width for row in a_rows):
        raise ValueError("matrix_a rows must have equal width")
    if any(len(row) != b_width for row in b_rows):
        raise ValueError("matrix_b rows must have equal width")
    if width != len(b_rows):
        raise ValueError("matrix dimensions are incompatible")
    result: list[tuple[int, ...]] = []
    for row in a_rows:
        out_row: list[int] = []
        for col_idx in range(b_width):
            total = 0
            for k in range(width):
                total += row[k] * b_rows[k][col_idx]
            out_row.append(total)
        result.append(tuple(out_row))
    return {"matrix": tuple(result)}


def _op_fixed_point_mac(payload: _JSONValue) -> _JSONValue:
    data = _require_mapping(payload, "payload")
    vector_a = data.get("vector_a")
    vector_b = data.get("vector_b")
    if not isinstance(vector_a, tuple) or not isinstance(vector_b, tuple):
        raise ValueError("fixed_point_mac vectors must be sequences")
    if len(vector_a) != len(vector_b):
        raise ValueError("fixed_point_mac vectors must be equal length")
    accumulator = _require_int(data.get("accumulator", 0), "payload.accumulator")
    scale = _require_int(data.get("scale", 1), "payload.scale")
    if scale <= 0:
        raise ValueError("payload.scale must be > 0")
    products = tuple(_require_int(a, "vector_a value") * _require_int(b, "vector_b value") for a, b in zip(vector_a, vector_b))
    mac = accumulator + sum(products)
    fixed_point = mac // scale
    return {
        "products": products,
        "accumulator": accumulator,
        "mac": mac,
        "scale": scale,
        "fixed_point": fixed_point,
    }


def _run_operation(operation_type: str, payload: _JSONValue) -> _JSONValue:
    if operation_type == "identity_pass":
        return _op_identity_pass(payload)
    if operation_type == "lookup_table":
        return _op_lookup_table(payload)
    if operation_type == "bitfield_transform":
        return _op_bitfield_transform(payload)
    if operation_type == "integer_matrix":
        return _op_integer_matrix(payload)
    if operation_type == "fixed_point_mac":
        return _op_fixed_point_mac(payload)
    raise ValueError("unsupported operation_type")


def _deterministic_cycle_count(descriptor: CoProcessorDescriptor, output_payload: _JSONValue) -> int:
    payload_bytes = len(_canonical_bytes(descriptor.payload))
    output_bytes = len(_canonical_bytes(output_payload))
    op_weight = _OP_CYCLE_WEIGHTS[descriptor.operation_type]
    return payload_bytes + output_bytes + (op_weight * 17)


def run_coprocessor_kernel(descriptor: CoProcessorDescriptor) -> CoProcessorResult:
    descriptor = validate_coprocessor_descriptor(descriptor)
    output_payload = _run_operation(descriptor.operation_type, descriptor.payload)
    output_hash = _sha256_hex(output_payload)
    return CoProcessorResult(
        descriptor_id=descriptor.descriptor_id,
        output_payload=output_payload,
        output_hash=output_hash,
        cycle_count=_deterministic_cycle_count(descriptor, output_payload),
        epoch_id=descriptor.epoch_id,
        lane_id=descriptor.lane_id,
    )


def build_coprocessor_receipt(result: CoProcessorResult) -> CoProcessorReceipt:
    receipt = CoProcessorReceipt(
        receipt_hash="",
        descriptor_id=result.descriptor_id,
        output_hash=result.output_hash,
        epoch_id=result.epoch_id,
        lane_id=result.lane_id,
        cycle_count=result.cycle_count,
        validation_passed=True,
        schema_version=_SCHEMA_VERSION,
    )
    return CoProcessorReceipt(
        receipt_hash=receipt.stable_hash(),
        descriptor_id=receipt.descriptor_id,
        output_hash=receipt.output_hash,
        epoch_id=receipt.epoch_id,
        lane_id=receipt.lane_id,
        cycle_count=receipt.cycle_count,
        validation_passed=receipt.validation_passed,
        schema_version=receipt.schema_version,
    )


def stable_coprocessor_hash(report: CoProcessorKernelReport) -> str:
    return _sha256_hex(report.to_hash_payload_dict())


def compile_coprocessor_report(raw_input: Mapping[str, Any] | CoProcessorDescriptor) -> CoProcessorKernelReport:
    descriptor = validate_coprocessor_descriptor(normalize_coprocessor_descriptor(raw_input))
    result = run_coprocessor_kernel(descriptor)
    receipt = build_coprocessor_receipt(result)
    report = CoProcessorKernelReport(
        descriptor=descriptor,
        result=result,
        receipt=receipt,
        stable_hash="",
        schema_version=_SCHEMA_VERSION,
    )
    stable_hash = stable_coprocessor_hash(report)
    return CoProcessorKernelReport(
        descriptor=descriptor,
        result=result,
        receipt=receipt,
        stable_hash=stable_hash,
        schema_version=_SCHEMA_VERSION,
    )
