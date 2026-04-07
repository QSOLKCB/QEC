"""v137.11.1 — Integer / Matrix Offload Engine.

Deterministic Layer-4 integer offload substrate:
descriptor -> integer lane -> deterministic receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1
_SUPPORTED_OPERATION_TYPES = (
    "matmul",
    "matvec",
    "accumulate",
    "fixed_point_mac",
    "transpose",
    "identity_matrix_pass",
)
_SUPPORTED_LANES = ("integer_lane_0", "integer_lane_1", "fixed_point_lane")
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ValueError("floating point values are not allowed")
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


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _normalize_matrix(value: Any, field_name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if len(value) == 0:
        raise ValueError(f"{field_name} must be non-empty")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for row_idx, row in enumerate(value):
        if not isinstance(row, tuple):
            raise ValueError(f"{field_name} row {row_idx} must be a tuple")
        if len(row) == 0:
            raise ValueError(f"{field_name} rows must be non-empty")
        normalized_row = tuple(_require_int(cell, f"{field_name}[{row_idx}]") for cell in row)
        if width is None:
            width = len(normalized_row)
        elif len(normalized_row) != width:
            raise ValueError(f"{field_name} must be rectangular")
        rows.append(normalized_row)
    return tuple(rows)


def _apply_saturation(value: int, saturating: bool) -> tuple[int, bool]:
    if not saturating:
        return value, False
    if value < _INT32_MIN:
        return _INT32_MIN, True
    if value > _INT32_MAX:
        return _INT32_MAX, True
    return value, False


@dataclass(frozen=True)
class MatrixOffloadDescriptor:
    descriptor_id: str
    operation_type: str
    matrix_a: tuple[tuple[int, ...], ...]
    matrix_b: tuple[tuple[int, ...], ...]
    scale_factor: int
    fixed_point_shift: int
    saturating: bool
    epoch_id: str
    lane_id: str
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor_id": self.descriptor_id,
            "operation_type": self.operation_type,
            "matrix_a": self.matrix_a,
            "matrix_b": self.matrix_b,
            "scale_factor": self.scale_factor,
            "fixed_point_shift": self.fixed_point_shift,
            "saturating": self.saturating,
            "epoch_id": self.epoch_id,
            "lane_id": self.lane_id,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class MatrixOffloadResult:
    descriptor_id: str
    output_matrix: tuple[tuple[int, ...], ...]
    output_hash: str
    cycle_count: int
    overflow_detected: bool
    epoch_id: str
    lane_id: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor_id": self.descriptor_id,
            "output_matrix": self.output_matrix,
            "output_hash": self.output_hash,
            "cycle_count": self.cycle_count,
            "overflow_detected": self.overflow_detected,
            "epoch_id": self.epoch_id,
            "lane_id": self.lane_id,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class MatrixOffloadReceipt:
    receipt_hash: str
    descriptor_id: str
    output_hash: str
    cycle_count: int
    overflow_detected: bool
    epoch_id: str
    lane_id: str
    validation_passed: bool
    schema_version: int

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "descriptor_id": self.descriptor_id,
            "output_hash": self.output_hash,
            "cycle_count": self.cycle_count,
            "overflow_detected": self.overflow_detected,
            "epoch_id": self.epoch_id,
            "lane_id": self.lane_id,
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
class MatrixOffloadReport:
    descriptor: MatrixOffloadDescriptor
    result: MatrixOffloadResult
    receipt: MatrixOffloadReceipt
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


def normalize_matrix_offload_descriptor(
    raw_input: Mapping[str, Any] | MatrixOffloadDescriptor,
) -> MatrixOffloadDescriptor:
    if isinstance(raw_input, MatrixOffloadDescriptor):
        return raw_input
    data = _require_mapping(raw_input, "raw_input")

    descriptor = MatrixOffloadDescriptor(
        descriptor_id=str(data.get("descriptor_id", "")).strip(),
        operation_type=str(data.get("operation_type", "")).strip(),
        matrix_a=_normalize_matrix(_canonicalize_json(data.get("matrix_a", ())), "matrix_a"),
        matrix_b=_normalize_matrix(_canonicalize_json(data.get("matrix_b", ())), "matrix_b"),
        scale_factor=_require_int(data.get("scale_factor", 1), "scale_factor"),
        fixed_point_shift=_require_int(data.get("fixed_point_shift", 0), "fixed_point_shift"),
        saturating=_require_bool(data.get("saturating", False), "saturating"),
        epoch_id=str(data.get("epoch_id", "")).strip(),
        lane_id=_require_str(data.get("lane_id", ""), "lane_id").strip(),
        schema_version=_require_int(data.get("schema_version", _SCHEMA_VERSION), "schema_version"),
    )
    return descriptor


def validate_matrix_offload_descriptor(descriptor: MatrixOffloadDescriptor) -> MatrixOffloadDescriptor:
    if descriptor.descriptor_id == "":
        raise ValueError("descriptor_id must be non-empty")
    if descriptor.operation_type not in _SUPPORTED_OPERATION_TYPES:
        raise ValueError("unsupported operation_type")
    if descriptor.lane_id not in _SUPPORTED_LANES:
        raise ValueError("unsupported lane_id")
    if descriptor.epoch_id == "":
        raise ValueError("epoch_id must be non-empty")
    if descriptor.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if descriptor.fixed_point_shift < 0:
        raise ValueError("malformed shift parameters")

    a_rows = descriptor.matrix_a
    b_rows = descriptor.matrix_b
    if len(a_rows) == 0 or len(b_rows) == 0:
        raise ValueError("empty matrices are not allowed")

    a_width = len(a_rows[0])
    b_width = len(b_rows[0])

    if descriptor.operation_type == "matmul" and a_width != len(b_rows):
        raise ValueError("inconsistent dimensions")
    if descriptor.operation_type == "matvec":
        if b_width != 1:
            raise ValueError("inconsistent dimensions")
        if a_width != len(b_rows):
            raise ValueError("inconsistent dimensions")
    if descriptor.operation_type in ("accumulate", "fixed_point_mac"):
        if len(a_rows) != len(b_rows) or a_width != b_width:
            raise ValueError("inconsistent dimensions")
    return descriptor


def _matrix_multiply(
    matrix_a: tuple[tuple[int, ...], ...],
    matrix_b: tuple[tuple[int, ...], ...],
    saturating: bool,
) -> tuple[tuple[int, ...], bool]:
    b_width = len(matrix_b[0])
    shared = len(matrix_b)
    overflow = False
    out: list[tuple[int, ...]] = []
    for row in matrix_a:
        out_row: list[int] = []
        for col_idx in range(b_width):
            acc = 0
            for k in range(shared):
                acc += row[k] * matrix_b[k][col_idx]
            acc, did_overflow = _apply_saturation(acc, saturating)
            overflow = overflow or did_overflow
            out_row.append(acc)
        out.append(tuple(out_row))
    return tuple(out), overflow


def _matrix_add(
    matrix_a: tuple[tuple[int, ...], ...],
    matrix_b: tuple[tuple[int, ...], ...],
    saturating: bool,
) -> tuple[tuple[int, ...], bool]:
    overflow = False
    out: list[tuple[int, ...]] = []
    for row_idx, row_a in enumerate(matrix_a):
        row_b = matrix_b[row_idx]
        out_row: list[int] = []
        for col_idx, a_val in enumerate(row_a):
            value, did_overflow = _apply_saturation(a_val + row_b[col_idx], saturating)
            overflow = overflow or did_overflow
            out_row.append(value)
        out.append(tuple(out_row))
    return tuple(out), overflow


def _fixed_point_mac(
    matrix_a: tuple[tuple[int, ...], ...],
    matrix_b: tuple[tuple[int, ...], ...],
    scale_factor: int,
    fixed_point_shift: int,
    saturating: bool,
) -> tuple[tuple[int, ...], bool]:
    overflow = False
    out: list[tuple[int, ...]] = []
    for row_idx, row_a in enumerate(matrix_a):
        row_b = matrix_b[row_idx]
        acc = 0
        for col_idx, a_val in enumerate(row_a):
            acc += a_val * row_b[col_idx]
        scaled = (acc * scale_factor) >> fixed_point_shift
        scaled, did_overflow = _apply_saturation(scaled, saturating)
        overflow = overflow or did_overflow
        out.append((scaled,))
    return tuple(out), overflow


def _transpose(matrix: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    height = len(matrix)
    width = len(matrix[0])
    return tuple(tuple(matrix[row_idx][col_idx] for row_idx in range(height)) for col_idx in range(width))


def _cycle_count(descriptor: MatrixOffloadDescriptor, output_matrix: tuple[tuple[int, ...], ...]) -> int:
    in_cells = len(descriptor.matrix_a) * len(descriptor.matrix_a[0]) + len(descriptor.matrix_b) * len(descriptor.matrix_b[0])
    out_cells = len(output_matrix) * len(output_matrix[0])
    op_weight = _SUPPORTED_OPERATION_TYPES.index(descriptor.operation_type) + 1
    return in_cells + out_cells + (op_weight * 19)


def run_matrix_offload_engine(descriptor: MatrixOffloadDescriptor) -> MatrixOffloadResult:
    descriptor = validate_matrix_offload_descriptor(descriptor)

    overflow_detected = False
    if descriptor.operation_type == "matmul":
        output_matrix, overflow_detected = _matrix_multiply(descriptor.matrix_a, descriptor.matrix_b, descriptor.saturating)
    elif descriptor.operation_type == "matvec":
        output_matrix, overflow_detected = _matrix_multiply(descriptor.matrix_a, descriptor.matrix_b, descriptor.saturating)
    elif descriptor.operation_type == "accumulate":
        output_matrix, overflow_detected = _matrix_add(descriptor.matrix_a, descriptor.matrix_b, descriptor.saturating)
    elif descriptor.operation_type == "fixed_point_mac":
        output_matrix, overflow_detected = _fixed_point_mac(
            descriptor.matrix_a,
            descriptor.matrix_b,
            descriptor.scale_factor,
            descriptor.fixed_point_shift,
            descriptor.saturating,
        )
    elif descriptor.operation_type == "transpose":
        output_matrix = _transpose(descriptor.matrix_a)
    elif descriptor.operation_type == "identity_matrix_pass":
        output_matrix = descriptor.matrix_a
    else:
        raise ValueError("unsupported operation_type")

    output_hash = _sha256_hex(output_matrix)
    return MatrixOffloadResult(
        descriptor_id=descriptor.descriptor_id,
        output_matrix=output_matrix,
        output_hash=output_hash,
        cycle_count=_cycle_count(descriptor, output_matrix),
        overflow_detected=overflow_detected,
        epoch_id=descriptor.epoch_id,
        lane_id=descriptor.lane_id,
    )


def build_matrix_offload_receipt(result: MatrixOffloadResult) -> MatrixOffloadReceipt:
    receipt = MatrixOffloadReceipt(
        receipt_hash="",
        descriptor_id=result.descriptor_id,
        output_hash=result.output_hash,
        cycle_count=result.cycle_count,
        overflow_detected=result.overflow_detected,
        epoch_id=result.epoch_id,
        lane_id=result.lane_id,
        validation_passed=True,
        schema_version=_SCHEMA_VERSION,
    )
    return MatrixOffloadReceipt(
        receipt_hash=receipt.stable_hash(),
        descriptor_id=receipt.descriptor_id,
        output_hash=receipt.output_hash,
        cycle_count=receipt.cycle_count,
        overflow_detected=receipt.overflow_detected,
        epoch_id=receipt.epoch_id,
        lane_id=receipt.lane_id,
        validation_passed=receipt.validation_passed,
        schema_version=receipt.schema_version,
    )


def stable_matrix_offload_hash(report: MatrixOffloadReport) -> str:
    return _sha256_hex(report.to_hash_payload_dict())


def compile_matrix_offload_report(raw_input: Mapping[str, Any] | MatrixOffloadDescriptor) -> MatrixOffloadReport:
    descriptor = validate_matrix_offload_descriptor(normalize_matrix_offload_descriptor(raw_input))
    result = run_matrix_offload_engine(descriptor)
    receipt = build_matrix_offload_receipt(result)
    report = MatrixOffloadReport(
        descriptor=descriptor,
        result=result,
        receipt=receipt,
        stable_hash="",
        schema_version=_SCHEMA_VERSION,
    )
    stable_hash = stable_matrix_offload_hash(report)
    return MatrixOffloadReport(
        descriptor=descriptor,
        result=result,
        receipt=receipt,
        stable_hash=stable_hash,
        schema_version=_SCHEMA_VERSION,
    )
