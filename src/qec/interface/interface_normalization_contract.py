"""v137.21.4 — Interface / Normalization Layer Contract.

Deterministic boundary contract that normalizes external interface captures into
platform-neutral syndrome packages while keeping physical timing and suppression
artifacts as side-band receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

CONTRACT_VERSION = "v137.21.4"
SCHEMA_VERSION = "interface.normalization.v1"

_DTYPE_ALIASES: Mapping[str, str] = {
    "bool": "bool",
    "boolean": "bool",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float16": "float16",
    "half": "float16",
    "float32": "float32",
    "single": "float32",
    "float": "float32",
    "float64": "float64",
    "double": "float64",
}


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _validate_string_only_mapping_keys(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be mapping")
    for key in value.keys():
        if not isinstance(key, str):
            raise TypeError(f"{field} keys must be str")
    return value


def _canonicalize(value: Any, *, field: str) -> Any:
    if isinstance(value, Mapping):
        mapping = _validate_string_only_mapping_keys(value, field=field)
        return {key: _canonicalize(mapping[key], field=f"{field}.{key}") for key in sorted(mapping.keys())}
    if isinstance(value, tuple):
        return [_canonicalize(item, field=field) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item, field=field) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains non-finite float: {value!r}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"{field} contains non-canonical type: {type(value).__name__}")


def _normalize_shape(shape: Any) -> Tuple[int, ...]:
    if isinstance(shape, tuple):
        dims = shape
    elif isinstance(shape, list):
        dims = tuple(shape)
    elif isinstance(shape, int) and not isinstance(shape, bool):
        dims = (shape,)
    elif isinstance(shape, str):
        text = shape.strip().lower()
        if not text:
            raise ValueError("shape must be non-empty")
        for token in ("(", ")", "[", "]"):
            text = text.replace(token, "")
        for token in ("x", "*", ","):
            text = text.replace(token, " ")
        raw_parts = tuple(part for part in text.split() if part)
        if not raw_parts:
            raise ValueError("shape must contain dimensions")
        dims = tuple(int(part) for part in raw_parts)
    else:
        raise TypeError("shape must be tuple/list/int/str")

    normalized_dims: list[int] = []
    for dim in dims:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError("shape dimensions must be int")
        if dim < 0:
            raise ValueError("shape dimensions must be >= 0")
        normalized_dims.append(dim)
    return tuple(normalized_dims)


def _normalize_dtype(dtype: Any) -> str:
    if not isinstance(dtype, str):
        raise TypeError("dtype must be str")
    alias = dtype.strip().lower()
    if alias.startswith("np."):
        alias = alias[3:]
    if alias.startswith("numpy."):
        alias = alias[6:]
    alias = alias.replace("_", "")
    if alias in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[alias]
    raise ValueError(f"unsupported dtype alias: {dtype}")


def _normalize_bit(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        if value in (0, 1):
            return int(value)
        raise ValueError("syndrome bits must be 0/1")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("0", "false"):
            return 0
        if text in ("1", "true"):
            return 1
    raise TypeError("syndrome bit must be explicit boolean/int/string 0/1")


def _canonicalize_syndrome_ordering(payload: Any) -> Tuple[int, ...]:
    if isinstance(payload, Mapping):
        mapping = _validate_string_only_mapping_keys(payload, field="signal_payload")
        ordered_keys = tuple(sorted(mapping.keys()))
        return tuple(_normalize_bit(mapping[key]) for key in ordered_keys)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        return tuple(_normalize_bit(value) for value in payload)
    raise TypeError("signal_payload must be sequence or mapping")


def _shape_cardinality(shape: Tuple[int, ...]) -> int:
    if not shape:
        return 0
    cardinality = 1
    for dim in shape:
        cardinality *= dim
    return cardinality


@dataclass(frozen=True)
class RawInterfaceCapture:
    source_id: str
    capture_type: str
    signal_payload: Any
    shape: Tuple[int, ...]
    dtype: str
    timestamp_receipt_hash: str
    suppression_receipt_hash: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "capture_type": self.capture_type,
            "signal_payload": _canonicalize(self.signal_payload, field="signal_payload"),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "timestamp_receipt_hash": self.timestamp_receipt_hash,
            "suppression_receipt_hash": self.suppression_receipt_hash,
            "metadata": _canonicalize(dict(self.metadata), field="metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class NormalizedSyndromePackage:
    syndrome_bits: Tuple[int, ...]
    shape: Tuple[int, ...]
    normalization_version: str
    logical_payload_hash: str
    sideband_receipt_hashes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "syndrome_bits": list(self.syndrome_bits),
            "shape": list(self.shape),
            "normalization_version": self.normalization_version,
            "logical_payload_hash": self.logical_payload_hash,
            "sideband_receipt_hashes": list(self.sideband_receipt_hashes),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InterfaceNormalizationReport:
    validation_passed: bool
    normalization_steps: Tuple[str, ...]
    warnings: Tuple[str, ...]
    shape_transformations: Tuple[str, ...]
    dropped_metadata_fields: Tuple[str, ...]
    contract_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_passed": self.validation_passed,
            "normalization_steps": list(self.normalization_steps),
            "warnings": list(self.warnings),
            "shape_transformations": list(self.shape_transformations),
            "dropped_metadata_fields": list(self.dropped_metadata_fields),
            "contract_version": self.contract_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InterfaceContractReceipt:
    input_hash: str
    output_hash: str
    report_hash: str
    contract_valid: bool
    schema_version: str
    rationale: Tuple[str, ...]
    version: str = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "report_hash": self.report_hash,
            "contract_valid": self.contract_valid,
            "schema_version": self.schema_version,
            "rationale": list(self.rationale),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InterfaceNormalizationContract:
    contract_version: str = CONTRACT_VERSION
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"contract_version": self.contract_version, "schema_version": self.schema_version}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())

    @staticmethod
    def validate_raw_capture_schema(raw_capture: Any) -> None:
        mapping = _validate_string_only_mapping_keys(raw_capture, field="raw_capture")
        required = (
            "source_id",
            "capture_type",
            "signal_payload",
            "shape",
            "dtype",
            "timestamp_receipt_hash",
            "suppression_receipt_hash",
            "metadata",
        )
        for key in required:
            if key not in mapping:
                raise ValueError(f"raw_capture missing required field: {key}")
        _normalize_text(mapping["source_id"], field="source_id")
        _normalize_text(mapping["capture_type"], field="capture_type")
        normalized_shape = _normalize_shape(mapping["shape"])
        _normalize_dtype(mapping["dtype"])
        _normalize_text(mapping["timestamp_receipt_hash"], field="timestamp_receipt_hash")
        _normalize_text(mapping["suppression_receipt_hash"], field="suppression_receipt_hash")
        _validate_string_only_mapping_keys(mapping["metadata"], field="metadata")
        syndrome_bits = _canonicalize_syndrome_ordering(mapping["signal_payload"])
        expected_bits = _shape_cardinality(normalized_shape)
        if len(syndrome_bits) != expected_bits:
            raise ValueError(
                "signal_payload bit count must match shape cardinality: "
                f"{len(syndrome_bits)} != {expected_bits}"
            )

    @staticmethod
    def build_raw_interface_capture(raw_capture: Any) -> RawInterfaceCapture:
        InterfaceNormalizationContract.validate_raw_capture_schema(raw_capture)
        mapping = _validate_string_only_mapping_keys(raw_capture, field="raw_capture")
        return RawInterfaceCapture(
            source_id=_normalize_text(mapping["source_id"], field="source_id"),
            capture_type=_normalize_text(mapping["capture_type"], field="capture_type"),
            signal_payload=_canonicalize(mapping["signal_payload"], field="signal_payload"),
            shape=_normalize_shape(mapping["shape"]),
            dtype=_normalize_dtype(mapping["dtype"]),
            timestamp_receipt_hash=_normalize_text(mapping["timestamp_receipt_hash"], field="timestamp_receipt_hash"),
            suppression_receipt_hash=_normalize_text(
                mapping["suppression_receipt_hash"], field="suppression_receipt_hash"
            ),
            metadata=_canonicalize(mapping["metadata"], field="metadata"),
        )

    def normalize(
        self, raw_capture: Any
    ) -> Tuple[NormalizedSyndromePackage, InterfaceNormalizationReport, InterfaceContractReceipt]:
        # Capture original shape representation before build_raw_interface_capture normalizes it
        try:
            original_shape_input: Any = raw_capture["shape"]  # type: ignore[index]
        except (KeyError, TypeError):
            original_shape_input = "<missing>"
        raw = self.build_raw_interface_capture(raw_capture)
        syndrome_bits = _canonicalize_syndrome_ordering(raw.signal_payload)
        normalized_shape = _normalize_shape(raw.shape)

        logical_payload = {
            "syndrome_bits": list(syndrome_bits),
            "shape": list(normalized_shape),
            "normalization_version": self.contract_version,
            "dtype": raw.dtype,
        }
        logical_payload_hash = _stable_hash(logical_payload)

        sideband_receipt_hashes = (
            raw.timestamp_receipt_hash,
            raw.suppression_receipt_hash,
        )

        package = NormalizedSyndromePackage(
            syndrome_bits=syndrome_bits,
            shape=normalized_shape,
            normalization_version=self.contract_version,
            logical_payload_hash=logical_payload_hash,
            sideband_receipt_hashes=sideband_receipt_hashes,
        )

        dropped_metadata_fields = tuple(sorted(raw.metadata.keys()))
        report = InterfaceNormalizationReport(
            validation_passed=True,
            normalization_steps=(
                "validate_raw_capture_schema",
                "normalize_shape",
                "normalize_dtype",
                "canonicalize_syndrome_ordering",
                "separate_sideband_metadata",
            ),
            warnings=(),
            shape_transformations=(f"{original_shape_input!r}->{list(normalized_shape)}",),
            dropped_metadata_fields=dropped_metadata_fields,
            contract_version=self.contract_version,
        )

        receipt = InterfaceContractReceipt(
            input_hash=raw.stable_hash(),
            output_hash=package.stable_hash(),
            report_hash=report.stable_hash(),
            contract_valid=report.validation_passed,
            schema_version=self.schema_version,
            rationale=(
                "Deterministic interface normalization contract applied.",
                "Timing and suppression receipts isolated to side-band fields.",
                "Logical syndrome payload preserved as decoder-safe canonical package.",
            ),
        )

        return package, report, receipt
