"""v137.11.6 — Deterministic Latent Decode Lane.

Layer-4 deterministic latent reconstruction pipeline for replay-safe artifact recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

_JSONScalar = str | int | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = "137.11.6"
_SUPPORTED_DECODE_STRATEGIES = (
    "latent_tiles",
    "quantized_blocks",
    "lookup_projection",
    "identity_pass",
)
_SUPPORTED_QUANTIZATION_BITS = (8, 16, 32)


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


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _normalize_latent_blocks(value: Any) -> tuple[dict[str, _JSONValue], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("latent_blocks must be a list or tuple")
    normalized: list[dict[str, _JSONValue]] = []
    for idx, raw_block in enumerate(value):
        block = _require_mapping(raw_block, f"latent_blocks[{idx}]")
        canonical = _canonicalize_json(block)
        if not isinstance(canonical, dict):
            raise ValueError("malformed latent block payload")
        normalized.append(canonical)
    if len(normalized) == 0:
        raise ValueError("latent_blocks must be non-empty")
    return tuple(normalized)


def _signed_int_bounds(bits: int) -> tuple[int, int]:
    half = 2 ** (bits - 1)
    return -half, half - 1


def _decode_quantized_values(values: tuple[int, ...], quantization_bits: int) -> bytes:
    low, high = _signed_int_bounds(quantization_bits)
    width = quantization_bits // 8
    out = bytearray()
    for idx, value in enumerate(values):
        integer = _require_int(value, f"quantized_values[{idx}]")
        if integer < low or integer > high:
            raise ValueError("quantized value out of range")
        out.extend(integer.to_bytes(width, byteorder="big", signed=True))
    return bytes(out)


def _extract_values_tuple(payload: Mapping[str, Any], field_name: str) -> tuple[int, ...]:
    values_raw = payload.get(field_name)
    if not isinstance(values_raw, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple")
    return tuple(_require_int(v, f"{field_name}[{i}]") for i, v in enumerate(values_raw))


@dataclass(frozen=True)
class DecodeDescriptor:
    artifact_id: str
    latent_blocks: tuple[dict[str, _JSONValue], ...]
    decode_strategy: str
    quantization_bits: int
    epoch_id: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "artifact_id": self.artifact_id,
            "latent_blocks": self.latent_blocks,
            "decode_strategy": self.decode_strategy,
            "quantization_bits": self.quantization_bits,
            "epoch_id": self.epoch_id,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DecodedBlock:
    block_id: str
    artifact_id: str
    block_index: int
    decoded_payload: bytes
    stable_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "block_id": self.block_id,
            "artifact_id": self.artifact_id,
            "block_index": self.block_index,
            "decoded_payload_hex": self.decoded_payload.hex(),
            "stable_hash": self.stable_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DecodeReceipt:
    receipt_hash: str
    artifact_id: str
    block_count: int
    decoded_hash: str
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "receipt_hash": self.receipt_hash,
            "artifact_id": self.artifact_id,
            "block_count": self.block_count,
            "decoded_hash": self.decoded_hash,
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


@dataclass(frozen=True)
class DecodeReport:
    descriptor: DecodeDescriptor
    decoded_blocks: tuple[DecodedBlock, ...]
    receipt: DecodeReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "decoded_blocks": tuple(block.to_dict() for block in self.decoded_blocks),
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "decoded_blocks": tuple(block.to_dict() for block in self.decoded_blocks),
            "receipt": self.receipt.to_dict(),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_decode_descriptor(raw_input: Mapping[str, Any] | DecodeDescriptor) -> DecodeDescriptor:
    if isinstance(raw_input, DecodeDescriptor):
        data: Mapping[str, Any] = raw_input.to_dict()
    else:
        data = _require_mapping(raw_input, "raw_input")

    return DecodeDescriptor(
        artifact_id=_require_str(data.get("artifact_id", ""), "artifact_id").strip(),
        latent_blocks=_normalize_latent_blocks(data.get("latent_blocks", ())),
        decode_strategy=_require_str(data.get("decode_strategy", ""), "decode_strategy").strip(),
        quantization_bits=_require_int(data.get("quantization_bits", 0), "quantization_bits"),
        epoch_id=_require_str(data.get("epoch_id", ""), "epoch_id").strip(),
        schema_version=_require_str(data.get("schema_version", _SCHEMA_VERSION), "schema_version").strip(),
    )


def validate_decode_descriptor(descriptor: DecodeDescriptor) -> DecodeDescriptor:
    if descriptor.artifact_id == "":
        raise ValueError("artifact_id must be non-empty")
    if descriptor.epoch_id == "":
        raise ValueError("epoch_id must be non-empty")
    if len(descriptor.latent_blocks) == 0:
        raise ValueError("latent_blocks must be non-empty")
    if descriptor.decode_strategy not in _SUPPORTED_DECODE_STRATEGIES:
        raise ValueError("unsupported decode_strategy")
    if descriptor.quantization_bits not in _SUPPORTED_QUANTIZATION_BITS:
        raise ValueError("unsupported quantization_bits")
    if descriptor.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema version")

    seen_ids: set[str] = set()
    for idx, block in enumerate(descriptor.latent_blocks):
        block_id = block.get("block_id")
        block_index = block.get("block_index")
        if not isinstance(block_id, str) or block_id.strip() == "":
            raise ValueError("malformed latent block payload")
        if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
            raise ValueError("malformed latent block payload")
        if block_id in seen_ids:
            raise ValueError("duplicate block IDs")
        seen_ids.add(block_id)
    return descriptor


def _decode_block_payload(
    *,
    block: Mapping[str, Any],
    strategy: str,
    quantization_bits: int,
) -> bytes:
    if strategy == "identity_pass":
        byte_values = _extract_values_tuple(block, "byte_values")
        for idx, value in enumerate(byte_values):
            if value < 0 or value > 255:
                raise ValueError(f"byte_values[{idx}] out of range")
        return bytes(byte_values)

    quantized_values = _extract_values_tuple(block, "quantized_values")
    if strategy == "lookup_projection":
        table_raw = block.get("lookup_table")
        table = _require_mapping(table_raw, "lookup_table")
        projected_values: list[int] = []
        for idx, source_value in enumerate(quantized_values):
            key = str(source_value)
            if key not in table:
                raise ValueError(f"lookup_table missing projection for value index {idx}")
            projected_values.append(_require_int(table[key], f"lookup_table[{key}]") )
        return _decode_quantized_values(tuple(projected_values), quantization_bits)

    if strategy in ("latent_tiles", "quantized_blocks"):
        return _decode_quantized_values(quantized_values, quantization_bits)

    raise ValueError("unsupported decode_strategy")


def decode_latent_blocks(descriptor: DecodeDescriptor) -> tuple[DecodedBlock, ...]:
    descriptor = validate_decode_descriptor(descriptor)
    decoded: list[DecodedBlock] = []
    for block in descriptor.latent_blocks:
        block_id = str(block["block_id"])
        block_index = int(block["block_index"])
        decoded_payload = _decode_block_payload(
            block=block,
            strategy=descriptor.decode_strategy,
            quantization_bits=descriptor.quantization_bits,
        )
        block_proto = DecodedBlock(
            block_id=block_id,
            artifact_id=descriptor.artifact_id,
            block_index=block_index,
            decoded_payload=decoded_payload,
            stable_hash="",
        )
        decoded.append(
            DecodedBlock(
                block_id=block_proto.block_id,
                artifact_id=block_proto.artifact_id,
                block_index=block_proto.block_index,
                decoded_payload=block_proto.decoded_payload,
                stable_hash=_sha256_hex(block_proto.to_hash_payload_dict()),
            )
        )
    return tuple(decoded)


def merge_decoded_blocks(blocks: tuple[DecodedBlock, ...]) -> bytes:
    ordered = sorted(blocks, key=lambda block: (block.block_index, block.block_id))
    merged = bytearray()
    for block in ordered:
        merged.extend(block.decoded_payload)
    return bytes(merged)


def build_decode_receipt(blocks: tuple[DecodedBlock, ...], merged_output: bytes) -> DecodeReceipt:
    if len(blocks) == 0:
        raise ValueError("blocks must be non-empty")
    artifact_ids = {block.artifact_id for block in blocks}
    if len(artifact_ids) != 1:
        raise ValueError("decoded blocks must share artifact_id")

    receipt = DecodeReceipt(
        receipt_hash="",
        artifact_id=next(iter(artifact_ids)),
        block_count=len(blocks),
        decoded_hash=hashlib.sha256(merged_output).hexdigest(),
        validation_passed=True,
        schema_version=_SCHEMA_VERSION,
    )
    return DecodeReceipt(
        receipt_hash=_sha256_hex(receipt.to_hash_payload_dict()),
        artifact_id=receipt.artifact_id,
        block_count=receipt.block_count,
        decoded_hash=receipt.decoded_hash,
        validation_passed=receipt.validation_passed,
        schema_version=receipt.schema_version,
    )


def stable_decode_report_hash(report: DecodeReport) -> str:
    return _sha256_hex(report.to_hash_payload_dict())


def compile_decode_report(raw_input: Mapping[str, Any] | DecodeDescriptor) -> DecodeReport:
    descriptor = validate_decode_descriptor(normalize_decode_descriptor(raw_input))
    decoded_blocks = decode_latent_blocks(descriptor)
    merged_output = merge_decoded_blocks(decoded_blocks)
    receipt = build_decode_receipt(decoded_blocks, merged_output)
    report = DecodeReport(
        descriptor=descriptor,
        decoded_blocks=decoded_blocks,
        receipt=receipt,
        stable_hash="",
        schema_version=_SCHEMA_VERSION,
    )
    return DecodeReport(
        descriptor=report.descriptor,
        decoded_blocks=report.decoded_blocks,
        receipt=report.receipt,
        stable_hash=stable_decode_report_hash(report),
        schema_version=report.schema_version,
    )
