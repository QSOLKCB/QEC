from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION = "v137.11.5"
_SUPPORTED_COMPRESSION_STRATEGIES = {
    "latent_tiles",
    "quantized_blocks",
    "lookup_projection",
    "identity_pass",
}
_SUPPORTED_QUANTIZATION_BITS = {8, 16, 32}

_LOOKUP_TABLE = bytes((i * 73 + 19) % 256 for i in range(256))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_no_callable(value: Any, *, name: str) -> None:
    if callable(value):
        raise ValueError(f"{name} must not contain callables")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_callable(item, name=f"{name}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_callable(item, name=f"{name}[{index}]")


def _normalize_token(value: Any, *, name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a non-empty string")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _normalize_payload(value: Any, *, name: str) -> bytes:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, bytearray):
        payload = bytes(value)
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    elif isinstance(value, (list, tuple)):
        byte_values: list[int] = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{name}[{index}] must be an integer byte")
            if item < 0 or item > 255:
                raise ValueError(f"{name}[{index}] must be within 0..255")
            byte_values.append(item)
        payload = bytes(byte_values)
    else:
        raise ValueError(f"{name} must be bytes-like, UTF-8 text, or integer byte sequence")

    if not payload:
        raise ValueError(f"{name} must be non-empty")
    return payload


def _normalize_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if normalized <= 0:
        raise ValueError(f"{name} must be > 0")
    return normalized


def _quantize_bytes(block: bytes, quantization_bits: int) -> bytes:
    if quantization_bits == 8:
        return block
    if quantization_bits == 16:
        out = bytearray(len(block) * 2)
        for index, value in enumerate(block):
            quantized = value << 8
            out[index * 2 : index * 2 + 2] = quantized.to_bytes(2, byteorder="big", signed=False)
        return bytes(out)
    if quantization_bits == 32:
        out = bytearray(len(block) * 4)
        for index, value in enumerate(block):
            quantized = value << 24
            out[index * 4 : index * 4 + 4] = quantized.to_bytes(4, byteorder="big", signed=False)
        return bytes(out)
    raise ValueError("unsupported quantization_bits")


def _compress_block(block: bytes, *, strategy: str, quantization_bits: int) -> bytes:
    if strategy == "identity_pass":
        return block
    if strategy == "quantized_blocks":
        return _quantize_bytes(block, quantization_bits)
    if strategy == "lookup_projection":
        return bytes(_LOOKUP_TABLE[b] for b in block)
    if strategy == "latent_tiles":
        if not block:
            return b""
        out = bytearray()
        previous = 0
        for index, value in enumerate(block):
            delta = value if index == 0 else (value - previous) % 256
            out.append(delta)
            previous = value
        return bytes(out)
    raise ValueError("unsupported compression_strategy")


@dataclass(frozen=True)
class CompressionDescriptor:
    artifact_id: str
    payload: bytes
    block_size: int
    compression_strategy: str
    quantization_bits: int
    epoch_id: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "payload_hex": self.payload.hex(),
            "block_size": int(self.block_size),
            "compression_strategy": self.compression_strategy,
            "quantization_bits": int(self.quantization_bits),
            "epoch_id": self.epoch_id,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class LatentBlock:
    block_id: str
    artifact_id: str
    block_index: int
    latent_payload: bytes
    quantization_bits: int
    stable_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "artifact_id": self.artifact_id,
            "block_index": int(self.block_index),
            "latent_payload_hex": self.latent_payload.hex(),
            "quantization_bits": int(self.quantization_bits),
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CompressionReceipt:
    receipt_hash: str
    artifact_id: str
    block_count: int
    compression_ratio: float
    latent_hash: str
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_hash": self.receipt_hash,
            "artifact_id": self.artifact_id,
            "block_count": int(self.block_count),
            "compression_ratio": float(self.compression_ratio),
            "latent_hash": self.latent_hash,
            "validation_passed": bool(self.validation_passed),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CompressionReport:
    descriptor: CompressionDescriptor
    blocks: tuple[LatentBlock, ...]
    receipt: CompressionReceipt
    stable_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "blocks": [block.to_dict() for block in self.blocks],
            "receipt": self.receipt.to_dict(),
            "stable_hash": self.stable_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_compression_descriptor(raw_input: Mapping[str, Any]) -> CompressionDescriptor:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")
    _assert_no_callable(raw_input, name="raw_input")

    descriptor = CompressionDescriptor(
        artifact_id=_normalize_token(raw_input.get("artifact_id", ""), name="artifact_id"),
        payload=_normalize_payload(raw_input.get("payload", b""), name="payload"),
        block_size=_normalize_positive_int(raw_input.get("block_size", 0), name="block_size"),
        compression_strategy=_normalize_token(raw_input.get("compression_strategy", ""), name="compression_strategy"),
        quantization_bits=_normalize_positive_int(raw_input.get("quantization_bits", 0), name="quantization_bits"),
        epoch_id=_normalize_token(raw_input.get("epoch_id", ""), name="epoch_id"),
        schema_version=_normalize_token(
            raw_input.get("schema_version", NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION),
            name="schema_version",
        ),
    )
    validate_compression_descriptor(descriptor)
    return descriptor


def validate_compression_descriptor(descriptor: CompressionDescriptor) -> None:
    if descriptor.compression_strategy not in _SUPPORTED_COMPRESSION_STRATEGIES:
        raise ValueError("unsupported compression_strategy")
    if descriptor.quantization_bits not in _SUPPORTED_QUANTIZATION_BITS:
        raise ValueError("unsupported quantization_bits")
    if descriptor.schema_version != NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")


def _build_block_id(*, artifact_id: str, block_index: int, latent_payload: bytes) -> str:
    seed = f"{artifact_id}:{block_index}".encode("utf-8") + b":" + latent_payload
    return _sha256_hex_bytes(seed)


def _build_block_hash(block: LatentBlock) -> str:
    payload = {
        "block_id": block.block_id,
        "artifact_id": block.artifact_id,
        "block_index": int(block.block_index),
        "latent_payload_hex": block.latent_payload.hex(),
        "quantization_bits": int(block.quantization_bits),
    }
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


def compress_to_latent_blocks(descriptor: CompressionDescriptor) -> tuple[LatentBlock, ...]:
    validate_compression_descriptor(descriptor)
    blocks: list[LatentBlock] = []
    for block_index, start in enumerate(range(0, len(descriptor.payload), descriptor.block_size)):
        chunk = descriptor.payload[start : start + descriptor.block_size]
        latent_payload = _compress_block(
            chunk,
            strategy=descriptor.compression_strategy,
            quantization_bits=descriptor.quantization_bits,
        )
        block_id = _build_block_id(
            artifact_id=descriptor.artifact_id,
            block_index=block_index,
            latent_payload=latent_payload,
        )
        candidate = LatentBlock(
            block_id=block_id,
            artifact_id=descriptor.artifact_id,
            block_index=block_index,
            latent_payload=latent_payload,
            quantization_bits=descriptor.quantization_bits,
            stable_hash="",
        )
        blocks.append(replace(candidate, stable_hash=_build_block_hash(candidate)))

    if not blocks:
        raise ValueError("payload must produce at least one latent block")

    return tuple(sorted(blocks, key=lambda block: (block.block_index, block.block_id)))


def merge_latent_blocks(blocks: Sequence[LatentBlock]) -> bytes:
    ordered = tuple(sorted(tuple(blocks), key=lambda block: (block.block_index, block.block_id)))
    return b"".join(block.latent_payload for block in ordered)


def build_compression_receipt(
    blocks: Sequence[LatentBlock],
    merged_latent: bytes,
    original_payload_size: int,
) -> CompressionReceipt:
    ordered = tuple(sorted(tuple(blocks), key=lambda block: (block.block_index, block.block_id)))
    if not ordered:
        raise ValueError("blocks must be non-empty")
    artifact_ids = {block.artifact_id for block in ordered}
    if len(artifact_ids) != 1:
        raise ValueError("all blocks must share the same artifact_id")

    canonical_merged_latent = merge_latent_blocks(ordered)
    expected_hashes = tuple(_build_block_hash(block) for block in ordered)
    block_hashes_match = expected_hashes == tuple(block.stable_hash for block in ordered)
    merged_latent_matches = merged_latent == canonical_merged_latent
    validation_passed = block_hashes_match and merged_latent_matches
    latent_hash = _sha256_hex_bytes(canonical_merged_latent)

    denominator = original_payload_size if original_payload_size > 0 else 1
    compression_ratio = round(float(len(canonical_merged_latent)) / float(denominator), 12)

    payload = {
        "artifact_id": ordered[0].artifact_id,
        "block_count": len(ordered),
        "compression_ratio": compression_ratio,
        "latent_hash": latent_hash,
        "validation_passed": validation_passed,
        "schema_version": NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION,
    }
    receipt_hash = _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))
    return CompressionReceipt(
        receipt_hash=receipt_hash,
        artifact_id=ordered[0].artifact_id,
        block_count=len(ordered),
        compression_ratio=compression_ratio,
        latent_hash=latent_hash,
        validation_passed=validation_passed,
        schema_version=NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION,
    )


def stable_compression_report_hash(report: CompressionReport) -> str:
    payload = {
        "descriptor": report.descriptor.to_dict(),
        "blocks": [block.to_dict() for block in report.blocks],
        "receipt": report.receipt.to_dict(),
        "schema_version": report.schema_version,
    }
    return _sha256_hex_bytes(_canonical_json(payload).encode("utf-8"))


def compile_compression_report(raw_input: Mapping[str, Any]) -> CompressionReport:
    descriptor = normalize_compression_descriptor(raw_input)
    validate_compression_descriptor(descriptor)
    blocks = compress_to_latent_blocks(descriptor)
    merged_latent = merge_latent_blocks(blocks)
    receipt = build_compression_receipt(blocks, merged_latent, len(descriptor.payload))
    interim = CompressionReport(
        descriptor=descriptor,
        blocks=blocks,
        receipt=receipt,
        stable_hash="",
        schema_version=NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION,
    )
    stable_hash = stable_compression_report_hash(interim)
    return replace(interim, stable_hash=stable_hash)
