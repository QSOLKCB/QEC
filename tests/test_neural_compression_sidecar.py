from __future__ import annotations

import pytest

from qec.analysis.neural_compression_sidecar import (
    NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION,
    compile_compression_report,
)


def _base_input(strategy: str, quantization_bits: int) -> dict[str, object]:
    return {
        "artifact_id": "artifact-neural-01",
        "payload": b"deterministic-payload-0123456789",
        "block_size": 7,
        "compression_strategy": strategy,
        "quantization_bits": quantization_bits,
        "epoch_id": "epoch-001",
        "schema_version": NEURAL_COMPRESSION_SIDECAR_SCHEMA_VERSION,
    }


def test_latent_tile_determinism() -> None:
    a = compile_compression_report(_base_input("latent_tiles", 8))
    b = compile_compression_report(_base_input("latent_tiles", 8))
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_quantized_block_determinism() -> None:
    a = compile_compression_report(_base_input("quantized_blocks", 16))
    b = compile_compression_report(_base_input("quantized_blocks", 16))
    assert a.receipt.latent_hash == b.receipt.latent_hash


def test_lookup_projection_determinism() -> None:
    a = compile_compression_report(_base_input("lookup_projection", 8))
    b = compile_compression_report(_base_input("lookup_projection", 8))
    assert a.blocks == b.blocks


def test_identity_pass_determinism() -> None:
    a = compile_compression_report(_base_input("identity_pass", 8))
    b = compile_compression_report(_base_input("identity_pass", 8))
    assert a.stable_hash == b.stable_hash


def test_stable_block_hashes() -> None:
    report = compile_compression_report(_base_input("identity_pass", 8))
    repeated = compile_compression_report(_base_input("identity_pass", 8))
    assert tuple(block.stable_hash for block in report.blocks) == tuple(block.stable_hash for block in repeated.blocks)


def test_stable_receipts() -> None:
    a = compile_compression_report(_base_input("latent_tiles", 8))
    b = compile_compression_report(_base_input("latent_tiles", 8))
    assert a.receipt.to_canonical_bytes() == b.receipt.to_canonical_bytes()


def test_repeated_run_byte_identity() -> None:
    outputs = tuple(compile_compression_report(_base_input("lookup_projection", 8)).to_canonical_bytes() for _ in range(5))
    assert all(payload == outputs[0] for payload in outputs[1:])


def test_unsupported_quantization_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported quantization_bits"):
        compile_compression_report(_base_input("quantized_blocks", 12))


def test_malformed_payload_rejection() -> None:
    raw = _base_input("identity_pass", 8)
    raw["payload"] = object()
    with pytest.raises(ValueError, match="payload"):
        compile_compression_report(raw)


def test_schema_rejection() -> None:
    raw = _base_input("identity_pass", 8)
    raw["schema_version"] = "v0.0.0"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_compression_report(raw)


def test_block_ordering_stability() -> None:
    report = compile_compression_report(_base_input("identity_pass", 8))
    ordering = tuple((block.block_index, block.block_id) for block in report.blocks)
    assert ordering == tuple(sorted(ordering))


def test_compression_ratio_stability() -> None:
    a = compile_compression_report(_base_input("quantized_blocks", 32))
    b = compile_compression_report(_base_input("quantized_blocks", 32))
    assert a.receipt.compression_ratio == b.receipt.compression_ratio
    assert a.receipt.compression_ratio == 1.0
