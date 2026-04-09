from __future__ import annotations

import hashlib

import pytest

from qec.analysis.deterministic_latent_decode_lane import (
    compile_decode_report,
    merge_decoded_blocks,
)


def _base_input(strategy: str) -> dict[str, object]:
    return {
        "artifact_id": "artifact-A",
        "latent_blocks": (
            {
                "block_id": "b-2",
                "block_index": 1,
                "quantized_values": (1, -2, 3),
                "byte_values": (7, 8),
                "lookup_table": {"1": 3, "-2": 1, "3": -3},
            },
            {
                "block_id": "b-1",
                "block_index": 0,
                "quantized_values": (-4, 5),
                "byte_values": (9, 10, 11),
                "lookup_table": {"-4": -1, "5": 2},
            },
        ),
        "decode_strategy": strategy,
        "quantization_bits": 8,
        "epoch_id": "epoch-7",
        "schema_version": "v137.11.6",
    }


def test_latent_tile_decode_determinism() -> None:
    raw = _base_input("latent_tiles")
    a = compile_decode_report(raw)
    b = compile_decode_report(raw)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_quantized_block_decode_determinism() -> None:
    raw = _base_input("quantized_blocks")
    a = compile_decode_report(raw)
    b = compile_decode_report(raw)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_lookup_projection_determinism() -> None:
    raw = _base_input("lookup_projection")
    a = compile_decode_report(raw)
    b = compile_decode_report(raw)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_identity_pass_determinism() -> None:
    raw = _base_input("identity_pass")
    a = compile_decode_report(raw)
    b = compile_decode_report(raw)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_stable_decoded_hashes_and_receipts() -> None:
    report = compile_decode_report(_base_input("latent_tiles"))
    for block in report.decoded_blocks:
        assert len(block.stable_hash) == 64
        int(block.stable_hash, 16)
    assert len(report.receipt.receipt_hash) == 64
    assert report.receipt.validation_passed is True


def test_repeated_run_byte_identity() -> None:
    raw = _base_input("lookup_projection")
    artifacts = tuple(compile_decode_report(raw).to_canonical_bytes() for _ in range(8))
    assert len(set(artifacts)) == 1


def test_unsupported_quantization_rejection() -> None:
    raw = _base_input("latent_tiles")
    raw["quantization_bits"] = 12
    with pytest.raises(ValueError, match="unsupported quantization_bits"):
        compile_decode_report(raw)


def test_malformed_payload_rejection() -> None:
    raw = _base_input("latent_tiles")
    raw["latent_blocks"] = ({"block_id": "b-1", "block_index": 0, "quantized_values": (1.5,)},)
    with pytest.raises(ValueError, match="floating point values are not allowed"):
        compile_decode_report(raw)


def test_duplicate_block_rejection() -> None:
    raw = _base_input("latent_tiles")
    raw["latent_blocks"] = (
        {"block_id": "dup", "block_index": 0, "quantized_values": (1,)},
        {"block_id": "dup", "block_index": 1, "quantized_values": (2,)},
    )
    with pytest.raises(ValueError, match="duplicate block IDs"):
        compile_decode_report(raw)


def test_schema_rejection() -> None:
    raw = _base_input("latent_tiles")
    raw["schema_version"] = "137.11.5"
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_decode_report(raw)


def test_merge_ordering_stability() -> None:
    report = compile_decode_report(_base_input("identity_pass"))
    merged = merge_decoded_blocks(report.decoded_blocks)
    expected = bytes((9, 10, 11, 7, 8))
    assert merged == expected
    assert report.receipt.decoded_hash == hashlib.sha256(expected).hexdigest()
