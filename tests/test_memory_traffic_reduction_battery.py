from __future__ import annotations

import pytest

from qec.analysis.memory_traffic_reduction_battery import (
    TRAFFIC_BATTERY_SCHEMA_VERSION,
    TrafficBatteryInput,
    compile_traffic_battery_report,
    normalize_traffic_battery_input,
    validate_traffic_battery_input,
)

_ORIG_CONTENT_HASH = "0" * 64
_DECODED_CONTENT_HASH_MATCH = "0" * 64
_DECODED_CONTENT_HASH_MISMATCH = "1" * 64


def _base_input() -> dict[str, object]:
    return {
        "artifact_id": "artifact-alpha",
        "original_bytes": 1000,
        "latent_bytes": 400,
        "decoded_bytes": 1000,
        "epoch_id": "epoch-001",
        "schema_version": TRAFFIC_BATTERY_SCHEMA_VERSION,
        "original_content_hash": _ORIG_CONTENT_HASH,
        "decoded_content_hash": _DECODED_CONTENT_HASH_MATCH,
    }


def test_bytes_saved_and_reduction_ratio_correctness() -> None:
    report = compile_traffic_battery_report(_base_input())
    assert report.metrics.bytes_saved == 600
    assert report.metrics.traffic_reduction_ratio == 0.6
    assert report.metrics.compression_ratio == 0.4


def test_decode_match_passes_and_validation_true() -> None:
    report = compile_traffic_battery_report(_base_input())
    assert report.metrics.decode_byte_match is True
    assert report.comparison.byte_identical is True
    assert report.receipt.validation_passed is True


def test_decode_mismatch_fails_validation() -> None:
    raw = _base_input()
    raw["decoded_bytes"] = 999
    raw["decoded_content_hash"] = _DECODED_CONTENT_HASH_MISMATCH
    report = compile_traffic_battery_report(raw)
    assert report.metrics.decode_byte_match is False
    assert report.comparison.byte_identical is False
    assert report.receipt.validation_passed is False


def test_stable_hashes_and_receipts_are_deterministic() -> None:
    a = compile_traffic_battery_report(_base_input())
    b = compile_traffic_battery_report(_base_input())
    assert a.metrics.stable_hash == b.metrics.stable_hash
    assert a.stable_hash == b.stable_hash
    assert a.receipt.receipt_hash == b.receipt.receipt_hash
    assert a.receipt.to_canonical_bytes() == b.receipt.to_canonical_bytes()


def test_repeated_run_byte_identity() -> None:
    artifacts = tuple(compile_traffic_battery_report(_base_input()).to_canonical_bytes() for _ in range(25))
    assert len(set(artifacts)) == 1


def test_malformed_input_rejection() -> None:
    with pytest.raises(ValueError, match="raw_input must be a mapping"):
        compile_traffic_battery_report(object())

    raw = _base_input()
    raw["artifact_id"] = ""
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        compile_traffic_battery_report(raw)

    for field in ("original_content_hash", "decoded_content_hash"):
        raw_missing = _base_input()
        del raw_missing[field]
        with pytest.raises(ValueError, match=f"{field} must be a string"):
            compile_traffic_battery_report(raw_missing)

        raw_empty = _base_input()
        raw_empty[field] = ""
        with pytest.raises(ValueError, match=f"{field} must be non-empty"):
            compile_traffic_battery_report(raw_empty)


def test_validate_catches_directly_constructed_invariants() -> None:
    bad = TrafficBatteryInput(
        artifact_id="a",
        original_bytes=0,
        latent_bytes=400,
        decoded_bytes=1000,
        epoch_id="epoch-001",
        schema_version=TRAFFIC_BATTERY_SCHEMA_VERSION,
        original_content_hash=_ORIG_CONTENT_HASH,
        decoded_content_hash=_DECODED_CONTENT_HASH_MATCH,
    )
    with pytest.raises(ValueError, match="original_bytes must be positive"):
        validate_traffic_battery_input(bad)

    bad_id = TrafficBatteryInput(
        artifact_id="",
        original_bytes=100,
        latent_bytes=40,
        decoded_bytes=100,
        epoch_id="epoch-001",
        schema_version=TRAFFIC_BATTERY_SCHEMA_VERSION,
        original_content_hash=_ORIG_CONTENT_HASH,
        decoded_content_hash=_DECODED_CONTENT_HASH_MATCH,
    )
    with pytest.raises(ValueError, match="artifact_id must be non-empty"):
        validate_traffic_battery_input(bad_id)


def test_zero_and_negative_byte_rejection() -> None:
    for field_name in ("original_bytes", "latent_bytes", "decoded_bytes"):
        raw_zero = _base_input()
        raw_zero[field_name] = 0
        with pytest.raises(ValueError, match=f"{field_name} must be positive"):
            compile_traffic_battery_report(raw_zero)

        raw_negative = _base_input()
        raw_negative[field_name] = -1
        with pytest.raises(ValueError, match=f"{field_name} must be positive"):
            compile_traffic_battery_report(raw_negative)


def test_schema_rejection() -> None:
    raw = _base_input()
    raw["schema_version"] = "v0.0.0"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        compile_traffic_battery_report(raw)


def test_ratio_stability_rounds_to_twelve_decimals() -> None:
    raw = _base_input()
    raw["original_bytes"] = 3
    raw["latent_bytes"] = 2
    raw["decoded_bytes"] = 3
    report = compile_traffic_battery_report(raw)
    assert report.metrics.compression_ratio == 0.666666666667
    assert report.metrics.traffic_reduction_ratio == 0.333333333333


def test_byte_identical_and_decode_byte_match_are_distinct_signals() -> None:
    # Content hashes match but byte counts differ: decode_byte_match=False, byte_identical=True
    raw = _base_input()
    raw["decoded_bytes"] = 999
    # keep decoded_content_hash == original_content_hash (both "0" * 64)
    report = compile_traffic_battery_report(raw)
    assert report.metrics.decode_byte_match is False
    assert report.comparison.byte_identical is True
    assert report.receipt.validation_passed is True  # byte_identical=True, reduction_verified=True

    # Content hashes differ but byte counts match: decode_byte_match=True, byte_identical=False
    raw2 = _base_input()
    raw2["decoded_content_hash"] = _DECODED_CONTENT_HASH_MISMATCH
    report2 = compile_traffic_battery_report(raw2)
    assert report2.metrics.decode_byte_match is True
    assert report2.comparison.byte_identical is False
    assert report2.receipt.validation_passed is False


def test_callable_leakage_rejection() -> None:
    raw = _base_input()
    raw["artifact_id"] = lambda: "artifact-alpha"
    with pytest.raises(ValueError, match="callable leakage is not allowed"):
        normalize_traffic_battery_input(raw)
