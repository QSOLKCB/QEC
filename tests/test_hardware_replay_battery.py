from __future__ import annotations

import pytest

from qec.analysis.hardware_replay_battery import (
    HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    build_replay_battery_receipt,
    compile_replay_report,
    normalize_replay_input,
)


def _h(char: str) -> str:
    return char * 64


def _base_input() -> dict[str, object]:
    return {
        "artifact_id": "artifact-01",
        "artifact_type": "scheduler_receipt",
        "canonical_input_hash": _h("1"),
        "canonical_output_hash": _h("2"),
        "epoch_id": "epoch-007",
        "shard_ids": ["shard-b", "shard-a"],
        "schema_version": HARDWARE_REPLAY_BATTERY_SCHEMA_VERSION,
    }


def _base_runs() -> list[dict[str, object]]:
    return [
        {
            "run_id": "run-0",
            "artifact_id": "artifact-01",
            "input_hash": _h("1"),
            "output_hash": _h("2"),
            "epoch_id": "epoch-007",
            "run_index": 0,
            "shard_ids": ["shard-a", "shard-b"],
        },
        {
            "run_id": "run-1",
            "artifact_id": "artifact-01",
            "input_hash": _h("1"),
            "output_hash": _h("2"),
            "epoch_id": "epoch-007",
            "run_index": 1,
            "shard_ids": ["shard-b", "shard-a"],
        },
    ]


def test_byte_identical_replay_pass() -> None:
    report = compile_replay_report(_base_input(), _base_runs())
    assert report.comparison.byte_identical is True
    assert report.receipt.validation_passed is True
    assert report.receipt.fail_count == 0


def test_hash_mismatch_fail() -> None:
    raw_input = _base_input()
    raw_runs = _base_runs()
    raw_runs[0]["output_hash"] = _h("a")
    report = compile_replay_report(raw_input, raw_runs)
    assert report.comparison.byte_identical is False
    assert report.receipt.validation_passed is False
    assert report.receipt.fail_count >= 1


def test_epoch_mismatch_fail() -> None:
    raw_runs = _base_runs()
    raw_runs[1]["epoch_id"] = "epoch-008"
    report = compile_replay_report(_base_input(), raw_runs)
    assert report.comparison.epoch_match is False
    assert report.receipt.validation_passed is False


def test_shard_mismatch_fail() -> None:
    raw_runs = _base_runs()
    raw_runs[1]["shard_ids"] = ["shard-z"]
    report = compile_replay_report(_base_input(), raw_runs)
    assert report.comparison.shard_match is False
    assert report.receipt.validation_passed is False


def test_repeated_run_identity() -> None:
    reports = [compile_replay_report(_base_input(), _base_runs()) for _ in range(5)]
    assert all(r.to_canonical_bytes() == reports[0].to_canonical_bytes() for r in reports[1:])


def test_stable_receipts() -> None:
    comparison = compile_replay_report(_base_input(), _base_runs()).comparison
    receipt_a = build_replay_battery_receipt(comparison)
    receipt_b = build_replay_battery_receipt(comparison)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_stable_report_hashes() -> None:
    report_a = compile_replay_report(_base_input(), _base_runs())
    report_b = compile_replay_report(_base_input(), _base_runs())
    assert report_a.stable_hash == report_b.stable_hash


def test_malformed_input_rejection() -> None:
    with pytest.raises(ValueError, match="canonical_input_hash"):
        compile_replay_report({**_base_input(), "canonical_input_hash": "not-a-hash"}, _base_runs())


def test_schema_rejection() -> None:
    with pytest.raises(ValueError, match="unsupported schema version"):
        compile_replay_report({**_base_input(), "schema_version": "v0.0.0"}, _base_runs())


def test_repeated_battery_determinism() -> None:
    bytes_a = tuple(compile_replay_report(_base_input(), _base_runs()).to_canonical_bytes() for _ in range(10))
    assert all(payload == bytes_a[0] for payload in bytes_a[1:])


def test_explicit_fail_count_correctness() -> None:
    raw_runs = _base_runs()
    raw_runs[1]["output_hash"] = _h("f")
    raw_runs[1]["epoch_id"] = "epoch-x"
    raw_runs[1]["shard_ids"] = ["shard-x"]
    report = compile_replay_report(_base_input(), raw_runs)
    assert report.receipt.fail_count == 3
    assert report.receipt.pass_count == 1


def test_receipt_stability() -> None:
    report = compile_replay_report(_base_input(), _base_runs())
    assert report.receipt.receipt_hash == build_replay_battery_receipt(report.comparison).receipt_hash


def test_callable_leakage_rejected() -> None:
    bad_input = _base_input()
    bad_input["artifact_id"] = lambda: "artifact"
    with pytest.raises(ValueError, match="callables"):
        normalize_replay_input(bad_input)
