from __future__ import annotations

import hashlib

import pytest

from qec.analysis.scientific_replay_battery import (
    SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    ReplayBatteryReport,
    ReplayCase,
    compile_replay_battery,
    normalize_replay_case,
    run_replay_battery,
    run_replay_case,
    stable_replay_battery_hash,
)


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _case(case_id: str, payload: bytes, *, verdict: str = "pass") -> dict[str, object]:
    return {
        "case_id": case_id,
        "experiment_hash": "a" * 64,
        "evidence_hash": "b" * 64,
        "audit_hash": "c" * 64,
        "canonical_input_bytes": payload,
        "expected_output_hash": _hash_bytes(payload),
        "expected_verdict": verdict,
        "schema_version": SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    }


def _replay_ok(payload: bytes) -> dict[str, object]:
    return {
        "replay_hash": _hash_bytes(payload),
        "replay_verdict": "pass",
        "canonical_output_bytes": payload,
    }


def test_identical_replay_passes() -> None:
    case = normalize_replay_case(_case("c1", b"alpha"))
    result_a = run_replay_case(case, replay_fn=_replay_ok)
    result_b = run_replay_case(case, replay_fn=_replay_ok)

    assert result_a == result_b
    assert result_a.deterministic_pass is True
    assert result_a.failure_reason == ""


def test_hash_mismatch_fails() -> None:
    case = normalize_replay_case(_case("c1", b"alpha"))

    result = run_replay_case(
        case,
        replay_fn=lambda payload: {
            "replay_hash": _hash_bytes(payload + b"x"),
            "replay_verdict": "pass",
            "canonical_output_bytes": payload,
        },
    )

    assert result.deterministic_pass is False
    assert result.failure_reason == "hash mismatch"


def test_verdict_mismatch_fails() -> None:
    case = normalize_replay_case(_case("c1", b"alpha", verdict="accept"))

    result = run_replay_case(
        case,
        replay_fn=lambda payload: {
            "replay_hash": _hash_bytes(payload),
            "replay_verdict": "reject",
            "canonical_output_bytes": payload,
        },
    )

    assert result.deterministic_pass is False
    assert result.failure_reason == "verdict mismatch"


def test_byte_mismatch_fails() -> None:
    case = normalize_replay_case(_case("c1", b"alpha"))

    result = run_replay_case(
        case,
        replay_fn=lambda payload: {
            "replay_hash": _hash_bytes(payload),
            "replay_verdict": "pass",
            "canonical_output_bytes": payload + b"z",
        },
    )

    assert result.deterministic_pass is False
    assert result.failure_reason == "byte mismatch"


def test_duplicate_case_rejection() -> None:
    case_a = normalize_replay_case(_case("dup", b"a"))
    case_b = normalize_replay_case(_case("dup", b"b"))

    with pytest.raises(ValueError, match="duplicate case IDs"):
        run_replay_battery((case_a, case_b), replay_fn=_replay_ok)


def test_ordering_independence() -> None:
    case_a = normalize_replay_case(_case("b", b"payload-b"))
    case_b = normalize_replay_case(_case("a", b"payload-a"))

    report_1 = run_replay_battery((case_a, case_b), replay_fn=_replay_ok)
    report_2 = run_replay_battery((case_b, case_a), replay_fn=_replay_ok)

    assert tuple(result.case_id for result in report_1.results) == ("a", "b")
    assert report_1 == report_2


def test_receipt_stability() -> None:
    cases = (_case("a", b"one"), _case("b", b"two"))

    report_a, receipt_a = compile_replay_battery(cases, replay_fn=_replay_ok)
    report_b, receipt_b = compile_replay_battery(cases, replay_fn=_replay_ok)

    assert report_a == report_b
    assert receipt_a == receipt_b
    assert receipt_a.validation_passed is True


def test_stable_battery_hash() -> None:
    report, _ = compile_replay_battery((_case("a", b"one"),), replay_fn=_replay_ok)

    assert stable_replay_battery_hash(report) == stable_replay_battery_hash(report)


def test_defensive_copy_behavior() -> None:
    raw = _case("a", b"seed")
    report_before, _ = compile_replay_battery((raw,), replay_fn=_replay_ok)

    raw["case_id"] = "mutated"
    raw["canonical_input_bytes"] = b"changed"
    report_after, _ = compile_replay_battery((_case("a", b"seed"),), replay_fn=_replay_ok)

    assert report_before == report_after


def test_schema_rejection() -> None:
    raw = _case("a", b"payload")
    raw["schema_version"] = "v0"

    with pytest.raises(ValueError, match="unsupported schema_version"):
        normalize_replay_case(raw)


def test_replay_case_hash_validation() -> None:
    """Ensure that malformed hash fields are rejected even when passing a ReplayCase instance."""
    invalid_case = ReplayCase(
        case_id="test",
        experiment_hash="invalid",  # Not a valid 64-char hex SHA-256
        evidence_hash="b" * 64,
        audit_hash="c" * 64,
        canonical_input_bytes=b"payload",
        expected_output_hash="d" * 64,
        expected_verdict="pass",
        schema_version=SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="experiment_hash must be 64-char lowercase hex SHA-256"):
        normalize_replay_case(invalid_case)


def test_callable_leakage_rejected() -> None:
    report = ReplayBatteryReport(
        battery_id="x" * 64,
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        pass_rate=1.0,
        results=(),
        schema_version=SCIENTIFIC_REPLAY_BATTERY_SCHEMA_VERSION,
    )

    payload = report.to_dict()
    payload["leak"] = lambda: None

    with pytest.raises(ValueError, match="callable leakage into serialization"):
        # Uses the module's canonical serializer by round-tripping through constructor path.
        from qec.analysis.scientific_replay_battery import _canonical_json

        _canonical_json(payload)
