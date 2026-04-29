from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_agent_failure_injection import (
    AdversarialFailureCase,
    AdversarialGovernanceReceipt,
    run_multi_agent_failure_injection,
)


def _h(seed: str) -> str:
    return sha256_hex({"seed": seed})


def _case(
    case_id: str,
    failure_type: str,
    payload: dict[str, object],
    target: str | None = None,
    expected: str = "EXPECTED",
) -> AdversarialFailureCase:
    return AdversarialFailureCase(
        case_id=case_id,
        failure_type=failure_type,
        target_hash=target or _h(case_id),
        payload=payload,
        expected_rejection_reason=expected,
    )


def test_receipt_deterministic_and_frozen() -> None:
    cases = (
        _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}, expected="MALFORMED_DECISION_HASH"),
        _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN", "agent_conflict": True}),
        _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}, expected="MEMORY_HASH_MISMATCH"),
    )
    r1 = run_multi_agent_failure_injection("s", (_h("i0"),), cases)
    r2 = run_multi_agent_failure_injection("s", (_h("i0"),), tuple(reversed(cases)))
    assert r1.stable_hash == r2.stable_hash
    assert r1.to_canonical_json() == r2.to_canonical_json()
    with pytest.raises(FrozenInstanceError):
        r1.status = "X"


def test_reordered_cases_produce_identical_receipt_hash() -> None:
    cases_a = (
        _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}, expected="MALFORMED_DECISION_HASH"),
        _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN"}, expected="UNKNOWN_ROLE"),
        _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}, expected="MEMORY_HASH_MISMATCH"),
    )
    cases_b = tuple(reversed(cases_a))
    receipt_a = run_multi_agent_failure_injection("scenario", (_h("i0"),), cases_a)
    receipt_b = run_multi_agent_failure_injection("scenario", (_h("i0"),), cases_b)
    assert receipt_a.stable_hash == receipt_b.stable_hash


def test_invalid_decision_variants_detected_rejected() -> None:
    receipt = run_multi_agent_failure_injection(
        "s",
        (_h("i0"),),
        (
            _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}, expected="MALFORMED_DECISION_HASH"),
            _case("d2", "INVALID_DECISION", {"decision_hash": _h("d2"), "status": "INVALID"}, expected="INVALID_DECISION_STATUS"),
            _case("d3", "INVALID_DECISION", {"decision_hash": _h("d3"), "status": "ACCEPT", "score": True}, expected="INVALID_DECISION_SCORE"),
            _case("d4", "INVALID_DECISION", {"decision_hash": _h("d4"), "status": "ACCEPT", "conflicting_identity": True}, expected="CONFLICTING_DECISION_IDENTITY"),
            _case("d5", "INVALID_DECISION", {"decision_hash": _h("d5"), "status": "ACCEPT", "score": 1.1}, expected="INVALID_DECISION_SCORE"),
        ),
    )
    assert all(result.detected and result.rejected for result in receipt.failure_results)


def test_role_and_memory_variants_detected_rejected() -> None:
    receipt = run_multi_agent_failure_injection(
        "s",
        (_h("i0"),),
        (
            _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN"}, expected="UNKNOWN_ROLE"),
            _case("r2", "CONFLICTING_ROLE", {"role": "CONTROL", "agent_conflict": True}, expected="AGENT_ROLE_CONFLICT"),
            _case("r3", "CONFLICTING_ROLE", {"role": "CONTROL", "role_decision_conflict": True}, expected="ROLE_DECISION_CONFLICT"),
            _case("r4", "CONFLICTING_ROLE", {"role": "CONTROL", "agent_multi_role_conflict": True}, expected="AGENT_MULTI_ROLE_CONFLICT"),
            _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}, expected="MEMORY_HASH_MISMATCH"),
            _case("m2", "INCONSISTENT_MEMORY", {"duplicate_key_conflict": True}, expected="MEMORY_KEY_CONFLICT"),
            _case("m3", "INCONSISTENT_MEMORY", {"lineage_mismatch": True}, expected="MEMORY_LINEAGE_MISMATCH"),
            _case("m4", "INCONSISTENT_MEMORY", {"missing_lineage_hash": True}, expected="MISSING_LINEAGE_HASH"),
            _case("m5", "INCONSISTENT_MEMORY", {"unknown_decision_reference": True}, expected="UNKNOWN_DECISION_REFERENCE"),
        ),
    )
    assert receipt.status == "VALIDATED"
    assert receipt.accepted_invalid_count == 0
    assert receipt.detected_count == len(receipt.failure_results)
    assert receipt.rejected_count == len(receipt.failure_results)


def test_validation_failures() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("", "INVALID_DECISION", {"decision_hash": "bad"})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("x", "UNKNOWN", {"a": 1})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("x", "INVALID_DECISION", {"a": float("nan")})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("x", "INVALID_DECISION", {"a": ""})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("x", "INVALID_DECISION", {})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_multi_agent_failure_injection("s", ("bad",), (_case("x", "INCONSISTENT_MEMORY", {"hash_mismatch": True}),))
    dup = (
        _case("x", "INCONSISTENT_MEMORY", {"hash_mismatch": True}, target=_h("t0")),
        _case("x", "CONFLICTING_ROLE", {"role": "UNKNOWN"}, target=_h("t1")),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_multi_agent_failure_injection("s", (_h("i0"),), dup)
    ambiguous = (
        _case("x1", "INCONSISTENT_MEMORY", {"a": 1, "b": 2}, target=_h("t0")),
        _case("x2", "INCONSISTENT_MEMORY", {"b": 2, "a": 1}, target=_h("t0")),
    )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run_multi_agent_failure_injection("s", (_h("i0"),), ambiguous)


def test_rejects_non_string_payload_keys() -> None:
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        _case("k1", "INVALID_DECISION", {1: "x"})  # type: ignore[arg-type]


def test_payload_is_deeply_immutable() -> None:
    case = _case("imm", "INCONSISTENT_MEMORY", {"nested": {"x": 1}})
    with pytest.raises(TypeError):
        case.payload["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        case.payload["nested"]["x"] = 2  # type: ignore[index]


def test_receipt_enforces_sorted_case_and_result_ordering() -> None:
    c1 = _case("a", "CONFLICTING_ROLE", {"role": "UNKNOWN"}, expected="UNKNOWN_ROLE")
    c2 = _case("b", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}, expected="MALFORMED_DECISION_HASH")
    valid = run_multi_agent_failure_injection("s", (_h("i0"),), (c1, c2))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AdversarialGovernanceReceipt(
            version=valid.version,
            scenario_id=valid.scenario_id,
            input_hashes=valid.input_hashes,
            failure_cases=tuple(reversed(valid.failure_cases)),
            failure_results=valid.failure_results,
            detected_count=valid.detected_count,
            rejected_count=valid.rejected_count,
            accepted_invalid_count=valid.accepted_invalid_count,
            status=valid.status,
            stable_hash=valid.stable_hash,
        )
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AdversarialGovernanceReceipt(
            version=valid.version,
            scenario_id=valid.scenario_id,
            input_hashes=valid.input_hashes,
            failure_cases=valid.failure_cases,
            failure_results=tuple(reversed(valid.failure_results)),
            detected_count=valid.detected_count,
            rejected_count=valid.rejected_count,
            accepted_invalid_count=valid.accepted_invalid_count,
            status=valid.status,
            stable_hash=valid.stable_hash,
        )


def test_validated_status_invariant_lock() -> None:
    r = run_multi_agent_failure_injection(
        "s",
        (_h("i0"),),
        (
            _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}, expected="MEMORY_HASH_MISMATCH"),
        ),
    )
    if r.status == "VALIDATED":
        assert r.accepted_invalid_count == 0
        assert r.detected_count == len(r.failure_results)
