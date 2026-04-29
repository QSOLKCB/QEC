from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.multi_agent_failure_injection import (
    AdversarialFailureCase,
    run_multi_agent_failure_injection,
)


def _h(seed: str) -> str:
    return sha256_hex({"seed": seed})


def _case(case_id: str, failure_type: str, payload: dict[str, object], target: str | None = None) -> AdversarialFailureCase:
    return AdversarialFailureCase(
        case_id=case_id,
        failure_type=failure_type,
        target_hash=target or _h(case_id),
        payload=payload,
        expected_rejection_reason="EXPECTED",
    )


def test_receipt_deterministic_and_frozen() -> None:
    cases = (
        _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}),
        _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN", "agent_conflict": True}),
        _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}),
    )
    r1 = run_multi_agent_failure_injection("s", (_h("i0"),), cases)
    r2 = run_multi_agent_failure_injection("s", (_h("i0"),), tuple(reversed(cases)))
    assert r1.stable_hash == r2.stable_hash
    assert r1.to_canonical_json() == r1.to_canonical_json()
    with pytest.raises(FrozenInstanceError):
        r1.status = "X"


def test_reordered_cases_produce_identical_receipt_hash() -> None:
    cases_a = (
        _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}),
        _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN"}),
        _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}),
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
            _case("d1", "INVALID_DECISION", {"decision_hash": "bad", "status": "ACCEPT"}),
            _case("d2", "INVALID_DECISION", {"decision_hash": _h("d2"), "status": "INVALID"}),
            _case("d3", "INVALID_DECISION", {"decision_hash": _h("d3"), "status": "ACCEPT", "score": True}),
            _case("d4", "INVALID_DECISION", {"decision_hash": _h("d4"), "status": "ACCEPT", "conflicting_identity": True}),
            _case("d5", "INVALID_DECISION", {"decision_hash": _h("d5"), "status": "ACCEPT", "score": 1.1}),
        ),
    )
    assert all(result.detected and result.rejected for result in receipt.failure_results)


def test_role_and_memory_variants_detected_rejected() -> None:
    receipt = run_multi_agent_failure_injection(
        "s",
        (_h("i0"),),
        (
            _case("r1", "CONFLICTING_ROLE", {"role": "UNKNOWN"}),
            _case("r2", "CONFLICTING_ROLE", {"role": "CONTROL", "agent_conflict": True}),
            _case("r3", "CONFLICTING_ROLE", {"role": "CONTROL", "role_decision_conflict": True}),
            _case("r4", "CONFLICTING_ROLE", {"role": "CONTROL", "agent_multi_role_conflict": True}),
            _case("m1", "INCONSISTENT_MEMORY", {"hash_mismatch": True}),
            _case("m2", "INCONSISTENT_MEMORY", {"duplicate_key_conflict": True}),
            _case("m3", "INCONSISTENT_MEMORY", {"lineage_mismatch": True}),
            _case("m4", "INCONSISTENT_MEMORY", {"missing_lineage_hash": True}),
            _case("m5", "INCONSISTENT_MEMORY", {"unknown_decision_reference": True}),
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
