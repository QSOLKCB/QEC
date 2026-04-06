from __future__ import annotations

import pytest

from qec.analysis.replay_certification_battery import (
    certify_chain_integrity,
    export_certification_bytes,
    generate_certification_receipt,
    run_replay_certification_battery,
    verify_byte_identity,
)


def _sovereignty_events() -> tuple[dict[str, object], ...]:
    return (
        {"event_id": "sev-2", "scope": "ops", "decision": "maintain"},
        {"event_id": "sev-1", "scope": "core", "decision": "lock"},
    )


def _capability_decisions() -> tuple[dict[str, object], ...]:
    return (
        {"decision_id": "cap-2", "capability": "repair", "allowed": False},
        {"decision_id": "cap-1", "capability": "audit", "allowed": True},
    )


def _provenance_chain() -> tuple[dict[str, object], ...]:
    return (
        {"link_id": "prov-1", "payload": {"step": "origin", "proof": "a"}},
        {"link_id": "prov-2", "payload": {"step": "handoff", "proof": "b"}},
    )


def _policy_evidence() -> tuple[dict[str, object], ...]:
    return (
        {"evidence_id": "pol-2", "policy": "integrity", "status": "pass"},
        {"evidence_id": "pol-1", "policy": "safety", "status": "pass"},
    )


def test_repeated_run_determinism() -> None:
    report_a = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    report_b = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    assert report_a == report_b


def test_identical_inputs_produce_identical_certification_bytes() -> None:
    report_a = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    report_b = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    assert export_certification_bytes(report_a) == export_certification_bytes(report_b)


def test_stable_certification_root() -> None:
    report_a = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    report_b = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    assert report_a.certification_root_hash == report_b.certification_root_hash
    assert len(report_a.certification_root_hash) == 64


def test_tamper_rejection_on_chain_parent_hash() -> None:
    tampered_chain = (
        {"link_id": "prov-1", "parent_hash": "0" * 64, "payload": {"step": "origin"}},
    )
    with pytest.raises(ValueError):
        certify_chain_integrity(tampered_chain)


def test_byte_identity_enforcement() -> None:
    with pytest.raises(ValueError):
        verify_byte_identity(b"abc", b"abd")


def test_certification_receipt_stability() -> None:
    report = run_replay_certification_battery(
        _sovereignty_events(),
        _capability_decisions(),
        _provenance_chain(),
        _policy_evidence(),
    )
    receipt_a = generate_certification_receipt(report)
    receipt_b = generate_certification_receipt(report)
    assert receipt_a == receipt_b
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError):
        run_replay_certification_battery((), _capability_decisions(), _provenance_chain(), _policy_evidence())
    with pytest.raises(ValueError):
        run_replay_certification_battery(_sovereignty_events(), (), _provenance_chain(), _policy_evidence())
    with pytest.raises(ValueError):
        run_replay_certification_battery(_sovereignty_events(), _capability_decisions(), (), _policy_evidence())
    with pytest.raises(ValueError):
        run_replay_certification_battery(_sovereignty_events(), _capability_decisions(), _provenance_chain(), ())
