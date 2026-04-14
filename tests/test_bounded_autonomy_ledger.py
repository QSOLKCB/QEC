from __future__ import annotations

import json

from qec.orchestration.bounded_autonomy_ledger import (
    GENESIS_CONTINUITY_HASH,
    LEDGER_VERSION,
    AutonomyLedgerEntry,
    AutonomyLedgerSnapshot,
    BoundedAutonomyLedger,
    append_autonomy_ledger_entry,
    build_autonomy_ledger_receipt,
    compare_autonomy_ledger_replay,
    summarize_autonomy_ledger,
    validate_bounded_autonomy_ledger,
)


def _h(seed: str) -> str:
    return (seed * 64)[:64]


def _decision(idx: int, replay: str = "agent-a", decision: str = "allow") -> dict:
    return {
        "replay_identity": replay,
        "transition_id": f"t-{idx}",
        "boundary_report_hash": _h(f"a{idx % 10}"),
        "firewall_decision_hash": _h(f"b{idx % 10}"),
        "decision": decision,
        "rule_hit_reasons": ("r2", "r1", "r1"),
    }


def _build(n: int = 3, replay: str = "agent-a") -> BoundedAutonomyLedger:
    ledger = None
    for idx in range(n):
        ledger, _ = append_autonomy_ledger_entry(_decision(idx, replay=replay, decision="allow" if idx % 2 == 0 else "deny"), ledger)
    assert ledger is not None
    return ledger


def test_deterministic_repeated_append() -> None:
    ledger_a = None
    ledger_b = None
    for idx in range(4):
        ledger_a, _ = append_autonomy_ledger_entry(_decision(idx), ledger_a)
        ledger_b, _ = append_autonomy_ledger_entry(_decision(idx), ledger_b)
    assert ledger_a == ledger_b


def test_stable_hash_reproducibility() -> None:
    hashes = {_build(5).stable_hash() for _ in range(5)}
    assert len(hashes) == 1


def test_continuity_validation_detects_break() -> None:
    ledger = _build(3)
    bad_entry = AutonomyLedgerEntry(
        **{**ledger.entries[1].to_dict(), "prior_continuity_hash": GENESIS_CONTINUITY_HASH}
    )
    bad = BoundedAutonomyLedger(
        ledger_version=ledger.ledger_version,
        entries=(ledger.entries[0], bad_entry, ledger.entries[2]),
        receipt_chain=ledger.receipt_chain,
        ledger_hash=ledger.ledger_hash,
    )
    report = validate_bounded_autonomy_ledger(bad)
    assert "continuity break" in report["violations"]


def test_duplicate_transition_id_detected() -> None:
    ledger = _build(2)
    bad_dict = ledger.to_dict()
    bad_dict["entries"][1]["transition_id"] = bad_dict["entries"][0]["transition_id"]
    report = validate_bounded_autonomy_ledger(bad_dict)
    assert "duplicate transition_id" in report["violations"]


def test_duplicate_decision_hash_detected() -> None:
    ledger = _build(2)
    bad_dict = ledger.to_dict()
    bad_dict["entries"][1]["firewall_decision_hash"] = bad_dict["entries"][0]["firewall_decision_hash"]
    report = validate_bounded_autonomy_ledger(bad_dict)
    assert "duplicate decision hash" in report["violations"]


def test_missing_receipt_chain_detected() -> None:
    ledger = _build(2)
    bad = {**ledger.to_dict(), "receipt_chain": [ledger.receipt_chain[0]]}
    report = validate_bounded_autonomy_ledger(bad)
    assert "missing receipt chain" in report["violations"]


def test_malformed_entry_ordering_detected() -> None:
    ledger = _build(3)
    bad = ledger.to_dict()
    bad["entries"][2]["entry_index"] = 9
    report = validate_bounded_autonomy_ledger(bad)
    assert "malformed entry ordering" in report["violations"]


def test_ledger_hash_drift_detected() -> None:
    ledger = _build(2)
    bad = {**ledger.to_dict(), "ledger_hash": _h("f")}
    report = validate_bounded_autonomy_ledger(bad)
    assert "ledger hash drift" in report["violations"]


def test_replay_comparison_stability() -> None:
    ledger_a = _build(4)
    ledger_b = _build(4)
    cmp = compare_autonomy_ledger_replay(ledger_a, ledger_b)
    assert cmp["replay_stable"] is True
    assert cmp["violations"] == ()


def test_replay_identity_mismatch_detected() -> None:
    ledger_a = _build(2, replay="agent-a")
    ledger_b = _build(2, replay="agent-b")
    cmp = compare_autonomy_ledger_replay(ledger_a, ledger_b)
    assert "replay identity mismatch" in cmp["violations"]


def test_canonical_json_round_trip() -> None:
    ledger = _build(3)
    payload = ledger.to_canonical_json()
    parsed = json.loads(payload)
    rebuilt = BoundedAutonomyLedger(
        ledger_version=parsed["ledger_version"],
        entries=tuple(AutonomyLedgerEntry(**entry) for entry in parsed["entries"]),
        receipt_chain=tuple(parsed["receipt_chain"]),
        ledger_hash=parsed["ledger_hash"],
    )
    assert rebuilt.to_canonical_json() == payload


def test_validator_never_raises_on_malformed_input() -> None:
    report = validate_bounded_autonomy_ledger({"entries": "not-a-sequence"})
    assert report["is_valid"] is False
    assert report["violations"]


def test_summary_deterministic_ordering() -> None:
    ledger = _build(3)
    summary_a = summarize_autonomy_ledger(ledger)
    summary_b = summarize_autonomy_ledger(ledger)
    assert summary_a == summary_b
    assert summary_a["rule_hit_reason_counts"] == (("r1", 3), ("r2", 3))


def test_receipt_chain_reproducible() -> None:
    ledger = _build(4)
    receipt_a = build_autonomy_ledger_receipt(ledger)
    receipt_b = build_autonomy_ledger_receipt(ledger)
    assert receipt_a == receipt_b
    assert receipt_a.receipt_hash == ledger.receipt_chain[-1]


def test_snapshot_hash_stability() -> None:
    ledger = _build(3)
    snapshot_a = AutonomyLedgerSnapshot(
        ledger_version=LEDGER_VERSION,
        entry_count=len(ledger.entries),
        continuity_chain=tuple(entry.continuity_hash for entry in ledger.entries),
        receipt_chain=ledger.receipt_chain,
        ledger_hash=ledger.ledger_hash,
        snapshot_hash="0" * 64,
    )
    snapshot_b = AutonomyLedgerSnapshot(**snapshot_a.to_dict())
    assert snapshot_a.to_canonical_json() == snapshot_b.to_canonical_json()


def test_dataclass_hash_methods() -> None:
    ledger = _build(1)
    entry = ledger.entries[0]
    assert entry.stable_hash() == entry.continuity_hash
    assert ledger.stable_hash() == ledger.ledger_hash


def test_allow_decision_inconsistency_in_raw_dict_rejected() -> None:
    ledger = _build(1)
    bad_dict = ledger.to_dict()
    bad_dict["entries"][0]["allow"] = not bad_dict["entries"][0]["allow"]
    report = validate_bounded_autonomy_ledger(bad_dict)
    assert report["is_valid"] is False
    assert report["violations"]


def test_validator_detects_allow_decision_inconsistency() -> None:
    ledger = _build(1)
    entry = ledger.entries[0]
    bad_entry = AutonomyLedgerEntry(
        entry_index=entry.entry_index,
        replay_identity=entry.replay_identity,
        transition_id=entry.transition_id,
        boundary_report_hash=entry.boundary_report_hash,
        firewall_decision_hash=entry.firewall_decision_hash,
        allow=not entry.allow,
        decision=entry.decision,
        rule_hit_reasons=entry.rule_hit_reasons,
        prior_continuity_hash=entry.prior_continuity_hash,
        continuity_hash=entry.continuity_hash,
    )
    bad = BoundedAutonomyLedger(
        ledger_version=ledger.ledger_version,
        entries=(bad_entry,),
        receipt_chain=ledger.receipt_chain,
        ledger_hash=ledger.ledger_hash,
    )
    report = validate_bounded_autonomy_ledger(bad)
    assert "allow/decision inconsistency" in report["violations"]


def test_replay_comparison_asymmetric_normalization_failure() -> None:
    ledger_a = _build(2)
    bad_dict = {"entries": "not-a-sequence", "ledger_version": LEDGER_VERSION}
    cmp = compare_autonomy_ledger_replay(ledger_a, bad_dict)
    assert cmp["replay_stable"] is False
    assert cmp["ledger_hash_match"] is False
    assert cmp["receipt_chain_match"] is False
    assert cmp["replay_identity_match"] is False
