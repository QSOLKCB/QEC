from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.policy_decision_evidence_kernel import (
    EvidenceReceipt,
    append_evidence_receipt,
    compute_evidence_root,
    create_policy_decision_artifact,
    export_policy_evidence_bytes,
    generate_decision_lineage_receipt,
    verify_policy_evidence_chain,
)


def _h(ch: str) -> str:
    return ch * 64


def _make_artifact():
    return create_policy_decision_artifact(
        policy_id="policy.alpha",
        decision_verdict="ALLOW",
        supporting_evidence_hashes=(_h("c"), _h("a"), _h("b")),
        parent_provenance_root=_h("1"),
        originating_sovereignty_event_hash=_h("2"),
        originating_privilege_decision_hash=_h("3"),
    )


def _build_receipts():
    artifact = _make_artifact()
    receipts: tuple[EvidenceReceipt, ...] = ()
    for evidence_hash in artifact.supporting_evidence_hashes:
        receipts = append_evidence_receipt(artifact, receipts, evidence_hash=evidence_hash)
    return artifact, receipts


def test_repeated_run_determinism_and_identical_bytes() -> None:
    artifact1, receipts1 = _build_receipts()
    artifact2, receipts2 = _build_receipts()

    assert artifact1 == artifact2
    assert receipts1 == receipts2
    assert export_policy_evidence_bytes(artifact1, receipts1) == export_policy_evidence_bytes(artifact2, receipts2)


def test_stable_evidence_root_and_receipt_stability() -> None:
    _, receipts1 = _build_receipts()
    _, receipts2 = _build_receipts()

    assert compute_evidence_root(receipts1) == compute_evidence_root(receipts2)
    assert generate_decision_lineage_receipt(*_build_receipts()) == generate_decision_lineage_receipt(*_build_receipts())


def test_append_only_chain_enforcement() -> None:
    artifact = _make_artifact()
    receipts: tuple[EvidenceReceipt, ...] = ()

    receipts = append_evidence_receipt(artifact, receipts, evidence_hash=artifact.supporting_evidence_hashes[0])
    assert len(receipts) == 1
    with pytest.raises(ValueError, match="append order"):
        append_evidence_receipt(artifact, receipts, evidence_hash=artifact.supporting_evidence_hashes[2])


def test_tamper_rejection_on_receipt_and_artifact() -> None:
    artifact, receipts = _build_receipts()

    tampered_receipts = receipts[:-1] + (replace(receipts[-1], evidence_hash=_h("d")),)
    with pytest.raises(ValueError, match="receipt evidence hash mismatch"):
        verify_policy_evidence_chain(artifact, tampered_receipts)

    tampered_artifact = replace(artifact, stable_decision_hash=_h("9"))
    with pytest.raises(ValueError, match="stable_decision_hash mismatch"):
        verify_policy_evidence_chain(tampered_artifact, receipts)


def test_replay_fidelity_same_input_same_bytes() -> None:
    artifact, receipts = _build_receipts()
    exported_a = export_policy_evidence_bytes(artifact, receipts)
    exported_b = export_policy_evidence_bytes(artifact, receipts)

    assert exported_a == exported_b
    assert verify_policy_evidence_chain(artifact, receipts) is True


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="policy_id must be non-empty"):
        create_policy_decision_artifact(
            policy_id="  ",
            decision_verdict="ALLOW",
            supporting_evidence_hashes=(_h("a"),),
            parent_provenance_root=_h("1"),
            originating_sovereignty_event_hash=_h("2"),
            originating_privilege_decision_hash=_h("3"),
        )

    with pytest.raises(ValueError, match="must be unique"):
        create_policy_decision_artifact(
            policy_id="policy.alpha",
            decision_verdict="ALLOW",
            supporting_evidence_hashes=(_h("a"), _h("a")),
            parent_provenance_root=_h("1"),
            originating_sovereignty_event_hash=_h("2"),
            originating_privilege_decision_hash=_h("3"),
        )

    artifact = _make_artifact()
    with pytest.raises(ValueError, match="not declared by artifact"):
        append_evidence_receipt(artifact, (), evidence_hash=_h("d"))

    with pytest.raises(ValueError, match="receipt count exceeds"):
        verify_policy_evidence_chain(artifact, (EvidenceReceipt(
            index=0,
            schema_version=1,
            parent_receipt_hash=_h("0"),
            policy_decision_hash=artifact.stable_decision_hash,
            evidence_hash=artifact.supporting_evidence_hashes[0],
            evidence_position=0,
            receipt_hash=_h("e"),
        ),) * 4)
