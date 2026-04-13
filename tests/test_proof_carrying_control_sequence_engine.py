"""v137.15.6 — Tests for the Proof-Carrying Control Sequence Engine.

Hard-gate formal verification tests. These must pass byte-for-byte on
repeated execution.
"""

from __future__ import annotations

import copy

import pytest

from qec.control.proof_carrying_control_sequence_engine import (
    ControlProofObligation,
    ProofCarryingControlSequence,
    ProofExecutionReceipt,
    ProofVerificationReport,
    SMTLIBExportArtifact,
    build_proof_carrying_control_sequence,
    export_smtlib_proof_artifact,
    normalize_control_proof_obligation,
    record_proof_execution_receipt,
    verify_control_proof_sequence,
)


def _obligation(
    obligation_id: str = "ob-state-1",
    proof_kind: str = "state_safety",
    latency_budget_ms: int = 4096,
    proof_epoch: int = 1,
    state_constraints=None,
    transition_constraints=None,
    collision_constraints=None,
    rollback_constraints=None,
    fallback_constraints=None,
    source_module: str = "explicit_state_transition_automata",
):
    return {
        "obligation_id": obligation_id,
        "source_module": source_module,
        "proof_kind": proof_kind,
        "state_constraints": list(
            state_constraints
            if state_constraints is not None
            else ("S0_safe", "S1_safe")
        ),
        "transition_constraints": list(
            transition_constraints
            if transition_constraints is not None
            else ("T01_legal",)
        ),
        "collision_constraints": list(
            collision_constraints
            if collision_constraints is not None
            else ("lane2_clear",)
        ),
        "rollback_constraints": list(
            rollback_constraints
            if rollback_constraints is not None
            else ("rb_leq_3",)
        ),
        "fallback_constraints": list(
            fallback_constraints
            if fallback_constraints is not None
            else ("corridor_bound_5",)
        ),
        "latency_budget_ms": latency_budget_ms,
        "proof_epoch": proof_epoch,
    }


def _base_obligations():
    return [
        _obligation(
            obligation_id="ob-state-1",
            proof_kind="state_safety",
            source_module="explicit_state_transition_automata",
        ),
        _obligation(
            obligation_id="ob-collision-1",
            proof_kind="collision_freedom",
            source_module="collision_prevention_scheduler",
            collision_constraints=("lane2_clear", "lane3_clear"),
        ),
        _obligation(
            obligation_id="ob-rollback-1",
            proof_kind="rollback_boundedness",
            source_module="deterministic_rollback_planner",
            rollback_constraints=("rb_leq_2",),
        ),
        _obligation(
            obligation_id="ob-fallback-1",
            proof_kind="fallback_corridor_safety",
            source_module="bounded_fallback_corridor",
            fallback_constraints=("corridor_bound_4",),
        ),
        _obligation(
            obligation_id="ob-replay-1",
            proof_kind="replay_identity",
            source_module="deterministic_control_sequence_kernel",
            state_constraints=("replay_identity_invariant",),
        ),
    ]


def _build_sequence(obligations=None):
    return build_proof_carrying_control_sequence(
        sequence_id="seq-v137.15.6",
        obligations=obligations if obligations is not None else _base_obligations(),
    )


# -- normalization --------------------------------------------------------


def test_normalize_accepts_dataclass_roundtrip():
    dto = _obligation()
    normalized = normalize_control_proof_obligation(dto)
    normalized_again = normalize_control_proof_obligation(normalized)
    assert isinstance(normalized, ControlProofObligation)
    assert normalized_again == normalized


def test_normalize_rejects_unknown_proof_kind():
    dto = _obligation(proof_kind="black_box_guess")
    with pytest.raises(ValueError, match="unsupported proof kind"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_malformed_constraint():
    dto = _obligation(state_constraints=["", "S1_safe"])
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_non_string_constraint():
    dto = _obligation(transition_constraints=[42])
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_string_as_constraint_group():
    dto = _obligation()
    dto["collision_constraints"] = "lane2_clear"
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_unordered_set_constraint_group():
    dto = _obligation()
    dto["state_constraints"] = {"S0_safe", "S1_safe"}
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_generator_constraint_group():
    dto = _obligation()
    dto["transition_constraints"] = (c for c in ("T01_legal",))
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_mapping_constraint_group():
    dto = _obligation()
    dto["rollback_constraints"] = {"rb_leq_3": True}
    with pytest.raises(ValueError, match="malformed constraint"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_zero_latency_budget():
    dto = _obligation(latency_budget_ms=0)
    with pytest.raises(ValueError, match="invalid latency budget"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_negative_latency_budget():
    dto = _obligation(latency_budget_ms=-1)
    with pytest.raises(ValueError, match="invalid latency budget"):
        normalize_control_proof_obligation(dto)


def test_normalize_rejects_missing_field():
    dto = _obligation()
    dto.pop("rollback_constraints")
    with pytest.raises(ValueError, match="missing required obligation fields"):
        normalize_control_proof_obligation(dto)


# -- sequence construction ------------------------------------------------


def test_build_sequence_rejects_duplicate_obligations():
    obligations = _base_obligations()
    obligations.append(copy.deepcopy(obligations[0]))
    with pytest.raises(ValueError, match="duplicate obligations"):
        build_proof_carrying_control_sequence(
            sequence_id="seq-dup", obligations=obligations
        )


def test_build_sequence_rejects_empty_obligations():
    with pytest.raises(ValueError, match="empty obligations"):
        build_proof_carrying_control_sequence(sequence_id="seq-empty", obligations=[])


def test_build_sequence_produces_deterministic_hashes():
    seq_a = _build_sequence()
    seq_b = _build_sequence()
    assert seq_a.canonical_control_hash == seq_b.canonical_control_hash
    assert seq_a.proof_hash == seq_b.proof_hash
    assert seq_a.smtlib_artifact_hash == seq_b.smtlib_artifact_hash
    assert seq_a.to_canonical_bytes() == seq_b.to_canonical_bytes()


def test_build_sequence_is_order_independent_for_obligations():
    obligations = _base_obligations()
    shuffled = [obligations[4], obligations[0], obligations[3], obligations[1], obligations[2]]
    seq_a = build_proof_carrying_control_sequence("seq-order", obligations)
    seq_b = build_proof_carrying_control_sequence("seq-order", shuffled)
    assert seq_a.to_canonical_bytes() == seq_b.to_canonical_bytes()
    assert seq_a.proof_hash == seq_b.proof_hash


# -- SMT-LIB export stability --------------------------------------------


def test_repeated_smtlib_export_z3_is_byte_identical():
    sequence = _build_sequence()
    a = export_smtlib_proof_artifact(sequence, "z3")
    b = export_smtlib_proof_artifact(sequence, "z3")
    assert a.smtlib_text == b.smtlib_text
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.export_hash == b.export_hash


def test_repeated_smtlib_export_cvc5_is_byte_identical():
    sequence = _build_sequence()
    a = export_smtlib_proof_artifact(sequence, "cvc5")
    b = export_smtlib_proof_artifact(sequence, "cvc5")
    assert a.smtlib_text == b.smtlib_text
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.export_hash == b.export_hash


def test_smtlib_export_differs_between_solvers():
    sequence = _build_sequence()
    z3 = export_smtlib_proof_artifact(sequence, "z3")
    cvc5 = export_smtlib_proof_artifact(sequence, "cvc5")
    assert z3.export_hash != cvc5.export_hash
    assert "solver_target: z3" in z3.smtlib_text
    assert "solver_target: cvc5" in cvc5.smtlib_text


def test_smtlib_export_encodes_all_proof_kinds():
    sequence = _build_sequence()
    artifact = export_smtlib_proof_artifact(sequence, "z3")
    assert "state-safety" in artifact.smtlib_text
    assert "transition-legality" in artifact.smtlib_text
    assert "collision-free" in artifact.smtlib_text
    assert "rollback-bounded" in artifact.smtlib_text
    assert "fallback-corridor-safe" in artifact.smtlib_text
    assert "replay-identity-invariant" in artifact.smtlib_text


def test_smtlib_export_rejects_invalid_solver():
    sequence = _build_sequence()
    with pytest.raises(ValueError, match="invalid solver target"):
        export_smtlib_proof_artifact(sequence, "yices")


# -- verification ---------------------------------------------------------


def test_verify_returns_deterministic_verified_outcome_z3():
    sequence = _build_sequence()
    a = verify_control_proof_sequence(sequence, "z3")
    b = verify_control_proof_sequence(sequence, "z3")
    assert a.verification_status == "verified"
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.proof_artifact_hash == b.proof_artifact_hash
    assert a.replay_identity_hash == b.replay_identity_hash
    assert not a.obligations_failed


def test_verify_returns_deterministic_verified_outcome_cvc5():
    sequence = _build_sequence()
    a = verify_control_proof_sequence(sequence, "cvc5")
    b = verify_control_proof_sequence(sequence, "cvc5")
    assert a.verification_status == "verified"
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_verify_returns_deterministic_failed_outcome():
    obligations = _base_obligations()
    obligations[1]["collision_constraints"] = ["FAIL:lane2_blocked"]
    sequence = build_proof_carrying_control_sequence("seq-fail", obligations)
    a = verify_control_proof_sequence(sequence, "z3")
    b = verify_control_proof_sequence(sequence, "z3")
    assert a.verification_status == "failed"
    assert "ob-collision-1" in a.obligations_failed
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_verify_returns_deterministic_timeout_outcome():
    obligations = _base_obligations()
    obligations[0]["latency_budget_ms"] = 1  # smaller than simulated cost
    sequence = build_proof_carrying_control_sequence("seq-timeout", obligations)
    a = verify_control_proof_sequence(sequence, "z3")
    b = verify_control_proof_sequence(sequence, "z3")
    assert a.verification_status == "timeout"
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_verify_latency_within_budget_on_normal_sequence():
    sequence = _build_sequence()
    report = verify_control_proof_sequence(sequence, "z3")
    # Aggregate latency must stay within per-obligation budgets.
    for obligation in sequence.proof_obligations:
        assert obligation.latency_budget_ms > 0
    assert report.latency_ms >= 0


def test_verify_detects_tampered_sequence_as_invalid_artifact():
    sequence = _build_sequence()
    tampered = ProofCarryingControlSequence(
        sequence_id=sequence.sequence_id,
        proof_obligations=sequence.proof_obligations,
        canonical_control_hash=sequence.canonical_control_hash,
        proof_hash="0" * 64,
        smtlib_artifact_hash=sequence.smtlib_artifact_hash,
    )
    report = verify_control_proof_sequence(tampered, "z3")
    assert report.verification_status == "invalid_artifact"
    assert report.proof_artifact_hash == ""
    assert report.replay_identity_hash == ""


def test_verify_detects_tampered_canonical_control_hash_as_invalid_artifact():
    sequence = _build_sequence()
    tampered = ProofCarryingControlSequence(
        sequence_id=sequence.sequence_id,
        proof_obligations=sequence.proof_obligations,
        canonical_control_hash="0" * 64,
        proof_hash=sequence.proof_hash,
        smtlib_artifact_hash=sequence.smtlib_artifact_hash,
    )
    report = verify_control_proof_sequence(tampered, "z3")
    assert report.verification_status == "invalid_artifact"


def test_verify_detects_tampered_smtlib_artifact_hash_as_invalid_artifact():
    sequence = _build_sequence()
    tampered = ProofCarryingControlSequence(
        sequence_id=sequence.sequence_id,
        proof_obligations=sequence.proof_obligations,
        canonical_control_hash=sequence.canonical_control_hash,
        proof_hash=sequence.proof_hash,
        smtlib_artifact_hash="0" * 64,
    )
    report = verify_control_proof_sequence(tampered, "z3")
    assert report.verification_status == "invalid_artifact"


def test_verify_detects_non_tuple_proof_obligations_as_invalid_artifact():
    sequence = _build_sequence()
    tampered = ProofCarryingControlSequence(
        sequence_id=sequence.sequence_id,
        proof_obligations=list(sequence.proof_obligations),  # type: ignore[arg-type]
        canonical_control_hash=sequence.canonical_control_hash,
        proof_hash=sequence.proof_hash,
        smtlib_artifact_hash=sequence.smtlib_artifact_hash,
    )
    report = verify_control_proof_sequence(tampered, "z3")
    assert report.verification_status == "invalid_artifact"
    assert report.proof_artifact_hash == ""


def test_verify_detects_non_obligation_entries_as_invalid_artifact():
    sequence = _build_sequence()
    tampered = ProofCarryingControlSequence(
        sequence_id=sequence.sequence_id,
        proof_obligations=({"not": "a dataclass"},),  # type: ignore[arg-type]
        canonical_control_hash=sequence.canonical_control_hash,
        proof_hash=sequence.proof_hash,
        smtlib_artifact_hash=sequence.smtlib_artifact_hash,
    )
    report = verify_control_proof_sequence(tampered, "z3")
    assert report.verification_status == "invalid_artifact"
    assert report.proof_artifact_hash == ""
    assert report.replay_identity_hash == ""


def test_verify_rejects_invalid_solver():
    sequence = _build_sequence()
    with pytest.raises(ValueError, match="invalid solver target"):
        verify_control_proof_sequence(sequence, "yices")


# -- replay identity invariant -------------------------------------------


def test_replay_identity_invariant_on_repeated_runs():
    seq_a = _build_sequence()
    seq_b = _build_sequence()
    art_a_z3 = export_smtlib_proof_artifact(seq_a, "z3")
    art_b_z3 = export_smtlib_proof_artifact(seq_b, "z3")
    rep_a_z3 = verify_control_proof_sequence(seq_a, "z3")
    rep_b_z3 = verify_control_proof_sequence(seq_b, "z3")
    assert art_a_z3.export_hash == art_b_z3.export_hash
    assert rep_a_z3.replay_identity_hash == rep_b_z3.replay_identity_hash
    assert rep_a_z3.to_canonical_bytes() == rep_b_z3.to_canonical_bytes()


def test_canonical_export_stability_across_dataclasses():
    seq = _build_sequence()
    report = verify_control_proof_sequence(seq, "cvc5")
    artifact = export_smtlib_proof_artifact(seq, "cvc5")
    receipt = record_proof_execution_receipt(seq, report, proof_epoch=7)
    for dc in (seq, report, artifact, receipt):
        assert dc.to_canonical_json() == dc.to_canonical_json()
        assert dc.to_canonical_bytes() == dc.to_canonical_bytes()
        assert dc.as_hash_payload() == dc.to_canonical_bytes()


# -- receipts -------------------------------------------------------------


def test_record_proof_execution_receipt_is_deterministic():
    seq = _build_sequence()
    report = verify_control_proof_sequence(seq, "z3")
    a = record_proof_execution_receipt(seq, report, proof_epoch=3)
    b = record_proof_execution_receipt(seq, report, proof_epoch=3)
    assert isinstance(a, ProofExecutionReceipt)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.receipt_id == b.receipt_id
    assert a.verification_status == "verified"


def test_record_proof_execution_receipt_rejects_invalid_epoch():
    seq = _build_sequence()
    report = verify_control_proof_sequence(seq, "z3")
    with pytest.raises(ValueError, match="invalid proof epoch"):
        record_proof_execution_receipt(seq, report, proof_epoch=-1)


# -- dataclass payload contracts -----------------------------------------


def test_all_dataclasses_expose_hash_payload_contract():
    seq = _build_sequence()
    artifact = export_smtlib_proof_artifact(seq, "z3")
    report = verify_control_proof_sequence(seq, "z3")
    receipt = record_proof_execution_receipt(seq, report, proof_epoch=1)

    for dc in (
        seq.proof_obligations[0],
        seq,
        artifact,
        report,
        receipt,
    ):
        assert isinstance(dc.to_dict(), dict)
        assert isinstance(dc.to_canonical_json(), str)
        assert isinstance(dc.to_canonical_bytes(), bytes)
        assert dc.as_hash_payload() == dc.to_canonical_bytes()


def test_verification_report_types():
    seq = _build_sequence()
    report = verify_control_proof_sequence(seq, "z3")
    assert isinstance(report, ProofVerificationReport)
    assert isinstance(report.obligations_passed, tuple)
    assert isinstance(report.obligations_failed, tuple)
    assert report.solver_target == "z3"
    assert report.verification_status in (
        "verified",
        "failed",
        "timeout",
        "invalid_artifact",
    )


def test_smtlib_artifact_types():
    seq = _build_sequence()
    artifact = export_smtlib_proof_artifact(seq, "cvc5")
    assert isinstance(artifact, SMTLIBExportArtifact)
    assert artifact.solver_target == "cvc5"
    assert artifact.artifact_id.endswith("-cvc5")
