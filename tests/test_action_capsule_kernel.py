from __future__ import annotations

import json

import pytest

from qec.analysis.action_capsule_kernel import (
    ActionCapsuleProofReceipt,
    ActionDescriptor,
    ProofCarryingActionCapsule,
    build_action_capsule,
)
from qec.analysis.bounded_refinement_kernel import refine_transition_policy
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.deterministic_transition_policy import TransitionDecision, TransitionPolicyReceipt
from qec.analysis.governed_orchestration_layer import (
    GovernancePolicy,
    evaluate_governed_orchestration,
)


def _transition() -> TransitionPolicyReceipt:
    decision_payload = {
        "selected_ordering_signature": "sig_a",
        "selected_score": round(0.82, 12),
        "decision_rank": 1,
        "margin_to_next": round(0.21, 12),
        "decision_confidence": round(0.86, 12),
        "decision_type": "clear_winner",
    }
    decision = TransitionDecision(
        selected_ordering_signature="sig_a",
        selected_score=0.82,
        decision_rank=1,
        margin_to_next=0.21,
        decision_confidence=0.86,
        decision_type="clear_winner",
        stable_hash=sha256_hex(decision_payload),
    )
    payload = {
        "input_receipt_hash": "1" * 64,
        "candidate_count": 2,
        "selected_decision": decision.to_dict(),
        "selection_rule": "ordered_scores_margin_dominance_v1",
        "classification": "stable_transition",
    }
    return TransitionPolicyReceipt(
        input_receipt_hash="1" * 64,
        candidate_count=2,
        selected_decision=decision,
        selection_rule="ordered_scores_margin_dominance_v1",
        classification="stable_transition",
        stable_hash=sha256_hex(payload),
    )


def _policy() -> GovernancePolicy:
    payload = {
        "min_required_score": round(0.5, 12),
        "min_required_confidence": round(0.5, 12),
        "min_required_margin": round(0.1, 12),
        "min_required_convergence": round(0.1, 12),
        "allow_tie_break": True,
        "allow_no_improvement": True,
        "require_stable_transition": False,
    }
    return GovernancePolicy(
        min_required_score=0.5,
        min_required_confidence=0.5,
        min_required_margin=0.1,
        min_required_convergence=0.1,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
        stable_hash=sha256_hex(payload),
    )


def _triplet():
    transition = _transition()
    refinement = refine_transition_policy(transition)
    governance = evaluate_governed_orchestration(_policy(), transition, refinement)
    return transition, refinement, governance


def test_deterministic_replay_and_cross_instance_determinism() -> None:
    transition, refinement, governance = _triplet()
    capsule1, proof1 = build_action_capsule(transition, refinement, governance)
    capsule2, proof2 = build_action_capsule(transition, refinement, governance)

    assert capsule1.to_canonical_bytes() == capsule2.to_canonical_bytes()
    assert capsule1.to_canonical_json() == capsule2.to_canonical_json()
    assert capsule1.capsule_hash == capsule2.capsule_hash
    assert proof1.proof_receipt_hash == proof2.proof_receipt_hash
    assert capsule1.replay_identity == capsule2.replay_identity


def test_invalid_lineage_rejected() -> None:
    transition, refinement, governance = _triplet()
    bad_refinement = refinement
    object.__setattr__(bad_refinement, "input_policy_hash", "2" * 64)
    payload = {
        "input_policy_hash": bad_refinement.input_policy_hash,
        "steps": tuple(step.to_dict() for step in bad_refinement.steps),
        "final_vector": tuple(round(v, 12) for v in bad_refinement.final_vector),
        "iteration_count": bad_refinement.iteration_count,
        "converged": bad_refinement.converged,
        "convergence_metric": round(bad_refinement.convergence_metric, 12),
        "classification": bad_refinement.classification,
    }
    object.__setattr__(bad_refinement, "stable_hash", sha256_hex(payload))
    with pytest.raises(ValueError, match="input_policy_hash"):
        build_action_capsule(transition, bad_refinement, governance)

    transition2, refinement2, governance2 = _triplet()
    bad_governance = governance2
    object.__setattr__(bad_governance, "input_refinement_hash", "3" * 64)
    g_payload = {
        "policy": bad_governance.policy.to_dict(),
        "input_transition_hash": bad_governance.input_transition_hash,
        "input_refinement_hash": bad_governance.input_refinement_hash,
        "checks": tuple(check.to_dict() for check in bad_governance.checks),
        "verdict": bad_governance.verdict.to_dict(),
    }
    object.__setattr__(bad_governance, "stable_hash", sha256_hex(g_payload))
    with pytest.raises(ValueError, match="input_refinement_hash"):
        build_action_capsule(transition2, refinement2, bad_governance)


def test_governance_not_allow_rejected() -> None:
    transition = _transition()
    refinement = refine_transition_policy(transition)
    hold_policy_payload = {
        "min_required_score": round(0.5, 12),
        "min_required_confidence": round(0.95, 12),
        "min_required_margin": round(0.1, 12),
        "min_required_convergence": round(0.1, 12),
        "allow_tie_break": True,
        "allow_no_improvement": True,
        "require_stable_transition": False,
    }
    hold_policy = GovernancePolicy(
        min_required_score=0.5,
        min_required_confidence=0.95,
        min_required_margin=0.1,
        min_required_convergence=0.1,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
        stable_hash=sha256_hex(hold_policy_payload),
    )
    bad_governance = evaluate_governed_orchestration(hold_policy, transition, refinement)
    with pytest.raises(ValueError, match="governance verdict"):
        build_action_capsule(transition, refinement, bad_governance)

    transition2 = _transition()
    decision_payload = {
        "selected_ordering_signature": "sig_a",
        "selected_score": round(0.2, 12),
        "decision_rank": 1,
        "margin_to_next": round(0.01, 12),
        "decision_confidence": round(0.2, 12),
        "decision_type": "clear_winner",
    }
    decision = TransitionDecision(
        selected_ordering_signature="sig_a",
        selected_score=0.2,
        decision_rank=1,
        margin_to_next=0.01,
        decision_confidence=0.2,
        decision_type="clear_winner",
        stable_hash=sha256_hex(decision_payload),
    )
    t_payload = {
        "input_receipt_hash": "1" * 64,
        "candidate_count": 2,
        "selected_decision": decision.to_dict(),
        "selection_rule": "ordered_scores_margin_dominance_v1",
        "classification": "stable_transition",
    }
    transition2 = TransitionPolicyReceipt(
        input_receipt_hash="1" * 64,
        candidate_count=2,
        selected_decision=decision,
        selection_rule="ordered_scores_margin_dominance_v1",
        classification="stable_transition",
        stable_hash=sha256_hex(t_payload),
    )
    refinement2 = refine_transition_policy(transition2)
    governance2 = evaluate_governed_orchestration(_policy(), transition2, refinement2)
    with pytest.raises(ValueError, match="governance verdict"):
        build_action_capsule(transition2, refinement2, governance2)


def test_tampered_receipt_rejected() -> None:
    transition, refinement, governance = _triplet()
    object.__setattr__(transition, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="transition_receipt stable_hash"):
        build_action_capsule(transition, refinement, governance)


def test_canonical_reconstruction_and_hash_stability_under_reconstruction() -> None:
    transition, refinement, governance = _triplet()
    capsule, proof = build_action_capsule(transition, refinement, governance)
    capsule2, proof2 = build_action_capsule(transition, refinement, governance)

    assert capsule.to_canonical_json() == capsule2.to_canonical_json()
    assert capsule.capsule_hash == capsule2.capsule_hash
    assert proof.proof_receipt_hash == proof2.proof_receipt_hash

    reconstructed_payload = json.loads(capsule.to_canonical_json())
    reconstructed_payload["action_descriptor"] = ActionDescriptor(**reconstructed_payload["action_descriptor"])
    reconstructed_payload["admissibility_reasons"] = tuple(reconstructed_payload["admissibility_reasons"])
    reconstructed_payload["certification_reasons"] = tuple(reconstructed_payload["certification_reasons"])
    reconstructed_payload["validation_notes"] = tuple(reconstructed_payload["validation_notes"])
    reconstructed = ProofCarryingActionCapsule(**reconstructed_payload)
    reproof_payload = json.loads(proof.to_canonical_json())
    reproof_payload["verification_checks"] = tuple(reproof_payload["verification_checks"])
    reproof = ActionCapsuleProofReceipt(**reproof_payload)
    assert reconstructed.to_canonical_json() == capsule.to_canonical_json()
    assert reconstructed.capsule_hash == capsule.capsule_hash
    assert reproof.proof_receipt_hash == proof.proof_receipt_hash


def test_invalid_payload_rejection() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": float("nan"), "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )

    with pytest.raises(ValueError, match="non-finite"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": float("inf"), "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )

    with pytest.raises(ValueError, match="non-canonicalizable"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": 0.1, "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": object()},
            representation_only=True,
            payload_schema_version="v146.0",
        )

    with pytest.raises(ValueError, match="execution semantics"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": 0.1, "refinement_hash": "2" * 64},
                "governance_linkage": {
                    "verdict": "allow",
                    "verdict_hash": "3" * 64,
                    "governance_hash": "4" * 64,
                    "command": "run",
                },
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )


def test_strict_invariants_and_tuple_ordering_and_self_consistency() -> None:
    transition, refinement, governance = _triplet()
    capsule, proof = build_action_capsule(transition, refinement, governance)

    assert capsule.admissibility_reasons == (
        "governance_allow_verdict",
        "transition_refinement_governance_linkage_valid",
    )
    assert proof.verification_checks == (
        "transition_receipt_hash_valid",
        "refinement_receipt_hash_valid",
        "governance_receipt_hash_valid",
        "governance_verdict_allow",
        "capsule_hash_self_valid",
        "replay_identity_valid",
    )
    assert capsule.capsule_hash == capsule.stable_hash()
    assert proof.proof_receipt_hash == proof.stable_hash()

    with pytest.raises(ValueError, match="representation_only"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": 0.1, "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=False,
            payload_schema_version="v146.0",
        )

    bad_capsule = capsule.to_dict()
    bad_capsule["action_descriptor"] = ActionDescriptor(**bad_capsule["action_descriptor"])
    bad_capsule["analysis_only"] = False
    with pytest.raises(ValueError, match="analysis_only"):
        ProofCarryingActionCapsule(**bad_capsule)

    bad_capsule2 = capsule.to_dict()
    bad_capsule2["action_descriptor"] = ActionDescriptor(**bad_capsule2["action_descriptor"])
    bad_capsule2["non_executing"] = False
    with pytest.raises(ValueError, match="non_executing"):
        ProofCarryingActionCapsule(**bad_capsule2)

    bad_capsule3 = capsule.to_dict()
    bad_capsule3["action_descriptor"] = ActionDescriptor(**bad_capsule3["action_descriptor"])
    bad_capsule3["side_effect_free"] = False
    with pytest.raises(ValueError, match="side_effect_free"):
        ProofCarryingActionCapsule(**bad_capsule3)

    with pytest.raises(ValueError, match="action_type"):
        ActionDescriptor(
            action_type="bad_action",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64},
                "refined_outcome": {"classification": "bounded", "convergence_metric": 0.1, "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )

    bad_hash_capsule = capsule.to_dict()
    bad_hash_capsule["action_descriptor"] = ActionDescriptor(**bad_hash_capsule["action_descriptor"])
    bad_hash_capsule["mesh_hash"] = "abc"
    with pytest.raises(ValueError, match="64-char"):
        ProofCarryingActionCapsule(**bad_hash_capsule)

    bad_hash_capsule2 = capsule.to_dict()
    bad_hash_capsule2["action_descriptor"] = ActionDescriptor(**bad_hash_capsule2["action_descriptor"])
    bad_hash_capsule2["mesh_hash"] = "A" * 64
    with pytest.raises(ValueError, match="64-char"):
        ProofCarryingActionCapsule(**bad_hash_capsule2)

    bad_hash_capsule3 = capsule.to_dict()
    bad_hash_capsule3["action_descriptor"] = ActionDescriptor(**bad_hash_capsule3["action_descriptor"])
    bad_hash_capsule3["mesh_hash"] = "a" * 63
    with pytest.raises(ValueError, match="64-char"):
        ProofCarryingActionCapsule(**bad_hash_capsule3)


def test_nested_descriptor_schema_validation() -> None:
    """Issue 4: ActionDescriptor enforces nested key sets and hash formats."""
    _base_payload = {
        "selected_transition": {"ordering_signature": "sig_a", "transition_hash": "1" * 64},
        "refined_outcome": {"classification": "bounded", "convergence_metric": 0.5, "refinement_hash": "2" * 64},
        "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
    }

    def _make(**overrides: object) -> ActionDescriptor:
        import copy
        payload = copy.deepcopy(_base_payload)
        for path, val in overrides.items():
            section, _, key = path.partition(".")
            if key:
                payload[section][key] = val  # type: ignore[index]
            else:
                payload[section] = val  # type: ignore[assignment]
        return ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload=payload,
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )

    # Missing key in selected_transition
    with pytest.raises(ValueError, match="selected_transition"):
        _make(**{"selected_transition": {"transition_hash": "1" * 64}})

    # Extra key in selected_transition
    with pytest.raises(ValueError, match="selected_transition"):
        _make(**{"selected_transition": {"ordering_signature": "s", "transition_hash": "1" * 64, "extra": "x"}})

    # Invalid SHA-256 in transition_hash
    with pytest.raises(ValueError, match="transition_hash"):
        _make(**{"selected_transition.transition_hash": "notahex"})

    # Missing key in refined_outcome
    with pytest.raises(ValueError, match="refined_outcome"):
        _make(**{"refined_outcome": {"classification": "bounded", "refinement_hash": "2" * 64}})

    # Extra key in refined_outcome
    with pytest.raises(ValueError, match="refined_outcome"):
        _make(**{"refined_outcome": {"classification": "bounded", "convergence_metric": 0.5, "refinement_hash": "2" * 64, "extra": "x"}})

    # Invalid SHA-256 in refinement_hash
    with pytest.raises(ValueError, match="refinement_hash"):
        _make(**{"refined_outcome.refinement_hash": "short"})

    # Missing key in governance_linkage
    with pytest.raises(ValueError, match="governance_linkage"):
        _make(**{"governance_linkage": {"verdict": "allow", "governance_hash": "4" * 64}})

    # Extra key in governance_linkage
    with pytest.raises(ValueError, match="governance_linkage"):
        _make(**{"governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64, "extra": "x"}})

    # governance_linkage.verdict != "allow"
    with pytest.raises(ValueError, match="governance_linkage.verdict"):
        _make(**{"governance_linkage.verdict": "hold"})

    # Invalid SHA-256 in verdict_hash
    with pytest.raises(ValueError, match="verdict_hash"):
        _make(**{"governance_linkage.verdict_hash": "bad"})

    # Invalid SHA-256 in governance_hash
    with pytest.raises(ValueError, match="governance_hash"):
        _make(**{"governance_linkage.governance_hash": "G" * 64})


def test_descriptor_capsule_lineage_cross_check() -> None:
    """Issue 1: ProofCarryingActionCapsule cross-checks descriptor lineage against top-level fields."""
    transition, refinement, governance = _triplet()
    capsule, _ = build_action_capsule(transition, refinement, governance)

    ap = capsule.action_descriptor.action_payload
    bc = capsule.action_descriptor.bound_constraints

    # Build an alternative descriptor with a different transition_hash
    alt_transition_hash = "a" * 64
    alt_descriptor = ActionDescriptor(
        action_type="governed_transition_commitment",
        target_scope="orchestration",
        action_payload={
            "selected_transition": {
                "ordering_signature": ap["selected_transition"]["ordering_signature"],
                "transition_hash": alt_transition_hash,
            },
            "refined_outcome": dict(ap["refined_outcome"]),
            "governance_linkage": dict(ap["governance_linkage"]),
        },
        bound_constraints=dict(bc),
        representation_only=True,
        payload_schema_version="v146.0",
    )

    # Build capsule kwargs with alt descriptor but original transition_receipt_hash
    kwargs = capsule.to_dict()
    kwargs["action_descriptor"] = alt_descriptor
    # Recompute capsule_hash so the self-hash check passes, exposing the lineage cross-check
    payload_without_hash = {k: v for k, v in kwargs.items() if k != "capsule_hash"}
    payload_without_hash["action_descriptor"] = alt_descriptor.to_dict()
    kwargs["capsule_hash"] = sha256_hex(payload_without_hash)

    with pytest.raises(ValueError, match="transition_hash mismatch"):
        ProofCarryingActionCapsule(**kwargs)


def test_build_validates_nested_stable_hashes() -> None:
    """Issue 2: build_action_capsule validates nested stable_hash fields."""
    transition, refinement, governance = _triplet()

    # Tamper with selected_decision.stable_hash and recompute parent receipt hash
    original_decision = transition.selected_decision
    tampered_decision_hash = "b" * 64
    # Create a valid decision first, then bypass its validation to inject tampered hash
    tampered_decision = TransitionDecision(
        selected_ordering_signature=original_decision.selected_ordering_signature,
        selected_score=original_decision.selected_score,
        decision_rank=original_decision.decision_rank,
        margin_to_next=original_decision.margin_to_next,
        decision_confidence=original_decision.decision_confidence,
        decision_type=original_decision.decision_type,
        stable_hash=original_decision.stable_hash,
    )
    object.__setattr__(tampered_decision, "stable_hash", tampered_decision_hash)

    # Recompute transition receipt hash so _require_self_hash passes (top-level)
    t_payload = {
        "input_receipt_hash": transition.input_receipt_hash,
        "candidate_count": transition.candidate_count,
        "selected_decision": tampered_decision.to_dict(),
        "selection_rule": transition.selection_rule,
        "classification": transition.classification,
    }
    tampered_transition = TransitionPolicyReceipt(
        input_receipt_hash=transition.input_receipt_hash,
        candidate_count=transition.candidate_count,
        selected_decision=tampered_decision,
        selection_rule=transition.selection_rule,
        classification=transition.classification,
        stable_hash=sha256_hex(t_payload),
    )

    with pytest.raises(ValueError, match="selected_decision stable_hash"):
        build_action_capsule(tampered_transition, refinement, governance)


def test_nested_replay_recompute_detects_step_hash_mismatch() -> None:
    transition, refinement, governance = _triplet()
    bad_refinement = refinement
    bad_step = bad_refinement.steps[0]
    tampered_step_payload = {
        "iteration": bad_step.iteration,
        "input_vector": tuple(round(v, 12) for v in bad_step.input_vector),
        "output_vector": tuple(round(v, 12) for v in bad_step.output_vector),
        "delta_norm": round(min(1.0, bad_step.delta_norm + 0.001), 12),
    }
    object.__setattr__(bad_step, "stable_hash", sha256_hex(tampered_step_payload))
    bad_refinement_payload = {
        "input_policy_hash": bad_refinement.input_policy_hash,
        "steps": tuple(step.to_dict() for step in bad_refinement.steps),
        "final_vector": tuple(round(v, 12) for v in bad_refinement.final_vector),
        "iteration_count": bad_refinement.iteration_count,
        "converged": bad_refinement.converged,
        "convergence_metric": round(bad_refinement.convergence_metric, 12),
        "classification": bad_refinement.classification,
    }
    object.__setattr__(bad_refinement, "stable_hash", sha256_hex(bad_refinement_payload))
    with pytest.raises(ValueError, match="canonical field recomputation hash is invalid|stable_hash is invalid"):
        build_action_capsule(transition, bad_refinement, governance)


def test_canonical_round_trip_stability_for_capsule_objects() -> None:
    transition, refinement, governance = _triplet()
    capsule, proof = build_action_capsule(transition, refinement, governance)

    descriptor_json = capsule.action_descriptor.to_canonical_json()
    descriptor = ActionDescriptor(**json.loads(descriptor_json))
    assert descriptor.to_canonical_json() == descriptor_json

    capsule_payload = json.loads(capsule.to_canonical_json())
    capsule_payload["action_descriptor"] = ActionDescriptor(**capsule_payload["action_descriptor"])
    capsule_payload["admissibility_reasons"] = tuple(capsule_payload["admissibility_reasons"])
    capsule_payload["certification_reasons"] = tuple(capsule_payload["certification_reasons"])
    capsule_payload["validation_notes"] = tuple(capsule_payload["validation_notes"])
    reconstructed_capsule = ProofCarryingActionCapsule(**capsule_payload)
    assert reconstructed_capsule.to_canonical_json() == capsule.to_canonical_json()

    proof_payload = json.loads(proof.to_canonical_json())
    proof_payload["verification_checks"] = tuple(proof_payload["verification_checks"])
    reconstructed_proof = ActionCapsuleProofReceipt(**proof_payload)
    assert reconstructed_proof.to_canonical_json() == proof.to_canonical_json()


def test_replay_identity_tamper_rejected() -> None:
    transition, refinement, governance = _triplet()
    capsule, _ = build_action_capsule(transition, refinement, governance)
    kwargs = capsule.to_dict()
    kwargs["action_descriptor"] = ActionDescriptor(**kwargs["action_descriptor"])
    kwargs["transition_receipt_hash"] = "a" * 64
    kwargs["transition_hash"] = "a" * 64
    payload_without_hash = {k: v for k, v in kwargs.items() if k != "capsule_hash"}
    payload_without_hash["action_descriptor"] = kwargs["action_descriptor"].to_dict()
    kwargs["capsule_hash"] = sha256_hex(payload_without_hash)
    with pytest.raises(ValueError, match="replay_identity mismatch"):
        ProofCarryingActionCapsule(**kwargs)


def test_capsule_and_proof_hash_recompute_match() -> None:
    transition, refinement, governance = _triplet()
    capsule, proof = build_action_capsule(transition, refinement, governance)
    assert sha256_hex(capsule._payload_without_hash()) == capsule.capsule_hash
    assert sha256_hex(proof._payload_without_hash()) == proof.proof_receipt_hash


def test_deep_schema_extra_key_rejected() -> None:
    with pytest.raises(ValueError, match="selected_transition must have exactly keys"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {
                    "ordering_signature": "sig_a",
                    "transition_hash": "1" * 64,
                    "extra_key": "x",
                },
                "refined_outcome": {"classification": "bounded", "convergence_metric": 0.5, "refinement_hash": "2" * 64},
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )


def test_float_normalization_requires_pre_rounded_values() -> None:
    with pytest.raises(ValueError, match="pre-rounded"):
        ActionDescriptor(
            action_type="governed_transition_commitment",
            target_scope="orchestration",
            action_payload={
                "selected_transition": {"ordering_signature": "sig_a", "transition_hash": "1" * 64},
                "refined_outcome": {
                    "classification": "bounded",
                    "convergence_metric": 0.1234567890123,
                    "refinement_hash": "2" * 64,
                },
                "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
            },
            bound_constraints={"bounded_refinement": True},
            representation_only=True,
            payload_schema_version="v146.0",
        )


def test_convergence_metric_int_normalized_to_float() -> None:
    descriptor = ActionDescriptor(
        action_type="governed_transition_commitment",
        target_scope="orchestration",
        action_payload={
            "selected_transition": {"ordering_signature": "sig_a", "transition_hash": "1" * 64},
            "refined_outcome": {
                "classification": "bounded",
                "convergence_metric": 1,
                "refinement_hash": "2" * 64,
            },
            "governance_linkage": {"verdict": "allow", "verdict_hash": "3" * 64, "governance_hash": "4" * 64},
        },
        bound_constraints={"bounded_refinement": True},
        representation_only=True,
        payload_schema_version="v146.0",
    )
    assert descriptor.action_payload["refined_outcome"]["convergence_metric"] == 1.0
    assert isinstance(descriptor.action_payload["refined_outcome"]["convergence_metric"], float)


def test_cross_field_inconsistency_rejected() -> None:
    transition, refinement, governance = _triplet()
    capsule, _ = build_action_capsule(transition, refinement, governance)
    kwargs = capsule.to_dict()
    descriptor_payload = dict(kwargs["action_descriptor"])
    descriptor_payload["action_payload"] = dict(descriptor_payload["action_payload"])
    descriptor_payload["action_payload"]["governance_linkage"] = dict(
        descriptor_payload["action_payload"]["governance_linkage"]
    )
    descriptor_payload["action_payload"]["governance_linkage"]["verdict"] = "allow"
    kwargs["action_descriptor"] = ActionDescriptor(**descriptor_payload)
    kwargs["governance_verdict"] = "allow"
    kwargs["governance_hash"] = "b" * 64
    kwargs["governance_receipt_hash"] = "b" * 64
    kwargs["replay_identity"] = sha256_hex({
        "transition_receipt_hash": kwargs["transition_receipt_hash"],
        "refinement_receipt_hash": kwargs["refinement_receipt_hash"],
        "governance_receipt_hash": kwargs["governance_receipt_hash"],
    })
    payload_without_hash = {k: v for k, v in kwargs.items() if k != "capsule_hash"}
    payload_without_hash["action_descriptor"] = kwargs["action_descriptor"].to_dict()
    kwargs["capsule_hash"] = sha256_hex(payload_without_hash)
    with pytest.raises(ValueError, match="governance_hash mismatch"):
        ProofCarryingActionCapsule(**kwargs)


def test_input_mutation_safety() -> None:
    transition, refinement, governance = _triplet()
    transition_before = transition.to_canonical_json()
    refinement_before = refinement.to_canonical_json()
    governance_before = governance.to_canonical_json()

    build_action_capsule(transition, refinement, governance)

    assert transition.to_canonical_json() == transition_before
    assert refinement.to_canonical_json() == refinement_before
    assert governance.to_canonical_json() == governance_before


def test_hash_receipt_hash_equality_enforced() -> None:
    transition, refinement, governance = _triplet()
    capsule, _ = build_action_capsule(transition, refinement, governance)
    kwargs = capsule.to_dict()
    kwargs["action_descriptor"] = ActionDescriptor(**kwargs["action_descriptor"])
    kwargs["transition_hash"] = "a" * 64
    payload_without_hash = {k: v for k, v in kwargs.items() if k != "capsule_hash"}
    payload_without_hash["action_descriptor"] = kwargs["action_descriptor"].to_dict()
    kwargs["capsule_hash"] = sha256_hex(payload_without_hash)
    with pytest.raises(ValueError, match="transition_hash must equal transition_receipt_hash"):
        ProofCarryingActionCapsule(**kwargs)

    kwargs2 = capsule.to_dict()
    kwargs2["action_descriptor"] = ActionDescriptor(**kwargs2["action_descriptor"])
    kwargs2["refinement_hash"] = "b" * 64
    payload_without_hash2 = {k: v for k, v in kwargs2.items() if k != "capsule_hash"}
    payload_without_hash2["action_descriptor"] = kwargs2["action_descriptor"].to_dict()
    kwargs2["capsule_hash"] = sha256_hex(payload_without_hash2)
    with pytest.raises(ValueError, match="refinement_hash must equal refinement_receipt_hash"):
        ProofCarryingActionCapsule(**kwargs2)

    kwargs3 = capsule.to_dict()
    kwargs3["action_descriptor"] = ActionDescriptor(**kwargs3["action_descriptor"])
    kwargs3["governance_hash"] = "c" * 64
    payload_without_hash3 = {k: v for k, v in kwargs3.items() if k != "capsule_hash"}
    payload_without_hash3["action_descriptor"] = kwargs3["action_descriptor"].to_dict()
    kwargs3["capsule_hash"] = sha256_hex(payload_without_hash3)
    with pytest.raises(ValueError, match="governance_hash must equal governance_receipt_hash"):
        ProofCarryingActionCapsule(**kwargs3)
