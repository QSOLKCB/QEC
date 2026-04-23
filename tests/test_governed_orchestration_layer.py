from __future__ import annotations

import pytest

from qec.analysis.bounded_refinement_kernel import RefinementReceipt, refine_transition_policy
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.deterministic_transition_policy import (
    TransitionDecision,
    TransitionPolicyReceipt,
)
from qec.analysis.governed_orchestration_layer import (
    CHECK_CONFIDENCE,
    CHECK_CONVERGENCE,
    CHECK_MARGIN,
    CHECK_NO_IMPROVEMENT,
    CHECK_SELECTED_SCORE,
    CHECK_STABLE_TRANSITION,
    CHECK_TIE_BREAK,
    GovernancePolicy,
    GovernedOrchestrationReceipt,
    OrchestrationVerdict,
    REASON_LOW_CONFIDENCE,
    REASON_LOW_CONVERGENCE,
    REASON_MARGIN_TOO_LOW,
    REASON_NO_IMPROVEMENT,
    REASON_OK,
    REASON_SCORE_TOO_LOW,
    REASON_TIE_BREAK_DISALLOWED,
    REASON_UNSTABLE_TRANSITION,
    VERDICT_ALLOW,
    VERDICT_HOLD,
    VERDICT_REJECT,
    evaluate_governed_orchestration,
)


def _policy(
    *,
    min_required_score: float = 0.6,
    min_required_confidence: float = 0.7,
    min_required_margin: float = 0.2,
    min_required_convergence: float = 0.45,
    allow_tie_break: bool = True,
    allow_no_improvement: bool = True,
    require_stable_transition: bool = False,
) -> GovernancePolicy:
    payload = {
        "min_required_score": round(min_required_score, 12),
        "min_required_confidence": round(min_required_confidence, 12),
        "min_required_margin": round(min_required_margin, 12),
        "min_required_convergence": round(min_required_convergence, 12),
        "allow_tie_break": allow_tie_break,
        "allow_no_improvement": allow_no_improvement,
        "require_stable_transition": require_stable_transition,
    }
    return GovernancePolicy(
        min_required_score=min_required_score,
        min_required_confidence=min_required_confidence,
        min_required_margin=min_required_margin,
        min_required_convergence=min_required_convergence,
        allow_tie_break=allow_tie_break,
        allow_no_improvement=allow_no_improvement,
        require_stable_transition=require_stable_transition,
        stable_hash=sha256_hex(payload),
    )


def _transition(
    *,
    selected_score: float = 0.7,
    decision_confidence: float = 0.85,
    margin_to_next: float = 0.3,
    decision_type: str = "clear_winner",
    classification: str = "stable_transition",
    ordering_signature: str = "sig_a",
) -> TransitionPolicyReceipt:
    decision_payload = {
        "selected_ordering_signature": ordering_signature,
        "selected_score": round(selected_score, 12),
        "decision_rank": 1,
        "margin_to_next": round(margin_to_next, 12),
        "decision_confidence": round(decision_confidence, 12),
        "decision_type": decision_type,
    }
    decision = TransitionDecision(
        selected_ordering_signature=ordering_signature,
        selected_score=selected_score,
        decision_rank=1,
        margin_to_next=margin_to_next,
        decision_confidence=decision_confidence,
        decision_type=decision_type,
        stable_hash=sha256_hex(decision_payload),
    )
    receipt_payload = {
        "input_receipt_hash": "1" * 64,
        "candidate_count": 2,
        "selected_decision": decision.to_dict(),
        "selection_rule": "ordered_scores_margin_dominance_v1",
        "classification": classification,
    }
    return TransitionPolicyReceipt(
        input_receipt_hash="1" * 64,
        candidate_count=2,
        selected_decision=decision,
        selection_rule="ordered_scores_margin_dominance_v1",
        classification=classification,
        stable_hash=sha256_hex(receipt_payload),
    )


def _refinement(
    transition: TransitionPolicyReceipt,
    *,
    force_classification: str | None = None,
    force_convergence: float | None = None,
) -> RefinementReceipt:
    receipt = refine_transition_policy(transition)
    if force_classification is not None:
        object.__setattr__(receipt, "classification", force_classification)
    if force_convergence is not None:
        object.__setattr__(receipt, "convergence_metric", force_convergence)
    if force_classification is not None or force_convergence is not None:
        payload = {
            "input_policy_hash": receipt.input_policy_hash,
            "steps": tuple(step.to_dict() for step in receipt.steps),
            "final_vector": tuple(round(value, 12) for value in receipt.final_vector),
            "iteration_count": receipt.iteration_count,
            "converged": receipt.converged,
            "convergence_metric": round(receipt.convergence_metric, 12),
            "classification": receipt.classification,
        }
        object.__setattr__(receipt, "stable_hash", sha256_hex(payload))
    return receipt


def _base_triplet() -> tuple[GovernancePolicy, TransitionPolicyReceipt, RefinementReceipt]:
    transition = _transition()
    refinement = _refinement(transition)
    return _policy(), transition, refinement


def test_deterministic_replay() -> None:
    policy, transition, refinement = _base_triplet()
    first = evaluate_governed_orchestration(policy, transition, refinement)
    second = evaluate_governed_orchestration(policy, transition, refinement)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_allow_verdict() -> None:
    policy, transition, refinement = _base_triplet()
    result = evaluate_governed_orchestration(policy, transition, refinement)
    assert result.verdict.verdict == VERDICT_ALLOW
    assert result.verdict.admissible is True
    assert result.verdict.reason_code == REASON_OK


def test_reject_on_tie_break_disallowed() -> None:
    policy = _policy(allow_tie_break=False)
    transition = _transition(decision_type="tie_break")
    refinement = _refinement(transition)
    result = evaluate_governed_orchestration(policy, transition, refinement)
    assert result.verdict.verdict == VERDICT_REJECT
    assert result.verdict.reason_code == REASON_TIE_BREAK_DISALLOWED


def test_hold_on_low_confidence() -> None:
    policy = _policy(min_required_confidence=0.9)
    transition = _transition(decision_confidence=0.4)
    refinement = _refinement(transition)
    result = evaluate_governed_orchestration(policy, transition, refinement)
    assert result.verdict.verdict == VERDICT_HOLD
    assert result.verdict.reason_code == REASON_LOW_CONFIDENCE


def test_reject_on_low_score() -> None:
    policy = _policy(min_required_score=0.8)
    transition = _transition(selected_score=0.7)
    refinement = _refinement(transition)
    result = evaluate_governed_orchestration(policy, transition, refinement)
    assert result.verdict.verdict == VERDICT_REJECT
    assert result.verdict.reason_code == REASON_SCORE_TOO_LOW


def test_reject_on_low_margin() -> None:
    policy = _policy(min_required_margin=0.2)
    transition = _transition(margin_to_next=0.1)
    refinement = _refinement(transition)
    receipt = evaluate_governed_orchestration(policy, transition, refinement)
    assert receipt.verdict.verdict == VERDICT_REJECT
    assert receipt.verdict.reason_code == REASON_MARGIN_TOO_LOW


def test_hold_on_low_convergence() -> None:
    policy, transition, refinement = _base_triplet()
    low_refinement = _refinement(transition, force_convergence=0.1)
    result = evaluate_governed_orchestration(policy, transition, low_refinement)
    assert result.verdict.verdict == VERDICT_HOLD
    assert result.verdict.reason_code == REASON_LOW_CONVERGENCE


def test_no_improvement_policy_behavior() -> None:
    transition = _transition()
    no_improvement = _refinement(transition, force_classification="no_improvement")
    hold_policy = _policy(allow_no_improvement=False)
    allow_policy = _policy(allow_no_improvement=True)

    held = evaluate_governed_orchestration(hold_policy, transition, no_improvement)
    allowed = evaluate_governed_orchestration(allow_policy, transition, no_improvement)

    assert held.verdict.verdict == VERDICT_HOLD
    assert held.verdict.reason_code == REASON_NO_IMPROVEMENT
    assert allowed.verdict.verdict == VERDICT_ALLOW


def test_hold_on_unstable_transition_when_required() -> None:
    policy = _policy(require_stable_transition=True)
    transition = _transition(classification="uncertain_transition")
    refinement = _refinement(transition)
    receipt = evaluate_governed_orchestration(policy, transition, refinement)
    assert receipt.verdict.verdict == VERDICT_HOLD
    assert receipt.verdict.reason_code == REASON_UNSTABLE_TRANSITION


def test_linkage_validation_mismatch_rejected() -> None:
    policy, transition, refinement = _base_triplet()
    mismatched = refinement
    object.__setattr__(mismatched, "input_policy_hash", "2" * 64)
    payload = {
        "input_policy_hash": mismatched.input_policy_hash,
        "steps": tuple(step.to_dict() for step in mismatched.steps),
        "final_vector": tuple(round(value, 12) for value in mismatched.final_vector),
        "iteration_count": mismatched.iteration_count,
        "converged": mismatched.converged,
        "convergence_metric": round(mismatched.convergence_metric, 12),
        "classification": mismatched.classification,
    }
    object.__setattr__(mismatched, "stable_hash", sha256_hex(payload))
    with pytest.raises(ValueError, match="input_policy_hash"):
        evaluate_governed_orchestration(policy, transition, mismatched)


def test_tampered_nested_artifact_rejected() -> None:
    policy, transition, refinement = _base_triplet()

    tampered_transition = transition
    object.__setattr__(tampered_transition, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="transition_receipt"):
        evaluate_governed_orchestration(policy, tampered_transition, refinement)

    policy2, transition2, refinement2 = _base_triplet()
    tampered_refinement = refinement2
    object.__setattr__(tampered_refinement, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="refinement_receipt"):
        evaluate_governed_orchestration(policy2, transition2, tampered_refinement)

    policy3, transition3, refinement3 = _base_triplet()
    tampered_policy = policy3
    object.__setattr__(tampered_policy, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="policy"):
        evaluate_governed_orchestration(tampered_policy, transition3, refinement3)


def test_strict_enum_validation() -> None:
    with pytest.raises(ValueError, match="verdict is invalid"):
        OrchestrationVerdict(
            verdict="maybe",
            admissible=True,
            reason_code=REASON_OK,
            selected_ordering_signature="sig_a",
            decision_type="clear_winner",
            transition_classification="stable_transition",
            refinement_classification="converged",
            stable_hash="0" * 64,
        )


def test_check_ordering_stability() -> None:
    policy, transition, refinement = _base_triplet()
    result = evaluate_governed_orchestration(policy, transition, refinement)
    assert tuple(check.check_name for check in result.checks) == (
        CHECK_SELECTED_SCORE,
        CHECK_CONFIDENCE,
        CHECK_MARGIN,
        CHECK_TIE_BREAK,
        CHECK_STABLE_TRANSITION,
        CHECK_CONVERGENCE,
        CHECK_NO_IMPROVEMENT,
    )


def test_canonical_reconstruction_stability() -> None:
    policy, transition, refinement = _base_triplet()
    result = evaluate_governed_orchestration(policy, transition, refinement)
    policy_dict = result.policy.to_dict()
    rebuilt_policy = GovernancePolicy(**policy_dict)
    rebuilt_checks = tuple(type(check)(**check.to_dict()) for check in result.checks)
    rebuilt_verdict = OrchestrationVerdict(**result.verdict.to_dict())
    reconstructed = GovernedOrchestrationReceipt(
        policy=rebuilt_policy,
        input_transition_hash=result.input_transition_hash,
        input_refinement_hash=result.input_refinement_hash,
        checks=rebuilt_checks,
        verdict=rebuilt_verdict,
        stable_hash=result.stable_hash,
    )
    assert reconstructed.stable_hash == result.stable_hash


@pytest.mark.parametrize("bad_value", [True, False])
def test_policy_numeric_fields_reject_bool(bad_value: bool) -> None:
    with pytest.raises(ValueError, match="min_required_score"):
        _policy(min_required_score=bad_value)  # type: ignore[arg-type]
