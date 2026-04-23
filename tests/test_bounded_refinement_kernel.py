from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.bounded_refinement_kernel import (
    CLASSIFICATION_BOUNDED,
    CLASSIFICATION_CONVERGED,
    CLASSIFICATION_NO_IMPROVEMENT,
    MAX_ITERATIONS,
    RefinementReceipt,
    refine_transition_policy,
)
from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.deterministic_transition_policy import TransitionDecision, TransitionPolicyReceipt


def _decision(ordering_signature: str = "sig_a") -> TransitionDecision:
    payload = {
        "selected_ordering_signature": ordering_signature,
        "selected_score": round(0.7, 12),
        "decision_rank": 1,
        "margin_to_next": round(0.2, 12),
        "decision_confidence": round(0.84, 12),
        "decision_type": "clear_winner",
    }
    return TransitionDecision(
        selected_ordering_signature=ordering_signature,
        selected_score=0.7,
        decision_rank=1,
        margin_to_next=0.2,
        decision_confidence=0.84,
        decision_type="clear_winner",
        stable_hash=sha256_hex(payload),
    )


def _policy_receipt(ordering_signature: str = "sig_a") -> TransitionPolicyReceipt:
    decision = _decision(ordering_signature)
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


_SIGNATURE_SEARCH_LIMIT = 2000
_CLASSIFICATION_RECEIPT_CACHE: dict[str, TransitionPolicyReceipt] = {}


def _find_receipt_for_classification(target: str) -> TransitionPolicyReceipt:
    """Return a deterministic receipt for a requested refinement classification."""
    cached_receipt = _CLASSIFICATION_RECEIPT_CACHE.get(target)
    if cached_receipt is not None:
        return cached_receipt

    for idx in range(_SIGNATURE_SEARCH_LIMIT):
        candidate = _policy_receipt(f"sig_{idx:04d}")
        classification = refine_transition_policy(candidate).classification
        _CLASSIFICATION_RECEIPT_CACHE.setdefault(classification, candidate)
        if classification == target:
            return candidate

    raise AssertionError(
        f"unable to find deterministic signature for classification={target}"
    )
def test_deterministic_replay() -> None:
    receipt = _policy_receipt("deterministic_sig")
    first = refine_transition_policy(receipt)
    second = refine_transition_policy(receipt)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_bounded_iterations_never_exceed_max() -> None:
    result = refine_transition_policy(_policy_receipt("iter_bound"))
    assert 1 <= result.iteration_count <= MAX_ITERATIONS
    assert len(result.steps) == result.iteration_count


def test_convergence_case() -> None:
    receipt = _find_receipt_for_classification(CLASSIFICATION_CONVERGED)
    result = refine_transition_policy(receipt)
    assert result.classification == CLASSIFICATION_CONVERGED
    assert result.converged is True


def test_bounded_case() -> None:
    receipt = _find_receipt_for_classification(CLASSIFICATION_BOUNDED)
    result = refine_transition_policy(receipt)
    assert result.classification == CLASSIFICATION_BOUNDED
    assert result.converged is False


def test_no_improvement_case() -> None:
    receipt = _find_receipt_for_classification(CLASSIFICATION_NO_IMPROVEMENT)
    result = refine_transition_policy(receipt)
    assert result.classification == CLASSIFICATION_NO_IMPROVEMENT


def test_vector_bounds_and_rounding_stability() -> None:
    result_a = refine_transition_policy(_policy_receipt("rounding_sig"))
    result_b = refine_transition_policy(_policy_receipt("rounding_sig"))
    for step in result_a.steps:
        assert all(0.0 <= value <= 1.0 for value in step.input_vector)
        assert all(0.0 <= value <= 1.0 for value in step.output_vector)
        assert step.delta_norm == round(step.delta_norm, 12)
    assert all(0.0 <= value <= 1.0 for value in result_a.final_vector)
    assert result_a.to_canonical_json() == result_b.to_canonical_json()


def test_invalid_input_type_rejected() -> None:
    with pytest.raises(ValueError, match="TransitionPolicyReceipt"):
        refine_transition_policy("invalid")  # type: ignore[arg-type]


def test_tampered_receipt_hash_rejected() -> None:
    tampered = _policy_receipt("tampered")
    object.__setattr__(tampered, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="stable_hash"):
        refine_transition_policy(tampered)


@pytest.mark.parametrize("bad_margin", [1.1, -0.1, 2.0])
def test_out_of_range_margin_to_next_rejected(bad_margin: float) -> None:
    tampered = _policy_receipt("bad_margin")
    object.__setattr__(tampered.selected_decision, "margin_to_next", bad_margin)
    object.__setattr__(tampered.selected_decision, "stable_hash", tampered.selected_decision.computed_stable_hash())
    object.__setattr__(tampered, "stable_hash", tampered.computed_stable_hash())
    with pytest.raises(ValueError, match="margin_to_next"):
        refine_transition_policy(tampered)


def test_non_canonical_signature_rejected() -> None:
    receipt = _policy_receipt("sig_canon")
    object.__setattr__(receipt.selected_decision, "selected_ordering_signature", " sig_canon ")
    object.__setattr__(
        receipt.selected_decision,
        "stable_hash",
        receipt.selected_decision.computed_stable_hash(),
    )
    object.__setattr__(receipt, "stable_hash", receipt.computed_stable_hash())
    with pytest.raises(ValueError, match="selected_ordering_signature"):
        refine_transition_policy(receipt)


def test_immutability_and_hash_self_validation() -> None:
    result = refine_transition_policy(_policy_receipt("immut_sig"))
    with pytest.raises(FrozenInstanceError):
        result.classification = CLASSIFICATION_BOUNDED  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.steps[0].delta_norm = 0.5  # type: ignore[misc]
    with pytest.raises(ValueError, match="stable_hash"):
        RefinementReceipt(
            input_policy_hash=result.input_policy_hash,
            steps=result.steps,
            final_vector=result.final_vector,
            iteration_count=result.iteration_count,
            converged=result.converged,
            convergence_metric=result.convergence_metric,
            classification=result.classification,
            stable_hash="f" * 64,
        )
