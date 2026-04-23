from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.deterministic_transition_policy import (
    CLASSIFICATION_STABLE_TRANSITION,
    CLASSIFICATION_UNCERTAIN_TRANSITION,
    DECISION_TYPE_CLEAR_WINNER,
    DECISION_TYPE_NARROW_MARGIN,
    DECISION_TYPE_TIE_BREAK,
    TransitionPolicyReceipt,
    select_deterministic_transition,
)
from qec.analysis.state_conditioned_filter_mesh import _round12
from qec.analysis.state_conditioned_filter_mesh import (
    CLASSIFICATION_CONFLICTED,
    FilterMeshReceipt,
    FilterMeshState,
    OrderingScore,
)


def _state() -> FilterMeshState:
    return FilterMeshState(
        invariant_class="parity",
        geometry_class="surface",
        spectral_regime="high_frequency",
        hardware_class="superconducting",
        recurrence_class="stable",
        thermal_pressure=0.3,
        latency_drift=0.2,
        timing_skew=0.2,
        power_pressure=0.1,
        consensus_instability=0.2,
    )


def _score(ordering_signature: str, total_score: float, rank: int) -> OrderingScore:
    payload = {
        "ordering_signature": ordering_signature,
        "invariant_alignment": round(total_score, 12),
        "hardware_alignment": round(total_score, 12),
        "recurrence_avoidance": round(total_score, 12),
        "stability_alignment": round(total_score, 12),
        "total_score": round(total_score, 12),
        "rank": rank,
    }
    return OrderingScore(
        ordering_signature=ordering_signature,
        invariant_alignment=total_score,
        hardware_alignment=total_score,
        recurrence_avoidance=total_score,
        stability_alignment=total_score,
        total_score=total_score,
        rank=rank,
        stable_hash=sha256_hex(payload),
    )


def _receipt(total_scores: tuple[tuple[str, float], ...]) -> FilterMeshReceipt:
    ordered = tuple(_score(sig, score, rank) for rank, (sig, score) in enumerate(total_scores, start=1))
    top = ordered[0]
    payload = {
        "state": _state().to_dict(),
        "candidate_count": len(ordered),
        "ordered_scores": tuple(item.to_dict() for item in ordered),
        "dominant_ordering_signature": top.ordering_signature,
        "dominant_score": round(top.total_score, 12),
        "classification": CLASSIFICATION_CONFLICTED,
    }
    return FilterMeshReceipt(
        state=_state(),
        candidate_count=len(ordered),
        ordered_scores=ordered,
        dominant_ordering_signature=top.ordering_signature,
        dominant_score=top.total_score,
        classification=CLASSIFICATION_CONFLICTED,
        stable_hash=sha256_hex(payload),
    )


def test_deterministic_replay() -> None:
    receipt = _receipt((("sig_a", 0.90), ("sig_b", 0.50)))
    first = select_deterministic_transition(receipt)
    second = select_deterministic_transition(receipt)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_clear_winner_large_margin() -> None:
    receipt = _receipt((("sig_a", 0.88), ("sig_b", 0.60)))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.decision_type == DECISION_TYPE_CLEAR_WINNER


def test_narrow_margin() -> None:
    receipt = _receipt((("sig_a", 0.62), ("sig_b", 0.56)))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.decision_type == DECISION_TYPE_NARROW_MARGIN


def test_tie_break_case_uses_existing_ordering() -> None:
    receipt = _receipt((("sig_a", 0.600000000001), ("sig_b", 0.6)))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.decision_type == DECISION_TYPE_TIE_BREAK
    assert result.selected_decision.selected_ordering_signature == "sig_a"


def test_single_candidate_is_clear_winner() -> None:
    receipt = _receipt((("only", 0.42),))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.decision_type == DECISION_TYPE_CLEAR_WINNER
    assert result.selected_decision.margin_to_next == 1.0


def test_confidence_is_bounded_and_rounded() -> None:
    receipt = _receipt((("sig_a", 0.3333333333337), ("sig_b", 0.2222222222222)))
    result = select_deterministic_transition(receipt)
    assert 0.0 <= result.selected_decision.decision_confidence <= 1.0
    assert result.selected_decision.decision_confidence == round(result.selected_decision.decision_confidence, 12)


def test_classification_stable_vs_uncertain() -> None:
    stable = select_deterministic_transition(_receipt((("sig_a", 0.60), ("sig_b", 0.20))))
    uncertain = select_deterministic_transition(_receipt((("sig_a", 0.30), ("sig_b", 0.25))))
    assert stable.classification == CLASSIFICATION_STABLE_TRANSITION
    assert uncertain.classification == CLASSIFICATION_UNCERTAIN_TRANSITION


def test_invalid_receipt_type_rejected() -> None:
    with pytest.raises(ValueError, match="FilterMeshReceipt"):
        select_deterministic_transition("invalid")  # type: ignore[arg-type]


def test_tampered_receipt_hash_rejected() -> None:
    tampered = _receipt((("sig_a", 0.7), ("sig_b", 0.5)))
    object.__setattr__(tampered, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="stable_hash"):
        select_deterministic_transition(tampered)


def test_immutability() -> None:
    result = select_deterministic_transition(_receipt((("sig_a", 0.9), ("sig_b", 0.1))))
    with pytest.raises(FrozenInstanceError):
        result.classification = CLASSIFICATION_UNCERTAIN_TRANSITION  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.selected_decision.decision_rank = 2  # type: ignore[misc]


def test_margin_correctness() -> None:
    receipt = _receipt((("sig_a", 0.812345678912), ("sig_b", 0.712345678901)))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.margin_to_next == round(0.812345678912 - 0.712345678901, 12)


def test_canonical_margin_uses_rounded_scores_only() -> None:
    receipt = _receipt((("sig_a", 0.5000000000004), ("sig_b", 0.50000000000049)))
    result = select_deterministic_transition(receipt)
    assert _round12(0.5000000000004) == _round12(0.50000000000049)
    assert result.selected_decision.margin_to_next == 0.0


def test_margin_normalizes_negative_zero() -> None:
    receipt = _receipt((("sig_a", 0.5000000000004), ("sig_b", 0.50000000000049)))
    result = select_deterministic_transition(receipt)
    assert result.selected_decision.margin_to_next == 0.0
    assert repr(result.selected_decision.margin_to_next) != "-0.0"


def test_confidence_replay_stability_from_canonical_scores() -> None:
    receipt_a = _receipt((("sig_a", 0.2499999999996),))
    receipt_b = _receipt((("sig_a", 0.25),))
    assert receipt_a.stable_hash == receipt_b.stable_hash
    result_a = select_deterministic_transition(receipt_a)
    result_b = select_deterministic_transition(receipt_b)
    assert result_a.selected_decision.decision_confidence == result_b.selected_decision.decision_confidence
    assert result_a.classification == result_b.classification


@pytest.mark.parametrize("invalid_score", [True, float("nan"), float("inf")])
def test_invalid_total_score_types_rejected(invalid_score: float) -> None:
    receipt = _receipt((("sig_a", 0.8), ("sig_b", 0.4)))
    tampered_first = receipt.ordered_scores[0]
    with pytest.raises(ValueError):
        object.__setattr__(tampered_first, "total_score", invalid_score)
        object.__setattr__(tampered_first, "stable_hash", tampered_first.computed_stable_hash())
        object.__setattr__(receipt, "stable_hash", receipt.computed_stable_hash())
        select_deterministic_transition(receipt)


def test_invalid_input_receipt_hash_rejected() -> None:
    result = select_deterministic_transition(_receipt((("sig_a", 0.8), ("sig_b", 0.4))))
    with pytest.raises(ValueError, match="input_receipt_hash must be 64-char hex string"):
        TransitionPolicyReceipt(
            input_receipt_hash="abc",
            candidate_count=result.candidate_count,
            selected_decision=result.selected_decision,
            selection_rule=result.selection_rule,
            classification=result.classification,
            stable_hash=result.stable_hash,
        )


@pytest.mark.parametrize("candidate_count", [True, 1.5])
def test_candidate_count_strict_typing_rejected(candidate_count: int) -> None:
    receipt = _receipt((("sig_a", 0.9), ("sig_b", 0.1)))
    object.__setattr__(receipt, "candidate_count", candidate_count)
    object.__setattr__(receipt, "stable_hash", receipt.computed_stable_hash())
    with pytest.raises(ValueError, match="candidate_count must be int"):
        select_deterministic_transition(receipt)


def test_receipt_hash_self_validation() -> None:
    result = select_deterministic_transition(_receipt((("sig_a", 0.8), ("sig_b", 0.4))))
    with pytest.raises(ValueError, match="stable_hash"):
        TransitionPolicyReceipt(
            input_receipt_hash=result.input_receipt_hash,
            candidate_count=result.candidate_count,
            selected_decision=result.selected_decision,
            selection_rule=result.selection_rule,
            classification=result.classification,
            stable_hash="f" * 64,
        )
