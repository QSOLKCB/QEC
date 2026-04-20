from __future__ import annotations

import pytest

from qec.orchestration.ternary_decode_lane import (
    TernaryDecodeCandidate,
    run_ternary_decode_lane,
    _rank_candidates,
)


def test_determinism_repeated_runs_are_byte_identical() -> None:
    receipt_a = run_ternary_decode_lane((0, 1, 2, 0, 1))
    receipt_b = run_ternary_decode_lane((0, 1, 2, 0, 1))
    assert receipt_a.selected_correction == receipt_b.selected_correction
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_semantic_change_changes_hash_or_selection() -> None:
    receipt_a = run_ternary_decode_lane((0, 1, 2, 0, 1))
    receipt_b = run_ternary_decode_lane((0, 1, 2, 2, 1))
    assert (
        receipt_a.receipt_hash != receipt_b.receipt_hash
        or receipt_a.selected_candidate_id != receipt_b.selected_candidate_id
        or receipt_a.selected_correction != receipt_b.selected_correction
    )


def test_validation_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="ternary_symbols must be non-empty"):
        run_ternary_decode_lane(())


def test_validation_rejects_invalid_symbol() -> None:
    with pytest.raises(ValueError, match="canonical basis"):
        run_ternary_decode_lane((0, 1, 3))


def test_validation_rejects_non_integer_symbol() -> None:
    with pytest.raises(ValueError, match="must contain only integers"):
        run_ternary_decode_lane((0, 1, 1.5))


def test_balanced_ternary_boundary_canonicalizes_to_gf3_basis() -> None:
    receipt = run_ternary_decode_lane((-1, 0, 1), allow_balanced_ternary=True)
    assert receipt.canonical_ternary_syndrome == (2, 0, 1)
    assert set(receipt.canonical_ternary_syndrome).issubset({0, 1, 2})


def test_balanced_ternary_rejected_without_explicit_opt_in() -> None:
    with pytest.raises(ValueError, match="canonical basis"):
        run_ternary_decode_lane((-1, 0, 1))


def test_ranking_tie_break_order_is_explicit_and_deterministic() -> None:
    left = TernaryDecodeCandidate(
        candidate_id="cand_a",
        proposed_correction=(0, 0),
        correction_weight=0,
        syndrome_mismatch_count=0,
        metric_bundle={
            "syndrome_match_score": 1.0,
            "correction_sparsity_score": 1.0,
            "gf3_consistency_score": 1.0,
            "hardware_lane_readiness": 1.0,
            "bounded_confidence": 1.0,
        },
        composite_score=1.0,
    )
    right = TernaryDecodeCandidate(
        candidate_id="cand_b",
        proposed_correction=(0, 0),
        correction_weight=0,
        syndrome_mismatch_count=0,
        metric_bundle={
            "syndrome_match_score": 1.0,
            "correction_sparsity_score": 1.0,
            "gf3_consistency_score": 1.0,
            "hardware_lane_readiness": 1.0,
            "bounded_confidence": 1.0,
        },
        composite_score=1.0,
    )
    ranked = _rank_candidates((right, left))
    assert tuple(c.candidate_id for c in ranked) == ("cand_a", "cand_b")


def test_ranking_final_tie_break_uses_correction_tuple() -> None:
    first = TernaryDecodeCandidate(
        candidate_id="cand_z",
        proposed_correction=(0, 1),
        correction_weight=1,
        syndrome_mismatch_count=1,
        metric_bundle={
            "syndrome_match_score": 0.5,
            "correction_sparsity_score": 0.5,
            "gf3_consistency_score": 0.5,
            "hardware_lane_readiness": 1.0,
            "bounded_confidence": 0.625,
        },
        composite_score=0.55,
    )
    second = TernaryDecodeCandidate(
        candidate_id="cand_z",
        proposed_correction=(1, 0),
        correction_weight=1,
        syndrome_mismatch_count=1,
        metric_bundle={
            "syndrome_match_score": 0.5,
            "correction_sparsity_score": 0.5,
            "gf3_consistency_score": 0.5,
            "hardware_lane_readiness": 1.0,
            "bounded_confidence": 0.625,
        },
        composite_score=0.55,
    )
    ranked = _rank_candidates((second, first))
    assert tuple(c.proposed_correction for c in ranked) == ((0, 1), (1, 0))


def test_metrics_and_composite_score_are_bounded() -> None:
    receipt = run_ternary_decode_lane((2, 1, 0, 2))
    for value in receipt.selected_metric_bundle.values():
        assert 0.0 <= value <= 1.0
    assert 0.0 <= receipt.selected_composite_score <= 1.0


def test_receipt_invariants_and_hash_exclusion_behavior() -> None:
    receipt = run_ternary_decode_lane((0, 2, 1))
    assert receipt.release_version == "v138.4.0"
    assert receipt.lane_kind == "ternary_decode_lane"
    assert receipt.advisory_only is True
    assert receipt.decoder_core_modified is False
    assert receipt.receipt_hash == receipt.stable_hash()


def test_selected_metric_bundle_is_immutable_and_hash_stable() -> None:
    receipt = run_ternary_decode_lane((0, 2, 1))
    baseline_hash = receipt.receipt_hash
    with pytest.raises(TypeError):
        receipt.selected_metric_bundle["bounded_confidence"] = 0.0
    assert receipt.receipt_hash == baseline_hash
    assert receipt.stable_hash() == baseline_hash


def test_candidate_validation_rejects_malformed_correction_symbol() -> None:
    with pytest.raises(ValueError, match=r"proposed_correction\[1\] must be one of 0, 1, 2"):
        TernaryDecodeCandidate(
            candidate_id="cand_bad",
            proposed_correction=(0, 4),
            correction_weight=1,
            syndrome_mismatch_count=1,
            metric_bundle={
                "syndrome_match_score": 0.5,
                "correction_sparsity_score": 0.5,
                "gf3_consistency_score": 0.5,
                "hardware_lane_readiness": 0.5,
                "bounded_confidence": 0.5,
            },
            composite_score=0.5,
        )


def test_canonical_json_is_stable_and_ascii() -> None:
    receipt = run_ternary_decode_lane((0, 1, 2))
    payload = receipt.to_canonical_json()
    assert payload == receipt.to_canonical_json()
    assert "\n" not in payload
    assert payload.isascii()
