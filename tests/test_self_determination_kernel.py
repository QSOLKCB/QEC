"""Tests for v143.3 self-determination kernel.

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

import pytest

from qec.analysis.ensemble_consistency_engine import EnsembleClass, EnsembleConsistencyReceipt
from qec.analysis.generalized_invariant_detector import InvariantDecision, InvariantDetectionReceipt, InvariantPattern, InvariantSignal
from qec.analysis.invariant_geometry_embedding import InvariantClass, InvariantGeometryReceipt
from qec.analysis.self_determination_kernel import SelfDeterminationReceipt, TransitionOption, evaluate_self_determination_kernel
from qec.analysis.spectral_structure_kernel import evaluate_spectral_structure_kernel


def _invariant_receipt(*, support: int = 2) -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.2,
            fixed_point_score=0.7,
            plateau_score=0.1,
            oscillation_score=0.0,
            invariant_pressure=0.7,
        ),
        decision=InvariantDecision(
            dominant_invariant="fixed_point",
            invariant_rank=1,
            invariant_detected=True,
            invariant_confidence=0.7,
            rationale="fixed_point_detected",
        ),
        patterns=(InvariantPattern(pattern_type="fixed_point", key="S0", support=support, confidence=0.7),),
        control_mode="generalized_invariant_advisory",
        observatory_only=True,
    )


def _geometry_receipt(*, geometric: float = 0.9, stability: float = 0.9) -> InvariantGeometryReceipt:
    invariant_class = InvariantClass(
        class_id="fixed:cls",
        member_state_ids=("S0",),
        invariant_type="fixed_point",
        embedding_vector=(0.5, 0.5, 0.5, 0.5),
        invariant_signature="1" * 64,
    )
    return InvariantGeometryReceipt(
        version="v143.0",
        control_mode="invariant_geometry_embedding_advisory",
        observatory_only=True,
        convergence_label="stable",
        trace_length=1,
        invariant_classes=(invariant_class,),
        embedding_dimension=4,
        class_count=1,
        geometric_consistency_score=geometric,
        embedding_stability_score=stability,
    )


def _ensemble_receipt(
    classes: tuple[tuple[str, tuple[float, ...], float], ...],
    *,
    invariant_hash: str,
    global_consistency_score: float = 0.9,
) -> EnsembleConsistencyReceipt:
    ensembles = tuple(
        EnsembleClass(
            class_id=class_id,
            member_state_ids=(f"{class_id}:state",),
            centroid_vector=centroid_vector,
            max_deviation=mean_deviation,
            mean_deviation=mean_deviation,
            consistency_label="consistent",
        )
        for class_id, centroid_vector, mean_deviation in classes
    )
    sorted_ensembles = tuple(sorted(ensembles, key=lambda item: item.class_id))
    return EnsembleConsistencyReceipt(
        ensembles=sorted_ensembles,
        ensemble_count=len(sorted_ensembles),
        global_consistency_score=global_consistency_score,
        inconsistent_count=0,
        invariant_receipt_stable_hash=invariant_hash,
        version="v143.1",
        control_mode="ensemble_consistency_engine_advisory",
        observatory_only=True,
    )


def _spectral_receipt(
    ensemble: EnsembleConsistencyReceipt,
    geometry: InvariantGeometryReceipt,
    invariant: InvariantDetectionReceipt,
):
    return evaluate_spectral_structure_kernel(ensemble, geometry, invariant)


def test_deterministic_replay_same_input_same_selected_transition() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (0.9, 0.6), 0.1), ("b", (0.8, 0.7), 0.1)),
        invariant_hash=invariant.stable_hash,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    first = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)
    second = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    assert first.selected_transition_id == second.selected_transition_id
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_no_admissible_transitions_graceful_fallback() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt(geometric=0.2, stability=0.2)
    ensemble = _ensemble_receipt(
        (("a", (1.0, 1.0), 0.9), ("b", (1.0, 1.0), 0.9), ("c", (1.0, 1.0), 0.9)),
        invariant_hash=invariant.stable_hash,
        global_consistency_score=0.1,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    receipt = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    assert receipt.transition_count > 0
    assert receipt.admissible_count == 0
    assert receipt.selected_transition_id == "no_admissible_transition"
    assert receipt.selected_transition_score == 0.0
    assert receipt.selection_confidence == 0.0


def test_transition_priorities_are_differentiated() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (0.5, 0.5), 0.0), ("b", (0.5, 0.5), 0.0)),
        invariant_hash=invariant.stable_hash,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    receipt = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    priorities = {transition.transition_id: transition.priority_score for transition in receipt.allowed_transitions}
    assert len(set(priorities.values())) > 1


def test_selection_is_score_driven_not_lexicographic_fallback() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (1.0, 1.0), 0.9), ("b", (1.0, 1.0), 0.9), ("c", (1.0, 1.0), 0.9)),
        invariant_hash=invariant.stable_hash,
        global_consistency_score=0.8,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    receipt = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    admissible_transition_ids = sorted(
        transition.transition_id for transition in receipt.allowed_transitions if transition.admissible
    )
    assert receipt.selected_transition_id == "safe_hold"
    assert receipt.selected_transition_id != admissible_transition_ids[0]


def test_selection_confidence_uses_admissible_subset_only() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt(geometric=0.25, stability=0.25)
    ensemble = _ensemble_receipt(
        (("a", (1.0, 1.0), 0.6), ("b", (1.0, 1.0), 0.6), ("c", (1.0, 1.0), 0.6)),
        invariant_hash=invariant.stable_hash,
        global_consistency_score=0.7,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    receipt = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    assert receipt.admissible_count > 0
    assert any(not transition.admissible for transition in receipt.allowed_transitions)
    assert receipt.selection_confidence == 1.0


def test_score_bounds_are_in_unit_interval() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (0.7, 0.9), 0.2), ("b", (0.8, 0.8), 0.1)),
        invariant_hash=invariant.stable_hash,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    receipt = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)

    assert 0.0 <= receipt.selected_transition_score <= 1.0
    assert 0.0 <= receipt.selection_confidence <= 1.0
    for transition in receipt.allowed_transitions:
        assert 0.0 <= transition.priority_score <= 1.0


def test_permutation_invariance() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    classes = (("b", (0.8, 0.7), 0.1), ("a", (0.9, 0.6), 0.1))
    ensemble_one = _ensemble_receipt(classes, invariant_hash=invariant.stable_hash)
    ensemble_two = _ensemble_receipt(tuple(reversed(classes)), invariant_hash=invariant.stable_hash)

    spectral_one = _spectral_receipt(ensemble_one, geometry, invariant)
    spectral_two = _spectral_receipt(ensemble_two, geometry, invariant)

    receipt_one = evaluate_self_determination_kernel(spectral_one, ensemble_one, geometry, invariant)
    receipt_two = evaluate_self_determination_kernel(spectral_two, ensemble_two, geometry, invariant)

    assert receipt_one.to_canonical_json() == receipt_two.to_canonical_json()
    assert receipt_one.stable_hash == receipt_two.stable_hash


def test_upstream_hash_dependency_changes_receipt_hash() -> None:
    invariant = _invariant_receipt()
    geometry_one = _geometry_receipt(geometric=0.9, stability=0.9)
    geometry_two = _geometry_receipt(geometric=0.8, stability=0.9)
    ensemble = _ensemble_receipt(
        (("a", (0.9, 0.6), 0.1), ("b", (0.8, 0.7), 0.1)),
        invariant_hash=invariant.stable_hash,
    )

    spectral_one = _spectral_receipt(ensemble, geometry_one, invariant)
    spectral_two = _spectral_receipt(ensemble, geometry_two, invariant)

    receipt_one = evaluate_self_determination_kernel(spectral_one, ensemble, geometry_one, invariant)
    receipt_two = evaluate_self_determination_kernel(spectral_two, ensemble, geometry_two, invariant)

    assert receipt_one.stable_hash != receipt_two.stable_hash


def test_stable_hash_invariant_to_omitted_optional_invariant_receipt() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (0.9, 0.6), 0.1), ("b", (0.8, 0.7), 0.1)),
        invariant_hash=invariant.stable_hash,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    with_optional = evaluate_self_determination_kernel(spectral, ensemble, geometry, invariant)
    without_optional = evaluate_self_determination_kernel(spectral, ensemble, geometry)

    assert with_optional.invariant_receipt_stable_hash == invariant.stable_hash
    assert without_optional.invariant_receipt_stable_hash == invariant.stable_hash
    assert with_optional.stable_hash == without_optional.stable_hash


def test_invalid_input_raises_value_error() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (("a", (0.9, 0.6), 0.1), ("b", (0.8, 0.7), 0.1)),
        invariant_hash=invariant.stable_hash,
    )
    spectral = _spectral_receipt(ensemble, geometry, invariant)

    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_self_determination_kernel(object(), ensemble, geometry, invariant)  # type: ignore[arg-type]

    mismatched_invariant = _invariant_receipt(support=3)
    with pytest.raises(ValueError, match="invariant receipt stable_hash mismatch"):
        evaluate_self_determination_kernel(spectral, ensemble, geometry, mismatched_invariant)


def test_invalid_selected_transition_id_is_rejected() -> None:
    transitions = (
        TransitionOption(
            transition_id="safe_hold",
            description="hold",
            priority_score=0.9,
            admissible=True,
        ),
    )

    with pytest.raises(ValueError, match="selected_transition_id not in allowed_transitions"):
        SelfDeterminationReceipt(
            allowed_transitions=transitions,
            selected_transition_id="not_a_real_transition",
            selected_transition_score=0.9,
            selection_confidence=1.0,
            posture_label="highly_coupled_dynamic",
            transition_count=1,
            admissible_count=1,
            spectral_receipt_stable_hash="1" * 64,
            ensemble_receipt_stable_hash="2" * 64,
            geometry_receipt_stable_hash="3" * 64,
            invariant_receipt_stable_hash="4" * 64,
            version="v143.3",
            control_mode="self_determination_advisory",
            observatory_only=True,
        )
