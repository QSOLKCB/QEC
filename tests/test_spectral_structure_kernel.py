"""Tests for v143.2 spectral structure kernel.

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

import math

import pytest

from qec.analysis.ensemble_consistency_engine import EnsembleClass, EnsembleConsistencyReceipt
from qec.analysis.generalized_invariant_detector import InvariantDecision, InvariantDetectionReceipt, InvariantPattern, InvariantSignal
from qec.analysis.invariant_geometry_embedding import InvariantClass, InvariantGeometryReceipt
from qec.analysis.spectral_structure_kernel import (
    SPECTRAL_STRUCTURE_KERNEL_VERSION,
    _build_ensemble_operator,
    evaluate_spectral_structure_kernel,
)


def _invariant_receipt(*, support: int = 2) -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.1,
            fixed_point_score=0.8,
            plateau_score=0.1,
            oscillation_score=0.0,
            invariant_pressure=0.8,
        ),
        decision=InvariantDecision(
            dominant_invariant="fixed_point",
            invariant_rank=1,
            invariant_detected=True,
            invariant_confidence=0.8,
            rationale="fixed_point_detected",
        ),
        patterns=(InvariantPattern(pattern_type="fixed_point", key="S0", support=support, confidence=0.8),),
        control_mode="generalized_invariant_advisory",
        observatory_only=True,
    )


def _geometry_receipt() -> InvariantGeometryReceipt:
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
        geometric_consistency_score=1.0,
        embedding_stability_score=1.0,
    )


def _ensemble_receipt(
    classes: tuple[tuple[str, tuple[float, ...], float], ...],
    *,
    invariant_hash: str,
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
    ensembles_sorted = tuple(sorted(ensembles, key=lambda item: item.class_id))
    return EnsembleConsistencyReceipt(
        ensembles=ensembles_sorted,
        ensemble_count=len(ensembles_sorted),
        global_consistency_score=1.0,
        inconsistent_count=0,
        invariant_receipt_stable_hash=invariant_hash,
        version="v143.1",
        control_mode="ensemble_consistency_engine_advisory",
        observatory_only=True,
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.9, 0.8), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )

    first = evaluate_spectral_structure_kernel(ensemble, geometry, invariant)
    second = evaluate_spectral_structure_kernel(ensemble, geometry, invariant)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_symmetric_operator_construction_and_stability() -> None:
    invariant = _invariant_receipt()
    ensemble = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.75, 0.933012701892), 0.1),
            ("c", (0.5, 1.0), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )

    matrix = _build_ensemble_operator(ensemble.ensembles)
    assert matrix.shape == (3, 3)
    assert (matrix.T == matrix).all()
    assert float(matrix.min()) >= -1.0
    assert float(matrix.max()) <= 1.0


def test_bounded_descriptor_scores() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.9, 0.8), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )

    receipt = evaluate_spectral_structure_kernel(ensemble, geometry, invariant)

    assert 0.0 <= receipt.spectral_dispersion_score <= 1.0
    assert 0.0 <= receipt.coupling_density_score <= 1.0
    assert 0.0 <= receipt.diagonal_dominance_score <= 1.0
    assert 0.0 <= receipt.ensemble_symmetry_score <= 1.0


def test_label_classification_paths() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()

    rigid = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.01),
            ("b", (0.5, 1.0), 0.01),
        ),
        invariant_hash=invariant.stable_hash,
    )
    structured = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.5),
            ("b", (0.75, 0.933012701892), 0.5),
        ),
        invariant_hash=invariant.stable_hash,
    )
    coupled = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.0),
            ("b", (0.9, 0.8), 0.0),
        ),
        invariant_hash=invariant.stable_hash,
    )
    highly_coupled = _ensemble_receipt(
        (
            ("a", (1.0, 1.0), 0.9),
            ("b", (1.0, 1.0), 0.9),
            ("c", (1.0, 1.0), 0.9),
        ),
        invariant_hash=invariant.stable_hash,
    )

    assert evaluate_spectral_structure_kernel(rigid, geometry, invariant).dynamics_label == "rigid"
    assert evaluate_spectral_structure_kernel(structured, geometry, invariant).dynamics_label == "structured"
    assert evaluate_spectral_structure_kernel(coupled, geometry, invariant).dynamics_label == "coupled"
    assert evaluate_spectral_structure_kernel(highly_coupled, geometry, invariant).dynamics_label == "highly_coupled"


def test_permutation_invariance_of_ensemble_ordering() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()

    classes = (
        ("b", (0.9, 0.8), 0.1),
        ("a", (1.0, 0.5), 0.1),
    )
    first = _ensemble_receipt(classes, invariant_hash=invariant.stable_hash)
    second = _ensemble_receipt(tuple(reversed(classes)), invariant_hash=invariant.stable_hash)

    receipt_one = evaluate_spectral_structure_kernel(first, geometry, invariant)
    receipt_two = evaluate_spectral_structure_kernel(second, geometry, invariant)

    assert receipt_one.to_canonical_json() == receipt_two.to_canonical_json()
    assert receipt_one.stable_hash == receipt_two.stable_hash


def test_invalid_input_types_raise_value_error() -> None:
    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_spectral_structure_kernel(object(), _geometry_receipt())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_spectral_structure_kernel(_ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=_invariant_receipt().stable_hash), object())  # type: ignore[arg-type]


def test_invariant_receipt_mismatch_raises() -> None:
    invariant = _invariant_receipt(support=2)
    mismatched_invariant = _invariant_receipt(support=3)
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.9, 0.8), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )

    with pytest.raises(
        ValueError,
        match=(
            "invariant receipt stable_hash does not match "
            "ensemble_receipt.invariant_receipt_stable_hash"
        ),
    ):
        evaluate_spectral_structure_kernel(ensemble, geometry, mismatched_invariant)


def test_upstream_hash_dependency_changes_receipt_hash() -> None:
    invariant = _invariant_receipt()
    geometry_one = _geometry_receipt()
    geometry_two = InvariantGeometryReceipt(
        version=geometry_one.version,
        control_mode=geometry_one.control_mode,
        observatory_only=geometry_one.observatory_only,
        convergence_label="metastable",
        trace_length=geometry_one.trace_length,
        invariant_classes=geometry_one.invariant_classes,
        embedding_dimension=geometry_one.embedding_dimension,
        class_count=geometry_one.class_count,
        geometric_consistency_score=geometry_one.geometric_consistency_score,
        embedding_stability_score=geometry_one.embedding_stability_score,
    )
    ensemble_one = _ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=invariant.stable_hash)
    ensemble_two = _ensemble_receipt((("a", (0.5, 0.5), 0.2),), invariant_hash=invariant.stable_hash)

    receipt_one = evaluate_spectral_structure_kernel(ensemble_one, geometry_one, invariant)
    receipt_two = evaluate_spectral_structure_kernel(ensemble_one, geometry_two, invariant)
    receipt_three = evaluate_spectral_structure_kernel(ensemble_two, geometry_one, invariant)

    assert receipt_one.geometry_receipt_stable_hash != receipt_two.geometry_receipt_stable_hash
    assert receipt_one.ensemble_receipt_stable_hash != receipt_three.ensemble_receipt_stable_hash
    assert receipt_one.stable_hash != receipt_two.stable_hash
    assert receipt_one.stable_hash != receipt_three.stable_hash


def test_ensemble_symmetry_score_varies_across_inputs() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    reflection_symmetric = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.8, 0.8), 0.1),
            ("c", (0.5, 1.0), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )
    reflection_asymmetric = _ensemble_receipt(
        (
            ("a", (1.0, 0.5), 0.1),
            ("b", (0.6, 0.9), 0.1),
            ("c", (0.7, 0.6), 0.1),
        ),
        invariant_hash=invariant.stable_hash,
    )

    symmetric_receipt = evaluate_spectral_structure_kernel(reflection_symmetric, geometry, invariant)
    asymmetric_receipt = evaluate_spectral_structure_kernel(reflection_asymmetric, geometry, invariant)

    assert symmetric_receipt.ensemble_symmetry_score > asymmetric_receipt.ensemble_symmetry_score


def test_stable_hash_depends_on_invariant_receipt_hash() -> None:
    invariant_one = _invariant_receipt(support=2)
    invariant_two = _invariant_receipt(support=3)
    geometry = _geometry_receipt()
    ensemble_one = _ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=invariant_one.stable_hash)
    ensemble_two = _ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=invariant_two.stable_hash)

    receipt_one = evaluate_spectral_structure_kernel(ensemble_one, geometry, invariant_one)
    receipt_two = evaluate_spectral_structure_kernel(ensemble_two, geometry, invariant_two)

    assert receipt_one.invariant_receipt_stable_hash == invariant_one.stable_hash
    assert receipt_two.invariant_receipt_stable_hash == invariant_two.stable_hash
    assert receipt_one.stable_hash != receipt_two.stable_hash


def test_output_finite_value_enforcement() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=invariant.stable_hash)

    receipt = evaluate_spectral_structure_kernel(ensemble, geometry, invariant)

    numeric_values = (
        receipt.trace_value,
        receipt.frobenius_norm,
        receipt.diagonal_energy,
        receipt.off_diagonal_energy,
        receipt.spectral_radius_proxy,
        receipt.spectral_gap_proxy,
        receipt.spectral_dispersion_score,
        receipt.coupling_density_score,
        receipt.diagonal_dominance_score,
        receipt.ensemble_symmetry_score,
    )
    assert all(math.isfinite(value) for value in numeric_values)


def test_version_passthrough() -> None:
    invariant = _invariant_receipt()
    geometry = _geometry_receipt()
    ensemble = _ensemble_receipt((("a", (0.5, 0.5), 0.1),), invariant_hash=invariant.stable_hash)

    receipt = evaluate_spectral_structure_kernel(
        ensemble,
        geometry,
        invariant,
        version=SPECTRAL_STRUCTURE_KERNEL_VERSION,
    )
    assert receipt.version == SPECTRAL_STRUCTURE_KERNEL_VERSION
