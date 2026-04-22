"""Tests for v143.0 invariant geometry embedding kernel.

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

import pytest

from qec.analysis.convergence_engine import ConvergenceDecision, ConvergenceReceipt, ConvergenceSignal
from qec.analysis.generalized_invariant_detector import (
    InvariantDecision,
    InvariantDetectionReceipt,
    InvariantPattern,
    InvariantSignal,
)
from qec.analysis.invariant_geometry_embedding import (
    INVARIANT_GEOMETRY_EMBEDDING_VERSION,
    evaluate_invariant_geometry_embedding,
)


def _invariant_receipt(patterns: tuple[InvariantPattern, ...]) -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.0,
            fixed_point_score=0.6,
            plateau_score=0.4,
            oscillation_score=0.7,
            invariant_pressure=0.7,
        ),
        decision=InvariantDecision(
            dominant_invariant="oscillation",
            invariant_rank=4,
            invariant_detected=True,
            invariant_confidence=0.7,
            rationale="oscillation_detected",
        ),
        patterns=patterns,
        control_mode="generalized_invariant_advisory",
        observatory_only=True,
    )


def _convergence_receipt() -> ConvergenceReceipt:
    return ConvergenceReceipt(
        version="v142.2",
        signal=ConvergenceSignal(
            mean_convergence=0.5,
            invariant_pressure=0.7,
            terminal_convergence=0.6,
            plateau_component=0.4,
            oscillation_component=0.7,
            convergence_pressure=0.55,
            efficiency_score=0.4,
        ),
        decision=ConvergenceDecision(
            convergence_label="oscillating",
            convergence_rank=4,
            early_termination_advised=False,
            termination_confidence=0.4,
            rationale="oscillating_detected",
        ),
        control_mode="convergence_engine_advisory",
        observatory_only=True,
    )


def test_deterministic_replay() -> None:
    patterns = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="plateau", key="plateau:1-4", support=4, confidence=0.6),
        InvariantPattern(pattern_type="oscillation", key="S0<->S1", support=5, confidence=0.8),
    )
    first = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt())
    second = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt())

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_invariant_equivalence_identical_embedding() -> None:
    patterns = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="fixed_point", key="S2", support=3, confidence=1.0),
    )

    receipt = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt())
    assert receipt.class_count == 1

    invariant_class = receipt.invariant_classes[0]
    replay = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt()).invariant_classes[0]
    assert invariant_class.embedding_vector == replay.embedding_vector
    assert invariant_class.invariant_signature == replay.invariant_signature


def test_permutation_invariance() -> None:
    ordered = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="plateau", key="plateau:1-4", support=4, confidence=0.6),
        InvariantPattern(pattern_type="oscillation", key="S0<->S1", support=5, confidence=0.8),
    )
    permuted = (
        ordered[2],
        ordered[0],
        ordered[1],
    )

    first = evaluate_invariant_geometry_embedding(_invariant_receipt(ordered), _convergence_receipt())
    second = evaluate_invariant_geometry_embedding(_invariant_receipt(permuted), _convergence_receipt())

    assert first.stable_hash == second.stable_hash
    assert tuple(c.invariant_signature for c in first.invariant_classes) == tuple(
        c.invariant_signature for c in second.invariant_classes
    )


def test_geometric_constraint_stability() -> None:
    patterns = (
        InvariantPattern(pattern_type="oscillation", key="S0<->S1", support=8, confidence=1.0),
    )

    receipt = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt())
    same = evaluate_invariant_geometry_embedding(_invariant_receipt(patterns), _convergence_receipt())

    assert receipt.embedding_stability_score == 1.0
    assert receipt.geometric_consistency_score == 1.0
    assert receipt.invariant_classes[0].embedding_vector == same.invariant_classes[0].embedding_vector


@pytest.mark.parametrize(
    "invariant_receipt,convergence_receipt,error_match",
    [
        (object(), _convergence_receipt(), "invalid input type"),
        (
            _invariant_receipt(()),
            object(),
            "invalid input type",
        ),
    ],
)
def test_invalid_input_raises_value_error(invariant_receipt: object, convergence_receipt: object, error_match: str) -> None:
    with pytest.raises(ValueError, match=error_match):
        evaluate_invariant_geometry_embedding(invariant_receipt, convergence_receipt)  # type: ignore[arg-type]


def test_metric_bounds_in_unit_interval() -> None:
    patterns = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="plateau", key="plateau:1-4", support=4, confidence=0.6),
    )

    receipt = evaluate_invariant_geometry_embedding(
        _invariant_receipt(patterns),
        _convergence_receipt(),
        version=INVARIANT_GEOMETRY_EMBEDDING_VERSION,
    )

    assert 0.0 <= receipt.geometric_consistency_score <= 1.0
    assert 0.0 <= receipt.embedding_stability_score <= 1.0
    assert receipt.embedding_dimension == 4
