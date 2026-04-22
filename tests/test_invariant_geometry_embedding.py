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
    InvariantClass,
    evaluate_invariant_geometry_embedding,
)
from qec.analysis.iterative_system_abstraction_layer import (
    IterativeExecutionTrace,
    IterativeStateSnapshot,
    IterativeTransition,
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


def _execution_trace(total_steps: int = 3) -> IterativeExecutionTrace:
    snapshots = tuple(
        IterativeStateSnapshot(
            step_index=index,
            state_id=f"S{index}",
            state_payload={"index": index},
            convergence_metric=min(1.0, 0.4 + 0.2 * index),
            active=index == total_steps - 1,
        )
        for index in range(total_steps)
    )
    transitions = tuple(
        IterativeTransition(
            from_state_id=f"S{index}",
            to_state_id=f"S{index + 1}",
            delta_magnitude=0.2,
            transition_label="advance",
        )
        for index in range(max(total_steps - 1, 0))
    )
    return IterativeExecutionTrace(
        snapshots=snapshots,
        transitions=transitions,
        total_steps=total_steps,
        final_state_id=snapshots[-1].state_id if snapshots else "",
        mean_convergence=0.6,
        converged=False,
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


def test_receipt_schema_and_input_usage() -> None:
    patterns = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="plateau", key="plateau:1-4", support=4, confidence=0.6),
    )
    trace = _execution_trace(4)

    receipt = evaluate_invariant_geometry_embedding(
        _invariant_receipt(patterns),
        _convergence_receipt(),
        execution_trace=trace,
        version="v143.0-test",
    )

    assert receipt.version == "v143.0-test"
    assert receipt.control_mode == "invariant_geometry_embedding_advisory"
    assert receipt.observatory_only is True
    assert receipt.convergence_label == "oscillating"
    assert receipt.trace_length == 4
    payload = receipt.to_dict()
    assert payload["version"] == "v143.0-test"
    assert payload["control_mode"] == "invariant_geometry_embedding_advisory"
    assert payload["observatory_only"] is True
    assert payload["convergence_label"] == "oscillating"
    assert payload["trace_length"] == 4


def test_geometric_consistency_varies_across_inputs() -> None:
    one_pattern = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
    )
    three_patterns = (
        InvariantPattern(pattern_type="fixed_point", key="S2", support=2, confidence=1.0),
        InvariantPattern(pattern_type="plateau", key="plateau:1-4", support=4, confidence=0.6),
        InvariantPattern(pattern_type="oscillation", key="S0<->S1", support=5, confidence=0.8),
    )

    score_one = evaluate_invariant_geometry_embedding(_invariant_receipt(one_pattern), _convergence_receipt()).geometric_consistency_score
    score_three = evaluate_invariant_geometry_embedding(_invariant_receipt(three_patterns), _convergence_receipt()).geometric_consistency_score

    assert score_one != score_three


def test_invariant_signature_requires_lowercase_hex() -> None:
    valid = InvariantClass(
        class_id="fixed_point:abc",
        member_state_ids=("S1",),
        invariant_type="fixed_point",
        embedding_vector=(0.1, 0.2, 0.3, 0.4),
        invariant_signature="a" * 64,
    )
    assert valid.invariant_signature == "a" * 64

    with pytest.raises(ValueError, match="lowercase sha256 hex"):
        InvariantClass(
            class_id="fixed_point:def",
            member_state_ids=("S2",),
            invariant_type="fixed_point",
            embedding_vector=(0.1, 0.2, 0.3, 0.4),
            invariant_signature="A" * 64,
        )


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
