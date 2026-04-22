"""Tests for v143.1 ensemble consistency engine.

Attribution:
This module incorporates concepts from:
Marc Brendecke (2026)
Quantum Sphaera Companion v3.30.0
DOI: https://doi.org/10.5281/zenodo.19682951
License: CC-BY-4.0
"""

from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.ensemble_consistency_engine import (
    ENSEMBLE_CONSISTENCY_ENGINE_VERSION,
    evaluate_ensemble_consistency_engine,
)
from qec.analysis.generalized_invariant_detector import (
    InvariantDecision,
    InvariantDetectionReceipt,
    InvariantPattern,
    InvariantSignal,
)
from qec.analysis.invariant_geometry_embedding import InvariantClass, InvariantGeometryReceipt
from qec.analysis.iterative_system_abstraction_layer import (
    IterativeExecutionTrace,
    IterativeStateSnapshot,
    IterativeTransition,
)


def _invariant_receipt() -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.0,
            fixed_point_score=0.7,
            plateau_score=0.0,
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
        patterns=(InvariantPattern(pattern_type="fixed_point", key="S0", support=2, confidence=0.7),),
        control_mode="generalized_invariant_advisory",
        observatory_only=True,
    )


def _signature(invariant_type: str, member_state_ids: tuple[str, ...]) -> str:
    return sha256_hex({"invariant_type": invariant_type, "member_state_ids": member_state_ids})


def _geometry_receipt(member_state_ids: tuple[str, ...]) -> InvariantGeometryReceipt:
    signature = _signature("fixed_point", member_state_ids)
    invariant_class = InvariantClass(
        class_id="fixed_point:cls",
        member_state_ids=member_state_ids,
        invariant_type="fixed_point",
        embedding_vector=(0.5, 0.5, 0.5, 0.5),
        invariant_signature=signature,
    )
    return InvariantGeometryReceipt(
        version="v143.0",
        control_mode="invariant_geometry_embedding_advisory",
        observatory_only=True,
        convergence_label="stable",
        trace_length=None,
        invariant_classes=(invariant_class,),
        embedding_dimension=4,
        class_count=1,
        geometric_consistency_score=1.0,
        embedding_stability_score=1.0,
    )


def _trace(vectors: dict[str, tuple[float, ...]]) -> IterativeExecutionTrace:
    items = tuple(vectors.items())
    snapshots = tuple(
        IterativeStateSnapshot(
            step_index=index,
            state_id=state_id,
            state_payload={"embedding_vector": vector},
            convergence_metric=0.5,
            active=index == (len(vectors) - 1),
        )
        for index, (state_id, vector) in enumerate(items)
    )
    transitions = tuple(
        IterativeTransition(
            from_state_id=items[index][0],
            to_state_id=items[index + 1][0],
            delta_magnitude=0.1,
            transition_label="advance",
        )
        for index in range(max(len(items) - 1, 0))
    )
    return IterativeExecutionTrace(
        snapshots=snapshots,
        transitions=transitions,
        total_steps=len(snapshots),
        final_state_id=snapshots[-1].state_id if snapshots else "",
        mean_convergence=0.5,
        converged=False,
    )


def test_deterministic_replay() -> None:
    geometry = _geometry_receipt(("S0", "S1"))
    trace = _trace({"S0": (0.5, 0.5, 0.5, 0.5), "S1": (0.6, 0.5, 0.5, 0.5)})

    first = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace)
    second = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_identical_embeddings_fully_consistent() -> None:
    geometry = _geometry_receipt(("S0", "S1"))
    trace = _trace({"S0": (0.5, 0.5, 0.5, 0.5), "S1": (0.5, 0.5, 0.5, 0.5)})

    receipt = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace)
    assert receipt.ensembles[0].max_deviation == 0.0
    assert receipt.ensembles[0].consistency_label == "fully_consistent"


def test_small_deviation_consistent() -> None:
    geometry = _geometry_receipt(("S0", "S1"))
    trace = _trace({"S0": (0.5, 0.5, 0.5, 0.5), "S1": (0.502, 0.5, 0.5, 0.5)})

    receipt = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace, epsilon=0.01)
    assert receipt.ensembles[0].max_deviation > 0.0
    assert receipt.ensembles[0].consistency_label == "consistent"


def test_large_deviation_inconsistent() -> None:
    geometry = _geometry_receipt(("S0", "S1"))
    trace = _trace({"S0": (0.0, 0.0, 0.0, 0.0), "S1": (1.0, 1.0, 1.0, 1.0)})

    receipt = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace, epsilon=0.1)
    assert receipt.ensembles[0].max_deviation >= 0.5
    assert receipt.ensembles[0].consistency_label == "inconsistent"


def test_permutation_invariance() -> None:
    geometry = _geometry_receipt(("S0", "S1", "S2"))
    trace_one = _trace({"S0": (0.1, 0.2, 0.3, 0.4), "S1": (0.2, 0.3, 0.4, 0.5), "S2": (0.3, 0.4, 0.5, 0.6)})
    trace_two = _trace({"S2": (0.3, 0.4, 0.5, 0.6), "S1": (0.2, 0.3, 0.4, 0.5), "S0": (0.1, 0.2, 0.3, 0.4)})

    receipt_one = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace_one)
    receipt_two = evaluate_ensemble_consistency_engine(geometry, _invariant_receipt(), execution_trace=trace_two)

    assert receipt_one.stable_hash == receipt_two.stable_hash
    assert receipt_one.to_canonical_json() == receipt_two.to_canonical_json()


def test_metric_bounds_in_unit_interval() -> None:
    geometry = _geometry_receipt(("S0", "S1"))
    trace = _trace({"S0": (0.2, 0.2, 0.2, 0.2), "S1": (0.8, 0.8, 0.8, 0.8)})

    receipt = evaluate_ensemble_consistency_engine(
        geometry,
        _invariant_receipt(),
        execution_trace=trace,
        version=ENSEMBLE_CONSISTENCY_ENGINE_VERSION,
    )

    assert 0.0 <= receipt.global_consistency_score <= 1.0
    assert 0.0 <= receipt.ensembles[0].max_deviation <= 1.0
    assert 0.0 <= receipt.ensembles[0].mean_deviation <= 1.0


@pytest.mark.parametrize(
    "geometry,invariant,trace,epsilon,error_match",
    [
        (object(), _invariant_receipt(), None, 1e-3, "invalid input type"),
        (_geometry_receipt(("S0",)), object(), None, 1e-3, "invalid input type"),
        (_geometry_receipt(("S0",)), _invariant_receipt(), object(), 1e-3, "invalid input type"),
        (_geometry_receipt(("S0",)), _invariant_receipt(), None, -0.1, r"epsilon must be in \[0,1\]"),
    ],
)
def test_invalid_input_raises_value_error(
    geometry: object,
    invariant: object,
    trace: object,
    epsilon: float,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        evaluate_ensemble_consistency_engine(
            geometry,  # type: ignore[arg-type]
            invariant,  # type: ignore[arg-type]
            execution_trace=trace,  # type: ignore[arg-type]
            epsilon=epsilon,
        )
