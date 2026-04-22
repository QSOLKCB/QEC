from __future__ import annotations

import pytest

from qec.analysis.convergence_engine import CONVERGENCE_ENGINE_VERSION, evaluate_convergence_engine
from qec.analysis.generalized_invariant_detector import (
    GENERALIZED_INVARIANT_DETECTOR_VERSION,
    evaluate_generalized_invariant_detector,
)
from qec.analysis.iterative_system_abstraction_layer import (
    IterativeStateSnapshot,
    evaluate_iterative_system_abstraction,
)


def _snapshot(step_index: int, state_id: str, convergence_metric: float) -> IterativeStateSnapshot:
    return IterativeStateSnapshot(
        step_index=step_index,
        state_id=state_id,
        state_payload={"id": state_id, "step": step_index},
        convergence_metric=convergence_metric,
        active=True,
    )


def _eval(*snapshots: IterativeStateSnapshot):
    execution = evaluate_iterative_system_abstraction(tuple(snapshots))
    invariant = evaluate_generalized_invariant_detector(
        execution,
        version=GENERALIZED_INVARIANT_DETECTOR_VERSION,
    )
    return evaluate_convergence_engine(execution, invariant, version=CONVERGENCE_ENGINE_VERSION)


def test_deterministic_replay_identical_json_and_hash() -> None:
    first = _eval(
        _snapshot(0, "A", 0.2),
        _snapshot(1, "B", 0.4),
        _snapshot(2, "C", 0.6),
    )
    second = _eval(
        _snapshot(0, "A", 0.2),
        _snapshot(1, "B", 0.4),
        _snapshot(2, "C", 0.6),
    )

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_empty_trace_unconverged_no_early_termination() -> None:
    out = _eval()

    assert out.signal.terminal_convergence == 0.0
    assert out.decision.convergence_label == "unconverged"
    assert out.decision.early_termination_advised is False


def test_oscillating_path_from_oscillation_component_threshold() -> None:
    out = _eval(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.20),
        _snapshot(2, "A", 0.30),
        _snapshot(3, "B", 0.40),
        _snapshot(4, "A", 0.50),
    )

    assert out.signal.oscillation_component >= 0.5
    assert out.decision.convergence_label == "oscillating"


def test_converged_path_trace_converged_true() -> None:
    out = _eval(
        _snapshot(0, "A", 0.90),
        _snapshot(1, "B", 0.9995),
    )

    assert out.decision.convergence_label == "converged"
    assert out.decision.convergence_rank == 3


def test_near_converged_path_terminal_high_without_trace_converged() -> None:
    out = _eval(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.95),
    )

    assert out.decision.convergence_label == "near_converged"
    assert out.decision.convergence_rank == 2


def test_progressing_path_moderate_convergence_pressure() -> None:
    out = _eval(
        _snapshot(0, "A", 0.20),
        _snapshot(1, "B", 0.80),
    )

    assert out.signal.convergence_pressure >= 0.40
    assert out.signal.terminal_convergence < 0.90
    assert out.decision.convergence_label == "progressing"


def test_unconverged_path_low_metrics() -> None:
    out = _eval(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.12),
    )

    assert out.signal.convergence_pressure < 0.40
    assert out.decision.convergence_label == "unconverged"


def test_early_termination_advisory_gated_by_label_and_oscillation() -> None:
    near = _eval(
        _snapshot(0, "A", 0.20),
        _snapshot(1, "B", 0.95),
    )
    converged = _eval(
        _snapshot(0, "A", 0.90),
        _snapshot(1, "B", 0.9995),
    )
    oscillating = _eval(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.20),
        _snapshot(2, "A", 0.30),
        _snapshot(3, "B", 0.40),
        _snapshot(4, "A", 0.50),
    )

    assert near.decision.early_termination_advised is True
    assert converged.decision.early_termination_advised is True
    assert oscillating.decision.early_termination_advised is False


def test_efficiency_decreases_with_higher_oscillation() -> None:
    low_osc = _eval(
        _snapshot(0, "A", 0.2),
        _snapshot(1, "B", 0.4),
        _snapshot(2, "C", 0.6),
    )
    high_osc = _eval(
        _snapshot(0, "A", 0.2),
        _snapshot(1, "B", 0.4),
        _snapshot(2, "A", 0.6),
        _snapshot(3, "B", 0.7),
        _snapshot(4, "A", 0.8),
    )

    assert high_osc.signal.oscillation_component > low_osc.signal.oscillation_component
    assert high_osc.signal.efficiency_score < low_osc.signal.efficiency_score


def test_validation_invalid_input_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_convergence_engine(object(), object(), version=CONVERGENCE_ENGINE_VERSION)  # type: ignore[arg-type]


def test_hash_stability_repeated_runs_identical_stable_hash() -> None:
    execution = evaluate_iterative_system_abstraction(
        (
            _snapshot(0, "A", 0.3),
            _snapshot(1, "B", 0.6),
            _snapshot(2, "C", 0.8),
        )
    )
    invariant = evaluate_generalized_invariant_detector(execution, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    first = evaluate_convergence_engine(execution, invariant, version=CONVERGENCE_ENGINE_VERSION)
    second = evaluate_convergence_engine(execution, invariant, version=CONVERGENCE_ENGINE_VERSION)
    assert first.stable_hash == second.stable_hash


def test_canonical_serialization_replay_safe() -> None:
    out = _eval(
        _snapshot(0, "A", 0.3),
        _snapshot(1, "B", 0.6),
    )

    assert out.to_canonical_json() == out.to_canonical_json()
