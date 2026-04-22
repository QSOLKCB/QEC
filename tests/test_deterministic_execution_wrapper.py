from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.canonical_hashing import canonical_json
from qec.analysis.convergence_engine import ConvergenceDecision, ConvergenceReceipt, ConvergenceSignal
from qec.analysis.deterministic_execution_wrapper import (
    DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    evaluate_deterministic_execution_wrapper,
)
from qec.analysis.generalized_invariant_detector import (
    InvariantDecision,
    InvariantDetectionReceipt,
    InvariantSignal,
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


def _execution(total_steps: int = 2):
    snapshots = tuple(_snapshot(i, f"S{i}", 0.1 + 0.1 * i) for i in range(total_steps))
    return evaluate_iterative_system_abstraction(snapshots)


def _invariant_receipt(invariant_pressure: float, oscillation_score: float) -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.0,
            fixed_point_score=0.0,
            plateau_score=0.0,
            oscillation_score=oscillation_score,
            invariant_pressure=invariant_pressure,
        ),
        decision=InvariantDecision(
            dominant_invariant="none",
            invariant_rank=0,
            invariant_detected=False,
            invariant_confidence=0.0,
            rationale="no_invariant",
        ),
        patterns=(),
        control_mode="generalized_invariant_advisory",
        observatory_only=True,
    )


def _convergence_receipt(
    convergence_pressure: float,
    terminal_convergence: float,
    efficiency_score: float,
    *,
    early_termination_advised: bool,
    termination_confidence: float,
    convergence_label: str = "progressing",
) -> ConvergenceReceipt:
    return ConvergenceReceipt(
        version="v142.2",
        signal=ConvergenceSignal(
            mean_convergence=0.0,
            invariant_pressure=0.0,
            terminal_convergence=terminal_convergence,
            plateau_component=0.0,
            oscillation_component=0.0,
            convergence_pressure=convergence_pressure,
            efficiency_score=efficiency_score,
        ),
        decision=ConvergenceDecision(
            convergence_label=convergence_label,
            convergence_rank={
                "unconverged": 0,
                "progressing": 1,
                "near_converged": 2,
                "converged": 3,
                "oscillating": 4,
            }[convergence_label],
            early_termination_advised=early_termination_advised,
            termination_confidence=termination_confidence,
            rationale={
                "unconverged": "unconverged_detected",
                "progressing": "progressing_detected",
                "near_converged": "near_converged_detected",
                "converged": "converged_detected",
                "oscillating": "oscillating_detected",
            }[convergence_label],
        ),
        control_mode="convergence_engine_advisory",
        observatory_only=True,
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    execution = _execution(total_steps=2)
    invariant = _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.0)
    convergence = _convergence_receipt(0.10, 0.10, 0.10, early_termination_advised=False, termination_confidence=0.10)

    first = evaluate_deterministic_execution_wrapper(execution, invariant, convergence, version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION)
    second = evaluate_deterministic_execution_wrapper(execution, invariant, convergence, version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_continue_path_low_pressures() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=2),
        _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.0),
        _convergence_receipt(0.10, 0.10, 0.10, early_termination_advised=False, termination_confidence=0.10),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.decision.execution_label == "continue"
    assert out.decision.execution_rank == 0
    assert out.plan.allowed_next_steps == 3
    assert out.plan.pruning_budget == 0.0
    assert out.plan.canonical_output_mode == "full"


def test_gate_path_moderate_gating_pressure() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=2),
        _invariant_receipt(invariant_pressure=0.20, oscillation_score=0.0),
        _convergence_receipt(0.60, 0.30, 0.30, early_termination_advised=False, termination_confidence=0.20),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.signal.gating_pressure >= 0.40
    assert out.signal.pruning_pressure < 0.70
    assert out.decision.execution_label == "gate"
    assert out.plan.allowed_next_steps == 2


def test_prune_path_high_pruning_pressure() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=2),
        _invariant_receipt(invariant_pressure=1.0, oscillation_score=0.0),
        _convergence_receipt(0.20, 0.20, 1.0, early_termination_advised=False, termination_confidence=0.20),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.signal.pruning_pressure >= 0.70
    assert out.decision.execution_label == "prune"
    assert out.decision.pruning_enabled is True
    assert out.plan.canonical_output_mode == "reduced"


def test_terminate_advisory_path() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=3),
        _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.0),
        _convergence_receipt(0.20, 0.30, 0.20, early_termination_advised=True, termination_confidence=0.90),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.decision.execution_label == "terminate_advisory"
    assert out.decision.early_termination_advised is True
    assert out.plan.allowed_next_steps == 0
    assert out.plan.pruning_budget == 0.0
    assert out.plan.state_retention_budget == 1.0
    assert out.plan.canonical_output_mode == "terminal"


def test_oscillation_hold_path_precedence() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=3),
        _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.60),
        _convergence_receipt(0.20, 0.30, 0.20, early_termination_advised=True, termination_confidence=0.90),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.signal.oscillation_component >= 0.5
    assert out.decision.execution_label == "oscillation_hold"
    assert out.decision.pruning_enabled is True
    assert out.plan.allowed_next_steps == 1
    assert out.plan.pruning_budget > 0.0
    assert out.plan.canonical_output_mode == "reduced"


def test_standardization_and_plan_signature_deterministic() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=1),
        _invariant_receipt(invariant_pressure=0.20, oscillation_score=0.0),
        _convergence_receipt(0.60, 0.30, 0.30, early_termination_advised=False, termination_confidence=0.20),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert out.decision.output_standardized is True
    assert out.plan.plan_signature == f"{out.decision.execution_label}::{out.plan.canonical_output_mode}::{out.plan.allowed_next_steps}"


def test_oscillation_reduces_wrapper_confidence() -> None:
    base = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=2),
        _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.0),
        _convergence_receipt(0.20, 0.30, 0.20, early_termination_advised=False, termination_confidence=0.50),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )
    oscillating = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=2),
        _invariant_receipt(invariant_pressure=0.10, oscillation_score=0.60),
        _convergence_receipt(0.20, 0.30, 0.20, early_termination_advised=False, termination_confidence=0.50),
        version=DETERMINISTIC_EXECUTION_WRAPPER_VERSION,
    )

    assert oscillating.decision.execution_label == "oscillation_hold"
    assert oscillating.decision.wrapper_confidence < base.decision.wrapper_confidence


def test_validation_invalid_input_types_raise_value_error() -> None:
    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_deterministic_execution_wrapper(object(), object(), object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("execution_version", "invariant_version", "convergence_version", "error_match"),
    [
        ("v0.0", "v142.1", "v142.2", "unsupported execution_receipt version"),
        ("v142.0", "v0.0", "v142.2", "unsupported invariant_receipt version"),
        ("v142.0", "v142.1", "v0.0", "unsupported convergence_receipt version"),
    ],
)
def test_validation_rejects_unsupported_upstream_versions(
    execution_version: str,
    invariant_version: str,
    convergence_version: str,
    error_match: str,
) -> None:
    execution = _execution(total_steps=2)
    invariant = _invariant_receipt(invariant_pressure=0.20, oscillation_score=0.0)
    convergence = _convergence_receipt(0.60, 0.30, 0.30, early_termination_advised=False, termination_confidence=0.20)
    execution = replace(execution, version=execution_version)
    invariant = replace(invariant, version=invariant_version)
    convergence = replace(convergence, version=convergence_version)

    with pytest.raises(ValueError, match=error_match):
        evaluate_deterministic_execution_wrapper(execution, invariant, convergence)


def test_hash_stability_repeated_runs_identical_stable_hash() -> None:
    execution = _execution(total_steps=2)
    invariant = _invariant_receipt(invariant_pressure=0.20, oscillation_score=0.0)
    convergence = _convergence_receipt(0.60, 0.30, 0.30, early_termination_advised=False, termination_confidence=0.20)

    first = evaluate_deterministic_execution_wrapper(execution, invariant, convergence)
    second = evaluate_deterministic_execution_wrapper(execution, invariant, convergence)
    assert first.stable_hash == second.stable_hash


def test_canonical_serialization_replay_safe_json_output() -> None:
    out = evaluate_deterministic_execution_wrapper(
        _execution(total_steps=1),
        _invariant_receipt(invariant_pressure=0.20, oscillation_score=0.0),
        _convergence_receipt(0.60, 0.30, 0.30, early_termination_advised=False, termination_confidence=0.20),
    )

    assert out.to_canonical_json() == canonical_json(out.to_dict())
