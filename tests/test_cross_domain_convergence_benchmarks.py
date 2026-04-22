from __future__ import annotations

import pytest

from qec.analysis.canonical_hashing import canonical_json
from qec.analysis.convergence_engine import ConvergenceDecision, ConvergenceReceipt, ConvergenceSignal
from qec.analysis.cross_domain_convergence_benchmarks import (
    CROSS_DOMAIN_BENCHMARK_VERSION,
    evaluate_cross_domain_benchmark,
)
from qec.analysis.deterministic_execution_wrapper import (
    ExecutionPlan,
    ExecutionWrapperDecision,
    ExecutionWrapperReceipt,
    ExecutionWrapperSignal,
)
from qec.analysis.generalized_invariant_detector import (
    InvariantDecision,
    InvariantDetectionReceipt,
    InvariantSignal,
)
from qec.analysis.iterative_system_abstraction_layer import IterativeStateSnapshot, evaluate_iterative_system_abstraction


def _snapshot(step_index: int, state_id: str, convergence_metric: float) -> IterativeStateSnapshot:
    return IterativeStateSnapshot(
        step_index=step_index,
        state_id=state_id,
        state_payload={"id": state_id, "step": step_index},
        convergence_metric=convergence_metric,
        active=True,
    )


def _execution(total_steps: int) -> object:
    snapshots = tuple(_snapshot(i, f"S{i}", min(0.99, 0.1 + 0.1 * i)) for i in range(total_steps))
    return evaluate_iterative_system_abstraction(snapshots)


def _execution_from_sequence(convergence_values: tuple[float, ...], state_ids: tuple[str, ...] | None = None) -> object:
    if state_ids is None:
        state_ids = tuple(f"S{i}" for i in range(len(convergence_values)))
    elif len(state_ids) != len(convergence_values):
        raise ValueError(
            "state_ids must have the same length as convergence_values: "
            f"got {len(state_ids)} and {len(convergence_values)}"
        )
    snapshots = tuple(_snapshot(i, state_ids[i], convergence_values[i]) for i in range(len(convergence_values)))
    return evaluate_iterative_system_abstraction(snapshots)


def _invariant_receipt(invariant_pressure: float) -> InvariantDetectionReceipt:
    return InvariantDetectionReceipt(
        version="v142.1",
        signal=InvariantSignal(
            repeated_state_score=0.0,
            fixed_point_score=0.0,
            plateau_score=0.0,
            oscillation_score=0.0,
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
    *,
    early_termination_advised: bool,
    convergence_label: str = "unconverged",
) -> ConvergenceReceipt:
    valid_labels = {
        "unconverged",
        "progressing",
        "near_converged",
        "converged",
        "oscillating",
    }
    if convergence_label not in valid_labels:
        raise ValueError(f"invalid convergence_label: {convergence_label}")

    return ConvergenceReceipt(
        version="v142.2",
        signal=ConvergenceSignal(
            mean_convergence=0.0,
            invariant_pressure=0.0,
            terminal_convergence=0.0,
            plateau_component=0.0,
            oscillation_component=0.0,
            convergence_pressure=convergence_pressure,
            efficiency_score=0.0,
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
            termination_confidence=0.0,
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


def _wrapper_receipt(allowed_next_steps: int) -> ExecutionWrapperReceipt:
    return ExecutionWrapperReceipt(
        version="v142.3",
        signal=ExecutionWrapperSignal(
            invariant_pressure=0.0,
            convergence_pressure=0.0,
            terminal_convergence=0.0,
            oscillation_component=0.0,
            efficiency_score=0.0,
            gating_pressure=0.0,
            pruning_pressure=0.0,
            standardization_score=0.0,
        ),
        decision=ExecutionWrapperDecision(
            execution_label="continue",
            execution_rank=0,
            early_termination_advised=False,
            pruning_enabled=False,
            output_standardized=True,
            wrapper_confidence=0.0,
            rationale="continue_execution",
        ),
        plan=ExecutionPlan(
            allowed_next_steps=allowed_next_steps,
            pruning_budget=0.0,
            state_retention_budget=1.0,
            canonical_output_mode="full",
            plan_signature=f"continue::full::{allowed_next_steps}",
        ),
        control_mode="execution_wrapper_advisory",
        observatory_only=True,
    )


def test_deterministic_replay_json_and_hash_stable() -> None:
    execution = _execution(total_steps=4)
    invariant = _invariant_receipt(0.4)
    convergence = _convergence_receipt(0.5, early_termination_advised=False)
    wrapper = _wrapper_receipt(allowed_next_steps=2)

    first = evaluate_cross_domain_benchmark("routing", execution, invariant, convergence, wrapper, version=CROSS_DOMAIN_BENCHMARK_VERSION)
    second = evaluate_cross_domain_benchmark("routing", execution, invariant, convergence, wrapper, version=CROSS_DOMAIN_BENCHMARK_VERSION)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_zero_iteration_case() -> None:
    out = evaluate_cross_domain_benchmark(
        "empty",
        _execution(total_steps=0),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=5),
    )

    assert out.signal.iterations_total == 0
    assert out.signal.iterations_effective == 0
    assert out.signal.cutoff_step == -1
    assert out.signal.redundancy_ratio == 0.0
    assert out.signal.structural_redundancy_ratio == 0.0


def test_convergence_cutoff_uses_threshold_crossing() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.2, 0.5, 0.999, 0.999, 0.999)),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=2),
    )

    assert out.signal.cutoff_step == 2
    assert out.signal.iterations_effective == 3
    assert out.signal.redundancy_ratio > 0.0


def test_plateau_cutoff_uses_first_three_snapshot_plateau_index() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.20, 0.30, 0.305, 0.308, 0.309, 0.60)),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=6),
    )

    assert out.signal.cutoff_step == 2




def test_plateau_cutoff_detects_exact_three_snapshot_plateau() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.20, 0.204, 0.207)),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=3),
    )

    assert out.signal.cutoff_step == 0
    assert out.signal.iterations_effective == 1

def test_fixed_point_cutoff_uses_repeat_state() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.10, 0.30, 0.305, 0.36), state_ids=("A", "B", "B", "C")),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=4),
    )

    assert out.signal.cutoff_step == 2


def test_oscillation_guard_disables_redundancy() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.2, 0.4, 0.5, 0.6, 0.7)),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False, convergence_label="oscillating"),
        _wrapper_receipt(allowed_next_steps=0),
    )

    assert out.signal.cutoff_step == 4
    assert out.signal.redundancy_ratio == 0.0
    assert out.signal.structural_redundancy_ratio == 0.0


def test_full_length_fallback_has_zero_redundancy() -> None:
    out = evaluate_cross_domain_benchmark(
        "compute",
        _execution_from_sequence((0.10, 0.20, 0.30)),
        _invariant_receipt(0.0),
        _convergence_receipt(0.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=1),
    )

    assert out.signal.cutoff_step == 2
    assert out.signal.iterations_effective == 3
    assert out.signal.redundancy_ratio == 0.0


def test_early_termination_propagation() -> None:
    early = evaluate_cross_domain_benchmark(
        "vision",
        _execution(total_steps=3),
        _invariant_receipt(0.1),
        _convergence_receipt(0.1, early_termination_advised=True),
        _wrapper_receipt(allowed_next_steps=1),
    )
    late = evaluate_cross_domain_benchmark(
        "vision",
        _execution(total_steps=3),
        _invariant_receipt(0.1),
        _convergence_receipt(0.1, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=1),
    )

    assert early.signal.early_termination_rate == 1.0
    assert late.signal.early_termination_rate == 0.0


def test_classification_boundary_correctness() -> None:
    execution = _execution(total_steps=10)
    wrapper_full = _wrapper_receipt(allowed_next_steps=10)

    low = evaluate_cross_domain_benchmark(
        "boundary",
        execution,
        _invariant_receipt(0.4),
        _convergence_receipt(0.4, early_termination_advised=False),
        wrapper_full,
    )
    moderate = evaluate_cross_domain_benchmark(
        "boundary",
        execution,
        _invariant_receipt(0.5),
        _convergence_receipt(1.0 / 3.0, early_termination_advised=False),
        wrapper_full,
    )
    high = evaluate_cross_domain_benchmark(
        "boundary",
        execution,
        _invariant_receipt(1.0),
        _convergence_receipt(2.0 / 3.0, early_termination_advised=False),
        wrapper_full,
    )
    extreme = evaluate_cross_domain_benchmark(
        "boundary",
        execution,
        _invariant_receipt(1.0),
        _convergence_receipt(1.0, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=0),
    )

    assert low.decision.benchmark_label == "low"
    assert moderate.signal.efficiency_gain == pytest.approx(0.25)
    assert moderate.decision.benchmark_label == "moderate"
    assert high.signal.efficiency_gain == pytest.approx(0.5)
    assert high.decision.benchmark_label == "high"
    assert extreme.signal.efficiency_gain == pytest.approx(0.6)
    assert extreme.decision.benchmark_label == "high"


def test_domain_validation_empty_string_error() -> None:
    with pytest.raises(ValueError, match="domain must be a non-empty str"):
        evaluate_cross_domain_benchmark(
            "",
            _execution(total_steps=1),
            _invariant_receipt(0.0),
            _convergence_receipt(0.0, early_termination_advised=False),
            _wrapper_receipt(allowed_next_steps=1),
        )


def test_hash_stability() -> None:
    execution = _execution(total_steps=6)
    invariant = _invariant_receipt(0.2)
    convergence = _convergence_receipt(0.3, early_termination_advised=False)
    wrapper = _wrapper_receipt(allowed_next_steps=4)

    first = evaluate_cross_domain_benchmark("audio", execution, invariant, convergence, wrapper)
    second = evaluate_cross_domain_benchmark("audio", execution, invariant, convergence, wrapper)

    assert first.stable_hash == second.stable_hash


def test_canonical_json_consistency() -> None:
    out = evaluate_cross_domain_benchmark(
        "nlp",
        _execution(total_steps=4),
        _invariant_receipt(0.2),
        _convergence_receipt(0.1, early_termination_advised=False),
        _wrapper_receipt(allowed_next_steps=3),
    )

    assert out.to_canonical_json() == canonical_json(out.to_dict())
