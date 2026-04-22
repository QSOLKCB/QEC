from __future__ import annotations

import pytest

from qec.analysis.generalized_invariant_detector import (
    GENERALIZED_INVARIANT_DETECTOR_VERSION,
    evaluate_generalized_invariant_detector,
)
from qec.analysis.iterative_system_abstraction_layer import (
    IterativeExecutionReceipt,
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


def _receipt_from_snapshots(*snapshots: IterativeStateSnapshot) -> IterativeExecutionReceipt:
    return evaluate_iterative_system_abstraction(tuple(snapshots))


def test_deterministic_replay_hash_stable() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.20),
        _snapshot(2, "A", 0.30),
        _snapshot(3, "B", 0.40),
    )
    first = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)
    second = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert first.stable_hash == second.stable_hash
    assert first.to_canonical_json() == second.to_canonical_json()


def test_empty_trace_no_invariant() -> None:
    receipt = _receipt_from_snapshots()
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.decision.dominant_invariant == "none"
    assert out.decision.invariant_detected is False
    assert out.decision.invariant_confidence == 0.0
    assert out.patterns == ()


def test_repeated_state_detection() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.20),
        _snapshot(2, "A", 0.30),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.signal.repeated_state_score == pytest.approx(1.0 - (2.0 / 3.0))
    repeated = [p for p in out.patterns if p.pattern_type == "repeated_state"]
    assert len(repeated) == 1
    assert repeated[0].key == "A"
    assert repeated[0].support == 2


def test_fixed_point_detection() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.50),
        _snapshot(1, "A", 0.505),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.signal.fixed_point_score == 1.0
    assert out.decision.dominant_invariant == "fixed_point"
    assert out.patterns[0].pattern_type == "fixed_point"


def test_plateau_detection() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.105),
        _snapshot(2, "C", 0.109),
        _snapshot(3, "D", 0.115),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.signal.plateau_score > 0.0
    plateau = [p for p in out.patterns if p.pattern_type == "plateau"]
    assert len(plateau) == 1
    assert plateau[0].support >= 3


def test_oscillation_detection() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.20),
        _snapshot(2, "A", 0.30),
        _snapshot(3, "B", 0.40),
        _snapshot(4, "A", 0.50),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.signal.oscillation_score > 0.0
    oscillation = [p for p in out.patterns if p.pattern_type == "oscillation"]
    assert len(oscillation) == 1
    assert oscillation[0].key == "A<->B"


def test_oscillation_key_reflects_states() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "X", 0.10),
        _snapshot(1, "Y", 0.20),
        _snapshot(2, "X", 0.30),
        _snapshot(3, "Y", 0.40),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    oscillation = [p for p in out.patterns if p.pattern_type == "oscillation"]
    assert len(oscillation) == 1
    assert oscillation[0].key == "X<->Y"


def test_tie_break_prefers_fixed_point_then_repeated_then_plateau_then_oscillation() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "A", 0.105),
        _snapshot(2, "A", 0.109),
        _snapshot(3, "A", 0.115),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.signal.fixed_point_score == 1.0
    assert out.signal.plateau_score == 1.0
    assert out.decision.dominant_invariant == "fixed_point"
    assert out.decision.invariant_rank == 1


def test_pattern_ordering_deterministic() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.10),
        _snapshot(1, "B", 0.105),
        _snapshot(2, "A", 0.109),
        _snapshot(3, "B", 0.115),
        _snapshot(4, "A", 0.119),
        _snapshot(5, "A", 0.120),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert tuple(p.pattern_type for p in out.patterns) == (
        "fixed_point",
        "repeated_state",
        "repeated_state",
        "plateau",
        "oscillation",
    )
    repeated_keys = [p.key for p in out.patterns if p.pattern_type == "repeated_state"]
    assert repeated_keys == sorted(repeated_keys)


def test_validation_failure_on_bad_input() -> None:
    with pytest.raises(ValueError, match="invalid input type"):
        evaluate_generalized_invariant_detector(object(), version=GENERALIZED_INVARIANT_DETECTOR_VERSION)  # type: ignore[arg-type]


def test_nan_rejected() -> None:
    from qec.analysis.generalized_invariant_detector import InvariantSignal

    with pytest.raises(ValueError, match="repeated_state_score must be finite"):
        InvariantSignal(
            repeated_state_score=float("nan"),
            fixed_point_score=0.0,
            plateau_score=0.0,
            oscillation_score=0.0,
            invariant_pressure=0.0,
        )


def test_canonical_json_stable() -> None:
    receipt = _receipt_from_snapshots(
        _snapshot(0, "A", 0.1),
        _snapshot(1, "A", 0.105),
    )
    out = evaluate_generalized_invariant_detector(receipt, version=GENERALIZED_INVARIANT_DETECTOR_VERSION)

    assert out.to_canonical_json() == out.to_canonical_json()
