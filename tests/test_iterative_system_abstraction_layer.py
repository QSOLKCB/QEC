from __future__ import annotations

import pytest

from qec.analysis.iterative_system_abstraction_layer import (
    ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION,
    IterativeStateSnapshot,
    evaluate_iterative_system_abstraction,
)


def _snapshot(step_index: int, state_id: str, convergence_metric: float, *, active: bool = True) -> IterativeStateSnapshot:
    return IterativeStateSnapshot(
        step_index=step_index,
        state_id=state_id,
        state_payload={"id": state_id, "step": step_index},
        convergence_metric=convergence_metric,
        active=active,
    )


def _snapshot_with_nested_payload() -> IterativeStateSnapshot:
    return IterativeStateSnapshot(
        step_index=0,
        state_id="nested",
        state_payload={"nested": {"x": 0}, "series": [1, {"k": "v"}]},
        convergence_metric=0.4,
        active=True,
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    snapshots = (
        _snapshot(0, "s0", 0.10),
        _snapshot(1, "s1", 0.35),
        _snapshot(2, "s2", 0.60),
    )
    first = evaluate_iterative_system_abstraction(snapshots)
    second = evaluate_iterative_system_abstraction(snapshots)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_empty_trace_receipt_defaults() -> None:
    receipt = evaluate_iterative_system_abstraction(())

    assert receipt.version == ITERATIVE_SYSTEM_ABSTRACTION_LAYER_VERSION
    assert receipt.trace.total_steps == 0
    assert receipt.trace.final_state_id == ""
    assert receipt.trace.mean_convergence == 0.0
    assert receipt.trace.converged is False
    assert receipt.trace.transitions == ()


def test_transition_generation_count_matches_adjacent_pairs() -> None:
    snapshots = (
        _snapshot(0, "a", 0.20),
        _snapshot(1, "b", 0.30),
        _snapshot(2, "c", 0.50),
        _snapshot(3, "d", 0.55),
    )
    receipt = evaluate_iterative_system_abstraction(snapshots)

    assert len(receipt.trace.transitions) == len(snapshots) - 1


def test_transition_labels_cover_all_rules() -> None:
    advance_receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.10), _snapshot(1, "b", 0.25)))
    stabilize_receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.50), _snapshot(1, "b", 0.505)))
    converge_receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.80), _snapshot(1, "b", 0.999)))
    stall_receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.80), _snapshot(1, "b", 0.70)))

    assert advance_receipt.trace.transitions[0].transition_label == "advance"
    assert stabilize_receipt.trace.transitions[0].transition_label == "stabilize"
    assert converge_receipt.trace.transitions[0].transition_label == "converge"
    assert stall_receipt.trace.transitions[0].transition_label == "stall"


def test_step_ordering_validation_raises_for_nonzero_start_and_skipped_indices() -> None:
    with pytest.raises(ValueError, match="missing step 0"):
        evaluate_iterative_system_abstraction((_snapshot(1, "x", 0.1),))

    with pytest.raises(ValueError, match="invalid step ordering"):
        evaluate_iterative_system_abstraction((_snapshot(0, "x", 0.1), _snapshot(2, "y", 0.2)))


def test_mean_convergence_computed_correctly_and_bounded() -> None:
    receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.25), _snapshot(1, "b", 0.75), _snapshot(2, "c", 1.0)))
    assert receipt.trace.mean_convergence == pytest.approx((0.25 + 0.75 + 1.0) / 3.0)
    assert 0.0 <= receipt.trace.mean_convergence <= 1.0


def test_final_state_matches_final_snapshot() -> None:
    snapshots = (_snapshot(0, "s0", 0.0), _snapshot(1, "s1", 0.3), _snapshot(2, "s2", 0.4))
    receipt = evaluate_iterative_system_abstraction(snapshots)
    assert receipt.trace.final_state_id == snapshots[-1].state_id


def test_canonical_serialization_replay_safe_for_public_dataclasses() -> None:
    snapshot = _snapshot(0, "a", 0.4)
    transition_receipt = evaluate_iterative_system_abstraction((_snapshot(0, "a", 0.1), _snapshot(1, "b", 0.2)))
    transition = transition_receipt.trace.transitions[0]
    trace = transition_receipt.trace

    assert snapshot.to_canonical_json() == snapshot.to_canonical_json()
    assert transition.to_canonical_json() == transition.to_canonical_json()
    assert trace.to_canonical_json() == trace.to_canonical_json()
    assert transition_receipt.to_canonical_json() == transition_receipt.to_canonical_json()


def test_hash_stability_repeated_runs_identical() -> None:
    snapshots = (_snapshot(0, "a", 0.15), _snapshot(1, "b", 0.45), _snapshot(2, "a", 0.80))
    hashes = [evaluate_iterative_system_abstraction(snapshots).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1


def test_validation_rejects_malformed_payload_and_invalid_snapshot_types() -> None:
    with pytest.raises(ValueError, match="malformed payload"):
        IterativeStateSnapshot(
            step_index=0,
            state_id="bad",
            state_payload={"bad": {"not-json-set"}},
            convergence_metric=0.5,
            active=True,
        )

    with pytest.raises(ValueError, match="invalid snapshot types"):
        evaluate_iterative_system_abstraction((object(),))  # type: ignore[arg-type]


def test_payload_is_deeply_immutable() -> None:
    snapshot = _snapshot_with_nested_payload()
    with pytest.raises(TypeError):
        snapshot.state_payload["nested"]["x"] = 1

    with pytest.raises(TypeError):
        snapshot.state_payload["series"][1]["k"] = "changed"


def test_payload_union_mutation_blocked() -> None:
    snapshot = _snapshot(0, "s0", 0.25)
    with pytest.raises(TypeError):
        snapshot.state_payload |= {"k": 1}
