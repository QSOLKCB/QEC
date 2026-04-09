"""Tests for v137.13.1 — Morphology Transition Kernel."""

from __future__ import annotations

import pytest

from qec.analysis.hybrid_signal_interface import build_hybrid_signal_trace
from qec.analysis.neuromorphic_substrate_simulator import compile_substrate_report
from qec.analysis.synthetic_signal_geometry_kernel import build_geometry_trajectory
from qec.analysis.morphology_transition_kernel import (
    SCHEMA_VERSION,
    MorphologyState,
    MorphologyTransitionPath,
    build_transition_path,
    detect_morphology_states,
    run_morphology_transition_kernel,
)


def _base_input() -> dict[str, object]:
    return {
        "simulation_id": "sim-001",
        "node_count": 2,
        "input_signal": [1, 2, 3, 4],
        "threshold": 5,
        "time_steps": 4,
        "decay_factor": 0.5,
        "epoch_id": "epoch-1",
        "schema_version": "v137.12.0",
    }


def _make_trajectory():
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    return build_geometry_trajectory(trace)


def test_same_input_same_bytes() -> None:
    traj = _make_trajectory()
    payloads = tuple(run_morphology_transition_kernel(traj)[0].to_canonical_bytes() for _ in range(4))
    assert len(set(payloads)) == 1


def test_same_input_same_hash() -> None:
    traj = _make_trajectory()
    result_a, receipt_a = run_morphology_transition_kernel(traj)
    result_b, receipt_b = run_morphology_transition_kernel(traj)
    assert result_a.stable_hash == result_b.stable_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_repeated_run_byte_identity() -> None:
    traj = _make_trajectory()
    artifacts = tuple(run_morphology_transition_kernel(traj) for _ in range(3))
    result_bytes = tuple(a[0].to_canonical_bytes() for a in artifacts)
    receipt_bytes = tuple(a[1].to_canonical_bytes() for a in artifacts)
    assert len(set(result_bytes)) == 1
    assert len(set(receipt_bytes)) == 1


def test_broken_trajectory_rejection() -> None:
    traj = _make_trajectory()
    broken = type(traj)(
        config=traj.config,
        input_trace_hash=traj.input_trace_hash,
        node_count=0,
        nodes=(),
        edges=(),
        shape_label=traj.shape_label,
        shape_scores=traj.shape_scores,
        stable_hash=traj.stable_hash,
        schema_version=traj.schema_version,
    )
    with pytest.raises(ValueError, match="empty paths"):
        run_morphology_transition_kernel(broken)


def test_duplicate_state_rejection() -> None:
    traj = _make_trajectory()
    path = build_transition_path(traj)
    duplicate_id = path.states[0].state_id
    dup_state = MorphologyState(
        state_id=duplicate_id,
        source_index=path.states[1].source_index,
        state_label=path.states[1].state_label,
        activity_centroid=path.states[1].activity_centroid,
        spike_density_coordinate=path.states[1].spike_density_coordinate,
        continuity_coordinate=path.states[1].continuity_coordinate,
        state_score=path.states[1].state_score,
    )
    bad_path = MorphologyTransitionPath(
        config=path.config,
        input_trajectory_hash=path.input_trajectory_hash,
        states=(path.states[0], dup_state),
        edges=(path.edges[0],),
        stable_hash=path.stable_hash,
        schema_version=path.schema_version,
    )
    from qec.analysis.morphology_transition_kernel import _validate_path

    with pytest.raises(ValueError, match="duplicate state ids"):
        _validate_path(bad_path)


def test_canonical_export_stability() -> None:
    traj = _make_trajectory()
    result, receipt = run_morphology_transition_kernel(traj)
    states = detect_morphology_states(traj)
    path = build_transition_path(traj)
    assert states[0].to_canonical_json() == states[0].to_canonical_json()
    assert path.to_canonical_bytes() == path.to_canonical_bytes()
    assert result.to_canonical_json() == result.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt.to_canonical_bytes()


def test_bounded_metric_validation() -> None:
    traj = _make_trajectory()
    result, receipt = run_morphology_transition_kernel(traj)
    assert 0.0 <= result.transition_integrity_score <= 1.0
    assert 0.0 <= result.phase_stability_score <= 1.0
    assert 0.0 <= result.morphology_consistency_score <= 1.0
    assert 0.0 <= result.transition_continuity_score <= 1.0
    assert 0.0 <= receipt.transition_integrity_score <= 1.0
    assert 0.0 <= receipt.phase_stability_score <= 1.0
    assert 0.0 <= receipt.morphology_consistency_score <= 1.0
    assert 0.0 <= receipt.transition_continuity_score <= 1.0


def test_transition_ordering_integrity() -> None:
    traj = _make_trajectory()
    path = build_transition_path(traj)
    indices = tuple(state.source_index for state in path.states)
    assert indices == tuple(sorted(indices))
    for i, edge in enumerate(path.edges):
        assert edge.source_index == path.states[i].source_index
        assert edge.target_index == path.states[i + 1].source_index


def test_wrapper_manual_equivalence() -> None:
    traj = _make_trajectory()
    manual_path = build_transition_path(traj)
    from qec.analysis.morphology_transition_kernel import _compute_kernel_metrics

    manual_metrics = _compute_kernel_metrics(manual_path)
    result, _ = run_morphology_transition_kernel(traj)

    assert result.path.to_canonical_bytes() == manual_path.to_canonical_bytes()
    assert result.transition_integrity_score == manual_metrics["transition_integrity_score"]
    assert result.phase_stability_score == manual_metrics["phase_stability_score"]
    assert result.morphology_consistency_score == manual_metrics["morphology_consistency_score"]
    assert result.transition_continuity_score == manual_metrics["transition_continuity_score"]


def test_receipt_integrity() -> None:
    traj = _make_trajectory()
    result, receipt = run_morphology_transition_kernel(traj)
    assert receipt.output_stable_hash == result.stable_hash
    assert receipt.input_trajectory_hash == traj.stable_hash
    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.kernel_version == SCHEMA_VERSION
    assert receipt.validation_passed is True
    assert receipt.receipt_chain[1] == result.path.stable_hash
    assert receipt.receipt_chain[2] == result.stable_hash
