"""Tests for v137.13.2 — Phase Boundary Topology Kernel."""

from __future__ import annotations

import pytest

from qec.analysis.hybrid_signal_interface import build_hybrid_signal_trace
from qec.analysis.morphology_transition_kernel import (
    MorphologyState,
    MorphologyTransitionEdge,
    MorphologyTransitionPath,
    build_transition_path,
)
from qec.analysis.neuromorphic_substrate_simulator import compile_substrate_report
from qec.analysis.phase_boundary_topology_kernel import (
    SCHEMA_VERSION,
    PhaseRegion,
    PhaseTopologyPath,
    _sha256_hex,
    _validate_transition_path,
    build_phase_topology_path,
    detect_phase_regions,
    run_phase_boundary_topology_kernel,
)
from qec.analysis.synthetic_signal_geometry_kernel import build_geometry_trajectory


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


def _make_transition_path():
    report = compile_substrate_report(_base_input())
    trace = build_hybrid_signal_trace(report)
    trajectory = build_geometry_trajectory(trace)
    return build_transition_path(trajectory)


def test_same_input_same_bytes() -> None:
    path = _make_transition_path()
    payloads = tuple(run_phase_boundary_topology_kernel(path)[0].to_canonical_bytes() for _ in range(4))
    assert len(set(payloads)) == 1


def test_same_input_same_hash() -> None:
    path = _make_transition_path()
    result_a, receipt_a = run_phase_boundary_topology_kernel(path)
    result_b, receipt_b = run_phase_boundary_topology_kernel(path)
    assert result_a.stable_hash == result_b.stable_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_repeated_run_byte_identity() -> None:
    path = _make_transition_path()
    artifacts = tuple(run_phase_boundary_topology_kernel(path) for _ in range(3))
    result_bytes = tuple(a[0].to_canonical_bytes() for a in artifacts)
    receipt_bytes = tuple(a[1].to_canonical_bytes() for a in artifacts)
    assert len(set(result_bytes)) == 1
    assert len(set(receipt_bytes)) == 1


def test_broken_path_rejection() -> None:
    path = _make_transition_path()
    broken = type(path)(
        config=path.config,
        input_trajectory_hash=path.input_trajectory_hash,
        states=(),
        edges=(),
        stable_hash=path.stable_hash,
        schema_version=path.schema_version,
    )
    with pytest.raises(ValueError, match="broken lineage|empty path"):
        run_phase_boundary_topology_kernel(broken)


def test_duplicate_region_rejection() -> None:
    path = _make_transition_path()
    topology_path = build_phase_topology_path(path)
    duplicate_id = topology_path.regions[0].region_id
    duplicated_region = PhaseRegion(
        region_id=duplicate_id,
        source_start_index=topology_path.regions[0].source_start_index + 1,
        source_end_index=topology_path.regions[0].source_end_index + 1,
        region_label=topology_path.regions[0].region_label,
        region_score=topology_path.regions[0].region_score,
        continuity_mean=topology_path.regions[0].continuity_mean,
        morphology_mean=topology_path.regions[0].morphology_mean,
    )
    bad_topology = PhaseTopologyPath(
        config=topology_path.config,
        input_transition_hash=topology_path.input_transition_hash,
        regions=(topology_path.regions[0], duplicated_region),
        boundaries=(topology_path.boundaries[0],),
        stable_hash=topology_path.stable_hash,
        schema_version=topology_path.schema_version,
    )
    from qec.analysis.phase_boundary_topology_kernel import _validate_topology_path

    with pytest.raises(ValueError, match="duplicate region ids"):
        _validate_topology_path(bad_topology)


def test_canonical_export_stability() -> None:
    path = _make_transition_path()
    result, receipt = run_phase_boundary_topology_kernel(path)
    regions = detect_phase_regions(path)
    topology_path = build_phase_topology_path(path)
    assert regions[0].to_canonical_json() == regions[0].to_canonical_json()
    assert topology_path.to_canonical_bytes() == topology_path.to_canonical_bytes()
    assert result.to_canonical_json() == result.to_canonical_json()
    assert receipt.to_canonical_bytes() == receipt.to_canonical_bytes()


def test_bounded_metric_validation() -> None:
    path = _make_transition_path()
    result, receipt = run_phase_boundary_topology_kernel(path)
    assert 0.0 <= result.boundary_integrity_score <= 1.0
    assert 0.0 <= result.topology_stability_score <= 1.0
    assert 0.0 <= result.region_consistency_score <= 1.0
    assert 0.0 <= result.boundary_continuity_score <= 1.0
    assert 0.0 <= receipt.boundary_integrity_score <= 1.0
    assert 0.0 <= receipt.topology_stability_score <= 1.0
    assert 0.0 <= receipt.region_consistency_score <= 1.0
    assert 0.0 <= receipt.boundary_continuity_score <= 1.0


def test_boundary_ordering_integrity() -> None:
    path = _make_transition_path()
    topology_path = build_phase_topology_path(path)
    starts = tuple(region.source_start_index for region in topology_path.regions)
    assert starts == tuple(sorted(starts))
    for i, boundary in enumerate(topology_path.boundaries):
        assert boundary.source_region_index == i
        assert boundary.target_region_index == i + 1


def test_wrapper_manual_equivalence() -> None:
    path = _make_transition_path()
    manual_path = build_phase_topology_path(path)
    from qec.analysis.phase_boundary_topology_kernel import _compute_kernel_metrics

    manual_metrics = _compute_kernel_metrics(manual_path)
    result, _ = run_phase_boundary_topology_kernel(path)

    assert result.path.to_canonical_bytes() == manual_path.to_canonical_bytes()
    assert result.boundary_integrity_score == manual_metrics["boundary_integrity_score"]
    assert result.topology_stability_score == manual_metrics["topology_stability_score"]
    assert result.region_consistency_score == manual_metrics["region_consistency_score"]
    assert result.boundary_continuity_score == manual_metrics["boundary_continuity_score"]


def test_receipt_integrity() -> None:
    path = _make_transition_path()
    result, receipt = run_phase_boundary_topology_kernel(path)
    assert receipt.output_stable_hash == result.stable_hash
    assert receipt.input_transition_hash == path.stable_hash
    assert receipt.schema_version == SCHEMA_VERSION
    assert receipt.kernel_version == SCHEMA_VERSION
    assert receipt.validation_passed is True
    assert receipt.receipt_chain[1] == result.path.stable_hash
    assert receipt.receipt_chain[2] == result.stable_hash


def _make_path_with_stable_hash(
    path: MorphologyTransitionPath,
    states: tuple,
    edges: tuple,
) -> MorphologyTransitionPath:
    """Build a MorphologyTransitionPath with a correctly-computed stable_hash."""
    proto = MorphologyTransitionPath(
        config=path.config,
        input_trajectory_hash=path.input_trajectory_hash,
        states=states,
        edges=edges,
        stable_hash="",
        schema_version=path.schema_version,
    )
    return MorphologyTransitionPath(
        config=path.config,
        input_trajectory_hash=path.input_trajectory_hash,
        states=states,
        edges=edges,
        stable_hash=_sha256_hex(proto.to_hash_payload_dict()),
        schema_version=path.schema_version,
    )


def test_duplicate_state_id_rejection() -> None:
    """A path whose states contain duplicate state_id values is rejected."""
    path = _make_transition_path()
    if len(path.states) < 2:
        pytest.skip("path has fewer than 2 states")
    duplicate_state = MorphologyState(
        state_id=path.states[0].state_id,
        source_index=path.states[1].source_index,
        state_label=path.states[1].state_label,
        activity_centroid=path.states[1].activity_centroid,
        spike_density_coordinate=path.states[1].spike_density_coordinate,
        continuity_coordinate=path.states[1].continuity_coordinate,
        state_score=path.states[1].state_score,
    )
    new_states = (path.states[0], duplicate_state) + path.states[2:]
    malformed = _make_path_with_stable_hash(path, new_states, path.edges)
    with pytest.raises(ValueError, match="duplicate state ids"):
        _validate_transition_path(malformed)


def test_misaligned_edge_index_rejection() -> None:
    """A path with an edge whose indices don't match consecutive states is rejected."""
    path = _make_transition_path()
    if len(path.edges) == 0:
        pytest.skip("path has no edges")
    bad_edge = MorphologyTransitionEdge(
        source_index=9999,
        target_index=9998,
        source_state=path.edges[0].source_state,
        target_state=path.edges[0].target_state,
        transition_magnitude=path.edges[0].transition_magnitude,
        stability_delta=path.edges[0].stability_delta,
        continuity_delta=path.edges[0].continuity_delta,
    )
    new_edges = (bad_edge,) + path.edges[1:]
    malformed = _make_path_with_stable_hash(path, path.states, new_edges)
    with pytest.raises(ValueError, match="edge indices do not align"):
        _validate_transition_path(malformed)


def test_misaligned_edge_label_rejection() -> None:
    """A path with an edge whose state labels don't match consecutive states is rejected."""
    path = _make_transition_path()
    if len(path.edges) == 0:
        pytest.skip("path has no edges")
    valid_labels = ("stable_oscillation", "transient_burst", "refractory_suppression")
    mismatched_label = next(
        lbl for lbl in valid_labels if lbl != path.edges[0].source_state
    )
    bad_edge = MorphologyTransitionEdge(
        source_index=path.edges[0].source_index,
        target_index=path.edges[0].target_index,
        source_state=mismatched_label,
        target_state=path.edges[0].target_state,
        transition_magnitude=path.edges[0].transition_magnitude,
        stability_delta=path.edges[0].stability_delta,
        continuity_delta=path.edges[0].continuity_delta,
    )
    new_edges = (bad_edge,) + path.edges[1:]
    malformed = _make_path_with_stable_hash(path, path.states, new_edges)
    with pytest.raises(ValueError, match="edge/state label mismatch"):
        _validate_transition_path(malformed)
