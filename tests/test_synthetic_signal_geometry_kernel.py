"""Tests for v137.13.0 — Synthetic Signal Geometry Kernel.

Covers determinism, byte identity, receipt integrity, bounded metrics,
trajectory ordering, validation rejection, and wrapper/manual equivalence.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.hybrid_signal_interface import (
    build_hybrid_signal_trace,
)
from qec.analysis.neuromorphic_substrate_simulator import (
    compile_substrate_report,
)
from qec.analysis.synthetic_signal_geometry_kernel import (
    SCHEMA_VERSION,
    SignalGeometryEdge,
    SignalGeometryKernelResult,
    SignalGeometryNode,
    SignalGeometryReceipt,
    SignalGeometryTrajectory,
    SyntheticSignalGeometryConfig,
    build_ascii_geometry_summary,
    build_geometry_trajectory,
    compute_geometry_similarity,
    project_trace_to_geometry,
    run_signal_geometry_kernel,
)


# ---------------------------------------------------------------------------
# Deterministic test input
# ---------------------------------------------------------------------------


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


def _make_trace():
    report = compile_substrate_report(_base_input())
    return build_hybrid_signal_trace(report)


# ---------------------------------------------------------------------------
# Frozen dataclass behaviour
# ---------------------------------------------------------------------------


class TestFrozenDataclasses:
    def test_config_frozen(self) -> None:
        config = SyntheticSignalGeometryConfig()
        with pytest.raises(FrozenInstanceError):
            config.schema_version = "v0"

    def test_node_frozen(self) -> None:
        trace = _make_trace()
        nodes = project_trace_to_geometry(trace)
        with pytest.raises(FrozenInstanceError):
            nodes[0].frame_index = 99

    def test_trajectory_frozen(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        with pytest.raises(FrozenInstanceError):
            traj.shape_label = "invalid"

    def test_result_frozen(self) -> None:
        trace = _make_trace()
        result, _ = run_signal_geometry_kernel(trace)
        with pytest.raises(FrozenInstanceError):
            result.stable_hash = "bad"

    def test_edge_frozen(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        assert len(traj.edges) > 0
        with pytest.raises(FrozenInstanceError):
            traj.edges[0].source_index = 99

    def test_receipt_frozen(self) -> None:
        trace = _make_trace()
        _, receipt = run_signal_geometry_kernel(trace)
        with pytest.raises(FrozenInstanceError):
            receipt.receipt_hash = "bad"


# ---------------------------------------------------------------------------
# Determinism: same input → same bytes
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_bytes(self) -> None:
        trace = _make_trace()
        artifacts = tuple(
            run_signal_geometry_kernel(trace)[0].to_canonical_bytes()
            for _ in range(4)
        )
        assert len(set(artifacts)) == 1

    def test_same_input_same_hash(self) -> None:
        trace = _make_trace()
        result_a, receipt_a = run_signal_geometry_kernel(trace)
        result_b, receipt_b = run_signal_geometry_kernel(trace)
        assert result_a.stable_hash == result_b.stable_hash
        assert receipt_a.receipt_hash == receipt_b.receipt_hash

    def test_repeated_run_byte_identity(self) -> None:
        trace = _make_trace()
        results = tuple(run_signal_geometry_kernel(trace) for _ in range(3))
        bytes_results = tuple(r[0].to_canonical_bytes() for r in results)
        bytes_receipts = tuple(r[1].to_canonical_bytes() for r in results)
        assert len(set(bytes_results)) == 1
        assert len(set(bytes_receipts)) == 1

    def test_trajectory_bytes_stable(self) -> None:
        trace = _make_trace()
        traj_a = build_geometry_trajectory(trace)
        traj_b = build_geometry_trajectory(trace)
        assert traj_a.to_canonical_bytes() == traj_b.to_canonical_bytes()
        assert traj_a.stable_hash == traj_b.stable_hash

    def test_node_projection_deterministic(self) -> None:
        trace = _make_trace()
        nodes_a = project_trace_to_geometry(trace)
        nodes_b = project_trace_to_geometry(trace)
        assert tuple(n.stable_hash for n in nodes_a) == tuple(
            n.stable_hash for n in nodes_b
        )


# ---------------------------------------------------------------------------
# Canonical export stability
# ---------------------------------------------------------------------------


class TestCanonicalExport:
    def test_config_canonical_export(self) -> None:
        config = SyntheticSignalGeometryConfig()
        assert config.to_canonical_json() == config.to_canonical_json()
        assert config.to_canonical_bytes() == config.to_canonical_bytes()

    def test_trajectory_canonical_export(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        assert traj.to_canonical_json() == traj.to_canonical_json()
        assert traj.to_canonical_bytes() == traj.to_canonical_bytes()

    def test_result_canonical_export(self) -> None:
        trace = _make_trace()
        result, _ = run_signal_geometry_kernel(trace)
        assert result.to_canonical_json() == result.to_canonical_json()
        assert result.to_canonical_bytes() == result.to_canonical_bytes()

    def test_receipt_canonical_export(self) -> None:
        trace = _make_trace()
        _, receipt = run_signal_geometry_kernel(trace)
        assert receipt.to_canonical_json() == receipt.to_canonical_json()
        assert receipt.to_canonical_bytes() == receipt.to_canonical_bytes()


# ---------------------------------------------------------------------------
# Bounded metric validation
# ---------------------------------------------------------------------------


class TestBoundedMetrics:
    def test_kernel_metrics_bounded(self) -> None:
        trace = _make_trace()
        result, _ = run_signal_geometry_kernel(trace)
        assert 0.0 <= result.geometry_integrity_score <= 1.0
        assert 0.0 <= result.continuity_score <= 1.0
        assert 0.0 <= result.similarity_score <= 1.0
        assert 0.0 <= result.path_stability_score <= 1.0

    def test_receipt_metrics_bounded(self) -> None:
        trace = _make_trace()
        _, receipt = run_signal_geometry_kernel(trace)
        assert 0.0 <= receipt.geometry_integrity_score <= 1.0
        assert 0.0 <= receipt.continuity_score <= 1.0
        assert 0.0 <= receipt.similarity_score <= 1.0
        assert 0.0 <= receipt.path_stability_score <= 1.0

    def test_shape_scores_bounded(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        for key, value in traj.shape_scores.items():
            assert 0.0 <= value <= 1.0, f"shape_scores[{key}] out of bounds: {value}"

    def test_node_coordinates_bounded(self) -> None:
        trace = _make_trace()
        nodes = project_trace_to_geometry(trace)
        for node in nodes:
            assert 0.0 <= node.activity_centroid <= 1.0
            assert 0.0 <= node.spike_density_coordinate <= 1.0
            assert 0.0 <= node.continuity_coordinate <= 1.0
            for v in node.trajectory_vector:
                assert 0.0 <= v <= 1.0

    def test_similarity_metrics_bounded(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        sim = compute_geometry_similarity(traj, traj)
        for key, value in sim.items():
            assert 0.0 <= value <= 1.0, f"similarity[{key}] out of bounds: {value}"


# ---------------------------------------------------------------------------
# Trajectory ordering integrity
# ---------------------------------------------------------------------------


class TestTrajectoryOrdering:
    def test_node_ordering_monotonic(self) -> None:
        trace = _make_trace()
        nodes = project_trace_to_geometry(trace)
        indices = tuple(n.frame_index for n in nodes)
        assert indices == tuple(sorted(indices))
        assert len(set(indices)) == len(indices)

    def test_edge_ordering_matches_nodes(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        for i, edge in enumerate(traj.edges):
            assert edge.source_index == traj.nodes[i].frame_index
            assert edge.target_index == traj.nodes[i + 1].frame_index

    def test_edge_count_matches_expected(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        assert len(traj.edges) == len(traj.nodes) - 1

    def test_node_count_matches_frames(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        assert traj.node_count == trace.frame_count
        assert len(traj.nodes) == traj.node_count


# ---------------------------------------------------------------------------
# Validation rejection
# ---------------------------------------------------------------------------


class TestValidation:
    def test_broken_trace_rejected(self) -> None:
        trace = _make_trace()
        broken = type(trace)(
            config=trace.config,
            input_stable_hash=trace.input_stable_hash,
            node_ids=trace.node_ids,
            frame_count=0,
            frames=(),
            stable_hash=trace.stable_hash,
            schema_version=trace.schema_version,
        )
        with pytest.raises(ValueError, match="frame_count must be > 0"):
            run_signal_geometry_kernel(broken)

    def test_frame_count_mismatch_rejected(self) -> None:
        trace = _make_trace()
        broken = type(trace)(
            config=trace.config,
            input_stable_hash=trace.input_stable_hash,
            node_ids=trace.node_ids,
            frame_count=999,
            frames=trace.frames,
            stable_hash=trace.stable_hash,
            schema_version=trace.schema_version,
        )
        with pytest.raises(ValueError, match="frames length must equal frame_count"):
            run_signal_geometry_kernel(broken)

    def test_duplicate_frame_rejection(self) -> None:
        trace = _make_trace()
        # Create nodes with duplicate frame_index
        config = SyntheticSignalGeometryConfig()
        nodes = project_trace_to_geometry(trace, config)
        dup_node = SignalGeometryNode(
            frame_index=nodes[0].frame_index,
            activity_centroid=nodes[0].activity_centroid,
            spike_density_coordinate=nodes[0].spike_density_coordinate,
            continuity_coordinate=nodes[0].continuity_coordinate,
            trajectory_vector=nodes[0].trajectory_vector,
            stable_hash=nodes[0].stable_hash,
        )
        # Manually create a bad node tuple with duplicates
        from qec.analysis.synthetic_signal_geometry_kernel import _validate_nodes

        bad_nodes = (dup_node, dup_node)
        with pytest.raises(ValueError, match="strictly ordered"):
            _validate_nodes(bad_nodes, config)

    def test_invalid_schema_version_rejected(self) -> None:
        config = SyntheticSignalGeometryConfig(schema_version="v0.0.0")
        trace = _make_trace()
        with pytest.raises(ValueError, match="unsupported schema version"):
            run_signal_geometry_kernel(trace, config)

    def test_invalid_coordinate_dimensions_rejected(self) -> None:
        with pytest.raises(ValueError, match="coordinate_dimensions must be 3"):
            SyntheticSignalGeometryConfig(coordinate_dimensions=5)

    def test_trajectory_vector_length_validated(self) -> None:
        from qec.analysis.synthetic_signal_geometry_kernel import _validate_nodes

        bad_node = SignalGeometryNode(
            frame_index=0,
            activity_centroid=0.5,
            spike_density_coordinate=0.5,
            continuity_coordinate=0.0,
            trajectory_vector=(0.5, 0.5),  # length 2, should be 3
            stable_hash="a" * 64,
        )
        config = SyntheticSignalGeometryConfig()
        with pytest.raises(ValueError, match="trajectory_vector length"):
            _validate_nodes((bad_node,), config)

    def test_tampered_trace_hash_rejected(self) -> None:
        trace = _make_trace()
        tampered = type(trace)(
            config=trace.config,
            input_stable_hash=trace.input_stable_hash,
            node_ids=trace.node_ids,
            frame_count=trace.frame_count,
            frames=trace.frames,
            stable_hash="0" * 64,
            schema_version=trace.schema_version,
        )
        with pytest.raises(ValueError, match="stable_hash does not match"):
            run_signal_geometry_kernel(tampered)

    def test_first_frame_continuity_is_zero(self) -> None:
        trace = _make_trace()
        nodes = project_trace_to_geometry(trace)
        assert nodes[0].continuity_coordinate == 0.0


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


class TestReceiptIntegrity:
    def test_receipt_links_to_result(self) -> None:
        trace = _make_trace()
        result, receipt = run_signal_geometry_kernel(trace)
        assert receipt.output_stable_hash == result.stable_hash
        assert receipt.input_trace_hash == trace.stable_hash
        assert receipt.schema_version == SCHEMA_VERSION
        assert receipt.kernel_version == SCHEMA_VERSION
        assert receipt.validation_passed is True

    def test_receipt_metrics_match_result(self) -> None:
        trace = _make_trace()
        result, receipt = run_signal_geometry_kernel(trace)
        assert receipt.geometry_integrity_score == result.geometry_integrity_score
        assert receipt.continuity_score == result.continuity_score
        assert receipt.similarity_score == result.similarity_score
        assert receipt.path_stability_score == result.path_stability_score

    def test_receipt_node_count_matches(self) -> None:
        trace = _make_trace()
        result, receipt = run_signal_geometry_kernel(trace)
        assert receipt.node_count == result.trajectory.node_count

    def test_receipt_shape_label_matches(self) -> None:
        trace = _make_trace()
        result, receipt = run_signal_geometry_kernel(trace)
        assert receipt.shape_label == result.trajectory.shape_label


# ---------------------------------------------------------------------------
# Wrapper / manual equivalence
# ---------------------------------------------------------------------------


class TestWrapperEquivalence:
    def test_wrapper_matches_manual_pipeline(self) -> None:
        trace = _make_trace()

        # Manual pipeline
        manual_traj = build_geometry_trajectory(trace)
        from qec.analysis.synthetic_signal_geometry_kernel import _compute_kernel_metrics

        manual_metrics = _compute_kernel_metrics(manual_traj)

        # Wrapper pipeline
        result, receipt = run_signal_geometry_kernel(trace)

        assert result.trajectory.to_canonical_bytes() == manual_traj.to_canonical_bytes()
        assert result.geometry_integrity_score == manual_metrics["geometry_integrity_score"]
        assert result.continuity_score == manual_metrics["continuity_score"]
        assert result.similarity_score == manual_metrics["similarity_score"]
        assert result.path_stability_score == manual_metrics["path_stability_score"]


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_self_similarity_is_perfect(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        sim = compute_geometry_similarity(traj, traj)
        assert sim["geometric_continuity"] == 1.0
        assert sim["structural_similarity"] == 1.0
        assert sim["trajectory_overlap"] == 1.0
        assert sim["centroid_drift_score"] == 1.0

    def test_similarity_deterministic(self) -> None:
        trace = _make_trace()
        traj = build_geometry_trajectory(trace)
        sim_a = compute_geometry_similarity(traj, traj)
        sim_b = compute_geometry_similarity(traj, traj)
        assert sim_a == sim_b


# ---------------------------------------------------------------------------
# ASCII summary
# ---------------------------------------------------------------------------


class TestAsciiSummary:
    def test_ascii_summary_deterministic(self) -> None:
        trace = _make_trace()
        result, _ = run_signal_geometry_kernel(trace)
        summary_a = build_ascii_geometry_summary(result)
        summary_b = build_ascii_geometry_summary(result)
        assert summary_a == summary_b
        assert SCHEMA_VERSION in summary_a

    def test_ascii_summary_contains_shape(self) -> None:
        trace = _make_trace()
        result, _ = run_signal_geometry_kernel(trace)
        summary = build_ascii_geometry_summary(result)
        assert result.trajectory.shape_label in summary
