"""Tests for rotation-aware trajectory geometry diagnostics (v104.1.0).

Covers:
- state vector construction
- trajectory building
- 3D projection
- turning angles
- angular velocity
- spiral score
- axis lock
- curvature
- coupling metrics
- event prediction
- full pipeline
- determinism guarantees
- no-mutation guarantees
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

from qec.analysis.trajectory_geometry import (
    build_state_vector,
    build_trajectory,
    compute_angular_velocity,
    compute_axis_lock,
    compute_coupling_metrics,
    compute_curvature,
    compute_spiral_score,
    compute_turning_angles,
    format_trajectory_geometry_summary,
    predict_events,
    project_to_3d,
    run_trajectory_geometry_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_step(**kwargs: float) -> Dict[str, Any]:
    """Build a run step dict with given metric overrides."""
    return {"metrics": dict(kwargs)}


def _make_runs(n: int = 5) -> List[Dict[str, Any]]:
    """Build a simple increasing trajectory of *n* steps."""
    runs: List[Dict[str, Any]] = []
    for i in range(n):
        runs.append(
            _make_step(
                residual_norm=float(i) * 0.1,
                instability_score=float(i) * 0.05,
                barrier_estimate=float(i) * 0.02,
                boundary_distance=0.5,
                spectral_radius_proxy=0.9,
                convergence_signal=0.1 * i,
                control_signal=0.0,
                basin_switch_score=0.0,
            )
        )
    return runs


def _make_spiral_runs(n: int = 10) -> List[Dict[str, Any]]:
    """Build a spiraling inward trajectory."""
    runs: List[Dict[str, Any]] = []
    for i in range(n):
        r = 1.0 - i * 0.08
        angle = i * 0.8
        runs.append(
            _make_step(
                residual_norm=r * math.cos(angle),
                instability_score=r * math.sin(angle),
                barrier_estimate=0.5 - i * 0.03,
            )
        )
    return runs


# ---------------------------------------------------------------------------
# State Vector
# ---------------------------------------------------------------------------


class TestBuildStateVector:
    def test_empty_step(self) -> None:
        result = build_state_vector({})
        assert len(result) == 8
        assert all(v == 0.0 for v in result)

    def test_with_metrics(self) -> None:
        step = _make_step(residual_norm=0.5, instability_score=0.3)
        result = build_state_vector(step)
        assert result[0] == 0.5
        assert result[1] == 0.3
        assert result[2] == 0.0  # barrier_estimate missing

    def test_fixed_ordering(self) -> None:
        step = _make_step(
            basin_switch_score=0.9,
            residual_norm=0.1,
        )
        result = build_state_vector(step)
        assert result[0] == 0.1  # residual_norm first
        assert result[7] == 0.9  # basin_switch_score last

    def test_rounding(self) -> None:
        step = _make_step(residual_norm=1.0 / 3.0)
        result = build_state_vector(step)
        assert result[0] == round(1.0 / 3.0, 12)

    def test_direct_keys(self) -> None:
        """Keys read directly from step when no 'metrics' key."""
        step = {"residual_norm": 0.42}
        result = build_state_vector(step)
        assert result[0] == 0.42


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestBuildTrajectory:
    def test_empty(self) -> None:
        assert build_trajectory([]) == []

    def test_single(self) -> None:
        runs = [_make_step(residual_norm=0.5)]
        traj = build_trajectory(runs)
        assert len(traj) == 1
        assert traj[0][0] == 0.5

    def test_ordering(self) -> None:
        runs = _make_runs(3)
        traj = build_trajectory(runs)
        assert len(traj) == 3
        assert traj[0][0] < traj[1][0] < traj[2][0]


# ---------------------------------------------------------------------------
# 3D Projection
# ---------------------------------------------------------------------------


class TestProjectTo3d:
    def test_basic(self) -> None:
        state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        x, y, z = project_to_3d(state)
        assert x == 0.1
        assert y == 0.2
        assert z == 0.3

    def test_short_vector(self) -> None:
        x, y, z = project_to_3d([0.5])
        assert x == 0.5
        assert y == 0.0
        assert z == 0.0

    def test_empty_vector(self) -> None:
        x, y, z = project_to_3d([])
        assert x == 0.0
        assert y == 0.0
        assert z == 0.0


# ---------------------------------------------------------------------------
# Turning Angles
# ---------------------------------------------------------------------------


class TestTurningAngles:
    def test_fewer_than_3(self) -> None:
        assert compute_turning_angles([]) == []
        assert compute_turning_angles([(0, 0, 0)]) == []
        assert compute_turning_angles([(0, 0, 0), (1, 0, 0)]) == []

    def test_straight_line(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        angles = compute_turning_angles(traj)
        assert len(angles) == 1
        assert abs(angles[0]) < 1e-10

    def test_right_angle(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        angles = compute_turning_angles(traj)
        assert len(angles) == 1
        assert abs(angles[0] - math.pi / 2) < 1e-10

    def test_reversal(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
        angles = compute_turning_angles(traj)
        assert abs(angles[0] - math.pi) < 1e-10

    def test_stationary_point(self) -> None:
        traj = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        angles = compute_turning_angles(traj)
        assert angles[0] == 0.0


# ---------------------------------------------------------------------------
# Angular Velocity
# ---------------------------------------------------------------------------


class TestAngularVelocity:
    def test_empty(self) -> None:
        assert compute_angular_velocity([]) == 0.0

    def test_single(self) -> None:
        assert compute_angular_velocity([0.5]) == 0.5

    def test_constant_angles(self) -> None:
        assert compute_angular_velocity([0.5, 0.5, 0.5]) == 0.0

    def test_varying_angles(self) -> None:
        result = compute_angular_velocity([0.0, 0.5, 0.0])
        assert result > 0.0


# ---------------------------------------------------------------------------
# Spiral Score
# ---------------------------------------------------------------------------


class TestSpiralScore:
    def test_fewer_than_3(self) -> None:
        assert compute_spiral_score([]) == 0.0
        assert compute_spiral_score([(0, 0, 0), (1, 0, 0)]) == 0.0

    def test_straight_outward(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        score = compute_spiral_score(traj)
        # Straight line, angle=0, rotation_fraction=0 -> score=0
        assert score == 0.0

    def test_bounded(self) -> None:
        runs = _make_spiral_runs(10)
        traj = build_trajectory(runs)
        traj3d = [project_to_3d(s) for s in traj]
        score = compute_spiral_score(traj3d)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Axis Lock
# ---------------------------------------------------------------------------


class TestAxisLock:
    def test_single_point(self) -> None:
        assert compute_axis_lock([(0, 0, 0)]) == 0.0

    def test_single_axis(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        score = compute_axis_lock(traj)
        # All motion along x, but variance of dx is 0 when steps are equal.
        # dx=[1,1], dy=[0,0], dz=[0,0] => var_x=0, so total=0 => 0.0
        assert score == 0.0

    def test_mixed_axes(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0),
                (2.0, 1.0, 0.0), (2.0, 2.0, 0.0)]
        score = compute_axis_lock(traj)
        assert 0.0 <= score <= 1.0

    def test_bounded(self) -> None:
        runs = _make_runs(5)
        traj = build_trajectory(runs)
        traj3d = [project_to_3d(s) for s in traj]
        score = compute_axis_lock(traj3d)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    def test_fewer_than_3(self) -> None:
        assert compute_curvature([]) == 0.0
        assert compute_curvature([(0, 0, 0), (1, 0, 0)]) == 0.0

    def test_straight_line(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
        assert compute_curvature(traj) == 0.0

    def test_curved(self) -> None:
        traj = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
        curv = compute_curvature(traj)
        assert curv > 0.0

    def test_non_negative(self) -> None:
        runs = _make_spiral_runs(10)
        traj = build_trajectory(runs)
        traj3d = [project_to_3d(s) for s in traj]
        assert compute_curvature(traj3d) >= 0.0


# ---------------------------------------------------------------------------
# Coupling Metrics
# ---------------------------------------------------------------------------


class TestCouplingMetrics:
    def test_empty(self) -> None:
        result = compute_coupling_metrics([])
        assert result["plane_coupling_score"] == 0.0
        assert result["multi_axis_variation"] == 0
        assert result["dimensional_activity"] == 0

    def test_single_step(self) -> None:
        result = compute_coupling_metrics([[0.1, 0.2, 0.3]])
        assert result["dimensional_activity"] == 0

    def test_multi_step(self) -> None:
        traj = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        result = compute_coupling_metrics(traj)
        assert isinstance(result["plane_coupling_score"], float)
        assert isinstance(result["dimensional_activity"], int)

    def test_bounded_coupling(self) -> None:
        runs = _make_runs(5)
        traj = build_trajectory(runs)
        result = compute_coupling_metrics(traj)
        assert 0.0 <= result["plane_coupling_score"] <= 1.0


# ---------------------------------------------------------------------------
# Event Prediction
# ---------------------------------------------------------------------------


class TestPredictEvents:
    def test_all_low(self) -> None:
        metrics = {
            "angular_velocity": 0.0,
            "spiral_score": 0.0,
            "curvature": 0.0,
            "total_displacement": 1.0,
            "displacement_variance": 1.0,
            "mean_instability": 0.0,
        }
        pred = predict_events(metrics)
        assert pred["convergence"] == "low"
        assert pred["oscillation"] == "low"
        assert pred["basin_switch_risk"] == "low"
        assert pred["metastable"] == "low"

    def test_high_oscillation(self) -> None:
        metrics = {
            "angular_velocity": 0.8,
            "spiral_score": 0.0,
            "curvature": 0.0,
            "total_displacement": 1.0,
            "displacement_variance": 1.0,
            "mean_instability": 0.0,
        }
        pred = predict_events(metrics)
        assert pred["oscillation"] == "high"

    def test_convergence(self) -> None:
        metrics = {
            "angular_velocity": 0.0,
            "spiral_score": 0.8,
            "curvature": 0.0,
            "total_displacement": 1.0,
            "displacement_variance": 1.0,
            "mean_instability": 0.0,
        }
        pred = predict_events(metrics)
        assert pred["convergence"] == "likely"

    def test_basin_switch(self) -> None:
        metrics = {
            "angular_velocity": 0.0,
            "spiral_score": 0.0,
            "curvature": 0.7,
            "total_displacement": 1.0,
            "displacement_variance": 1.0,
            "mean_instability": 0.8,
        }
        pred = predict_events(metrics)
        assert pred["basin_switch_risk"] == "high"

    def test_metastable(self) -> None:
        metrics = {
            "angular_velocity": 0.0,
            "spiral_score": 0.0,
            "curvature": 0.0,
            "total_displacement": 0.01,
            "displacement_variance": 0.01,
            "mean_instability": 0.0,
        }
        pred = predict_events(metrics)
        assert pred["metastable"] == "likely"

    def test_returns_valid_levels(self) -> None:
        metrics = {
            "angular_velocity": 0.3,
            "spiral_score": 0.4,
            "curvature": 0.3,
            "total_displacement": 0.08,
            "displacement_variance": 0.08,
            "mean_instability": 0.4,
        }
        pred = predict_events(metrics)
        valid = {"low", "moderate", "high", "likely", "medium"}
        for key, val in pred.items():
            assert val in valid, f"{key}={val} not in {valid}"


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------


class TestRunTrajectoryGeometryAnalysis:
    def test_empty_runs(self) -> None:
        result = run_trajectory_geometry_analysis([])
        assert result["trajectory_length"] == 0
        assert result["rotation_metrics"]["angular_velocity"] == 0.0

    def test_single_run(self) -> None:
        runs = [_make_step(residual_norm=0.5)]
        result = run_trajectory_geometry_analysis(runs)
        assert result["trajectory_length"] == 1

    def test_basic_pipeline(self) -> None:
        runs = _make_runs(5)
        result = run_trajectory_geometry_analysis(runs)
        assert result["trajectory_length"] == 5
        assert "rotation_metrics" in result
        assert "coupling_metrics" in result
        assert "predictions" in result

    def test_predictions_present(self) -> None:
        runs = _make_runs(5)
        result = run_trajectory_geometry_analysis(runs)
        pred = result["predictions"]
        assert "convergence" in pred
        assert "oscillation" in pred
        assert "basin_switch_risk" in pred
        assert "metastable" in pred


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatSummary:
    def test_basic(self) -> None:
        runs = _make_runs(5)
        result = run_trajectory_geometry_analysis(runs)
        summary = format_trajectory_geometry_summary(result)
        assert "=== Trajectory Geometry ===" in summary
        assert "Angular Velocity:" in summary
        assert "Spiral Score:" in summary
        assert "Axis Lock:" in summary
        assert "Curvature:" in summary
        assert "Plane Coupling:" in summary
        assert "Convergence:" in summary

    def test_empty_result(self) -> None:
        result = run_trajectory_geometry_analysis([])
        summary = format_trajectory_geometry_summary(result)
        assert "=== Trajectory Geometry ===" in summary


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_pipeline_deterministic(self) -> None:
        runs = _make_spiral_runs(10)
        results = [run_trajectory_geometry_analysis(runs) for _ in range(20)]
        for r in results[1:]:
            assert r == results[0]

    def test_state_vector_deterministic(self) -> None:
        step = _make_step(residual_norm=0.123456789012345)
        results = [build_state_vector(step) for _ in range(20)]
        assert all(r == results[0] for r in results)

    def test_coupling_deterministic(self) -> None:
        traj = build_trajectory(_make_runs(5))
        results = [compute_coupling_metrics(traj) for _ in range(20)]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# No-Mutation
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_pipeline_no_mutation(self) -> None:
        runs = _make_runs(5)
        runs_copy = copy.deepcopy(runs)
        run_trajectory_geometry_analysis(runs)
        assert runs == runs_copy

    def test_state_vector_no_mutation(self) -> None:
        step = _make_step(residual_norm=0.5, instability_score=0.3)
        step_copy = copy.deepcopy(step)
        build_state_vector(step)
        assert step == step_copy

    def test_coupling_no_mutation(self) -> None:
        traj = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        traj_copy = copy.deepcopy(traj)
        compute_coupling_metrics(traj)
        assert traj == traj_copy
