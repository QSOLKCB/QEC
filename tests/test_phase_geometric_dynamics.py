"""Tests for v87.0.0 — Geometric Trajectory Dynamics."""

from __future__ import annotations

import copy
import math

import pytest

from qec.experiments.phase_geometric_dynamics import (
    compute_curvature,
    compute_spread,
    compute_step_vectors,
    compute_trajectory_length,
    detect_attractors,
    run_geometric_dynamics,
)


# ---------------------------------------------------------------------------
# Test 1 — Trajectory length correctness
# ---------------------------------------------------------------------------


class TestTrajectoryLength:
    def test_two_points(self):
        series = [(0, 0, 0, 0), (1, 0, 0, 0)]
        assert compute_trajectory_length(series) == 1.0

    def test_three_collinear(self):
        series = [(0, 0, 0, 0), (1, 0, 0, 0), (2, 0, 0, 0)]
        assert compute_trajectory_length(series) == 2.0

    def test_diagonal_step(self):
        series = [(0, 0, 0, 0), (1, 1, 0, 0)]
        assert math.isclose(compute_trajectory_length(series), math.sqrt(2))

    def test_round_trip(self):
        series = [(1, 1, 1, 1), (-1, -1, -1, -1), (1, 1, 1, 1)]
        # Each leg = sqrt(16) = 4.0
        assert math.isclose(compute_trajectory_length(series), 8.0)

    def test_empty(self):
        assert compute_trajectory_length([]) == 0.0

    def test_single_point(self):
        assert compute_trajectory_length([(1, 0, -1, 1)]) == 0.0


# ---------------------------------------------------------------------------
# Test 2 — Step vectors
# ---------------------------------------------------------------------------


class TestStepVectors:
    def test_basic(self):
        series = [(0, 0, 0, 0), (1, -1, 0, 1)]
        vectors = compute_step_vectors(series)
        assert vectors == [(1, -1, 0, 1)]

    def test_multiple_steps(self):
        series = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)]
        vectors = compute_step_vectors(series)
        assert vectors == [(1, 0, 0, 0), (0, 1, 0, 0)]

    def test_empty(self):
        assert compute_step_vectors([]) == []

    def test_single_point(self):
        assert compute_step_vectors([(1, 0, 0, 0)]) == []


# ---------------------------------------------------------------------------
# Test 3 — Curvature (straight vs turning)
# ---------------------------------------------------------------------------


class TestCurvature:
    def test_straight_line_zero_curvature(self):
        # Same direction steps → cos_theta = 1 → curvature = 0
        vectors = [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]
        c = compute_curvature(vectors)
        assert c["mean_curvature"] == 0.0
        assert c["max_curvature"] == 0.0

    def test_right_angle_turn(self):
        # Perpendicular vectors → cos_theta = 0 → curvature = 1
        vectors = [(1, 0, 0, 0), (0, 1, 0, 0)]
        c = compute_curvature(vectors)
        assert math.isclose(c["mean_curvature"], 1.0)
        assert math.isclose(c["max_curvature"], 1.0)

    def test_reversal(self):
        # Opposite direction → cos_theta = -1 → curvature = 2
        vectors = [(1, 0, 0, 0), (-1, 0, 0, 0)]
        c = compute_curvature(vectors)
        assert math.isclose(c["mean_curvature"], 2.0)
        assert math.isclose(c["max_curvature"], 2.0)

    def test_zero_vector_skipped(self):
        # Zero vector should be skipped
        vectors = [(1, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 0)]
        c = compute_curvature(vectors)
        # Both pairs include zero vector → no valid curvature
        assert c["mean_curvature"] is None
        assert c["max_curvature"] is None

    def test_empty(self):
        c = compute_curvature([])
        assert c["mean_curvature"] is None
        assert c["max_curvature"] is None

    def test_single_vector(self):
        c = compute_curvature([(1, 0, 0, 0)])
        assert c["mean_curvature"] is None
        assert c["max_curvature"] is None

    def test_mixed_curvature(self):
        # straight then turn
        vectors = [(1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)]
        c = compute_curvature(vectors)
        # First pair: curvature = 0, second pair: curvature = 1
        assert math.isclose(c["mean_curvature"], 0.5)
        assert math.isclose(c["max_curvature"], 1.0)


# ---------------------------------------------------------------------------
# Test 4 — Attractor detection
# ---------------------------------------------------------------------------


class TestAttractorDetection:
    def test_strong_attractor(self):
        # All same state
        series = [(1, 1, 1, 1)] * 5
        a = detect_attractors(series)
        assert a["has_attractor"] is True
        assert a["attractor_state"] == (1, 1, 1, 1)
        assert a["strength"] == 1.0

    def test_no_repeat(self):
        # All different states, each appears once, run length = 1
        series = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)]
        a = detect_attractors(series)
        assert a["has_attractor"] is False
        assert a["strength"] == pytest.approx(1.0 / 3.0)

    def test_partial_attractor(self):
        series = [(0, 0, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)]
        a = detect_attractors(series)
        assert a["has_attractor"] is True
        assert a["attractor_state"] == (1, 1, 1, 1)
        assert a["strength"] == 0.75

    def test_empty(self):
        a = detect_attractors([])
        assert a["has_attractor"] is False
        assert a["attractor_state"] is None
        assert a["strength"] == 0.0

    def test_single_point_is_attractor(self):
        # Single state → only one state exists → has_attractor
        a = detect_attractors([(1, 0, -1, 1)])
        assert a["has_attractor"] is True
        assert a["attractor_state"] == (1, 0, -1, 1)
        assert a["strength"] == 1.0


# ---------------------------------------------------------------------------
# Test 5 — Spread computation
# ---------------------------------------------------------------------------


class TestSpread:
    def test_single_point(self):
        assert compute_spread([(1, 0, -1, 1)]) == 0.0

    def test_two_opposite_corners(self):
        series = [(1, 1, 1, 1), (-1, -1, -1, -1)]
        # Per dimension: max-min = 2, sum = 8
        assert compute_spread(series) == 8.0

    def test_collinear(self):
        series = [(0, 0, 0, 0), (1, 0, 0, 0)]
        # Only first dim varies: 1-0 = 1
        assert compute_spread(series) == 1.0

    def test_empty(self):
        assert compute_spread([]) == 0.0


# ---------------------------------------------------------------------------
# Test 6 — Full analysis integration
# ---------------------------------------------------------------------------


class TestRunGeometricDynamics:
    def test_basic(self):
        series = [
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            (-1, -1, -1, -1),
            (-1, -1, -1, -1),
        ]
        out = run_geometric_dynamics(series)
        assert "trajectory_length" in out
        assert "mean_curvature" in out
        assert "max_curvature" in out
        assert "spread" in out
        assert "attractor" in out
        assert out["trajectory_length"] > 0.0
        assert out["spread"] == 8.0
        assert out["attractor"]["has_attractor"] is True

    def test_empty(self):
        out = run_geometric_dynamics([])
        assert out["trajectory_length"] == 0.0
        assert out["spread"] == 0.0
        assert out["mean_curvature"] is None
        assert out["attractor"]["has_attractor"] is False


# ---------------------------------------------------------------------------
# Test 7 — Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_full_analysis_determinism(self):
        series = [
            (1, 0, -1, 1),
            (0, 1, 0, -1),
            (-1, -1, 1, 0),
            (1, 0, -1, 1),
        ]
        r1 = run_geometric_dynamics(series)
        r2 = run_geometric_dynamics(series)
        assert r1 == r2

    def test_length_determinism(self):
        series = [(0, 0, 0, 0), (1, 1, 1, 1), (-1, -1, -1, -1)]
        l1 = compute_trajectory_length(series)
        l2 = compute_trajectory_length(series)
        assert l1 == l2


# ---------------------------------------------------------------------------
# Test 8 — No mutation
# ---------------------------------------------------------------------------


class TestNoMutation:
    def test_length_no_mutation(self):
        series = [(0, 0, 0, 0), (1, 1, 1, 1)]
        original = copy.deepcopy(series)
        compute_trajectory_length(series)
        assert series == original

    def test_step_vectors_no_mutation(self):
        series = [(0, 0, 0, 0), (1, 1, 1, 1)]
        original = copy.deepcopy(series)
        compute_step_vectors(series)
        assert series == original

    def test_curvature_no_mutation(self):
        vectors = [(1, 0, 0, 0), (0, 1, 0, 0)]
        original = copy.deepcopy(vectors)
        compute_curvature(vectors)
        assert vectors == original

    def test_attractors_no_mutation(self):
        series = [(1, 1, 1, 1), (1, 1, 1, 1), (0, 0, 0, 0)]
        original = copy.deepcopy(series)
        detect_attractors(series)
        assert series == original

    def test_spread_no_mutation(self):
        series = [(0, 0, 0, 0), (1, 1, 1, 1)]
        original = copy.deepcopy(series)
        compute_spread(series)
        assert series == original

    def test_full_analysis_no_mutation(self):
        series = [(0, 0, 0, 0), (1, 0, -1, 1), (-1, 1, 0, 0)]
        original = copy.deepcopy(series)
        run_geometric_dynamics(series)
        assert series == original
