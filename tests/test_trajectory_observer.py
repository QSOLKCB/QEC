"""
Tests for v80.5.0 — Neural ODE-Style Trajectory Observer.

Validates determinism, correctness, classification, and edge cases.
"""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import numpy as np
import pytest

from qec.controller.trajectory_observer import (
    analyze_trajectory,
    classify_trajectory,
    compute_flow_metrics,
    estimate_derivative,
    extract_time_series,
    interpolate_trajectory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(scores, epsilons=None, reject_cycles=None):
    """Build a minimal FSM history from a list of stability scores."""
    history = []
    for i, s in enumerate(scores):
        entry = {
            "from_state": "INVARIANT",
            "to_state": "EVALUATE",
            "stability_score": s,
            "phase": "stable_region",
            "epsilon": epsilons[i] if epsilons else 1e-3,
            "reject_cycle": reject_cycles[i] if reject_cycles else 0,
            "decision": "CONTINUE",
        }
        history.append(entry)
    return history


def _make_history_with_nones(scores_with_nones):
    """Build history where some entries have None stability_score."""
    history = []
    for s in scores_with_nones:
        history.append({
            "from_state": "INIT",
            "to_state": "ANALYZE",
            "stability_score": s,
            "phase": None,
            "epsilon": 1e-3,
            "reject_cycle": 0,
            "decision": "CONTINUE",
        })
    return history


# ---------------------------------------------------------------------------
# Test: Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Trajectory observer must produce identical results across runs."""

    def test_determinism_full_pipeline(self):
        history = _make_history([1.0, 0.8, 0.5, 0.3, 0.2, 0.15])
        r1 = analyze_trajectory(history)
        r2 = analyze_trajectory(history)
        assert r1 == r2

    def test_determinism_interpolation(self):
        t = np.arange(5, dtype=np.float64)
        y = np.array([1.0, 0.8, 0.5, 0.3, 0.2])
        t1, y1 = interpolate_trajectory(t, y)
        t2, y2 = interpolate_trajectory(t, y)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(y1, y2)

    def test_determinism_derivative(self):
        t = np.linspace(0, 4, 41)
        y = np.sin(t)
        d1 = estimate_derivative(t, y)
        d2 = estimate_derivative(t, y)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# Test: No Mutation
# ---------------------------------------------------------------------------

class TestNoMutation:
    """Observer must not mutate input history."""

    def test_history_not_mutated(self):
        history = _make_history([1.0, 0.5, 0.3])
        original = copy.deepcopy(history)
        analyze_trajectory(history)
        assert history == original

    def test_arrays_not_mutated(self):
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([1.0, 0.8, 0.5, 0.3])
        t_orig = t.copy()
        y_orig = y.copy()
        interpolate_trajectory(t, y)
        estimate_derivative(t, y)
        np.testing.assert_array_equal(t, t_orig)
        np.testing.assert_array_equal(y, y_orig)


# ---------------------------------------------------------------------------
# Test: Derivative Correctness
# ---------------------------------------------------------------------------

class TestDerivative:
    """Verify derivative estimation on simple known sequences."""

    def test_constant_signal_zero_velocity(self):
        t = np.arange(10, dtype=np.float64)
        y = np.full(10, 5.0)
        dy = estimate_derivative(t, y)
        np.testing.assert_allclose(dy, 0.0, atol=1e-12)

    def test_linear_signal_constant_derivative(self):
        t = np.arange(10, dtype=np.float64)
        y = 2.0 * t + 1.0  # slope = 2
        dy = estimate_derivative(t, y)
        np.testing.assert_allclose(dy, 2.0, atol=1e-12)

    def test_monotonic_increasing_positive_derivative(self):
        t = np.arange(5, dtype=np.float64)
        y = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        dy = estimate_derivative(t, y)
        assert np.all(dy > 0), "Monotonic increasing should have positive derivative"

    def test_monotonic_decreasing_negative_derivative(self):
        t = np.arange(5, dtype=np.float64)
        y = np.array([10.0, 6.0, 3.0, 1.0, 0.0])
        dy = estimate_derivative(t, y)
        assert np.all(dy < 0), "Monotonic decreasing should have negative derivative"

    def test_single_point_zero_derivative(self):
        t = np.array([0.0])
        y = np.array([5.0])
        dy = estimate_derivative(t, y)
        np.testing.assert_array_equal(dy, np.array([0.0]))


# ---------------------------------------------------------------------------
# Test: Oscillation Detection
# ---------------------------------------------------------------------------

class TestOscillation:
    """Verify zero-crossing and oscillation score computation."""

    def test_oscillating_signal_detected(self):
        t = np.arange(20, dtype=np.float64)
        y = np.sin(t)
        dy = estimate_derivative(t, y)
        metrics = compute_flow_metrics(y, dy)
        assert metrics["zero_crossings"] > 0
        assert metrics["oscillation_score"] > 0.0

    def test_monotonic_no_oscillation(self):
        t = np.arange(10, dtype=np.float64)
        y = np.exp(-0.5 * t)  # Decaying exponential
        dy = estimate_derivative(t, y)
        metrics = compute_flow_metrics(y, dy)
        assert metrics["zero_crossings"] == 0
        assert metrics["oscillation_score"] == 0.0


# ---------------------------------------------------------------------------
# Test: Classification
# ---------------------------------------------------------------------------

class TestClassification:
    """Phase classification correctness."""

    def test_stable_plateau(self):
        metrics = {
            "mean_velocity": 0.0,
            "max_velocity": 0.0,
            "oscillation_score": 0.0,
            "convergence_rate": 0.0,
        }
        assert classify_trajectory(metrics) == "stable_plateau"

    def test_oscillatory(self):
        metrics = {
            "mean_velocity": 1.0,
            "max_velocity": 2.0,
            "oscillation_score": 0.5,
            "convergence_rate": 0.1,
        }
        assert classify_trajectory(metrics) == "oscillatory"

    def test_convergent(self):
        metrics = {
            "mean_velocity": 0.5,
            "max_velocity": 1.0,
            "oscillation_score": 0.0,
            "convergence_rate": 0.01,
        }
        assert classify_trajectory(metrics) == "convergent"

    def test_divergent(self):
        metrics = {
            "mean_velocity": 1.0,
            "max_velocity": 2.0,
            "oscillation_score": 0.0,
            "convergence_rate": 0.8,
        }
        assert classify_trajectory(metrics) == "divergent"


# ---------------------------------------------------------------------------
# Test: Finite Outputs
# ---------------------------------------------------------------------------

class TestFiniteOutputs:
    """All numeric outputs must be finite."""

    def test_all_outputs_finite(self):
        history = _make_history([1.0, 0.8, 0.5, 0.3, 0.2, 0.15, 0.1])
        result = analyze_trajectory(history)
        assert math.isfinite(result["mean_velocity"])
        assert math.isfinite(result["max_velocity"])
        assert math.isfinite(result["oscillation_score"])
        assert math.isfinite(result["convergence_rate"])
        assert isinstance(result["n_points"], int)
        assert isinstance(result["classification"], str)

    def test_interpolation_finite(self):
        t = np.arange(5, dtype=np.float64)
        y = np.array([100.0, 0.01, 50.0, 0.001, 99.0])
        t_d, y_d = interpolate_trajectory(t, y)
        assert np.all(np.isfinite(t_d))
        assert np.all(np.isfinite(y_d))


# ---------------------------------------------------------------------------
# Test: Edge Cases (Empty / Short History)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Handle degenerate inputs gracefully."""

    def test_empty_history(self):
        result = analyze_trajectory([])
        assert result["n_points"] == 0
        assert result["classification"] == "stable_plateau"
        assert result["mean_velocity"] == 0.0

    def test_single_point_history(self):
        history = _make_history([0.5])
        result = analyze_trajectory(history)
        assert result["n_points"] == 1
        assert result["classification"] == "stable_plateau"

    def test_two_point_history(self):
        history = _make_history([1.0, 0.5])
        result = analyze_trajectory(history)
        assert result["n_points"] == 2
        assert isinstance(result["classification"], str)

    def test_history_with_none_scores_filtered(self):
        history = _make_history_with_nones([None, 1.0, None, 0.5, 0.3, None])
        series = extract_time_series(history)
        assert len(series["stability_score"]) == 3
        np.testing.assert_array_equal(
            series["stability_score"], [1.0, 0.5, 0.3]
        )


# ---------------------------------------------------------------------------
# Test: JSON Output
# ---------------------------------------------------------------------------

class TestJSONOutput:
    """Verify optional JSON serialization."""

    def test_writes_json_file(self):
        history = _make_history([1.0, 0.8, 0.5, 0.3])
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyze_trajectory(history, output_dir=tmpdir)
            path = os.path.join(tmpdir, "trajectory_analysis.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == result

    def test_no_file_without_output_dir(self):
        history = _make_history([1.0, 0.5])
        result = analyze_trajectory(history)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Test: Extract Time Series
# ---------------------------------------------------------------------------

class TestExtractTimeSeries:
    """Extraction from FSM history."""

    def test_extracts_epsilon_and_reject_cycle(self):
        history = _make_history(
            [0.5, 0.3],
            epsilons=[1e-3, 5e-4],
            reject_cycles=[0, 1],
        )
        series = extract_time_series(history)
        np.testing.assert_array_equal(series["epsilon"], [1e-3, 5e-4])
        np.testing.assert_array_equal(series["reject_cycle"], [0.0, 1.0])

    def test_time_is_arange(self):
        history = _make_history([1.0, 0.8, 0.6, 0.4])
        series = extract_time_series(history)
        np.testing.assert_array_equal(series["time"], [0, 1, 2, 3])
