"""
Tests for BP Jacobian spectral radius estimator (v6.0).

Verifies:
  - Stable values for synthetic converging sequences
  - Stable values for synthetic diverging sequences
  - Edge cases (too few iterations)
  - JSON serialization safety
  - Determinism
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.bp_jacobian_estimator import (
    estimate_bp_jacobian_spectral_radius,
)


class TestBPJacobianEstimator:
    def test_converging_sequence(self):
        """Geometrically converging sequence → spectral radius < 1."""
        # x_t = x_0 * 0.5^t
        T, N = 10, 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        history = np.array([x0 * (0.5 ** t) for t in range(T)])

        result = estimate_bp_jacobian_spectral_radius(history)

        assert result["jacobian_spectral_radius_est"] > 0.0
        assert result["jacobian_spectral_radius_est"] < 1.0 + 1e-6
        assert result["tail_iterations_used"] > 0

    def test_diverging_sequence(self):
        """Geometrically diverging sequence → spectral radius > 1."""
        T, N = 10, 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        history = np.array([x0 * (2.0 ** t) for t in range(T)])

        result = estimate_bp_jacobian_spectral_radius(history)

        assert result["jacobian_spectral_radius_est"] > 1.0 - 1e-6
        assert result["tail_iterations_used"] > 0

    def test_constant_sequence(self):
        """Constant sequence → zero norms → spectral radius 0."""
        T, N = 10, 5
        history = np.ones((T, N))

        result = estimate_bp_jacobian_spectral_radius(history)

        assert result["jacobian_spectral_radius_est"] == 0.0
        assert result["tail_iterations_used"] == 0

    def test_too_few_iterations(self):
        """Fewer than 3 iterations → fallback result."""
        history = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = estimate_bp_jacobian_spectral_radius(history)

        assert result["jacobian_spectral_radius_est"] == 0.0
        assert result["tail_iterations_used"] == 0

    def test_1d_input_fallback(self):
        """1D input → fallback result."""
        history = np.array([1.0, 2.0, 3.0])

        result = estimate_bp_jacobian_spectral_radius(history)

        assert result["jacobian_spectral_radius_est"] == 0.0
        assert result["tail_iterations_used"] == 0

    def test_determinism(self):
        """Repeated calls produce identical results."""
        T, N = 10, 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        history = np.array([x0 * (0.8 ** t) for t in range(T)])

        r1 = estimate_bp_jacobian_spectral_radius(history)
        r2 = estimate_bp_jacobian_spectral_radius(history)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        T, N = 10, 5
        x0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        history = np.array([x0 * (0.5 ** t) for t in range(T)])

        result = estimate_bp_jacobian_spectral_radius(history)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_converging_ratio_near_decay_rate(self):
        """For geometric decay, ratio should approximate the decay rate."""
        T, N = 20, 3
        rate = 0.7
        x0 = np.array([1.0, 1.0, 1.0])
        history = np.array([x0 * (rate ** t) for t in range(T)])

        result = estimate_bp_jacobian_spectral_radius(history)

        # Should be close to 0.7 (the decay rate).
        assert abs(result["jacobian_spectral_radius_est"] - rate) < 0.05

    def test_no_input_mutation(self):
        """Input array is not modified."""
        history = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        history_copy = history.copy()
        estimate_bp_jacobian_spectral_radius(history)
        np.testing.assert_array_equal(history, history_copy)
