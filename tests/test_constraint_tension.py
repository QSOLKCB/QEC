"""
Tests for the v12.0.0 ConstraintTensionAnalyzer.

Verifies:
  - deterministic kappa
  - triadic labels only {0, 1, 2}
  - no NaNs
  - stable ordering
  - no input mutation
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.constraint_tension import ConstraintTensionAnalyzer
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


def _small_H():
    """Create a small (3, 6) regular parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _medium_H():
    """Create a (4, 8) parity-check matrix."""
    return np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)


class TestConstraintTensionAnalyzer:
    """Tests for ConstraintTensionAnalyzer.compute_tension."""

    def test_deterministic(self):
        H = _small_H()
        analyzer = ConstraintTensionAnalyzer()
        r1 = analyzer.compute_tension(H)
        r2 = analyzer.compute_tension(H)
        assert r1["tension"] == r2["tension"]
        np.testing.assert_array_equal(r1["state_labels"], r2["state_labels"])

    def test_deterministic_with_flow(self):
        H = _small_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        analyzer = ConstraintTensionAnalyzer()
        r1 = analyzer.compute_tension(H, flow=flow)
        r2 = analyzer.compute_tension(H, flow=flow)
        assert r1["tension"] == r2["tension"]
        np.testing.assert_array_equal(r1["state_labels"], r2["state_labels"])

    def test_triadic_labels_only(self):
        H = _medium_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        result = ConstraintTensionAnalyzer().compute_tension(H, flow=flow)
        labels = result["state_labels"]
        assert set(np.unique(labels)).issubset({0, 1, 2})

    def test_state_labels_shape(self):
        H = _small_H()
        result = ConstraintTensionAnalyzer().compute_tension(H)
        assert result["state_labels"].shape == (6,)

    def test_state_labels_shape_medium(self):
        H = _medium_H()
        result = ConstraintTensionAnalyzer().compute_tension(H)
        assert result["state_labels"].shape == (8,)

    def test_no_nans(self):
        H = _medium_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        residual_map = np.random.RandomState(42).rand(8)
        result = ConstraintTensionAnalyzer().compute_tension(
            H, flow=flow, residual_map=residual_map,
        )
        assert not np.isnan(result["tension"])
        assert not np.isnan(result["residual_component"])
        assert not np.isnan(result["flow_component"])
        assert not np.isnan(result["cluster_component"])
        assert not np.isnan(result["trapping_component"])
        assert not np.any(np.isnan(result["state_labels"]))

    def test_tension_non_negative(self):
        H = _small_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        result = ConstraintTensionAnalyzer().compute_tension(H, flow=flow)
        assert result["tension"] >= 0

    def test_components_non_negative(self):
        H = _medium_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        residual_map = np.random.RandomState(42).rand(8)
        result = ConstraintTensionAnalyzer().compute_tension(
            H, flow=flow, residual_map=residual_map,
        )
        assert result["residual_component"] >= 0
        assert result["flow_component"] >= 0
        assert result["cluster_component"] >= 0
        assert result["trapping_component"] >= 0

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        ConstraintTensionAnalyzer().compute_tension(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((3, 0), dtype=np.float64)
        result = ConstraintTensionAnalyzer().compute_tension(H)
        assert result["tension"] == 0.0
        assert result["state_labels"].shape == (0,)

    def test_zero_tension_without_inputs(self):
        H = _small_H()
        result = ConstraintTensionAnalyzer().compute_tension(H)
        # Without residual_map, flow, or clusters, tension should be 0.
        assert result["tension"] == 0.0

    def test_tension_increases_with_residuals(self):
        H = _medium_H()
        r_zero = ConstraintTensionAnalyzer().compute_tension(H)
        residual_map = np.ones(8, dtype=np.float64)
        r_with = ConstraintTensionAnalyzer().compute_tension(
            H, residual_map=residual_map,
        )
        assert r_with["tension"] >= r_zero["tension"]

    def test_custom_weights(self):
        H = _medium_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        residual_map = np.random.RandomState(42).rand(8)
        r1 = ConstraintTensionAnalyzer(wr=2.0).compute_tension(
            H, flow=flow, residual_map=residual_map,
        )
        r2 = ConstraintTensionAnalyzer(wr=0.0).compute_tension(
            H, flow=flow, residual_map=residual_map,
        )
        # With higher residual weight, tension should be at least as high.
        assert r1["tension"] >= r2["tension"]

    def test_stable_ordering(self):
        """Running multiple times produces identical label ordering."""
        H = _medium_H()
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        analyzer = ConstraintTensionAnalyzer()
        results = [
            analyzer.compute_tension(H, flow=flow) for _ in range(5)
        ]
        for r in results[1:]:
            np.testing.assert_array_equal(
                results[0]["state_labels"], r["state_labels"],
            )
