"""
Tests for the v12.0.0 NonBacktrackingFlowAnalyzer.

Verifies:
  - deterministic output
  - correct array shapes
  - non-negative flows
  - sparse/dense equivalence
  - no input mutation
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.sparse

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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


class TestNonBacktrackingFlowAnalyzer:
    """Tests for NonBacktrackingFlowAnalyzer.compute_flow."""

    def test_deterministic(self):
        H = _small_H()
        analyzer = NonBacktrackingFlowAnalyzer()
        r1 = analyzer.compute_flow(H)
        r2 = analyzer.compute_flow(H)
        np.testing.assert_array_equal(r1["variable_flow"], r2["variable_flow"])
        np.testing.assert_array_equal(r1["edge_flow"], r2["edge_flow"])
        np.testing.assert_array_equal(r1["check_flow"], r2["check_flow"])
        np.testing.assert_array_equal(
            r1["directed_edge_flow"], r2["directed_edge_flow"],
        )
        assert r1["max_flow"] == r2["max_flow"]
        assert r1["mean_flow"] == r2["mean_flow"]
        assert r1["flow_localization"] == r2["flow_localization"]

    def test_variable_flow_shape(self):
        H = _small_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["variable_flow"].shape == (6,)

    def test_check_flow_shape(self):
        H = _small_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["check_flow"].shape == (3,)

    def test_edge_flow_shape(self):
        H = _small_H()
        num_edges = int(H.sum())
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["edge_flow"].shape == (num_edges,)

    def test_directed_edge_flow_shape(self):
        H = _small_H()
        num_edges = int(H.sum())
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["directed_edge_flow"].shape == (2 * num_edges,)

    def test_non_negative_flows(self):
        H = _medium_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert np.all(result["variable_flow"] >= 0)
        assert np.all(result["edge_flow"] >= 0)
        assert np.all(result["check_flow"] >= 0)
        assert np.all(result["directed_edge_flow"] >= 0)
        assert result["max_flow"] >= 0
        assert result["mean_flow"] >= 0
        assert result["flow_localization"] >= 0

    def test_variable_flow_normalized(self):
        H = _medium_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        vf = result["variable_flow"]
        assert np.all(vf >= 0)
        assert np.all(vf <= 1.0 + 1e-12)

    def test_sparse_dense_equivalence(self):
        H_dense = _small_H()
        H_sparse = scipy.sparse.csr_matrix(H_dense)
        analyzer = NonBacktrackingFlowAnalyzer()
        r_dense = analyzer.compute_flow(H_dense)
        r_sparse = analyzer.compute_flow(H_sparse)
        np.testing.assert_array_equal(
            r_dense["variable_flow"], r_sparse["variable_flow"],
        )
        np.testing.assert_array_equal(
            r_dense["edge_flow"], r_sparse["edge_flow"],
        )
        assert r_dense["max_flow"] == r_sparse["max_flow"]
        assert r_dense["mean_flow"] == r_sparse["mean_flow"]
        assert r_dense["flow_localization"] == r_sparse["flow_localization"]

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        NonBacktrackingFlowAnalyzer().compute_flow(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 5), dtype=np.float64)
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["variable_flow"].shape == (5,)
        assert result["max_flow"] == 0.0

    def test_zero_matrix(self):
        H = np.zeros((3, 6), dtype=np.float64)
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["max_flow"] == 0.0
        assert np.all(result["variable_flow"] == 0)

    def test_no_nans(self):
        H = _medium_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert not np.any(np.isnan(result["variable_flow"]))
        assert not np.any(np.isnan(result["edge_flow"]))
        assert not np.any(np.isnan(result["check_flow"]))
        assert not np.isnan(result["max_flow"])
        assert not np.isnan(result["mean_flow"])
        assert not np.isnan(result["flow_localization"])

    def test_medium_matrix_shapes(self):
        H = _medium_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert result["variable_flow"].shape == (8,)
        assert result["check_flow"].shape == (4,)
        num_edges = int(H.sum())
        assert result["edge_flow"].shape == (num_edges,)
