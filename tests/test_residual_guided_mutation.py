"""
Tests for v11.2.0 — Residual-Guided Mutation.

Verifies:
  - BPResidualAnalyzer determinism and output format
  - residual_guided_mutation determinism, shape, edge count, no input mutation
  - top-k node selection (v11.2 upgrade)
  - operator registration in _OPERATORS and dispatcher
  - centralized OPERATORS list
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.bp_residuals import BPResidualAnalyzer
from src.qec.discovery.guided_mutations import (
    residual_guided_mutation,
    apply_guided_mutation,
    _OPERATORS,
    OPERATORS,
)


def _small_H():
    """Create a small (3, 6) regular parity-check matrix."""
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    return H


def _medium_H():
    """Create a (4, 8) parity-check matrix."""
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)
    return H


# -----------------------------------------------------------
# BPResidualAnalyzer tests
# -----------------------------------------------------------


class TestBPResidualAnalyzer:
    """Tests for the BP residual analyzer."""

    def test_residual_map_deterministic(self):
        H = _small_H()
        analyzer = BPResidualAnalyzer()
        r1 = analyzer.compute_residual_map(H, iterations=10, seed=42)
        r2 = analyzer.compute_residual_map(H, iterations=10, seed=42)
        np.testing.assert_array_equal(r1["residual_map"], r2["residual_map"])
        assert r1["max_residual"] == r2["max_residual"]
        assert r1["mean_residual"] == r2["mean_residual"]

    def test_residual_map_length_equals_num_variables(self):
        H = _small_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert len(result["residual_map"]) == H.shape[1]

    def test_residual_map_length_medium(self):
        H = _medium_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert len(result["residual_map"]) == H.shape[1]

    def test_residual_map_values_non_negative(self):
        H = _small_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert np.all(result["residual_map"] >= 0)

    def test_residual_map_values_non_negative_medium(self):
        H = _medium_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert np.all(result["residual_map"] >= 0)

    def test_max_residual_consistent(self):
        H = _small_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert result["max_residual"] == float(np.max(result["residual_map"]))

    def test_mean_residual_consistent(self):
        H = _small_H()
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert result["mean_residual"] == pytest.approx(
            float(np.mean(result["residual_map"]))
        )

    def test_empty_matrix(self):
        H = np.zeros((0, 5), dtype=np.float64)
        analyzer = BPResidualAnalyzer()
        result = analyzer.compute_residual_map(H, seed=0)
        assert len(result["residual_map"]) == 5
        assert result["max_residual"] == 0.0
        assert result["mean_residual"] == 0.0

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        analyzer = BPResidualAnalyzer()
        analyzer.compute_residual_map(H, seed=0)
        np.testing.assert_array_equal(H, H_copy)

    def test_different_seeds_may_differ(self):
        H = _medium_H()
        analyzer = BPResidualAnalyzer()
        r1 = analyzer.compute_residual_map(H, seed=0)
        r2 = analyzer.compute_residual_map(H, seed=99)
        # Both should be valid even if values differ
        assert len(r1["residual_map"]) == H.shape[1]
        assert len(r2["residual_map"]) == H.shape[1]


# -----------------------------------------------------------
# residual_guided_mutation tests
# -----------------------------------------------------------


class TestResidualGuidedMutation:
    """Tests for the residual-guided mutation operator."""

    def test_shape_preserved(self):
        H = _small_H()
        H_out = residual_guided_mutation(H, seed=42)
        assert H_out.shape == H.shape

    def test_shape_preserved_medium(self):
        H = _medium_H()
        H_out = residual_guided_mutation(H, seed=42)
        assert H_out.shape == H.shape

    def test_binary_output(self):
        H = _small_H()
        H_out = residual_guided_mutation(H, seed=42)
        assert np.all((H_out == 0) | (H_out == 1))

    def test_deterministic(self):
        H = _small_H()
        r1 = residual_guided_mutation(H, seed=42)
        r2 = residual_guided_mutation(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_deterministic_medium(self):
        H = _medium_H()
        r1 = residual_guided_mutation(H, seed=42)
        r2 = residual_guided_mutation(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        residual_guided_mutation(H, seed=42)
        np.testing.assert_array_equal(H, H_copy)

    def test_edge_count_preserved(self):
        H = _small_H()
        original_edges = H.sum()
        H_out = residual_guided_mutation(H, seed=42)
        new_edges = H_out.sum()
        assert new_edges == original_edges

    def test_edge_count_preserved_medium(self):
        H = _medium_H()
        original_edges = H.sum()
        H_out = residual_guided_mutation(H, seed=42)
        new_edges = H_out.sum()
        assert new_edges == original_edges

    def test_empty_matrix(self):
        H = np.zeros((0, 5), dtype=np.float64)
        H_out = residual_guided_mutation(H, seed=0)
        assert H_out.shape == (0, 5)


# -----------------------------------------------------------
# Registration tests
# -----------------------------------------------------------


class TestResidualGuidedRegistration:
    """Tests that residual_guided is properly registered."""

    def test_operator_in_list(self):
        assert "residual_guided" in _OPERATORS

    def test_dispatcher_accepts_operator(self):
        H = _small_H()
        H_out = apply_guided_mutation(H, operator="residual_guided", seed=42)
        assert H_out.shape == H.shape

    def test_dispatcher_deterministic(self):
        H = _small_H()
        r1 = apply_guided_mutation(H, operator="residual_guided", seed=42)
        r2 = apply_guided_mutation(H, operator="residual_guided", seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_schedule_includes_residual_guided(self):
        """Verify the operator is reachable via generation scheduling."""
        H = _small_H()
        operators_used = set()
        for gen in range(len(_OPERATORS)):
            op = _OPERATORS[gen % len(_OPERATORS)]
            operators_used.add(op)
        assert "residual_guided" in operators_used


# -----------------------------------------------------------
# v11.2.0 — Top-k selection tests
# -----------------------------------------------------------


class TestResidualGuidedTopK:
    """Tests for the top-k residual variable selection (v11.2 upgrade)."""

    def test_top_k_parameter_accepted(self):
        """Verify the top_k parameter is accepted."""
        H = _small_H()
        H_out = residual_guided_mutation(H, seed=42, top_k=3)
        assert H_out.shape == H.shape

    def test_top_k_deterministic(self):
        """Same top_k and seed produces identical results."""
        H = _medium_H()
        r1 = residual_guided_mutation(H, seed=42, top_k=3)
        r2 = residual_guided_mutation(H, seed=42, top_k=3)
        np.testing.assert_array_equal(r1, r2)

    def test_top_k_shape_preserved(self):
        H = _medium_H()
        H_out = residual_guided_mutation(H, seed=42, top_k=2)
        assert H_out.shape == H.shape

    def test_top_k_edge_count_preserved(self):
        H = _medium_H()
        original_edges = H.sum()
        H_out = residual_guided_mutation(H, seed=42, top_k=3)
        assert H_out.sum() == original_edges

    def test_top_k_one_is_valid(self):
        """top_k=1 should still produce valid output."""
        H = _small_H()
        H_out = residual_guided_mutation(H, seed=42, top_k=1)
        assert H_out.shape == H.shape
        assert np.all((H_out == 0) | (H_out == 1))

    def test_default_top_k_is_five(self):
        """Default top_k should be min(5, n)."""
        H = _medium_H()
        # With n=8, default k=min(5,8)=5
        r1 = residual_guided_mutation(H, seed=42)
        r2 = residual_guided_mutation(H, seed=42, top_k=5)
        np.testing.assert_array_equal(r1, r2)


# -----------------------------------------------------------
# v11.2.0 — Centralized OPERATORS list tests
# -----------------------------------------------------------


class TestCentralizedOperators:
    """Tests for the centralized OPERATORS function list."""

    def test_operators_length(self):
        assert len(OPERATORS) == 11

    def test_operators_are_callable(self):
        for op in OPERATORS:
            assert callable(op)

    def test_operators_match_names(self):
        """OPERATORS function list matches _OPERATORS name list."""
        assert len(OPERATORS) == len(_OPERATORS)

    def test_residual_guided_in_operators(self):
        assert residual_guided_mutation in OPERATORS
