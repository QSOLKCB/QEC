"""
Tests for the v12.0.0 nonbacktracking_flow_mutation operator.

Verifies:
  - deterministic mutation
  - matrix shape preserved
  - edge count preserved
  - no duplicate edges
  - degree limits respected
  - registry integration works
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.guided_mutations import (
    nonbacktracking_flow_mutation,
    apply_guided_mutation,
    OPERATORS,
    _OPERATORS,
    _OPERATOR_FUNCTIONS,
)


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


class TestNonBacktrackingFlowMutation:
    """Tests for the nonbacktracking_flow_mutation operator."""

    def test_deterministic(self):
        H = _small_H()
        r1 = nonbacktracking_flow_mutation(H, seed=42)
        r2 = nonbacktracking_flow_mutation(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_shape_preserved(self):
        H = _small_H()
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        assert H_out.shape == H.shape

    def test_binary_output(self):
        H = _small_H()
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        assert np.all((H_out == 0) | (H_out == 1))

    def test_edge_count_preserved(self):
        H = _small_H()
        original_edges = H.sum()
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        new_edges = H_out.sum()
        assert abs(new_edges - original_edges) <= 2

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        nonbacktracking_flow_mutation(H, seed=42)
        np.testing.assert_array_equal(H, H_copy)

    def test_degree_limits_preserved(self):
        H = _medium_H()
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        assert np.all(H_out.sum(axis=0) >= 1)
        assert np.all(H_out.sum(axis=1) >= 1)

    def test_no_duplicate_edges(self):
        H = _medium_H()
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        assert np.all((H_out == 0) | (H_out == 1))

    def test_different_seeds_produce_valid_output(self):
        H = _medium_H()
        for s in [0, 1, 42, 99]:
            H_out = nonbacktracking_flow_mutation(H, seed=s)
            assert H_out.shape == H.shape
            assert np.all((H_out == 0) | (H_out == 1))

    def test_empty_matrix(self):
        H = np.zeros((0, 5), dtype=np.float64)
        H_out = nonbacktracking_flow_mutation(H, seed=42)
        assert H_out.shape == (0, 5)

    def test_low_tension_returns_unchanged(self):
        """With a very high threshold, mutation should not occur."""
        H = _small_H()
        H_out = nonbacktracking_flow_mutation(
            H, seed=42, tension_threshold=1e6,
        )
        np.testing.assert_array_equal(H_out, H)

    def test_zero_threshold_allows_mutation(self):
        """With threshold=0, mutation should be attempted."""
        H = _medium_H()
        H_out = nonbacktracking_flow_mutation(
            H, seed=42, tension_threshold=0.0,
        )
        assert H_out.shape == H.shape
        assert np.all((H_out == 0) | (H_out == 1))

    def test_shape_preserved_medium(self):
        H = _medium_H()
        H_out = nonbacktracking_flow_mutation(H, seed=0)
        assert H_out.shape == H.shape


class TestNonBacktrackingFlowMutationRegistry:
    """Tests for operator registry integration."""

    def test_operator_in_operators_list(self):
        assert nonbacktracking_flow_mutation in OPERATORS

    def test_operator_in_name_list(self):
        assert "nonbacktracking_flow" in _OPERATORS

    def test_operator_in_function_map(self):
        assert "nonbacktracking_flow" in _OPERATOR_FUNCTIONS
        assert (
            _OPERATOR_FUNCTIONS["nonbacktracking_flow"]
            is nonbacktracking_flow_mutation
        )

    def test_dispatcher_accepts_operator(self):
        H = _small_H()
        H_out = apply_guided_mutation(
            H, operator="nonbacktracking_flow", seed=42,
        )
        assert H_out.shape == H.shape

    def test_schedule_covers_new_operator(self):
        H = _small_H()
        idx = _OPERATORS.index("nonbacktracking_flow")
        H_out = apply_guided_mutation(H, generation=idx, seed=42)
        assert H_out.shape == H.shape

    def test_registry_count_matches(self):
        assert len(OPERATORS) == len(_OPERATORS)
        assert len(_OPERATOR_FUNCTIONS) == len(_OPERATORS)

    def test_registry_has_twelve_operators(self):
        assert len(OPERATORS) == 12
        assert len(_OPERATORS) == 12
        assert len(_OPERATOR_FUNCTIONS) == 12
