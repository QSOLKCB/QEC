"""
Tests for the v10.1.0 IPR trapping pressure mutation operator.

Verifies:
  - deterministic behavior (same seed → identical output)
  - output preserves matrix shape
  - no input mutation
  - valid Tanner graph after mutation
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
    ipr_trapping_pressure_mutation,
    apply_guided_mutation,
    _OPERATORS,
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


class TestIPRTrappingPressureDeterminism:
    """Determinism: same seed must produce identical output."""

    def test_deterministic_small(self):
        H = _small_H()
        r1 = ipr_trapping_pressure_mutation(H, seed=42)
        r2 = ipr_trapping_pressure_mutation(H, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_deterministic_medium(self):
        H = _medium_H()
        r1 = ipr_trapping_pressure_mutation(H, seed=99)
        r2 = ipr_trapping_pressure_mutation(H, seed=99)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_may_differ(self):
        H = _medium_H()
        r1 = ipr_trapping_pressure_mutation(H, seed=0)
        r2 = ipr_trapping_pressure_mutation(H, seed=12345)
        # Not guaranteed to differ but seeds are different
        # Just verify both are valid
        assert r1.shape == H.shape
        assert r2.shape == H.shape


class TestIPRTrappingPressureShapePreservation:
    """Matrix shape must be preserved."""

    def test_shape_small(self):
        H = _small_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert result.shape == H.shape

    def test_shape_medium(self):
        H = _medium_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert result.shape == H.shape


class TestIPRTrappingPressureNoInputMutation:
    """Input matrix must not be modified."""

    def test_no_mutation_small(self):
        H = _small_H()
        H_copy = H.copy()
        _ = ipr_trapping_pressure_mutation(H, seed=0)
        np.testing.assert_array_equal(H, H_copy)

    def test_no_mutation_medium(self):
        H = _medium_H()
        H_copy = H.copy()
        _ = ipr_trapping_pressure_mutation(H, seed=0)
        np.testing.assert_array_equal(H, H_copy)


class TestIPRTrappingPressureValidGraph:
    """After mutation, the graph must remain valid."""

    def test_binary_values(self):
        H = _small_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_variable_degrees_positive(self):
        H = _medium_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        col_sums = result.sum(axis=0)
        assert np.all(col_sums >= 1), "All variable nodes must have degree >= 1"

    def test_check_degrees_positive(self):
        H = _medium_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        row_sums = result.sum(axis=1)
        assert np.all(row_sums >= 1), "All check nodes must have degree >= 1"

    def test_edge_count_preserved(self):
        """Total edge count should be preserved (one remove + one add)."""
        H = _medium_H()
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert H.sum() == result.sum()


class TestIPRTrappingPressureRegistration:
    """Operator must be registered in the dispatch system."""

    def test_in_operators_list(self):
        assert "ipr_trapping_pressure" in _OPERATORS

    def test_dispatch_by_name(self):
        H = _small_H()
        result = apply_guided_mutation(
            H, operator="ipr_trapping_pressure", seed=0,
        )
        assert result.shape == H.shape

    def test_dispatch_by_generation(self):
        """Operator should be reachable via generation scheduling."""
        idx = _OPERATORS.index("ipr_trapping_pressure")
        H = _small_H()
        result = apply_guided_mutation(H, generation=idx, seed=0)
        assert result.shape == H.shape


class TestIPRTrappingPressureEdgeCases:
    """Edge cases for robustness."""

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert result.shape == (0, 0)

    def test_single_edge(self):
        """Matrix with minimal edges should return safely."""
        H = np.array([[1, 0], [0, 1]], dtype=np.float64)
        result = ipr_trapping_pressure_mutation(H, seed=0)
        assert result.shape == H.shape
