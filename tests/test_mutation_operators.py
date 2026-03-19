"""
Tests for v9.0.0 mutation operators.

Verifies:
  - mutation preserves degree constraints (approximately)
  - mutation is deterministic
  - all operators produce valid matrices
  - operator scheduling is correct
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.mutation_operators import (
    edge_swap,
    local_rewire,
    cycle_break,
    degree_preserving_rotation,
    seeded_reconstruction,
    mutate_tanner_graph,
    get_operator_for_generation,
)


def _small_H():
    """3x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


class TestEdgeSwap:
    def test_returns_valid_binary(self):
        H = _small_H()
        H_mut = edge_swap(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_deterministic(self):
        H = _small_H()
        H1 = edge_swap(H, seed=42)
        H2 = edge_swap(H, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_no_zero_rows(self):
        H = _small_H()
        H_mut = edge_swap(H, seed=42)
        assert np.all(H_mut.sum(axis=1) > 0)

    def test_no_zero_cols(self):
        H = _small_H()
        H_mut = edge_swap(H, seed=42)
        assert np.all(H_mut.sum(axis=0) > 0)

    def test_with_target_edges(self):
        H = _small_H()
        targets = [(0, 0), (1, 1)]
        H_mut = edge_swap(H, seed=42, target_edges=targets)
        assert np.all((H_mut == 0) | (H_mut == 1))


class TestLocalRewire:
    def test_returns_valid_binary(self):
        H = _small_H()
        H_mut = local_rewire(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_deterministic(self):
        H = _small_H()
        H1 = local_rewire(H, seed=42)
        H2 = local_rewire(H, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_preserves_variable_degree(self):
        H = _small_H()
        H_mut = local_rewire(H, seed=42)
        # Variable degree should be preserved exactly
        np.testing.assert_array_equal(H.sum(axis=0), H_mut.sum(axis=0))


class TestCycleBreak:
    def test_returns_valid_binary(self):
        H = _small_H()
        H_mut = cycle_break(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_deterministic(self):
        H = _small_H()
        H1 = cycle_break(H, seed=42)
        H2 = cycle_break(H, seed=42)
        np.testing.assert_array_equal(H1, H2)


class TestDegreePreservingRotation:
    def test_returns_valid_binary(self):
        H = _small_H()
        H_mut = degree_preserving_rotation(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_deterministic(self):
        H = _small_H()
        H1 = degree_preserving_rotation(H, seed=42)
        H2 = degree_preserving_rotation(H, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_preserves_degrees(self):
        H = _small_H()
        H_mut = degree_preserving_rotation(H, seed=42)
        # Both row and column sums should be preserved
        np.testing.assert_array_equal(H.sum(axis=0), H_mut.sum(axis=0))
        np.testing.assert_array_equal(H.sum(axis=1), H_mut.sum(axis=1))


class TestSeededReconstruction:
    def test_returns_valid_binary(self):
        H = _small_H()
        H_mut = seeded_reconstruction(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))

    def test_deterministic(self):
        H = _small_H()
        H1 = seeded_reconstruction(H, seed=42)
        H2 = seeded_reconstruction(H, seed=42)
        np.testing.assert_array_equal(H1, H2)


class TestMutateTannerGraph:
    def test_scheduled_operator(self):
        assert get_operator_for_generation(0) == "edge_swap"
        assert get_operator_for_generation(1) == "local_rewire"
        assert get_operator_for_generation(5) == "cycle_guided_mutation"
        assert get_operator_for_generation(6) == "spectral_pressure_guided_mutation"
        assert get_operator_for_generation(7) == "edge_swap"

    def test_returns_valid(self):
        H = _small_H()
        H_mut, op = mutate_tanner_graph(H, seed=42)
        assert np.all((H_mut == 0) | (H_mut == 1))
        assert op in [
            "edge_swap", "local_rewire", "cycle_break",
            "degree_preserving_rotation", "seeded_reconstruction",
            "cycle_guided_mutation", "spectral_pressure_guided_mutation",
        ]

    def test_explicit_operator(self):
        H = _small_H()
        H_mut, op = mutate_tanner_graph(H, operator="local_rewire", seed=42)
        assert op == "local_rewire"

    def test_deterministic(self):
        H = _small_H()
        H1, op1 = mutate_tanner_graph(H, seed=42, generation=3)
        H2, op2 = mutate_tanner_graph(H, seed=42, generation=3)
        np.testing.assert_array_equal(H1, H2)
        assert op1 == op2

    def test_unknown_operator_raises(self):
        H = _small_H()
        with pytest.raises(ValueError, match="Unknown mutation operator"):
            mutate_tanner_graph(H, operator="nonexistent")

    def test_does_not_mutate_input(self):
        H = _small_H()
        H_orig = H.copy()
        mutate_tanner_graph(H, seed=42)
        np.testing.assert_array_equal(H, H_orig)
