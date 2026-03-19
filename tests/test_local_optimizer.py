"""
Tests for the v11.4.0 Local Graph Optimizer.

Verifies:
  - deterministic output
  - no input mutation
  - shape preserved
  - edge count preserved
  - fitness improves or stays stable
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.local_optimizer import LocalGraphOptimizer, _try_rewire
from qec.fitness.fitness_engine import FitnessEngine


def _make_test_matrix():
    """Create a small test parity-check matrix (3x6, variable degree 2)."""
    H = np.array([
        [1, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0, 1],
    ], dtype=np.float64)
    return H


def _make_larger_test_matrix():
    """Create a larger test matrix (4x8) for more diverse rewiring."""
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)
    return H


class TestTryRewire:
    """Tests for the _try_rewire helper."""

    def test_valid_rewire(self):
        H = _make_test_matrix()
        result = _try_rewire(H, 0, 0, 2, 4)
        assert result is not None
        assert result[0, 0] == 0
        assert result[2, 4] == 1

    def test_no_source_edge(self):
        H = _make_test_matrix()
        result = _try_rewire(H, 0, 4, 2, 4)
        assert result is None

    def test_target_occupied(self):
        H = _make_test_matrix()
        result = _try_rewire(H, 0, 0, 0, 1)
        assert result is None

    def test_same_position(self):
        H = _make_test_matrix()
        result = _try_rewire(H, 0, 0, 0, 0)
        assert result is None

    def test_no_input_mutation(self):
        H = _make_test_matrix()
        H_copy = H.copy()
        _try_rewire(H, 0, 0, 2, 4)
        np.testing.assert_array_equal(H, H_copy)

    def test_preserves_edge_count(self):
        H = _make_test_matrix()
        original_edges = int(np.count_nonzero(H))
        result = _try_rewire(H, 0, 0, 2, 4)
        assert result is not None
        assert int(np.count_nonzero(result)) == original_edges


class TestLocalGraphOptimizer:
    """Tests for the LocalGraphOptimizer class."""

    def test_init_defaults(self):
        opt = LocalGraphOptimizer()
        assert opt.max_steps == 10
        assert opt.seed == 42

    def test_init_custom(self):
        opt = LocalGraphOptimizer(max_steps=5, seed=99)
        assert opt.max_steps == 5
        assert opt.seed == 99

    def test_output_shape_preserved(self):
        H = _make_test_matrix()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        result = opt.optimize(H, engine)
        assert result.shape == H.shape

    def test_output_edge_count_preserved(self):
        H = _make_test_matrix()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        original_edges = int(np.count_nonzero(H))
        result = opt.optimize(H, engine)
        assert int(np.count_nonzero(result)) == original_edges

    def test_no_input_mutation(self):
        H = _make_test_matrix()
        H_copy = H.copy()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        opt.optimize(H, engine)
        np.testing.assert_array_equal(H, H_copy)

    def test_deterministic_output(self):
        H = _make_test_matrix()
        engine1 = FitnessEngine(seed=42)
        engine2 = FitnessEngine(seed=42)

        opt1 = LocalGraphOptimizer(max_steps=3, seed=42)
        opt2 = LocalGraphOptimizer(max_steps=3, seed=42)

        result1 = opt1.optimize(H.copy(), engine1)
        result2 = opt2.optimize(H.copy(), engine2)

        np.testing.assert_array_equal(result1, result2)

    def test_fitness_does_not_decrease(self):
        H = _make_test_matrix()
        engine = FitnessEngine(seed=42)

        original_fitness = engine.evaluate(H)["composite"]

        opt = LocalGraphOptimizer(max_steps=5, seed=42)
        result = opt.optimize(H, engine)

        new_fitness = engine.evaluate(result)["composite"]
        assert new_fitness >= original_fitness

    def test_zero_steps_returns_unchanged(self):
        H = _make_test_matrix()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=0, seed=42)
        result = opt.optimize(H, engine)
        np.testing.assert_array_equal(result, H)

    def test_binary_output(self):
        H = _make_test_matrix()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        result = opt.optimize(H, engine)
        unique_vals = set(np.unique(result).tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_larger_matrix(self):
        H = _make_larger_test_matrix()
        engine = FitnessEngine(seed=42)
        opt = LocalGraphOptimizer(max_steps=5, seed=42)

        original_fitness = engine.evaluate(H)["composite"]
        result = opt.optimize(H, engine)

        assert result.shape == H.shape
        assert int(np.count_nonzero(result)) == int(np.count_nonzero(H))
        new_fitness = engine.evaluate(result)["composite"]
        assert new_fitness >= original_fitness

    def test_different_seeds_may_differ(self):
        H = _make_larger_test_matrix()
        engine1 = FitnessEngine(seed=42)
        engine2 = FitnessEngine(seed=42)

        opt1 = LocalGraphOptimizer(max_steps=5, seed=0)
        opt2 = LocalGraphOptimizer(max_steps=5, seed=99)

        result1 = opt1.optimize(H.copy(), engine1)
        result2 = opt2.optimize(H.copy(), engine2)

        # Both should be valid
        assert result1.shape == H.shape
        assert result2.shape == H.shape


class TestLocalSearchOperators:
    """Tests for individual local search operator methods."""

    def test_absorbing_set_repair_returns_list(self):
        H = _make_test_matrix()
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        candidates = opt._absorbing_set_repair(H, seed=42)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, np.ndarray)
            assert c.shape == H.shape

    def test_residual_hotspot_returns_list(self):
        H = _make_test_matrix()
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        candidates = opt._residual_hotspot_smoothing(H, seed=42)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, np.ndarray)
            assert c.shape == H.shape

    def test_cycle_irregularity_returns_list(self):
        H = _make_test_matrix()
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        candidates = opt._cycle_irregularity_reduction(H, seed=42)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, np.ndarray)
            assert c.shape == H.shape

    def test_bethe_hessian_returns_list(self):
        H = _make_test_matrix()
        opt = LocalGraphOptimizer(max_steps=3, seed=42)
        candidates = opt._bethe_hessian_improvement(H, seed=42)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, np.ndarray)
            assert c.shape == H.shape

    def test_operators_preserve_edge_count(self):
        H = _make_test_matrix()
        original_edges = int(np.count_nonzero(H))
        opt = LocalGraphOptimizer(max_steps=3, seed=42)

        for method in [
            opt._absorbing_set_repair,
            opt._residual_hotspot_smoothing,
            opt._cycle_irregularity_reduction,
            opt._bethe_hessian_improvement,
        ]:
            candidates = method(H, seed=42)
            for c in candidates:
                assert int(np.count_nonzero(c)) == original_edges, (
                    f"{method.__name__} changed edge count"
                )

    def test_operators_no_input_mutation(self):
        H = _make_test_matrix()
        H_copy = H.copy()
        opt = LocalGraphOptimizer(max_steps=3, seed=42)

        for method in [
            opt._absorbing_set_repair,
            opt._residual_hotspot_smoothing,
            opt._cycle_irregularity_reduction,
            opt._bethe_hessian_improvement,
        ]:
            method(H, seed=42)
            np.testing.assert_array_equal(H, H_copy)
