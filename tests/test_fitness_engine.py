"""
Tests for the v10.0.0 fitness engine.

Verifies:
  - spectral metrics produce deterministic results
  - fitness engine composite scores are deterministic
  - caching works correctly
  - sparse matrix integrity is maintained
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.fitness.spectral_metrics import (
    compute_nbt_spectral_radius,
    compute_girth_spectrum,
    compute_ace_spectrum,
    estimate_eigenvector_ipr,
)
from qec.fitness.fitness_engine import FitnessEngine


def _small_H():
    """Create a small (3, 6) regular parity-check matrix."""
    H = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    return H


def _tiny_H():
    """Create a minimal (2, 4) matrix."""
    H = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ], dtype=np.float64)
    return H


class TestNBTSpectralRadius:
    """Tests for compute_nbt_spectral_radius."""

    def test_returns_float(self):
        H = _small_H()
        result = compute_nbt_spectral_radius(H)
        assert isinstance(result, float)

    def test_non_negative(self):
        H = _small_H()
        result = compute_nbt_spectral_radius(H)
        assert result >= 0.0

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_nbt_spectral_radius(H)
        r2 = compute_nbt_spectral_radius(H)
        assert r1 == r2

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        assert compute_nbt_spectral_radius(H) == 0.0

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        compute_nbt_spectral_radius(H)
        np.testing.assert_array_equal(H, H_copy)


class TestGirthSpectrum:
    """Tests for compute_girth_spectrum."""

    def test_returns_dict(self):
        H = _small_H()
        result = compute_girth_spectrum(H)
        assert "girth" in result
        assert "cycle_counts" in result

    def test_girth_positive(self):
        H = _small_H()
        result = compute_girth_spectrum(H)
        assert result["girth"] >= 0

    def test_cycle_counts_keys(self):
        H = _small_H()
        result = compute_girth_spectrum(H)
        assert 4 in result["cycle_counts"]
        assert 6 in result["cycle_counts"]
        assert 8 in result["cycle_counts"]

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_girth_spectrum(H)
        r2 = compute_girth_spectrum(H)
        assert r1 == r2

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        compute_girth_spectrum(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_tree_like_no_short_cycles(self):
        # A tree-like structure has no cycles
        H = _tiny_H()
        result = compute_girth_spectrum(H)
        assert result["cycle_counts"][4] == 0


class TestACESpectrum:
    """Tests for compute_ace_spectrum."""

    def test_returns_ndarray(self):
        H = _small_H()
        result = compute_ace_spectrum(H)
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        H = _small_H()
        result = compute_ace_spectrum(H)
        assert result.shape == (6,)

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_ace_spectrum(H)
        r2 = compute_ace_spectrum(H)
        np.testing.assert_array_equal(r1, r2)

    def test_non_negative(self):
        H = _small_H()
        result = compute_ace_spectrum(H)
        assert np.all(result >= 0)

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        compute_ace_spectrum(H)
        np.testing.assert_array_equal(H, H_copy)


class TestEigenvectorIPR:
    """Tests for estimate_eigenvector_ipr."""

    def test_returns_dict(self):
        H = _small_H()
        result = estimate_eigenvector_ipr(H)
        assert "mean_ipr" in result
        assert "max_ipr" in result

    def test_non_negative(self):
        H = _small_H()
        result = estimate_eigenvector_ipr(H)
        assert result["mean_ipr"] >= 0.0
        assert result["max_ipr"] >= 0.0

    def test_deterministic(self):
        H = _small_H()
        r1 = estimate_eigenvector_ipr(H)
        r2 = estimate_eigenvector_ipr(H)
        assert r1 == r2

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        estimate_eigenvector_ipr(H)
        np.testing.assert_array_equal(H, H_copy)


class TestFitnessEngine:
    """Tests for FitnessEngine."""

    def test_evaluate_returns_composite(self):
        engine = FitnessEngine()
        H = _small_H()
        result = engine.evaluate(H)
        assert "composite" in result
        assert "components" in result
        assert "metrics" in result

    def test_composite_is_float(self):
        engine = FitnessEngine()
        H = _small_H()
        result = engine.evaluate(H)
        assert isinstance(result["composite"], float)

    def test_deterministic(self):
        engine = FitnessEngine()
        H = _small_H()
        r1 = engine.evaluate(H)
        r2 = engine.evaluate(H)
        assert r1["composite"] == r2["composite"]

    def test_cache_works(self):
        engine = FitnessEngine()
        H = _small_H()
        r1 = engine.evaluate(H)
        r2 = engine.evaluate(H)
        # Should be the exact same object due to caching
        assert r1 is r2

    def test_clear_cache(self):
        engine = FitnessEngine()
        H = _small_H()
        r1 = engine.evaluate(H)
        engine.clear_cache()
        r2 = engine.evaluate(H)
        assert r1 is not r2
        assert r1["composite"] == r2["composite"]

    def test_custom_weights(self):
        engine1 = FitnessEngine()
        engine2 = FitnessEngine(weights={
            "girth": 10.0,
            "nbt_spectral_radius": 0.0,
            "ace_variance": 0.0,
            "expansion": 0.0,
            "cycle_density": 0.0,
            "sparsity": 0.0,
        })
        H = _small_H()
        r1 = engine1.evaluate(H)
        r2 = engine2.evaluate(H)
        # Different weights should give different composites
        assert r1["composite"] != r2["composite"]

    def test_metrics_present(self):
        engine = FitnessEngine()
        H = _small_H()
        result = engine.evaluate(H)
        metrics = result["metrics"]
        assert "nbt_spectral_radius" in metrics
        assert "girth" in metrics
        assert "expansion" in metrics
        assert "sparsity" in metrics

    def test_no_input_mutation(self):
        engine = FitnessEngine()
        H = _small_H()
        H_copy = H.copy()
        engine.evaluate(H)
        np.testing.assert_array_equal(H, H_copy)
