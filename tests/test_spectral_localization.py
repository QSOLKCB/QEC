"""
Tests for the v11.6.0 SpectralLocalizationAnalyzer.

Verifies:
  - deterministic results across identical runs
  - pressure arrays have valid shape
  - no NaN values in outputs
  - stable ordering across calls
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.spectral_localization import SpectralLocalizationAnalyzer


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


class TestSpectralLocalizationAnalyzer:
    """Tests for SpectralLocalizationAnalyzer.compute_pressure."""

    def test_deterministic(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        r1 = analyzer.compute_pressure(H)
        r2 = analyzer.compute_pressure(H)
        np.testing.assert_array_equal(
            r1["variable_pressure"], r2["variable_pressure"],
        )
        np.testing.assert_array_equal(
            r1["edge_pressure"], r2["edge_pressure"],
        )
        assert r1["max_pressure"] == r2["max_pressure"]
        assert r1["mean_pressure"] == r2["mean_pressure"]

    def test_variable_pressure_shape(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        result = analyzer.compute_pressure(H)
        assert result["variable_pressure"].shape == (H.shape[1],)

    def test_edge_pressure_shape(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        result = analyzer.compute_pressure(H)
        num_edges = int(H.sum())
        assert result["edge_pressure"].shape == (num_edges,)

    def test_no_nans_small(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        result = analyzer.compute_pressure(H)
        assert not np.any(np.isnan(result["variable_pressure"]))
        assert not np.any(np.isnan(result["edge_pressure"]))
        assert not np.isnan(result["max_pressure"])
        assert not np.isnan(result["mean_pressure"])

    def test_no_nans_medium(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _medium_H()
        result = analyzer.compute_pressure(H)
        assert not np.any(np.isnan(result["variable_pressure"]))
        assert not np.any(np.isnan(result["edge_pressure"]))

    def test_pressure_non_negative(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        result = analyzer.compute_pressure(H)
        assert np.all(result["variable_pressure"] >= 0.0)
        assert np.all(result["edge_pressure"] >= 0.0)
        assert result["max_pressure"] >= 0.0
        assert result["mean_pressure"] >= 0.0

    def test_max_pressure_consistent(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _medium_H()
        result = analyzer.compute_pressure(H)
        computed_max = float(result["variable_pressure"].max())
        assert abs(result["max_pressure"] - computed_max) < 1e-10

    def test_mean_pressure_consistent(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _medium_H()
        result = analyzer.compute_pressure(H)
        computed_mean = float(result["variable_pressure"].mean())
        assert abs(result["mean_pressure"] - computed_mean) < 1e-10

    def test_stable_ordering(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _medium_H()
        r1 = analyzer.compute_pressure(H)
        r2 = analyzer.compute_pressure(H)
        # Same ranking order
        order1 = np.argsort(-r1["variable_pressure"])
        order2 = np.argsort(-r2["variable_pressure"])
        np.testing.assert_array_equal(order1, order2)

    def test_empty_matrix(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = np.zeros((0, 5), dtype=np.float64)
        result = analyzer.compute_pressure(H)
        assert result["variable_pressure"].shape == (5,)
        assert result["max_pressure"] == 0.0
        assert result["mean_pressure"] == 0.0

    def test_zero_matrix(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = np.zeros((3, 6), dtype=np.float64)
        result = analyzer.compute_pressure(H)
        assert result["max_pressure"] == 0.0
        assert result["edge_pressure"].shape == (0,)

    def test_no_input_mutation(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _small_H()
        H_copy = H.copy()
        analyzer.compute_pressure(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_normalized_variable_pressure(self):
        analyzer = SpectralLocalizationAnalyzer()
        H = _medium_H()
        result = analyzer.compute_pressure(H)
        # Variable pressure should be normalized to [0, 1]
        assert result["variable_pressure"].max() <= 1.0 + 1e-12
