"""
Tests for v11.2.0 — Bethe Hessian Stability Analyzer.

Verifies:
  - Deterministic eigenvalue computation
  - Stability score sign interpretation
  - Behavior on simple graphs
  - Empty and edge-case matrices
  - No input mutation
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer


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


def _identity_H():
    """Create a simple identity-like (3, 3) matrix (no shared checks)."""
    H = np.eye(3, dtype=np.float64)
    return H


# -----------------------------------------------------------
# Determinism tests
# -----------------------------------------------------------


class TestBetheHessianDeterminism:
    """Verify deterministic eigenvalue computation."""

    def test_deterministic_small(self):
        H = _small_H()
        analyzer = BetheHessianAnalyzer()
        r1 = analyzer.compute_bethe_hessian_stability(H)
        r2 = analyzer.compute_bethe_hessian_stability(H)
        assert r1["bethe_hessian_min_eigenvalue"] == r2["bethe_hessian_min_eigenvalue"]
        assert r1["bethe_hessian_stability_score"] == r2["bethe_hessian_stability_score"]

    def test_deterministic_medium(self):
        H = _medium_H()
        analyzer = BetheHessianAnalyzer()
        r1 = analyzer.compute_bethe_hessian_stability(H)
        r2 = analyzer.compute_bethe_hessian_stability(H)
        assert r1["bethe_hessian_min_eigenvalue"] == r2["bethe_hessian_min_eigenvalue"]
        assert r1["bethe_hessian_stability_score"] == r2["bethe_hessian_stability_score"]

    def test_deterministic_across_instances(self):
        H = _small_H()
        a1 = BetheHessianAnalyzer()
        a2 = BetheHessianAnalyzer()
        r1 = a1.compute_bethe_hessian_stability(H)
        r2 = a2.compute_bethe_hessian_stability(H)
        assert r1 == r2


# -----------------------------------------------------------
# Output format tests
# -----------------------------------------------------------


class TestBetheHessianOutput:
    """Verify output dictionary structure and types."""

    def test_output_keys(self):
        H = _small_H()
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert "bethe_hessian_min_eigenvalue" in result
        assert "bethe_hessian_stability_score" in result

    def test_output_types(self):
        H = _small_H()
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert isinstance(result["bethe_hessian_min_eigenvalue"], float)
        assert isinstance(result["bethe_hessian_stability_score"], float)


# -----------------------------------------------------------
# Stability score interpretation tests
# -----------------------------------------------------------


class TestBetheHessianInterpretation:
    """Verify stability score sign interpretation on known graphs."""

    def test_identity_matrix_stability(self):
        """Identity matrix has no variable-variable adjacency -> stable."""
        H = _identity_H()
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        # With no shared checks, the adjacency is zero and H_B is diagonal
        # with positive entries, so min eigenvalue should be non-negative
        assert result["bethe_hessian_min_eigenvalue"] >= 0.0

    def test_dense_graph_computable(self):
        """A dense matrix with many shared checks is computable."""
        H = np.ones((3, 6), dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert isinstance(result["bethe_hessian_min_eigenvalue"], float)

    def test_small_graph_produces_finite_values(self):
        H = _small_H()
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert np.isfinite(result["bethe_hessian_min_eigenvalue"])
        assert np.isfinite(result["bethe_hessian_stability_score"])


# -----------------------------------------------------------
# Edge case tests
# -----------------------------------------------------------


class TestBetheHessianEdgeCases:
    """Verify behavior on edge-case matrices."""

    def test_empty_rows(self):
        H = np.zeros((0, 5), dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert result["bethe_hessian_min_eigenvalue"] == 0.0
        assert result["bethe_hessian_stability_score"] == 0.0

    def test_empty_cols(self):
        H = np.zeros((3, 0), dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert result["bethe_hessian_min_eigenvalue"] == 0.0
        assert result["bethe_hessian_stability_score"] == 0.0

    def test_zero_matrix(self):
        H = np.zeros((3, 6), dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert result["bethe_hessian_min_eigenvalue"] == 0.0
        assert result["bethe_hessian_stability_score"] == 0.0

    def test_single_edge(self):
        H = np.zeros((2, 3), dtype=np.float64)
        H[0, 0] = 1.0
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert isinstance(result["bethe_hessian_min_eigenvalue"], float)

    def test_no_input_mutation(self):
        H = _small_H()
        H_copy = H.copy()
        analyzer = BetheHessianAnalyzer()
        analyzer.compute_bethe_hessian_stability(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_two_variable_graph(self):
        """Minimal 2-variable graph."""
        H = np.array([[1, 1]], dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        result = analyzer.compute_bethe_hessian_stability(H)
        assert isinstance(result["bethe_hessian_min_eigenvalue"], float)
        assert np.isfinite(result["bethe_hessian_min_eigenvalue"])
