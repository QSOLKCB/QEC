"""
Tests for Bethe Hessian spectral diagnostics (v6.0).

Verifies:
  - Deterministic spectral results on small synthetic graphs
  - Eigenvalue ordering (ascending)
  - JSON serialization safety
  - Custom r parameter
  - Automatic r computation
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.diagnostics.bethe_hessian import compute_bethe_hessian


class TestBetheHessian:
    def test_small_repetition_code(self):
        """Repetition code H = [[1,1,0],[0,1,1]] produces valid spectrum."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_bethe_hessian(H)

        assert "bethe_eigenvalues" in result
        assert "min_eigenvalue" in result
        assert "num_negative" in result
        assert "r_used" in result

        assert isinstance(result["min_eigenvalue"], float)
        assert isinstance(result["num_negative"], int)
        assert result["num_negative"] >= 0
        assert result["r_used"] > 0.0

    def test_eigenvalue_ordering(self):
        """Eigenvalues are sorted ascending."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_bethe_hessian(H)

        evs = result["bethe_eigenvalues"]
        for i in range(len(evs) - 1):
            assert evs[i] <= evs[i + 1] + 1e-10

    def test_custom_r(self):
        """Custom r parameter is used correctly."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_bethe_hessian(H, r=2.0)

        assert result["r_used"] == 2.0

    def test_determinism(self):
        """Repeated calls produce identical results."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        r1 = compute_bethe_hessian(H)
        r2 = compute_bethe_hessian(H)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_bethe_hessian(H)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_num_eigenvalues_matches_matrix_size(self):
        """Number of eigenvalues equals total graph nodes."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        m, n = H.shape
        result = compute_bethe_hessian(H)
        assert len(result["bethe_eigenvalues"]) == n + m

    def test_min_eigenvalue_consistency(self):
        """min_eigenvalue matches first element of sorted eigenvalues."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.float64)
        result = compute_bethe_hessian(H)
        assert abs(result["min_eigenvalue"] - result["bethe_eigenvalues"][0]) < 1e-10

    def test_num_negative_consistency(self):
        """num_negative matches count of negative eigenvalues."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_bethe_hessian(H)
        expected = sum(1 for v in result["bethe_eigenvalues"] if v < 0.0)
        assert result["num_negative"] == expected

    def test_no_input_mutation(self):
        """Input matrix is not modified."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        H_copy = H.copy()
        compute_bethe_hessian(H)
        np.testing.assert_array_equal(H, H_copy)

from src.qec.analysis.bethe_hessian import BetheHessianAnalyzer, estimate_nishimori_temperature


class TestBetheHessianV132Analysis:
    def test_smallest_eigenvalue_deterministic(self):
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        r = 1.2
        v1 = analyzer.smallest_eigenvalue(H, r=r)
        v2 = analyzer.smallest_eigenvalue(H, r=r)
        assert v1 == v2

    def test_sparse_support_matches_dense(self):
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        analyzer = BetheHessianAnalyzer()
        A = analyzer._build_variable_adjacency(H)
        dense = A.toarray()
        assert np.array_equal((dense > 0).astype(int), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int))

    def test_nishimori_temperature_deterministic(self):
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        r1 = estimate_nishimori_temperature(H)
        r2 = estimate_nishimori_temperature(H)
        assert r1 == r2
        assert r1 > 0.0
