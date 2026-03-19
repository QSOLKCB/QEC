"""
Tests for non-backtracking spectrum diagnostics (v6.0).

Verifies:
  - Deterministic spectral results on small synthetic graphs
  - Deterministic ordering of eigenvalues
  - JSON serialization safety
  - Expected behavior on identity-like and trivial matrices
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

from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
)


class TestNonBacktrackingSpectrum:
    def test_small_repetition_code(self):
        """Repetition code H = [[1,1,0],[0,1,1]] produces valid spectrum."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)

        assert "nb_eigenvalues" in result
        assert "spectral_radius" in result
        assert "num_eigenvalues" in result

        assert result["num_eigenvalues"] > 0
        assert result["spectral_radius"] >= 0.0
        assert len(result["nb_eigenvalues"]) == result["num_eigenvalues"]

    def test_single_check_node(self):
        """Single check node H = [[1,1,1]] produces valid spectrum."""
        H = np.array([[1, 1, 1]], dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)

        assert result["num_eigenvalues"] > 0
        assert result["spectral_radius"] >= 0.0

    def test_empty_matrix(self):
        """All-zero matrix returns empty spectrum."""
        H = np.zeros((2, 3), dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)

        assert result["nb_eigenvalues"] == []
        assert result["spectral_radius"] == 0.0
        assert result["num_eigenvalues"] == 0

    def test_determinism(self):
        """Repeated calls produce identical results."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        r1 = compute_non_backtracking_spectrum(H)
        r2 = compute_non_backtracking_spectrum(H)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_eigenvalue_ordering(self):
        """Eigenvalues are sorted by magnitude descending."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)

        mags = [abs(complex(ev[0], ev[1])) for ev in result["nb_eigenvalues"]]
        for i in range(len(mags) - 1):
            assert mags[i] >= mags[i + 1] - 1e-10

    def test_spectral_radius_equals_max_magnitude(self):
        """Spectral radius equals the magnitude of the first eigenvalue."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_non_backtracking_spectrum(H)

        if result["nb_eigenvalues"]:
            ev = result["nb_eigenvalues"][0]
            expected_mag = abs(complex(ev[0], ev[1]))
            assert abs(result["spectral_radius"] - expected_mag) < 1e-10

    def test_no_input_mutation(self):
        """Input matrix is not modified."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        H_copy = H.copy()
        compute_non_backtracking_spectrum(H)
        np.testing.assert_array_equal(H, H_copy)
