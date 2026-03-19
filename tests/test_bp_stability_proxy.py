"""
Tests for BP stability proxy diagnostics (v6.0).

Verifies:
  - Stability score is well-defined and reproducible
  - Score reflects expected behavior (positive for stable, negative for unstable)
  - JSON serialization safety
  - Determinism
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.diagnostics.bp_stability_proxy import estimate_bp_stability


class TestBPStabilityProxy:
    def test_positive_stability(self):
        """Positive min eigenvalue and nonzero radius → positive score."""
        nb_result = {"spectral_radius": 2.0}
        bethe_result = {"min_eigenvalue": 1.0, "num_negative": 0}

        result = estimate_bp_stability(nb_result, bethe_result)

        assert result["bp_stability_score"] == 0.5  # 1/2 * 1.0
        assert result["spectral_radius"] == 2.0
        assert result["bethe_min_eigenvalue"] == 1.0
        assert result["num_negative_bethe"] == 0

    def test_negative_stability(self):
        """Negative min eigenvalue → negative score."""
        nb_result = {"spectral_radius": 2.0}
        bethe_result = {"min_eigenvalue": -1.0, "num_negative": 1}

        result = estimate_bp_stability(nb_result, bethe_result)

        assert result["bp_stability_score"] == -0.5
        assert result["num_negative_bethe"] == 1

    def test_zero_spectral_radius(self):
        """Zero spectral radius → zero stability score."""
        nb_result = {"spectral_radius": 0.0}
        bethe_result = {"min_eigenvalue": 1.0, "num_negative": 0}

        result = estimate_bp_stability(nb_result, bethe_result)

        assert result["bp_stability_score"] == 0.0

    def test_determinism(self):
        """Repeated calls produce identical results."""
        nb_result = {"spectral_radius": 3.0}
        bethe_result = {"min_eigenvalue": -0.5, "num_negative": 2}

        r1 = estimate_bp_stability(nb_result, bethe_result)
        r2 = estimate_bp_stability(nb_result, bethe_result)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        nb_result = {"spectral_radius": 2.5}
        bethe_result = {"min_eigenvalue": 0.1, "num_negative": 0}

        result = estimate_bp_stability(nb_result, bethe_result)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_all_fields_present(self):
        """All required output fields are present."""
        nb_result = {"spectral_radius": 1.0}
        bethe_result = {"min_eigenvalue": 0.0, "num_negative": 0}

        result = estimate_bp_stability(nb_result, bethe_result)

        assert "bp_stability_score" in result
        assert "spectral_radius" in result
        assert "bethe_min_eigenvalue" in result
        assert "num_negative_bethe" in result

    def test_formula_correctness(self):
        """Score = (1/spectral_radius) * min_eigenvalue."""
        nb_result = {"spectral_radius": 4.0}
        bethe_result = {"min_eigenvalue": 2.0, "num_negative": 0}

        result = estimate_bp_stability(nb_result, bethe_result)
        expected = (1.0 / 4.0) * 2.0
        assert abs(result["bp_stability_score"] - expected) < 1e-10
