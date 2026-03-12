"""
Tests for the v12.2.0 EigenvectorLocalizationAnalyzer (IPR).

Verifies:
  - deterministic output
  - distributed vector classification
  - strong localization classification
  - mild localization classification
  - edge cases (near-zero vectors, precision)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.eigenvector_localization import (
    EigenvectorLocalizationAnalyzer,
)


class TestDeterminism:
    """IPR computation must be deterministic across repeated runs."""

    def test_repeated_runs_identical(self):
        v = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        r1 = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        r2 = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert r1["ipr"] == r2["ipr"]
        assert r1["localization_type"] == r2["localization_type"]
        assert r1["vector_norm"] == r2["vector_norm"]


class TestDistributedVector:
    """Uniform vector should yield IPR ~ 1/n (distributed)."""

    def test_uniform_vector(self):
        n = 100
        v = np.ones(n) / np.sqrt(n)
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        expected_ipr = 1.0 / n
        assert abs(result["ipr"] - expected_ipr) < 1e-10
        assert result["localization_type"] == "distributed"

    def test_large_uniform_vector(self):
        n = 1000
        v = np.ones(n) / np.sqrt(n)
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert result["localization_type"] == "distributed"
        assert result["ipr"] < 0.05


class TestStrongLocalization:
    """Single-site vector should yield IPR ~ 1 (strong localization)."""

    def test_single_nonzero(self):
        v = np.zeros(10)
        v[0] = 1.0
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert result["ipr"] == 1.0
        assert result["localization_type"] == "strong_localization"

    def test_two_site_localization(self):
        v = np.zeros(20)
        v[0] = 1.0
        v[1] = 1.0
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert abs(result["ipr"] - 0.5) < 1e-10
        assert result["localization_type"] == "strong_localization"


class TestMildLocalization:
    """Partially concentrated vector should yield mild localization."""

    def test_mild_concentration(self):
        # Build a vector where several entries dominate moderately.
        n = 40
        v = np.ones(n, dtype=np.float64) * 0.05
        v[:10] = [0.3, 0.28, 0.25, 0.22, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08]
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert 0.05 <= result["ipr"] < 0.15
        assert result["localization_type"] == "mild_localization"


class TestEdgeCases:
    """Edge cases: near-zero vectors, single-element vectors."""

    def test_zero_vector(self):
        v = np.zeros(10)
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert result["ipr"] == 0.0
        assert result["vector_norm"] == 0.0

    def test_very_small_vector(self):
        v = np.array([1e-20, 1e-20, 1e-20])
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        # Should not raise or produce NaN.
        assert np.isfinite(result["ipr"])
        assert 0.0 <= result["ipr"] <= 1.0

    def test_single_element(self):
        v = np.array([3.14])
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert result["ipr"] == 1.0
        assert result["localization_type"] == "strong_localization"

    def test_ipr_clamped_to_unit(self):
        """IPR must always be in [0, 1]."""
        for _ in range(5):
            v = np.random.RandomState(42).randn(20)
            result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
            assert 0.0 <= result["ipr"] <= 1.0


class TestVectorNorm:
    """vector_norm must match np.linalg.norm."""

    def test_norm_matches(self):
        v = np.array([3.0, 4.0])
        result = EigenvectorLocalizationAnalyzer.compute_ipr(v)
        assert result["vector_norm"] == round(5.0, 12)
