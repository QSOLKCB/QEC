"""
Tests for v11.0.0 — Trapping Set Detector.

Covers:
  - deterministic detection
  - no input mutation
  - well-formed result structure
  - identity-like matrix behavior
  - runtime safety bound for larger matrices
"""

import numpy as np
import pytest

from src.qec.analysis.trapping_sets import TrappingSetDetector


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix with known structure."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


class TestTrappingSetDetector:
    def test_result_structure(self):
        H = _make_small_ldpc()
        detector = TrappingSetDetector()
        result = detector.detect(H)

        assert "min_size" in result
        assert "counts" in result
        assert "total" in result
        assert "variable_participation" in result

        assert isinstance(result["min_size"], int)
        assert isinstance(result["counts"], dict)
        assert isinstance(result["total"], int)
        assert isinstance(result["variable_participation"], list)
        assert len(result["variable_participation"]) == H.shape[1]

    def test_deterministic(self):
        H = _make_small_ldpc()
        detector = TrappingSetDetector()
        r1 = detector.detect(H)
        r2 = detector.detect(H)

        assert r1["min_size"] == r2["min_size"]
        assert r1["counts"] == r2["counts"]
        assert r1["total"] == r2["total"]
        assert r1["variable_participation"] == r2["variable_participation"]

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        detector = TrappingSetDetector()
        detector.detect(H)

        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        detector = TrappingSetDetector()
        result = detector.detect(H)

        assert result["total"] == 0
        assert result["min_size"] == 0
        assert result["counts"] == {}
        assert result["variable_participation"] == []

    def test_identity_like_matrix(self):
        """Identity matrix has no shared check nodes between variables,
        so all variable subsets of size >= 2 have only degree-1 induced
        checks. This tests that the detector handles sparse connectivity."""
        H = np.eye(4, dtype=np.float64)
        detector = TrappingSetDetector()
        result = detector.detect(H)

        assert isinstance(result["total"], int)
        assert result["total"] >= 0
        assert len(result["variable_participation"]) == 4

    def test_counts_keys_are_tuples(self):
        H = _make_small_ldpc()
        detector = TrappingSetDetector()
        result = detector.detect(H)

        for key in result["counts"]:
            assert isinstance(key, tuple)
            assert len(key) == 2
            a, b = key
            assert isinstance(a, int)
            assert isinstance(b, int)
            assert 1 <= a <= 6
            assert 0 <= b <= 4

    def test_budget_limits_runtime(self):
        """With a small budget, detection stops early."""
        H = _make_small_ldpc()
        detector_small = TrappingSetDetector(budget=10)
        result_small = detector_small.detect(H)

        detector_large = TrappingSetDetector(budget=500_000)
        result_large = detector_large.detect(H)

        # Small budget may find fewer or equal trapping sets
        assert result_small["total"] <= result_large["total"]

    def test_participation_nonnegative(self):
        H = _make_small_ldpc()
        detector = TrappingSetDetector()
        result = detector.detect(H)

        for count in result["variable_participation"]:
            assert count >= 0

    def test_larger_matrix_does_not_crash(self):
        """Ensure a larger matrix with budget doesn't explode."""
        rng = np.random.RandomState(123)
        H = (rng.random((20, 40)) < 0.15).astype(np.float64)
        detector = TrappingSetDetector(max_a=4, max_b=3, budget=50_000)
        result = detector.detect(H)

        assert isinstance(result["total"], int)
        assert len(result["variable_participation"]) == 40
