"""Tests for structural diversity regulator (v9.3.0)."""

from __future__ import annotations

import numpy as np
import pytest

from qec.discovery.diversity import (
    compute_structure_signature,
    compute_diversity_penalty,
)


@pytest.fixture
def sample_H():
    """Small parity-check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


@pytest.fixture
def different_H():
    """A different parity-check matrix with distinct degree distribution."""
    return np.array([
        [1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1],
    ], dtype=np.float64)


class TestComputeStructureSignature:
    def test_returns_tuple(self, sample_H):
        sig = compute_structure_signature(sample_H)
        assert isinstance(sig, tuple)

    def test_has_four_components(self, sample_H):
        sig = compute_structure_signature(sample_H)
        assert len(sig) == 4

    def test_all_floats(self, sample_H):
        sig = compute_structure_signature(sample_H)
        for v in sig:
            assert isinstance(v, float)

    def test_deterministic(self, sample_H):
        sig1 = compute_structure_signature(sample_H)
        sig2 = compute_structure_signature(sample_H)
        assert sig1 == sig2

    def test_different_matrices_different_signatures(self, sample_H, different_H):
        sig1 = compute_structure_signature(sample_H)
        sig2 = compute_structure_signature(different_H)
        assert sig1 != sig2

    def test_mean_variable_degree(self, sample_H):
        sig = compute_structure_signature(sample_H)
        expected_mean = float(np.mean(np.sum(sample_H, axis=0)))
        assert sig[0] == pytest.approx(expected_mean)

    def test_mean_check_degree(self, sample_H):
        sig = compute_structure_signature(sample_H)
        expected_mean = float(np.mean(np.sum(sample_H, axis=1)))
        assert sig[2] == pytest.approx(expected_mean)


class TestComputeDiversityPenalty:
    def test_empty_archive_returns_zero(self, sample_H):
        sig = compute_structure_signature(sample_H)
        penalty = compute_diversity_penalty(sig, [])
        assert penalty == 0.0

    def test_identical_signature_high_penalty(self, sample_H):
        sig = compute_structure_signature(sample_H)
        penalty = compute_diversity_penalty(sig, [sig])
        assert penalty == pytest.approx(1.0)

    def test_different_signature_lower_penalty(self, sample_H, different_H):
        sig1 = compute_structure_signature(sample_H)
        sig2 = compute_structure_signature(different_H)
        penalty = compute_diversity_penalty(sig1, [sig2])
        assert 0.0 < penalty < 1.0

    def test_penalty_bounded(self, sample_H, different_H):
        sig1 = compute_structure_signature(sample_H)
        sig2 = compute_structure_signature(different_H)
        penalty = compute_diversity_penalty(sig1, [sig2])
        assert 0.0 <= penalty < 1.0

    def test_penalty_deterministic(self, sample_H, different_H):
        sig1 = compute_structure_signature(sample_H)
        sig2 = compute_structure_signature(different_H)
        p1 = compute_diversity_penalty(sig1, [sig2])
        p2 = compute_diversity_penalty(sig1, [sig2])
        assert p1 == p2

    def test_more_similar_higher_penalty(self):
        sig = (3.0, 0.5, 3.0, 0.5)
        close = (3.1, 0.5, 3.0, 0.5)
        far = (5.0, 2.0, 5.0, 2.0)
        penalty_close = compute_diversity_penalty(sig, [close])
        penalty_far = compute_diversity_penalty(sig, [far])
        assert penalty_close > penalty_far

    def test_multiple_archive_entries(self, sample_H, different_H):
        sig = compute_structure_signature(sample_H)
        archive = [
            compute_structure_signature(different_H),
            compute_structure_signature(sample_H),  # identical
        ]
        penalty = compute_diversity_penalty(sig, archive)
        # Should find the identical one and return high penalty
        assert penalty == pytest.approx(1.0)
