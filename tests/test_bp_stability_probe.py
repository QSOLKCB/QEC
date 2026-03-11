"""
Tests for v11.0.0 — BP Stability Probe and BP Instability Estimate.

Covers:
  - deterministic behavior
  - valid metric ranges
  - trials=0 edge case
  - no input mutation
  - Jacobian spectral radius estimation
"""

import numpy as np
import pytest

from src.qec.decoder.stability_probe import BPStabilityProbe, estimate_bp_instability


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


class TestBPStabilityProbe:
    def test_result_structure(self):
        H = _make_small_ldpc()
        probe = BPStabilityProbe(trials=10, iterations=5, seed=42)
        result = probe.probe(H)

        assert "bp_stability_score" in result
        assert "divergence_rate" in result
        assert "stagnation_rate" in result
        assert "oscillation_score" in result
        assert "average_iterations" in result

    def test_deterministic(self):
        H = _make_small_ldpc()
        probe = BPStabilityProbe(trials=10, iterations=5, seed=42)
        r1 = probe.probe(H)
        r2 = probe.probe(H)

        assert r1 == r2

    def test_valid_ranges(self):
        H = _make_small_ldpc()
        probe = BPStabilityProbe(trials=20, iterations=5, seed=42)
        result = probe.probe(H)

        assert 0.0 <= result["bp_stability_score"] <= 1.0
        assert 0.0 <= result["divergence_rate"] <= 1.0
        assert 0.0 <= result["stagnation_rate"] <= 1.0
        assert result["oscillation_score"] >= 0.0
        assert result["average_iterations"] >= 0.0

    def test_trials_zero(self):
        H = _make_small_ldpc()
        probe = BPStabilityProbe(trials=0, iterations=5, seed=42)
        result = probe.probe(H)

        assert result["bp_stability_score"] == 1.0
        assert result["divergence_rate"] == 0.0
        assert result["stagnation_rate"] == 0.0
        assert result["oscillation_score"] == 0.0
        assert result["average_iterations"] == 0.0

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        probe = BPStabilityProbe(trials=10, iterations=5, seed=42)
        probe.probe(H)

        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        probe = BPStabilityProbe(trials=5, iterations=5, seed=42)
        result = probe.probe(H)

        assert result["bp_stability_score"] == 1.0

    def test_different_seeds_may_differ(self):
        H = _make_small_ldpc()
        probe1 = BPStabilityProbe(trials=20, iterations=5, seed=1)
        probe2 = BPStabilityProbe(trials=20, iterations=5, seed=999)
        r1 = probe1.probe(H)
        r2 = probe2.probe(H)

        # Different seeds may produce different results
        # (but both should be valid)
        assert 0.0 <= r1["bp_stability_score"] <= 1.0
        assert 0.0 <= r2["bp_stability_score"] <= 1.0


class TestEstimateBPInstability:
    def test_result_structure(self):
        H = _make_small_ldpc()
        result = estimate_bp_instability(H, seed=42)

        assert "jacobian_spectral_radius_est" in result
        assert isinstance(result["jacobian_spectral_radius_est"], float)

    def test_deterministic(self):
        H = _make_small_ldpc()
        r1 = estimate_bp_instability(H, seed=42)
        r2 = estimate_bp_instability(H, seed=42)

        assert r1 == r2

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        estimate_bp_instability(H, seed=42)

        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        result = estimate_bp_instability(H, seed=42)

        assert result["jacobian_spectral_radius_est"] == 0.0

    def test_nonnegative(self):
        H = _make_small_ldpc()
        result = estimate_bp_instability(H, seed=42)

        assert result["jacobian_spectral_radius_est"] >= 0.0
