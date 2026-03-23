"""Tests for the deterministic field metrics layer."""

import numpy as np
import pytest

from qec.analysis.field_metrics import (
    PHI,
    compute_complexity,
    compute_curvature,
    compute_field_metrics,
    compute_nonlinear_response,
    compute_phi_alignment,
    compute_resonance,
    compute_symmetry_score,
    compute_triality_balance,
)


# ---------------------------------------------------------------------------
# φ alignment
# ---------------------------------------------------------------------------


class TestPhiAlignment:
    def test_constant_phi_ratio_high_score(self):
        """Successive ratios equal to φ should give alignment = 1.0."""
        values = [1.0, PHI, PHI**2, PHI**3]
        score = compute_phi_alignment(values)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_single_value_returns_zero(self):
        assert compute_phi_alignment([5.0]) == 0.0

    def test_empty_returns_zero(self):
        assert compute_phi_alignment([]) == 0.0

    def test_zero_denominator_skipped(self):
        score = compute_phi_alignment([0.0, 3.0, 3.0 * PHI])
        # First ratio skipped (zero denom), second = PHI → perfect
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_all_zeros(self):
        assert compute_phi_alignment([0.0, 0.0, 0.0]) == 0.0

    def test_non_phi_ratio_lower(self):
        values = [1.0, 2.0, 4.0, 8.0]  # ratio = 2, not φ
        score = compute_phi_alignment(values)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# symmetry score
# ---------------------------------------------------------------------------


class TestSymmetryScore:
    def test_equal_values_high(self):
        score = compute_symmetry_score([5.0, 5.0, 5.0, 5.0])
        assert score == pytest.approx(1.0)

    def test_spread_values_lower(self):
        score = compute_symmetry_score([0.0, 100.0])
        assert score < 0.1

    def test_single_value(self):
        assert compute_symmetry_score([7.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# triality balance
# ---------------------------------------------------------------------------


class TestTrialityBalance:
    def test_balanced_partitions(self):
        # indices 0,3,6 → group0; 1,4,7 → group1; 2,5,8 → group2
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        score = compute_triality_balance(values)
        assert score == pytest.approx(1.0)

    def test_imbalanced_lower(self):
        values = [100.0, 0.0, 0.0, 100.0, 0.0, 0.0]
        score = compute_triality_balance(values)
        assert score < 0.1

    def test_empty(self):
        assert compute_triality_balance([]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# nonlinear response
# ---------------------------------------------------------------------------


class TestNonlinearResponse:
    def test_small_values_low(self):
        score = compute_nonlinear_response([0.01, 0.02, 0.01])
        assert score < 0.01

    def test_large_values_higher(self):
        score = compute_nonlinear_response([10.0, 20.0, 30.0])
        assert score > 0.5

    def test_zeros(self):
        assert compute_nonlinear_response([0.0, 0.0]) == pytest.approx(0.0)

    def test_bounded_below_one(self):
        score = compute_nonlinear_response([1e6, 1e6])
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------


class TestCurvature:
    def test_linear_zero_curvature(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_curvature(values)
        assert result["abs_curvature"] == pytest.approx(0.0, abs=1e-12)
        assert result["curvature_variation"] == pytest.approx(0.0, abs=1e-12)

    def test_oscillating_nonzero(self):
        values = [0.0, 1.0, 0.0, 1.0, 0.0]
        result = compute_curvature(values)
        assert result["abs_curvature"] > 0.0

    def test_too_short(self):
        result = compute_curvature([1.0, 2.0])
        assert result == {"abs_curvature": 0.0, "curvature_variation": 0.0}


# ---------------------------------------------------------------------------
# resonance
# ---------------------------------------------------------------------------


class TestResonance:
    def test_repeating_pattern_detected(self):
        values = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        score = compute_resonance(values)
        assert score > 0.5

    def test_constant_full_resonance(self):
        values = [3.0] * 10
        score = compute_resonance(values)
        assert score == pytest.approx(1.0)

    def test_no_pattern(self):
        values = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]
        score = compute_resonance(values)
        assert score < 0.5

    def test_short_sequence(self):
        assert compute_resonance([1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# complexity
# ---------------------------------------------------------------------------


class TestComplexity:
    def test_constant_low(self):
        assert compute_complexity([5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_varied_higher(self):
        score = compute_complexity([0.0, 10.0, 0.0, 10.0])
        assert score > 0.5

    def test_bounded(self):
        score = compute_complexity([1e6, -1e6])
        assert 0.0 <= score < 1.0


# ---------------------------------------------------------------------------
# integration — compute_field_metrics
# ---------------------------------------------------------------------------


class TestComputeFieldMetrics:
    def test_all_keys_present(self):
        result = compute_field_metrics([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_keys = {
            "phi_alignment",
            "symmetry_score",
            "triality_balance",
            "nonlinear_response",
            "curvature",
            "resonance",
            "complexity",
        }
        assert set(result.keys()) == expected_keys

    def test_curvature_is_dict(self):
        result = compute_field_metrics([1.0, 2.0, 3.0])
        assert isinstance(result["curvature"], dict)
        assert "abs_curvature" in result["curvature"]
        assert "curvature_variation" in result["curvature"]

    def test_deterministic(self):
        values = [1.0, 1.618, 2.618, 4.236]
        r1 = compute_field_metrics(values)
        r2 = compute_field_metrics(values)
        assert r1 == r2

    def test_no_mutation(self):
        original = [1.0, 2.0, 3.0, 4.0]
        copy = list(original)
        compute_field_metrics(original)
        assert original == copy

    def test_scalar_types(self):
        result = compute_field_metrics([1.0, 2.0, 3.0, 4.0, 5.0])
        for key in ["phi_alignment", "symmetry_score", "triality_balance",
                     "nonlinear_response", "resonance", "complexity"]:
            assert isinstance(result[key], float), f"{key} is not float"
