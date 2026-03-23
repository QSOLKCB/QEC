"""Tests for multi-scale field metrics."""

import numpy as np
import pytest

from qec.analysis.multiscale_metrics import (
    compute_multiscale_metrics,
    compute_multiscale_summary,
    compute_scale_consistency,
    compute_scale_divergence,
    downsample,
)


# ---------------------------------------------------------------------------
# downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_basic_factor_2(self):
        result = downsample([1.0, 3.0, 5.0, 7.0], 2)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_basic_factor_4(self):
        result = downsample([1.0, 2.0, 3.0, 4.0], 4)
        np.testing.assert_array_almost_equal(result, [2.5])

    def test_remainder_ignored(self):
        result = downsample([1.0, 3.0, 5.0, 7.0, 9.0], 2)
        np.testing.assert_array_almost_equal(result, [2.0, 6.0])

    def test_empty_input(self):
        result = downsample([], 2)
        assert len(result) == 0

    def test_input_shorter_than_factor(self):
        result = downsample([1.0], 2)
        assert len(result) == 0

    def test_factor_1_identity(self):
        vals = [1.0, 2.0, 3.0]
        result = downsample(vals, 1)
        np.testing.assert_array_almost_equal(result, vals)

    def test_invalid_factor_raises(self):
        with pytest.raises(ValueError):
            downsample([1.0], 0)

    def test_deterministic(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        r1 = downsample(vals, 2)
        r2 = downsample(vals, 2)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# compute_multiscale_metrics
# ---------------------------------------------------------------------------


class TestMultiscaleMetrics:
    def test_fine_always_present(self):
        result = compute_multiscale_metrics([1.0, 2.0, 3.0])
        assert result["fine"] is not None

    def test_scale_2_present_for_4_values(self):
        result = compute_multiscale_metrics([1.0, 2.0, 3.0, 4.0])
        assert result["scale_2"] is not None

    def test_scale_4_present_for_8_values(self):
        result = compute_multiscale_metrics(list(range(8)))
        assert result["scale_4"] is not None

    def test_scale_4_none_for_short_input(self):
        result = compute_multiscale_metrics([1.0, 2.0, 3.0, 4.0])
        assert result["scale_4"] is None

    def test_scale_2_none_for_very_short_input(self):
        result = compute_multiscale_metrics([1.0, 2.0])
        assert result["scale_2"] is None

    def test_no_input_mutation(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        original = vals.copy()
        compute_multiscale_metrics(vals)
        np.testing.assert_array_equal(vals, original)


# ---------------------------------------------------------------------------
# compute_scale_consistency
# ---------------------------------------------------------------------------


class TestScaleConsistency:
    def test_uniform_values_high_consistency(self):
        vals = [5.0] * 16
        ms = compute_multiscale_metrics(vals)
        c = compute_scale_consistency(ms)
        assert c > 0.9

    def test_single_scale_returns_one(self):
        ms = {"fine": {"phi_alignment": 0.5, "symmetry_score": 0.5,
                       "triality_balance": 0.5, "nonlinear_response": 0.5,
                       "resonance": 0.5, "complexity": 0.5},
              "scale_2": None, "scale_4": None}
        assert compute_scale_consistency(ms) == 1.0

    def test_consistency_bounded_0_1(self):
        vals = list(range(16))
        ms = compute_multiscale_metrics(vals)
        c = compute_scale_consistency(ms)
        assert 0.0 < c <= 1.0

    def test_deterministic(self):
        vals = [float(i) for i in range(16)]
        ms = compute_multiscale_metrics(vals)
        c1 = compute_scale_consistency(ms)
        c2 = compute_scale_consistency(ms)
        assert c1 == c2


# ---------------------------------------------------------------------------
# compute_scale_divergence
# ---------------------------------------------------------------------------


class TestScaleDivergence:
    def test_uniform_values_low_divergence(self):
        vals = [3.0] * 16
        ms = compute_multiscale_metrics(vals)
        d = compute_scale_divergence(ms)
        assert d < 0.1

    def test_divergence_bounded_0_1(self):
        vals = list(range(16))
        ms = compute_multiscale_metrics(vals)
        d = compute_scale_divergence(ms)
        assert 0.0 <= d < 1.0

    def test_single_scale_returns_zero(self):
        ms = {"fine": {"phi_alignment": 0.5}, "scale_2": None, "scale_4": None}
        assert compute_scale_divergence(ms) == 0.0

    def test_no_fine_returns_zero(self):
        assert compute_scale_divergence({"fine": None}) == 0.0

    def test_deterministic(self):
        vals = [float(i) for i in range(16)]
        ms = compute_multiscale_metrics(vals)
        d1 = compute_scale_divergence(ms)
        d2 = compute_scale_divergence(ms)
        assert d1 == d2


# ---------------------------------------------------------------------------
# compute_multiscale_summary
# ---------------------------------------------------------------------------


class TestMultiscaleSummary:
    def test_keys_present(self):
        result = compute_multiscale_summary(list(range(16)))
        assert "multiscale" in result
        assert "scale_consistency" in result
        assert "scale_divergence" in result

    def test_consistency_is_float(self):
        result = compute_multiscale_summary(list(range(8)))
        assert isinstance(result["scale_consistency"], float)

    def test_divergence_is_float(self):
        result = compute_multiscale_summary(list(range(8)))
        assert isinstance(result["scale_divergence"], float)

    def test_deterministic(self):
        vals = [float(x) for x in range(16)]
        r1 = compute_multiscale_summary(vals)
        r2 = compute_multiscale_summary(vals)
        assert r1["scale_consistency"] == r2["scale_consistency"]
        assert r1["scale_divergence"] == r2["scale_divergence"]

    def test_no_input_mutation(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        original = vals.copy()
        compute_multiscale_summary(vals)
        np.testing.assert_array_equal(vals, original)
