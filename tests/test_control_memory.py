"""Tests for temporal smoothing and trend memory layer (v105.5.1)."""

from qec.analysis.control_memory import (
    TREND_FALL_THRESHOLD,
    TREND_RISE_THRESHOLD,
    classify_control_trend,
    compute_ema,
    compute_trend_delta,
    run_control_memory,
)


class TestComputeEma:
    def test_equal_weight(self):
        result = compute_ema(0.4, 0.8, alpha=0.5)
        assert abs(result - 0.6) < 1e-15

    def test_alpha_zero_returns_previous(self):
        assert compute_ema(0.7, 0.3, alpha=0.0) == 0.7

    def test_alpha_one_returns_current(self):
        assert compute_ema(0.7, 0.3, alpha=1.0) == 0.3

    def test_alpha_clamp_negative(self):
        # alpha < 0 clamps to 0 -> returns previous
        assert compute_ema(0.5, 0.9, alpha=-0.5) == 0.5

    def test_alpha_clamp_above_one(self):
        # alpha > 1 clamps to 1 -> returns current
        assert compute_ema(0.5, 0.9, alpha=2.0) == 0.9


class TestComputeTrendDelta:
    def test_no_change(self):
        assert compute_trend_delta(0.5, 0.5) == 0.0

    def test_positive_delta(self):
        assert compute_trend_delta(0.4, 0.8) == 0.4

    def test_negative_delta(self):
        assert compute_trend_delta(0.8, 0.4) == -0.4


class TestClassifyControlTrend:
    def test_rising(self):
        assert classify_control_trend(0.2) == "rising"

    def test_falling(self):
        assert classify_control_trend(-0.2) == "falling"

    def test_stable_zero(self):
        assert classify_control_trend(0.0) == "stable"

    def test_stable_boundary_positive(self):
        assert classify_control_trend(0.1) == "stable"

    def test_stable_boundary_negative(self):
        assert classify_control_trend(-0.1) == "stable"


class TestRunControlMemory:
    def test_stable(self):
        result = run_control_memory(0.5, 0.5)
        assert result["trend_delta"] == 0.0
        assert result["trend_state"] == "stable"
        assert result["smoothed_signal"] == 0.5

    def test_rising(self):
        result = run_control_memory(0.4, 0.8)
        assert result["trend_state"] == "rising"
        assert result["trend_delta"] == 0.4
        assert result["smoothed_signal"] == 0.6

    def test_falling(self):
        result = run_control_memory(0.8, 0.4)
        assert result["trend_state"] == "falling"
        assert result["trend_delta"] == -0.4
        assert result["smoothed_signal"] == 0.6

    def test_rounding_at_api_boundary(self):
        """Rounding is applied at API level, not in helpers."""
        result = run_control_memory(0.35, 0.75)
        assert round(result["smoothed_signal"], 12) == result["smoothed_signal"]
        assert round(result["trend_delta"], 12) == result["trend_delta"]

    def test_determinism(self):
        """Repeated runs must produce identical results."""
        for _ in range(10):
            r = run_control_memory(0.35, 0.75)
            assert r == {
                "smoothed_signal": 0.55,
                "trend_delta": 0.4,
                "trend_state": "rising",
            }


class TestHelpersReturnRawFloats:
    def test_ema_returns_raw_float(self):
        """compute_ema must return a raw float, not pre-rounded."""
        result = compute_ema(0.1, 0.2, alpha=0.3)
        assert isinstance(result, float)
        # Result is raw: 0.3*0.2 + 0.7*0.1 = 0.13
        assert result == 0.3 * 0.2 + 0.7 * 0.1

    def test_trend_delta_returns_raw_float(self):
        """compute_trend_delta must return a raw float, not pre-rounded."""
        result = compute_trend_delta(0.1, 0.2)
        assert isinstance(result, float)
        assert result == 0.2 - 0.1


class TestThresholdConstants:
    def test_constants_values(self):
        assert TREND_RISE_THRESHOLD == 0.1
        assert TREND_FALL_THRESHOLD == -0.1

    def test_exactly_at_rise_threshold_is_stable(self):
        assert classify_control_trend(0.1) == "stable"

    def test_above_rise_threshold_is_rising(self):
        assert classify_control_trend(0.1 + 1e-9) == "rising"

    def test_exactly_at_fall_threshold_is_stable(self):
        assert classify_control_trend(-0.1) == "stable"

    def test_below_fall_threshold_is_falling(self):
        assert classify_control_trend(-0.1 - 1e-9) == "falling"
