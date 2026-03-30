"""Tests for temporal smoothing and trend memory layer (v105.5.0)."""

from qec.analysis.control_memory import (
    classify_control_trend,
    compute_ema,
    compute_trend_delta,
    run_control_memory,
)


class TestComputeEma:
    def test_equal_weight(self):
        assert compute_ema(0.4, 0.8, alpha=0.5) == 0.6

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

    def test_determinism(self):
        """Repeated runs must produce identical results."""
        for _ in range(10):
            r = run_control_memory(0.35, 0.75)
            assert r == {
                "smoothed_signal": 0.55,
                "trend_delta": 0.4,
                "trend_state": "rising",
            }
