"""Tests for adaptive trend-aware damping controller."""

from qec.analysis.adaptive_control import (
    compute_adaptive_damping,
    compute_response_mode,
    run_adaptive_control,
)


def _make_memory(trend_state, smoothed=0.5, delta=0.0):
    return {
        "smoothed_signal": smoothed,
        "trend_delta": delta,
        "trend_state": trend_state,
    }


class TestComputeAdaptiveDamping:
    def test_stable(self):
        assert compute_adaptive_damping(0.8, "stable") == 0.8

    def test_rising(self):
        assert compute_adaptive_damping(0.8, "rising") == 0.7

    def test_falling(self):
        assert compute_adaptive_damping(0.8, "falling") == 0.9

    def test_lower_clamp(self):
        assert compute_adaptive_damping(0.4, "rising") == 0.4

    def test_upper_clamp(self):
        assert compute_adaptive_damping(1.0, "falling") == 1.0


class TestComputeResponseMode:
    def test_rising(self):
        assert compute_response_mode("rising") == "stabilize"

    def test_falling(self):
        assert compute_response_mode("falling") == "recover"

    def test_stable(self):
        assert compute_response_mode("stable") == "hold"


class TestRunAdaptiveControl:
    def test_stable(self):
        result = run_adaptive_control(0.8, _make_memory("stable"))
        assert result["adaptive_damping"] == 0.8
        assert result["response_mode"] == "hold"
        assert result["control_gain"] == 1.0

    def test_rising(self):
        result = run_adaptive_control(0.8, _make_memory("rising"))
        assert result["adaptive_damping"] == 0.7
        assert result["response_mode"] == "stabilize"
        assert result["control_gain"] < 1.0

    def test_falling(self):
        result = run_adaptive_control(0.8, _make_memory("falling"))
        assert result["adaptive_damping"] == 0.9
        assert result["response_mode"] == "recover"
        assert result["control_gain"] > 1.0

    def test_lower_clamp(self):
        result = run_adaptive_control(0.4, _make_memory("rising"))
        assert result["adaptive_damping"] == 0.4

    def test_upper_clamp(self):
        result = run_adaptive_control(1.0, _make_memory("falling"))
        assert result["adaptive_damping"] == 1.0

    def test_determinism(self):
        mem = _make_memory("rising", smoothed=0.6, delta=0.05)
        results = [run_adaptive_control(0.8, mem) for _ in range(50)]
        assert all(r == results[0] for r in results)
