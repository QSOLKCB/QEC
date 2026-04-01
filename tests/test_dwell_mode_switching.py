"""v132.3.0 — Deterministic tests for dwell-time guard, hysteresis filter,
and atomic mode switcher.

All tests are pure, deterministic, and replay-safe.
No system clock reads. No randomness.
"""

from __future__ import annotations

import pickle

import pytest

from qec.control.dwell_time_guard import can_switch_now
from qec.control.hysteresis_filter import passes_hysteresis
from qec.control.mode_switcher import (
    FailSafeState,
    ModeSwitchResult,
    evaluate_mode_switch,
    _detect_chatter,
)


# ===================================================================
# Dwell-time guard
# ===================================================================


class TestCanSwitchNow:
    """Deterministic dwell-time guard tests."""

    def test_switch_allowed_after_dwell(self) -> None:
        assert can_switch_now(0, 1000, 1000) is True

    def test_switch_allowed_well_past_dwell(self) -> None:
        assert can_switch_now(0, 5000, 1000) is True

    def test_switch_blocked_before_dwell(self) -> None:
        assert can_switch_now(0, 999, 1000) is False

    def test_switch_blocked_zero_elapsed(self) -> None:
        assert can_switch_now(100, 100, 1000) is False

    def test_zero_dwell_always_allows(self) -> None:
        assert can_switch_now(100, 100, 0) is True

    def test_exact_boundary(self) -> None:
        assert can_switch_now(500, 1500, 1000) is True

    def test_one_ms_short(self) -> None:
        assert can_switch_now(500, 1499, 1000) is False

    def test_wrap_safe_64bit(self) -> None:
        """Monotonic counter wrap at 2^64 boundary."""
        max_64 = (1 << 64) - 1
        # last switch near max, current wrapped to small value
        # elapsed = (10 - max_64) & mask = 11
        assert can_switch_now(max_64, 10, 5) is True
        assert can_switch_now(max_64, 10, 100) is False

    def test_negative_dwell_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            can_switch_now(0, 100, -1)

    def test_deterministic_replay(self) -> None:
        """Same inputs must produce identical outputs on repeated calls."""
        args = (1000, 2500, 1000)
        results = [can_switch_now(*args) for _ in range(100)]
        assert all(r is True for r in results)


# ===================================================================
# Hysteresis filter
# ===================================================================


class TestPassesHysteresis:
    """Deterministic hysteresis filter tests."""

    def test_enter_at_threshold(self) -> None:
        assert passes_hysteresis(10.0, 10.0, 5.0, "inactive") is True

    def test_no_enter_below_threshold(self) -> None:
        assert passes_hysteresis(9.9, 10.0, 5.0, "inactive") is False

    def test_stay_active_above_exit(self) -> None:
        assert passes_hysteresis(6.0, 10.0, 5.0, "active") is True

    def test_exit_at_threshold(self) -> None:
        """At exactly exit_threshold, mode should exit (not > exit)."""
        assert passes_hysteresis(5.0, 10.0, 5.0, "active") is False

    def test_exit_below_threshold(self) -> None:
        assert passes_hysteresis(4.0, 10.0, 5.0, "active") is False

    def test_hysteresis_band_prevents_chatter(self) -> None:
        """Metric in the dead band doesn't trigger entry or exit."""
        # In dead band (between 5 and 10), inactive stays inactive
        assert passes_hysteresis(7.0, 10.0, 5.0, "inactive") is False
        # In dead band, active stays active
        assert passes_hysteresis(7.0, 10.0, 5.0, "active") is True

    def test_integer_thresholds(self) -> None:
        assert passes_hysteresis(10, 10, 5, "inactive") is True
        assert passes_hysteresis(4, 10, 5, "active") is False

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="enter_threshold"):
            passes_hysteresis(5.0, 3.0, 8.0, "inactive")

    def test_equal_thresholds(self) -> None:
        """When enter == exit, no dead band — behaves as simple threshold."""
        assert passes_hysteresis(10.0, 10.0, 10.0, "inactive") is True
        assert passes_hysteresis(10.0, 10.0, 10.0, "active") is False

    def test_deterministic_replay(self) -> None:
        args = (7.5, 10.0, 5.0, "inactive")
        results = [passes_hysteresis(*args) for _ in range(100)]
        assert all(r is False for r in results)


# ===================================================================
# Chatter detection
# ===================================================================


class TestDetectChatter:

    def test_abab_is_chatter(self) -> None:
        assert _detect_chatter(("A", "B", "A", "B")) is True

    def test_abac_is_not_chatter(self) -> None:
        assert _detect_chatter(("A", "B", "A", "C")) is False

    def test_short_sequence_no_chatter(self) -> None:
        assert _detect_chatter(("A", "B", "A")) is False

    def test_longer_trailing_chatter(self) -> None:
        assert _detect_chatter(("X", "A", "B", "A", "B")) is True

    def test_same_mode_no_chatter(self) -> None:
        assert _detect_chatter(("A", "A", "A", "A")) is False


# ===================================================================
# Fail-safe state
# ===================================================================


class TestFailSafeState:

    def test_initial_state(self) -> None:
        fs = FailSafeState.initial()
        assert fs.blocked_count == 0
        assert fs.latched is False
        assert fs.recent_modes == ()

    def test_latch_after_max_blocked(self) -> None:
        fs = FailSafeState.initial()
        fs = fs.record_blocked("B")
        fs = fs.record_blocked("B")
        assert fs.latched is False
        fs = fs.record_blocked("B")
        assert fs.latched is True

    def test_latch_on_chatter_pattern(self) -> None:
        fs = FailSafeState.initial()
        fs = fs.record_blocked("A")
        fs = fs.record_blocked("B")
        fs = fs.record_blocked("A")
        # 3 blocked → latched by count threshold
        assert fs.latched is True

    def test_latch_permanent_until_reset(self) -> None:
        fs = FailSafeState.initial()
        for _ in range(3):
            fs = fs.record_blocked("X")
        assert fs.latched is True
        fs = fs.record_success()
        assert fs.latched is True  # latch survives success
        fs = fs.reset()
        assert fs.latched is False

    def test_immutability(self) -> None:
        fs = FailSafeState.initial()
        fs2 = fs.record_blocked("A")
        assert fs.blocked_count == 0  # original unchanged
        assert fs2.blocked_count == 1


# ===================================================================
# Mode switcher — atomic evaluation
# ===================================================================


class TestEvaluateModeSwitch:

    def test_legal_switch(self) -> None:
        result, fs = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=2000,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        assert result.switch_allowed is True
        assert result.new_mode == "warning"
        assert result.reason_code == "switched"
        assert result.dwell_guard_passed is True
        assert result.hysteresis_passed is True
        assert result.fail_safe_latched is False

    def test_dwell_blocked(self) -> None:
        result, fs = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=500,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        assert result.switch_allowed is False
        assert result.new_mode == "nominal"
        assert result.reason_code == "dwell_blocked"
        assert result.dwell_guard_passed is False

    def test_hysteresis_blocked(self) -> None:
        result, fs = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=2000,
            dwell_time_ms=1000,
            metric_value=5.0,  # below enter_threshold
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        assert result.switch_allowed is False
        assert result.new_mode == "nominal"
        assert result.reason_code == "hysteresis_blocked"
        assert result.dwell_guard_passed is True
        assert result.hysteresis_passed is False

    def test_no_change_same_mode(self) -> None:
        result, fs = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="nominal",
            last_switch_time_ms=0,
            current_time_ms=2000,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        assert result.switch_allowed is False
        assert result.reason_code == "no_change"
        assert result.new_mode == "nominal"

    def test_fail_safe_latch_after_repeated_blocks(self) -> None:
        fs = FailSafeState.initial()
        for i in range(4):
            result, fs = evaluate_mode_switch(
                current_mode="nominal",
                candidate_mode="warning",
                last_switch_time_ms=0,
                current_time_ms=500,  # always too early
                dwell_time_ms=1000,
                metric_value=10.0,
                enter_threshold=8.0,
                exit_threshold=4.0,
                fail_safe=fs,
            )
        assert fs.latched is True
        assert result.fail_safe_latched is True

    def test_fail_safe_blocks_even_valid_switch(self) -> None:
        """Once latched, even a valid switch is blocked."""
        fs = FailSafeState(blocked_count=5, latched=True, recent_modes=())
        result, fs = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=99999,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
            fail_safe=fs,
        )
        assert result.switch_allowed is False
        assert result.reason_code == "fail_safe_latched"
        assert result.new_mode == "nominal"

    def test_result_is_frozen(self) -> None:
        result, _ = evaluate_mode_switch(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=2000,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        with pytest.raises(AttributeError):
            result.new_mode = "hacked"  # type: ignore[misc]

    def test_deterministic_byte_identical_replay(self) -> None:
        """Critical determinism test: same inputs must produce
        byte-identical serialised outputs across repeated calls."""
        kwargs = dict(
            current_mode="nominal",
            candidate_mode="warning",
            last_switch_time_ms=0,
            current_time_ms=2000,
            dwell_time_ms=1000,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        result_a, fs_a = evaluate_mode_switch(**kwargs)
        result_b, fs_b = evaluate_mode_switch(**kwargs)

        # Structural equality
        assert result_a == result_b
        assert fs_a == fs_b

        # Byte-identical serialisation
        assert pickle.dumps(result_a) == pickle.dumps(result_b)
        assert pickle.dumps(fs_a) == pickle.dumps(fs_b)

    def test_wrap_safe_timing_in_full_pipeline(self) -> None:
        """64-bit wrap in integrated pipeline."""
        max_64 = (1 << 64) - 1
        result, _ = evaluate_mode_switch(
            current_mode="A",
            candidate_mode="B",
            last_switch_time_ms=max_64,
            current_time_ms=500,
            dwell_time_ms=100,
            metric_value=10.0,
            enter_threshold=8.0,
            exit_threshold=4.0,
        )
        assert result.switch_allowed is True
        assert result.reason_code == "switched"
