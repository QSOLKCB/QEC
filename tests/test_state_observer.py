"""Tests for v117.0.0 deterministic state observer layer."""

from __future__ import annotations

import math

from qec.analysis.state_observer import (
    BoundedRollingWindow,
    TRANSITION_EVENTS,
    aggregate_warning_score,
    classify_observer_state,
    compute_drift_score,
    compute_stability_score,
    compute_variance_score,
    run_state_observer,
)


class TestBoundedRollingWindow:
    def test_capacity_overwrite(self) -> None:
        window = BoundedRollingWindow(capacity=3)
        window.append(0.1)
        window.append(0.2)
        window.append(0.3)
        window.append(0.4)
        assert window.values() == [0.2, 0.3, 0.4]
        assert len(window) == 3

    def test_capacity_zero_defaults_to_one(self) -> None:
        window = BoundedRollingWindow(capacity=0)
        window.append(0.1)
        window.append(0.2)
        assert window.values() == [0.2]
        assert len(window) == 1

    def test_negative_capacity_defaults_to_one(self) -> None:
        window = BoundedRollingWindow(capacity=-5)
        window.append(0.3)
        window.append(0.4)
        assert window.values() == [0.4]
        assert len(window) == 1

    def test_clear(self) -> None:
        window = BoundedRollingWindow(capacity=3)
        window.append(0.1)
        window.append(0.2)
        window.clear()
        assert window.values() == []
        assert len(window) == 0

    def test_wrap_around_ordering(self) -> None:
        window = BoundedRollingWindow(capacity=3)
        for value in [0.1, 0.2, 0.3, 0.4, 0.5]:
            window.append(value)
        assert window.values() == [0.3, 0.4, 0.5]

    def test_clamp_behavior(self) -> None:
        window = BoundedRollingWindow(capacity=3)
        window.append(-1.0)
        window.append(1.5)
        assert window.values() == [0.0, 1.0]

    def test_repeated_wrap_around(self) -> None:
        window = BoundedRollingWindow(capacity=2)
        for value in [0.0, 0.1, 0.2, 0.3, 0.4]:
            window.append(value)
        assert window.values() == [0.3, 0.4]


class TestMetricsBounds:
    def test_variance_bounds_zero_one(self) -> None:
        score = compute_variance_score([0.0, 1.0, 0.0, 1.0])
        assert 0.0 <= score <= 1.0

    def test_drift_bounds_zero_one(self) -> None:
        score = compute_drift_score(0.9, 0.1)
        assert 0.0 <= score <= 1.0

    def test_stability_bounds_zero_one(self) -> None:
        score = compute_stability_score(0.8, 0.2)
        assert 0.0 <= score <= 1.0

    def test_warning_aggregation_bounds_zero_one(self) -> None:
        score = aggregate_warning_score(0.6, 0.4, 0.7)
        assert 0.0 <= score <= 1.0


class TestClassification:
    def test_safe_classification(self) -> None:
        assert classify_observer_state(0.2) == "safe"

    def test_warning_classification(self) -> None:
        assert classify_observer_state(0.5) == "warning"

    def test_critical_classification(self) -> None:
        assert classify_observer_state(0.9) == "critical"


class TestObserverBehavior:
    def test_transition_label_correctness(self) -> None:
        assert TRANSITION_EVENTS[("safe", "warning")] == "safe_to_warning"
        assert TRANSITION_EVENTS[("warning", "critical")] == "warning_to_critical"
        assert TRANSITION_EVENTS[("critical", "warning")] == "critical_to_warning"

    def test_transition_label_remain_warning(self) -> None:
        assert TRANSITION_EVENTS[("warning", "warning")] == "remain_warning"

    def test_transition_label_safe_to_critical(self) -> None:
        assert TRANSITION_EVENTS[("safe", "critical")] == "safe_to_critical"

    def test_transition_label_remain_critical(self) -> None:
        assert TRANSITION_EVENTS[("critical", "critical")] == "remain_critical"

    def test_nan_handling(self) -> None:
        result, window = run_state_observer(math.nan)
        assert result["warning_score"] == 0.0
        assert result["variance_score"] == 0.0
        assert result["drift_score"] == 0.0
        assert result["stability_score"] == 1.0
        assert result["confidence_score"] == 1.0
        assert window.values() == [0.0]

    def test_empty_history_handling(self) -> None:
        window = BoundedRollingWindow()
        window.clear()
        result, _ = run_state_observer(0.0, rolling_window=window)
        assert result["history_depth"] > 0.0
        assert result["observer_state"] == "safe"

    def test_deterministic_repeated_identical_output(self) -> None:
        sequence = [0.1, 0.2, 0.25, 0.3]

        window_a = BoundedRollingWindow(capacity=8)
        state_a = "safe"
        last_a = None
        for metric in sequence:
            last_a, _ = run_state_observer(metric, rolling_window=window_a, previous_state=state_a)
            state_a = str(last_a["observer_state"])

        window_b = BoundedRollingWindow(capacity=8)
        state_b = "safe"
        last_b = None
        for metric in sequence:
            last_b, _ = run_state_observer(metric, rolling_window=window_b, previous_state=state_b)
            state_b = str(last_b["observer_state"])

        assert last_a == last_b
