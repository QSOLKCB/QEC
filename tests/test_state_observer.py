"""Tests for v117.0.0 deterministic state observer layer."""

from __future__ import annotations

import math

from qec.analysis.state_observer import (
    BoundedRollingWindow,
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
        _, window = run_state_observer(0.0)
        warning_result, _ = run_state_observer(1.0, rolling_window=window, previous_state="safe")
        escalated_result, _ = run_state_observer(1.0, rolling_window=window, previous_state="warning")
        deescalated_result, _ = run_state_observer(0.0, previous_state="critical")

        assert warning_result["state_transition_event"] == "safe_to_warning"
        assert escalated_result["state_transition_event"] == "warning_to_critical"
        assert deescalated_result["state_transition_event"] == "critical_to_warning"

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
