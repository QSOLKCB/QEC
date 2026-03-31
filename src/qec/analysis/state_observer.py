"""v117.0.0 — Deterministic state observer layer.

Transforms raw metric observations into bounded warning-state summaries.
All computations are deterministic, bounded, and additive.
"""

from __future__ import annotations

import math
from typing import Iterable


SAFE_WARNING_THRESHOLD = 0.30
CRITICAL_WARNING_THRESHOLD = 0.70
ROLLING_WINDOW_CAPACITY = 32


def _clamp_unit(value: float) -> float:
    """Clamp to [0, 1], with NaN/inf-safe fallback to 0.0."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return 0.0
    if numeric_value <= 0.0:
        return 0.0
    if numeric_value >= 1.0:
        return 1.0
    return numeric_value


class BoundedRollingWindow:
    """Deterministic fixed-capacity rolling history with constant memory."""

    def __init__(self, capacity: int = ROLLING_WINDOW_CAPACITY) -> None:
        self.capacity = max(1, int(capacity))
        self._buffer = [0.0] * self.capacity
        self._size = 0
        self._start = 0

    def append(self, value: float) -> None:
        """Append a value, overwriting the oldest when full."""
        sanitized_value = _clamp_unit(value)
        if self._size < self.capacity:
            write_index = (self._start + self._size) % self.capacity
            self._buffer[write_index] = sanitized_value
            self._size += 1
            return

        self._buffer[self._start] = sanitized_value
        self._start = (self._start + 1) % self.capacity

    def values(self) -> list[float]:
        """Return values in insertion order (oldest to newest)."""
        return [
            self._buffer[(self._start + index) % self.capacity]
            for index in range(self._size)
        ]

    def clear(self) -> None:
        """Reset window to empty state."""
        self._size = 0
        self._start = 0

    def __len__(self) -> int:
        return self._size


def compute_variance_score(history: Iterable[float]) -> float:
    """Compute normalized variance score [0, 1] from rolling values."""
    values = [_clamp_unit(value) for value in history]
    if len(values) < 2:
        return 0.0

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    # For values constrained to [0,1], population variance is <= 0.25.
    return _clamp_unit(variance / 0.25)


def compute_drift_score(current_value: float, rolling_mean: float) -> float:
    """Compute normalized morphology drift score [0, 1]."""
    current = _clamp_unit(current_value)
    mean = _clamp_unit(rolling_mean)
    return _clamp_unit(abs(current - mean))


def compute_stability_score(variance_score: float, drift_score: float) -> float:
    """Compute inverse instability score, bounded in [0, 1]."""
    variance = _clamp_unit(variance_score)
    drift = _clamp_unit(drift_score)
    return _clamp_unit(1.0 - max(variance, drift))


def aggregate_warning_score(
    variance_score: float,
    drift_score: float,
    stability_score: float,
) -> float:
    """Compute deterministic warning score [0, 1]."""
    variance = _clamp_unit(variance_score)
    drift = _clamp_unit(drift_score)
    stability = _clamp_unit(stability_score)
    inverse_stability = 1.0 - stability
    warning_score = (0.4 * variance) + (0.4 * drift) + (0.2 * inverse_stability)
    return _clamp_unit(warning_score)


def classify_observer_state(warning_score: float) -> str:
    """Classify observer state as safe / warning / critical."""
    score = _clamp_unit(warning_score)
    if score < SAFE_WARNING_THRESHOLD:
        return "safe"
    if score < CRITICAL_WARNING_THRESHOLD:
        return "warning"
    return "critical"


def _compute_transition_event(previous_state: str, observer_state: str) -> str:
    """Return deterministic transition label from finite state transitions."""
    if previous_state == "critical":
        if observer_state == "critical":
            return "remain_critical"
        return "critical_to_warning"

    if previous_state == "warning":
        if observer_state == "safe":
            return "warning_to_safe"
        return "warning_to_critical"

    if observer_state == "safe":
        return "remain_safe"
    return "safe_to_warning"


def run_state_observer(
    metric_value: float,
    rolling_window: BoundedRollingWindow | None = None,
    previous_state: str = "safe",
) -> tuple[dict[str, float | str | bool], BoundedRollingWindow]:
    """Run deterministic observer layer and return bounded summary + window."""
    window = rolling_window if rolling_window is not None else BoundedRollingWindow()
    window.append(metric_value)

    history = window.values()
    history_depth = _clamp_unit(len(window) / window.capacity)

    if history:
        rolling_mean = sum(history) / len(history)
    else:
        rolling_mean = 0.0

    variance_score = compute_variance_score(history)
    drift_score = compute_drift_score(history[-1] if history else 0.0, rolling_mean)
    stability_score = compute_stability_score(variance_score, drift_score)
    warning_score = aggregate_warning_score(variance_score, drift_score, stability_score)
    confidence_score = _clamp_unit(stability_score * (1.0 - warning_score))

    observer_state = classify_observer_state(warning_score)
    transition_event = _compute_transition_event(previous_state, observer_state)

    result: dict[str, float | str | bool] = {
        "observer_state": observer_state,
        "warning_score": warning_score,
        "variance_score": variance_score,
        "drift_score": drift_score,
        "stability_score": stability_score,
        "confidence_score": confidence_score,
        "history_depth": history_depth,
        "state_transition_event": transition_event,
        "warning_triggered": observer_state in {"warning", "critical"},
    }
    return result, window


__all__ = [
    "SAFE_WARNING_THRESHOLD",
    "CRITICAL_WARNING_THRESHOLD",
    "ROLLING_WINDOW_CAPACITY",
    "BoundedRollingWindow",
    "compute_variance_score",
    "compute_drift_score",
    "compute_stability_score",
    "aggregate_warning_score",
    "classify_observer_state",
    "run_state_observer",
]
