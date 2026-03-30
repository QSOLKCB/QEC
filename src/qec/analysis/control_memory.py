"""v105.5.1 — Deterministic temporal smoothing and trend memory layer.

Consumes outputs from run_control_flow() and distinguishes
one-off instability spikes from sustained collapse trends.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- stdlib only

Dependencies: none (stdlib only).
"""

from __future__ import annotations

TREND_RISE_THRESHOLD = 0.1
TREND_FALL_THRESHOLD = -0.1


def compute_ema(previous: float, current: float, alpha: float = 0.5) -> float:
    """Standard deterministic exponential moving average.

    alpha is automatically clamped to [0.0, 1.0].
    """
    alpha = min(1.0, max(0.0, alpha))
    return alpha * current + (1.0 - alpha) * previous


def compute_trend_delta(previous: float, current: float) -> float:
    """Simple difference: current - previous."""
    return current - previous


def classify_control_trend(delta: float) -> str:
    """Classify trend direction from delta value."""
    if delta > TREND_RISE_THRESHOLD:
        return "rising"
    if delta < TREND_FALL_THRESHOLD:
        return "falling"
    return "stable"


def run_control_memory(previous_signal: float, current_signal: float) -> dict:
    """Compute smoothed signal, trend delta, and trend classification."""
    smoothed = compute_ema(previous_signal, current_signal)
    delta = compute_trend_delta(previous_signal, current_signal)
    state = classify_control_trend(delta)
    return {
        "smoothed_signal": round(smoothed, 12),
        "trend_delta": round(delta, 12),
        "trend_state": state,
    }
