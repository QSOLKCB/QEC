"""v105.5.0 — Deterministic temporal smoothing and trend memory layer.

Consumes outputs from run_control_flow() and distinguishes
one-off instability spikes from sustained collapse trends.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- stdlib only

Dependencies: none (stdlib only).
"""

from __future__ import annotations


def compute_ema(previous: float, current: float, alpha: float = 0.5) -> float:
    """Standard deterministic exponential moving average."""
    alpha = min(1.0, max(0.0, alpha))
    return round(alpha * current + (1.0 - alpha) * previous, 12)


def compute_trend_delta(previous: float, current: float) -> float:
    """Simple difference: current - previous."""
    return round(current - previous, 12)


def classify_control_trend(delta: float) -> str:
    """Classify trend direction from delta value."""
    if delta > 0.1:
        return "rising"
    if delta < -0.1:
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
