"""Deterministic curriculum-learning metrics."""

from __future__ import annotations

import numpy as np


def curriculum_success_rate(successful_codes: int, attempted_codes: int) -> np.float64:
    """Compute deterministic success rate with safe division."""
    attempts = int(attempted_codes)
    if attempts <= 0:
        return np.float64(0.0)
    rate = np.float64(successful_codes) / np.float64(attempts)
    return np.float64(min(max(float(rate), 0.0), 1.0))


def curriculum_progress(
    successful_codes: int,
    attempted_codes: int,
    threshold: float,
) -> dict[str, float | bool]:
    """Compute curriculum progress scalar and threshold status."""
    rate = curriculum_success_rate(successful_codes, attempted_codes)
    th = float(np.float64(threshold))
    progress = np.float64(min(float(rate) / th, 1.0)) if th > 0.0 else np.float64(1.0)
    return {
        "success_rate": float(rate),
        "progress": float(progress),
        "advance": bool(float(rate) >= th),
    }
