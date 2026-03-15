"""Deterministic predictor recalibration helpers."""

from __future__ import annotations

import numpy as np

_ROUND = 12


def compute_recalibration_bias(predictions, actuals):
    """Deterministically compute predictor bias correction."""

    if not predictions:
        return 0.0

    p = np.asarray(predictions, dtype=np.float64)
    a = np.asarray(actuals, dtype=np.float64)

    bias = float(np.mean(a - p))
    return round(bias, _ROUND)


def apply_recalibration(value, bias):
    return round(float(value + bias), _ROUND)
