"""Bosonic analog interface — experimental design layer.

Normalizes analog (continuous) inputs to the [-1, 1] range for
downstream ternary quantization. Pure, deterministic, no side effects.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def normalize_to_bipolar(
    values: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Normalize an array of analog values to the [-1, 1] range.

    Uses min-max normalization mapped to [-1, 1]. If all values are
    identical, returns zeros (neutral).

    Parameters
    ----------
    values : np.ndarray
        Raw analog input values (1-D float array).
    clip : bool
        If True, clip outputs to exactly [-1, 1] to guard against
        floating-point drift.

    Returns
    -------
    np.ndarray
        Normalized values in [-1, 1].
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)

    vmin = values.min()
    vmax = values.max()
    span = vmax - vmin

    if span == 0.0:
        return np.zeros_like(values)

    normalized = 2.0 * (values - vmin) / span - 1.0

    if clip:
        np.clip(normalized, -1.0, 1.0, out=normalized)

    return normalized


def prepare_bosonic_input(
    raw_signals: np.ndarray,
    *,
    clip: bool = True,
) -> Dict[str, Any]:
    """Prepare a bosonic input packet for the ternary pipeline.

    Parameters
    ----------
    raw_signals : np.ndarray
        Raw analog signal values.
    clip : bool
        Whether to clip normalized outputs.

    Returns
    -------
    dict
        Keys: ``"normalized"`` (np.ndarray in [-1,1]),
        ``"length"`` (int), ``"input_range"`` (tuple of float).
    """
    raw = np.asarray(raw_signals, dtype=np.float64)
    normalized = normalize_to_bipolar(raw, clip=clip)
    return {
        "normalized": normalized,
        "length": int(raw.size),
        "input_range": (float(raw.min()), float(raw.max())) if raw.size > 0 else (0.0, 0.0),
    }
