"""Decoder design baselines — experimental design layer.

Simple deterministic baseline decoders for comparison against the
ternary message-passing pipeline. Pure functions, no side effects.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def hard_threshold_decoder(
    values: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Hard-threshold binary decoder.

    Maps values > threshold to +1, values <= threshold to -1.

    Parameters
    ----------
    values : np.ndarray
        Analog input values.
    threshold : float
        Decision boundary (default 0.0).

    Returns
    -------
    np.ndarray
        Binary decisions in {-1, +1}.
    """
    values = np.asarray(values, dtype=np.float64)
    result = np.where(values > threshold, 1, -1).astype(np.int8)
    return result


def signed_soft_decoder(
    values: np.ndarray,
) -> np.ndarray:
    """Signed soft decoder.

    Returns the sign of each value as the decision, with zero mapping
    to +1. Preserves the original magnitude as a soft confidence proxy.

    Parameters
    ----------
    values : np.ndarray
        Analog input values.

    Returns
    -------
    np.ndarray
        Soft decisions: sign(value) * |value|, where sign(0) = +1.
    """
    values = np.asarray(values, dtype=np.float64)
    signs = np.sign(values)
    signs[signs == 0] = 1.0
    return signs * np.abs(values)


def run_baselines(
    values: np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Run all baseline decoders on the same input.

    Parameters
    ----------
    values : np.ndarray
        Analog input values.
    threshold : float
        Threshold for hard decoder.

    Returns
    -------
    dict
        ``"hard_threshold"`` and ``"signed_soft"`` arrays.
    """
    return {
        "hard_threshold": hard_threshold_decoder(values, threshold),
        "signed_soft": signed_soft_decoder(values),
    }
