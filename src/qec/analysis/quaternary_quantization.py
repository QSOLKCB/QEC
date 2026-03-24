"""Quaternary quantization — experimental design layer.

Maps bipolar [-1, 1] values to the quaternary alphabet {-1.0, -0.5, 0.5, 1.0}
using deterministic nearest-neighbor mapping. Pure, no side effects.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


QUATERNARY_STATES: tuple[float, ...] = (-1.0, -0.5, 0.5, 1.0)

_QUATERNARY_ARRAY = np.array(QUATERNARY_STATES, dtype=np.float64)


def quantize_quaternary(values: np.ndarray) -> np.ndarray:
    """Quantize bipolar values to quaternary {-1.0, -0.5, 0.5, 1.0}.

    Each value is mapped to the nearest quaternary state using
    deterministic nearest-neighbor assignment.

    Parameters
    ----------
    values : np.ndarray
        Input values, expected in [-1, 1].

    Returns
    -------
    np.ndarray
        Float64 array with values in {-1.0, -0.5, 0.5, 1.0}.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.array([], dtype=np.float64)

    # Compute absolute distance to each quaternary state
    # Shape: (len(values), 4)
    distances = np.abs(values[:, np.newaxis] - _QUATERNARY_ARRAY[np.newaxis, :])

    # Nearest-neighbor: argmin along axis 1
    indices = np.argmin(distances, axis=1)

    return _QUATERNARY_ARRAY[indices].copy()


def quaternary_stats(quaternary: np.ndarray) -> Dict[str, float]:
    """Compute statistics over a quaternary array.

    Parameters
    ----------
    quaternary : np.ndarray
        Array with values in {-1.0, -0.5, 0.5, 1.0}.

    Returns
    -------
    dict
        Counts for each state and the fraction of soft states
        (those at -0.5 or 0.5).
    """
    n = max(quaternary.size, 1)
    n_strong_pos = int(np.sum(quaternary == 1.0))
    n_strong_neg = int(np.sum(quaternary == -1.0))
    n_soft_pos = int(np.sum(quaternary == 0.5))
    n_soft_neg = int(np.sum(quaternary == -0.5))
    soft_count = n_soft_pos + n_soft_neg
    return {
        "n_strong_positive": n_strong_pos,
        "n_strong_negative": n_strong_neg,
        "n_soft_positive": n_soft_pos,
        "n_soft_negative": n_soft_neg,
        "soft_fraction": soft_count / n,
    }
