"""Ternary quantization — experimental design layer.

Maps bipolar [-1, 1] values to the ternary alphabet {-1, 0, +1}
using a deterministic threshold. Pure, no side effects.

Dependencies: numpy.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def quantize_ternary(
    values: np.ndarray,
    threshold: float = 0.3,
) -> np.ndarray:
    """Quantize bipolar values to ternary {-1, 0, +1}.

    Values with absolute magnitude below *threshold* map to 0 (neutral).
    Values >= threshold map to +1; values <= -threshold map to -1.

    Parameters
    ----------
    values : np.ndarray
        Input values, expected in [-1, 1].
    threshold : float
        Dead-zone half-width. Must be in (0, 1].

    Returns
    -------
    np.ndarray
        Integer array with values in {-1, 0, +1}.

    Raises
    ------
    ValueError
        If threshold is not in (0, 1].
    """
    if not (0.0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    values = np.asarray(values, dtype=np.float64)
    result = np.zeros(values.shape, dtype=np.int8)
    result[values >= threshold] = 1
    result[values <= -threshold] = -1
    return result


def ternary_stats(ternary: np.ndarray) -> Dict[str, float]:
    """Compute statistics over a ternary array.

    Parameters
    ----------
    ternary : np.ndarray
        Array with values in {-1, 0, +1}.

    Returns
    -------
    dict
        ``"n_positive"``, ``"n_negative"``, ``"n_neutral"``,
        ``"neutral_fraction"`` (float in [0,1]).
    """
    n = max(ternary.size, 1)
    n_pos = int(np.sum(ternary == 1))
    n_neg = int(np.sum(ternary == -1))
    n_neut = int(np.sum(ternary == 0))
    return {
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_neutral": n_neut,
        "neutral_fraction": n_neut / n,
    }
