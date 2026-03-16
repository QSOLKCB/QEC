"""Deterministic basin stagnation detection helpers."""

from __future__ import annotations

import numpy as np


def detect_basin_stagnation(assignments: list[int] | np.ndarray, window: int = 10) -> bool:
    """Return ``True`` when recent basin assignments are constant.

    Parameters
    ----------
    assignments : list[int] or np.ndarray
        Basin assignment history in temporal order.
    window : int
        Number of latest assignments required for stagnation detection.

    Returns
    -------
    bool
        ``True`` if the last ``window`` assignments are equal.
    """
    if window <= 0:
        return False

    arr = np.asarray(assignments, dtype=np.int64)
    if arr.size < int(window):
        return False

    recent = arr[-int(window):]
    return bool(np.all(recent == recent[0]))
