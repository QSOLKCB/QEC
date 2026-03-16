"""v43.0.0 — Deterministic spectral target selection for experiments."""

from __future__ import annotations

import numpy as np


def choose_experiment_target(gap_candidates: list[np.ndarray] | np.ndarray) -> np.ndarray | None:
    """Choose a deterministic target spectrum from gap candidates."""
    arr = np.asarray(gap_candidates, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    magnitudes = np.linalg.norm(arr, axis=1)
    # Stable deterministic ordering: largest magnitude, then lexicographic vector.
    lex_keys = [arr[:, k] for k in range(arr.shape[1] - 1, -1, -1)]
    order = np.lexsort(tuple(lex_keys + [-magnitudes]))
    return arr[int(order[0])].astype(np.float64, copy=False)
