"""v43.0.0 — Deterministic metrics for autonomous scheduling."""

from __future__ import annotations

import numpy as np

from src.qec.analysis.landscape_gaps import detect_landscape_gaps


def _centers_from_memory(memory) -> np.ndarray:
    if hasattr(memory, "centers") and callable(memory.centers):
        centers = memory.centers()
    else:
        centers = memory
    arr = np.asarray(centers, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def landscape_gap_count(memory, gap_radius: float = 0.3, max_gaps: int = 16) -> int:
    """Count deterministic gap candidates from landscape memory."""
    return int(len(detect_landscape_gaps(memory, gap_radius=float(gap_radius), max_gaps=int(max_gaps))))


def mean_gap_distance(memory, gap_radius: float = 0.3, max_gaps: int = 16) -> float:
    """Mean nearest-center distance for deterministic gap candidates."""
    gaps = detect_landscape_gaps(memory, gap_radius=float(gap_radius), max_gaps=int(max_gaps))
    if not gaps:
        return 0.0
    centers = _centers_from_memory(memory)
    if centers.size == 0:
        return 0.0

    distances = []
    for gap in gaps:
        diff = centers - np.asarray(gap, dtype=np.float64)
        d = np.linalg.norm(diff, axis=1)
        distances.append(float(np.min(d)))
    return float(np.mean(np.asarray(distances, dtype=np.float64)))


def scheduled_experiment_targets(queue) -> list[np.ndarray]:
    """Return queued target spectra in deterministic FIFO order."""
    if hasattr(queue, "_items"):
        return [np.asarray(t, dtype=np.float64).copy() for t in queue._items]
    if isinstance(queue, list):
        return [np.asarray(t, dtype=np.float64).copy() for t in queue]
    return []
