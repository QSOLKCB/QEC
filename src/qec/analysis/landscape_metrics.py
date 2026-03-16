"""Deterministic spectral landscape metrics."""

from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory


def novelty_score(
    spectrum: np.ndarray | list[float],
    memory: SpectralLandscapeMemory,
) -> float:
    """Compute normalized novelty from nearest-region squared distance."""
    spec = np.asarray(spectrum, dtype=np.float64)
    if spec.ndim != 1:
        raise ValueError("spectrum must be 1D")
    if memory.region_count == 0:
        return 1.0
    centers = memory.centers_array[: memory.region_count]
    diff = centers - spec
    dists2 = np.sum(diff * diff, axis=1)
    min_distance2 = float(np.min(dists2))
    return float(np.float64(min_distance2) / (np.float64(1.0) + np.float64(min_distance2)))


def landscape_coverage(memory: SpectralLandscapeMemory) -> float:
    """Return number of discovered regions as float for stable artifact schema."""
    return float(np.float64(memory.region_count))
