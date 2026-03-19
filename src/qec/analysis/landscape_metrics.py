"""Deterministic spectral landscape metrics."""

from __future__ import annotations

import numpy as np

from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory


def novelty_score(
    spectrum: np.ndarray | list[float],
    memory: SpectralLandscapeMemory,
    *,
    reuse_kd_tree: bool = False,
) -> float:
    """Compute normalized novelty from nearest-region squared distance."""
    spec = np.asarray(spectrum, dtype=np.float64)
    if spec.ndim != 1:
        raise ValueError("spectrum must be 1D")
    if memory.region_count == 0:
        return 1.0
    min_distance2 = memory.nearest_distance2(spec, reuse_index=bool(reuse_kd_tree))
    return float(np.float64(min_distance2) / (np.float64(1.0) + np.float64(min_distance2)))


def landscape_coverage(memory: SpectralLandscapeMemory) -> float:
    """Return number of discovered regions as float for stable artifact schema."""
    return float(np.float64(memory.region_count))
