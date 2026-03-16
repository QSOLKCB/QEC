"""Deterministic spectral uncertainty estimation (float64)."""

from __future__ import annotations

import numpy as np


def estimate_spectral_uncertainty(memory: object, spectrum: np.ndarray | list[float]) -> float:
    """Estimate uncertainty from nearest known region center.

    Uses: uncertainty = d / (1 + d), where d is the minimum squared distance
    from ``spectrum`` to known region centers.
    """
    spec = np.asarray(spectrum, dtype=np.float64)
    if spec.ndim != 1:
        raise ValueError("spectrum must be 1D")

    centers = memory.centers()
    if centers is None or len(centers) == 0:
        return float(np.float64(1.0))

    centers = np.asarray(centers, dtype=np.float64)

    diff = centers - spec
    dists2 = np.sum(diff * diff, axis=1, dtype=np.float64)
    d = np.float64(np.min(dists2))
    return float(d / (np.float64(1.0) + d))

