"""Spectral-space trust region controller for mutation acceptance."""

from __future__ import annotations

import numpy as np


class SpectralTrustRegion:
    """Limits mutation step size in spectral space."""

    def __init__(self, radius: float = 0.25):
        self.radius = float(radius)

    def spectral_distance(self, s_old, s_new) -> float:
        s_old_arr = np.asarray(s_old, dtype=np.float64)
        s_new_arr = np.asarray(s_new, dtype=np.float64)
        return float(np.linalg.norm(s_new_arr - s_old_arr))

    def allow(self, s_old, s_new) -> bool:
        return self.spectral_distance(s_old, s_new) <= self.radius
