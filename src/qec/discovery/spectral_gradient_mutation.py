"""Helpers for optional gradient-guided spectral mutation targeting."""

from __future__ import annotations

import numpy as np


def propose_gradient_step(current_spectrum, gradient, step: float = 0.1) -> np.ndarray:
    """Compute a deterministic target spectrum by taking a gradient step."""
    return np.asarray(current_spectrum, dtype=np.float64) + (
        float(step) * np.asarray(gradient, dtype=np.float64)
    )
