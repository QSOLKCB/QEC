"""Deterministic spectral entropy utilities for NB eigenvalue magnitudes."""

from __future__ import annotations

import numpy as np


def spectral_entropy(eigvals: np.ndarray, precision: int = 12) -> float:
    """Compute normalized spectral entropy of NB eigenvalues."""
    vals = np.asarray(np.abs(np.asarray(eigvals)), dtype=np.float64)
    total = float(np.sum(vals, dtype=np.float64))
    if total == 0.0:
        return 0.0

    p = vals / total
    entropy = -np.sum(p * np.log(p + 1e-15), dtype=np.float64)
    return round(float(entropy), int(precision))
