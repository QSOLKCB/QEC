"""Deterministic spectral-space geometry helpers."""

from __future__ import annotations

import numpy as np


def spectral_distance(a, b) -> float:
    """Euclidean distance between two spectral feature vectors."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def spectral_entropy(eigs) -> float:
    """Shannon entropy of normalized spectral magnitudes."""
    eigs_arr = np.abs(np.asarray(eigs, dtype=np.float64))
    total = float(eigs_arr.sum())
    if total <= 0.0:
        return 0.0
    p = eigs_arr / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def spectral_diversity(history) -> float:
    """Mean consecutive spectral distance across a trajectory."""
    if len(history) < 2:
        return 0.0
    dists = [spectral_distance(history[i], history[i - 1]) for i in range(1, len(history))]
    return float(np.mean(np.asarray(dists, dtype=np.float64)))
