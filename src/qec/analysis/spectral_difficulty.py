"""Deterministic spectral difficulty estimation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_WEIGHTS = {
    "instability_weight": np.float64(0.5),
    "distance_weight": np.float64(0.3),
    "uncertainty_weight": np.float64(0.2),
}


def _as_spectrum(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        return np.zeros((0,), dtype=np.float64)
    return arr


def _extract_successful_regions(memory: Any) -> np.ndarray:
    if memory is None:
        return np.zeros((0, 0), dtype=np.float64)
    if hasattr(memory, "successful_regions") and callable(memory.successful_regions):
        regions = np.asarray(memory.successful_regions(), dtype=np.float64)
        if regions.ndim == 2:
            return regions
    if hasattr(memory, "centers") and callable(memory.centers):
        centers = np.asarray(memory.centers(), dtype=np.float64)
        if centers.ndim == 2:
            return centers
    if isinstance(memory, dict) and "successful_regions" in memory:
        regions = np.asarray(memory.get("successful_regions"), dtype=np.float64)
        if regions.ndim == 2:
            return regions
    return np.zeros((0, 0), dtype=np.float64)


def _extract_recent_spectrum(memory: Any) -> np.ndarray:
    if memory is None:
        return np.zeros((0,), dtype=np.float64)
    if hasattr(memory, "recent_spectrum"):
        return _as_spectrum(getattr(memory, "recent_spectrum"))
    if isinstance(memory, dict):
        return _as_spectrum(memory.get("recent_spectrum", []))
    return np.zeros((0,), dtype=np.float64)


def _extract_uncertainty(memory: Any, spectrum: np.ndarray) -> np.float64:
    if memory is None:
        return np.float64(0.0)
    if hasattr(memory, "predict_uncertainty") and callable(memory.predict_uncertainty):
        return np.float64(memory.predict_uncertainty(spectrum))
    if isinstance(memory, dict):
        return np.float64(memory.get("spectral_uncertainty", 0.0))
    return np.float64(0.0)


def _extract_weight(memory: Any, key: str) -> np.float64:
    if isinstance(memory, dict) and key in memory:
        return np.float64(memory[key])
    return np.float64(_DEFAULT_WEIGHTS[key])


def estimate_spectral_difficulty(memory: Any, spectrum: np.ndarray) -> np.float64:
    """Estimate deterministic spectral-region difficulty as float64.

    Combines instability score, distance to successful regions, and
    spectral uncertainty with fixed deterministic weights.
    """
    spec = _as_spectrum(spectrum)
    previous = _extract_recent_spectrum(memory)
    if previous.shape == spec.shape and spec.size > 0:
        instability_score = np.float64(np.linalg.norm(spec - previous))
    else:
        instability_score = np.float64(0.0)

    successful = _extract_successful_regions(memory)
    if successful.size > 0 and successful.shape[1] == spec.shape[0]:
        distances = np.linalg.norm(successful - spec[np.newaxis, :], axis=1)
        distance_to_success = np.float64(np.min(distances))
    else:
        distance_to_success = np.float64(0.0)

    spectral_uncertainty = _extract_uncertainty(memory, spec)

    difficulty = (
        _extract_weight(memory, "instability_weight") * instability_score
        + _extract_weight(memory, "distance_weight") * distance_to_success
        + _extract_weight(memory, "uncertainty_weight") * spectral_uncertainty
    )
    return np.float64(difficulty)
