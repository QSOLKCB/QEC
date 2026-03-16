"""Deterministic uncertainty estimation over spectral phase diagrams."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_phase_uncertainty(phase_surface: dict[str, Any]) -> dict[str, np.ndarray]:
    """Estimate local phase uncertainty via deterministic 3x3 window variance.

    Parameters
    ----------
    phase_surface : dict[str, Any]
        Dictionary containing ``grid_x``, ``grid_y``, and ``grid_z`` arrays.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with ``grid_x``, ``grid_y``, and ``uncertainty_map``.
    """
    grid_x = np.asarray(phase_surface.get("grid_x", []), dtype=np.float64)
    grid_y = np.asarray(phase_surface.get("grid_y", []), dtype=np.float64)
    grid_z = np.asarray(phase_surface.get("grid_z", []), dtype=np.float64)

    if grid_z.size == 0:
        return {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "uncertainty_map": np.zeros((0, 0), dtype=np.float64),
        }
    if grid_z.ndim != 2:
        raise ValueError("phase_surface['grid_z'] must be a 2D grid")

    padded = np.pad(grid_z, ((1, 1), (1, 1)), mode="edge")
    uncertainty = np.zeros_like(grid_z, dtype=np.float64)

    for i in range(grid_z.shape[0]):
        for j in range(grid_z.shape[1]):
            window = padded[i:i + 3, j:j + 3]
            uncertainty[i, j] = np.var(window, dtype=np.float64)

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "uncertainty_map": uncertainty.astype(np.float64, copy=False),
    }

