"""Deterministic experiment target extraction from phase uncertainty maps."""

from __future__ import annotations

from typing import Any

import numpy as np


def detect_high_uncertainty_regions(
    uncertainty_map: dict[str, Any] | np.ndarray,
    threshold: float,
) -> list[dict[str, float]]:
    """Return deterministic coordinates above uncertainty threshold."""
    if isinstance(uncertainty_map, dict):
        grid_x = np.asarray(uncertainty_map.get("grid_x", []), dtype=np.float64)
        grid_y = np.asarray(uncertainty_map.get("grid_y", []), dtype=np.float64)
        scores = np.asarray(uncertainty_map.get("uncertainty_map", []), dtype=np.float64)
    else:
        grid_x = np.asarray([], dtype=np.float64)
        grid_y = np.asarray([], dtype=np.float64)
        scores = np.asarray(uncertainty_map, dtype=np.float64)

    if scores.size == 0:
        return []
    if scores.ndim != 2:
        raise ValueError("uncertainty map must be 2D")

    rows, cols = np.where(scores >= float(threshold))
    if rows.size == 0:
        return []

    vals = scores[rows, cols]
    idx = np.arange(rows.size, dtype=np.int64)
    order = np.lexsort((idx, -vals))

    out: list[dict[str, float]] = []
    for o in order:
        i = int(rows[o])
        j = int(cols[o])
        radius = float(grid_x[i]) if grid_x.size > i else float(i)
        bethe = float(grid_y[j]) if grid_y.size > j else float(j)
        out.append(
            {
                "spectral_radius": radius,
                "bethe_eigenvalue": bethe,
                "uncertainty_score": float(vals[o]),
            }
        )
    return out


def generate_experiment_targets(
    regions: list[dict[str, float]],
    max_targets: int = 10,
) -> list[np.ndarray]:
    """Convert uncertainty regions into deterministic float64 spectral vectors."""
    if not regions or int(max_targets) <= 0:
        return []

    rows: list[tuple[float, float, float, int]] = []
    for i, region in enumerate(regions):
        rows.append(
            (
                float(np.float64(region.get("uncertainty_score", 0.0))),
                float(np.float64(region.get("spectral_radius", 0.0))),
                float(np.float64(region.get("bethe_eigenvalue", 0.0))),
                int(i),
            )
        )

    scores = np.asarray([r[0] for r in rows], dtype=np.float64)
    radii = np.asarray([r[1] for r in rows], dtype=np.float64)
    bethe = np.asarray([r[2] for r in rows], dtype=np.float64)
    idx = np.asarray([r[3] for r in rows], dtype=np.int64)
    order = np.lexsort((idx, bethe, radii, -scores))

    targets: list[np.ndarray] = []
    for k in order[: int(max_targets)]:
        targets.append(
            np.asarray([radii[k], bethe[k], scores[k], 0.0], dtype=np.float64)
        )
    return targets
