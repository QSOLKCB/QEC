"""Export helpers for spectral basin topology artifacts."""

from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_phase_space import project_basin_centers_2d, project_basin_centers_3d


def export_basin_map(
    basins: np.ndarray,
    assignments: np.ndarray,
    transitions: list[int],
    *,
    include_phase_space_projections: bool = False,
    projection_2d_dims: tuple[int, int] = (0, 1),
    projection_3d_dims: tuple[int, int, int] = (0, 1, 2),
) -> dict[str, object]:
    """Build a JSON-safe basin topology payload."""
    basin_arr = np.asarray(basins, dtype=np.float64)
    assignment_arr = np.asarray(assignments, dtype=np.int64)
    payload: dict[str, object] = {
        "num_basins": int(basin_arr.shape[0]),
        "basin_centers": basin_arr.tolist(),
        "assignments": assignment_arr.tolist(),
        "transitions": [int(t) for t in transitions],
    }

    if include_phase_space_projections:
        payload["basin_centers_projected_2d"] = project_basin_centers_2d(
            basin_arr,
            dims=projection_2d_dims,
        ).tolist()
        payload["basin_centers_projected_3d"] = project_basin_centers_3d(
            basin_arr,
            dims=projection_3d_dims,
        ).tolist()
    return payload
