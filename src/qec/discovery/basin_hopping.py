"""Deterministic basin-hopping proposal utilities."""

from __future__ import annotations

import numpy as np


def propose_basin_hop(
    current_vector: list[float] | np.ndarray,
    basins: list[dict[str, object]],
) -> list[float]:
    """Propose a deterministic spectral hop toward a different basin."""
    current = np.asarray(current_vector, dtype=np.float64)
    if current.ndim != 1:
        raise ValueError("current_vector must be 1D")
    if not basins:
        return current.tolist()

    ordered_basins = sorted(
        basins,
        key=lambda rec: (
            int(rec.get("basin_id", 0)),
            tuple(float(np.float64(v)) for v in rec.get("centroid", [])),
        ),
    )

    dim = current.shape[0]
    centroids = np.asarray([b.get("centroid", [0.0] * dim) for b in ordered_basins], dtype=np.float64)
    basin_ids = np.asarray([int(b.get("basin_id", i)) for i, b in enumerate(ordered_basins)], dtype=np.int64)

    if centroids.shape[1] != dim:
        raise ValueError("basin centroid dimensionality must match current_vector")

    deltas = centroids - current
    distances = np.sqrt(np.sum(np.square(deltas, dtype=np.float64), axis=1), dtype=np.float64)
    nearest_order = np.lexsort((basin_ids, distances))
    current_basin_id = int(basin_ids[int(nearest_order[0])])

    candidate_ids = [int(bid) for bid in basin_ids.tolist() if int(bid) != current_basin_id]
    if not candidate_ids:
        candidate_ids = [current_basin_id]
    target_basin_id = int(sorted(candidate_ids)[0])

    target_index = int(np.where(basin_ids == target_basin_id)[0][0])
    target_centroid = np.asarray(centroids[target_index], dtype=np.float64)

    offset = (np.float64((target_basin_id % 7) + 1) * np.float64(1e-3))
    direction = np.arange(1, dim + 1, dtype=np.float64)
    direction /= np.sqrt(np.sum(np.square(direction), dtype=np.float64), dtype=np.float64)
    proposal = target_centroid + (offset * direction)
    return np.asarray(proposal, dtype=np.float64).tolist()
