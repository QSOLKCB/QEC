"""Deterministic spectral basin identification utilities."""

from __future__ import annotations

import numpy as np


_DISTANCE_EPS = np.float64(1e-12)
_DEFAULT_BASIN_THRESHOLD = np.float64(0.25)


def _as_float64_matrix(points: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("points must be a 2D array")
    return arr


def detect_spectral_basins(points: list[list[float]] | np.ndarray) -> list[dict[str, object]]:
    """Deterministically cluster nearby spectral points into basins."""
    pts = _as_float64_matrix(points)
    if pts.shape[0] == 0:
        return []

    threshold = float(_DEFAULT_BASIN_THRESHOLD)
    centers: list[np.ndarray] = []
    members: list[list[int]] = []

    for idx, point in enumerate(pts):
        if not centers:
            centers.append(np.asarray(point, dtype=np.float64))
            members.append([int(idx)])
            continue

        center_array = np.asarray(centers, dtype=np.float64)
        deltas = center_array - point
        distances = np.sqrt(np.sum(np.square(deltas, dtype=np.float64), axis=1), dtype=np.float64)
        distances = np.maximum(distances, _DISTANCE_EPS)
        indices = np.arange(distances.shape[0], dtype=np.int64)
        order = np.lexsort((indices, distances))
        best_id = int(order[0])

        if float(distances[best_id]) <= threshold:
            members[best_id].append(int(idx))
        else:
            centers.append(np.asarray(point, dtype=np.float64))
            members.append([int(idx)])

    basin_records: list[dict[str, object]] = []
    for basin_idx, basin_members in enumerate(members):
        basin_points = pts[np.asarray(basin_members, dtype=np.int64)]
        centroid = np.mean(basin_points, axis=0, dtype=np.float64)
        deltas = basin_points - centroid
        distances = np.sqrt(np.sum(np.square(deltas, dtype=np.float64), axis=1), dtype=np.float64)
        distances = np.maximum(distances, _DISTANCE_EPS)
        radius = np.sqrt(np.mean(np.square(distances), dtype=np.float64), dtype=np.float64)
        basin_records.append(
            {
                "basin_id": int(basin_idx),
                "centroid": centroid.tolist(),
                "basin_radius": float(np.float64(radius)),
                "member_indices": [int(i) for i in basin_members],
            }
        )

    sorted_basins = sorted(
        basin_records,
        key=lambda rec: (
            tuple(float(np.float64(v)) for v in rec["centroid"]),
            int(rec["basin_id"]),
        ),
    )
    for new_id, basin in enumerate(sorted_basins):
        basin["basin_id"] = int(new_id)
    return sorted_basins


def build_basin_transition_graph(
    trajectory: list[list[float]] | np.ndarray,
    basins: list[dict[str, object]],
) -> dict[str, object]:
    """Build deterministic basin visit and transition statistics."""
    traj = _as_float64_matrix(trajectory)
    if traj.shape[0] == 0 or not basins:
        return {"basin_visit_counts": {}, "basin_transitions": []}

    ordered_basins = sorted(
        basins,
        key=lambda rec: (
            int(rec.get("basin_id", 0)),
            tuple(float(np.float64(v)) for v in rec.get("centroid", [])),
        ),
    )
    basin_ids = np.asarray([int(b["basin_id"]) for b in ordered_basins], dtype=np.int64)
    centroids = np.asarray([b["centroid"] for b in ordered_basins], dtype=np.float64)

    assignments: list[int] = []
    for point in traj:
        deltas = centroids - point
        distances = np.sqrt(np.sum(np.square(deltas, dtype=np.float64), axis=1), dtype=np.float64)
        distances = np.maximum(distances, _DISTANCE_EPS)
        order = np.lexsort((basin_ids, distances))
        assignments.append(int(basin_ids[int(order[0])]))

    visit_counts: dict[int, int] = {int(bid): 0 for bid in sorted(int(x) for x in basin_ids.tolist())}
    transitions: dict[tuple[int, int], int] = {}

    for basin_id in assignments:
        visit_counts[int(basin_id)] = int(visit_counts.get(int(basin_id), 0) + 1)
    for i in range(1, len(assignments)):
        prev_id = int(assignments[i - 1])
        curr_id = int(assignments[i])
        if curr_id != prev_id:
            key = (prev_id, curr_id)
            transitions[key] = int(transitions.get(key, 0) + 1)

    transition_records = [
        {"from_basin_id": int(src), "to_basin_id": int(dst), "count": int(count)}
        for (src, dst), count in sorted(transitions.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    sorted_visits = {int(k): int(visit_counts[k]) for k in sorted(visit_counts.keys())}
    return {
        "basin_visit_counts": sorted_visits,
        "basin_transitions": transition_records,
    }


def identify_spectral_basins(
    trajectory: np.ndarray,
    threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition trajectory states into distance-threshold spectral basins."""
    traj = _as_float64_matrix(trajectory)
    if traj.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, traj.shape[1]), dtype=np.float64)

    basins: list[np.ndarray] = []
    assignments: list[int] = []
    distance_threshold = float(np.float64(threshold))

    for state in traj:
        if basins:
            centers = np.asarray(basins, dtype=np.float64)
            deltas = centers - state
            distances = np.sqrt(np.sum(np.square(deltas, dtype=np.float64), axis=1), dtype=np.float64)
            distances = np.maximum(distances, _DISTANCE_EPS)
            indices = np.arange(distances.shape[0], dtype=np.int64)
            order = np.lexsort((indices, distances))
            best = int(order[0])
            if float(distances[best]) <= distance_threshold:
                assignments.append(best)
                continue

        basins.append(np.asarray(state, dtype=np.float64))
        assignments.append(len(basins) - 1)

    return np.asarray(assignments, dtype=np.int64), np.asarray(basins, dtype=np.float64)
