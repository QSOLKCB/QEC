"""Deterministic spectral ridge detection and ridge-topology utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


_DISTANCE_EPS = np.float64(1e-12)


def _as_float64_matrix(points: list[list[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("points must be a 2D array")
    return arr


def detect_spectral_ridges(points: list[list[float]] | np.ndarray) -> list[dict[str, Any]]:
    """Detect deterministic ridge points from a spectral trajectory."""
    pts = _as_float64_matrix(points)
    n_points = int(pts.shape[0])
    if n_points == 0:
        return []

    gradients = np.zeros((n_points, pts.shape[1]), dtype=np.float64)
    curvature_vectors = np.zeros((n_points, pts.shape[1]), dtype=np.float64)

    for idx in range(n_points):
        left = pts[idx - 1] if idx > 0 else pts[idx]
        right = pts[idx + 1] if idx < n_points - 1 else pts[idx]
        gradients[idx] = (right - left) / np.float64(2.0)
        curvature_vectors[idx] = right - np.float64(2.0) * pts[idx] + left

    gradient_magnitudes = np.linalg.norm(gradients, axis=1).astype(np.float64, copy=False)
    curvatures = np.linalg.norm(curvature_vectors, axis=1).astype(np.float64, copy=False)

    curvature_threshold = float(
        np.float64(np.mean(curvatures, dtype=np.float64) + np.std(curvatures, dtype=np.float64))
    )
    ridge_indices = np.flatnonzero(curvatures >= np.float64(curvature_threshold)).astype(np.int64, copy=False)
    if ridge_indices.size == 0:
        return []

    lex_keys = [pts[ridge_indices, dim] for dim in range(pts.shape[1] - 1, -1, -1)]
    order = np.lexsort(tuple(lex_keys + [ridge_indices]))
    ordered_indices = ridge_indices[order]

    ridges: list[dict[str, Any]] = []
    for ridge_id, point_index in enumerate(ordered_indices.tolist()):
        ridges.append(
            {
                "ridge_id": int(ridge_id),
                "location": pts[int(point_index)].astype(np.float64, copy=False).tolist(),
                "curvature": float(np.float64(curvatures[int(point_index)])),
                "gradient_magnitude": float(np.float64(gradient_magnitudes[int(point_index)])),
            }
        )
    return ridges


def build_ridge_graph(ridges: list[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic proximity graph among detected ridges."""
    if not ridges:
        return {"ridge_nodes": [], "ridge_edges": []}

    ordered_nodes = sorted(
        [{
            "ridge_id": int(rec.get("ridge_id", 0)),
            "location": np.asarray(rec.get("location", []), dtype=np.float64).tolist(),
            "curvature": float(np.float64(rec.get("curvature", 0.0))),
            "gradient_magnitude": float(np.float64(rec.get("gradient_magnitude", 0.0))),
        } for rec in ridges],
        key=lambda rec: (
            tuple(float(np.float64(v)) for v in rec["location"]),
            int(rec["ridge_id"]),
        ),
    )
    for idx, node in enumerate(ordered_nodes):
        node["ridge_id"] = int(idx)

    locations = np.asarray([n["location"] for n in ordered_nodes], dtype=np.float64)
    n_nodes = int(locations.shape[0])
    if n_nodes <= 1:
        return {"ridge_nodes": ordered_nodes, "ridge_edges": []}

    pair_distances: list[float] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            pair_distances.append(float(np.float64(np.linalg.norm(locations[i] - locations[j]))))

    threshold = float(np.float64(np.median(np.asarray(pair_distances, dtype=np.float64)))) if pair_distances else 0.0
    edges: list[dict[str, Any]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            distance = float(np.float64(np.linalg.norm(locations[i] - locations[j])))
            if distance <= threshold + float(_DISTANCE_EPS):
                edges.append(
                    {
                        "from_ridge_id": int(i),
                        "to_ridge_id": int(j),
                        "distance": float(np.float64(distance)),
                    }
                )

    edges = sorted(
        edges,
        key=lambda rec: (
            int(rec["from_ridge_id"]),
            int(rec["to_ridge_id"]),
            float(np.float64(rec["distance"])),
        ),
    )
    return {"ridge_nodes": ordered_nodes, "ridge_edges": edges}


def map_ridges_to_basins(ridges: list[dict[str, Any]], basins: list[dict[str, Any]]) -> dict[str, Any]:
    """Map ridges to nearby basins and emit deterministic boundary segments."""
    if not ridges or not basins:
        return {"basin_boundary_segments": []}

    ordered_basins = sorted(
        basins,
        key=lambda rec: (
            tuple(float(np.float64(v)) for v in rec.get("centroid", [])),
            int(rec.get("basin_id", 0)),
        ),
    )
    segments: list[dict[str, Any]] = []

    for ridge in sorted(ridges, key=lambda rec: int(rec.get("ridge_id", 0))):
        location = np.asarray(ridge.get("location", []), dtype=np.float64)
        adjacent: list[int] = []
        for basin in ordered_basins:
            basin_id = int(basin.get("basin_id", 0))
            centroid = np.asarray(basin.get("centroid", []), dtype=np.float64)
            radius = float(np.float64(basin.get("basin_radius", 0.0)))
            if centroid.shape != location.shape:
                continue
            distance = float(np.float64(np.linalg.norm(location - centroid)))
            if distance <= radius + float(_DISTANCE_EPS):
                adjacent.append(basin_id)

        adjacent_sorted = sorted(set(int(x) for x in adjacent))
        if len(adjacent_sorted) >= 2:
            segments.append(
                {
                    "ridge_id": int(ridge.get("ridge_id", 0)),
                    "adjacent_basins": adjacent_sorted,
                    "location": location.tolist(),
                }
            )

    segments = sorted(
        segments,
        key=lambda rec: (
            tuple(int(v) for v in rec["adjacent_basins"]),
            tuple(float(np.float64(v)) for v in rec["location"]),
            int(rec["ridge_id"]),
        ),
    )
    return {"basin_boundary_segments": segments}

