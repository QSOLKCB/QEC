"""Deterministic spectral phase-map reconstruction utilities."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


_FALLBACK_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\x0b\xe7\x02\x9d"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _as_point(value: Any, dims: int | None = None) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    if dims is not None and arr.size < dims:
        return None
    if dims is not None:
        arr = arr[:dims]
    return arr.astype(np.float64, copy=False)


def _ordered_basins(basins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, basin in enumerate(basins):
        members = [int(m) for m in basin.get("members", [])]
        centroid = _as_point(basin.get("centroid", []))
        if centroid is None:
            continue
        normalized.append(
            {
                "_source_index": int(idx),
                "centroid": centroid,
                "basin_members": sorted(set(members)),
            }
        )
    normalized.sort(
        key=lambda rec: (
            tuple(float(np.float64(v)) for v in rec["centroid"]),
            int(rec["_source_index"]),
        )
    )
    return normalized


def _ridge_membership_for_phase(centroid: np.ndarray, ridges: list[dict[str, Any]]) -> list[int]:
    memberships: list[tuple[float, int, int]] = []
    for ridx, ridge in enumerate(ridges):
        location = _as_point(ridge.get("location", []), dims=int(centroid.size))
        if location is None:
            continue
        distance = float(np.float64(np.linalg.norm(location - centroid)))
        memberships.append((distance, int(ridge.get("ridge_id", ridx)), int(ridx)))
    memberships.sort(key=lambda rec: (rec[0], rec[1], rec[2]))
    return [int(rec[1]) for rec in memberships[:2]]


def label_phases(basins: list[dict[str, Any]], ridges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign deterministic phase IDs from basin centroids and nearby ridges."""
    ordered = _ordered_basins(basins)
    phases: list[dict[str, Any]] = []
    for phase_id, basin in enumerate(ordered):
        centroid = basin["centroid"].astype(np.float64, copy=False)
        phases.append(
            {
                "phase_id": int(phase_id),
                "centroid": centroid.tolist(),
                "basin_members": list(basin["basin_members"]),
                "boundary_ridges": _ridge_membership_for_phase(centroid, ridges),
            }
        )
    return phases


def _build_phase_adjacency(phase_regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ridge_to_phases: dict[int, list[int]] = {}
    for rec in phase_regions:
        phase_id = int(rec.get("phase_id", 0))
        for ridge_id in rec.get("boundary_ridges", []):
            rid = int(ridge_id)
            ridge_to_phases.setdefault(rid, []).append(phase_id)

    edges: list[dict[str, Any]] = []
    for ridge_id in sorted(ridge_to_phases.keys()):
        attached = sorted(set(int(v) for v in ridge_to_phases[ridge_id]))
        for i in range(len(attached)):
            for j in range(i + 1, len(attached)):
                edges.append(
                    {
                        "from_phase_id": int(attached[i]),
                        "to_phase_id": int(attached[j]),
                        "ridge_id": int(ridge_id),
                    }
                )
    edges.sort(
        key=lambda rec: (
            int(rec["from_phase_id"]),
            int(rec["to_phase_id"]),
            int(rec["ridge_id"]),
        )
    )
    return edges


def _assign_point_to_phase(point: np.ndarray, phase_regions: list[dict[str, Any]]) -> int | None:
    if not phase_regions:
        return None
    distances: list[tuple[float, int]] = []
    for rec in phase_regions:
        phase_id = int(rec.get("phase_id", 0))
        centroid = _as_point(rec.get("centroid", []), dims=int(point.size))
        if centroid is None:
            continue
        distances.append((float(np.float64(np.linalg.norm(point - centroid))), phase_id))
    if not distances:
        return None
    distances.sort(key=lambda rec: (rec[0], rec[1]))
    return int(distances[0][1])


def _trajectory_segments(
    trajectory: list[list[float]] | np.ndarray,
    phase_regions: list[dict[str, Any]],
    dims: int,
) -> list[dict[str, Any]]:
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < dims:
        return []
    points = arr[:, :dims].astype(np.float64, copy=False)

    segments: list[dict[str, Any]] = []
    for idx in range(points.shape[0] - 1):
        phase_id = _assign_point_to_phase(points[idx], phase_regions)
        if phase_id is None:
            continue
        segments.append(
            {
                "segment_id": int(idx),
                "phase_id": int(phase_id),
                "start_index": int(idx),
                "end_index": int(idx + 1),
                "start": points[idx].tolist(),
                "end": points[idx + 1].tolist(),
            }
        )
    segments.sort(key=lambda rec: (int(rec["start_index"]), int(rec["segment_id"])))
    for seg_id, rec in enumerate(segments):
        rec["segment_id"] = int(seg_id)
    return segments


def construct_phase_map(
    basins: list[dict[str, Any]],
    ridges: list[dict[str, Any]],
    phase_surface: dict[str, Any] | None,
    trajectory: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """Construct deterministic phase-map artifacts from basin/ridge/surface/trajectory data."""
    phase_regions = label_phases(basins, ridges)
    phase_boundaries = sorted(
        [
            {
                "ridge_id": int(rec.get("ridge_id", idx)),
                "location": np.asarray(rec.get("location", []), dtype=np.float64).reshape(-1).tolist(),
            }
            for idx, rec in enumerate(ridges)
            if np.asarray(rec.get("location", []), dtype=np.float64).size > 0
        ],
        key=lambda rec: (
            tuple(float(np.float64(v)) for v in rec["location"]),
            int(rec["ridge_id"]),
        ),
    )
    phase_adjacency = _build_phase_adjacency(phase_regions)

    dims = 2
    if phase_regions:
        dims = max(1, len(phase_regions[0].get("centroid", [])))
    elif phase_boundaries:
        dims = max(1, len(phase_boundaries[0].get("location", [])))
    elif isinstance(phase_surface, dict):
        gx = np.asarray(phase_surface.get("grid_x", []), dtype=np.float64)
        dims = max(1, 1 if gx.size > 0 else 2)

    trajectory_segments = _trajectory_segments(trajectory, phase_regions, dims=dims)

    return {
        "phase_regions": phase_regions,
        "phase_boundaries": phase_boundaries,
        "phase_adjacency": phase_adjacency,
        "trajectory_segments": trajectory_segments,
    }


def render_phase_map(phase_map: dict[str, Any], output_path: str) -> dict[str, Any]:
    """Render deterministic phase-map PNG; use fixed fallback when matplotlib is unavailable."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        _has_mpl = importlib.util.find_spec("matplotlib") is not None
    except (ModuleNotFoundError, ValueError):
        _has_mpl = False
    if not _has_mpl:
        out.write_bytes(_FALLBACK_PNG_1X1)
        return {"output_path": str(out), "backend": "fallback"}

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    centroids = np.asarray(
        [rec.get("centroid", []) for rec in phase_map.get("phase_regions", [])],
        dtype=np.float64,
    )
    boundaries = np.asarray(
        [rec.get("location", []) for rec in phase_map.get("phase_boundaries", [])],
        dtype=np.float64,
    )

    if centroids.ndim == 2 and centroids.shape[1] >= 2 and centroids.shape[0] > 0:
        ax.scatter(centroids[:, 0], centroids[:, 1], c="tab:blue", marker="o", s=24)
    if boundaries.ndim == 2 and boundaries.shape[1] >= 2 and boundaries.shape[0] > 0:
        ax.scatter(boundaries[:, 0], boundaries[:, 1], c="tab:red", marker="x", s=28)

    trajectory_points: list[list[float]] = []
    for seg in sorted(
        phase_map.get("trajectory_segments", []),
        key=lambda rec: (int(rec.get("start_index", 0)), int(rec.get("segment_id", 0))),
    ):
        start = _as_point(seg.get("start", []), dims=2)
        if start is not None:
            trajectory_points.append(start.tolist())
    if phase_map.get("trajectory_segments"):
        last_end = _as_point(phase_map["trajectory_segments"][-1].get("end", []), dims=2)
        if last_end is not None:
            trajectory_points.append(last_end.tolist())
    if trajectory_points:
        traj = np.asarray(trajectory_points, dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], color="tab:green", linewidth=1.25)

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    if centroids.ndim == 2 and centroids.shape[0] > 0 and centroids.shape[1] >= 2:
        all_x.append(centroids[:, 0])
        all_y.append(centroids[:, 1])
    if boundaries.ndim == 2 and boundaries.shape[0] > 0 and boundaries.shape[1] >= 2:
        all_x.append(boundaries[:, 0])
        all_y.append(boundaries[:, 1])
    if trajectory_points:
        traj = np.asarray(trajectory_points, dtype=np.float64)
        all_x.append(traj[:, 0])
        all_y.append(traj[:, 1])

    if all_x:
        xs = np.concatenate(all_x).astype(np.float64, copy=False)
        ys = np.concatenate(all_y).astype(np.float64, copy=False)
        min_x = float(np.min(xs))
        max_x = float(np.max(xs))
        min_y = float(np.min(ys))
        max_y = float(np.max(ys))
        pad_x = float(np.float64(0.05 * (max_x - min_x))) if max_x > min_x else 0.5
        pad_y = float(np.float64(0.05 * (max_y - min_y))) if max_y > min_y else 0.5
        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_xlabel("Spectral Axis 1")
    ax.set_ylabel("Spectral Axis 2")
    ax.set_title("Spectral Phase Map")
    fig.tight_layout()
    fig.savefig(str(out), dpi=120, format="png", metadata={"Software": "matplotlib", "Creation Time": ""})
    plt.close(fig)
    return {"output_path": str(out), "backend": "matplotlib"}
