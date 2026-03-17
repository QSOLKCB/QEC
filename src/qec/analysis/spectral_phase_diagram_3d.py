"""Deterministic 3D spectral phase-surface rendering utilities."""

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


def _normalize_trajectory(trajectory_points: Any) -> np.ndarray:
    if trajectory_points is None:
        return np.zeros((0, 3), dtype=np.float64)
    rows: list[tuple[float, float, float, int]] = []
    for i, point in enumerate(trajectory_points):
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            continue
        rows.append((float(arr[0]), float(arr[1]), float(arr[2]), int(i)))
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    y = np.asarray([r[1] for r in rows], dtype=np.float64)
    z = np.asarray([r[2] for r in rows], dtype=np.float64)
    order = np.lexsort((z, y, x))
    return np.column_stack((x[order], y[order], z[order])).astype(np.float64, copy=False)


def _surface_to_mesh(surface_points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(surface_points, dict):
        gx = np.asarray(surface_points.get("grid_x", []), dtype=np.float64)
        gy = np.asarray(surface_points.get("grid_y", []), dtype=np.float64)
        gz = np.asarray(surface_points.get("grid_z", surface_points.get("phase_surface", [])), dtype=np.float64)
        if gx.ndim == 1 and gy.ndim == 1 and gz.ndim == 2:
            X, Y = np.meshgrid(gx, gy, indexing="ij")
            return X, Y, gz
        if gx.shape == gy.shape == gz.shape and gz.ndim == 2:
            return gx, gy, gz
    points = np.asarray(surface_points, dtype=np.float64)
    if points.ndim == 2 and points.shape[1] >= 3:
        xs = np.unique(points[:, 0])
        ys = np.unique(points[:, 1])
        if xs.size * ys.size == points.shape[0]:
            Z = np.zeros((xs.size, ys.size), dtype=np.float64)
            for row in points:
                xi = int(np.where(xs == row[0])[0][0])
                yi = int(np.where(ys == row[1])[0][0])
                Z[xi, yi] = np.float64(row[2])
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            return X, Y, Z
    return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)


def generate_phase_surface_3d(
    surface_points: Any,
    trajectory_points: Any = None,
    output_path: str | None = None,
    **legacy_kwargs: Any,
) -> dict[str, Any]:
    """Generate deterministic phase-surface PNG with optional trajectory overlay."""
    if output_path is None and isinstance(trajectory_points, str):
        output_path = trajectory_points
        trajectory_points = None

    # Backward-compatibility: generate_phase_surface_3d(None, phase_surface, output_path, planned_targets=...)
    if surface_points is None and isinstance(trajectory_points, dict):
        surface_points = trajectory_points
        trajectory_points = legacy_kwargs.get("planned_targets")

    if output_path is None:
        output_path = "artifacts/phase_diagram_3d.png"

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    trajectory = _normalize_trajectory(trajectory_points)
    X, Y, Z = _surface_to_mesh(surface_points)

    try:
        _has_mpl = importlib.util.find_spec("matplotlib") is not None
    except (ModuleNotFoundError, ValueError):
        _has_mpl = False
    if not _has_mpl:
        out.write_bytes(_FALLBACK_PNG_1X1)
        return {
            "surface_path": str(out),
            "trajectory_points": trajectory.tolist(),
            "trajectory_length": int(trajectory.shape[0]),
            "num_targets": int(trajectory.shape[0]),
            "target_points": [[float(v[0]), float(v[1])] for v in trajectory],
        }

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    if Z.size > 0:
        norm = matplotlib.colors.Normalize(vmin=float(np.min(Z)), vmax=float(np.max(Z)))
        ax.plot_surface(X, Y, Z, cmap="viridis", norm=norm, linewidth=0.0, antialiased=False, shade=False)

    if trajectory.shape[0] > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="red", linewidth=1.5)

    all_x = trajectory[:, 0] if (trajectory.shape[0] > 0 and Z.size == 0) else (np.concatenate((X.reshape(-1), trajectory[:, 0])) if trajectory.shape[0] > 0 and Z.size > 0 else X.reshape(-1))
    all_y = trajectory[:, 1] if (trajectory.shape[0] > 0 and Z.size == 0) else (np.concatenate((Y.reshape(-1), trajectory[:, 1])) if trajectory.shape[0] > 0 and Z.size > 0 else Y.reshape(-1))
    all_z = trajectory[:, 2] if (trajectory.shape[0] > 0 and Z.size == 0) else (np.concatenate((Z.reshape(-1), trajectory[:, 2])) if trajectory.shape[0] > 0 and Z.size > 0 else Z.reshape(-1))
    if all_x.size > 0:
        ax.set_xlim(float(np.min(all_x)), float(np.max(all_x)))
        ax.set_ylim(float(np.min(all_y)), float(np.max(all_y)))
        ax.set_zlim(float(np.min(all_z)), float(np.max(all_z)))

    ax.view_init(elev=30.0, azim=135.0)
    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Bethe Eigenvalue")
    ax.set_zlabel("Stability Surface")
    fig.tight_layout()
    fig.savefig(str(out), dpi=120, format="png", metadata={"Software": "matplotlib", "Creation Time": ""})
    plt.close(fig)

    return {
        "surface_path": str(out),
        "trajectory_points": trajectory.tolist(),
        "trajectory_length": int(trajectory.shape[0]),
        "num_targets": int(trajectory.shape[0]),
        "target_points": [[float(v[0]), float(v[1])] for v in trajectory],
    }
