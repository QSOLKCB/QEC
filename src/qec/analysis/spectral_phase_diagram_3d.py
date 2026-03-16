"""Deterministic 3D spectral phase-diagram rendering utilities."""

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


def _to_float64_mesh(phase_surface: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid_x = np.asarray(phase_surface.get("grid_x", []), dtype=np.float64)
    grid_y = np.asarray(phase_surface.get("grid_y", []), dtype=np.float64)
    grid_z = np.asarray(phase_surface.get("grid_z", []), dtype=np.float64)

    if grid_z.ndim != 2:
        raise ValueError("phase_surface['grid_z'] must be a 2D grid")

    if grid_x.ndim == 1 and grid_y.ndim == 1 and grid_x.size > 0 and grid_y.size > 0:
        X, Y = np.meshgrid(grid_x, grid_y, indexing="ij")
    elif grid_x.shape == grid_z.shape and grid_y.shape == grid_z.shape:
        X = np.asarray(grid_x, dtype=np.float64)
        Y = np.asarray(grid_y, dtype=np.float64)
    else:
        raise ValueError("phase_surface grid_x/grid_y must be 1D axes or 2D meshes matching grid_z")

    if X.shape != grid_z.shape or Y.shape != grid_z.shape:
        raise ValueError("phase_surface mesh shapes must match grid_z")

    return X, Y, grid_z


def _normalize_targets(planned_targets: list[np.ndarray | list[float]] | None) -> tuple[np.ndarray, list[list[float]]]:
    if planned_targets is None:
        return np.zeros((0, 2), dtype=np.float64), []

    rows: list[tuple[float, float, int]] = []
    for i, target in enumerate(planned_targets):
        arr = np.asarray(target, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            continue
        rows.append((float(arr[0]), float(arr[1]), int(i)))

    if not rows:
        return np.zeros((0, 2), dtype=np.float64), []

    x = np.asarray([r[0] for r in rows], dtype=np.float64)
    y = np.asarray([r[1] for r in rows], dtype=np.float64)
    idx = np.asarray([r[2] for r in rows], dtype=np.int64)
    order = np.lexsort((idx, y, x))
    sorted_xy = np.column_stack((x[order], y[order])).astype(np.float64, copy=False)
    points = [[float(v0), float(v1)] for v0, v1 in sorted_xy.tolist()]
    return sorted_xy, points


def generate_phase_surface_3d(
    phase_grid: Any,
    phase_surface: dict[str, Any],
    output_path: str,
    planned_targets: list[np.ndarray | list[float]] | None = None,
    title: str = "Spectral Phase Diagram",
) -> dict[str, Any]:
    """Generate a deterministic 3D phase-surface artifact with optional overlays."""
    del phase_grid
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mpl_available = importlib.util.find_spec("matplotlib.pyplot") is not None
    if not mpl_available:
        out.write_bytes(_FALLBACK_PNG_1X1)
        _, target_points = _normalize_targets(planned_targets)
        return {
            "surface_path": str(out),
            "num_targets": len(target_points),
            "target_points": target_points,
        }

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    X, Y, Z = _to_float64_mesh(phase_surface)
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    z = np.asarray(Z, dtype=np.float64)

    target_xy, target_points = _normalize_targets(planned_targets)
    target_z = np.zeros(target_xy.shape[0], dtype=np.float64)
    if target_xy.shape[0] > 0:
        x_vals = x[:, 0]
        y_vals = y[0, :]
        for i in range(target_xy.shape[0]):
            xi = int(np.argmin(np.abs(x_vals - target_xy[i, 0])))
            yi = int(np.argmin(np.abs(y_vals - target_xy[i, 1])))
            target_z[i] = float(z[xi, yi])

    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    norm = matplotlib.colors.Normalize(vmin=float(np.min(z)), vmax=float(np.max(z)))

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        cmap="viridis",
        norm=norm,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )

    if target_xy.shape[0] > 0:
        ax.scatter(
            target_xy[:, 0],
            target_xy[:, 1],
            target_z,
            c="red",
            s=24,
            marker="o",
            depthshade=False,
        )

    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)), float(np.max(y)))
    ax.set_zlim(float(np.min(z)), float(np.max(z)))
    ax.view_init(elev=30.0, azim=135.0)

    ax.set_xlabel("Spectral Radius")
    ax.set_ylabel("Bethe Eigenvalue")
    ax.set_zlabel("Stability Surface")
    ax.set_title(str(title))

    fig.tight_layout()
    fig.savefig(
        str(out),
        dpi=120,
        format="png",
        metadata={"Software": "matplotlib", "Creation Time": ""},
    )
    plt.close(fig)

    return {
        "surface_path": str(out),
        "num_targets": int(target_xy.shape[0]),
        "target_points": target_points,
    }
