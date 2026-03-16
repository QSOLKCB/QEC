"""Deterministic optional 3D spectral phase-surface rendering."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

import numpy as np


def _as_points(points: list[np.ndarray] | np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    return arr[:, :3].astype(np.float64, copy=False)


def generate_phase_surface_3d(
    surface_points: list[np.ndarray] | np.ndarray,
    *,
    output_path: str | None = None,
    trajectory_points: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Render deterministic 3D surface + optional trajectory line when matplotlib exists."""
    pts = _as_points(surface_points)
    traj = _as_points(trajectory_points)

    if traj.shape[0] > 1:
        order = np.lexsort((np.arange(traj.shape[0], dtype=np.int64), traj[:, 0]))
        traj = traj[order]

    result: dict[str, Any] = {
        "rendered": False,
        "num_surface_points": int(pts.shape[0]),
        "trajectory_length": int(traj.shape[0]),
        "output_path": output_path,
    }

    if importlib.util.find_spec("matplotlib") is None or importlib.util.find_spec("matplotlib.pyplot") is None:
        return result

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    if pts.shape[0] > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=14, alpha=0.8)
    if traj.shape[0] > 1:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="black", linewidth=1.5, alpha=0.9, zorder=5)
    ax.set_xlabel("spectral_radius")
    ax.set_ylabel("bethe_min_eigenvalue")
    ax.set_zlabel("bp_stability")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        result["rendered"] = True
    plt.close(fig)
    return result
