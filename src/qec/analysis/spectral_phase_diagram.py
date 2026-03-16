"""Deterministic spectral phase diagram generation utilities (v51.0.0)."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np


_ROUND = 12
_FALLBACK_PNG_1X1 = bytes(
    [
        137, 80, 78, 71, 13, 10, 26, 10,
        0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1,
        8, 6, 0, 0, 0, 31, 21, 196,
        137, 0, 0, 0, 13, 73, 68, 65,
        84, 120, 156, 99, 248, 255, 255, 255,
        127, 0, 9, 251, 3, 253, 42, 134,
        227, 182, 0, 0, 0, 0, 73, 69,
        78, 68, 174, 66, 96, 130,
    ]
)


def _write_fallback_png(output_path: str) -> str:
    """Write deterministic 1x1 PNG fallback when matplotlib is unavailable."""
    with open(output_path, "wb") as f:
        f.write(_FALLBACK_PNG_1X1)
    return output_path


def _unique_archive_entries(archive: dict[str, Any]) -> list[dict[str, Any]]:
    categories = archive.get("categories", {})
    seen: dict[str, dict[str, Any]] = {}
    for category_name in sorted(categories.keys()):
        for entry in categories.get(category_name, []):
            candidate_id = str(entry.get("candidate_id", ""))
            if candidate_id:
                seen[candidate_id] = entry
    ordered_ids = sorted(seen.keys())
    return [seen[cid] for cid in ordered_ids]


def _objective_value(objectives: dict[str, Any], key: str) -> float:
    return float(np.float64(objectives.get(key, 0.0)))


def _stability_metric(objectives: dict[str, Any]) -> float:
    if "bp_stability_score" in objectives:
        return _objective_value(objectives, "bp_stability_score")
    if "composite_threshold_metric" in objectives:
        return _objective_value(objectives, "composite_threshold_metric")

    threshold_estimate = _objective_value(
        objectives,
        "threshold_estimate",
    )
    if threshold_estimate == 0.0:
        threshold_estimate = _objective_value(objectives, "threshold")
    if threshold_estimate == 0.0:
        threshold_estimate = _objective_value(objectives, "bethe_margin")

    instability = _objective_value(objectives, "instability_score")
    return float(np.float64(threshold_estimate - instability))


def build_phase_diagram_dataset(archive: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract deterministic float64 vectors for phase-diagram construction."""
    entries = _unique_archive_entries(archive)
    if not entries:
        return {
            "x": np.zeros((0,), dtype=np.float64),
            "y": np.zeros((0,), dtype=np.float64),
            "z": np.zeros((0,), dtype=np.float64),
        }

    x_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []

    for entry in entries:
        objectives = entry.get("objectives", {})
        x_vals.append(_objective_value(objectives, "spectral_radius"))
        y_vals.append(_objective_value(objectives, "bethe_min_eigenvalue"))
        z_vals.append(_stability_metric(objectives))

    return {
        "x": np.asarray(x_vals, dtype=np.float64),
        "y": np.asarray(y_vals, dtype=np.float64),
        "z": np.asarray(z_vals, dtype=np.float64),
    }


def construct_phase_grid(
    dataset: dict[str, np.ndarray],
    grid_resolution: int = 50,
) -> dict[str, np.ndarray]:
    """Construct deterministic nearest-neighbor phase grid from dataset."""
    res = max(2, int(grid_resolution))
    x = np.asarray(dataset.get("x", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    z = np.asarray(dataset.get("z", np.zeros((0,), dtype=np.float64)), dtype=np.float64)

    if x.size == 0 or y.size == 0 or z.size == 0:
        axis = np.linspace(0.0, 1.0, res, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
        return {
            "grid_x": grid_x.astype(np.float64, copy=False),
            "grid_y": grid_y.astype(np.float64, copy=False),
            "grid_z": np.zeros_like(grid_x, dtype=np.float64),
        }

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_min == x_max:
        x_max = float(np.float64(x_min + 1e-9))
    if y_min == y_max:
        y_max = float(np.float64(y_min + 1e-9))

    x_axis = np.linspace(x_min, x_max, res, dtype=np.float64)
    y_axis = np.linspace(y_min, y_max, res, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis, indexing="xy")

    samples = np.column_stack((x, y)).astype(np.float64, copy=False)
    grid_z = np.zeros_like(grid_x, dtype=np.float64)
    index = np.arange(samples.shape[0], dtype=np.int64)

    for row in range(grid_x.shape[0]):
        for col in range(grid_x.shape[1]):
            gx = grid_x[row, col]
            gy = grid_y[row, col]
            d2 = (samples[:, 0] - gx) ** 2 + (samples[:, 1] - gy) ** 2
            order = np.lexsort((index, d2))
            grid_z[row, col] = z[int(order[0])]

    return {
        "grid_x": grid_x.astype(np.float64, copy=False),
        "grid_y": grid_y.astype(np.float64, copy=False),
        "grid_z": grid_z.astype(np.float64, copy=False),
    }


def estimate_stability_surface(
    dataset: dict[str, np.ndarray],
    grid: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Estimate deterministic cell-wise average stability over the phase grid."""
    x = np.asarray(dataset.get("x", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    y = np.asarray(dataset.get("y", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    z = np.asarray(dataset.get("z", np.zeros((0,), dtype=np.float64)), dtype=np.float64)

    grid_x = np.asarray(grid.get("grid_x"), dtype=np.float64)
    grid_y = np.asarray(grid.get("grid_y"), dtype=np.float64)
    grid_z = np.asarray(grid.get("grid_z"), dtype=np.float64)

    phase_surface = grid_z.astype(np.float64, copy=True)
    counts = np.zeros_like(phase_surface, dtype=np.float64)
    sums = np.zeros_like(phase_surface, dtype=np.float64)

    if x.size > 0 and y.size > 0 and z.size > 0:
        x_axis = grid_x[0, :]
        y_axis = grid_y[:, 0]
        for i in range(x.size):
            xi = int(np.argmin(np.abs(x_axis - x[i])))
            yi = int(np.argmin(np.abs(y_axis - y[i])))
            sums[yi, xi] += z[i]
            counts[yi, xi] += 1.0

        mask = counts > 0.0
        phase_surface[mask] = sums[mask] / counts[mask]

    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "phase_surface": phase_surface.astype(np.float64, copy=False),
        "sample_counts": counts.astype(np.float64, copy=False),
    }


def detect_phase_boundaries(memory: Any) -> list[dict[str, float]]:
    """Detect deterministic boundary segments from landscape-memory centers."""
    if memory is None:
        return []

    if hasattr(memory, "centers"):
        centers = np.asarray(memory.centers(), dtype=np.float64)
    elif hasattr(memory, "as_array"):
        centers = np.asarray(memory.as_array(), dtype=np.float64)
    else:
        centers = np.asarray(memory, dtype=np.float64)

    if centers.ndim != 2 or centers.shape[0] < 2 or centers.shape[1] < 2:
        return []

    points = centers[:, :2].astype(np.float64, copy=False)
    index = np.arange(points.shape[0], dtype=np.int64)
    order = np.lexsort((index, points[:, 1], points[:, 0]))
    sorted_points = points[order]

    if sorted_points.shape[0] < 2:
        return []

    deltas = sorted_points[1:] - sorted_points[:-1]
    distances = np.sqrt(np.sum(deltas * deltas, axis=1, dtype=np.float64), dtype=np.float64)
    if distances.size == 0:
        return []

    threshold = float(np.median(distances))
    boundaries: list[dict[str, float]] = []
    for i in range(distances.size):
        if float(distances[i]) >= threshold:
            p0 = sorted_points[i]
            p1 = sorted_points[i + 1]
            boundaries.append(
                {
                    "x0": round(float(np.float64(p0[0])), _ROUND),
                    "y0": round(float(np.float64(p0[1])), _ROUND),
                    "x1": round(float(np.float64(p1[0])), _ROUND),
                    "y1": round(float(np.float64(p1[1])), _ROUND),
                }
            )
    return boundaries


def generate_phase_heatmap(
    phase_surface: dict[str, np.ndarray | list[dict[str, float]]],
    output_path: str = "spectral_phase_diagram.png",
) -> str:
    """Generate deterministic spectral phase heatmap image artifact."""
    if importlib.util.find_spec("matplotlib.pyplot") is None:
        return _write_fallback_png(output_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return _write_fallback_png(output_path)

    grid_x = np.asarray(phase_surface.get("grid_x"), dtype=np.float64)
    grid_y = np.asarray(phase_surface.get("grid_y"), dtype=np.float64)
    surface = np.asarray(phase_surface.get("phase_surface"), dtype=np.float64)
    boundaries = list(phase_surface.get("phase_boundaries", []))

    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=120)
    extent = [
        float(np.min(grid_x)),
        float(np.max(grid_x)),
        float(np.min(grid_y)),
        float(np.max(grid_y)),
    ]
    image = ax.imshow(
        surface,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Decoding stability", fontsize=10)

    for boundary in boundaries:
        ax.plot(
            [float(boundary["x0"]), float(boundary["x1"])],
            [float(boundary["y0"]), float(boundary["y1"])],
            color="white",
            linewidth=1.0,
            linestyle="-",
        )

    ax.set_xlabel("Spectral radius")
    ax.set_ylabel("Bethe eigenvalue")
    ax.set_title("Spectral Phase Diagram")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
