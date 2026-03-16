from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_geometry import (
    estimate_basin_geometry,
    estimate_local_curvature,
    spectral_distance,
    trajectory_arc_length,
)


def test_spectral_distance_euclidean() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 3.0, 7.0], dtype=np.float64)
    assert spectral_distance(a, b) == np.sqrt(17.0)


def test_trajectory_arc_length_deterministic() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 4.0],
            [0.0, 6.0, 8.0],
        ],
        dtype=np.float64,
    )
    expected = np.float64(10.0)
    assert trajectory_arc_length(points) == float(expected)
    assert trajectory_arc_length(points) == float(expected)


def test_local_curvature_deterministic() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    curv1 = estimate_local_curvature(points)
    curv2 = estimate_local_curvature(points)
    assert curv1 == curv2
    assert curv1 == [np.pi / 2.0, np.pi / 2.0]


def test_basin_geometry_metrics_reproducible() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    m1 = estimate_basin_geometry(points)
    m2 = estimate_basin_geometry(points)
    assert m1 == m2
    assert set(m1.keys()) == {
        "basin_radius_estimate",
        "mean_step_length",
        "mean_curvature",
        "spectral_dispersion",
    }
