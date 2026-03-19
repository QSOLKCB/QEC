from __future__ import annotations

import numpy as np

from qec.analysis.spectral_basins import (
    build_basin_transition_graph,
    detect_spectral_basins,
)


def test_detect_spectral_basins_deterministic() -> None:
    points = [
        [0.10, 0.20],
        [0.11, 0.21],
        [1.00, 1.10],
        [1.01, 1.09],
    ]
    b1 = detect_spectral_basins(points)
    b2 = detect_spectral_basins(points)

    assert b1 == b2
    assert len(b1) == 2
    assert b1[0]["member_indices"] == [0, 1]
    assert b1[1]["member_indices"] == [2, 3]


def test_detect_spectral_basins_stable_centroids() -> None:
    points = np.asarray(
        [
            [0.0, 0.0],
            [0.15, 0.15],
            [2.0, 2.0],
            [2.15, 2.15],
        ],
        dtype=np.float64,
    )
    basins = detect_spectral_basins(points)

    c0 = np.asarray(basins[0]["centroid"], dtype=np.float64)
    c1 = np.asarray(basins[1]["centroid"], dtype=np.float64)
    np.testing.assert_allclose(c0, np.asarray([0.075, 0.075], dtype=np.float64))
    np.testing.assert_allclose(c1, np.asarray([2.075, 2.075], dtype=np.float64))


def test_detect_spectral_basins_stable_ids_ordered_by_centroid() -> None:
    points = [
        [1.0, 1.0],
        [1.05, 1.05],
        [-1.0, -1.0],
        [-1.03, -1.02],
    ]
    basins = detect_spectral_basins(points)

    assert basins[0]["basin_id"] == 0
    assert basins[1]["basin_id"] == 1
    assert basins[0]["centroid"][0] < basins[1]["centroid"][0]


def test_build_basin_transition_graph_deterministic() -> None:
    points = [
        [0.0, 0.0],
        [0.05, 0.05],
        [1.0, 1.0],
        [1.05, 1.05],
        [0.02, 0.02],
    ]
    basins = detect_spectral_basins(points)

    g1 = build_basin_transition_graph(points, basins)
    g2 = build_basin_transition_graph(points, basins)

    assert g1 == g2
    assert g1["basin_visit_counts"] == {0: 3, 1: 2}
    assert g1["basin_transitions"] == [
        {"from_basin_id": 0, "to_basin_id": 1, "count": 1},
        {"from_basin_id": 1, "to_basin_id": 0, "count": 1},
    ]
