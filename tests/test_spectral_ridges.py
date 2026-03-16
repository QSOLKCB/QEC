from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_basins import detect_spectral_basins
from src.qec.analysis.spectral_ridges import (
    build_ridge_graph,
    detect_spectral_ridges,
    map_ridges_to_basins,
)


def _sample_points() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [2.0, 4.0],
            [3.0, 4.0],
            [4.0, 4.0],
            [5.0, 4.0],
        ],
        dtype=np.float64,
    )


def test_detect_spectral_ridges_is_deterministic_with_stable_ids() -> None:
    points = _sample_points()
    r1 = detect_spectral_ridges(points)
    r2 = detect_spectral_ridges(points)

    assert r1 == r2
    assert [rec["ridge_id"] for rec in r1] == list(range(len(r1)))
    locations = [tuple(rec["location"]) for rec in r1]
    assert locations == sorted(locations)


def test_build_ridge_graph_is_reproducible() -> None:
    ridges = detect_spectral_ridges(_sample_points())
    g1 = build_ridge_graph(ridges)
    g2 = build_ridge_graph(ridges)

    assert g1 == g2
    edge_pairs = [(e["from_ridge_id"], e["to_ridge_id"]) for e in g1["ridge_edges"]]
    assert edge_pairs == sorted(edge_pairs)


def test_map_ridges_to_basins_is_deterministic() -> None:
    points = _sample_points()
    ridges = detect_spectral_ridges(points)
    basins = detect_spectral_basins(points)

    m1 = map_ridges_to_basins(ridges, basins)
    m2 = map_ridges_to_basins(ridges, basins)

    assert m1 == m2
    segments = m1["basin_boundary_segments"]
    assert segments == sorted(
        segments,
        key=lambda rec: (
            tuple(rec["adjacent_basins"]),
            tuple(float(np.float64(v)) for v in rec["location"]),
            int(rec["ridge_id"]),
        ),
    )
