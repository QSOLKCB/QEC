from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.qec.analysis.spectral_phase_map import (
    construct_phase_map,
    label_phases,
    render_phase_map,
)


def _basins() -> list[dict[str, object]]:
    return [
        {"basin_id": 3, "centroid": [2.0, 2.0], "members": [4, 5]},
        {"basin_id": 1, "centroid": [0.0, 0.0], "members": [0, 1]},
        {"basin_id": 2, "centroid": [1.0, 0.5], "members": [2, 3]},
    ]


def _ridges() -> list[dict[str, object]]:
    return [
        {"ridge_id": 8, "location": [0.5, 0.2]},
        {"ridge_id": 9, "location": [1.5, 1.0]},
        {"ridge_id": 10, "location": [2.1, 2.2]},
    ]


def _trajectory() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.8, 0.4],
            [1.6, 1.2],
            [2.2, 2.1],
        ],
        dtype=np.float64,
    )


def test_label_phases_stable_ids_sorted_by_centroid() -> None:
    phases1 = label_phases(_basins(), _ridges())
    phases2 = label_phases(_basins(), _ridges())

    assert phases1 == phases2
    assert [p["phase_id"] for p in phases1] == [0, 1, 2]
    centroids = [tuple(p["centroid"]) for p in phases1]
    assert centroids == sorted(centroids)


def test_construct_phase_map_deterministic_with_stable_adjacency() -> None:
    m1 = construct_phase_map(_basins(), _ridges(), {"grid_x": [0.0], "grid_y": [0.0]}, _trajectory())
    m2 = construct_phase_map(_basins(), _ridges(), {"grid_x": [0.0], "grid_y": [0.0]}, _trajectory())

    assert m1 == m2
    assert [r["phase_id"] for r in m1["phase_regions"]] == [0, 1, 2]
    edge_pairs = [(e["from_phase_id"], e["to_phase_id"], e["ridge_id"]) for e in m1["phase_adjacency"]]
    assert edge_pairs == sorted(edge_pairs)
    assert [s["segment_id"] for s in m1["trajectory_segments"]] == list(range(len(m1["trajectory_segments"])))


def test_render_phase_map_is_deterministic(tmp_path: Path) -> None:
    phase_map = construct_phase_map(_basins(), _ridges(), None, _trajectory())
    p1 = tmp_path / "phase_map_1.png"
    p2 = tmp_path / "phase_map_2.png"

    render_phase_map(phase_map, str(p1))
    render_phase_map(phase_map, str(p2))

    h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
    assert h1 == h2
