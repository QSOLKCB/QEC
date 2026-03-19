from __future__ import annotations

import numpy as np

from qec.discovery.basin_hopping import propose_basin_hop


def _basins() -> list[dict[str, object]]:
    return [
        {"basin_id": 0, "centroid": [0.0, 0.0], "basin_radius": 0.1, "member_indices": [0, 1]},
        {"basin_id": 1, "centroid": [1.0, 1.0], "basin_radius": 0.1, "member_indices": [2, 3]},
    ]


def test_propose_basin_hop_deterministic() -> None:
    current = np.asarray([0.01, 0.02], dtype=np.float64)
    p1 = propose_basin_hop(current, _basins())
    p2 = propose_basin_hop(current, _basins())
    np.testing.assert_allclose(np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64))


def test_propose_basin_hop_centroid_offset_logic() -> None:
    current = np.asarray([0.01, 0.02], dtype=np.float64)
    proposal = np.asarray(propose_basin_hop(current, _basins()), dtype=np.float64)
    target_centroid = np.asarray([1.0, 1.0], dtype=np.float64)

    assert proposal.shape == (2,)
    assert np.linalg.norm(proposal - target_centroid) > 0.0
    assert np.linalg.norm(proposal - target_centroid) < 0.01


def test_propose_basin_hop_reproducible_across_runs() -> None:
    current = np.asarray([0.95, 0.95], dtype=np.float64)
    basins = _basins()
    expected = np.asarray(propose_basin_hop(current, basins), dtype=np.float64)
    for _ in range(5):
        np.testing.assert_allclose(expected, np.asarray(propose_basin_hop(current, basins), dtype=np.float64))
