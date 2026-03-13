import numpy as np

from src.qec.analysis.basin_depth import adaptive_beam_width, basin_depth_from_diagnostics, compute_basin_depth
from src.qec.discovery.eigenmode_guided_swap import find_best_swap


def test_basin_depth_computation_is_deterministic() -> None:
    diagnostics = {
        "flow_ipr": 0.8,
        "edge_reuse_rate": 0.6,
        "unstable_mode_persistence": 0.4,
        "energy_slope": -0.2,
    }
    d1 = basin_depth_from_diagnostics(diagnostics)
    d2 = basin_depth_from_diagnostics(diagnostics)
    expected = compute_basin_depth(
        flow_ipr=0.8,
        edge_reuse_rate=0.6,
        unstable_mode_persistence=0.4,
        energy_slope=-0.2,
    )
    assert d1 == d2
    assert d1 == expected


def test_adaptive_width_increases_with_depth() -> None:
    low = adaptive_beam_width(-1.5, beam_min=3, beam_max=10, depth_scale=3.0)
    mid = adaptive_beam_width(0.0, beam_min=3, beam_max=10, depth_scale=3.0)
    high = adaptive_beam_width(1.5, beam_min=3, beam_max=10, depth_scale=3.0)
    assert 3 <= low <= 10
    assert 3 <= mid <= 10
    assert 3 <= high <= 10
    assert low <= mid <= high


def test_adaptive_beam_swap_is_deterministic() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    G = {
        (0, 0): 1.0,
        (0, 1): 0.7,
        (1, 1): 0.9,
        (1, 2): 0.8,
        (2, 0): 0.6,
        (2, 3): 0.5,
        (0, 2): -0.3,
        (1, 0): -0.2,
        (2, 1): -0.1,
        (0, 3): -0.4,
    }
    diag = {
        "flow_ipr": 0.9,
        "edge_reuse_rate": 0.8,
        "unstable_mode_persistence": 0.6,
        "energy_slope": -0.05,
    }

    r1 = find_best_swap(H, G, adaptive_beam=True, basin_diagnostics=diag, return_metadata=True)
    r2 = find_best_swap(H, G, adaptive_beam=True, basin_diagnostics=diag, return_metadata=True)
    assert r1 == r2


def test_adaptive_swap_preserves_degree_sequence_and_structure() -> None:
    H = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    G = {(0, 0): 1.0, (1, 1): 0.9, (2, 3): 0.8, (0, 2): -0.1, (1, 3): -0.2, (2, 1): -0.3}

    result = find_best_swap(H, G, adaptive_beam=True)
    if result is None:
        return

    H2 = H.copy()
    for ci, vi in result["remove"]:
        H2[ci, vi] = 0.0
    for ci, vi in result["add"]:
        H2[ci, vi] = 1.0

    assert H2.shape == H.shape
    assert np.array_equal(np.sum(H, axis=1), np.sum(H2, axis=1))
    assert np.array_equal(np.sum(H, axis=0), np.sum(H2, axis=0))
    assert np.all((H2 == 0.0) | (H2 == 1.0))
