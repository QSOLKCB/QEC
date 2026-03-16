from __future__ import annotations

import numpy as np
import pytest

from src.qec.analysis.basin_depth import BasinDepthConfig, compute_basin_depth
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator
from src.qec.discovery.spectral_beam_search import adaptive_beam_width, select_beam_candidates


def test_basin_depth_is_deterministic() -> None:
    cfg = BasinDepthConfig(w1=1.0, w2=0.5, w3=0.25, w4=0.75, slope_window=3, precision=12)
    kwargs = {
        "flow_ipr": 0.33,
        "edge_reuse_rate": 0.21,
        "unstable_mode_persistence": 0.8,
        "energy_deltas": [0.1, 0.2, -0.1, 0.05],
        "config": cfg,
    }
    d1 = compute_basin_depth(**kwargs)
    d2 = compute_basin_depth(**kwargs)
    assert d1 == d2


def test_adaptive_beam_scaling_increases_with_depth() -> None:
    low = adaptive_beam_width(basin_depth=-1.0, beam_min=3, beam_max=10, depth_scale=3.0)
    high = adaptive_beam_width(basin_depth=1.0, beam_min=3, beam_max=10, depth_scale=3.0)
    assert 3 <= low <= 10
    assert 3 <= high <= 10
    assert high > low


def test_beam_diversity_guard_rejects_overlapping_edges() -> None:
    candidates = [
        {"score": -4.0, "swap_index": 0, "remove": ((0, 0), (1, 1)), "add": ((0, 1), (1, 0))},
        {"score": -3.0, "swap_index": 1, "remove": ((0, 0), (2, 2)), "add": ((0, 2), (2, 0))},
        {"score": -2.0, "swap_index": 2, "remove": ((3, 3), (4, 4)), "add": ((3, 4), (4, 3))},
    ]
    selected = select_beam_candidates(candidates, beam_width=3, beam_diversity=True)
    assert len(selected) == 2
    assert selected[0]["swap_index"] == 0
    assert selected[1]["swap_index"] == 2


def test_nb_alignment_bias_prefers_high_alignment_swaps() -> None:
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float64)
    mut = NBGradientMutator(enabled=True, avoid_4cycles=False, nb_alignment_bias=True, eta_nb=5.0)

    gradient = {
        "edge_scores": {(0, 0): 2.0, (1, 1): 2.0},
        "node_instability": {0: 5.0, 1: 5.0, 2: 0.0, 3: 0.0, 4: 0.0},
        "gradient_direction": {(0, 0): 7.0, (1, 1): 7.0},
    }

    mut._find_partner_check = lambda ci, vi, vj, *_: 1 if ci == 0 else 0  # type: ignore[assignment]

    directed_edges = [(0, 2), (1, 3), (2, 0), (2, 1), (3, 1), (3, 2)]
    directed_flow = np.array([3.0, 0.1, -3.0, 0.1, -0.1, 0.1], dtype=np.float64)
    flow = {
        "directed_edges": directed_edges,
        "directed_edge_flow": directed_flow,
        "edge_flow": np.array([1.0, 0.5, 0.25], dtype=np.float64),
        "variable_flow": np.array([1.0, 0.5, 0.25], dtype=np.float64),
    }

    alignment = mut._compute_nb_alignment_map(H, flow)
    candidates = mut._enumerate_swap_candidates(H, gradient, nb_alignment_map=alignment)
    ranked = sorted(candidates, key=lambda c: (float(c["score"]), int(c["swap_index"])))
    assert ranked[0]["removed_edge"] == (0, 0)


def test_adaptive_beam_swap_choice_is_deterministic_and_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    H = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.float64,
    )
    row_deg_before = H.sum(axis=1).copy()
    col_deg_before = H.sum(axis=0).copy()

    mut = NBGradientMutator(
        enabled=True,
        avoid_4cycles=False,
        enable_spectral_beam_search=True,
        adaptive_beam=True,
        beam_min=3,
        beam_max=6,
        depth_scale=4.0,
        beam_diversity=True,
        nb_alignment_bias=True,
        eta_nb=0.2,
        beam_width=4,
    )

    gradient = {
        "edge_scores": {(0, 0): 3.0, (1, 1): 2.0, (2, 0): 1.0},
        "node_instability": {0: 6.0, 1: 6.0, 2: 6.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
        "gradient_direction": {(0, 0): 7.0, (1, 1): 7.0, (2, 0): 7.0},
    }
    mut._analyzer.compute_gradient = lambda _H: gradient  # type: ignore[assignment]
    mut._trapping_predictor.predict_trapping_regions = lambda _H: {"ipr": 0.4, "risk_score": 0.5}  # type: ignore[assignment]
    mut._find_partner_check = lambda ci, vi, vj, *_: (1 if ci != 1 else 0)  # type: ignore[assignment]

    flow = {
        "directed_edges": [],
        "directed_edge_flow": np.zeros(0, dtype=np.float64),
        "edge_flow": np.array([1.0, 0.5, 0.25], dtype=np.float64),
        "variable_flow": np.array([1.0, 0.2, 0.1, 0.3], dtype=np.float64),
    }
    mut._nb_flow.compute_flow = lambda _H: flow  # type: ignore[assignment]

    H1, log1 = mut.mutate_flow(H, iterations=1)
    H2, log2 = mut.mutate_flow(H, iterations=1)

    np.testing.assert_array_equal(H1, H2)
    assert log1 == log2
    assert log1[0]["beam_search_used"] is True
    assert mut.beam_min <= log1[0]["beam_width"] <= mut.beam_max
    assert log1[0]["planner_depth"] == 2

    assert set(np.unique(H1)).issubset({0.0, 1.0})
    np.testing.assert_allclose(H1.sum(axis=1), row_deg_before)
    np.testing.assert_allclose(H1.sum(axis=0), col_deg_before)
