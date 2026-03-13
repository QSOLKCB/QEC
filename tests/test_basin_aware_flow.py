from __future__ import annotations

import numpy as np

from src.qec.discovery.basin_aware_flow import BasinAwareFlowConfig, BasinAwareSpectralFlow


def _descent_swap(H: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    out = H.copy()
    out[0, 0] = 0.0
    out[1, 1] = 0.0
    out[0, 1] = 1.0
    out[1, 0] = 1.0
    return out, {"action": "descent_swap"}


def _explore_swap(H: np.ndarray) -> np.ndarray:
    out = H.copy()
    out[0, 2] = 0.0
    out[1, 0] = 0.0
    out[0, 0] = 1.0
    out[1, 2] = 1.0
    return out


def test_escape_triggered_for_localized_trap() -> None:
    H = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )

    engine = BasinAwareSpectralFlow(
        config=BasinAwareFlowConfig(
            enabled=True,
            max_steps=1,
            trap_ipr_threshold=0.3,
            edge_reuse_threshold=0.4,
            escape_candidate_limit=5,
            escape_blacklist_size=0,
        ),
    )

    # Force diagnostics trajectory to classify as localized trap.
    engine.diagnostics.energy_trace = [4.0, 4.0]
    engine.diagnostics.recent_hot_edges = [(0, 1), (0, 1), (0, 1)]

    result = engine.run(H)
    assert result["trajectory"]
    step = result["trajectory"][0]
    assert step["basin_state"] == "localized_trap"
    assert step["escape_triggered"] is True


def test_basin_aware_flow_determinism() -> None:
    H = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    cfg = BasinAwareFlowConfig(enabled=True, max_steps=3, escape_blacklist_size=0)

    e1 = BasinAwareSpectralFlow(config=cfg, descent_step=_descent_swap, exploration_step=_explore_swap)
    e2 = BasinAwareSpectralFlow(config=cfg, descent_step=_descent_swap, exploration_step=_explore_swap)

    r1 = e1.run(H)
    r2 = e2.run(H)

    assert np.array_equal(r1["H"], r2["H"])
    assert r1["trajectory"] == r2["trajectory"]


def test_escape_swap_preserves_degrees_and_binary() -> None:
    H = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    engine = BasinAwareSpectralFlow(
        config=BasinAwareFlowConfig(enabled=True, max_steps=1, escape_blacklist_size=0),
    )

    out, meta = engine._escape_step(
        H,
        {
            "hot_edges": [],
            "candidates": [(0, 0, 1, 1)],
        },
    )
    assert meta["action"] == "escape_swap"
    assert np.all((out == 0.0) | (out == 1.0))
    assert np.array_equal(np.sum(H, axis=0), np.sum(out, axis=0))
    assert np.array_equal(np.sum(H, axis=1), np.sum(out, axis=1))


def test_candidate_enumeration_pruning_is_deterministic() -> None:
    H = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    c1 = BasinAwareSpectralFlow._enumerate_swap_candidates(H, max_row_candidates=1, max_col_candidates=1)
    c2 = BasinAwareSpectralFlow._enumerate_swap_candidates(H, max_row_candidates=1, max_col_candidates=1)
    assert c1 == c2


def test_rank_candidates_respects_max_ipr_evaluations_zero() -> None:
    H = np.array(
        [
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    engine = BasinAwareSpectralFlow(config=BasinAwareFlowConfig(enabled=True, max_ipr_evaluations=0, escape_blacklist_size=0))
    candidates = [(0, 0, 1, 1)]
    ranked = engine._rank_candidates(H, candidates, set())
    assert ranked
    assert ranked[0]["ipr_after"] == engine._evaluate_ipr(H)
