from __future__ import annotations

import numpy as np

from src.qec.analysis.nonbacktracking_flow import NBFlowConfig, NonBacktrackingEigenvectorFlowAnalyzer, canonical_directed_edges
from src.qec.discovery.nb_flow_mutation import NBFlowMutationConfig, NonBacktrackingFlowMutator


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def _flow_ranking(H: np.ndarray, *, enable_second_mode: bool) -> list[tuple[tuple[int, int], float]]:
    analyzer = NonBacktrackingEigenvectorFlowAnalyzer(config=NBFlowConfig(num_nb_eigenvalues=256, precision=12, bulk_radius_mode="fixed", bulk_radius_value=0.0))
    eigvals, eigvecs = analyzer.compute_modes(H)
    assert eigvals.size >= 1

    mut = NonBacktrackingFlowMutator(
        config=NBFlowMutationConfig(
            enabled=True,
            enable_second_mode=enable_second_mode,
            second_mode_weight=0.25,
        ),
    )
    v, _, _ = mut._build_flow_vector(eigvals, eigvecs)
    undirected, _, edge_index, _, _, n = canonical_directed_edges(H)
    flow = mut._edge_flow_map(
        undirected=undirected,
        edge_index=edge_index,
        n=n,
        flow_vector=v,
    )
    return sorted(flow.items(), key=lambda item: (-abs(item[1]), item[0]))


def test_deterministic_mutation() -> None:
    H = _matrix()
    cfg = NBFlowMutationConfig(enabled=True, max_flow_edges=6, swap_candidates_per_edge=4)
    mut = NonBacktrackingFlowMutator(config=cfg)

    out_a, log_a = mut.mutate(H)
    out_b, log_b = mut.mutate(H)

    np.testing.assert_array_equal(out_a, out_b)
    assert log_a == log_b


def test_degree_preservation() -> None:
    H = _matrix()
    mut = NonBacktrackingFlowMutator(config=NBFlowMutationConfig(enabled=True, max_flow_edges=8))

    out, _ = mut.mutate(H)

    np.testing.assert_array_equal(H.sum(axis=0), out.sum(axis=0))
    np.testing.assert_array_equal(H.sum(axis=1), out.sum(axis=1))
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_flow_localization_prefers_short_cycle_region() -> None:
    H = _matrix()
    mut = NonBacktrackingFlowMutator(config=NBFlowMutationConfig(enabled=True, max_flow_edges=4))
    _, log = mut.mutate(H)

    if not log:
        return

    selected = tuple(log[0]["swap_selected"])
    # In this toy graph, the high-flow short-cycle core is concentrated
    # around rows 0-2 and columns 0-3.
    short_cycle_rows = {0, 1, 2}
    short_cycle_cols = {0, 1, 2, 3}
    assert selected[0] in short_cycle_rows or selected[2] in short_cycle_rows
    assert selected[1] in short_cycle_cols or selected[3] in short_cycle_cols


def test_two_mode_changes_flow_ranking_deterministically() -> None:
    H = _matrix()
    r_single_a = _flow_ranking(H, enable_second_mode=False)
    r_single_b = _flow_ranking(H, enable_second_mode=False)
    r_dual_a = _flow_ranking(H, enable_second_mode=True)
    r_dual_b = _flow_ranking(H, enable_second_mode=True)

    assert r_single_a == r_single_b
    assert r_dual_a == r_dual_b
    assert r_single_a != r_dual_a
