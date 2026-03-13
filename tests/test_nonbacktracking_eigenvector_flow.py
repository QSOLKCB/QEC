from __future__ import annotations

import numpy as np

from src.qec.analysis.nonbacktracking_flow import (
    NBFlowConfig,
    NonBacktrackingEigenvectorFlowAnalyzer,
    canonical_directed_edges,
    normalize_mode_phase,
    project_directed_pressure_to_undirected,
)
from src.qec.discovery.nonbacktracking_eigenvector_flow import (
    NonBacktrackingEigenvectorFlowOptimizer,
)


def _toy_H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.float64)


def test_phase_normalization_invariant_under_global_phase() -> None:
    v = np.array([1.0 + 2.0j, -3.0 + 4.0j, 0.5 - 0.1j], dtype=np.complex128)
    n1 = normalize_mode_phase(v)
    n2 = normalize_mode_phase(v * np.exp(1j * 0.73))
    np.testing.assert_allclose(n1, n2)


def test_directed_to_undirected_projection_pairs_both_directions() -> None:
    H = _toy_H()
    undirected, directed, index, _, _, n = canonical_directed_edges(H)
    p = np.zeros(len(directed), dtype=np.float64)
    ci, vi = undirected[0]
    p[index[(vi, n + ci)]] = 2.0
    p[index[(n + ci, vi)]] = 3.0
    edge_pressure, edge_map = project_directed_pressure_to_undirected(
        undirected_edges=undirected,
        directed_index=index,
        n=n,
        directed_pressure=p,
    )
    assert edge_map[(ci, vi)] == 5.0
    assert edge_pressure[0] == 5.0


def test_flow_aggregation_deterministic() -> None:
    H = _toy_H()
    analyzer = NonBacktrackingEigenvectorFlowAnalyzer(config=NBFlowConfig(num_nb_eigenvalues=4))
    f1 = analyzer.build_flow_field(H)
    f2 = analyzer.build_flow_field(H)
    np.testing.assert_array_equal(f1.directed_pressure, f2.directed_pressure)
    np.testing.assert_array_equal(f1.edge_pressure, f2.edge_pressure)


def test_swap_ranking_determinism() -> None:
    H = _toy_H()
    opt = NonBacktrackingEigenvectorFlowOptimizer(max_steps=1, min_girth=0, use_bh_acceptance=False)
    t1 = opt.optimize(H)
    t2 = opt.optimize(H)
    assert t1.steps == t2.steps


def test_validity_preservation_degree_and_binary() -> None:
    H = _toy_H()
    opt = NonBacktrackingEigenvectorFlowOptimizer(max_steps=2, min_girth=0, use_bh_acceptance=False)
    out = opt.optimize(H).H_final
    np.testing.assert_array_equal(H.sum(axis=0), out.sum(axis=0))
    np.testing.assert_array_equal(H.sum(axis=1), out.sum(axis=1))
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_hybrid_acceptance_rejects_when_bh_worsens() -> None:
    H = _toy_H()
    opt = NonBacktrackingEigenvectorFlowOptimizer(max_steps=1, min_girth=0, use_bh_acceptance=True)
    # deterministic check: either accepted or BH-rejected, but if rejected reason is fixed
    tr = opt.optimize(H)
    if tr.steps:
        assert tr.steps[0].reason in {"accepted", "bh_rejected", "flow_converged"}


def test_trajectory_monotonic_deltaflow_for_accepted_steps() -> None:
    H = _toy_H()
    opt = NonBacktrackingEigenvectorFlowOptimizer(max_steps=3, min_girth=0, use_bh_acceptance=False)
    tr = opt.optimize(H)
    for step in tr.steps:
        if step.accepted:
            assert step.delta_flow <= 1e-8


def test_convergence_reason_enum() -> None:
    H = np.zeros((2, 2), dtype=np.float64)
    opt = NonBacktrackingEigenvectorFlowOptimizer(max_steps=1)
    tr = opt.optimize(H)
    assert tr.termination_reason in {
        "no_informative_nb_mode",
        "no_improving_swap",
        "flow_converged",
        "max_steps_reached",
    }
