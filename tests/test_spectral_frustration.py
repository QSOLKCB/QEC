from __future__ import annotations

import numpy as np

from qec.analysis.eigenmode_mutation import build_bethe_hessian
from qec.analysis.spectral_frustration import (
    SpectralFrustrationAnalyzer,
    SpectralFrustrationConfig,
    count_negative_modes,
)
from qec.discovery.spectral_descent_loop import spectral_descent


def _base_graph() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.float64)


def _more_cyclic_graph() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 1],
    ], dtype=np.float64)


def test_frustration_increases_with_additional_short_cycles() -> None:
    analyzer = SpectralFrustrationAnalyzer()
    low = analyzer.compute_frustration(_base_graph())
    high = analyzer.compute_frustration(_more_cyclic_graph())
    assert high.frustration_score >= low.frustration_score


def test_negative_mode_count_matches_inertia() -> None:
    H = _more_cyclic_graph()
    B_sparse, _ = build_bethe_hessian(H)
    B = B_sparse.toarray().astype(np.float64, copy=False)
    inertia_neg = count_negative_modes(B)
    eig_neg = int(np.sum(np.linalg.eigvalsh(B) < 0.0))
    assert inertia_neg == eig_neg


def test_frustration_deterministic_across_runs() -> None:
    analyzer = SpectralFrustrationAnalyzer()
    H = _more_cyclic_graph()
    r1 = analyzer.compute_frustration(H)
    r2 = analyzer.compute_frustration(H)
    assert r1 == r2


def test_detect_trap_nodes_localizes_inside_subgraph() -> None:
    H = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ], dtype=np.float64)
    analyzer = SpectralFrustrationAnalyzer(SpectralFrustrationConfig(trap_threshold=0.12))
    nodes = analyzer.detect_trap_nodes(H)
    assert len(nodes) > 0
    assert set(nodes).issubset(set(range(H.shape[0] + H.shape[1])))
    comp_a = {0, 1, 2, 4, 5, 6}
    comp_b = {3, 7, 8}
    assert set(nodes).issubset(comp_a) or set(nodes).issubset(comp_b)


def test_optimizer_opt_in_does_not_increase_frustration() -> None:
    H = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
    ], dtype=np.float64)
    analyzer = SpectralFrustrationAnalyzer()
    before = analyzer.compute_frustration(H).frustration_score
    H_opt = spectral_descent(H, max_iter=10, eta_frustration=0.25)
    after = analyzer.compute_frustration(H_opt.toarray()).frustration_score
    assert after <= before + 1e-12
