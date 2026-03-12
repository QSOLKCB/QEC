"""Tests for v12.6.0 NB instability-gradient mutation."""

from __future__ import annotations

import numpy as np
import scipy.sparse

from src.qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def _assert_binary(H: np.ndarray) -> None:
    assert set(np.unique(H)).issubset({0.0, 1.0})


class TestGradientAnalyzer:
    def test_deterministic_rounding_and_keys(self) -> None:
        H = _matrix()
        analyzer = NBInstabilityGradientAnalyzer()
        g1 = analyzer.compute_gradient(H)
        g2 = analyzer.compute_gradient(H)

        assert g1 == g2
        assert set(g1.keys()) == {
            "edge_scores", "node_instability", "gradient_direction",
        }
        for value in g1["edge_scores"].values():
            assert round(value, 12) == value

    def test_sparse_matches_dense(self) -> None:
        H = _matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        analyzer = NBInstabilityGradientAnalyzer()

        dense = analyzer.compute_gradient(H)
        sparse = analyzer.compute_gradient(H_sp)
        assert dense == sparse


class TestGradientMutator:
    def test_disabled_is_noop(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=False)
        H_new, log = mut.mutate(H, steps=3)
        np.testing.assert_array_equal(H_new, H)
        assert log == []

    def test_deterministic_results(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)

        H1, log1 = mut.mutate(H, steps=4)
        H2, log2 = mut.mutate(H, steps=4)
        np.testing.assert_array_equal(H1, H2)
        assert log1 == log2

    def test_degree_preservation_and_no_duplicates(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)
        H_new, _ = mut.mutate(H, steps=5)

        np.testing.assert_array_equal(H.sum(axis=0), H_new.sum(axis=0))
        np.testing.assert_array_equal(H.sum(axis=1), H_new.sum(axis=1))
        _assert_binary(H_new)

    def test_gradient_monotonicity_per_step(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=False)
        _, log = mut.mutate(H, steps=5)

        for step in log:
            assert step["source_gradient"] > step["target_gradient"]

    def test_mutate_flow_stability(self) -> None:
        H = _matrix()
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)
        H_flow_a, log_a = mut.mutate_flow(H, iterations=6)
        H_flow_b, log_b = mut.mutate_flow(H, iterations=6)

        np.testing.assert_array_equal(H_flow_a, H_flow_b)
        assert log_a == log_b

    def test_sparse_input(self) -> None:
        H = _matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        mut = NBGradientMutator(enabled=True, avoid_4cycles=True)

        H_dense, log_dense = mut.mutate(H, steps=3)
        H_sparse, log_sparse = mut.mutate(H_sp, steps=3)

        np.testing.assert_array_equal(H_dense, H_sparse)
        assert log_dense == log_sparse
