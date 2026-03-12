"""
v12.3.0 — Tests for NB Eigenvector Guided Mutation Operator.

Validates determinism, degree preservation, bipartite structure,
duplicate-edge avoidance, score ranking, and mutation log correctness.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from src.qec.discovery.mutation_nb_guided import NBGuidedMutator


def _make_test_matrix() -> np.ndarray:
    """Small 3x6 parity-check matrix with known structure."""
    H = np.array([
        [1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1, 1],
    ], dtype=np.float64)
    return H


def _make_larger_matrix() -> np.ndarray:
    """4x8 parity-check matrix with more rewiring room."""
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 0, 0, 1],
    ], dtype=np.float64)
    return H


class TestDeterminism:
    """Running mutation twice on the same input produces identical results."""

    def test_deterministic_output(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)

        H1, log1 = mutator.mutate(H)
        H2, log2 = mutator.mutate(H)

        np.testing.assert_array_equal(H1, H2)
        assert len(log1) == len(log2)
        for a, b in zip(log1, log2):
            assert a["removed_edge"] == b["removed_edge"]
            assert a["added_edge"] == b["added_edge"]
            assert a["score"] == b["score"]

    def test_deterministic_larger(self) -> None:
        H = _make_larger_matrix()
        mutator = NBGuidedMutator(k=3, enabled=True)

        H1, log1 = mutator.mutate(H)
        H2, log2 = mutator.mutate(H)

        np.testing.assert_array_equal(H1, H2)
        assert len(log1) == len(log2)


class TestValidTannerGraph:
    """After mutation the graph must remain a valid Tanner graph."""

    def test_bipartite_structure(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)
        H_mut, _ = mutator.mutate(H)

        # All entries are 0 or 1.
        assert set(np.unique(H_mut)).issubset({0.0, 1.0})

    def test_degrees_preserved(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)
        H_mut, _ = mutator.mutate(H)

        # Row sums (check degrees) unchanged.
        np.testing.assert_array_equal(
            H.sum(axis=1), H_mut.sum(axis=1),
        )
        # Column sums (variable degrees) unchanged.
        np.testing.assert_array_equal(
            H.sum(axis=0), H_mut.sum(axis=0),
        )

    def test_degrees_preserved_larger(self) -> None:
        H = _make_larger_matrix()
        mutator = NBGuidedMutator(k=3, enabled=True)
        H_mut, _ = mutator.mutate(H)

        np.testing.assert_array_equal(
            H.sum(axis=1), H_mut.sum(axis=1),
        )
        np.testing.assert_array_equal(
            H.sum(axis=0), H_mut.sum(axis=0),
        )

    def test_no_duplicate_edges(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)
        H_mut, _ = mutator.mutate(H)

        # No entry exceeds 1.
        assert H_mut.max() <= 1.0

    def test_no_input_mutation(self) -> None:
        H = _make_test_matrix()
        H_orig = H.copy()
        mutator = NBGuidedMutator(k=2, enabled=True)
        mutator.mutate(H)

        np.testing.assert_array_equal(H, H_orig)


class TestScoreRanking:
    """Edge scores must be deterministically ordered."""

    def test_scores_descending(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=5, enabled=True)

        edges = mutator._collect_edges(H)
        from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        scores = mutator._compute_edge_scores(
            H, edges, flow["directed_edge_flow"],
            flow["directed_edges"],
        )
        ranked = mutator._rank_edges(edges, scores)

        # Verify descending score order.
        ranked_scores = [scores[e] for e in ranked]
        for i in range(len(ranked_scores) - 1):
            assert ranked_scores[i] >= ranked_scores[i + 1]

    def test_lexicographic_tiebreak(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=5, enabled=True)

        edges = mutator._collect_edges(H)
        from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
        flow = NonBacktrackingFlowAnalyzer().compute_flow(H)
        scores = mutator._compute_edge_scores(
            H, edges, flow["directed_edge_flow"],
            flow["directed_edges"],
        )
        ranked = mutator._rank_edges(edges, scores)

        # Among edges with equal score, order must be lexicographic.
        for i in range(len(ranked) - 1):
            si = scores[ranked[i]]
            sj = scores[ranked[i + 1]]
            if si == sj:
                assert ranked[i] < ranked[i + 1]


class TestMutationLog:
    """Mutation log must contain correct fields."""

    def test_log_fields(self) -> None:
        H = _make_larger_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)
        _, log = mutator.mutate(H)

        for entry in log:
            assert "removed_edge" in entry
            assert "added_edge" in entry
            assert "score" in entry
            assert isinstance(entry["removed_edge"], tuple)
            assert isinstance(entry["added_edge"], tuple)
            assert isinstance(entry["score"], float)

    def test_log_length_respects_k(self) -> None:
        H = _make_larger_matrix()
        for k_val in [1, 2, 3]:
            mutator = NBGuidedMutator(k=k_val, enabled=True)
            _, log = mutator.mutate(H)
            assert len(log) <= k_val


class TestSparseInput:
    """Mutation must accept scipy sparse matrices."""

    def test_sparse_produces_same_result(self) -> None:
        H = _make_larger_matrix()
        H_sp = scipy.sparse.csr_matrix(H)
        mutator = NBGuidedMutator(k=2, enabled=True)

        H_dense, log_dense = mutator.mutate(H)
        H_sparse, log_sparse = mutator.mutate(H_sp)

        np.testing.assert_array_equal(H_dense, H_sparse)
        assert len(log_dense) == len(log_sparse)


class TestEdgeCases:
    """Edge cases: empty matrix, k=0."""

    def test_empty_matrix(self) -> None:
        H = np.zeros((3, 6), dtype=np.float64)
        mutator = NBGuidedMutator(k=2, enabled=True)
        H_mut, log = mutator.mutate(H)
        np.testing.assert_array_equal(H_mut, H)
        assert log == []

    def test_k_zero(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=0)
        H_mut, log = mutator.mutate(H)
        np.testing.assert_array_equal(H_mut, H)
        assert log == []

    def test_k_override(self) -> None:
        H = _make_larger_matrix()
        mutator = NBGuidedMutator(k=5, enabled=True)
        _, log = mutator.mutate(H, k=1)
        assert len(log) <= 1


class TestEnabledFlag:
    """The enabled flag must gate mutation behavior."""

    def test_disabled_returns_unchanged(self) -> None:
        H = _make_test_matrix()
        mutator = NBGuidedMutator(k=2, enabled=False)
        H_mut, log = mutator.mutate(H)

        np.testing.assert_array_equal(H_mut, H)
        assert log == []

    def test_enabled_may_mutate(self) -> None:
        H = _make_larger_matrix()
        mutator = NBGuidedMutator(k=2, enabled=True)
        H_mut, log = mutator.mutate(H)

        # When enabled, mutation should produce changes (for this matrix).
        assert len(log) > 0


class TestPartitionSeparation:
    """Check and variable node namespaces must not collide."""

    def test_no_false_conflict_across_partitions(self) -> None:
        """A swap using check 0 must not block edges at variable 0."""
        # 3x3 matrix where check and variable indices fully overlap.
        H = np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=np.float64)
        mutator = NBGuidedMutator(k=3, avoid_4cycles=False, enabled=True)
        H_mut, log = mutator.mutate(H)

        # Degrees must still be preserved.
        np.testing.assert_array_equal(
            H.sum(axis=1), H_mut.sum(axis=1),
        )
        np.testing.assert_array_equal(
            H.sum(axis=0), H_mut.sum(axis=0),
        )
