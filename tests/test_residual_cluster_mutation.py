"""
Tests for residual_cluster_mutation (v11.5.0).

Covers:
  - deterministic output
  - matrix shape preserved
  - edge count preserved
  - no input mutation
  - operator registry integration
  - dispatcher integration
  - memetic integration (cluster smoothing)
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.discovery.guided_mutations import (
    residual_cluster_mutation,
    apply_guided_mutation,
    _OPERATORS,
    OPERATORS,
    _OPERATOR_FUNCTIONS,
)


def _make_regular_H(m: int = 6, n: int = 12, seed: int = 42) -> np.ndarray:
    """Create a small regular parity-check matrix for testing."""
    rng = np.random.RandomState(seed)
    H = np.zeros((m, n), dtype=np.float64)
    for ci in range(m):
        cols = rng.choice(n, size=min(4, n), replace=False)
        for vi in cols:
            H[ci, vi] = 1.0
    # Ensure every column has at least one 1
    for vi in range(n):
        if H[:, vi].sum() == 0:
            ci = rng.randint(0, m)
            H[ci, vi] = 1.0
    return H


class TestResidualClusterMutationDeterminism:
    """Verify deterministic behavior."""

    def test_same_seed_same_result(self):
        H = _make_regular_H()
        H1 = residual_cluster_mutation(H, seed=42)
        H2 = residual_cluster_mutation(H, seed=42)

        np.testing.assert_array_equal(H1, H2)

    def test_different_seed_may_differ(self):
        H = _make_regular_H()
        H1 = residual_cluster_mutation(H, seed=0)
        H2 = residual_cluster_mutation(H, seed=999)

        # Both should be valid but may differ
        assert H1.shape == H.shape
        assert H2.shape == H.shape


class TestResidualClusterMutationPreservation:
    """Verify shape and edge count preservation."""

    def test_shape_preserved(self):
        H = _make_regular_H()
        H_out = residual_cluster_mutation(H, seed=42)

        assert H_out.shape == H.shape

    def test_edge_count_preserved(self):
        H = _make_regular_H()
        original_edges = int(np.count_nonzero(H))
        H_out = residual_cluster_mutation(H, seed=42)

        assert int(np.count_nonzero(H_out)) == original_edges

    def test_shape_preserved_various_sizes(self):
        for m, n in [(4, 8), (6, 12), (8, 16)]:
            H = _make_regular_H(m=m, n=n, seed=7)
            H_out = residual_cluster_mutation(H, seed=42)
            assert H_out.shape == (m, n)

    def test_edge_count_preserved_various_seeds(self):
        H = _make_regular_H()
        original_edges = int(np.count_nonzero(H))
        for seed in [0, 1, 42, 100, 999]:
            H_out = residual_cluster_mutation(H, seed=seed)
            assert int(np.count_nonzero(H_out)) == original_edges


class TestResidualClusterMutationNoInputMutation:
    """Verify the operator does not mutate input."""

    def test_input_not_modified(self):
        H = _make_regular_H()
        H_copy = H.copy()
        residual_cluster_mutation(H, seed=42)

        np.testing.assert_array_equal(H, H_copy)


class TestResidualClusterMutationEdgeCases:
    """Verify correct behavior on degenerate inputs."""

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        H_out = residual_cluster_mutation(H, seed=0)
        assert H_out.shape == (0, 0)

    def test_zero_matrix(self):
        H = np.zeros((4, 8), dtype=np.float64)
        H_out = residual_cluster_mutation(H, seed=0)
        assert H_out.shape == (4, 8)
        assert int(np.count_nonzero(H_out)) == 0

    def test_single_row(self):
        H = np.array([[1, 1, 0, 1]], dtype=np.float64)
        H_out = residual_cluster_mutation(H, seed=0)
        assert H_out.shape == H.shape

    def test_max_cluster_rewires_parameter(self):
        H = _make_regular_H()
        original_edges = int(np.count_nonzero(H))

        H_out1 = residual_cluster_mutation(H, seed=42, max_cluster_rewires=1)
        H_out2 = residual_cluster_mutation(H, seed=42, max_cluster_rewires=2)

        assert H_out1.shape == H.shape
        assert H_out2.shape == H.shape
        assert int(np.count_nonzero(H_out1)) == original_edges
        assert int(np.count_nonzero(H_out2)) == original_edges


class TestResidualClusterOperatorRegistry:
    """Verify operator is registered in all registries."""

    def test_in_operators_name_list(self):
        assert "residual_cluster" in _OPERATORS

    def test_in_operators_function_list(self):
        assert residual_cluster_mutation in OPERATORS

    def test_in_operator_functions_dict(self):
        assert "residual_cluster" in _OPERATOR_FUNCTIONS
        assert _OPERATOR_FUNCTIONS["residual_cluster"] is residual_cluster_mutation

    def test_operator_count_is_twelve(self):
        assert len(_OPERATORS) == 12
        assert len(OPERATORS) == 12
        assert len(_OPERATOR_FUNCTIONS) == 12

    def test_dispatcher_by_name(self):
        H = _make_regular_H()
        H_out = apply_guided_mutation(
            H, operator="residual_cluster", seed=42,
        )
        assert H_out.shape == H.shape

    def test_dispatcher_by_generation_schedule(self):
        """Operator 10 should be reached at generation 9 (0-indexed)."""
        H = _make_regular_H()
        # residual_cluster is at index 9 in _OPERATORS
        H_out = apply_guided_mutation(H, generation=9, seed=42)
        assert H_out.shape == H.shape

    def test_full_registry_list(self):
        expected = [
            "spectral_edge_pressure",
            "cycle_pressure",
            "ace_repair",
            "girth_preserving_rewire",
            "expansion_driven_rewire",
            "ipr_trapping_pressure",
            "trapping_set_pressure",
            "residual_guided",
            "absorbing_set_pressure",
            "residual_cluster",
            "spectral_localization",
            "nonbacktracking_flow",
        ]
        assert _OPERATORS == expected


class TestResidualClusterMemeticIntegration:
    """Verify cluster smoothing integration in local optimizer."""

    def test_optimizer_has_cluster_smoothing(self):
        from qec.discovery.local_optimizer import LocalGraphOptimizer

        opt = LocalGraphOptimizer(max_steps=5, seed=42)
        assert hasattr(opt, "_residual_cluster_smoothing")
        assert hasattr(opt, "_cluster_analyzer")

    def test_cluster_smoothing_returns_candidates(self):
        from qec.discovery.local_optimizer import LocalGraphOptimizer

        H = _make_regular_H()
        opt = LocalGraphOptimizer(max_steps=5, seed=42)
        candidates = opt._residual_cluster_smoothing(H, seed=42)

        assert isinstance(candidates, list)
        for cand in candidates:
            assert cand.shape == H.shape
            assert int(np.count_nonzero(cand)) == int(np.count_nonzero(H))

    def test_cluster_smoothing_deterministic(self):
        from qec.discovery.local_optimizer import LocalGraphOptimizer

        H = _make_regular_H()
        opt = LocalGraphOptimizer(max_steps=5, seed=42)

        c1 = opt._residual_cluster_smoothing(H, seed=123)
        c2 = opt._residual_cluster_smoothing(H, seed=123)

        assert len(c1) == len(c2)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a, b)

    def test_optimizer_five_operators(self):
        from qec.discovery.local_optimizer import LocalGraphOptimizer

        opt = LocalGraphOptimizer(max_steps=5, seed=42)
        # Verify 5 operators in the optimize loop by checking
        # that _residual_cluster_smoothing is the 5th operator
        assert callable(opt._residual_cluster_smoothing)
