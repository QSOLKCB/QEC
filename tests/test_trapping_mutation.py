"""
Tests for v11.0.0 — Trapping Set Pressure Mutation.

Covers:
  - deterministic behavior
  - matrix shape preserved
  - input matrix unchanged
  - valid output graph structure
  - operator registration in guided mutation framework
"""

import numpy as np
import pytest

from qec.discovery.guided_mutations import (
    trapping_set_pressure_mutation,
    apply_guided_mutation,
    _OPERATORS,
    _OPERATOR_FUNCTIONS,
)


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


class TestTrappingSetPressureMutation:
    def test_deterministic(self):
        H = _make_small_ldpc()
        r1 = trapping_set_pressure_mutation(H, seed=42)
        r2 = trapping_set_pressure_mutation(H, seed=42)

        np.testing.assert_array_equal(r1, r2)

    def test_shape_preserved(self):
        H = _make_small_ldpc()
        H_out = trapping_set_pressure_mutation(H, seed=42)

        assert H_out.shape == H.shape

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        trapping_set_pressure_mutation(H, seed=42)

        np.testing.assert_array_equal(H, H_copy)

    def test_output_binary(self):
        H = _make_small_ldpc()
        H_out = trapping_set_pressure_mutation(H, seed=42)

        # All entries should be 0 or 1
        unique_vals = set(np.unique(H_out))
        assert unique_vals <= {0.0, 1.0}

    def test_edge_count_preserved_or_close(self):
        """Edge count should be preserved (remove one, add one)."""
        H = _make_small_ldpc()
        original_edges = H.sum()
        H_out = trapping_set_pressure_mutation(H, seed=42)
        new_edges = H_out.sum()

        # Should be equal (one remove + one add), or unchanged if no mutation
        assert abs(new_edges - original_edges) <= 1

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        H_out = trapping_set_pressure_mutation(H, seed=42)
        assert H_out.shape == (0, 0)

    def test_different_seeds(self):
        H = _make_small_ldpc()
        r1 = trapping_set_pressure_mutation(H, seed=1)
        r2 = trapping_set_pressure_mutation(H, seed=999)

        # Both should be valid matrices
        assert r1.shape == H.shape
        assert r2.shape == H.shape


class TestOperatorRegistration:
    def test_trapping_set_pressure_in_operators_list(self):
        assert "trapping_set_pressure" in _OPERATORS

    def test_trapping_set_pressure_in_functions_dict(self):
        assert "trapping_set_pressure" in _OPERATOR_FUNCTIONS

    def test_dispatch_via_apply_guided_mutation(self):
        H = _make_small_ldpc()
        H_out = apply_guided_mutation(
            H, operator="trapping_set_pressure", seed=42,
        )

        assert H_out.shape == H.shape
        unique_vals = set(np.unique(H_out))
        assert unique_vals <= {0.0, 1.0}

    def test_schedule_includes_trapping_set_pressure(self):
        """The round-robin schedule should eventually select the new operator."""
        found = False
        for gen in range(len(_OPERATORS)):
            selected = _OPERATORS[gen % len(_OPERATORS)]
            if selected == "trapping_set_pressure":
                found = True
                break
        assert found
