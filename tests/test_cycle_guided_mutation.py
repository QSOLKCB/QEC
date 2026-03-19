"""
v9.4.0 — Tests for the Cycle-Pressure Guided Mutation Operator.

Verifies:
- Mutation is deterministic.
- Matrix shape is unchanged.
- Degree constraints are preserved (no isolated nodes).
- Operator is registered in the mutation schedule.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.discovery.cycle_guided_mutation import (
    cycle_pressure_guided_mutation,
)
from qec.discovery.mutation_operators import (
    _OPERATORS,
    _OPERATOR_FUNCTIONS,
    mutate_tanner_graph,
)


def _make_simple_H() -> np.ndarray:
    """Create a small parity-check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


class TestCyclePressureGuidedMutation:
    """Tests for cycle_pressure_guided_mutation."""

    def test_deterministic(self):
        H = _make_simple_H()
        H1 = cycle_pressure_guided_mutation(H, seed=42)
        H2 = cycle_pressure_guided_mutation(H, seed=42)
        np.testing.assert_array_equal(H1, H2)

    def test_shape_unchanged(self):
        H = _make_simple_H()
        H_new = cycle_pressure_guided_mutation(H)
        assert H_new.shape == H.shape

    def test_no_isolated_rows(self):
        H = _make_simple_H()
        H_new = cycle_pressure_guided_mutation(H)
        for ci in range(H_new.shape[0]):
            assert H_new[ci].sum() >= 1, f"Row {ci} is isolated"

    def test_no_isolated_columns(self):
        H = _make_simple_H()
        H_new = cycle_pressure_guided_mutation(H)
        for vi in range(H_new.shape[1]):
            assert H_new[:, vi].sum() >= 1, f"Column {vi} is isolated"

    def test_does_not_modify_input(self):
        H = _make_simple_H()
        H_copy = H.copy()
        cycle_pressure_guided_mutation(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_binary_output(self):
        H = _make_simple_H()
        H_new = cycle_pressure_guided_mutation(H)
        unique_vals = set(H_new.flatten())
        assert unique_vals <= {0.0, 1.0}

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        H_new = cycle_pressure_guided_mutation(H)
        assert H_new.shape == (0, 0)

    def test_with_target_edges(self):
        H = _make_simple_H()
        target = [(0, 0), (1, 1)]
        H_new = cycle_pressure_guided_mutation(H, target_edges=target)
        assert H_new.shape == H.shape


class TestOperatorRegistration:
    """Tests for integration with mutation_operators."""

    def test_registered_in_schedule(self):
        assert "cycle_guided_mutation" in _OPERATORS

    def test_registered_in_functions(self):
        assert "cycle_guided_mutation" in _OPERATOR_FUNCTIONS

    def test_callable_via_mutate_tanner_graph(self):
        H = _make_simple_H()
        H_new, op_name = mutate_tanner_graph(
            H, operator="cycle_guided_mutation", seed=42,
        )
        assert op_name == "cycle_guided_mutation"
        assert H_new.shape == H.shape

    def test_scheduled_generation(self):
        """Cycle guided mutation appears in schedule rotation."""
        idx = _OPERATORS.index("cycle_guided_mutation")
        H = _make_simple_H()
        H_new, op_name = mutate_tanner_graph(H, generation=idx, seed=0)
        assert op_name == "cycle_guided_mutation"
