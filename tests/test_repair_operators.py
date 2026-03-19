"""
Tests for v9.0.0 repair operators.

Verifies:
  - repair fixes structural violations
  - repair is deterministic
  - validation detects known issues
  - combined repair pipeline works
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.repair_operators import (
    repair_degree_constraints,
    repair_duplicate_edges,
    repair_local_cycle_pressure,
    validate_tanner_graph,
    repair_tanner_graph,
)


def _small_H():
    """3x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


class TestValidation:
    def test_valid_graph(self):
        H = _small_H()
        result = validate_tanner_graph(H)
        assert result["is_valid"] is True
        assert result["violations"] == []

    def test_non_binary(self):
        H = _small_H()
        H[0, 0] = 2.0
        result = validate_tanner_graph(H)
        assert result["is_valid"] is False
        assert "non_binary_entries" in result["violations"]

    def test_zero_row(self):
        H = _small_H()
        H[0, :] = 0.0
        result = validate_tanner_graph(H)
        assert "zero_row" in result["violations"]

    def test_zero_column(self):
        H = _small_H()
        H[:, 0] = 0.0
        result = validate_tanner_graph(H)
        assert "zero_column" in result["violations"]


class TestRepairDuplicateEdges:
    def test_clips_to_binary(self):
        H = _small_H()
        H[0, 0] = 3.0
        H_fixed = repair_duplicate_edges(H)
        assert np.all((H_fixed == 0) | (H_fixed == 1))

    def test_deterministic(self):
        H = _small_H()
        H[0, 0] = 2.0
        H1 = repair_duplicate_edges(H)
        H2 = repair_duplicate_edges(H)
        np.testing.assert_array_equal(H1, H2)


class TestRepairDegreeConstraints:
    def test_deterministic(self):
        H = _small_H()
        H1 = repair_degree_constraints(H, target_variable_degree=2)
        H2 = repair_degree_constraints(H, target_variable_degree=2)
        np.testing.assert_array_equal(H1, H2)

    def test_output_binary(self):
        H = _small_H()
        H_fixed = repair_degree_constraints(H, target_variable_degree=2)
        assert np.all((H_fixed == 0) | (H_fixed == 1))

    def test_no_zero_rows_or_cols(self):
        H = _small_H()
        H_fixed = repair_degree_constraints(H, target_variable_degree=2)
        assert np.all(H_fixed.sum(axis=1) > 0)
        assert np.all(H_fixed.sum(axis=0) > 0)


class TestRepairLocalCyclePressure:
    def test_deterministic(self):
        H = _small_H()
        H1 = repair_local_cycle_pressure(H)
        H2 = repair_local_cycle_pressure(H)
        np.testing.assert_array_equal(H1, H2)

    def test_output_binary(self):
        H = _small_H()
        H_fixed = repair_local_cycle_pressure(H)
        assert np.all((H_fixed == 0) | (H_fixed == 1))


class TestRepairTannerGraph:
    def test_combined_repair(self):
        H = _small_H()
        H_repaired, validation = repair_tanner_graph(H)
        assert validation["is_valid"] is True

    def test_deterministic(self):
        H = _small_H()
        H1, v1 = repair_tanner_graph(H)
        H2, v2 = repair_tanner_graph(H)
        np.testing.assert_array_equal(H1, H2)
        assert v1 == v2

    def test_does_not_mutate_input(self):
        H = _small_H()
        H_orig = H.copy()
        repair_tanner_graph(H)
        np.testing.assert_array_equal(H, H_orig)
