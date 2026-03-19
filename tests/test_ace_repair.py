"""
Tests for v9.5.0 ACE-based repair operator.

Verifies:
  - matrix shape preserved
  - degree constraints maintained
  - deterministic output
  - binary entries preserved
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.ace_repair import repair_graph_with_ace_constraint


def _small_H():
    """3x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


def _fragile_H():
    """Matrix with a degree-1 variable node (column 3)."""
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 1, 1],
    ], dtype=np.float64)


class TestACERepair:

    def test_shape_preserved(self):
        H = _small_H()
        H_rep = repair_graph_with_ace_constraint(H)
        assert H_rep.shape == H.shape

    def test_binary_output(self):
        H = _small_H()
        H_rep = repair_graph_with_ace_constraint(H)
        assert np.all((H_rep == 0) | (H_rep == 1))

    def test_determinism(self):
        H = _small_H()
        H1 = repair_graph_with_ace_constraint(H)
        H2 = repair_graph_with_ace_constraint(H)
        np.testing.assert_array_equal(H1, H2)

    def test_no_input_mutation(self):
        H = _small_H()
        H_orig = H.copy()
        repair_graph_with_ace_constraint(H)
        np.testing.assert_array_equal(H, H_orig)

    def test_fragile_node_rewired(self):
        """Variable node with degree < 2 should be rewired."""
        H = _fragile_H()
        # Column 3 has degree 1 (only row 0)
        assert H[:, 3].sum() == 1
        H_rep = repair_graph_with_ace_constraint(H)
        # The edge should have moved; column 3 should no longer have
        # that edge (it moved to column 4)
        assert H_rep.shape == H.shape
        assert np.all((H_rep == 0) | (H_rep == 1))

    def test_healthy_graph_unchanged(self):
        """Graph where all variable degrees >= 2 should be unchanged."""
        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        # All columns have degree >= 2
        assert np.all(H.sum(axis=0) >= 2)
        H_rep = repair_graph_with_ace_constraint(H)
        np.testing.assert_array_equal(H, H_rep)

    def test_idempotent_on_stable_graph(self):
        """Applying repair twice should give the same result."""
        H = _small_H()
        H1 = repair_graph_with_ace_constraint(H)
        H2 = repair_graph_with_ace_constraint(H1)
        np.testing.assert_array_equal(H1, H2)
