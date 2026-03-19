"""
Integration tests for v12.1.0 flow alignment diagnostics.

Verifies that NonBacktrackingFlowAnalyzer returns flow_alignment
when a residual_map is provided, and omits it when not.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.sparse

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from qec.analysis.bp_residuals import BPResidualAnalyzer


def _small_H():
    """Create a small (3, 6) regular parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


class TestFlowAlignmentIntegration:
    """Integration tests for flow alignment in NB flow analyzer."""

    def test_no_alignment_without_residual(self):
        H = _small_H()
        result = NonBacktrackingFlowAnalyzer().compute_flow(H)
        assert "flow_alignment" not in result

    def test_alignment_with_residual(self):
        H = _small_H()
        residual_map = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4])
        result = NonBacktrackingFlowAnalyzer().compute_flow(
            H, residual_map=residual_map,
        )
        assert "flow_alignment" in result
        alignment = result["flow_alignment"]
        assert "alignment_score" in alignment
        assert "residual_norm" in alignment
        assert "flow_norm" in alignment
        assert "alignment_type" in alignment
        assert -1.0 <= alignment["alignment_score"] <= 1.0
        assert alignment["alignment_type"] in ("strong", "moderate", "weak")

    def test_alignment_deterministic(self):
        H = _small_H()
        residual_map = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4])
        analyzer = NonBacktrackingFlowAnalyzer()
        r1 = analyzer.compute_flow(H, residual_map=residual_map)
        r2 = analyzer.compute_flow(H, residual_map=residual_map)
        assert r1["flow_alignment"]["alignment_score"] == r2["flow_alignment"]["alignment_score"]
        assert r1["flow_alignment"]["alignment_type"] == r2["flow_alignment"]["alignment_type"]

    def test_alignment_with_bp_residual_analyzer(self):
        H = _small_H()
        bp = BPResidualAnalyzer()
        residuals = bp.compute_residual_map(H, iterations=5, seed=42)
        result = NonBacktrackingFlowAnalyzer().compute_flow(
            H, residual_map=residuals["residual_map"],
        )
        assert "flow_alignment" in result
        alignment = result["flow_alignment"]
        assert -1.0 <= alignment["alignment_score"] <= 1.0
        assert alignment["alignment_type"] in ("strong", "moderate", "weak")

    def test_alignment_sparse_matrix(self):
        H_dense = _small_H()
        H_sparse = scipy.sparse.csr_matrix(H_dense)
        residual_map = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4])
        r_dense = NonBacktrackingFlowAnalyzer().compute_flow(
            H_dense, residual_map=residual_map,
        )
        r_sparse = NonBacktrackingFlowAnalyzer().compute_flow(
            H_sparse, residual_map=residual_map,
        )
        assert r_dense["flow_alignment"]["alignment_score"] == r_sparse["flow_alignment"]["alignment_score"]
        assert r_dense["flow_alignment"]["alignment_type"] == r_sparse["flow_alignment"]["alignment_type"]

    def test_zero_residual_alignment(self):
        H = _small_H()
        residual_map = np.zeros(6)
        result = NonBacktrackingFlowAnalyzer().compute_flow(
            H, residual_map=residual_map,
        )
        assert result["flow_alignment"]["alignment_score"] == 0.0
        assert result["flow_alignment"]["alignment_type"] == "weak"

    def test_no_input_mutation_with_residual(self):
        H = _small_H()
        H_copy = H.copy()
        residual_map = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4])
        r_copy = residual_map.copy()
        NonBacktrackingFlowAnalyzer().compute_flow(
            H, residual_map=residual_map,
        )
        np.testing.assert_array_equal(H, H_copy)
        np.testing.assert_array_equal(residual_map, r_copy)
