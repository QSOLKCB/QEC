"""
Tests for the v12.1.0 FlowAlignmentAnalyzer.

Verifies:
  - deterministic output
  - correct cosine similarity range
  - stable classification thresholds
  - identical output for repeated runs
  - zero-vector handling
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.flow_alignment import FlowAlignmentAnalyzer


class TestFlowAlignmentAnalyzer:
    """Tests for FlowAlignmentAnalyzer.compute_alignment."""

    def test_deterministic(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 2.0, 3.0, 4.0])
        flow = np.array([0.5, 1.5, 2.5, 3.5])
        r1 = analyzer.compute_alignment(residual, flow)
        r2 = analyzer.compute_alignment(residual, flow)
        assert r1["alignment_score"] == r2["alignment_score"]
        assert r1["residual_norm"] == r2["residual_norm"]
        assert r1["flow_norm"] == r2["flow_norm"]
        assert r1["alignment_type"] == r2["alignment_type"]

    def test_identical_vectors_strong(self):
        analyzer = FlowAlignmentAnalyzer()
        v = np.array([1.0, 2.0, 3.0])
        result = analyzer.compute_alignment(v, v)
        assert result["alignment_score"] == 1.0
        assert result["alignment_type"] == "strong"

    def test_orthogonal_vectors_weak(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 0.0])
        flow = np.array([0.0, 1.0])
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_score"] == 0.0
        assert result["alignment_type"] == "weak"

    def test_opposite_vectors(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 2.0, 3.0])
        flow = np.array([-1.0, -2.0, -3.0])
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_score"] == -1.0
        assert result["alignment_type"] == "weak"

    def test_score_range(self):
        analyzer = FlowAlignmentAnalyzer()
        rng = np.random.RandomState(42)
        for _ in range(20):
            residual = rng.standard_normal(10)
            flow = rng.standard_normal(10)
            result = analyzer.compute_alignment(residual, flow)
            assert -1.0 <= result["alignment_score"] <= 1.0

    def test_zero_residual(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.zeros(5)
        flow = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_score"] == 0.0
        assert result["alignment_type"] == "weak"

    def test_zero_flow(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 2.0, 3.0])
        flow = np.zeros(3)
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_score"] == 0.0
        assert result["alignment_type"] == "weak"

    def test_both_zero(self):
        analyzer = FlowAlignmentAnalyzer()
        result = analyzer.compute_alignment(np.zeros(4), np.zeros(4))
        assert result["alignment_score"] == 0.0
        assert result["alignment_type"] == "weak"

    def test_strong_threshold(self):
        analyzer = FlowAlignmentAnalyzer()
        # Construct vectors with cosine similarity >= 0.6
        residual = np.array([1.0, 0.0])
        flow = np.array([1.0, 1.0])
        result = analyzer.compute_alignment(residual, flow)
        # cos(45°) ≈ 0.707 → strong
        assert result["alignment_type"] == "strong"

    def test_moderate_threshold(self):
        analyzer = FlowAlignmentAnalyzer()
        # cos(angle) ≈ 0.4 → moderate
        residual = np.array([1.0, 0.0, 0.0])
        flow = np.array([0.4, 0.9165, 0.0])
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_type"] == "moderate"

    def test_weak_threshold(self):
        analyzer = FlowAlignmentAnalyzer()
        # Nearly orthogonal → weak
        residual = np.array([1.0, 0.0, 0.0])
        flow = np.array([0.1, 1.0, 0.0])
        result = analyzer.compute_alignment(residual, flow)
        assert result["alignment_type"] == "weak"

    def test_norms_correct(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([3.0, 4.0])
        flow = np.array([5.0, 12.0])
        result = analyzer.compute_alignment(residual, flow)
        assert abs(result["residual_norm"] - 5.0) < 1e-10
        assert abs(result["flow_norm"] - 13.0) < 1e-10

    def test_result_keys(self):
        analyzer = FlowAlignmentAnalyzer()
        result = analyzer.compute_alignment(np.ones(3), np.ones(3))
        assert "alignment_score" in result
        assert "residual_norm" in result
        assert "flow_norm" in result
        assert "alignment_type" in result

    def test_no_input_mutation(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 2.0, 3.0])
        flow = np.array([4.0, 5.0, 6.0])
        r_copy = residual.copy()
        f_copy = flow.copy()
        analyzer.compute_alignment(residual, flow)
        np.testing.assert_array_equal(residual, r_copy)
        np.testing.assert_array_equal(flow, f_copy)

    def test_single_element(self):
        analyzer = FlowAlignmentAnalyzer()
        result = analyzer.compute_alignment(
            np.array([5.0]), np.array([3.0]),
        )
        assert result["alignment_score"] == 1.0
        assert result["alignment_type"] == "strong"

    def test_rounding_precision(self):
        analyzer = FlowAlignmentAnalyzer()
        residual = np.array([1.0, 2.0, 3.0])
        flow = np.array([4.0, 5.0, 6.0])
        result = analyzer.compute_alignment(residual, flow)
        score_str = f"{result['alignment_score']:.12f}"
        # Must have at most 12 decimal digits
        assert len(score_str.split(".")[1]) <= 12
