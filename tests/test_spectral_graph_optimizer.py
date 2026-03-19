"""
Tests for v7.5.0 — Deterministic Spectral Graph Optimization Pipeline.

Verifies:
  - Determinism: repeated runs produce identical artifacts
  - Improvement: final_score <= initial_score
  - Gradient heuristic: high-magnitude eigenvector edges prioritized
  - Curvature diagnostic: oscillatory BP traces detected
  - Ternary classification: correct LLR → ternary mapping
  - JSON artifact stability: serialization roundtrip
  - Decoder safety: optimizer does NOT import src/qec/decoder/
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.experiments.spectral_graph_optimizer import (
    _compute_graph_instability_score,
    _compute_spectral_edge_gradient,
    _generate_guided_repair_candidates,
    _compute_belief_curvature,
    _classify_ternary_llr,
    _compute_ternary_cluster_stats,
    run_spectral_graph_optimization,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_simple_code() -> np.ndarray:
    """Simple 3x5 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)


def _make_larger_code() -> np.ndarray:
    """Larger 4x7 parity-check matrix."""
    return np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
    ], dtype=np.float64)


# ── Test: Determinism ─────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs produce identical artifacts."""

    def test_instability_score_deterministic(self):
        H = _make_simple_code()
        s1 = _compute_graph_instability_score(H)
        s2 = _compute_graph_instability_score(H)
        assert s1 == s2

    def test_gradient_deterministic(self):
        H = _make_simple_code()
        g1 = _compute_spectral_edge_gradient(H)
        g2 = _compute_spectral_edge_gradient(H)
        assert g1 == g2

    def test_optimization_deterministic(self):
        H = _make_larger_code()
        r1 = run_spectral_graph_optimization(H, max_iterations=3)
        r2 = run_spectral_graph_optimization(H, max_iterations=3)
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_full_optimization_deterministic(self):
        H = _make_simple_code()
        r1 = run_spectral_graph_optimization(
            H, max_iterations=5, max_candidates=5,
        )
        r2 = run_spectral_graph_optimization(
            H, max_iterations=5, max_candidates=5,
        )
        assert r1 == r2


# ── Test: Improvement ─────────────────────────────────────────────────


class TestImprovement:
    """final_score <= initial_score."""

    def test_score_nonincreasing(self):
        H = _make_larger_code()
        result = run_spectral_graph_optimization(H, max_iterations=5)
        assert result["final_instability_score"] <= result["initial_instability_score"]

    def test_score_nonincreasing_simple(self):
        H = _make_simple_code()
        result = run_spectral_graph_optimization(H, max_iterations=3)
        assert result["final_instability_score"] <= result["initial_instability_score"]

    def test_spectral_radius_reduction_nonnegative(self):
        H = _make_larger_code()
        result = run_spectral_graph_optimization(H, max_iterations=5)
        assert result["spectral_radius_reduction"] >= 0.0


# ── Test: Gradient Heuristic ──────────────────────────────────────────


class TestGradientHeuristic:
    """High-magnitude eigenvector edges are prioritized."""

    def test_gradient_sorted_descending(self):
        H = _make_simple_code()
        gradient = _compute_spectral_edge_gradient(H)
        if len(gradient) >= 2:
            for i in range(len(gradient) - 1):
                assert gradient[i][1] >= gradient[i + 1][1]

    def test_gradient_nonnegative(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        for _, score in gradient:
            assert score >= 0.0

    def test_gradient_returns_all_edges(self):
        H = _make_simple_code()
        gradient = _compute_spectral_edge_gradient(H)
        num_edges = int(np.sum(H))
        assert len(gradient) == num_edges

    def test_gradient_edges_are_valid(self):
        H = _make_simple_code()
        m, n = H.shape
        gradient = _compute_spectral_edge_gradient(H)
        for (v, c), _ in gradient:
            assert 0 <= v < n
            assert n <= c < n + m
            assert H[c - n, v] != 0


# ── Test: Curvature Diagnostic ────────────────────────────────────────


class TestCurvatureDiagnostic:
    """Detect oscillatory BP traces."""

    def test_constant_trace(self):
        trace = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = _compute_belief_curvature(trace)
        assert result["max_abs_curvature"] == 0.0
        assert not result["oscillation_detected"]

    def test_linear_trace(self):
        trace = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _compute_belief_curvature(trace)
        assert result["max_abs_curvature"] == 0.0
        assert not result["oscillation_detected"]

    def test_oscillatory_trace(self):
        trace = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        result = _compute_belief_curvature(trace)
        assert result["max_abs_curvature"] > 0.0
        assert result["oscillation_detected"]
        assert result["num_sign_changes"] > 0

    def test_short_trace(self):
        result = _compute_belief_curvature([1.0, 2.0])
        assert result["curvature_values"] == []
        assert result["max_abs_curvature"] == 0.0

    def test_empty_trace(self):
        result = _compute_belief_curvature([])
        assert result["curvature_values"] == []

    def test_quadratic_trace(self):
        # x^2: [0, 1, 4, 9, 16] → second differences = [2, 2, 2]
        trace = [0.0, 1.0, 4.0, 9.0, 16.0]
        result = _compute_belief_curvature(trace)
        for c in result["curvature_values"]:
            assert abs(c - 2.0) < 1e-10
        assert not result["oscillation_detected"]


# ── Test: Ternary Classification ──────────────────────────────────────


class TestTernaryClassification:
    """Correct LLR → ternary mapping."""

    def test_strong_positive(self):
        labels = _classify_ternary_llr([5.0, 3.0, 2.0], threshold=1.0)
        assert labels == [1, 1, 1]

    def test_strong_negative(self):
        labels = _classify_ternary_llr([-5.0, -3.0, -2.0], threshold=1.0)
        assert labels == [-1, -1, -1]

    def test_metastable(self):
        labels = _classify_ternary_llr([0.5, -0.5, 0.0], threshold=1.0)
        assert labels == [0, 0, 0]

    def test_mixed(self):
        labels = _classify_ternary_llr([3.0, 0.0, -3.0], threshold=1.0)
        assert labels == [1, 0, -1]

    def test_boundary(self):
        labels = _classify_ternary_llr([1.0, -1.0], threshold=1.0)
        assert labels == [0, 0]

    def test_empty(self):
        labels = _classify_ternary_llr([])
        assert labels == []

    def test_cluster_stats(self):
        labels = [1, 1, 0, -1, 0]
        stats = _compute_ternary_cluster_stats(labels)
        assert stats["num_strong_zero"] == 2
        assert stats["num_strong_one"] == 1
        assert stats["num_metastable"] == 2
        assert stats["total"] == 5
        assert abs(stats["fraction_metastable"] - 0.4) < 1e-10


# ── Test: JSON Artifact Stability ─────────────────────────────────────


class TestJSONStability:
    """Serialization roundtrip stability."""

    def test_json_serializable(self):
        H = _make_simple_code()
        result = run_spectral_graph_optimization(H, max_iterations=2)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_json_roundtrip(self):
        H = _make_simple_code()
        result = run_spectral_graph_optimization(H, max_iterations=2)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized

    def test_all_output_keys_present(self):
        H = _make_larger_code()
        result = run_spectral_graph_optimization(H, max_iterations=3)
        expected_keys = {
            "initial_instability_score",
            "final_instability_score",
            "iterations",
            "swaps_applied",
            "spectral_radius_reduction",
            "optimizer_success",
            "curvature_events_detected",
            "ternary_clusters_detected",
            "initial_metrics",
            "final_metrics",
        }
        assert expected_keys.issubset(result.keys())

    def test_floats_rounded(self):
        H = _make_simple_code()
        result = run_spectral_graph_optimization(H, max_iterations=2)
        score = result["initial_instability_score"]
        assert isinstance(score, float)
        # Check 12-decimal rounding: multiply by 1e12 and check integer.
        assert score == round(score, 12)


# ── Test: Decoder Safety ──────────────────────────────────────────────


class TestDecoderSafety:
    """Optimizer module does NOT import src/qec/decoder/."""

    def test_no_decoder_import(self):
        import qec.experiments.spectral_graph_optimizer as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
        assert "from qec_qldpc_codes" not in source

    def test_no_bp_decode_import(self):
        import qec.experiments.spectral_graph_optimizer as mod
        source = open(mod.__file__).read()
        assert "import bp_decode" not in source


# ── Test: Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the optimizer."""

    def test_single_edge_matrix(self):
        H = np.array([[1]], dtype=np.float64)
        result = run_spectral_graph_optimization(H, max_iterations=2)
        assert result["iterations"] >= 0
        assert result["final_instability_score"] <= result["initial_instability_score"]

    def test_zero_iterations(self):
        H = _make_simple_code()
        result = run_spectral_graph_optimization(H, max_iterations=0)
        assert result["iterations"] == 0
        assert result["swaps_applied"] == []
        assert result["final_instability_score"] == result["initial_instability_score"]

    def test_with_bp_traces(self):
        H = _make_simple_code()
        traces = [
            [1.0, -1.0, 1.0, -1.0, 1.0],  # oscillatory
            [0.5, 0.6, 0.7, 0.8, 0.9],     # convergent
        ]
        result = run_spectral_graph_optimization(
            H, max_iterations=2, bp_traces=traces,
        )
        assert result["curvature_events_detected"] >= 0

    def test_with_ternary_llr(self):
        H = _make_simple_code()
        final_llr = [5.0, -3.0, 0.2, 2.0, -0.1]
        result = run_spectral_graph_optimization(
            H, max_iterations=2, final_llr=final_llr,
        )
        assert result["ternary_clusters_detected"] >= 0

    def test_optimizer_success_flag(self):
        H = _make_larger_code()
        result = run_spectral_graph_optimization(H, max_iterations=10)
        if result["final_instability_score"] < result["initial_instability_score"]:
            assert result["optimizer_success"] is True
        else:
            assert result["optimizer_success"] is False


# ── Test: Candidate Generation ────────────────────────────────────────


class TestCandidateGeneration:
    """Test guided repair candidate generation."""

    def test_candidates_generated(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        candidates = _generate_guided_repair_candidates(
            H, gradient, max_candidates=5,
        )
        assert isinstance(candidates, list)

    def test_candidates_have_required_keys(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        candidates = _generate_guided_repair_candidates(
            H, gradient, max_candidates=5,
        )
        for c in candidates:
            assert "remove" in c
            assert "add" in c
            assert "description" in c

    def test_max_candidates_respected(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        candidates = _generate_guided_repair_candidates(
            H, gradient, max_candidates=3,
        )
        assert len(candidates) <= 3

    def test_no_duplicate_edges_after_swap(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        candidates = _generate_guided_repair_candidates(
            H, gradient, max_candidates=10,
        )
        from qec.experiments.tanner_graph_repair import (
            _extract_edges, _apply_swap,
        )
        edges = _extract_edges(H)
        for c in candidates:
            new_edges = _apply_swap(edges, c)
            assert len(new_edges) == len(set(new_edges))

    def test_degree_preservation(self):
        H = _make_larger_code()
        gradient = _compute_spectral_edge_gradient(H)
        candidates = _generate_guided_repair_candidates(
            H, gradient, max_candidates=10,
        )
        from qec.experiments.tanner_graph_repair import (
            _extract_edges, _apply_swap,
        )
        edges = _extract_edges(H)

        def degree_map(edge_list):
            deg = {}
            for u, v in edge_list:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            return deg

        orig_deg = degree_map(edges)
        for c in candidates:
            new_edges = _apply_swap(edges, c)
            assert degree_map(new_edges) == orig_deg

    def test_empty_gradient(self):
        H = _make_simple_code()
        candidates = _generate_guided_repair_candidates(
            H, [], max_candidates=5,
        )
        assert candidates == []
