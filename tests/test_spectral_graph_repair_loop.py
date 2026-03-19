"""
Tests for spectral graph repair loop experiment (v7.3).

Verifies:
  - Determinism: repeated runs produce identical best repairs and artifacts
  - Candidate validity: degree preservation and no duplicate edges
  - Stable ordering: candidate ordering and tie-breaking are deterministic
  - Score improvement logic: selected best repair maximizes improvement
  - Empty / no-repair case: valid artifact with repair_applied=False
  - JSON stability: artifact serializes correctly
  - Decoder safety: no decoder imports or modifications
  - Pipeline integration: full spectral → predict → repair → decode stack
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

from qec.experiments.spectral_graph_repair_loop import (
    generate_repair_candidates,
    score_repair_candidate,
    select_best_repair,
    run_spectral_graph_repair_loop,
    compute_repair_loop_aggregate_metrics,
)
from qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _apply_swap,
    _edges_to_H,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_simple_code() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple parity-check matrix with LLR and syndrome.

    H = [[1,1,0,1,0],
         [0,1,1,0,1],
         [1,0,1,1,0]]

    Variable nodes: 0-4, Check nodes: 5-7.
    """
    H = np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)
    llr = np.array([2.0, -1.5, 0.5, 1.0, -0.8], dtype=np.float64)
    syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)
    return H, llr, syndrome_vec


def _make_larger_code() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a larger parity-check matrix for candidate generation tests.

    H = [[1,1,0,0,1,0,0],
         [0,1,1,0,0,1,0],
         [0,0,1,1,0,0,1],
         [1,0,0,1,0,1,0]]

    Variable nodes: 0-6, Check nodes: 7-10.
    """
    H = np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
    ], dtype=np.float64)
    llr = np.array([2.0, -1.5, 0.5, 1.0, -0.8, 0.3, -1.2], dtype=np.float64)
    syndrome_vec = np.array([0, 0, 0, 0], dtype=np.uint8)
    return H, llr, syndrome_vec


def _make_risk_result(
    node_risk_scores: list[list],
    cluster_risk_scores: list[float] | None = None,
    top_risk_clusters: list[int] | None = None,
    candidate_clusters: list[dict] | None = None,
) -> dict[str, object]:
    """Build a minimal v6.4-compatible risk result."""
    result = {
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores or [],
        "top_risk_clusters": top_risk_clusters or [],
        "cluster_risk_ranking": [],
        "max_cluster_risk": max((s for _, s in node_risk_scores), default=0.0),
        "mean_cluster_risk": 0.0,
        "num_high_risk_clusters": 0,
    }
    if candidate_clusters is not None:
        result["candidate_clusters"] = candidate_clusters
    return result


def _make_cluster_risk_result_larger():
    """Build risk result with clusters for the larger code."""
    return _make_risk_result(
        node_risk_scores=[[0, 0.8], [1, 0.9], [4, 0.5]],
        cluster_risk_scores=[0.85, 0.3],
        top_risk_clusters=[0, 1],
        candidate_clusters=[
            {"variable_nodes": [0, 1], "check_nodes": [0, 1]},
            {"variable_nodes": [4], "check_nodes": [0]},
        ],
    )


# ── Test: Determinism ────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs produce identical results."""

    def test_repair_loop_determinism(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result_1 = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=5,
            max_iters=50,
        )
        result_2 = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=5,
            max_iters=50,
        )

        json_1 = json.dumps(result_1, sort_keys=True)
        json_2 = json.dumps(result_2, sort_keys=True)
        assert json_1 == json_2

    def test_candidate_generation_determinism(self):
        H, _, _ = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        gen_1 = generate_repair_candidates(
            H,
            top_risk_clusters=risk["top_risk_clusters"],
            candidate_clusters=risk["candidate_clusters"],
            node_risk_scores=risk["node_risk_scores"],
            cluster_risk_scores=risk["cluster_risk_scores"],
            max_candidates=10,
        )
        gen_2 = generate_repair_candidates(
            H,
            top_risk_clusters=risk["top_risk_clusters"],
            candidate_clusters=risk["candidate_clusters"],
            node_risk_scores=risk["node_risk_scores"],
            cluster_risk_scores=risk["cluster_risk_scores"],
            max_candidates=10,
        )

        assert json.dumps(gen_1, sort_keys=True) == json.dumps(gen_2, sort_keys=True)


# ── Test: Candidate Validity ─────────────────────────────────────────


class TestCandidateValidity:
    """Every candidate preserves degrees and avoids duplicate edges."""

    def test_degree_preservation(self):
        H, _, _ = _make_larger_code()
        risk = _make_cluster_risk_result_larger()
        m, n = H.shape

        gen = generate_repair_candidates(
            H,
            top_risk_clusters=risk["top_risk_clusters"],
            candidate_clusters=risk["candidate_clusters"],
            node_risk_scores=risk["node_risk_scores"],
            cluster_risk_scores=risk["cluster_risk_scores"],
            max_candidates=20,
        )

        edges = _extract_edges(H)
        baseline_var_degrees = np.sum(H, axis=0)
        baseline_check_degrees = np.sum(H, axis=1)

        for candidate in gen["candidates"]:
            repaired_edges = _apply_swap(edges, candidate)
            H_repaired = _edges_to_H(repaired_edges, m, n)

            repaired_var_degrees = np.sum(H_repaired, axis=0)
            repaired_check_degrees = np.sum(H_repaired, axis=1)

            np.testing.assert_array_equal(
                baseline_var_degrees, repaired_var_degrees,
                err_msg=f"Variable degree mismatch for {candidate['description']}",
            )
            np.testing.assert_array_equal(
                baseline_check_degrees, repaired_check_degrees,
                err_msg=f"Check degree mismatch for {candidate['description']}",
            )

    def test_no_duplicate_edges(self):
        H, _, _ = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        gen = generate_repair_candidates(
            H,
            top_risk_clusters=risk["top_risk_clusters"],
            candidate_clusters=risk["candidate_clusters"],
            node_risk_scores=risk["node_risk_scores"],
            cluster_risk_scores=risk["cluster_risk_scores"],
            max_candidates=20,
        )

        edges = _extract_edges(H)

        for candidate in gen["candidates"]:
            repaired_edges = _apply_swap(edges, candidate)
            edge_set = set(repaired_edges)
            assert len(edge_set) == len(repaired_edges), (
                f"Duplicate edges after swap: {candidate['description']}"
            )

    def test_no_self_recreating_swaps(self):
        H, _, _ = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        gen = generate_repair_candidates(
            H,
            top_risk_clusters=risk["top_risk_clusters"],
            candidate_clusters=risk["candidate_clusters"],
            node_risk_scores=risk["node_risk_scores"],
            cluster_risk_scores=risk["cluster_risk_scores"],
            max_candidates=20,
        )

        edges = _extract_edges(H)
        edge_set = set(edges)

        for candidate in gen["candidates"]:
            add_0 = tuple(candidate["add"][0])
            add_1 = tuple(candidate["add"][1])
            assert add_0 not in edge_set, (
                f"Added edge already exists: {candidate['description']}"
            )
            assert add_1 not in edge_set, (
                f"Added edge already exists: {candidate['description']}"
            )


# ── Test: Stable Ordering ────────────────────────────────────────────


class TestStableOrdering:
    """Candidate ordering and tie-breaking are deterministic."""

    def test_select_best_deterministic_tiebreak(self):
        scored = [
            {
                "swap": {"remove": [[0, 5]], "add": [[0, 6]], "description": "swap B"},
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.5,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.2,
                "repaired_spectral_radius": 2.0,
            },
            {
                "swap": {"remove": [[1, 5]], "add": [[1, 6]], "description": "swap A"},
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.5,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.2,
                "repaired_spectral_radius": 1.8,
            },
        ]

        result = select_best_repair(scored)
        # Same improvement and same score_after → lexicographic tiebreak.
        assert result["best_swap"]["description"] == "swap A"

    def test_select_best_by_improvement(self):
        scored = [
            {
                "swap": {"remove": [[0, 5]], "add": [[0, 6]], "description": "swap B"},
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.6,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.1,
                "repaired_spectral_radius": 2.0,
            },
            {
                "swap": {"remove": [[1, 5]], "add": [[1, 6]], "description": "swap A"},
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.4,
                "predicted_instability_before": True,
                "predicted_instability_after": False,
                "score_improvement": 0.3,
                "repaired_spectral_radius": 1.8,
            },
        ]

        result = select_best_repair(scored)
        assert result["best_swap"]["description"] == "swap A"
        assert result["best_candidate_metrics"]["score_improvement"] == 0.3


# ── Test: Score Improvement Logic ────────────────────────────────────


class TestScoreImprovement:
    """Selected best repair maximizes deterministic improvement."""

    def test_best_maximizes_improvement(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=10,
            max_iters=50,
        )

        if result["repair_applied"]:
            assert result["score_improvement"] > 0.0
            assert (
                result["spectral_instability_score_after"]
                < result["spectral_instability_score_before"]
            )


# ── Test: Empty / No-Repair Case ────────────────────────────────────


class TestNoRepairCase:
    """When no valid candidate improves score, report repair_applied=False."""

    def test_empty_risk_result(self):
        H, llr, syndrome_vec = _make_simple_code()
        risk = _make_risk_result([], [], [])

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            max_iters=50,
        )

        assert result["repair_applied"] is False
        assert result["best_swap"] is None
        assert result["score_improvement"] == 0.0
        assert result["num_candidates_evaluated"] == 0

    def test_no_top_risk_clusters(self):
        H, llr, syndrome_vec = _make_simple_code()
        risk = _make_risk_result(
            node_risk_scores=[[0, 0.5]],
            cluster_risk_scores=[0.3],
            top_risk_clusters=[],
        )

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            max_iters=50,
        )

        assert result["repair_applied"] is False

    def test_no_decode_comparison(self):
        H, llr, syndrome_vec = _make_simple_code()
        risk = _make_risk_result([], [], [])

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            max_iters=50,
            enable_decode_comparison=False,
        )

        assert result["repair_applied"] is False
        assert result["decode_success_before"] is False
        assert result["decode_success_after"] is False
        assert result["iterations_before"] == 0
        assert result["iterations_after"] == 0


# ── Test: JSON Stability ─────────────────────────────────────────────


class TestJSONStability:
    """Artifact serializes correctly."""

    def test_json_serializable(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=5,
            max_iters=50,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_required_keys_present(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=5,
            max_iters=50,
        )

        required_keys = {
            "repair_applied",
            "best_swap",
            "num_candidates_evaluated",
            "spectral_instability_score_before",
            "spectral_instability_score_after",
            "score_improvement",
            "predicted_instability_before",
            "predicted_instability_after",
            "decode_success_before",
            "decode_success_after",
            "delta_success",
            "iterations_before",
            "iterations_after",
            "delta_iterations",
            "cluster_nodes",
            "cluster_risk_score",
        }

        assert required_keys.issubset(result.keys())

    def test_no_repair_keys_present(self):
        H, llr, syndrome_vec = _make_simple_code()
        risk = _make_risk_result([], [], [])

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            max_iters=50,
        )

        required_keys = {
            "repair_applied",
            "best_swap",
            "num_candidates_evaluated",
            "spectral_instability_score_before",
            "spectral_instability_score_after",
            "score_improvement",
            "predicted_instability_before",
            "predicted_instability_after",
            "decode_success_before",
            "decode_success_after",
            "delta_success",
            "iterations_before",
            "iterations_after",
            "delta_iterations",
            "cluster_nodes",
            "cluster_risk_score",
        }

        assert required_keys.issubset(result.keys())


# ── Test: Decoder Safety ─────────────────────────────────────────────


class TestDecoderSafety:
    """No decoder imports or modifications."""

    def test_no_decoder_imports(self):
        import qec.experiments.spectral_graph_repair_loop as module
        source_file = module.__file__
        with open(source_file, "r") as f:
            source = f.read()

        # Must not import from decoder module.
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ── Test: Pipeline Integration ───────────────────────────────────────


class TestPipelineIntegration:
    """Full stack: diagnostics → predict → repair → decode."""

    def test_full_pipeline_with_decode(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=10,
            max_iters=50,
            enable_decode_comparison=True,
        )

        assert isinstance(result["repair_applied"], bool)
        assert isinstance(result["spectral_instability_score_before"], float)
        assert isinstance(result["spectral_instability_score_after"], float)
        assert isinstance(result["decode_success_before"], bool)
        assert isinstance(result["decode_success_after"], bool)
        assert isinstance(result["iterations_before"], int)
        assert isinstance(result["iterations_after"], int)

    def test_full_pipeline_without_decode(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=10,
            max_iters=50,
            enable_decode_comparison=False,
        )

        assert isinstance(result["repair_applied"], bool)
        assert isinstance(result["spectral_instability_score_before"], float)


# ── Test: Aggregate Metrics ──────────────────────────────────────────


class TestAggregateMetrics:
    """Aggregate phase diagram metrics are computed correctly."""

    def test_aggregate_empty(self):
        metrics = compute_repair_loop_aggregate_metrics([])
        assert metrics["num_trials"] == 0
        assert metrics["mean_repair_score_improvement"] == 0.0
        assert metrics["repair_activation_rate"] == 0.0

    def test_aggregate_basic(self):
        trials = [
            {
                "repair_applied": True,
                "score_improvement": 0.1,
                "predicted_instability_before": True,
                "predicted_instability_after": False,
                "spectral_instability_score_after": 0.4,
                "delta_iterations": -2,
                "delta_success": 1,
                "num_candidates_evaluated": 5,
            },
            {
                "repair_applied": False,
                "score_improvement": 0.0,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "spectral_instability_score_after": 0.6,
                "delta_iterations": 0,
                "delta_success": 0,
                "num_candidates_evaluated": 3,
            },
        ]

        metrics = compute_repair_loop_aggregate_metrics(trials)
        assert metrics["num_trials"] == 2
        assert metrics["repair_activation_rate"] == 0.5
        assert metrics["repair_prediction_flip_rate"] == 0.5
        assert metrics["mean_repair_score_improvement"] == round(0.05, 12)
        assert metrics["mean_repaired_instability_score"] == 0.5
        assert metrics["mean_candidates_evaluated"] == 4.0
        assert metrics["repair_decode_improvement_rate"] == 0.5


# ── Test: Float Rounding ─────────────────────────────────────────────


class TestFloatRounding:
    """All float outputs rounded to 12 decimals."""

    def test_result_float_precision(self):
        H, llr, syndrome_vec = _make_larger_code()
        risk = _make_cluster_risk_result_larger()

        result = run_spectral_graph_repair_loop(
            H, llr, syndrome_vec, risk,
            nb_spectral_radius=2.5,
            spectral_instability_ratio=1.2,
            ipr_localization_score=0.3,
            avg_variable_degree=2.0,
            avg_check_degree=3.5,
            max_candidates=5,
            max_iters=50,
        )

        for key in [
            "spectral_instability_score_before",
            "spectral_instability_score_after",
            "score_improvement",
            "cluster_risk_score",
        ]:
            val = result[key]
            assert val == round(val, 12), f"{key} not rounded to 12 decimals"
