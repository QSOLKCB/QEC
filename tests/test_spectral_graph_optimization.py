"""
Tests for spectral Tanner graph optimization experiment (v6.7).

Verifies:
  - spectral_score computation
  - Deterministic candidate selection
  - Node degree preservation
  - No duplicate edges
  - Spectral score decreases when swap accepted
  - JSON output stability
  - Compatibility with v6.6 repair pipeline
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

from src.qec.experiments.tanner_graph_repair import (
    run_spectral_graph_optimization_experiment,
    spectral_score,
    _build_nb_matrix_from_edges,
    _build_nb_context,
    _spectral_score_with_swap,
    _spectral_radius_from_nb,
    _spectral_radius_power_iteration,
    _power_iteration_spectral_radius,
    _extract_edges,
    _build_edge_set,
    _build_adjacency,
    _get_cluster_edges,
    _get_boundary_edges,
    _generate_candidate_swaps,
    _apply_swap,
    _edges_to_H,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_risk_result(
    node_risk_scores: list[list],
    cluster_risk_scores: list[float] | None = None,
    top_risk_clusters: list[int] | None = None,
) -> dict[str, object]:
    """Build a minimal v6.4-compatible risk result."""
    return {
        "node_risk_scores": node_risk_scores,
        "cluster_risk_scores": cluster_risk_scores or [],
        "top_risk_clusters": top_risk_clusters or [],
        "cluster_risk_ranking": [],
        "max_cluster_risk": max((s for _, s in node_risk_scores), default=0.0),
        "mean_cluster_risk": 0.0,
        "num_high_risk_clusters": 0,
    }


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
    llr = np.array([1.5, -1.0, 0.8, -0.5, 1.2, -0.3, 0.7], dtype=np.float64)
    syndrome_vec = np.array([0, 0, 0, 0], dtype=np.uint8)
    return H, llr, syndrome_vec


# ── Test classes ──────────────────────────────────────────────────────


class TestSpectralScore:
    """Tests for spectral_score computation."""

    def test_empty_graph(self):
        """Empty edge list → spectral score 0."""
        assert spectral_score([]) == 0.0

    def test_single_edge(self):
        """Single edge → small non-backtracking matrix."""
        edges = [(0, 1)]
        score = spectral_score(edges)
        # Single edge produces 2 directed edges.
        # v6.0-compatible construction: spectral radius = 1.0.
        assert score == 1.0

    def test_positive_for_nontrivial_graph(self):
        """Nontrivial graph has positive spectral score."""
        H, _, _ = _make_simple_code()
        edges = _extract_edges(H)
        score = spectral_score(edges)
        assert score > 0.0

    def test_deterministic(self):
        """Repeated calls produce identical results."""
        H, _, _ = _make_simple_code()
        edges = _extract_edges(H)
        s1 = spectral_score(edges)
        s2 = spectral_score(edges)
        assert s1 == s2

    def test_returns_float(self):
        """Spectral score returns a float."""
        H, _, _ = _make_simple_code()
        edges = _extract_edges(H)
        score = spectral_score(edges)
        assert isinstance(score, float)

    def test_consistent_with_nb_spectrum(self):
        """Spectral score is consistent with the v6.0 NB spectrum module."""
        from src.qec.diagnostics.non_backtracking_spectrum import (
            compute_non_backtracking_spectrum,
        )
        H, _, _ = _make_simple_code()
        edges = _extract_edges(H)

        # v6.0 exact computation.
        nb_result = compute_non_backtracking_spectrum(H)
        exact_radius = nb_result["spectral_radius"]

        # v6.7 power iteration estimate.
        estimated_radius = spectral_score(edges)

        # Power iteration converges to the dominant eigenvalue magnitude.
        # Allow small tolerance for iterative estimation.
        assert abs(estimated_radius - exact_radius) < 0.1, (
            f"Power iteration estimate {estimated_radius} differs from "
            f"exact spectral radius {exact_radius} by more than 0.1"
        )


class TestNBMatrixConstruction:
    """Tests for non-backtracking matrix construction."""

    def test_empty_edges(self):
        """Empty edge list → empty matrix."""
        B = _build_nb_matrix_from_edges([])
        assert B.shape == (0, 0)

    def test_single_edge(self):
        """Single edge → 2x2 matrix."""
        B = _build_nb_matrix_from_edges([(0, 1)])
        assert B.shape == (2, 2)
        # v6.0-compatible construction: identity for single edge.
        assert np.allclose(B.toarray(), np.eye(2))

    def test_triangle(self):
        """Triangle graph has correct NB matrix size."""
        edges = [(0, 1), (1, 2), (0, 2)]
        B = _build_nb_matrix_from_edges(edges)
        # 3 edges → 6 directed edges.
        assert B.shape == (6, 6)
        # Non-zero entries exist for triangles.
        assert np.sum(B) > 0


class TestPowerIteration:
    """Tests for power iteration spectral radius estimation."""

    def test_identity(self):
        """Identity matrix has spectral radius 1."""
        M = np.eye(5, dtype=np.float64)
        r = _power_iteration_spectral_radius(M)
        assert abs(r - 1.0) < 1e-6

    def test_zero_matrix(self):
        """Zero matrix has spectral radius 0."""
        M = np.zeros((5, 5), dtype=np.float64)
        r = _power_iteration_spectral_radius(M)
        assert r == 0.0

    def test_empty_matrix(self):
        """Empty matrix has spectral radius 0."""
        M = np.zeros((0, 0), dtype=np.float64)
        r = _power_iteration_spectral_radius(M)
        assert r == 0.0

    def test_known_eigenvalue(self):
        """Diagonal matrix with known largest eigenvalue."""
        M = np.diag([3.0, 1.0, 2.0])
        r = _power_iteration_spectral_radius(M)
        assert abs(r - 3.0) < 1e-4

    def test_deterministic(self):
        """Repeated calls produce identical results."""
        M = np.array([[2, 1], [1, 3]], dtype=np.float64)
        r1 = _power_iteration_spectral_radius(M)
        r2 = _power_iteration_spectral_radius(M)
        assert r1 == r2


class TestIncrementalNBUpdate:
    """Tests that incremental NB matrix update matches full reconstruction."""

    def test_incremental_matches_full(self):
        """Incremental update produces same spectral score as full rebuild."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        B_base, directed, outgoing, edge_to_dir = _build_nb_context(edges)

        for swap in candidates:
            # Full rebuild path.
            trial_edges = _apply_swap(edges, swap)
            full_score = spectral_score(trial_edges)

            # Incremental path.
            incr_score = _spectral_score_with_swap(
                B_base, directed, outgoing, edge_to_dir, swap,
            )

            assert full_score == incr_score, (
                f"Mismatch for {swap['description']}: "
                f"full={full_score}, incremental={incr_score}"
            )

    def test_incremental_deterministic(self):
        """Incremental scores are deterministic across repeated calls."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=5,
        )

        B_base, directed, outgoing, edge_to_dir = _build_nb_context(edges)

        for swap in candidates:
            s1 = _spectral_score_with_swap(
                B_base, directed, outgoing, edge_to_dir, swap,
            )
            s2 = _spectral_score_with_swap(
                B_base, directed, outgoing, edge_to_dir, swap,
            )
            assert s1 == s2

    def test_base_matrix_not_mutated(self):
        """Incremental update does not mutate the base matrix."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=5,
        )
        if not candidates:
            pytest.skip("No candidates generated")

        B_base, directed, outgoing, edge_to_dir = _build_nb_context(edges)
        B_copy = B_base.copy()

        _spectral_score_with_swap(
            B_base, directed, outgoing, edge_to_dir, candidates[0],
        )

        np.testing.assert_array_equal(B_base.toarray(), B_copy.toarray())

    def test_nb_context_returns_correct_shapes(self):
        """_build_nb_context returns correctly shaped structures."""
        H, _, _ = _make_simple_code()
        edges = _extract_edges(H)

        B, directed, outgoing, edge_to_dir = _build_nb_context(edges)

        assert B.shape == (2 * len(edges), 2 * len(edges))
        assert len(directed) == 2 * len(edges)
        assert len(edge_to_dir) == len(edges)

    def test_incremental_matches_full_simple_code(self):
        """Incremental update matches full rebuild on simple code."""
        H, _, _ = _make_simple_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 5}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        B_base, directed, outgoing, edge_to_dir = _build_nb_context(edges)

        for swap in candidates:
            trial_edges = _apply_swap(edges, swap)
            full_score = spectral_score(trial_edges)
            incr_score = _spectral_score_with_swap(
                B_base, directed, outgoing, edge_to_dir, swap,
            )
            assert full_score == incr_score

    def test_experiment_results_unchanged(self):
        """Full experiment produces identical results with optimization."""
        H, llr, s = _make_larger_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.8], [2, 0.3], [3, 0.2]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
        )

        r1 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )
        r2 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )

        assert r1["spectral_score_before"] == r2["spectral_score_before"]
        assert r1["spectral_score_after"] == r2["spectral_score_after"]
        assert r1["spectral_improvement"] == r2["spectral_improvement"]
        assert r1["best_swap"] == r2["best_swap"]
        assert r1["candidate_swaps"] == r2["candidate_swaps"]


class TestCandidateSwapSpectral:
    """Tests that candidate swaps work with spectral scoring."""

    def test_swaps_preserve_degrees(self):
        """Node degrees are identical before and after swap."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        def degree_map(edge_list):
            deg = {}
            for u, v in edge_list:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            return deg

        original_degrees = degree_map(edges)

        for swap in candidates:
            new_edges = _apply_swap(edges, swap)
            new_degrees = degree_map(new_edges)
            assert new_degrees == original_degrees

    def test_no_duplicate_edges(self):
        """Applied swap produces no duplicate edges."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        for swap in candidates:
            new_edges = _apply_swap(edges, swap)
            assert len(new_edges) == len(set(new_edges))

    def test_spectral_score_computable_after_swap(self):
        """Spectral score is computable after each swap."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        for swap in candidates:
            new_edges = _apply_swap(edges, swap)
            score = spectral_score(new_edges)
            assert isinstance(score, float)
            assert score >= 0.0


class TestSpectralScoreDecrease:
    """Tests that spectral score decreases when a swap is accepted."""

    def test_improvement_nonnegative(self):
        """Spectral improvement is non-negative when a swap is selected."""
        H, llr, s = _make_larger_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.8], [2, 0.3], [3, 0.2]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )

        assert result["spectral_improvement"] >= 0.0
        assert result["spectral_score_after"] <= result["spectral_score_before"]


class TestExperimentOutputKeys:
    """Tests that experiment returns all required output keys."""

    def test_all_keys_present(self):
        """Experiment returns all expected output keys."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5], [2, 0.2]],
            cluster_risk_scores=[0.8],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        expected_keys = {
            "baseline_metrics", "optimized_metrics",
            "delta_iterations", "delta_success",
            "best_swap", "candidate_swaps",
            "spectral_score_before", "spectral_score_after",
            "spectral_improvement",
            "cluster_nodes",
            "node_risk_scores", "cluster_risk_scores", "top_risk_clusters",
        }
        assert expected_keys.issubset(result.keys())

    def test_baseline_metrics_structure(self):
        """Baseline metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        bm = result["baseline_metrics"]
        assert "iterations" in bm
        assert "success" in bm
        assert "residual_norms" in bm
        assert "final_residual_norm" in bm
        assert isinstance(bm["iterations"], int)
        assert isinstance(bm["success"], bool)

    def test_optimized_metrics_structure(self):
        """Optimized metrics contain expected fields."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        om = result["optimized_metrics"]
        assert "iterations" in om
        assert "success" in om
        assert "residual_norms" in om
        assert "final_residual_norm" in om

    def test_delta_consistency(self):
        """Delta values are consistent with metrics."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.8]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=20,
        )

        assert result["delta_iterations"] == (
            result["optimized_metrics"]["iterations"]
            - result["baseline_metrics"]["iterations"]
        )
        assert result["delta_success"] == (
            int(result["optimized_metrics"]["success"])
            - int(result["baseline_metrics"]["success"])
        )

    def test_no_risk_clusters(self):
        """Empty top_risk_clusters → baseline only."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result([], top_risk_clusters=[])

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert result["best_swap"] is None
        assert result["candidate_swaps"] == []
        assert result["spectral_improvement"] == 0.0
        assert "baseline_metrics" in result
        assert "optimized_metrics" in result

    def test_risk_passthrough(self):
        """Risk result fields are passed through to output."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5]],
            cluster_risk_scores=[0.8, 0.3],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert result["node_risk_scores"] == [[0, 1.0], [1, 0.5]]
        assert result["cluster_risk_scores"] == [0.8, 0.3]
        assert result["top_risk_clusters"] == [0]

    def test_spectral_scores_are_floats(self):
        """Spectral score fields are floats."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert isinstance(result["spectral_score_before"], float)
        assert isinstance(result["spectral_score_after"], float)
        assert isinstance(result["spectral_improvement"], float)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5], [2, 0.2]],
            cluster_risk_scores=[0.7],
            top_risk_clusters=[0],
        )

        r1 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=15,
        )
        r2 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=15,
        )

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is fully JSON-serializable."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [2, 0.8]],
            top_risk_clusters=[0],
        )

        result = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10,
        )

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_deterministic_candidate_selection(self):
        """Best swap selection is deterministic across runs."""
        H, llr, s = _make_larger_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.8], [2, 0.3], [3, 0.2]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
        )

        r1 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )
        r2 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )

        assert r1["best_swap"] == r2["best_swap"]
        assert r1["spectral_improvement"] == r2["spectral_improvement"]


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """Module does not import any decoder code."""
        import src.qec.experiments.tanner_graph_repair as mod
        source = open(mod.__file__).read()
        assert "import bp_decode" not in source
        assert "from src.qec.decoder" not in source
        assert "from src.qec_qldpc_codes" not in source


class TestV66Compatibility:
    """Tests compatibility with v6.6 repair pipeline."""

    def test_v66_still_works(self):
        """v6.6 repair experiment still works after v6.7 additions."""
        from src.qec.experiments.tanner_graph_repair import (
            run_tanner_graph_repair_experiment,
        )

        H, llr, s = _make_simple_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.5], [2, 0.2]],
            cluster_risk_scores=[0.8],
            top_risk_clusters=[0],
        )

        result = run_tanner_graph_repair_experiment(
            H, llr, s, risk, max_iters=10,
        )

        assert "baseline_metrics" in result
        assert "repaired_metrics" in result
        assert "repair_score_improvement" in result

    def test_shared_candidate_logic(self):
        """v6.6 and v6.7 produce same candidate swaps for same input."""
        from src.qec.experiments.tanner_graph_repair import (
            run_tanner_graph_repair_experiment,
        )

        H, llr, s = _make_larger_code()
        risk = _make_risk_result(
            [[0, 1.0], [1, 0.8], [2, 0.3], [3, 0.2]],
            cluster_risk_scores=[0.9],
            top_risk_clusters=[0],
        )

        r66 = run_tanner_graph_repair_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )
        r67 = run_spectral_graph_optimization_experiment(
            H, llr, s, risk, max_iters=10, max_candidates=10,
        )

        # Same candidate swap descriptions (same generation logic).
        descs_66 = [c["description"] for c in r66["candidate_swaps"]]
        descs_67 = [c["description"] for c in r67["candidate_swaps"]]
        assert descs_66 == descs_67


class TestPowerIterationCorrectness:
    """Tests that power iteration matches full eigenvalue computation."""

    def test_matches_eigvals_simple(self):
        """Power iteration result matches eigvals on a simple matrix."""
        M = np.array([[2, 1], [1, 3]], dtype=np.float64)
        rho_power = _spectral_radius_power_iteration(M)
        eigvals = np.linalg.eigvals(M)
        rho_eig = float(np.max(np.abs(eigvals)))
        assert abs(rho_power - rho_eig) < 1e-4

    def test_matches_eigvals_nb_matrix(self):
        """Power iteration matches eigvals on an NB matrix from a real graph."""
        H, _, _ = _make_larger_code()
        edges = _extract_edges(H)
        B = _build_nb_matrix_from_edges(edges)
        rho_power = _spectral_radius_power_iteration(B)
        eigvals = np.linalg.eigvals(B.toarray())
        rho_eig = float(np.max(np.abs(eigvals)))
        assert abs(rho_power - rho_eig) < 1e-4

    def test_matches_eigvals_diagonal(self):
        """Power iteration matches eigvals on a diagonal matrix."""
        M = np.diag([5.0, 3.0, 1.0, 4.0])
        rho_power = _spectral_radius_power_iteration(M)
        assert abs(rho_power - 5.0) < 1e-4

    def test_zero_matrix(self):
        """Power iteration returns 0 for zero matrix."""
        M = np.zeros((4, 4), dtype=np.float64)
        assert _spectral_radius_power_iteration(M) == 0.0


class TestPowerIterationDeterminism:
    """Tests that power iteration is fully deterministic."""

    def test_identical_results(self):
        """Two calls with identical input produce identical output."""
        M = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=np.float64)
        r1 = _spectral_radius_power_iteration(M)
        r2 = _spectral_radius_power_iteration(M)
        assert r1 == r2

    def test_identical_on_nb_matrix(self):
        """Deterministic on NB matrix from real graph."""
        H, _, _ = _make_larger_code()
        edges = _extract_edges(H)
        B = _build_nb_matrix_from_edges(edges)
        r1 = _spectral_radius_power_iteration(B)
        r2 = _spectral_radius_power_iteration(B)
        assert r1 == r2


class TestIncrementalSwapWithPowerIteration:
    """Tests that swap scoring with power iteration matches full rebuild."""

    def test_swap_consistency(self):
        """Incremental swap score matches full rebuild score."""
        H, _, _ = _make_larger_code()
        n = H.shape[1]
        edges = _extract_edges(H)
        edge_set = _build_edge_set(edges)
        adjacency = _build_adjacency(edges)

        cluster_nodes = {0, 1, 7, 8, 10}
        c_edges = _get_cluster_edges(edges, cluster_nodes)
        b_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

        candidates = _generate_candidate_swaps(
            c_edges, b_edges, edge_set, n, max_candidates=10,
        )

        B_base, directed, outgoing, edge_to_dir = _build_nb_context(edges)

        for swap in candidates:
            trial_edges = _apply_swap(edges, swap)
            full_score = spectral_score(trial_edges)
            incr_score = _spectral_score_with_swap(
                B_base, directed, outgoing, edge_to_dir, swap,
            )
            assert full_score == incr_score, (
                f"Mismatch for {swap['description']}: "
                f"full={full_score}, incremental={incr_score}"
            )


class TestEndToEndWithV64:
    """Tests end-to-end pipeline from v6.4 risk scoring to spectral optimization."""

    def test_full_pipeline(self):
        """Full pipeline: risk scoring -> spectral graph optimization."""
        from src.qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from src.qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )
        from src.qec.diagnostics.spectral_bp_alignment import (
            compute_spectral_bp_alignment,
        )
        from src.qec.diagnostics.spectral_failure_risk import (
            compute_spectral_failure_risk,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)

        loc = compute_nb_localization_metrics(H)
        trapping = compute_nb_trapping_candidates(H, loc)
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)
        risk = compute_spectral_failure_risk(loc, trapping, alignment)

        llr = np.array([2.0, -1.5, 0.5, 1.0], dtype=np.float64)
        syndrome_vec = np.array([0, 0, 0], dtype=np.uint8)

        result = run_spectral_graph_optimization_experiment(
            H, llr, syndrome_vec, risk, max_iters=20,
        )

        # All required keys present.
        assert "baseline_metrics" in result
        assert "optimized_metrics" in result
        assert "delta_iterations" in result
        assert "delta_success" in result
        assert "candidate_swaps" in result
        assert "spectral_score_before" in result
        assert "spectral_score_after" in result
        assert "spectral_improvement" in result

        # Spectral scores are valid.
        assert result["spectral_score_before"] >= 0.0
        assert result["spectral_score_after"] >= 0.0
        assert result["spectral_improvement"] >= 0.0

        # JSON-serializable.
        json.dumps(result)
