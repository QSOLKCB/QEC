"""
Tests for multi-step spectral graph repair loop (v7.3.x).

Verifies:
  - Determinism: repeated runs produce identical artifacts
  - Candidate validity: edge swaps preserve node degrees
  - Duplicate prevention: no duplicate edges after multi-step swaps
  - Depth behavior: depth=1 matches v7.3 single-step behavior
  - Pruning behavior: pruned branches skipped, best repair unchanged
  - JSON stability: artifact round-trip serialization
  - Decoder safety: no imports from src/qec/decoder/
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
    run_spectral_graph_repair_loop,
    _prune_branch,
    _generate_repair_sequences,
    _select_best_repair_sequence,
)
from qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _apply_swap,
    _edges_to_H,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_larger_code() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a larger parity-check matrix for multi-step tests.

    H = [[1,1,0,0,1,0,0],
         [0,1,1,0,0,1,0],
         [0,0,1,1,0,0,1],
         [1,0,0,1,0,1,0]]
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


def _make_risk_result():
    """Build risk result with clusters for the larger code."""
    return {
        "node_risk_scores": [[0, 0.8], [1, 0.9], [4, 0.5]],
        "cluster_risk_scores": [0.85, 0.3],
        "top_risk_clusters": [0, 1],
        "cluster_risk_ranking": [0, 1],
        "max_cluster_risk": 0.85,
        "mean_cluster_risk": 0.575,
        "num_high_risk_clusters": 2,
        "candidate_clusters": [
            {"variable_nodes": [0, 1], "check_nodes": [0, 1]},
            {"variable_nodes": [4], "check_nodes": [0]},
        ],
    }


_COMMON_KWARGS = {
    "nb_spectral_radius": 2.5,
    "spectral_instability_ratio": 1.2,
    "ipr_localization_score": 0.3,
    "avg_variable_degree": 2.0,
    "avg_check_degree": 3.5,
    "max_candidates": 5,
    "max_iters": 50,
}


# ── Test: Determinism ────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs produce identical results."""

    def test_multistep_determinism(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        r1 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=True,
            **_COMMON_KWARGS,
        )
        r2 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=True,
            **_COMMON_KWARGS,
        )

        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_multistep_no_pruning_determinism(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        r1 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=False,
            **_COMMON_KWARGS,
        )
        r2 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=False,
            **_COMMON_KWARGS,
        )

        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


# ── Test: Candidate Validity ─────────────────────────────────────────


class TestCandidateValidity:
    """Multi-step swaps preserve node degrees."""

    def test_degree_preservation_multistep(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()
        m, n = H.shape

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            **_COMMON_KWARGS,
        )

        if result["repair_applied"] and result["repair_sequence"]:
            edges = _extract_edges(H)
            for swap_step in result["repair_sequence"]:
                edges = _apply_swap(edges, swap_step)
            H_repaired = _edges_to_H(edges, m, n)

            baseline_var_degrees = np.sum(H, axis=0)
            baseline_check_degrees = np.sum(H, axis=1)
            repaired_var_degrees = np.sum(H_repaired, axis=0)
            repaired_check_degrees = np.sum(H_repaired, axis=1)

            np.testing.assert_array_equal(
                baseline_var_degrees, repaired_var_degrees,
            )
            np.testing.assert_array_equal(
                baseline_check_degrees, repaired_check_degrees,
            )


# ── Test: Duplicate Prevention ───────────────────────────────────────


class TestDuplicatePrevention:
    """No duplicate edges after multi-step swaps."""

    def test_no_duplicate_edges_multistep(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()
        m, n = H.shape

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            **_COMMON_KWARGS,
        )

        if result["repair_applied"] and result["repair_sequence"]:
            edges = _extract_edges(H)
            for swap_step in result["repair_sequence"]:
                edges = _apply_swap(edges, swap_step)
            assert len(set(edges)) == len(edges)


# ── Test: Depth Behavior ─────────────────────────────────────────────


class TestDepthBehavior:
    """depth=1 produces identical results to v7.3 single-step."""

    def test_depth_1_matches_single_step(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        single = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=False,
            max_repair_depth=1,
            **_COMMON_KWARGS,
        )
        multi_d1 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=1,
            **_COMMON_KWARGS,
        )

        # Core metrics must match.
        assert single["repair_applied"] == multi_d1["repair_applied"]
        assert single["score_improvement"] == multi_d1["score_improvement"]
        assert (
            single["spectral_instability_score_before"]
            == multi_d1["spectral_instability_score_before"]
        )
        assert (
            single["spectral_instability_score_after"]
            == multi_d1["spectral_instability_score_after"]
        )
        if single["best_swap"] is not None:
            assert (
                single["best_swap"]["description"]
                == multi_d1["best_swap"]["description"]
            )

    def test_multistep_disabled_ignores_depth(self):
        """When enable_multistep_repair=False, depth>1 is ignored."""
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        r1 = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=False,
            max_repair_depth=2,
            **_COMMON_KWARGS,
        )

        assert r1["repair_depth"] == 1
        assert r1["multistep_enabled"] is False


# ── Test: Pruning Behavior ───────────────────────────────────────────


class TestPruningBehavior:
    """Pruning skips branches but preserves the best repair."""

    def test_pruning_does_not_change_best(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        with_pruning = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=True,
            **_COMMON_KWARGS,
        )
        without_pruning = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=False,
            **_COMMON_KWARGS,
        )

        # Best repair must be identical.
        assert (
            with_pruning["score_improvement"]
            == without_pruning["score_improvement"]
        )
        assert (
            with_pruning["spectral_instability_score_after"]
            == without_pruning["spectral_instability_score_after"]
        )
        assert (
            with_pruning["repair_applied"]
            == without_pruning["repair_applied"]
        )

    def test_prune_branch_function(self):
        # Upper bound = 0.5 - 0.4 = 0.1, best = 0.15 → prune.
        assert _prune_branch(0.5, 0.4, 0.15) is True
        # Upper bound = 0.5 - 0.2 = 0.3, best = 0.15 → don't prune.
        assert _prune_branch(0.5, 0.2, 0.15) is False
        # Upper bound = best → prune (equal means no improvement).
        assert _prune_branch(0.5, 0.3, 0.2) is True

    def test_branches_pruned_reported(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            enable_pruning=True,
            **_COMMON_KWARGS,
        )

        assert "branches_pruned" in result
        assert isinstance(result["branches_pruned"], int)
        assert result["branches_pruned"] >= 0


# ── Test: JSON Stability ─────────────────────────────────────────────


class TestJSONStability:
    """Artifact round-trip serialization."""

    def test_multistep_json_serializable(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            **_COMMON_KWARGS,
        )

        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_multistep_artifact_keys(self):
        H, llr, s = _make_larger_code()
        risk = _make_risk_result()

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            **_COMMON_KWARGS,
        )

        multistep_keys = {
            "repair_depth",
            "repair_sequence",
            "sequence_length",
            "multistep_enabled",
            "pruning_enabled",
            "branches_pruned",
        }
        assert multistep_keys.issubset(result.keys())

    def test_no_repair_multistep_keys(self):
        """No-repair result also includes multistep fields."""
        H, llr, s = _make_larger_code()
        risk = {
            "node_risk_scores": [],
            "cluster_risk_scores": [],
            "top_risk_clusters": [],
            "cluster_risk_ranking": [],
            "max_cluster_risk": 0.0,
            "mean_cluster_risk": 0.0,
            "num_high_risk_clusters": 0,
        }

        result = run_spectral_graph_repair_loop(
            H, llr, s, risk,
            enable_multistep_repair=True,
            max_repair_depth=2,
            max_iters=50,
        )

        assert result["repair_applied"] is False
        assert result["repair_depth"] == 2
        assert result["multistep_enabled"] is True
        assert result["repair_sequence"] == []
        assert result["sequence_length"] == 0


# ── Test: Decoder Safety ─────────────────────────────────────────────


class TestDecoderSafety:
    """No decoder imports or modifications."""

    def test_no_decoder_imports(self):
        import qec.experiments.spectral_graph_repair_loop as module
        with open(module.__file__, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ── Test: Sequence Selection ─────────────────────────────────────────


class TestSequenceSelection:
    """_select_best_repair_sequence uses deterministic tie-breaking."""

    def test_selects_by_improvement(self):
        sequences = [
            {
                "repair_sequence": [
                    {"remove": [[0, 5]], "add": [[0, 6]],
                     "description": "swap A"},
                ],
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.5,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.2,
                "sequence_length": 1,
            },
            {
                "repair_sequence": [
                    {"remove": [[1, 5]], "add": [[1, 6]],
                     "description": "swap B"},
                    {"remove": [[2, 6]], "add": [[2, 7]],
                     "description": "swap C"},
                ],
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.3,
                "predicted_instability_before": True,
                "predicted_instability_after": False,
                "score_improvement": 0.4,
                "sequence_length": 2,
            },
        ]

        result = _select_best_repair_sequence(sequences)
        assert result["best_sequence_metrics"]["score_improvement"] == 0.4
        assert len(result["best_sequence"]) == 2

    def test_tiebreak_by_score_then_description(self):
        sequences = [
            {
                "repair_sequence": [
                    {"remove": [[0, 5]], "add": [[0, 6]],
                     "description": "swap B"},
                ],
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.5,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.2,
                "sequence_length": 1,
            },
            {
                "repair_sequence": [
                    {"remove": [[1, 5]], "add": [[1, 6]],
                     "description": "swap A"},
                ],
                "spectral_instability_score_before": 0.7,
                "spectral_instability_score_after": 0.5,
                "predicted_instability_before": True,
                "predicted_instability_after": True,
                "score_improvement": 0.2,
                "sequence_length": 1,
            },
        ]

        result = _select_best_repair_sequence(sequences)
        # Same improvement and score_after → lexicographic on description.
        assert result["best_sequence"][0]["description"] == "swap A"
