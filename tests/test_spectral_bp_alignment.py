"""
Tests for spectral–BP attractor alignment diagnostics (v6.3).

Verifies:
  - Alignment score computation on toy inputs
  - Deterministic thresholding of BP-active nodes
  - Cluster alignment computation correctness
  - JSON serialization stability
  - Compatibility with v6.2 candidate outputs
  - No decoder behavior changes (no decoder import)
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

from qec.diagnostics.spectral_bp_alignment import (
    compute_spectral_bp_alignment,
)


# ── Toy trapping-candidate outputs ────────────────────────────────────

def _make_trapping_result(
    cand_vars: list[int],
    clusters: list[dict[str, list[int]]] | None = None,
) -> dict[str, object]:
    """Build a minimal v6.2-compatible trapping candidate result."""
    if clusters is None:
        clusters = []
    return {
        "candidate_variable_nodes": cand_vars,
        "candidate_check_nodes": [],
        "candidate_clusters": clusters,
    }


class TestAlignmentScore:
    """Tests for global alignment score computation."""

    def test_perfect_overlap(self):
        """Candidate nodes == BP-active nodes → Jaccard = 1.0."""
        trapping = _make_trapping_result([0, 1, 2])
        bp_scores = {0: 10.0, 1: 8.0, 2: 6.0, 3: 0.5}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["spectral_bp_alignment_score"] == 1.0
        assert result["candidate_node_overlap_fraction"] == 1.0
        assert result["bp_node_overlap_fraction"] == 1.0

    def test_no_overlap(self):
        """Disjoint sets → Jaccard = 0.0."""
        trapping = _make_trapping_result([0, 1])
        # Nodes 2, 3 are active, 0, 1 are inactive.
        bp_scores = {0: 0.0, 1: 0.0, 2: 10.0, 3: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["spectral_bp_alignment_score"] == 0.0
        assert result["candidate_node_overlap_fraction"] == 0.0

    def test_partial_overlap(self):
        """Partial overlap gives expected Jaccard."""
        trapping = _make_trapping_result([0, 1, 2])
        # Node 2 is the only active candidate.
        bp_scores = {0: 0.0, 1: 0.0, 2: 10.0, 3: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        # intersection = {2}, union = {0, 1, 2, 3}, Jaccard = 1/4
        assert abs(result["spectral_bp_alignment_score"] - 0.25) < 1e-10
        # candidate overlap: 1/3
        assert abs(result["candidate_node_overlap_fraction"] - 1.0/3.0) < 1e-10
        # bp overlap: 1/2
        assert abs(result["bp_node_overlap_fraction"] - 0.5) < 1e-10

    def test_empty_candidates(self):
        """Empty candidate set → alignment = 0.0."""
        trapping = _make_trapping_result([])
        bp_scores = {0: 10.0, 1: 5.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        # union = {0, 1}, intersection = {}, Jaccard = 0
        assert result["spectral_bp_alignment_score"] == 0.0
        assert result["candidate_node_overlap_fraction"] == 0.0

    def test_empty_bp_scores(self):
        """No BP scores → no active nodes → alignment = 0.0."""
        trapping = _make_trapping_result([0, 1])
        result = compute_spectral_bp_alignment(trapping, {})

        assert result["spectral_bp_alignment_score"] == 0.0
        assert result["active_bp_nodes"] == []

    def test_both_empty(self):
        """Both empty → alignment = 0.0."""
        trapping = _make_trapping_result([])
        result = compute_spectral_bp_alignment(trapping, {})

        assert result["spectral_bp_alignment_score"] == 0.0


class TestThresholding:
    """Tests for deterministic BP-active node thresholding."""

    def test_default_threshold(self):
        """Default threshold (10% of max) selects correct nodes."""
        trapping = _make_trapping_result([])
        # max = 100, threshold = 10.0
        bp_scores = {0: 100.0, 1: 50.0, 2: 10.0, 3: 5.0, 4: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        # Nodes with score >= 10.0: 0, 1, 2
        assert result["active_bp_nodes"] == [0, 1, 2]

    def test_custom_threshold(self):
        """Custom threshold fraction works correctly."""
        trapping = _make_trapping_result([])
        bp_scores = {0: 100.0, 1: 50.0, 2: 10.0}
        result = compute_spectral_bp_alignment(
            trapping, bp_scores,
            activity_threshold_fraction=0.5,
        )

        # threshold = 50.0, nodes >= 50.0: 0, 1
        assert result["active_bp_nodes"] == [0, 1]

    def test_all_zero_scores(self):
        """All-zero scores → no active nodes."""
        trapping = _make_trapping_result([0])
        bp_scores = {0: 0.0, 1: 0.0, 2: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["active_bp_nodes"] == []

    def test_threshold_stores_param(self):
        """Activity threshold fraction is stored in output."""
        trapping = _make_trapping_result([])
        result = compute_spectral_bp_alignment(
            trapping, {0: 1.0},
            activity_threshold_fraction=0.25,
        )
        assert result["activity_threshold_fraction"] == 0.25


class TestClusterAlignment:
    """Tests for per-cluster alignment computation."""

    def test_single_cluster_full_overlap(self):
        """Cluster with all nodes active → alignment = 1.0."""
        trapping = _make_trapping_result(
            [0, 1],
            clusters=[{"variable_nodes": [0, 1], "check_nodes": []}],
        )
        bp_scores = {0: 10.0, 1: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["per_cluster_alignment_scores"] == [1.0]
        assert result["max_cluster_alignment"] == 1.0
        assert result["top_aligned_clusters"] == [0]

    def test_single_cluster_partial_overlap(self):
        """Cluster with partial overlap."""
        trapping = _make_trapping_result(
            [0, 1, 2],
            clusters=[{"variable_nodes": [0, 1, 2], "check_nodes": []}],
        )
        # Only node 0 is active.
        bp_scores = {0: 10.0, 1: 0.0, 2: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert len(result["per_cluster_alignment_scores"]) == 1
        assert abs(result["per_cluster_alignment_scores"][0] - 1.0/3.0) < 1e-10

    def test_multiple_clusters(self):
        """Multiple clusters with different alignments."""
        trapping = _make_trapping_result(
            [0, 1, 2, 3],
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
                {"variable_nodes": [2, 3], "check_nodes": []},
            ],
        )
        # Nodes 0, 1 active; 2, 3 inactive.
        bp_scores = {0: 10.0, 1: 10.0, 2: 0.0, 3: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert len(result["per_cluster_alignment_scores"]) == 2
        assert result["per_cluster_alignment_scores"][0] == 1.0
        assert result["per_cluster_alignment_scores"][1] == 0.0
        assert result["top_aligned_clusters"] == [0]
        assert result["max_cluster_alignment"] == 1.0

    def test_no_clusters(self):
        """No clusters → empty per-cluster scores."""
        trapping = _make_trapping_result([0, 1])
        bp_scores = {0: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["per_cluster_alignment_scores"] == []
        assert result["top_aligned_clusters"] == []
        assert result["max_cluster_alignment"] == 0.0

    def test_empty_cluster_variable_nodes(self):
        """Cluster with no variable nodes → alignment = 0.0."""
        trapping = _make_trapping_result(
            [],
            clusters=[{"variable_nodes": [], "check_nodes": [0]}],
        )
        bp_scores = {0: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["per_cluster_alignment_scores"] == [0.0]


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        trapping = _make_trapping_result(
            [0, 1, 2],
            clusters=[{"variable_nodes": [0, 1], "check_nodes": [0]}],
        )
        bp_scores = {0: 10.0, 1: 5.0, 2: 1.0, 3: 8.0}

        r1 = compute_spectral_bp_alignment(trapping, bp_scores)
        r2 = compute_spectral_bp_alignment(trapping, bp_scores)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        trapping = _make_trapping_result([0, 1])
        bp_scores = {0: 10.0, 1: 5.0, 2: 0.5}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        trapping = _make_trapping_result(
            [0, 1, 2],
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
                {"variable_nodes": [2], "check_nodes": [0]},
            ],
        )
        bp_scores = {0: 10.0, 1: 3.0, 2: 7.0, 3: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


class TestAlignedCandidateNodes:
    """Tests for aligned_candidate_nodes output."""

    def test_aligned_nodes_sorted(self):
        """Aligned candidate nodes are sorted."""
        trapping = _make_trapping_result([2, 0, 1])
        bp_scores = {0: 10.0, 1: 10.0, 2: 10.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["aligned_candidate_nodes"] == [0, 1, 2]
        assert result["num_aligned_candidate_nodes"] == 3

    def test_count_matches_list(self):
        """num_aligned_candidate_nodes matches length of list."""
        trapping = _make_trapping_result([0, 1, 2, 3])
        bp_scores = {0: 10.0, 1: 0.0, 2: 10.0, 3: 0.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        assert result["num_aligned_candidate_nodes"] == len(
            result["aligned_candidate_nodes"]
        )


class TestActivityScoresOutput:
    """Tests for bp_node_activity_scores output."""

    def test_scores_sorted_by_node(self):
        """Activity scores are sorted by node index."""
        trapping = _make_trapping_result([])
        bp_scores = {3: 1.0, 0: 5.0, 1: 2.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        pairs = result["bp_node_activity_scores"]
        assert pairs == [[0, 5.0], [1, 2.0], [3, 1.0]]


class TestIntegrationWithTrappingCandidates:
    """Tests for compatibility with v6.2 trapping candidate outputs."""

    def test_end_to_end_with_localization(self):
        """Full pipeline: localization → trapping → alignment."""
        from qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )

        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        loc = compute_nb_localization_metrics(H)
        trapping = compute_nb_trapping_candidates(H, loc)

        # Simulate BOI-like scores for 4 variable nodes.
        bp_scores = {0: 3.0, 1: 7.0, 2: 1.0, 3: 5.0}
        result = compute_spectral_bp_alignment(trapping, bp_scores)

        # All required keys present.
        assert "spectral_bp_alignment_score" in result
        assert "candidate_node_overlap_fraction" in result
        assert "bp_node_overlap_fraction" in result
        assert "active_bp_nodes" in result
        assert "aligned_candidate_nodes" in result
        assert "num_aligned_candidate_nodes" in result
        assert "bp_node_activity_scores" in result
        assert "per_cluster_alignment_scores" in result
        assert "top_aligned_clusters" in result
        assert "max_cluster_alignment" in result
        assert "activity_threshold_fraction" in result

        # Types are correct.
        assert isinstance(result["spectral_bp_alignment_score"], float)
        assert isinstance(result["candidate_node_overlap_fraction"], float)
        assert isinstance(result["active_bp_nodes"], list)
        assert isinstance(result["per_cluster_alignment_scores"], list)
        assert 0.0 <= result["spectral_bp_alignment_score"] <= 1.0

    def test_no_decoder_import(self):
        """This module does not import any decoder code."""
        import qec.diagnostics.spectral_bp_alignment as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from qec.decoder" not in source
