"""
Tests for spectral trapping-set candidate detection (v6.2).

Verifies:
  - Deterministic node participation counting
  - Cluster detection on toy Tanner graphs
  - JSON serialization stability
  - Compatibility with localization outputs
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

from qec.diagnostics.nb_trapping_candidates import (
    compute_nb_trapping_candidates,
)
from qec.diagnostics.nb_localization import (
    compute_nb_localization_metrics,
)


class TestNodeParticipation:
    """Tests for node participation counting."""

    def test_no_localized_modes(self):
        """No localized modes yields empty candidates."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [],
            "localized_check_nodes": [],
        }
        result = compute_nb_trapping_candidates(H, loc)

        assert result["node_participation_counts"] == {}
        assert result["candidate_variable_nodes"] == []
        assert result["candidate_check_nodes"] == []
        assert result["candidate_clusters"] == []
        assert result["max_node_participation"] == 0
        assert result["num_candidate_nodes"] == 0
        assert result["num_candidate_clusters"] == 0

    def test_single_mode_no_candidates(self):
        """Single localized mode: no node meets threshold >= 2."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0, 1]],
            "localized_check_nodes": [[0]],
        }
        result = compute_nb_trapping_candidates(H, loc)

        assert result["max_node_participation"] == 1
        assert result["candidate_variable_nodes"] == []
        assert result["candidate_check_nodes"] == []
        assert result["num_candidate_nodes"] == 0

    def test_two_modes_shared_node(self):
        """Node appearing in two modes is a candidate."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0, 1], [1, 2]],
            "localized_check_nodes": [[0], [1]],
        }
        result = compute_nb_trapping_candidates(H, loc)

        # Variable node 1 appears in both modes.
        assert 1 in result["candidate_variable_nodes"]
        assert result["node_participation_counts"]["v1"] == 2
        assert result["max_node_participation"] == 2

    def test_threshold_parameter(self):
        """Custom threshold filters correctly."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0, 1], [1, 2], [1]],
            "localized_check_nodes": [[0], [1], [0]],
        }
        # threshold=3: only node 1 (var) with count=3
        result = compute_nb_trapping_candidates(
            H, loc, participation_threshold=3,
        )
        assert result["candidate_variable_nodes"] == [1]
        assert result["candidate_check_nodes"] == []
        assert result["participation_threshold"] == 3

    def test_check_node_participation(self):
        """Check nodes are counted independently."""
        H = np.array([[1, 1], [1, 0]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0], [0]],
            "localized_check_nodes": [[0], [0]],
        }
        result = compute_nb_trapping_candidates(H, loc)

        assert 0 in result["candidate_variable_nodes"]
        assert 0 in result["candidate_check_nodes"]
        assert result["node_participation_counts"]["v0"] == 2
        assert result["node_participation_counts"]["c0"] == 2


class TestClusterDetection:
    """Tests for connected component (cluster) detection."""

    def test_single_cluster(self):
        """Adjacent candidate nodes form one cluster."""
        # H = [[1,1],[1,0]] : v0-c0, v1-c0, v0-c1
        H = np.array([[1, 1], [1, 0]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0, 1], [0, 1]],
            "localized_check_nodes": [[0], [0]],
        }
        result = compute_nb_trapping_candidates(H, loc)

        # v0, v1, c0 are all candidates; they're connected via c0.
        assert result["num_candidate_clusters"] == 1
        cluster = result["candidate_clusters"][0]
        assert 0 in cluster["variable_nodes"]
        assert 1 in cluster["variable_nodes"]

    def test_disconnected_clusters(self):
        """Disconnected candidate nodes form separate clusters."""
        # Block-diagonal: two disconnected components.
        H = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0, 1], [0, 1], [2, 3], [2, 3]],
            "localized_check_nodes": [[0], [0], [1], [1]],
        }
        result = compute_nb_trapping_candidates(H, loc)

        assert result["num_candidate_clusters"] == 2

    def test_empty_clusters(self):
        """No candidates yields no clusters."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [[0]],
            "localized_check_nodes": [[0]],
        }
        result = compute_nb_trapping_candidates(H, loc)
        assert result["candidate_clusters"] == []
        assert result["num_candidate_clusters"] == 0

    def test_cluster_sort_determinism(self):
        """Clusters are sorted deterministically by size then index."""
        H = np.array([
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1],
        ], dtype=np.float64)
        loc = {
            "localized_variable_nodes": [
                [0, 1], [0, 1],
                [2, 3, 4], [2, 3, 4],
            ],
            "localized_check_nodes": [
                [0], [0],
                [1], [1],
            ],
        }
        result = compute_nb_trapping_candidates(H, loc)

        assert result["num_candidate_clusters"] == 2
        # Larger cluster (3 var nodes + 1 check = 4) comes first.
        assert len(result["candidate_clusters"][0]["variable_nodes"]) >= \
               len(result["candidate_clusters"][1]["variable_nodes"])


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        loc = compute_nb_localization_metrics(H)

        r1 = compute_nb_trapping_candidates(H, loc)
        r2 = compute_nb_trapping_candidates(H, loc)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        loc = compute_nb_localization_metrics(H)
        result = compute_nb_trapping_candidates(H, loc)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        loc = compute_nb_localization_metrics(H)
        result = compute_nb_trapping_candidates(H, loc)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_no_input_mutation(self):
        """Input matrix and localization result are not modified."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        H_copy = H.copy()
        loc = compute_nb_localization_metrics(H)
        loc_copy = json.loads(json.dumps(loc))

        compute_nb_trapping_candidates(H, loc)

        np.testing.assert_array_equal(H, H_copy)
        assert json.dumps(loc, sort_keys=True) == \
               json.dumps(loc_copy, sort_keys=True)


class TestIntegrationWithLocalization:
    """Tests for compatibility with v6.1 localization outputs."""

    def test_end_to_end_small_code(self):
        """Full pipeline: localization → trapping candidates."""
        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        loc = compute_nb_localization_metrics(H)
        result = compute_nb_trapping_candidates(H, loc)

        # All required keys present.
        assert "node_participation_counts" in result
        assert "candidate_variable_nodes" in result
        assert "candidate_check_nodes" in result
        assert "candidate_clusters" in result
        assert "max_node_participation" in result
        assert "num_candidate_nodes" in result
        assert "num_candidate_clusters" in result
        assert "participation_threshold" in result

        # Types are correct.
        assert isinstance(result["node_participation_counts"], dict)
        assert isinstance(result["candidate_variable_nodes"], list)
        assert isinstance(result["candidate_check_nodes"], list)
        assert isinstance(result["candidate_clusters"], list)
        assert isinstance(result["max_node_participation"], int)
        assert isinstance(result["participation_threshold"], int)

    def test_empty_matrix(self):
        """All-zero matrix yields empty candidates."""
        H = np.zeros((2, 3), dtype=np.float64)
        loc = compute_nb_localization_metrics(H)
        result = compute_nb_trapping_candidates(H, loc)

        assert result["candidate_variable_nodes"] == []
        assert result["candidate_check_nodes"] == []
        assert result["max_node_participation"] == 0
        assert result["num_candidate_clusters"] == 0

    def test_forced_localization(self):
        """Forcing all modes localized produces valid candidates."""
        H = np.array([
            [1, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
        ], dtype=np.float64)
        loc = compute_nb_localization_metrics(
            H, ipr_localization_threshold=0.0,
        )
        result = compute_nb_trapping_candidates(H, loc)

        # With many localized modes, likely some nodes meet threshold.
        assert result["max_node_participation"] >= 1

        # All candidate variable nodes are valid indices.
        m, n = H.shape
        for vi in result["candidate_variable_nodes"]:
            assert 0 <= vi < n
        for ci in result["candidate_check_nodes"]:
            assert 0 <= ci < m
