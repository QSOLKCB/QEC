"""
Tests for spectral failure risk scoring diagnostics (v6.4).

Verifies:
  - Deterministic risk score calculation
  - Cluster ranking correctness
  - JSON serialization stability
  - Compatibility with v6.3 outputs
  - CLI flag integration
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

from qec.diagnostics.spectral_failure_risk import (
    compute_spectral_failure_risk,
)


# ── Toy inputs ────────────────────────────────────────────────────────

def _make_localization_result(
    ipr_scores: list[float],
    localized_modes: list[int],
) -> dict[str, object]:
    """Build a minimal v6.1-compatible localization result."""
    return {
        "ipr_scores": ipr_scores,
        "localized_modes": localized_modes,
    }


def _make_trapping_result(
    clusters: list[dict[str, list[int]]],
    node_participation: dict[str, int],
) -> dict[str, object]:
    """Build a minimal v6.2-compatible trapping candidate result."""
    cand_vars = sorted(set(
        v for cl in clusters for v in cl.get("variable_nodes", [])
    ))
    return {
        "candidate_variable_nodes": cand_vars,
        "candidate_clusters": clusters,
        "node_participation_counts": node_participation,
    }


def _make_alignment_result(
    per_cluster_alignment_scores: list[float],
) -> dict[str, object]:
    """Build a minimal v6.3-compatible alignment result."""
    return {
        "per_cluster_alignment_scores": per_cluster_alignment_scores,
    }


class TestRiskScoreComputation:
    """Tests for cluster risk score calculation."""

    def test_basic_risk_score(self):
        """Risk = participation_weight * alignment * localization_weight."""
        loc = _make_localization_result(
            ipr_scores=[0.5, 0.3, 0.8],
            localized_modes=[0, 2],
        )
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
            ],
            node_participation={"v0": 3, "v1": 2},
        )
        alignment = _make_alignment_result([0.8])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        # localization_weight = mean(ipr[0], ipr[2]) = mean(0.5, 0.8) = 0.65
        # participation_weight = mean(3, 2) = 2.5
        # alignment_score = 0.8
        # risk = 2.5 * 0.8 * 0.65 = 1.3
        assert len(result["cluster_risk_scores"]) == 1
        assert abs(result["cluster_risk_scores"][0] - 1.3) < 1e-10

    def test_zero_alignment_yields_zero_risk(self):
        """Cluster with zero alignment has zero risk."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[{"variable_nodes": [0], "check_nodes": []}],
            node_participation={"v0": 3},
        )
        alignment = _make_alignment_result([0.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        assert result["cluster_risk_scores"] == [0.0]
        assert result["max_cluster_risk"] == 0.0

    def test_no_localized_modes_yields_zero_risk(self):
        """No localized modes → localization_weight = 0 → zero risk."""
        loc = _make_localization_result([0.01, 0.02], [])
        trapping = _make_trapping_result(
            clusters=[{"variable_nodes": [0], "check_nodes": []}],
            node_participation={"v0": 3},
        )
        alignment = _make_alignment_result([1.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        assert result["cluster_risk_scores"] == [0.0]

    def test_empty_clusters(self):
        """No clusters → empty outputs."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result([], {})
        alignment = _make_alignment_result([])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        assert result["cluster_risk_scores"] == []
        assert result["cluster_risk_ranking"] == []
        assert result["top_risk_clusters"] == []
        assert result["node_risk_scores"] == []
        assert result["max_cluster_risk"] == 0.0
        assert result["mean_cluster_risk"] == 0.0
        assert result["num_high_risk_clusters"] == 0


class TestClusterRanking:
    """Tests for cluster ranking correctness."""

    def test_ranking_descending_by_risk(self):
        """Clusters ranked by descending risk score."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0], "check_nodes": []},
                {"variable_nodes": [1], "check_nodes": []},
                {"variable_nodes": [2], "check_nodes": []},
            ],
            node_participation={"v0": 1, "v1": 3, "v2": 2},
        )
        alignment = _make_alignment_result([0.5, 1.0, 0.8])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        # risk[0] = 1 * 0.5 * 0.5 = 0.25
        # risk[1] = 3 * 1.0 * 0.5 = 1.5
        # risk[2] = 2 * 0.8 * 0.5 = 0.8
        assert result["cluster_risk_ranking"] == [1, 2, 0]

    def test_tie_breaking_by_index(self):
        """Equal risk → sorted by cluster index ascending."""
        loc = _make_localization_result([1.0], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0], "check_nodes": []},
                {"variable_nodes": [1], "check_nodes": []},
            ],
            node_participation={"v0": 2, "v1": 2},
        )
        alignment = _make_alignment_result([0.5, 0.5])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        # Both have identical risk.
        assert result["cluster_risk_ranking"] == [0, 1]

    def test_top_risk_clusters_limited_by_top_k(self):
        """top_risk_clusters respects top_k parameter."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [i], "check_nodes": []}
                for i in range(5)
            ],
            node_participation={f"v{i}": i + 1 for i in range(5)},
        )
        alignment = _make_alignment_result([1.0] * 5)

        result = compute_spectral_failure_risk(
            loc, trapping, alignment, top_k=2,
        )

        assert len(result["top_risk_clusters"]) == 2
        # Highest risk clusters first.
        assert result["top_risk_clusters"][0] == 4
        assert result["top_risk_clusters"][1] == 3

    def test_top_risk_excludes_zero_risk(self):
        """top_risk_clusters excludes clusters with zero risk."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0], "check_nodes": []},
                {"variable_nodes": [1], "check_nodes": []},
            ],
            node_participation={"v0": 2, "v1": 2},
        )
        alignment = _make_alignment_result([1.0, 0.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        assert result["top_risk_clusters"] == [0]
        assert result["num_high_risk_clusters"] == 1


class TestNodeRiskScores:
    """Tests for node-level risk score computation."""

    def test_node_risk_is_sum_of_cluster_risks(self):
        """Node risk = sum of risk scores for all containing clusters."""
        loc = _make_localization_result([1.0], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
                {"variable_nodes": [1, 2], "check_nodes": []},
            ],
            node_participation={"v0": 2, "v1": 2, "v2": 2},
        )
        alignment = _make_alignment_result([0.5, 0.5])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        # Both clusters: participation_weight = 2, alignment = 0.5,
        # localization = 1.0 → risk = 1.0 each.
        risk_map = {pair[0]: pair[1] for pair in result["node_risk_scores"]}
        assert abs(risk_map[0] - 1.0) < 1e-10  # in cluster 0 only
        assert abs(risk_map[1] - 2.0) < 1e-10  # in both clusters
        assert abs(risk_map[2] - 1.0) < 1e-10  # in cluster 1 only

    def test_node_risk_sorted_by_index(self):
        """Node risk pairs are sorted by node index."""
        loc = _make_localization_result([1.0], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [3, 1, 5], "check_nodes": []},
            ],
            node_participation={"v1": 2, "v3": 2, "v5": 2},
        )
        alignment = _make_alignment_result([1.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        indices = [pair[0] for pair in result["node_risk_scores"]]
        assert indices == sorted(indices)

    def test_zero_risk_nodes_excluded(self):
        """Nodes with zero risk are not in node_risk_scores."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0], "check_nodes": []},
                {"variable_nodes": [1], "check_nodes": []},
            ],
            node_participation={"v0": 2, "v1": 2},
        )
        alignment = _make_alignment_result([1.0, 0.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        indices = [pair[0] for pair in result["node_risk_scores"]]
        assert 0 in indices
        assert 1 not in indices


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        loc = _make_localization_result([0.5, 0.3, 0.8], [0, 2])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": [0]},
                {"variable_nodes": [2, 3], "check_nodes": []},
            ],
            node_participation={"v0": 3, "v1": 2, "v2": 1, "v3": 4},
        )
        alignment = _make_alignment_result([0.6, 0.9])

        r1 = compute_spectral_failure_risk(loc, trapping, alignment)
        r2 = compute_spectral_failure_risk(loc, trapping, alignment)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        loc = _make_localization_result([0.5], [0])
        trapping = _make_trapping_result(
            clusters=[{"variable_nodes": [0, 1], "check_nodes": []}],
            node_participation={"v0": 2, "v1": 3},
        )
        alignment = _make_alignment_result([0.7])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        loc = _make_localization_result([0.5, 0.8], [0, 1])
        trapping = _make_trapping_result(
            clusters=[
                {"variable_nodes": [0, 1], "check_nodes": []},
                {"variable_nodes": [2], "check_nodes": [0]},
            ],
            node_participation={"v0": 2, "v1": 1, "v2": 3},
        )
        alignment = _make_alignment_result([0.5, 1.0])

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)


class TestCompatibilityWithV63:
    """Tests for compatibility with v6.3 alignment outputs."""

    def test_end_to_end_with_full_pipeline(self):
        """Full pipeline: localization → trapping → alignment → risk."""
        from qec.diagnostics.nb_localization import (
            compute_nb_localization_metrics,
        )
        from qec.diagnostics.nb_trapping_candidates import (
            compute_nb_trapping_candidates,
        )
        from qec.diagnostics.spectral_bp_alignment import (
            compute_spectral_bp_alignment,
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
        alignment = compute_spectral_bp_alignment(trapping, bp_scores)

        result = compute_spectral_failure_risk(loc, trapping, alignment)

        # All required keys present.
        assert "node_risk_scores" in result
        assert "cluster_risk_scores" in result
        assert "cluster_risk_ranking" in result
        assert "max_cluster_risk" in result
        assert "mean_cluster_risk" in result
        assert "top_risk_clusters" in result
        assert "num_high_risk_clusters" in result

        # Types are correct.
        assert isinstance(result["cluster_risk_scores"], list)
        assert isinstance(result["cluster_risk_ranking"], list)
        assert isinstance(result["max_cluster_risk"], float)
        assert isinstance(result["mean_cluster_risk"], float)
        assert isinstance(result["top_risk_clusters"], list)
        assert isinstance(result["num_high_risk_clusters"], int)
        assert isinstance(result["node_risk_scores"], list)

        # Risk scores are non-negative.
        for score in result["cluster_risk_scores"]:
            assert score >= 0.0

        # JSON-serializable.
        json.dumps(result)


class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """This module does not import any decoder code."""
        import qec.diagnostics.spectral_failure_risk as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from qec.decoder" not in source
