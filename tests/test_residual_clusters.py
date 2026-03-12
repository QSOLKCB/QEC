"""
Tests for ResidualClusterAnalyzer (v11.5.0).

Covers:
  - deterministic output
  - valid cluster structure
  - cluster counts non-negative
  - largest cluster size valid
  - no input mutation
  - empty / degenerate matrix handling
  - pre-computed residual map support
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.analysis.residual_clusters import ResidualClusterAnalyzer
from src.qec.analysis.bp_residuals import BPResidualAnalyzer


def _make_regular_H(m: int = 6, n: int = 12, seed: int = 42) -> np.ndarray:
    """Create a small regular parity-check matrix for testing."""
    rng = np.random.RandomState(seed)
    H = np.zeros((m, n), dtype=np.float64)
    for ci in range(m):
        cols = rng.choice(n, size=min(4, n), replace=False)
        for vi in cols:
            H[ci, vi] = 1.0
    # Ensure every column has at least one 1
    for vi in range(n):
        if H[:, vi].sum() == 0:
            ci = rng.randint(0, m)
            H[ci, vi] = 1.0
    return H


class TestResidualClusterAnalyzerDeterminism:
    """Verify deterministic output of the analyzer."""

    def test_same_seed_same_result(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()

        r1 = analyzer.find_clusters(H, seed=123)
        r2 = analyzer.find_clusters(H, seed=123)

        assert r1["num_clusters"] == r2["num_clusters"]
        assert r1["largest_cluster_size"] == r2["largest_cluster_size"]
        assert r1["max_cluster_residual"] == r2["max_cluster_residual"]
        assert len(r1["clusters"]) == len(r2["clusters"])

        for c1, c2 in zip(r1["clusters"], r2["clusters"]):
            assert c1["variables"] == c2["variables"]
            assert c1["size"] == c2["size"]
            assert c1["mean_residual"] == c2["mean_residual"]
            assert c1["max_residual"] == c2["max_residual"]

    def test_different_seed_may_differ(self):
        """Different seeds may produce different residual maps and clusters."""
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()

        r1 = analyzer.find_clusters(H, seed=0)
        r2 = analyzer.find_clusters(H, seed=999)

        # Structure should be valid for both
        assert r1["num_clusters"] >= 0
        assert r2["num_clusters"] >= 0


class TestResidualClusterStructure:
    """Verify cluster result structure and invariants."""

    def test_result_keys(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        assert "clusters" in result
        assert "num_clusters" in result
        assert "largest_cluster_size" in result
        assert "max_cluster_residual" in result

    def test_cluster_entry_keys(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        if result["clusters"]:
            cluster = result["clusters"][0]
            assert "variables" in cluster
            assert "size" in cluster
            assert "mean_residual" in cluster
            assert "max_residual" in cluster
            assert "boundary_score" in cluster
            assert "internal_density" in cluster

    def test_num_clusters_matches_list_length(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        assert result["num_clusters"] == len(result["clusters"])

    def test_cluster_sizes_positive(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        for cluster in result["clusters"]:
            assert cluster["size"] > 0
            assert cluster["size"] == len(cluster["variables"])

    def test_largest_cluster_size_valid(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        if result["clusters"]:
            sizes = [c["size"] for c in result["clusters"]]
            assert result["largest_cluster_size"] == max(sizes)
        else:
            assert result["largest_cluster_size"] == 0

    def test_max_cluster_residual_valid(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        if result["clusters"]:
            max_res = max(c["max_residual"] for c in result["clusters"])
            assert result["max_cluster_residual"] == max_res
        else:
            assert result["max_cluster_residual"] == 0.0

    def test_cluster_counts_non_negative(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        assert result["num_clusters"] >= 0
        assert result["largest_cluster_size"] >= 0
        assert result["max_cluster_residual"] >= 0.0

    def test_variables_are_sorted(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        for cluster in result["clusters"]:
            assert cluster["variables"] == sorted(cluster["variables"])

    def test_no_duplicate_variables_across_clusters(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        all_vars: list[int] = []
        for cluster in result["clusters"]:
            all_vars.extend(cluster["variables"])
        assert len(all_vars) == len(set(all_vars))

    def test_boundary_score_in_range(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        for cluster in result["clusters"]:
            assert 0.0 <= cluster["boundary_score"] <= 1.0

    def test_internal_density_in_range(self):
        H = _make_regular_H()
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=42)

        for cluster in result["clusters"]:
            assert 0.0 <= cluster["internal_density"] <= 1.0


class TestResidualClusterNoInputMutation:
    """Verify the analyzer does not modify input arrays."""

    def test_H_not_mutated(self):
        H = _make_regular_H()
        H_copy = H.copy()
        analyzer = ResidualClusterAnalyzer()
        analyzer.find_clusters(H, seed=42)

        np.testing.assert_array_equal(H, H_copy)

    def test_residual_map_not_mutated(self):
        H = _make_regular_H()
        bp = BPResidualAnalyzer()
        result = bp.compute_residual_map(H, seed=42)
        rmap = result["residual_map"]
        rmap_copy = rmap.copy()

        analyzer = ResidualClusterAnalyzer()
        analyzer.find_clusters(H, residual_map=rmap, seed=42)

        np.testing.assert_array_equal(rmap, rmap_copy)


class TestResidualClusterEdgeCases:
    """Verify correct behavior on edge cases."""

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=0)

        assert result["num_clusters"] == 0
        assert result["largest_cluster_size"] == 0
        assert result["max_cluster_residual"] == 0.0
        assert result["clusters"] == []

    def test_zero_matrix(self):
        H = np.zeros((4, 8), dtype=np.float64)
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=0)

        assert result["num_clusters"] >= 0

    def test_single_row(self):
        H = np.array([[1, 1, 0, 1]], dtype=np.float64)
        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, seed=0)

        assert result["num_clusters"] >= 0
        assert result["largest_cluster_size"] >= 0

    def test_precomputed_residual_map(self):
        H = _make_regular_H()
        n = H.shape[1]
        residual_map = np.arange(n, dtype=np.float64)

        analyzer = ResidualClusterAnalyzer()
        result = analyzer.find_clusters(H, residual_map=residual_map)

        assert result["num_clusters"] >= 0

    def test_custom_threshold_percentile(self):
        H = _make_regular_H()
        analyzer_low = ResidualClusterAnalyzer(threshold_percentile=50.0)
        analyzer_high = ResidualClusterAnalyzer(threshold_percentile=90.0)

        result_low = analyzer_low.find_clusters(H, seed=42)
        result_high = analyzer_high.find_clusters(H, seed=42)

        # Lower threshold should include more variables
        total_vars_low = sum(c["size"] for c in result_low["clusters"])
        total_vars_high = sum(c["size"] for c in result_high["clusters"])
        assert total_vars_low >= total_vars_high


class TestResidualClusterImport:
    """Verify the analyzer is properly exported."""

    def test_import_from_analysis(self):
        from src.qec.analysis import ResidualClusterAnalyzer as RCA
        assert RCA is not None

    def test_instance_creation(self):
        analyzer = ResidualClusterAnalyzer()
        assert analyzer is not None
        assert analyzer.threshold_percentile == 75.0
