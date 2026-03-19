"""
Tests for spectral trapping-set heatmaps (v7.7.0).

Verifies:
  - directed heat matches edge_energy
  - undirected heat aggregates correctly
  - node heat vector sizes correct
  - ranking deterministic
  - artifact schema valid
  - heat values finite and non-negative
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

from qec.diagnostics.spectral_heatmaps import (
    compute_spectral_heatmaps,
    rank_variable_nodes_by_heat,
    rank_check_nodes_by_heat,
    rank_edges_by_heat,
    _compute_undirected_edge_heat,
    _contrast_normalize,
)
from qec.diagnostics.spectral_nb import compute_nb_spectrum, _TannerGraph
from qec.diagnostics._spectral_utils import build_directed_edges
from qec.experiments.spectral_heatmap_experiment import (
    run_spectral_heatmap_experiment,
    serialize_heatmap_artifact,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


def _medium_H():
    """4x6 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


# ── compute_spectral_heatmaps tests ─────────────────────────────


class TestComputeSpectralHeatmaps:
    """Tests for the core heatmap computation."""

    def test_returns_all_keys(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        expected_keys = {
            "directed_edge_heat",
            "undirected_edge_heat",
            "variable_node_heat",
            "check_node_heat",
            "spectral_radius",
            "ipr",
            "eeec",
            "sis",
        }
        assert set(result.keys()) == expected_keys

    def test_directed_heat_matches_edge_energy(self):
        H = _small_H()
        spectrum = compute_nb_spectrum(H)
        heatmaps = compute_spectral_heatmaps(H)
        np.testing.assert_allclose(
            heatmaps["directed_edge_heat"],
            spectrum["edge_energy"],
            atol=1e-12,
        )

    def test_spectral_metrics_match(self):
        H = _small_H()
        spectrum = compute_nb_spectrum(H)
        heatmaps = compute_spectral_heatmaps(H)
        assert heatmaps["spectral_radius"] == spectrum["spectral_radius"]
        assert heatmaps["ipr"] == spectrum["ipr"]
        assert heatmaps["eeec"] == spectrum["eeec"]
        assert heatmaps["sis"] == spectrum["sis"]

    def test_variable_node_heat_size(self):
        H = _small_H()
        m, n = H.shape
        result = compute_spectral_heatmaps(H)
        assert len(result["variable_node_heat"]) == n

    def test_check_node_heat_size(self):
        H = _small_H()
        m, n = H.shape
        result = compute_spectral_heatmaps(H)
        assert len(result["check_node_heat"]) == m

    def test_directed_heat_finite(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(np.isfinite(result["directed_edge_heat"]))

    def test_directed_heat_non_negative(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(result["directed_edge_heat"] >= 0)

    def test_undirected_heat_finite(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(np.isfinite(result["undirected_edge_heat"]))

    def test_undirected_heat_non_negative(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(result["undirected_edge_heat"] >= 0)

    def test_variable_node_heat_finite(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(np.isfinite(result["variable_node_heat"]))

    def test_variable_node_heat_non_negative(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(result["variable_node_heat"] >= 0)

    def test_check_node_heat_finite(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(np.isfinite(result["check_node_heat"]))

    def test_check_node_heat_non_negative(self):
        H = _small_H()
        result = compute_spectral_heatmaps(H)
        assert np.all(result["check_node_heat"] >= 0)

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_spectral_heatmaps(H)
        r2 = compute_spectral_heatmaps(H)
        np.testing.assert_allclose(
            r1["directed_edge_heat"], r2["directed_edge_heat"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            r1["undirected_edge_heat"], r2["undirected_edge_heat"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            r1["variable_node_heat"], r2["variable_node_heat"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            r1["check_node_heat"], r2["check_node_heat"],
            atol=1e-12,
        )

    def test_medium_matrix(self):
        H = _medium_H()
        m, n = H.shape
        result = compute_spectral_heatmaps(H)
        assert len(result["variable_node_heat"]) == n
        assert len(result["check_node_heat"]) == m
        assert np.all(np.isfinite(result["undirected_edge_heat"]))


# ── Undirected edge heat tests ───────────────────────────────────


class TestUndirectedEdgeHeat:
    """Tests for undirected edge aggregation."""

    def test_aggregation_sums_pairs(self):
        H = _small_H()
        graph = _TannerGraph(H)
        directed_edges = build_directed_edges(graph)
        spectrum = compute_nb_spectrum(H)
        edge_energy = spectrum["edge_energy"]

        undirected_heat, undirected_edges = _compute_undirected_edge_heat(
            directed_edges, edge_energy,
        )

        # Each undirected edge should be the sum of its two directed edges
        edge_idx = {e: i for i, e in enumerate(directed_edges)}
        for ue_idx, (u, v) in enumerate(undirected_edges):
            forward = edge_idx.get((u, v), None)
            backward = edge_idx.get((v, u), None)
            expected = 0.0
            if forward is not None:
                expected += edge_energy[forward]
            if backward is not None:
                expected += edge_energy[backward]
            np.testing.assert_almost_equal(
                undirected_heat[ue_idx], expected, decimal=12,
            )

    def test_undirected_edges_sorted(self):
        H = _small_H()
        graph = _TannerGraph(H)
        directed_edges = build_directed_edges(graph)
        spectrum = compute_nb_spectrum(H)

        _, undirected_edges = _compute_undirected_edge_heat(
            directed_edges, spectrum["edge_energy"],
        )

        # Verify sorted order
        for i in range(len(undirected_edges) - 1):
            assert undirected_edges[i] <= undirected_edges[i + 1]

    def test_undirected_edges_canonical(self):
        H = _small_H()
        graph = _TannerGraph(H)
        directed_edges = build_directed_edges(graph)
        spectrum = compute_nb_spectrum(H)

        _, undirected_edges = _compute_undirected_edge_heat(
            directed_edges, spectrum["edge_energy"],
        )

        # All undirected edges must have u < v
        for u, v in undirected_edges:
            assert u < v


# ── Contrast normalization tests ─────────────────────────────────


class TestContrastNormalize:
    """Tests for contrast normalization."""

    def test_non_negative_output(self):
        heat = np.array([1.0, 2.0, 3.0, 0.5, 0.1])
        result = _contrast_normalize(heat)
        assert np.all(result >= 0)

    def test_empty_input(self):
        heat = np.array([], dtype=np.float64)
        result = _contrast_normalize(heat)
        assert len(result) == 0

    def test_uniform_input(self):
        heat = np.ones(5)
        result = _contrast_normalize(heat)
        # Uniform input: all values become (1-1)/std = 0
        np.testing.assert_array_almost_equal(result, np.zeros(5))

    def test_deterministic(self):
        heat = np.array([1.0, 3.0, 0.5, 2.0])
        r1 = _contrast_normalize(heat)
        r2 = _contrast_normalize(heat)
        np.testing.assert_array_equal(r1, r2)


# ── Ranking tests ────────────────────────────────────────────────


class TestRankings:
    """Tests for deterministic ranking utilities."""

    def test_variable_ranking_deterministic(self):
        H = _small_H()
        r1 = rank_variable_nodes_by_heat(H)
        r2 = rank_variable_nodes_by_heat(H)
        # Compare heat values with tolerance (node order may differ for ties)
        v1 = sorted([h for _, h in r1], reverse=True)
        v2 = sorted([h for _, h in r2], reverse=True)
        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_check_ranking_deterministic(self):
        H = _small_H()
        r1 = rank_check_nodes_by_heat(H)
        r2 = rank_check_nodes_by_heat(H)
        v1 = sorted([h for _, h in r1], reverse=True)
        v2 = sorted([h for _, h in r2], reverse=True)
        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_edge_ranking_deterministic(self):
        H = _small_H()
        r1 = rank_edges_by_heat(H)
        r2 = rank_edges_by_heat(H)
        v1 = sorted([h for _, h in r1], reverse=True)
        v2 = sorted([h for _, h in r2], reverse=True)
        np.testing.assert_allclose(v1, v2, atol=1e-6)

    def test_variable_ranking_sorted_descending(self):
        H = _small_H()
        ranking = rank_variable_nodes_by_heat(H)
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_check_ranking_sorted_descending(self):
        H = _small_H()
        ranking = rank_check_nodes_by_heat(H)
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_edge_ranking_sorted_descending(self):
        H = _small_H()
        ranking = rank_edges_by_heat(H)
        for i in range(len(ranking) - 1):
            assert ranking[i][1] >= ranking[i + 1][1]

    def test_variable_ranking_covers_all_nodes(self):
        H = _small_H()
        m, n = H.shape
        ranking = rank_variable_nodes_by_heat(H)
        assert len(ranking) == n
        indices = {r[0] for r in ranking}
        assert indices == set(range(n))

    def test_check_ranking_covers_all_nodes(self):
        H = _small_H()
        m, n = H.shape
        ranking = rank_check_nodes_by_heat(H)
        assert len(ranking) == m
        indices = {r[0] for r in ranking}
        assert indices == set(range(m))

    def test_edge_ranking_values_non_negative(self):
        H = _small_H()
        ranking = rank_edges_by_heat(H)
        for _, heat in ranking:
            assert heat >= 0

    def test_stable_tie_breaking(self):
        """When heat values tie, lower index comes first."""
        H = _small_H()
        ranking = rank_variable_nodes_by_heat(H)
        for i in range(len(ranking) - 1):
            if ranking[i][1] == ranking[i + 1][1]:
                assert ranking[i][0] < ranking[i + 1][0]


# ── Artifact experiment tests ────────────────────────────────────


class TestHeatmapExperiment:
    """Tests for the heatmap artifact experiment."""

    def test_artifact_has_required_keys(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        required_keys = {
            "schema_version",
            "spectral_radius",
            "ipr",
            "eeec",
            "sis",
            "max_variable_heat",
            "max_check_heat",
            "num_variable_nodes",
            "num_check_nodes",
            "num_undirected_edges",
            "top_variable_nodes",
            "top_check_nodes",
            "top_edges",
        }
        assert set(artifact.keys()) == required_keys

    def test_schema_version(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["schema_version"] == 1

    def test_num_variable_nodes(self):
        H = _small_H()
        m, n = H.shape
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["num_variable_nodes"] == n

    def test_num_check_nodes(self):
        H = _small_H()
        m, n = H.shape
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["num_check_nodes"] == m

    def test_top_variable_nodes_type(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert isinstance(artifact["top_variable_nodes"], list)
        for item in artifact["top_variable_nodes"]:
            assert "index" in item
            assert "heat" in item

    def test_top_check_nodes_type(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert isinstance(artifact["top_check_nodes"], list)
        for item in artifact["top_check_nodes"]:
            assert "index" in item
            assert "heat" in item

    def test_top_edges_type(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert isinstance(artifact["top_edges"], list)
        for item in artifact["top_edges"]:
            assert "index" in item
            assert "heat" in item

    def test_artifact_serializable(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        serialized = serialize_heatmap_artifact(artifact)
        loaded = json.loads(serialized)
        assert loaded["schema_version"] == 1

    def test_artifact_deterministic(self):
        H = _small_H()
        a1 = run_spectral_heatmap_experiment(H)
        a2 = run_spectral_heatmap_experiment(H)
        s1 = serialize_heatmap_artifact(a1)
        s2 = serialize_heatmap_artifact(a2)
        assert s1 == s2

    def test_max_heat_non_negative(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["max_variable_heat"] >= 0
        assert artifact["max_check_heat"] >= 0

    def test_spectral_radius_positive(self):
        H = _small_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["spectral_radius"] > 0

    def test_medium_matrix(self):
        H = _medium_H()
        artifact = run_spectral_heatmap_experiment(H)
        assert artifact["num_variable_nodes"] == 6
        assert artifact["num_check_nodes"] == 4
        serialized = serialize_heatmap_artifact(artifact)
        assert json.loads(serialized) is not None
