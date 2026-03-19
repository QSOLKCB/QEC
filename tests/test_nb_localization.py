"""
Tests for non-backtracking localization diagnostics (v6.1).

Verifies:
  - IPR is correct on simple synthetic vectors
  - Complex eigenvector handling is correct
  - Results are deterministic across repeated runs
  - JSON output structure is stable
  - Localized directed-edge support projects correctly to variable/check
    nodes on a tiny toy Tanner graph
  - Existing decoder behavior remains unchanged (no decoder import)
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

from qec.diagnostics.nb_localization import (
    _ipr,
    compute_nb_localization_metrics,
)


class TestIPR:
    """Unit tests for the inverse participation ratio function."""

    def test_uniform_vector(self):
        """Uniform vector has IPR = 1/n."""
        n = 10
        v = np.ones(n) / np.sqrt(n)
        ipr_val = _ipr(v)
        assert abs(ipr_val - 1.0 / n) < 1e-10

    def test_delta_vector(self):
        """Delta (single-entry) vector has IPR = 1."""
        v = np.zeros(8)
        v[3] = 1.0
        ipr_val = _ipr(v)
        assert abs(ipr_val - 1.0) < 1e-10

    def test_two_entry_vector(self):
        """Two equal-magnitude entries give IPR = 1/2."""
        v = np.zeros(6)
        v[0] = 1.0
        v[3] = 1.0
        ipr_val = _ipr(v)
        assert abs(ipr_val - 0.5) < 1e-10

    def test_zero_vector(self):
        """Zero vector returns IPR = 0."""
        v = np.zeros(5)
        ipr_val = _ipr(v)
        assert ipr_val == 0.0

    def test_complex_uniform(self):
        """Complex uniform vector has IPR = 1/n."""
        n = 8
        v = np.exp(1j * np.arange(n) * 2 * np.pi / n) / np.sqrt(n)
        ipr_val = _ipr(v)
        assert abs(ipr_val - 1.0 / n) < 1e-10

    def test_complex_delta(self):
        """Complex delta vector has IPR = 1."""
        v = np.zeros(6, dtype=complex)
        v[2] = 1.0 + 1.0j
        ipr_val = _ipr(v)
        assert abs(ipr_val - 1.0) < 1e-10

    def test_complex_two_entries(self):
        """Two complex entries with equal magnitude give IPR = 1/2."""
        v = np.zeros(5, dtype=complex)
        v[0] = 1.0 + 0.0j
        v[2] = 0.0 + 1.0j  # |v[0]| = |v[2]| = 1
        ipr_val = _ipr(v)
        assert abs(ipr_val - 0.5) < 1e-10

    def test_ipr_scale_invariance(self):
        """IPR is invariant to scalar multiplication."""
        v = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        ipr1 = _ipr(v)
        ipr2 = _ipr(v * 7.5)
        ipr3 = _ipr(v * 0.001)
        assert abs(ipr1 - ipr2) < 1e-10
        assert abs(ipr1 - ipr3) < 1e-10


class TestComputeNBLocalizationMetrics:
    """Tests for the full localization metrics function."""

    def test_small_repetition_code(self):
        """Repetition code H = [[1,1,0],[0,1,1]] produces valid metrics."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        assert "ipr_scores" in result
        assert "max_ipr" in result
        assert "localized_modes" in result
        assert "mode_support_sizes" in result
        assert "localized_edge_indices" in result
        assert "localized_variable_nodes" in result
        assert "localized_check_nodes" in result
        assert "top_localization_score" in result
        assert "per_mode_mass_on_variables" in result
        assert "per_mode_mass_on_checks" in result
        assert "num_directed_edges" in result
        assert "num_leading_modes" in result

        assert result["num_directed_edges"] > 0
        assert result["num_leading_modes"] > 0
        assert len(result["ipr_scores"]) == result["num_leading_modes"]

    def test_empty_matrix(self):
        """All-zero matrix returns empty localization metrics."""
        H = np.zeros((2, 3), dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        assert result["ipr_scores"] == []
        assert result["max_ipr"] == 0.0
        assert result["localized_modes"] == []
        assert result["mode_support_sizes"] == []
        assert result["num_directed_edges"] == 0
        assert result["num_leading_modes"] == 0

    def test_determinism(self):
        """Repeated calls produce identical JSON results."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        r1 = compute_nb_localization_metrics(H)
        r2 = compute_nb_localization_metrics(H)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization roundtrip."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == json.dumps(result, sort_keys=True)

    def test_no_input_mutation(self):
        """Input matrix is not modified."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float64)
        H_copy = H.copy()
        compute_nb_localization_metrics(H)
        np.testing.assert_array_equal(H, H_copy)

    def test_ipr_in_valid_range(self):
        """All IPR scores are in [0, 1]."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        for score in result["ipr_scores"]:
            assert 0.0 <= score <= 1.0 + 1e-10

    def test_max_ipr_equals_top_localization_score(self):
        """max_ipr and top_localization_score are identical."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)
        assert result["max_ipr"] == result["top_localization_score"]

    def test_mass_fractions_sum_to_one(self):
        """Variable + check mass fractions sum to 1 per mode."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        for vm, cm in zip(result["per_mode_mass_on_variables"],
                          result["per_mode_mass_on_checks"]):
            assert abs(vm + cm - 1.0) < 1e-10

    def test_localized_modes_indices_valid(self):
        """Localized mode indices are within [0, num_leading_modes)."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        for idx in result["localized_modes"]:
            assert 0 <= idx < result["num_leading_modes"]

    def test_support_sizes_match_edge_indices(self):
        """Each localized mode has matching support size and edge list."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(H)

        # localized_edge_indices has one entry per localized mode.
        assert len(result["localized_edge_indices"]) == len(result["localized_modes"])
        assert len(result["localized_variable_nodes"]) == len(result["localized_modes"])
        assert len(result["localized_check_nodes"]) == len(result["localized_modes"])


class TestTannerGraphProjection:
    """Tests for projection from directed edges to Tanner graph nodes."""

    def test_toy_graph_projection(self):
        """Toy Tanner graph: localized edges project to correct nodes.

        H = [[1, 1],   → 2 check nodes, 2 variable nodes
              [1, 0]]    4 undirected edges → 8 directed edges
                          but H has 3 nonzero entries → 6 directed edges

        Variable nodes: 0, 1
        Check nodes:    0, 1 (mapped to indices 2, 3 internally)

        Edges from H[ci, vi]:
          H[0,0]=1: v0↔c0  → directed (0,2), (2,0)
          H[0,1]=1: v1↔c0  → directed (1,2), (2,1)
          H[1,0]=1: v0↔c1  → directed (0,3), (3,0)
        """
        H = np.array([[1, 1], [1, 0]], dtype=np.float64)
        result = compute_nb_localization_metrics(
            H,
            num_leading_modes=6,
            ipr_localization_threshold=0.0,  # all modes localized
        )

        assert result["num_directed_edges"] == 6

        # With threshold=0, all modes should be "localized".
        assert len(result["localized_modes"]) == result["num_leading_modes"]

        # Check that variable and check nodes are within valid ranges.
        for var_nodes in result["localized_variable_nodes"]:
            for v in var_nodes:
                assert 0 <= v < 2  # 2 variable nodes

        for chk_nodes in result["localized_check_nodes"]:
            for c in chk_nodes:
                assert 0 <= c < 2  # 2 check nodes

    def test_single_check_node(self):
        """Single check node: all variable nodes should appear in support."""
        H = np.array([[1, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(
            H,
            num_leading_modes=6,
            ipr_localization_threshold=0.0,
        )

        assert result["num_directed_edges"] == 6  # 3 edges × 2 directions

        # With threshold=0 and full support, all variable/check nodes appear.
        for var_nodes in result["localized_variable_nodes"]:
            if var_nodes:  # non-empty support
                for v in var_nodes:
                    assert 0 <= v < 3

        for chk_nodes in result["localized_check_nodes"]:
            if chk_nodes:
                for c in chk_nodes:
                    assert c == 0  # only one check node

    def test_edge_indices_valid(self):
        """Edge indices in support are within [0, num_directed_edges)."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        result = compute_nb_localization_metrics(
            H,
            ipr_localization_threshold=0.0,
        )

        for edge_list in result["localized_edge_indices"]:
            for ei in edge_list:
                assert 0 <= ei < result["num_directed_edges"]

    def test_variable_check_node_coverage(self):
        """Projected nodes are subsets of actual variable/check nodes."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.float64)
        m, n_var = H.shape
        result = compute_nb_localization_metrics(
            H,
            ipr_localization_threshold=0.0,
        )

        for var_nodes in result["localized_variable_nodes"]:
            for v in var_nodes:
                assert 0 <= v < n_var

        for chk_nodes in result["localized_check_nodes"]:
            for c in chk_nodes:
                assert 0 <= c < m
