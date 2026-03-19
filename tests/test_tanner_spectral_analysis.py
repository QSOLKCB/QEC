"""
Tests for v5.4.0 — Tanner Spectral Fragility Diagnostics.

Verifies determinism, eigenvalue ordering, valid IPR ranges,
correct node counts, correct node selection ordering,
and JSON serialization.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.tanner_spectral_analysis import (
    compute_tanner_spectral_analysis,
)


# ── Synthetic parity-check matrices ─────────────────────────────────

def _small_H() -> np.ndarray:
    """Small 3×6 parity-check matrix (rate-0.5 repetition-like)."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _identity_H() -> np.ndarray:
    """4×4 identity (degenerate case)."""
    return np.eye(4, dtype=np.float64)


def _dense_H() -> np.ndarray:
    """4×8 denser parity-check matrix."""
    return np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)


def _single_check_H() -> np.ndarray:
    """1×3 single-check code."""
    return np.array([[1, 1, 1]], dtype=np.float64)


# ── Determinism tests ────────────────────────────────────────────────

class TestDeterminism:
    """Verify that outputs are byte-identical across repeated calls."""

    def test_small_H_determinism(self):
        H = _small_H()
        out1 = compute_tanner_spectral_analysis(H)
        out2 = compute_tanner_spectral_analysis(H)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_identity_H_determinism(self):
        H = _identity_H()
        out1 = compute_tanner_spectral_analysis(H)
        out2 = compute_tanner_spectral_analysis(H)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_dense_H_determinism(self):
        H = _dense_H()
        out1 = compute_tanner_spectral_analysis(H)
        out2 = compute_tanner_spectral_analysis(H)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── Output structure tests ───────────────────────────────────────────

class TestOutputStructure:
    """Verify all required keys are present and have correct types."""

    REQUIRED_KEYS = {
        "num_variable_nodes": int,
        "num_check_nodes": int,
        "num_edges": int,
        "largest_eigenvalue": float,
        "adjacency_spectral_gap": float,
        "laplacian_second_eigenvalue": float,
        "spectral_ratio": float,
        "mode_iprs": list,
        "variable_mode_iprs": list,
        "max_mode_ipr": float,
        "max_variable_mode_ipr": float,
        "most_localized_mode_index": int,
        "localized_variable_nodes": list,
        "localized_variable_weights": list,
        "localized_variable_fraction": float,
    }

    def test_all_keys_present(self):
        out = compute_tanner_spectral_analysis(_small_H())
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing key: {key}"

    def test_all_types_correct(self):
        out = compute_tanner_spectral_analysis(_small_H())
        for key, expected_type in self.REQUIRED_KEYS.items():
            assert isinstance(out[key], expected_type), (
                f"Key {key}: expected {expected_type}, got {type(out[key])}"
            )

    def test_json_serializable(self):
        out = compute_tanner_spectral_analysis(_small_H())
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out


# ── Node count tests ─────────────────────────────────────────────────

class TestNodeCounts:
    """Verify node counts match parity-check matrix dimensions."""

    def test_small_H_counts(self):
        H = _small_H()
        out = compute_tanner_spectral_analysis(H)
        assert out["num_check_nodes"] == 3
        assert out["num_variable_nodes"] == 6

    def test_identity_H_counts(self):
        H = _identity_H()
        out = compute_tanner_spectral_analysis(H)
        assert out["num_check_nodes"] == 4
        assert out["num_variable_nodes"] == 4

    def test_dense_H_counts(self):
        H = _dense_H()
        out = compute_tanner_spectral_analysis(H)
        assert out["num_check_nodes"] == 4
        assert out["num_variable_nodes"] == 8

    def test_edge_count(self):
        H = _small_H()
        out = compute_tanner_spectral_analysis(H)
        assert out["num_edges"] == int(np.sum(H != 0))

    def test_single_check_counts(self):
        H = _single_check_H()
        out = compute_tanner_spectral_analysis(H)
        assert out["num_check_nodes"] == 1
        assert out["num_variable_nodes"] == 3
        assert out["num_edges"] == 3


# ── Eigenvalue ordering tests ────────────────────────────────────────

class TestEigenvalueOrdering:
    """Verify spectral properties are correctly computed."""

    def test_largest_eigenvalue_positive(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["largest_eigenvalue"] > 0.0

    def test_spectral_gap_non_negative(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["adjacency_spectral_gap"] >= 0.0

    def test_laplacian_second_eigenvalue_non_negative(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["laplacian_second_eigenvalue"] >= 0.0

    def test_spectral_ratio_non_negative(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["spectral_ratio"] >= 0.0

    def test_identity_eigenvalues(self):
        """Identity H produces a known bipartite structure."""
        out = compute_tanner_spectral_analysis(_identity_H())
        # Bipartite adjacency of identity: eigenvalues are ±1
        assert abs(out["largest_eigenvalue"] - 1.0) < 1e-10


# ── IPR range tests ──────────────────────────────────────────────────

class TestIPRRanges:
    """Verify IPR values are in valid range [0, 1]."""

    def test_mode_iprs_valid(self):
        out = compute_tanner_spectral_analysis(_small_H())
        for ipr in out["mode_iprs"]:
            assert 0.0 <= ipr <= 1.0, f"IPR out of range: {ipr}"

    def test_variable_mode_iprs_valid(self):
        out = compute_tanner_spectral_analysis(_small_H())
        for ipr in out["variable_mode_iprs"]:
            assert 0.0 <= ipr <= 1.0, f"Variable IPR out of range: {ipr}"

    def test_max_mode_ipr_consistent(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["max_mode_ipr"] == max(out["mode_iprs"])

    def test_max_variable_mode_ipr_consistent(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert out["max_variable_mode_ipr"] == max(out["variable_mode_iprs"])

    def test_mode_iprs_count_matches_top_k(self):
        out = compute_tanner_spectral_analysis(_small_H(), top_k_modes=2)
        assert len(out["mode_iprs"]) == 2
        assert len(out["variable_mode_iprs"]) == 2

    def test_top_k_modes_clamped(self):
        """top_k_modes larger than eigenvectors doesn't crash."""
        H = _single_check_H()
        out = compute_tanner_spectral_analysis(H, top_k_modes=100)
        # 1×3 → 4×4 adjacency → 4 eigenvectors
        assert len(out["mode_iprs"]) == 4


# ── Node selection ordering tests ────────────────────────────────────

class TestNodeLocalization:
    """Verify localized node selection is correctly ordered."""

    def test_weights_descending(self):
        out = compute_tanner_spectral_analysis(_small_H())
        weights = out["localized_variable_weights"]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Weights not descending at index {i}: {weights}"
            )

    def test_nodes_are_variable_indices(self):
        H = _small_H()
        out = compute_tanner_spectral_analysis(H)
        n = H.shape[1]
        for node in out["localized_variable_nodes"]:
            assert 0 <= node < n, f"Node index {node} out of range [0, {n})"

    def test_top_k_nodes_respected(self):
        out = compute_tanner_spectral_analysis(_dense_H(), top_k_nodes=3)
        assert len(out["localized_variable_nodes"]) == 3
        assert len(out["localized_variable_weights"]) == 3

    def test_top_k_nodes_clamped(self):
        """top_k_nodes larger than n doesn't crash."""
        H = _single_check_H()
        out = compute_tanner_spectral_analysis(H, top_k_nodes=100)
        assert len(out["localized_variable_nodes"]) == 3  # n=3

    def test_most_localized_mode_index_valid(self):
        out = compute_tanner_spectral_analysis(_small_H(), top_k_modes=3)
        assert 0 <= out["most_localized_mode_index"] < len(out["variable_mode_iprs"])

    def test_weights_non_negative(self):
        out = compute_tanner_spectral_analysis(_dense_H())
        for w in out["localized_variable_weights"]:
            assert w >= 0.0

    def test_localized_variable_fraction_present(self):
        out = compute_tanner_spectral_analysis(_small_H())
        assert "localized_variable_fraction" in out

    def test_localized_variable_fraction_range(self):
        for H in [_small_H(), _identity_H(), _dense_H(), _single_check_H()]:
            out = compute_tanner_spectral_analysis(H)
            assert 0.0 <= out["localized_variable_fraction"] <= 1.0, (
                f"Fraction out of range: {out['localized_variable_fraction']}"
            )

    def test_localized_variable_fraction_all_nodes(self):
        """When top_k_nodes >= n, fraction should be 1.0."""
        H = _single_check_H()
        out = compute_tanner_spectral_analysis(H, top_k_nodes=100)
        assert abs(out["localized_variable_fraction"] - 1.0) < 1e-10


# ── Input handling tests ─────────────────────────────────────────────

class TestInputHandling:
    """Verify the function handles various input types."""

    def test_integer_input(self):
        """Integer parity-check matrix should work."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
        out = compute_tanner_spectral_analysis(H)
        assert out["num_variable_nodes"] == 3
        assert out["num_check_nodes"] == 2

    def test_no_mutation(self):
        """Input matrix must not be modified."""
        H = _small_H()
        H_copy = H.copy()
        compute_tanner_spectral_analysis(H)
        assert np.array_equal(H, H_copy)
