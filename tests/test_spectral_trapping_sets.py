"""
Tests for v5.6.0 — Spectral Trapping-Set Diagnostics.

Verifies correct cluster detection, deterministic repeated execution,
empty cluster handling, JSON serialization stability, and input
validation.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from qec.diagnostics.spectral_trapping_sets import (
    compute_spectral_trapping_sets,
)


# ── Test helpers ──────────────────────────────────────────────────────

def _localized_mode(n_var: int = 10, n_check: int = 5) -> np.ndarray:
    """Eigenvector with mass concentrated on first two variable nodes."""
    v = np.zeros(n_var + n_check, dtype=np.float64)
    v[0] = 5.0
    v[1] = 4.0
    # Small values on remaining variable nodes.
    for i in range(2, n_var):
        v[i] = 0.1
    return v


def _uniform_mode(n_var: int = 10, n_check: int = 5) -> np.ndarray:
    """Eigenvector with uniform mass — no localization."""
    v = np.ones(n_var + n_check, dtype=np.float64)
    return v


def _zero_mode(n_var: int = 10, n_check: int = 5) -> np.ndarray:
    """Zero eigenvector."""
    return np.zeros(n_var + n_check, dtype=np.float64)


def _mixed_modes() -> list[np.ndarray]:
    """Three modes: localized, uniform, and zero."""
    return [_localized_mode(), _uniform_mode(), _zero_mode()]


# ── Cluster detection tests ──────────────────────────────────────────

class TestClusterDetection:
    """Verify correct identification of localized clusters."""

    def test_localized_mode_detected(self):
        """Strongly localized mode should produce a cluster."""
        modes = [_localized_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        assert out["cluster_count"] >= 1
        # The first two nodes carry dominant mass.
        cluster = out["clusters"][0]
        assert 0 in cluster["nodes"]
        assert 1 in cluster["nodes"]

    def test_uniform_mode_no_cluster(self):
        """Uniform mode should produce no cluster (all equal importance)."""
        modes = [_uniform_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        # Uniform importance: no node exceeds mean + std.
        assert out["cluster_count"] == 0
        assert out["clusters"] == []
        assert out["largest_cluster_size"] == 0
        assert out["mean_cluster_size"] == 0.0

    def test_zero_mode_no_cluster(self):
        """Zero mode should produce no cluster."""
        modes = [_zero_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        assert out["cluster_count"] == 0

    def test_mixed_modes(self):
        """Only localized modes should produce clusters."""
        modes = _mixed_modes()
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        # Only the localized mode (index 0) should yield a cluster.
        assert out["cluster_count"] >= 1
        mode_indices = [c["mode_index"] for c in out["clusters"]]
        assert 0 in mode_indices

    def test_cluster_size_reasonable(self):
        """Cluster size should be smaller than total variable nodes."""
        modes = [_localized_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        for cluster in out["clusters"]:
            assert cluster["cluster_size"] < 10

    def test_importance_values(self):
        """Max importance should be >= mean importance in each cluster."""
        modes = [_localized_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        for cluster in out["clusters"]:
            assert cluster["max_importance"] >= cluster["mean_importance"]


# ── Determinism tests ────────────────────────────────────────────────

class TestDeterminism:
    """Verify byte-identical outputs across repeated calls."""

    def test_localized_determinism(self):
        modes = [_localized_mode()]
        out1 = compute_spectral_trapping_sets(modes, variable_node_count=10)
        out2 = compute_spectral_trapping_sets(modes, variable_node_count=10)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2

    def test_mixed_determinism(self):
        modes = _mixed_modes()
        out1 = compute_spectral_trapping_sets(modes, variable_node_count=10)
        out2 = compute_spectral_trapping_sets(modes, variable_node_count=10)
        j1 = json.dumps(out1, sort_keys=True)
        j2 = json.dumps(out2, sort_keys=True)
        assert j1 == j2


# ── Empty cluster handling ───────────────────────────────────────────

class TestEmptyClusters:
    """Verify correct behavior when no clusters are found."""

    def test_all_uniform(self):
        """All uniform modes should yield zero clusters."""
        modes = [_uniform_mode(), _uniform_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        assert out["cluster_count"] == 0
        assert out["clusters"] == []
        assert out["largest_cluster_size"] == 0
        assert out["mean_cluster_size"] == 0.0

    def test_all_zero(self):
        """All zero modes should yield zero clusters."""
        modes = [_zero_mode(), _zero_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        assert out["cluster_count"] == 0


# ── JSON serialization tests ─────────────────────────────────────────

class TestJsonSerialization:
    """Verify JSON roundtrip stability."""

    def test_json_roundtrip(self):
        modes = [_localized_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_json_roundtrip_empty(self):
        modes = [_uniform_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        s = json.dumps(out, sort_keys=True)
        roundtrip = json.loads(s)
        assert roundtrip == out

    def test_all_values_json_types(self):
        """All output values must be JSON-native types."""
        modes = [_localized_mode()]
        out = compute_spectral_trapping_sets(modes, variable_node_count=10)
        s = json.dumps(out)
        # Should not raise.
        assert isinstance(s, str)


# ── Input validation tests ───────────────────────────────────────────

class TestInputValidation:
    """Verify correct error handling for invalid inputs."""

    def test_empty_modes_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_spectral_trapping_sets([], variable_node_count=10)

    def test_zero_variable_count_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            compute_spectral_trapping_sets(
                [_localized_mode()], variable_node_count=0,
            )

    def test_negative_variable_count_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            compute_spectral_trapping_sets(
                [_localized_mode()], variable_node_count=-1,
            )

    def test_short_eigenvector_raises(self):
        """Eigenvector shorter than variable_node_count should raise."""
        short = np.array([1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="components"):
            compute_spectral_trapping_sets([short], variable_node_count=10)

    def test_no_input_mutation(self):
        """Input arrays must not be modified."""
        modes = [_localized_mode()]
        modes_copy = [m.copy() for m in modes]
        compute_spectral_trapping_sets(modes, variable_node_count=10)
        for orig, copy in zip(modes, modes_copy):
            assert np.array_equal(orig, copy)


# ── Output structure tests ───────────────────────────────────────────

class TestOutputStructure:
    """Verify all required keys are present and have correct types."""

    REQUIRED_KEYS = {
        "cluster_count": int,
        "clusters": list,
        "largest_cluster_size": int,
        "mean_cluster_size": float,
    }

    CLUSTER_KEYS = {
        "mode_index": int,
        "cluster_size": int,
        "nodes": list,
        "max_importance": float,
        "mean_importance": float,
    }

    def test_all_keys_present(self):
        out = compute_spectral_trapping_sets(
            [_localized_mode()], variable_node_count=10,
        )
        for key in self.REQUIRED_KEYS:
            assert key in out, f"Missing key: {key}"

    def test_all_types_correct(self):
        out = compute_spectral_trapping_sets(
            [_localized_mode()], variable_node_count=10,
        )
        for key, expected_type in self.REQUIRED_KEYS.items():
            assert isinstance(out[key], expected_type), (
                f"Key {key}: expected {expected_type}, got {type(out[key])}"
            )

    def test_cluster_keys_present(self):
        out = compute_spectral_trapping_sets(
            [_localized_mode()], variable_node_count=10,
        )
        for cluster in out["clusters"]:
            for key in self.CLUSTER_KEYS:
                assert key in cluster, f"Missing cluster key: {key}"

    def test_cluster_types_correct(self):
        out = compute_spectral_trapping_sets(
            [_localized_mode()], variable_node_count=10,
        )
        for cluster in out["clusters"]:
            for key, expected_type in self.CLUSTER_KEYS.items():
                assert isinstance(cluster[key], expected_type), (
                    f"Cluster key {key}: expected {expected_type}, "
                    f"got {type(cluster[key])}"
                )

    def test_node_indices_are_ints(self):
        out = compute_spectral_trapping_sets(
            [_localized_mode()], variable_node_count=10,
        )
        for cluster in out["clusters"]:
            for node in cluster["nodes"]:
                assert isinstance(node, int)
