"""
Tests for v7.6.0 — Deterministic Instability Sensitivity Maps.

Verifies:
  - Determinism: repeated runs produce identical artifacts
  - Proxy sensitivity: per-edge scores are non-negative, sorted, complete
  - Measured deltas: per-edge deltas are signed, baseline-referenced
  - Combined map: all sections present, JSON-serializable
  - Canonical ordering: deterministic tie-breaking
  - Layer safety: no decoder imports
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

from qec.diagnostics.sensitivity_map import (
    compute_proxy_sensitivity_scores,
    compute_measured_instability_deltas,
    compute_sensitivity_map,
)


# ── Toy inputs ────────────────────────────────────────────────────────


def _make_simple_code() -> np.ndarray:
    """Simple 3x5 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)


def _make_larger_code() -> np.ndarray:
    """Larger 4x7 parity-check matrix."""
    return np.array([
        [1, 1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 0],
    ], dtype=np.float64)


# ── Test: Determinism ─────────────────────────────────────────────────


class TestDeterminism:
    """Repeated runs produce identical artifacts."""

    def test_proxy_sensitivity_deterministic(self):
        H = _make_simple_code()
        s1 = compute_proxy_sensitivity_scores(H)
        s2 = compute_proxy_sensitivity_scores(H)
        assert s1 == s2

    def test_measured_deltas_deterministic(self):
        H = _make_simple_code()
        d1 = compute_measured_instability_deltas(H)
        d2 = compute_measured_instability_deltas(H)
        assert d1 == d2

    def test_sensitivity_map_deterministic(self):
        H = _make_simple_code()
        m1 = compute_sensitivity_map(H)
        m2 = compute_sensitivity_map(H)
        j1 = json.dumps(m1, sort_keys=True)
        j2 = json.dumps(m2, sort_keys=True)
        assert j1 == j2

    def test_larger_code_deterministic(self):
        H = _make_larger_code()
        m1 = compute_sensitivity_map(H)
        m2 = compute_sensitivity_map(H)
        j1 = json.dumps(m1, sort_keys=True)
        j2 = json.dumps(m2, sort_keys=True)
        assert j1 == j2


# ── Test: Proxy Sensitivity Scores ────────────────────────────────────


class TestProxySensitivity:
    """Per-edge proxy scores are non-negative, sorted, complete."""

    def test_scores_nonnegative(self):
        H = _make_simple_code()
        scores = compute_proxy_sensitivity_scores(H)
        for rec in scores:
            assert rec["proxy_sensitivity"] >= 0.0

    def test_scores_sorted_descending(self):
        H = _make_larger_code()
        scores = compute_proxy_sensitivity_scores(H)
        if len(scores) >= 2:
            for i in range(len(scores) - 1):
                assert scores[i]["proxy_sensitivity"] >= scores[i + 1]["proxy_sensitivity"]

    def test_covers_all_edges(self):
        H = _make_simple_code()
        scores = compute_proxy_sensitivity_scores(H)
        num_edges = int(np.sum(H))
        assert len(scores) == num_edges

    def test_edges_are_valid(self):
        H = _make_simple_code()
        m, n = H.shape
        scores = compute_proxy_sensitivity_scores(H)
        for rec in scores:
            vi = rec["variable_node"]
            cj = rec["check_node"]
            assert 0 <= vi < n
            assert n <= cj < n + m
            assert H[cj - n, vi] != 0

    def test_required_keys_present(self):
        H = _make_simple_code()
        scores = compute_proxy_sensitivity_scores(H)
        for rec in scores:
            assert "variable_node" in rec
            assert "check_node" in rec
            assert "proxy_sensitivity" in rec

    def test_floats_rounded_12(self):
        H = _make_larger_code()
        scores = compute_proxy_sensitivity_scores(H)
        for rec in scores:
            val = rec["proxy_sensitivity"]
            assert val == round(val, 12)


# ── Test: Measured Instability Deltas ─────────────────────────────────


class TestMeasuredDeltas:
    """Per-edge deltas are signed, baseline-referenced."""

    def test_deltas_have_baseline(self):
        H = _make_simple_code()
        deltas = compute_measured_instability_deltas(H)
        if deltas:
            baseline = deltas[0]["baseline_score"]
            for rec in deltas:
                assert rec["baseline_score"] == baseline

    def test_deltas_sorted_ascending(self):
        H = _make_larger_code()
        deltas = compute_measured_instability_deltas(H)
        if len(deltas) >= 2:
            for i in range(len(deltas) - 1):
                assert deltas[i]["instability_delta"] <= deltas[i + 1]["instability_delta"]

    def test_covers_all_edges(self):
        H = _make_simple_code()
        deltas = compute_measured_instability_deltas(H)
        num_edges = int(np.sum(H))
        assert len(deltas) == num_edges

    def test_delta_consistency(self):
        """delta = score_without_edge - baseline_score."""
        H = _make_simple_code()
        deltas = compute_measured_instability_deltas(H)
        for rec in deltas:
            expected = round(
                rec["score_without_edge"] - rec["baseline_score"], 12,
            )
            assert rec["instability_delta"] == expected

    def test_required_keys_present(self):
        H = _make_simple_code()
        deltas = compute_measured_instability_deltas(H)
        for rec in deltas:
            assert "variable_node" in rec
            assert "check_node" in rec
            assert "baseline_score" in rec
            assert "score_without_edge" in rec
            assert "instability_delta" in rec


# ── Test: Combined Sensitivity Map ────────────────────────────────────


class TestSensitivityMap:
    """Full map artifact is correct and complete."""

    def test_all_keys_present(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        expected_keys = {
            "matrix_shape",
            "num_edges",
            "baseline_instability_score",
            "proxy_sensitivities",
            "measured_deltas",
            "top_sensitive_edges",
            "summary",
        }
        assert expected_keys.issubset(result.keys())

    def test_matrix_shape_correct(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        assert result["matrix_shape"] == [3, 5]

    def test_num_edges_matches(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        assert result["num_edges"] == int(np.sum(H))

    def test_top_sensitive_edges_bounded(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        assert len(result["top_sensitive_edges"]) <= 3

    def test_top_sensitive_annotated(self):
        H = _make_larger_code()
        result = compute_sensitivity_map(H)
        for rec in result["top_sensitive_edges"]:
            assert "proxy_sensitivity" in rec
            assert "measured_delta" in rec

    def test_summary_keys(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        summary = result["summary"]
        assert "max_proxy_sensitivity" in summary
        assert "mean_proxy_sensitivity" in summary
        assert "min_instability_delta" in summary
        assert "max_instability_delta" in summary
        assert "mean_instability_delta" in summary


# ── Test: JSON Artifact Stability ─────────────────────────────────────


class TestJSONStability:
    """Serialization roundtrip stability."""

    def test_json_serializable(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_json_roundtrip(self):
        H = _make_simple_code()
        result = compute_sensitivity_map(H)
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized

    def test_proxy_json_roundtrip(self):
        H = _make_larger_code()
        scores = compute_proxy_sensitivity_scores(H)
        serialized = json.dumps(scores, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized


# ── Test: Layer Safety ────────────────────────────────────────────────


class TestLayerSafety:
    """Sensitivity map module does NOT import decoder or bench."""

    def test_no_decoder_import(self):
        import qec.diagnostics.sensitivity_map as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_bench_import(self):
        import qec.diagnostics.sensitivity_map as mod
        source = open(mod.__file__).read()
        assert "from bench" not in source
        assert "import src.bench" not in source


# ── Test: Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for sensitivity computations."""

    def test_single_edge_matrix(self):
        H = np.array([[1]], dtype=np.float64)
        scores = compute_proxy_sensitivity_scores(H)
        assert isinstance(scores, list)

    def test_single_edge_deltas(self):
        H = np.array([[1]], dtype=np.float64)
        deltas = compute_measured_instability_deltas(H)
        assert isinstance(deltas, list)

    def test_single_edge_map(self):
        H = np.array([[1]], dtype=np.float64)
        result = compute_sensitivity_map(H)
        assert result["num_edges"] <= 1
