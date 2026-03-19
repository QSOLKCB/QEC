"""
Tests for spectral Tanner graph design rules (v7.4).

Verifies:
  - Deterministic design score calculation
  - Score bounds [0, 1]
  - Spectral gap detection and structural risk
  - JSON serialization stability
  - Sorted node lists
  - No decoder import
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

from qec.experiments.spectral_graph_design_rules import (
    compute_spectral_design_score,
    compute_spectral_gap,
    detect_structural_risk_patterns,
    run_spectral_graph_design_analysis,
)


# ── Toy inputs ────────────────────────────────────────────────────────

def _make_nb_spectrum_result(
    eigenvalues: list[list[float]],
    spectral_radius: float,
) -> dict[str, object]:
    """Build a minimal v6.0-compatible NB spectrum result."""
    return {
        "nb_eigenvalues": eigenvalues,
        "spectral_radius": spectral_radius,
        "num_eigenvalues": len(eigenvalues),
    }


def _make_localization_result(
    max_ipr: float,
    localized_variable_nodes: list[int],
) -> dict[str, object]:
    """Build a minimal v6.1-compatible localization result."""
    return {
        "max_ipr": max_ipr,
        "localized_variable_nodes": localized_variable_nodes,
    }


def _make_risk_result(
    cluster_risk_scores: list[float],
    max_cluster_risk: float,
    top_risk_clusters: list[int],
) -> dict[str, object]:
    """Build a minimal v6.4-compatible risk result."""
    return {
        "cluster_risk_scores": cluster_risk_scores,
        "max_cluster_risk": max_cluster_risk,
        "top_risk_clusters": top_risk_clusters,
    }


# ── Test: Spectral Design Score ──────────────────────────────────────

class TestSpectralDesignScore:
    """Tests for compute_spectral_design_score."""

    def test_all_zero_yields_zero(self):
        """All-zero inputs produce score 0."""
        score = compute_spectral_design_score(0.0, 0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_one_yields_one(self):
        """All-one inputs produce score 1."""
        score = compute_spectral_design_score(1.0, 1.0, 1.0, 1.0, 1.0)
        assert score == 1.0

    def test_score_bounds(self):
        """Score is always in [0, 1]."""
        for vals in [
            (0.5, 0.5, 0.5, 0.5, 0.5),
            (0.1, 0.9, 0.0, 0.3, 0.7),
            (1.0, 0.0, 1.0, 0.0, 1.0),
        ]:
            score = compute_spectral_design_score(*vals)
            assert 0.0 <= score <= 1.0

    def test_input_clamping(self):
        """Inputs outside [0, 1] are clamped."""
        score = compute_spectral_design_score(2.0, -0.5, 1.5, 3.0, -1.0)
        # After clamping: (1.0, 0.0, 1.0, 1.0, 0.0)
        # = 0.30*1 + 0.20*0 + 0.20*1 + 0.15*1 + 0.15*0 = 0.65
        assert score == 0.65

    def test_weighted_computation(self):
        """Verify weighted sum with known values."""
        score = compute_spectral_design_score(0.5, 0.4, 0.3, 0.2, 0.1)
        expected = 0.30 * 0.5 + 0.20 * 0.4 + 0.20 * 0.3 + 0.15 * 0.2 + 0.15 * 0.1
        assert abs(score - round(expected, 12)) < 1e-12

    def test_custom_weights(self):
        """Custom weights are respected."""
        score = compute_spectral_design_score(
            1.0, 0.0, 0.0, 0.0, 0.0,
            w1=0.5, w2=0.1, w3=0.1, w4=0.1, w5=0.2,
        )
        assert score == 0.5

    def test_rounded_to_12_decimals(self):
        """Score is rounded to 12 decimal places."""
        score = compute_spectral_design_score(0.3, 0.3, 0.3, 0.3, 0.3)
        as_str = f"{score:.15f}"
        # At most 12 significant decimal digits.
        assert score == round(score, 12)


# ── Test: Spectral Gap ───────────────────────────────────────────────

class TestSpectralGap:
    """Tests for compute_spectral_gap."""

    def test_basic_gap(self):
        """Basic spectral gap computation."""
        eigenvalues = [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]]
        result = compute_spectral_gap(eigenvalues)
        assert result["spectral_gap"] == 1.0
        assert abs(result["spectral_gap_ratio"] - round(2.0 / 3.0, 12)) < 1e-12

    def test_complex_eigenvalues(self):
        """Works with complex eigenvalues."""
        # |3+4i| = 5, |3+0i| = 3
        eigenvalues = [[3.0, 4.0], [3.0, 0.0], [1.0, 0.0]]
        result = compute_spectral_gap(eigenvalues)
        assert result["spectral_gap"] == 2.0
        assert abs(result["spectral_gap_ratio"] - round(3.0 / 5.0, 12)) < 1e-12

    def test_single_eigenvalue(self):
        """Fewer than 2 eigenvalues returns zeros."""
        result = compute_spectral_gap([[5.0, 0.0]])
        assert result["spectral_gap"] == 0.0
        assert result["spectral_gap_ratio"] == 0.0

    def test_empty_eigenvalues(self):
        """Empty eigenvalue list returns zeros."""
        result = compute_spectral_gap([])
        assert result["spectral_gap"] == 0.0
        assert result["spectral_gap_ratio"] == 0.0

    def test_zero_dominant_eigenvalue(self):
        """Zero dominant eigenvalue avoids division by zero."""
        eigenvalues = [[0.0, 0.0], [0.0, 0.0]]
        result = compute_spectral_gap(eigenvalues)
        assert result["spectral_gap"] == 0.0
        assert result["spectral_gap_ratio"] == 0.0

    def test_rounded_to_12_decimals(self):
        """Gap values rounded to 12 decimals."""
        eigenvalues = [[3.0, 0.0], [1.0, 0.0]]
        result = compute_spectral_gap(eigenvalues)
        assert result["spectral_gap"] == round(result["spectral_gap"], 12)
        assert result["spectral_gap_ratio"] == round(result["spectral_gap_ratio"], 12)


# ── Test: Structural Risk Pattern Detection ──────────────────────────

class TestStructuralRiskPatterns:
    """Tests for detect_structural_risk_patterns."""

    def test_no_risk_detected(self):
        """No risk patterns when all metrics are safe."""
        result = detect_structural_risk_patterns(
            max_ipr=0.1,
            cluster_risk_scores=[0.1, 0.2],
            spectral_gap=1.0,
            spectral_instability_ratio=0.5,
            localized_variable_nodes=[],
            top_risk_clusters=[],
        )
        assert result["structural_risk_detected"] is False
        assert result["spectral_gap_small"] is False
        assert result["high_risk_clusters"] == []

    def test_small_spectral_gap_triggers_risk(self):
        """Small spectral gap triggers structural risk detection."""
        result = detect_structural_risk_patterns(
            max_ipr=0.1,
            cluster_risk_scores=[0.1],
            spectral_gap=0.1,
            spectral_instability_ratio=0.5,
            localized_variable_nodes=[],
            top_risk_clusters=[],
        )
        assert result["spectral_gap_small"] is True
        assert result["structural_risk_detected"] is True

    def test_high_ipr_triggers_risk(self):
        """High IPR triggers structural risk detection."""
        result = detect_structural_risk_patterns(
            max_ipr=0.5,
            cluster_risk_scores=[0.1],
            spectral_gap=1.0,
            spectral_instability_ratio=0.5,
            localized_variable_nodes=[3, 1, 5],
            top_risk_clusters=[],
        )
        assert result["structural_risk_detected"] is True
        assert result["high_localization_nodes"] == [1, 3, 5]

    def test_high_cluster_risk_triggers_risk(self):
        """High cluster risk triggers structural risk detection."""
        result = detect_structural_risk_patterns(
            max_ipr=0.1,
            cluster_risk_scores=[0.1, 0.8, 0.2],
            spectral_gap=1.0,
            spectral_instability_ratio=0.5,
            localized_variable_nodes=[],
            top_risk_clusters=[1],
        )
        assert result["structural_risk_detected"] is True
        assert result["high_risk_clusters"] == [1]

    def test_high_instability_ratio_triggers_risk(self):
        """High instability ratio triggers risk."""
        result = detect_structural_risk_patterns(
            max_ipr=0.1,
            cluster_risk_scores=[0.1],
            spectral_gap=1.0,
            spectral_instability_ratio=1.5,
            localized_variable_nodes=[],
            top_risk_clusters=[],
        )
        assert result["structural_risk_detected"] is True

    def test_node_lists_sorted(self):
        """Node and cluster lists are deterministically sorted."""
        result = detect_structural_risk_patterns(
            max_ipr=0.5,
            cluster_risk_scores=[0.8, 0.6, 0.9],
            spectral_gap=0.1,
            spectral_instability_ratio=0.5,
            localized_variable_nodes=[10, 3, 7, 1],
            top_risk_clusters=[2, 0],
        )
        assert result["high_localization_nodes"] == [1, 3, 7, 10]
        assert result["high_risk_clusters"] == sorted(result["high_risk_clusters"])


# ── Test: Full Pipeline ──────────────────────────────────────────────

class TestRunSpectralGraphDesignAnalysis:
    """Tests for run_spectral_graph_design_analysis."""

    def _run_basic_analysis(self):
        """Helper returning a basic analysis result."""
        nb = _make_nb_spectrum_result(
            eigenvalues=[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            spectral_radius=3.0,
        )
        loc = _make_localization_result(max_ipr=0.4, localized_variable_nodes=[1, 3])
        risk = _make_risk_result(
            cluster_risk_scores=[0.3, 0.7],
            max_cluster_risk=0.7,
            top_risk_clusters=[1],
        )
        return run_spectral_graph_design_analysis(
            nb_spectrum_result=nb,
            localization_result=loc,
            risk_result=risk,
            spectral_instability_ratio=0.8,
            avg_variable_degree=3.0,
            avg_check_degree=4.0,
        )

    def test_all_keys_present(self):
        """All expected artifact keys are present."""
        result = self._run_basic_analysis()
        expected_keys = {
            "graph_design_score",
            "nb_spectral_radius",
            "spectral_gap",
            "spectral_gap_ratio",
            "spectral_instability_ratio",
            "max_ipr",
            "max_cluster_risk",
            "structural_risk_detected",
            "high_localization_nodes",
            "high_risk_clusters",
        }
        assert set(result.keys()) == expected_keys

    def test_score_in_bounds(self):
        """Design score is in [0, 1]."""
        result = self._run_basic_analysis()
        assert 0.0 <= result["graph_design_score"] <= 1.0

    def test_json_serializable(self):
        """Output is JSON-serializable."""
        result = self._run_basic_analysis()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert json.dumps(deserialized, sort_keys=True) == \
               json.dumps(result, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON roundtrip."""
        result = self._run_basic_analysis()
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized

    def test_sorted_node_lists(self):
        """Node and cluster lists are sorted."""
        result = self._run_basic_analysis()
        assert result["high_localization_nodes"] == sorted(
            result["high_localization_nodes"]
        )
        assert result["high_risk_clusters"] == sorted(
            result["high_risk_clusters"]
        )

    def test_empty_inputs(self):
        """Handles empty/minimal inputs gracefully."""
        nb = _make_nb_spectrum_result([], 0.0)
        loc = _make_localization_result(0.0, [])
        risk = _make_risk_result([], 0.0, [])
        result = run_spectral_graph_design_analysis(
            nb, loc, risk, 0.0, 0.0, 0.0,
        )
        # Empty eigenvalues → spectral_gap=0 → gap_penalty=1.0 → 0.15*1.0=0.15
        assert result["graph_design_score"] == 0.15
        assert result["spectral_gap"] == 0.0

    def test_floats_rounded_to_12_decimals(self):
        """All float values are rounded to 12 decimals."""
        result = self._run_basic_analysis()
        for key in [
            "graph_design_score", "nb_spectral_radius", "spectral_gap",
            "spectral_gap_ratio", "spectral_instability_ratio",
            "max_ipr", "max_cluster_risk",
        ]:
            val = result[key]
            assert val == round(val, 12), f"{key} not rounded to 12 decimals"


# ── Test: Determinism ────────────────────────────────────────────────

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_calls_identical(self):
        """Repeated calls produce identical JSON results."""
        nb = _make_nb_spectrum_result(
            eigenvalues=[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
            spectral_radius=3.0,
        )
        loc = _make_localization_result(0.4, [1, 3])
        risk = _make_risk_result([0.3, 0.7], 0.7, [1])

        r1 = run_spectral_graph_design_analysis(nb, loc, risk, 0.8, 3.0, 4.0)
        r2 = run_spectral_graph_design_analysis(nb, loc, risk, 0.8, 3.0, 4.0)

        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_gap_determinism(self):
        """Spectral gap computation is deterministic."""
        eigenvalues = [[3.0, 4.0], [2.5, 1.0], [1.0, 0.0]]
        r1 = compute_spectral_gap(eigenvalues)
        r2 = compute_spectral_gap(eigenvalues)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_design_score_determinism(self):
        """Design score computation is deterministic."""
        s1 = compute_spectral_design_score(0.5, 0.4, 0.3, 0.2, 0.1)
        s2 = compute_spectral_design_score(0.5, 0.4, 0.3, 0.2, 0.1)
        assert s1 == s2


# ── Test: No Decoder Import ─────────────────────────────────────────

class TestNoDecoderImport:
    """Tests that this module does not import decoder code."""

    def test_no_decoder_import(self):
        """This module does not import any decoder code."""
        import qec.experiments.spectral_graph_design_rules as mod
        source = open(mod.__file__).read()
        assert "bp_decode" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
