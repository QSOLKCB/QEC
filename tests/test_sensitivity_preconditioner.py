"""
Tests for v7.6.0 — Sensitivity-Based Preconditioner for Graph Optimization.

Verifies:
  - Determinism: repeated runs produce identical artifacts
  - Baseline preservation: disable_preconditioner matches v7.5 behavior
  - Improvement: final_score <= initial_score
  - Candidate generation: sensitivity-weighted, degree-preserving
  - Experiment report: comparative metrics correct
  - JSON artifact stability
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

from qec.experiments.sensitivity_preconditioner import (
    _generate_sensitivity_weighted_candidates,
    run_sensitivity_preconditioned_optimization,
    run_sensitivity_preconditioner_experiment,
)
from qec.diagnostics.sensitivity_map import (
    compute_proxy_sensitivity_scores,
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

    def test_preconditioned_deterministic(self):
        H = _make_simple_code()
        r1 = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=True,
        )
        r2 = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=True,
        )
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_baseline_mode_deterministic(self):
        H = _make_simple_code()
        r1 = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=False,
        )
        r2 = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=False,
        )
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2

    def test_experiment_deterministic(self):
        H = _make_simple_code()
        r1 = run_sensitivity_preconditioner_experiment(
            H, max_iterations=2, max_candidates=5,
        )
        r2 = run_sensitivity_preconditioner_experiment(
            H, max_iterations=2, max_candidates=5,
        )
        j1 = json.dumps(r1, sort_keys=True)
        j2 = json.dumps(r2, sort_keys=True)
        assert j1 == j2


# ── Test: Baseline Preservation ───────────────────────────────────────


class TestBaselinePreservation:
    """Disabled preconditioner preserves baseline behavior."""

    def test_preconditioner_flag_respected(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=False,
        )
        assert result["preconditioner_enabled"] is False
        assert result["sensitivity_map_summary"] is None

    def test_preconditioner_enabled_flag(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=True,
        )
        assert result["preconditioner_enabled"] is True


# ── Test: Improvement ─────────────────────────────────────────────────


class TestImprovement:
    """final_score <= initial_score."""

    def test_score_nonincreasing_preconditioned(self):
        H = _make_larger_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=5, enable_preconditioner=True,
        )
        assert result["final_instability_score"] <= result["initial_instability_score"]

    def test_score_nonincreasing_baseline(self):
        H = _make_larger_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=5, enable_preconditioner=False,
        )
        assert result["final_instability_score"] <= result["initial_instability_score"]


# ── Test: Candidate Generation ────────────────────────────────────────


class TestCandidateGeneration:
    """Sensitivity-weighted candidates are valid."""

    def test_candidates_generated(self):
        H = _make_larger_code()
        proxy = compute_proxy_sensitivity_scores(H)
        candidates = _generate_sensitivity_weighted_candidates(
            H, proxy, max_candidates=5,
        )
        assert isinstance(candidates, list)

    def test_candidates_have_required_keys(self):
        H = _make_larger_code()
        proxy = compute_proxy_sensitivity_scores(H)
        candidates = _generate_sensitivity_weighted_candidates(
            H, proxy, max_candidates=5,
        )
        for c in candidates:
            assert "remove" in c
            assert "add" in c
            assert "description" in c
            assert "sensitivity_score" in c

    def test_max_candidates_respected(self):
        H = _make_larger_code()
        proxy = compute_proxy_sensitivity_scores(H)
        candidates = _generate_sensitivity_weighted_candidates(
            H, proxy, max_candidates=3,
        )
        assert len(candidates) <= 3

    def test_degree_preservation(self):
        H = _make_larger_code()
        proxy = compute_proxy_sensitivity_scores(H)
        candidates = _generate_sensitivity_weighted_candidates(
            H, proxy, max_candidates=10,
        )
        from qec.experiments.tanner_graph_repair import (
            _extract_edges, _apply_swap,
        )
        edges = _extract_edges(H)

        def degree_map(edge_list):
            deg = {}
            for u, v in edge_list:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            return deg

        orig_deg = degree_map(edges)
        for c in candidates:
            new_edges = _apply_swap(edges, c)
            assert degree_map(new_edges) == orig_deg

    def test_empty_proxy_scores(self):
        H = _make_simple_code()
        candidates = _generate_sensitivity_weighted_candidates(
            H, [], max_candidates=5,
        )
        assert candidates == []


# ── Test: Experiment Report ───────────────────────────────────────────


class TestExperimentReport:
    """Comparative experiment report is complete."""

    def test_report_keys_present(self):
        H = _make_simple_code()
        report = run_sensitivity_preconditioner_experiment(
            H, max_iterations=2,
        )
        assert "baseline" in report
        assert "preconditioned" in report
        assert "sensitivity_map" in report
        assert "comparison" in report

    def test_comparison_keys(self):
        H = _make_simple_code()
        report = run_sensitivity_preconditioner_experiment(
            H, max_iterations=2,
        )
        comp = report["comparison"]
        assert "baseline_final_score" in comp
        assert "preconditioned_final_score" in comp
        assert "improvement_delta" in comp
        assert "preconditioner_better" in comp

    def test_comparison_delta_consistent(self):
        H = _make_larger_code()
        report = run_sensitivity_preconditioner_experiment(
            H, max_iterations=3,
        )
        comp = report["comparison"]
        expected = round(
            comp["baseline_final_score"] - comp["preconditioned_final_score"],
            12,
        )
        assert comp["improvement_delta"] == expected


# ── Test: JSON Artifact Stability ─────────────────────────────────────


class TestJSONStability:
    """Serialization roundtrip stability."""

    def test_optimization_json_serializable(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=2, enable_preconditioner=True,
        )
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_optimization_json_roundtrip(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=2, enable_preconditioner=True,
        )
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized

    def test_experiment_json_roundtrip(self):
        H = _make_simple_code()
        report = run_sensitivity_preconditioner_experiment(
            H, max_iterations=2,
        )
        serialized = json.dumps(report, sort_keys=True)
        deserialized = json.loads(serialized)
        reserialized = json.dumps(deserialized, sort_keys=True)
        assert serialized == reserialized

    def test_all_output_keys_present(self):
        H = _make_larger_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=3, enable_preconditioner=True,
        )
        expected_keys = {
            "initial_instability_score",
            "final_instability_score",
            "iterations",
            "swaps_applied",
            "preconditioner_enabled",
            "optimizer_success",
            "sensitivity_map_summary",
        }
        assert expected_keys.issubset(result.keys())

    def test_floats_rounded(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=2, enable_preconditioner=True,
        )
        score = result["initial_instability_score"]
        assert isinstance(score, float)
        assert score == round(score, 12)


# ── Test: Layer Safety ────────────────────────────────────────────────


class TestLayerSafety:
    """Preconditioner module does NOT import decoder."""

    def test_no_decoder_import(self):
        import qec.experiments.sensitivity_preconditioner as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_bench_import(self):
        import qec.experiments.sensitivity_preconditioner as mod
        source = open(mod.__file__).read()
        assert "from bench" not in source
        assert "import src.bench" not in source


# ── Test: Edge Cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the preconditioned optimizer."""

    def test_zero_iterations(self):
        H = _make_simple_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=0, enable_preconditioner=True,
        )
        assert result["iterations"] == 0
        assert result["swaps_applied"] == []

    def test_single_edge_matrix(self):
        H = np.array([[1]], dtype=np.float64)
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=2, enable_preconditioner=True,
        )
        assert result["iterations"] >= 0
        assert result["final_instability_score"] <= result["initial_instability_score"]

    def test_optimizer_success_flag(self):
        H = _make_larger_code()
        result = run_sensitivity_preconditioned_optimization(
            H, max_iterations=10, enable_preconditioner=True,
        )
        if result["final_instability_score"] < result["initial_instability_score"]:
            assert result["optimizer_success"] is True
        else:
            assert result["optimizer_success"] is False
