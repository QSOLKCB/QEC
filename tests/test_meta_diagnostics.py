"""Tests for meta_diagnostics.py — v105.0.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.meta_diagnostics import (
    ROUND_PRECISION,
    classify_invariants,
    compare_runs,
    format_meta_diagnostics,
    run_meta_diagnostics_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_run_result(
    diagnosis: str = "oscillatory_trap",
    topology: str = "mixed",
    invariants: list | None = None,
) -> dict:
    if invariants is None:
        invariants = [
            {
                "name": "stability_monotonicity",
                "type": "geometry",
                "strength": 1.0,
                "support": 2,
                "total": 2,
            },
            {
                "name": "basin_switch_suppression",
                "type": "sign",
                "strength": 0.8,
                "support": 2,
                "total": 2,
            },
        ]
    return {
        "global_metrics": {
            "primary_diagnosis": diagnosis,
            "topology_type": topology,
        },
        "scored_invariants": invariants,
    }


def _make_registry(count: int = 5, total_runs: int = 5) -> dict:
    return {
        "geometry:stability_monotonicity": {
            "count": count,
            "avg_strength": 0.9,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": total_runs - 1,
        },
        "sign:basin_switch_suppression": {
            "count": max(1, count // 2),
            "avg_strength": 0.6,
            "max_strength": 0.8,
            "first_seen": 0,
            "last_seen": total_runs - 1,
        },
    }


# ---------------------------------------------------------------------------
# Tests: compare_runs
# ---------------------------------------------------------------------------


class TestCompareRuns:
    def test_empty_input(self):
        result = compare_runs([])
        assert result["run_count"] == 0
        assert result["invariant_overlap"] == []

    def test_single_run(self):
        result = compare_runs([_make_run_result()])
        assert result["run_count"] == 1
        assert len(result["invariant_overlap"]) == 2  # all present in one run

    def test_overlap_across_identical_runs(self):
        runs = [_make_run_result() for _ in range(3)]
        result = compare_runs(runs)
        assert len(result["invariant_overlap"]) == 2

    def test_divergence_when_different(self):
        r1 = _make_run_result(invariants=[
            {"name": "unique_a", "type": "sign", "strength": 1.0},
        ])
        r2 = _make_run_result(invariants=[
            {"name": "unique_b", "type": "sign", "strength": 1.0},
        ])
        result = compare_runs([r1, r2])
        assert len(result["invariant_divergence"]) == 2

    def test_shared_diagnoses(self):
        runs = [
            _make_run_result(diagnosis="oscillatory_trap"),
            _make_run_result(diagnosis="oscillatory_trap"),
            _make_run_result(diagnosis="slow_convergence"),
        ]
        result = compare_runs(runs)
        diag_map = {d["diagnosis"]: d for d in result["shared_diagnoses"]}
        assert diag_map["oscillatory_trap"]["count"] == 2
        assert diag_map["slow_convergence"]["count"] == 1

    def test_shared_topologies(self):
        runs = [
            _make_run_result(topology="mixed"),
            _make_run_result(topology="mixed"),
            _make_run_result(topology="star"),
        ]
        result = compare_runs(runs)
        topo_map = {t["topology"]: t for t in result["shared_topologies"]}
        assert topo_map["mixed"]["count"] == 2

    def test_deterministic(self):
        runs = [_make_run_result() for _ in range(3)]
        assert compare_runs(runs) == compare_runs(runs)

    def test_no_mutation(self):
        runs = [_make_run_result() for _ in range(3)]
        orig = copy.deepcopy(runs)
        compare_runs(runs)
        assert runs == orig

    def test_ratios_bounded(self):
        runs = [_make_run_result() for _ in range(5)]
        result = compare_runs(runs)
        for entry in result["shared_diagnoses"]:
            assert 0.0 <= entry["ratio"] <= 1.0
        for entry in result["shared_topologies"]:
            assert 0.0 <= entry["ratio"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: classify_invariants
# ---------------------------------------------------------------------------


class TestClassifyInvariants:
    def test_empty_registry(self):
        result = classify_invariants({})
        assert result == {"universal": [], "contextual": [], "rare": []}

    def test_universal_classification(self):
        reg = _make_registry(count=5, total_runs=5)
        result = classify_invariants(reg)
        universal_keys = [u["invariant"] for u in result["universal"]]
        assert "geometry:stability_monotonicity" in universal_keys

    def test_rare_classification(self):
        reg = {
            "sign:rare_inv": {
                "count": 1,
                "avg_strength": 0.5,
                "max_strength": 0.5,
                "first_seen": 0,
                "last_seen": 9,
            },
        }
        result = classify_invariants(reg)
        rare_keys = [r["invariant"] for r in result["rare"]]
        assert "sign:rare_inv" in rare_keys

    def test_contextual_classification(self):
        reg = {
            "sign:mid_inv": {
                "count": 3,
                "avg_strength": 0.7,
                "max_strength": 0.9,
                "first_seen": 0,
                "last_seen": 9,
            },
        }
        result = classify_invariants(reg)
        ctx_keys = [c["invariant"] for c in result["contextual"]]
        assert "sign:mid_inv" in ctx_keys

    def test_deterministic(self):
        reg = _make_registry()
        assert classify_invariants(reg) == classify_invariants(reg)

    def test_avg_strength_bounded(self):
        reg = _make_registry()
        result = classify_invariants(reg)
        for category in ("universal", "contextual", "rare"):
            for item in result[category]:
                assert 0.0 <= item["avg_strength"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: run_meta_diagnostics_analysis
# ---------------------------------------------------------------------------


class TestRunMetaDiagnosticsAnalysis:
    def test_returns_required_keys(self):
        runs = [_make_run_result() for _ in range(3)]
        result = run_meta_diagnostics_analysis(runs)
        assert "comparison" in result
        assert "classification" in result
        assert "registry" in result

    def test_with_precomputed_registry(self):
        runs = [_make_run_result() for _ in range(3)]
        reg = _make_registry()
        result = run_meta_diagnostics_analysis(runs, registry=reg)
        assert result["registry"] is reg

    def test_deterministic(self):
        runs = [_make_run_result() for _ in range(3)]
        r1 = run_meta_diagnostics_analysis(runs)
        r2 = run_meta_diagnostics_analysis(runs)
        assert r1 == r2

    def test_no_mutation(self):
        runs = [_make_run_result() for _ in range(3)]
        orig = copy.deepcopy(runs)
        run_meta_diagnostics_analysis(runs)
        assert runs == orig

    def test_empty_input(self):
        result = run_meta_diagnostics_analysis([])
        assert result["comparison"]["run_count"] == 0


# ---------------------------------------------------------------------------
# Tests: format_meta_diagnostics
# ---------------------------------------------------------------------------


class TestFormatMetaDiagnostics:
    def test_contains_header(self):
        runs = [_make_run_result() for _ in range(3)]
        result = run_meta_diagnostics_analysis(runs)
        text = format_meta_diagnostics(result)
        assert "Meta-Diagnostics" in text
        assert "Runs Compared: 3" in text

    def test_shows_classification(self):
        runs = [_make_run_result() for _ in range(3)]
        result = run_meta_diagnostics_analysis(runs)
        text = format_meta_diagnostics(result)
        assert "Universal" in text

    def test_empty_result(self):
        result = {
            "comparison": {"run_count": 0, "invariant_overlap": [],
                          "invariant_divergence": [], "shared_diagnoses": [],
                          "shared_topologies": []},
            "classification": {"universal": [], "contextual": [], "rare": []},
            "registry": {},
        }
        text = format_meta_diagnostics(result)
        assert "Runs Compared: 0" in text
