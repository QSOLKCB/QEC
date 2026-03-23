"""Tests for taxonomy-aware diagnostics layer (v93.1.0).

Deterministic, no randomness, no mutation of inputs.
Tests system class inference, taxonomy issues, and recommendations.
"""

import copy

import pytest

from qec.analysis.self_diagnostics import (
    classify_all_systems,
    detect_taxonomy_issues,
    get_mode_metrics,
    infer_system_class,
    print_diagnostics,
    recommend,
    run_self_diagnostics,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _mode_record(mode, comp=0.0, stab=0.0, gain=0, ub=6, ua=6):
    """Build a single aggregated record for a given mode."""
    return {
        "dfa_type": "test",
        "n": 5,
        "mode": mode,
        "compression_efficiency": comp,
        "stability_efficiency": stab,
        "stability_gain": gain,
        "unique_before": ub,
        "unique_after": ua,
    }


def _make_raw_result(dfa_name, n, mode, comp, stab, gain, ub, ua):
    """Build a run_suite()-style result dict."""
    return {
        "dfa_name": dfa_name,
        "n": n,
        "mode": mode,
        "metrics": {
            "compression_efficiency": comp,
            "stability_efficiency": stab,
            "stability_gain": gain,
            "unique_before": ub,
            "unique_after": ua,
            "syndrome_changes": 0,
            "mean_delta": 0.0,
            "stable_before": 4,
            "stable_after": 4,
            "directionality": 0,
        },
        "alignment": [],
    }


# ---------------------------------------------------------------------------
# PART 1 — get_mode_metrics TESTS
# ---------------------------------------------------------------------------


class TestGetModeMetrics:
    """Tests for the get_mode_metrics helper."""

    def test_found_mode(self):
        records = [
            _mode_record("d4", comp=0.5, stab=0.6),
            _mode_record("none", comp=0.0, stab=0.0),
        ]
        result = get_mode_metrics(records, "d4")
        assert result["compression_efficiency"] == 0.5
        assert result["stability_efficiency"] == 0.6

    def test_missing_mode_returns_zeros(self):
        records = [_mode_record("d4", comp=0.5, stab=0.6)]
        result = get_mode_metrics(records, "inv")
        assert result["compression_efficiency"] == 0.0
        assert result["stability_efficiency"] == 0.0

    def test_empty_records(self):
        result = get_mode_metrics([], "d4")
        assert result["compression_efficiency"] == 0.0
        assert result["stability_efficiency"] == 0.0


# ---------------------------------------------------------------------------
# PART 2 — SYSTEM CLASS INFERENCE TESTS
# ---------------------------------------------------------------------------


class TestInferSystemClass:
    """Tests for deterministic system class inference."""

    def test_degenerate_low_unique(self):
        records = [
            _mode_record("none", comp=0.0, stab=0.0, ub=2),
            _mode_record("d4", comp=0.3, stab=0.3, ub=2),
        ]
        assert infer_system_class(records) == "degenerate"

    def test_degenerate_single_state(self):
        records = [_mode_record("none", comp=0.0, stab=0.0, ub=1)]
        assert infer_system_class(records) == "degenerate"

    def test_chain_like(self):
        """d4 stability dominates square and is above threshold."""
        records = [
            _mode_record("none", comp=0.0, stab=0.0),
            _mode_record("d4", comp=0.5, stab=0.6),
            _mode_record("square", comp=0.3, stab=0.3),
            _mode_record("inv", comp=0.2, stab=0.1),
            _mode_record("d4+inv", comp=0.4, stab=0.5),
        ]
        assert infer_system_class(records) == "chain_like"

    def test_cycle_like(self):
        """d4 stability is weak, inv stability is stronger."""
        records = [
            _mode_record("none", comp=0.0, stab=0.0),
            _mode_record("d4", comp=0.1, stab=0.1),
            _mode_record("square", comp=0.1, stab=0.05),
            _mode_record("inv", comp=0.3, stab=0.3),
            _mode_record("d4+inv", comp=0.4, stab=0.5),
        ]
        assert infer_system_class(records) == "cycle_like"

    def test_branching_like(self):
        """inv compression dominates d4 compression."""
        records = [
            _mode_record("none", comp=0.0, stab=0.0),
            _mode_record("d4", comp=0.2, stab=0.3),
            _mode_record("square", comp=0.3, stab=0.35),
            _mode_record("inv", comp=0.6, stab=0.3),
            _mode_record("d4+inv", comp=0.5, stab=0.4),
        ]
        assert infer_system_class(records) == "branching_like"

    def test_basin_like(self):
        """Strong correction across multiple modes."""
        records = [
            _mode_record("none", comp=0.0, stab=0.0),
            _mode_record("d4", comp=0.4, stab=0.35),
            _mode_record("square", comp=0.5, stab=0.55),
            _mode_record("inv", comp=0.3, stab=0.35),
            _mode_record("d4+inv", comp=0.5, stab=0.5),
        ]
        assert infer_system_class(records) == "basin_like"

    def test_unknown_fallback(self):
        """No rule matches -> unknown."""
        records = [
            _mode_record("none", comp=0.0, stab=0.0),
            _mode_record("d4", comp=0.1, stab=0.25),
            _mode_record("square", comp=0.3, stab=0.3),
        ]
        assert infer_system_class(records) == "unknown"

    def test_empty_records(self):
        assert infer_system_class([]) == "unknown"

    def test_deterministic_across_calls(self):
        records = [
            _mode_record("d4", comp=0.5, stab=0.6),
            _mode_record("square", comp=0.3, stab=0.3),
        ]
        r1 = infer_system_class(records)
        r2 = infer_system_class(records)
        assert r1 == r2

    def test_no_mutation_of_input(self):
        records = [
            _mode_record("d4", comp=0.5, stab=0.6),
            _mode_record("square", comp=0.3, stab=0.3),
        ]
        original = copy.deepcopy(records)
        infer_system_class(records)
        assert records == original


# ---------------------------------------------------------------------------
# PART 3 — TAXONOMY ISSUE DETECTION TESTS
# ---------------------------------------------------------------------------


class TestDetectTaxonomyIssues:
    """Tests for taxonomy-aware issue detection."""

    def test_cycle_mismatch(self):
        issues = detect_taxonomy_issues("cycle_like", "d4", [])
        assert "cycle_mismatch" in issues

    def test_cycle_no_mismatch(self):
        issues = detect_taxonomy_issues("cycle_like", "d4+inv", [])
        assert "cycle_mismatch" not in issues

    def test_chain_underperformance(self):
        issues = detect_taxonomy_issues("chain_like", "square", [])
        assert "chain_underperformance" in issues

    def test_chain_no_underperformance(self):
        issues = detect_taxonomy_issues("chain_like", "d4", [])
        assert "chain_underperformance" not in issues

    def test_missing_invariant_guidance(self):
        issues = detect_taxonomy_issues("branching_like", "d4", [])
        assert "missing_invariant_guidance" in issues

    def test_branching_with_inv(self):
        issues = detect_taxonomy_issues("branching_like", "d4+inv", [])
        assert "missing_invariant_guidance" not in issues

    def test_overprocessing_simple_system(self):
        issues = detect_taxonomy_issues("degenerate", "d4", [])
        assert "overprocessing_simple_system" in issues

    def test_degenerate_none_mode_ok(self):
        issues = detect_taxonomy_issues("degenerate", "none", [])
        assert "overprocessing_simple_system" not in issues

    def test_unknown_class_no_issues(self):
        issues = detect_taxonomy_issues("unknown", "d4", [])
        assert issues == []

    def test_deterministic(self):
        r1 = detect_taxonomy_issues("cycle_like", "d4", [])
        r2 = detect_taxonomy_issues("cycle_like", "d4", [])
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART 4 — TAXONOMY RECOMMENDATION TESTS
# ---------------------------------------------------------------------------


class TestTaxonomyRecommendations:
    """Tests that taxonomy issues map to correct recommendations."""

    def test_cycle_mismatch_recommendation(self):
        assert recommend(["cycle_mismatch"]) == [
            "prefer_invariant_guided_modes"
        ]

    def test_chain_underperformance_recommendation(self):
        assert recommend(["chain_underperformance"]) == [
            "prefer_d4_projection"
        ]

    def test_missing_invariant_recommendation(self):
        assert recommend(["missing_invariant_guidance"]) == [
            "enable_invariant_guidance"
        ]

    def test_overprocessing_recommendation(self):
        assert recommend(["overprocessing_simple_system"]) == [
            "disable_correction"
        ]


# ---------------------------------------------------------------------------
# PART 5 — INTEGRATION TESTS
# ---------------------------------------------------------------------------


def _chain_system_raw():
    """Raw results for a chain-like system."""
    return [
        _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 6, 6),
        _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 6, 3),
        _make_raw_result("chain", 5, "square", 0.3, 0.3, 1, 6, 4),
        _make_raw_result("chain", 5, "inv", 0.2, 0.1, 0, 6, 5),
        _make_raw_result("chain", 5, "d4+inv", 0.4, 0.5, 2, 6, 3),
    ]


def _cycle_system_raw():
    """Raw results for a cycle-like system."""
    return [
        _make_raw_result("cycle", 10, "none", 0.0, 0.0, 0, 8, 8),
        _make_raw_result("cycle", 10, "d4", 0.1, 0.1, 0, 8, 7),
        _make_raw_result("cycle", 10, "square", 0.1, 0.05, 0, 8, 7),
        _make_raw_result("cycle", 10, "inv", 0.3, 0.3, 1, 8, 6),
        _make_raw_result("cycle", 10, "d4+inv", 0.4, 0.5, 3, 8, 5),
    ]


class TestPipelineIntegration:
    """Integration tests: full pipeline with taxonomy layer."""

    def test_system_classes_in_output(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        assert "system_classes" in report

    def test_chain_classified_correctly(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        classes = {
            (sc["dfa_type"], sc["n"]): sc["system_class"]
            for sc in report["system_classes"]
        }
        assert classes[("chain", 5)] == "chain_like"

    def test_cycle_classified_correctly(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        classes = {
            (sc["dfa_type"], sc["n"]): sc["system_class"]
            for sc in report["system_classes"]
        }
        assert classes[("cycle", 10)] == "cycle_like"

    def test_v93_keys_still_present(self):
        raw = _chain_system_raw()
        report = run_self_diagnostics(raw)
        assert "metrics" in report
        assert "best_modes" in report
        assert "issues" in report
        assert "recommendations" in report

    def test_deterministic_across_runs(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        r1 = run_self_diagnostics(raw)
        r2 = run_self_diagnostics(raw)
        assert r1 == r2

    def test_no_mutation(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        original = copy.deepcopy(raw)
        run_self_diagnostics(raw)
        assert raw == original

    def test_issues_include_taxonomy(self):
        """Cycle system with best_mode=d4+inv should not have cycle_mismatch.
        Chain system with best_mode=d4 should not have chain_underperformance.
        """
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        all_issues = []
        for ir in report["issues"]:
            all_issues.extend(ir["issues"])
        # Neither system should trigger taxonomy issues when best modes match.
        assert "cycle_mismatch" not in all_issues
        assert "chain_underperformance" not in all_issues

    def test_recommendations_match_issues_count(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        for iss, rec in zip(report["issues"], report["recommendations"]):
            assert len(iss["issues"]) == len(rec["recommendations"])


class TestPrintDiagnosticsWithTaxonomy:
    """Tests for print_diagnostics with taxonomy info."""

    def test_contains_class_label(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        output = print_diagnostics(report)
        assert "class: chain_like" in output
        assert "class: cycle_like" in output

    def test_deterministic_output(self):
        raw = _chain_system_raw() + _cycle_system_raw()
        report = run_self_diagnostics(raw)
        o1 = print_diagnostics(report)
        o2 = print_diagnostics(report)
        assert o1 == o2


class TestClassifyAllSystems:
    """Tests for classify_all_systems."""

    def test_returns_dict(self):
        from qec.analysis.self_diagnostics import (
            aggregate_metrics,
            normalize_results,
        )
        raw = _chain_system_raw() + _cycle_system_raw()
        records = normalize_results(raw)
        agg = aggregate_metrics(records)
        result = classify_all_systems(agg)
        assert isinstance(result, dict)
        assert ("chain", 5) in result
        assert ("cycle", 10) in result

    def test_deterministic(self):
        from qec.analysis.self_diagnostics import (
            aggregate_metrics,
            normalize_results,
        )
        raw = _chain_system_raw() + _cycle_system_raw()
        records = normalize_results(raw)
        agg = aggregate_metrics(records)
        r1 = classify_all_systems(agg)
        r2 = classify_all_systems(agg)
        assert r1 == r2
