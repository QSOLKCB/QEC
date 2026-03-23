"""Tests for self-diagnostics and recommendation layer (v93.0.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.self_diagnostics import (
    aggregate_metrics,
    best_mode_per_system,
    detect_all,
    detect_issues,
    normalize_results,
    print_diagnostics,
    recommend,
    recommend_all,
    run_self_diagnostics,
)


# ---------------------------------------------------------------------------
# FIXTURES — synthetic benchmark data
# ---------------------------------------------------------------------------


def _make_raw_result(
    dfa_name="chain",
    n=5,
    mode="none",
    comp=0.0,
    stab=0.0,
    gain=0,
    ub=4,
    ua=4,
):
    """Build a single run_suite()-style result dict."""
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


def _sample_raw_results():
    """Small set of raw results covering two systems and two modes."""
    return [
        _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
        _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 4, 2),
        _make_raw_result("cycle", 10, "none", 0.0, 0.0, 0, 6, 6),
        _make_raw_result("cycle", 10, "d4+inv", 0.4, 0.5, 3, 6, 4),
    ]


def _sample_summary():
    """Small summarize()-style output."""
    return {
        ("chain", 5): {
            "none": {
                "compression_efficiency": 0.0,
                "stability_efficiency": 0.0,
            },
            "d4": {
                "compression_efficiency": 0.5,
                "stability_efficiency": 0.6,
            },
        },
        ("cycle", 10): {
            "none": {
                "compression_efficiency": 0.0,
                "stability_efficiency": 0.0,
            },
            "d4+inv": {
                "compression_efficiency": 0.4,
                "stability_efficiency": 0.5,
            },
        },
    }


# ---------------------------------------------------------------------------
# PART 1 — NORMALIZATION TESTS
# ---------------------------------------------------------------------------


class TestNormalizeRawResults:
    """Tests for normalizing run_suite() output."""

    def test_basic_structure(self):
        raw = _sample_raw_results()
        records = normalize_results(raw)
        assert isinstance(records, list)
        assert len(records) == 4

    def test_flat_keys(self):
        raw = _sample_raw_results()
        records = normalize_results(raw)
        expected_keys = {
            "dfa_type", "n", "mode",
            "compression_efficiency", "stability_efficiency",
            "stability_gain", "unique_before", "unique_after",
        }
        for r in records:
            assert set(r.keys()) == expected_keys

    def test_sorted_output(self):
        raw = _sample_raw_results()
        records = normalize_results(raw)
        keys = [(r["dfa_type"], str(r["n"]), r["mode"]) for r in records]
        assert keys == sorted(keys)

    def test_no_mutation(self):
        raw = _sample_raw_results()
        original = copy.deepcopy(raw)
        normalize_results(raw)
        assert raw == original


class TestNormalizeSummary:
    """Tests for normalizing summarize() output."""

    def test_basic_structure(self):
        summary = _sample_summary()
        records = normalize_results(summary)
        assert isinstance(records, list)
        assert len(records) == 4

    def test_defaults_for_missing_fields(self):
        summary = _sample_summary()
        records = normalize_results(summary)
        for r in records:
            assert r["stability_gain"] == 0
            assert r["unique_before"] == 0
            assert r["unique_after"] == 0

    def test_sorted_output(self):
        summary = _sample_summary()
        records = normalize_results(summary)
        keys = [(r["dfa_type"], str(r["n"]), r["mode"]) for r in records]
        assert keys == sorted(keys)

    def test_no_mutation(self):
        summary = _sample_summary()
        original = copy.deepcopy(summary)
        normalize_results(summary)
        assert summary == original


class TestNormalizeInvalidInput:
    """Test that invalid input raises TypeError."""

    def test_string_raises(self):
        with pytest.raises(TypeError):
            normalize_results("bad input")

    def test_int_raises(self):
        with pytest.raises(TypeError):
            normalize_results(42)


# ---------------------------------------------------------------------------
# PART 2 — AGGREGATION TESTS
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    """Tests for metric aggregation."""

    def test_single_record_per_group(self):
        records = normalize_results(_sample_raw_results())
        agg = aggregate_metrics(records)
        assert len(agg) == 4

    def test_duplicate_averaging(self):
        records = [
            {
                "dfa_type": "chain", "n": 5, "mode": "d4",
                "compression_efficiency": 0.4,
                "stability_efficiency": 0.6,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
            {
                "dfa_type": "chain", "n": 5, "mode": "d4",
                "compression_efficiency": 0.6,
                "stability_efficiency": 0.8,
                "stability_gain": 4, "unique_before": 4, "unique_after": 2,
            },
        ]
        agg = aggregate_metrics(records)
        assert len(agg) == 1
        assert abs(agg[0]["compression_efficiency"] - 0.5) < 1e-10
        assert abs(agg[0]["stability_efficiency"] - 0.7) < 1e-10
        assert abs(agg[0]["stability_gain"] - 3.0) < 1e-10

    def test_deterministic_ordering(self):
        records = normalize_results(_sample_raw_results())
        agg = aggregate_metrics(records)
        keys = [(r["dfa_type"], str(r["n"]), r["mode"]) for r in agg]
        assert keys == sorted(keys)


class TestBestModePerSystem:
    """Tests for best mode selection."""

    def test_selects_best(self):
        records = normalize_results(_sample_raw_results())
        agg = aggregate_metrics(records)
        best = best_mode_per_system(agg)
        assert len(best) == 2  # chain, cycle
        chain_best = [b for b in best if b["dfa_type"] == "chain"][0]
        assert chain_best["best_mode"] == "d4"

    def test_tiebreak_by_compression(self):
        records = [
            {
                "dfa_type": "chain", "n": 5, "mode": "d4",
                "compression_efficiency": 0.6,
                "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
            {
                "dfa_type": "chain", "n": 5, "mode": "square",
                "compression_efficiency": 0.3,
                "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
        ]
        agg = aggregate_metrics(records)
        best = best_mode_per_system(agg)
        assert best[0]["best_mode"] == "d4"

    def test_tiebreak_by_name(self):
        records = [
            {
                "dfa_type": "chain", "n": 5, "mode": "d4",
                "compression_efficiency": 0.5,
                "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
            {
                "dfa_type": "chain", "n": 5, "mode": "square",
                "compression_efficiency": 0.5,
                "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
        ]
        agg = aggregate_metrics(records)
        best = best_mode_per_system(agg)
        # d4 < square lexicographically, so d4 wins
        assert best[0]["best_mode"] == "d4"

    def test_stable_across_runs(self):
        records = normalize_results(_sample_raw_results())
        agg = aggregate_metrics(records)
        best1 = best_mode_per_system(agg)
        best2 = best_mode_per_system(agg)
        assert best1 == best2


# ---------------------------------------------------------------------------
# PART 3 — ISSUE DETECTION TESTS
# ---------------------------------------------------------------------------


class TestDetectIssues:
    """Tests for individual issue detection rules."""

    def _best_map(self, stab=0.5):
        return {("chain", 5): {"stability_efficiency": stab,
                               "compression_efficiency": 0.5}}

    def test_no_effect(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "none",
            "compression_efficiency": 0.01,
            "stability_efficiency": 0.01,
            "stability_gain": 0, "unique_before": 4, "unique_after": 4,
        }
        result = detect_issues(r, self._best_map())
        assert "no_effect" in result["issues"]

    def test_over_correction(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "square",
            "compression_efficiency": 0.8,
            "stability_efficiency": 0.05,
            "stability_gain": 0, "unique_before": 10, "unique_after": 2,
        }
        result = detect_issues(r, self._best_map())
        assert "over_correction" in result["issues"]

    def test_destabilizing(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "square",
            "compression_efficiency": 0.3,
            "stability_efficiency": 0.3,
            "stability_gain": -2, "unique_before": 4, "unique_after": 3,
        }
        result = detect_issues(r, self._best_map())
        assert "destabilizing" in result["issues"]

    def test_low_diversity(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "none",
            "compression_efficiency": 0.0,
            "stability_efficiency": 0.0,
            "stability_gain": 0, "unique_before": 2, "unique_after": 2,
        }
        result = detect_issues(r, self._best_map())
        assert "low_diversity" in result["issues"]

    def test_globally_weak_correction(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "d4",
            "compression_efficiency": 0.3,
            "stability_efficiency": 0.15,
            "stability_gain": 1, "unique_before": 4, "unique_after": 3,
        }
        result = detect_issues(r, self._best_map(stab=0.15))
        assert "globally_weak_correction" in result["issues"]

    def test_no_issues_clean_record(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "d4",
            "compression_efficiency": 0.5,
            "stability_efficiency": 0.6,
            "stability_gain": 2, "unique_before": 6, "unique_after": 3,
        }
        result = detect_issues(r, self._best_map(stab=0.6))
        assert result["issues"] == []

    def test_multiple_issues(self):
        r = {
            "dfa_type": "chain", "n": 5, "mode": "none",
            "compression_efficiency": 0.01,
            "stability_efficiency": 0.01,
            "stability_gain": -1, "unique_before": 1, "unique_after": 1,
        }
        result = detect_issues(r, self._best_map(stab=0.1))
        assert "no_effect" in result["issues"]
        assert "destabilizing" in result["issues"]
        assert "low_diversity" in result["issues"]
        assert "globally_weak_correction" in result["issues"]


# ---------------------------------------------------------------------------
# PART 4 — RECOMMENDATION TESTS
# ---------------------------------------------------------------------------


class TestRecommend:
    """Tests for issue-to-recommendation mapping."""

    def test_no_effect_mapping(self):
        assert recommend(["no_effect"]) == ["increase_structure_guidance"]

    def test_over_correction_mapping(self):
        assert recommend(["over_correction"]) == [
            "reduce_projection_strength"
        ]

    def test_destabilizing_mapping(self):
        assert recommend(["destabilizing"]) == ["switch_to_d4_or_invariant"]

    def test_low_diversity_mapping(self):
        assert recommend(["low_diversity"]) == [
            "skip_correction_or_expand_input"
        ]

    def test_globally_weak_mapping(self):
        assert recommend(["globally_weak_correction"]) == [
            "enable_invariant_guidance"
        ]

    def test_empty_issues(self):
        assert recommend([]) == []

    def test_multiple_issues_ordered(self):
        result = recommend(["no_effect", "destabilizing"])
        assert result == [
            "increase_structure_guidance",
            "switch_to_d4_or_invariant",
        ]


class TestRecommendAll:
    """Tests for batch recommendation generation."""

    def test_basic(self):
        issue_records = [
            {"dfa_type": "chain", "n": 5, "mode": "none",
             "issues": ["no_effect"]},
            {"dfa_type": "chain", "n": 5, "mode": "d4",
             "issues": []},
        ]
        recs = recommend_all(issue_records)
        assert len(recs) == 2
        assert recs[0]["recommendations"] == [
            "increase_structure_guidance"
        ]
        assert recs[1]["recommendations"] == []


# ---------------------------------------------------------------------------
# PART 5 — FULL PIPELINE TESTS
# ---------------------------------------------------------------------------


class TestRunSelfDiagnostics:
    """Tests for the full pipeline."""

    def test_raw_input(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        assert "metrics" in report
        assert "best_modes" in report
        assert "issues" in report
        assert "recommendations" in report

    def test_summary_input(self):
        summary = _sample_summary()
        report = run_self_diagnostics(summary)
        assert "metrics" in report
        assert len(report["metrics"]) == 4

    def test_deterministic_across_runs(self):
        raw = _sample_raw_results()
        r1 = run_self_diagnostics(raw)
        r2 = run_self_diagnostics(raw)
        assert r1 == r2

    def test_no_mutation_raw(self):
        raw = _sample_raw_results()
        original = copy.deepcopy(raw)
        run_self_diagnostics(raw)
        assert raw == original

    def test_no_mutation_summary(self):
        summary = _sample_summary()
        original = copy.deepcopy(summary)
        run_self_diagnostics(summary)
        assert summary == original

    def test_issues_match_recommendations_count(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        for iss, rec in zip(report["issues"], report["recommendations"]):
            assert len(iss["issues"]) == len(rec["recommendations"])


# ---------------------------------------------------------------------------
# PART 6 — PRINT DIAGNOSTICS TESTS
# ---------------------------------------------------------------------------


class TestPrintDiagnostics:
    """Tests for print_diagnostics formatting."""

    def test_produces_string(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        output = print_diagnostics(report)
        assert isinstance(output, str)

    def test_contains_dfa_names(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        output = print_diagnostics(report)
        assert "chain" in output
        assert "cycle" in output

    def test_deterministic_output(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        o1 = print_diagnostics(report)
        o2 = print_diagnostics(report)
        assert o1 == o2

    def test_contains_best_mode(self):
        raw = _sample_raw_results()
        report = run_self_diagnostics(raw)
        output = print_diagnostics(report)
        assert "best_mode:" in output
