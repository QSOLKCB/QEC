"""Tests for law stability scoring — v105.1.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.invariant_registry import (
    ROUND_PRECISION,
    compute_law_stability,
    detect_emergent_laws,
    format_law_stability_summary,
    init_registry,
    run_incremental_invariant_analysis,
    update_registry_incremental,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_invariant(
    name: str = "stability_monotonicity",
    inv_type: str = "geometry",
    strength: float = 1.0,
) -> dict:
    return {
        "name": name,
        "type": inv_type,
        "description": f"{name} invariant",
        "strength": strength,
        "support": int(strength * 2),
        "total": 2,
    }


def _make_run_result(scored_invariants: list | None = None) -> dict:
    if scored_invariants is None:
        scored_invariants = [
            _make_invariant("stability_monotonicity", "geometry", 1.0),
            _make_invariant("basin_switch_suppression", "sign", 0.8),
        ]
    return {
        "scored_invariants": scored_invariants,
        "global_metrics": {
            "primary_diagnosis": "oscillatory_trap",
            "topology_type": "mixed",
        },
    }


# ---------------------------------------------------------------------------
# Tests: compute_law_stability
# ---------------------------------------------------------------------------


class TestComputeLawStability:
    def test_perfect_entry(self):
        entry = {
            "count": 10,
            "streak": 10,
            "break_count": 0,
            "avg_strength": 1.0,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
        }
        score = compute_law_stability(entry, total_runs=10)
        assert score == 1.0

    def test_zero_entry(self):
        entry = {
            "count": 1,
            "streak": 0,
            "break_count": 1,
            "avg_strength": 0.0,
            "max_strength": 0.0,
            "first_seen": 0,
            "last_seen": 1,
        }
        score = compute_law_stability(entry, total_runs=10)
        assert 0.0 <= score <= 1.0

    def test_score_in_range(self):
        for count in [1, 3, 5, 10]:
            for streak in range(0, count + 1):
                for brk in range(0, count + 1):
                    entry = {
                        "count": count,
                        "streak": streak,
                        "break_count": brk,
                        "avg_strength": 0.7,
                        "max_strength": 1.0,
                        "first_seen": 0,
                        "last_seen": count - 1,
                    }
                    score = compute_law_stability(entry, total_runs=10)
                    assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        entry = {
            "count": 5,
            "streak": 3,
            "break_count": 1,
            "avg_strength": 0.8,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 5,
        }
        s1 = compute_law_stability(entry, total_runs=10)
        s2 = compute_law_stability(entry, total_runs=10)
        assert s1 == s2

    def test_higher_streak_means_higher_stability(self):
        base = {
            "count": 10,
            "break_count": 0,
            "avg_strength": 0.8,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
        }
        low = dict(base, streak=2)
        high = dict(base, streak=10)
        assert compute_law_stability(low, 10) < compute_law_stability(high, 10)

    def test_more_breaks_means_lower_stability(self):
        base = {
            "count": 10,
            "streak": 5,
            "avg_strength": 0.8,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
        }
        no_break = dict(base, break_count=0)
        many_break = dict(base, break_count=8)
        assert compute_law_stability(no_break, 10) > compute_law_stability(many_break, 10)

    def test_auto_derives_total_runs(self):
        entry = {
            "count": 5,
            "streak": 5,
            "break_count": 0,
            "avg_strength": 0.8,
            "max_strength": 1.0,
            "first_seen": 0,
            "last_seen": 9,
        }
        # total_runs=0 => derived from span.
        score = compute_law_stability(entry, total_runs=0)
        assert 0.0 <= score <= 1.0

    def test_precision(self):
        entry = {
            "count": 3,
            "streak": 2,
            "break_count": 1,
            "avg_strength": 0.777,
            "max_strength": 0.9,
            "first_seen": 0,
            "last_seen": 5,
        }
        score = compute_law_stability(entry, total_runs=6)
        # Check rounding precision.
        assert score == round(score, ROUND_PRECISION)


# ---------------------------------------------------------------------------
# Tests: detect_emergent_laws with stability
# ---------------------------------------------------------------------------


class TestEmergentLawsWithStability:
    def test_laws_include_stability_score(self):
        reg = {
            "sign:test": {
                "count": 5,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
                "streak": 5,
                "break_count": 0,
                "last_observed": True,
            }
        }
        laws = detect_emergent_laws(reg)
        assert len(laws) >= 1
        assert "stability_score" in laws[0]
        assert "classification" in laws[0]

    def test_low_stability_filtered_out(self):
        reg = {
            "sign:weak": {
                "count": 3,
                "avg_strength": 0.65,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 20,
                "streak": 0,
                "break_count": 15,
                "last_observed": False,
            }
        }
        laws = detect_emergent_laws(reg)
        # Should be filtered out due to low stability.
        assert len(laws) == 0

    def test_deterministic(self):
        reg = {
            "sign:test": {
                "count": 5,
                "avg_strength": 0.9,
                "max_strength": 1.0,
                "first_seen": 0,
                "last_seen": 4,
                "streak": 5,
                "break_count": 0,
                "last_observed": True,
            }
        }
        assert detect_emergent_laws(reg) == detect_emergent_laws(reg)


# ---------------------------------------------------------------------------
# Tests: run_incremental_invariant_analysis
# ---------------------------------------------------------------------------


class TestRunIncrementalInvariantAnalysis:
    def test_returns_required_keys(self):
        results = [_make_run_result() for _ in range(3)]
        result = run_incremental_invariant_analysis(results)
        assert "registry" in result
        assert "emergent_laws" in result
        assert "top_invariants" in result
        assert "drift" in result
        assert "law_summary" in result

    def test_law_summary_structure(self):
        results = [_make_run_result() for _ in range(5)]
        result = run_incremental_invariant_analysis(results)
        summary = result["law_summary"]
        assert "law_stability_score" in summary
        assert "stable_law_count" in summary
        assert "emerging_law_count" in summary
        assert "drifting_invariant_count" in summary
        assert "total_invariants" in summary
        assert "total_runs" in summary
        assert summary["total_runs"] == 5

    def test_top_invariants_include_lifecycle(self):
        results = [_make_run_result() for _ in range(3)]
        result = run_incremental_invariant_analysis(results)
        for item in result["top_invariants"]:
            assert "stability_score" in item
            assert "classification" in item
            assert "lifecycle" in item

    def test_deterministic(self):
        results = [_make_run_result() for _ in range(3)]
        r1 = run_incremental_invariant_analysis(results)
        r2 = run_incremental_invariant_analysis(results)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        results = [_make_run_result() for _ in range(3)]
        orig = copy.deepcopy(results)
        run_incremental_invariant_analysis(results)
        assert results == orig

    def test_empty_input(self):
        result = run_incremental_invariant_analysis([])
        assert result["registry"] == {}
        assert result["emergent_laws"] == []
        assert result["top_invariants"] == []
        assert result["law_summary"]["total_runs"] == 0


# ---------------------------------------------------------------------------
# Tests: format_law_stability_summary
# ---------------------------------------------------------------------------


class TestFormatLawStabilitySummary:
    def test_contains_header(self):
        results = [_make_run_result() for _ in range(5)]
        analysis = run_incremental_invariant_analysis(results)
        text = format_law_stability_summary(analysis)
        assert "Law Stability Summary" in text

    def test_contains_statistics(self):
        results = [_make_run_result() for _ in range(5)]
        analysis = run_incremental_invariant_analysis(results)
        text = format_law_stability_summary(analysis)
        assert "Total invariants" in text
        assert "Total runs" in text
        assert "Avg stability score" in text

    def test_empty_analysis(self):
        analysis = run_incremental_invariant_analysis([])
        text = format_law_stability_summary(analysis)
        assert "Law Stability Summary" in text
        assert "Total invariants: 0" in text
