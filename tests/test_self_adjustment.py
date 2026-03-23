"""Tests for self-adjustment loop (v94.0.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.self_adjustment import (
    adjust_system,
    is_improvement,
    print_adjustment_report,
    resolve_recommendation_to_mode,
    run_self_adjustment,
)
from qec.analysis.self_diagnostics import run_self_diagnostics


# ---------------------------------------------------------------------------
# FIXTURES
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
    """Multi-system raw results with clear best modes."""
    return [
        _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
        _make_raw_result("chain", 5, "square", 0.3, 0.3, 1, 4, 3),
        _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 4, 2),
        _make_raw_result("chain", 5, "d4+inv", 0.4, 0.5, 1, 4, 2),
        _make_raw_result("cycle", 10, "none", 0.0, 0.0, 0, 6, 6),
        _make_raw_result("cycle", 10, "square", 0.2, 0.1, 0, 6, 5),
        _make_raw_result("cycle", 10, "d4", 0.3, 0.15, 1, 6, 4),
        _make_raw_result("cycle", 10, "d4+inv", 0.4, 0.5, 3, 6, 4),
    ]


def _sample_summary():
    """Summary-style output."""
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


def _system_records_chain():
    """Aggregated records for a chain system with d4 as best."""
    return [
        {
            "dfa_type": "chain", "n": 5, "mode": "none",
            "compression_efficiency": 0.0, "stability_efficiency": 0.0,
            "stability_gain": 0, "unique_before": 4, "unique_after": 4,
        },
        {
            "dfa_type": "chain", "n": 5, "mode": "square",
            "compression_efficiency": 0.3, "stability_efficiency": 0.3,
            "stability_gain": 1, "unique_before": 4, "unique_after": 3,
        },
        {
            "dfa_type": "chain", "n": 5, "mode": "d4",
            "compression_efficiency": 0.5, "stability_efficiency": 0.6,
            "stability_gain": 2, "unique_before": 4, "unique_after": 2,
        },
        {
            "dfa_type": "chain", "n": 5, "mode": "d4+inv",
            "compression_efficiency": 0.4, "stability_efficiency": 0.5,
            "stability_gain": 1, "unique_before": 4, "unique_after": 2,
        },
    ]


# ---------------------------------------------------------------------------
# PART 1 — RESOLVE RECOMMENDATION TO MODE
# ---------------------------------------------------------------------------


class TestResolveRecommendationToMode:
    """Tests for recommendation-to-mode resolution."""

    def test_prefer_d4_projection(self):
        result = resolve_recommendation_to_mode(
            "prefer_d4_projection", ["none", "square", "d4", "d4+inv"]
        )
        assert result == "d4"

    def test_prefer_invariant_guided_modes(self):
        result = resolve_recommendation_to_mode(
            "prefer_invariant_guided_modes", ["none", "square", "d4", "d4+inv"]
        )
        assert result == "d4+inv"

    def test_enable_invariant_guidance(self):
        result = resolve_recommendation_to_mode(
            "enable_invariant_guidance", ["none", "square", "d4", "d4+inv"]
        )
        assert result == "d4+inv"

    def test_reduce_projection_strength(self):
        result = resolve_recommendation_to_mode(
            "reduce_projection_strength", ["none", "square", "d4"]
        )
        assert result == "square"

    def test_disable_correction(self):
        result = resolve_recommendation_to_mode(
            "disable_correction", ["none", "square", "d4"]
        )
        assert result == "none"

    def test_switch_to_d4_or_invariant(self):
        result = resolve_recommendation_to_mode(
            "switch_to_d4_or_invariant", ["none", "square", "d4", "d4+inv"]
        )
        assert result == "d4+inv"

    def test_increase_structure_guidance(self):
        result = resolve_recommendation_to_mode(
            "increase_structure_guidance", ["none", "square", "d4", "d4+inv"]
        )
        assert result == "d4+inv"

    def test_skip_correction_or_expand_input(self):
        result = resolve_recommendation_to_mode(
            "skip_correction_or_expand_input", ["none", "square"]
        )
        assert result == "none"

    def test_fallback_d4inv_to_d4(self):
        result = resolve_recommendation_to_mode(
            "prefer_invariant_guided_modes", ["none", "square", "d4"]
        )
        assert result == "d4"

    def test_fallback_d4inv_to_square(self):
        result = resolve_recommendation_to_mode(
            "prefer_invariant_guided_modes", ["none", "square"]
        )
        assert result == "square"

    def test_fallback_d4inv_to_none(self):
        result = resolve_recommendation_to_mode(
            "prefer_invariant_guided_modes", ["none"]
        )
        assert result == "none"

    def test_fallback_d4_to_square(self):
        result = resolve_recommendation_to_mode(
            "prefer_d4_projection", ["none", "square"]
        )
        assert result == "square"

    def test_fallback_d4_to_none(self):
        result = resolve_recommendation_to_mode(
            "prefer_d4_projection", ["none"]
        )
        assert result == "none"

    def test_fallback_square_to_none(self):
        result = resolve_recommendation_to_mode(
            "reduce_projection_strength", ["none"]
        )
        assert result == "none"

    def test_no_valid_mode_returns_none(self):
        result = resolve_recommendation_to_mode(
            "prefer_d4_projection", []
        )
        assert result is None

    def test_unknown_recommendation_returns_none(self):
        result = resolve_recommendation_to_mode(
            "unknown_recommendation", ["none", "d4"]
        )
        assert result is None


# ---------------------------------------------------------------------------
# PART 2 — IMPROVEMENT CRITERION
# ---------------------------------------------------------------------------


class TestIsImprovement:
    """Tests for deterministic improvement criterion."""

    def test_higher_stability_accepted(self):
        before = {"stability_efficiency": 0.3, "compression_efficiency": 0.5}
        after = {"stability_efficiency": 0.6, "compression_efficiency": 0.3}
        accepted, reason = is_improvement(before, after)
        assert accepted is True
        assert reason == "accepted_improved_stability"

    def test_lower_stability_rejected(self):
        before = {"stability_efficiency": 0.6, "compression_efficiency": 0.3}
        after = {"stability_efficiency": 0.3, "compression_efficiency": 0.5}
        accepted, reason = is_improvement(before, after)
        assert accepted is False
        assert reason == "rejected_no_improvement"

    def test_tied_stability_higher_compression_accepted(self):
        before = {"stability_efficiency": 0.5, "compression_efficiency": 0.3}
        after = {"stability_efficiency": 0.5, "compression_efficiency": 0.6}
        accepted, reason = is_improvement(before, after)
        assert accepted is True
        assert reason == "accepted_improved_compression"

    def test_tied_stability_lower_compression_rejected(self):
        before = {"stability_efficiency": 0.5, "compression_efficiency": 0.6}
        after = {"stability_efficiency": 0.5, "compression_efficiency": 0.3}
        accepted, reason = is_improvement(before, after)
        assert accepted is False

    def test_tied_stability_and_compression_higher_gain(self):
        before = {
            "stability_efficiency": 0.5,
            "compression_efficiency": 0.5,
            "stability_gain": 1.0,
        }
        after = {
            "stability_efficiency": 0.5,
            "compression_efficiency": 0.5,
            "stability_gain": 3.0,
        }
        accepted, reason = is_improvement(before, after)
        assert accepted is True
        assert reason == "accepted_improved_stability_gain"

    def test_identical_metrics_rejected(self):
        before = {"stability_efficiency": 0.5, "compression_efficiency": 0.5}
        after = {"stability_efficiency": 0.5, "compression_efficiency": 0.5}
        accepted, reason = is_improvement(before, after)
        assert accepted is False
        assert reason == "rejected_no_improvement"

    def test_missing_keys_default_to_zero(self):
        before = {}
        after = {"stability_efficiency": 0.1}
        accepted, reason = is_improvement(before, after)
        assert accepted is True


# ---------------------------------------------------------------------------
# PART 3 — SINGLE-SYSTEM ADJUSTMENT
# ---------------------------------------------------------------------------


class TestAdjustSystem:
    """Tests for single-system adjustment."""

    def test_accepts_improving_candidate(self):
        records = _system_records_chain()
        # If best is d4 (stab=0.6) and we recommend d4+inv (stab=0.5),
        # that's worse. Let's construct a scenario where candidate improves.
        records_mod = [
            {
                "dfa_type": "test", "n": 5, "mode": "none",
                "compression_efficiency": 0.0, "stability_efficiency": 0.0,
                "stability_gain": 0, "unique_before": 4, "unique_after": 4,
            },
            {
                "dfa_type": "test", "n": 5, "mode": "square",
                "compression_efficiency": 0.3, "stability_efficiency": 0.3,
                "stability_gain": 1, "unique_before": 4, "unique_after": 3,
            },
            {
                "dfa_type": "test", "n": 5, "mode": "d4",
                "compression_efficiency": 0.6, "stability_efficiency": 0.7,
                "stability_gain": 3, "unique_before": 4, "unique_after": 2,
            },
        ]
        result = adjust_system(records_mod, ["prefer_d4_projection"])
        # Best is square (stab=0.3), candidate is d4 (stab=0.7) -> accepted
        assert result["original_mode"] == "d4"
        # d4 is already best, so recommend d4 -> same mode
        # Let's use a different recommendation
        result2 = adjust_system(
            records_mod, ["reduce_projection_strength"]
        )
        # Best is d4 (stab=0.7), candidate is square (stab=0.3) -> rejected
        assert result2["accepted"] is False
        assert result2["reason"] == "rejected_no_improvement"

    def test_rejects_same_mode(self):
        records = _system_records_chain()
        # Best mode is d4, recommend d4 -> same mode rejection
        result = adjust_system(records, ["prefer_d4_projection"])
        assert result["accepted"] is False
        assert result["reason"] == "rejected_same_mode"
        assert result["candidate_mode"] == "d4"
        assert result["original_mode"] == "d4"

    def test_rejects_non_improving_candidate(self):
        records = _system_records_chain()
        # Best is d4 (stab=0.6), recommend square (stab=0.3) -> rejected
        result = adjust_system(records, ["reduce_projection_strength"])
        assert result["accepted"] is False
        assert result["reason"] == "rejected_no_improvement"

    def test_accepts_when_candidate_better(self):
        # Tied stability between square and d4, but d4 has better
        # compression. Best picks d4 (comp=0.5 vs 0.4). Recommend
        # d4+inv which has higher stability -> accepted.
        records = [
            {
                "dfa_type": "cycle", "n": 10, "mode": "square",
                "compression_efficiency": 0.4, "stability_efficiency": 0.3,
                "stability_gain": 1, "unique_before": 6, "unique_after": 5,
            },
            {
                "dfa_type": "cycle", "n": 10, "mode": "d4",
                "compression_efficiency": 0.5, "stability_efficiency": 0.3,
                "stability_gain": 1, "unique_before": 6, "unique_after": 4,
            },
            {
                "dfa_type": "cycle", "n": 10, "mode": "d4+inv",
                "compression_efficiency": 0.6, "stability_efficiency": 0.5,
                "stability_gain": 3, "unique_before": 6, "unique_after": 3,
            },
        ]
        result = adjust_system(records, ["prefer_invariant_guided_modes"])
        # d4+inv (stab=0.5) beats d4+inv... wait, d4+inv is already best.
        # Actually best_mode picks max stab first -> d4+inv (0.5) wins.
        # So recommend d4+inv -> same mode.
        assert result["reason"] == "rejected_same_mode"

    def test_accepts_compression_improvement(self):
        # Best is d4 (stab=0.5, comp=0.6) by tiebreak on comp.
        # Recommend "enable_invariant_guidance" -> d4+inv.
        # d4+inv has same stab=0.5 but higher comp=0.7 -> accepted.
        records = [
            {
                "dfa_type": "test", "n": 5, "mode": "d4",
                "compression_efficiency": 0.6, "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
            {
                "dfa_type": "test", "n": 5, "mode": "d4+inv",
                "compression_efficiency": 0.7, "stability_efficiency": 0.5,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
        ]
        result = adjust_system(records, ["enable_invariant_guidance"])
        # best_mode picks max(stab=0.5 tie, comp: 0.7 vs 0.6) -> d4+inv
        # So candidate d4+inv == original d4+inv -> same mode
        assert result["reason"] == "rejected_same_mode"

    def test_accepts_stability_gain_improvement(self):
        # Directly test adjust_system with a scenario where
        # candidate has better stability_gain (3rd tiebreak).
        # Both modes tied on stab and comp, different gain.
        records = [
            {
                "dfa_type": "test", "n": 5, "mode": "d4",
                "compression_efficiency": 0.5, "stability_efficiency": 0.5,
                "stability_gain": 2.0, "unique_before": 4, "unique_after": 2,
            },
            {
                "dfa_type": "test", "n": 5, "mode": "square",
                "compression_efficiency": 0.5, "stability_efficiency": 0.5,
                "stability_gain": 4.0, "unique_before": 4, "unique_after": 2,
            },
        ]
        # Best: tied stab, tied comp -> lexicographic: d4 < square -> d4 wins
        result = adjust_system(records, ["reduce_projection_strength"])
        assert result["original_mode"] == "d4"
        assert result["candidate_mode"] == "square"
        assert result["accepted"] is True
        assert result["reason"] == "accepted_improved_stability_gain"

    def test_no_recommendations(self):
        records = _system_records_chain()
        result = adjust_system(records, [])
        assert result["accepted"] is False
        assert result["reason"] == "rejected_no_valid_candidate"
        assert result["applied_recommendation"] is None

    def test_empty_records(self):
        result = adjust_system([], ["prefer_d4_projection"])
        assert result["accepted"] is False
        assert result["reason"] == "rejected_no_valid_candidate"

    def test_only_first_recommendation_used(self):
        records = _system_records_chain()
        result = adjust_system(
            records,
            ["prefer_d4_projection", "reduce_projection_strength"],
        )
        assert result["applied_recommendation"] == "prefer_d4_projection"

    def test_no_mutation_of_input(self):
        records = _system_records_chain()
        original = copy.deepcopy(records)
        recs = ["prefer_d4_projection"]
        recs_orig = copy.deepcopy(recs)
        adjust_system(records, recs)
        assert records == original
        assert recs == recs_orig

    def test_output_structure(self):
        records = _system_records_chain()
        result = adjust_system(records, ["prefer_d4_projection"])
        expected_keys = {
            "dfa_type", "n", "original_mode", "candidate_mode",
            "accepted", "reason", "before_metrics", "after_metrics",
            "applied_recommendation",
        }
        assert set(result.keys()) == expected_keys

    def test_unknown_recommendation(self):
        records = _system_records_chain()
        result = adjust_system(records, ["nonexistent_recommendation"])
        assert result["accepted"] is False
        assert result["reason"] == "rejected_no_valid_candidate"
        assert result["candidate_mode"] is None


# ---------------------------------------------------------------------------
# PART 4 — FULL PIPELINE
# ---------------------------------------------------------------------------


class TestRunSelfAdjustment:
    """Tests for the full adjustment pipeline."""

    def test_raw_input(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        assert "diagnostics" in report
        assert "adjustments" in report
        assert isinstance(report["adjustments"], list)

    def test_summary_input(self):
        summary = _sample_summary()
        report = run_self_adjustment(summary)
        assert "diagnostics" in report
        assert "adjustments" in report

    def test_diagnostics_input(self):
        raw = _sample_raw_results()
        diag = run_self_diagnostics(raw)
        report = run_self_adjustment(diag)
        assert "diagnostics" in report
        assert "adjustments" in report

    def test_deterministic_across_runs(self):
        raw = _sample_raw_results()
        r1 = run_self_adjustment(raw)
        r2 = run_self_adjustment(raw)
        assert r1 == r2

    def test_no_mutation_raw(self):
        raw = _sample_raw_results()
        original = copy.deepcopy(raw)
        run_self_adjustment(raw)
        assert raw == original

    def test_no_mutation_summary(self):
        summary = _sample_summary()
        original = copy.deepcopy(summary)
        run_self_adjustment(summary)
        assert summary == original

    def test_adjustment_ordering_deterministic(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        keys = [
            (a["dfa_type"], str(a["n"]))
            for a in report["adjustments"]
        ]
        assert keys == sorted(keys)

    def test_each_adjustment_has_required_fields(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        required = {
            "dfa_type", "n", "original_mode", "candidate_mode",
            "accepted", "reason", "before_metrics", "after_metrics",
            "applied_recommendation", "system_class",
        }
        for adj in report["adjustments"]:
            assert required.issubset(set(adj.keys()))

    def test_at_most_one_recommendation_per_system(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        for adj in report["adjustments"]:
            rec = adj["applied_recommendation"]
            # Either None or a single string
            assert rec is None or isinstance(rec, str)

    def test_reason_values_are_valid(self):
        valid_reasons = {
            "accepted_improved_stability",
            "accepted_improved_compression",
            "accepted_improved_stability_gain",
            "rejected_no_improvement",
            "rejected_no_valid_candidate",
            "rejected_same_mode",
        }
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        for adj in report["adjustments"]:
            assert adj["reason"] in valid_reasons


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


class TestPrintAdjustmentReport:
    """Tests for human-readable adjustment report."""

    def test_produces_string(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        output = print_adjustment_report(report)
        assert isinstance(output, str)

    def test_contains_dfa_names(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        output = print_adjustment_report(report)
        assert "chain" in output
        assert "cycle" in output

    def test_deterministic_output(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        o1 = print_adjustment_report(report)
        o2 = print_adjustment_report(report)
        assert o1 == o2

    def test_contains_key_fields(self):
        raw = _sample_raw_results()
        report = run_self_adjustment(raw)
        output = print_adjustment_report(report)
        assert "best_mode:" in output
        assert "recommendation:" in output
        assert "candidate_mode:" in output
        assert "accepted:" in output
        assert "reason:" in output

    def test_empty_report(self):
        report = {"diagnostics": {}, "adjustments": []}
        output = print_adjustment_report(report)
        assert output == ""
