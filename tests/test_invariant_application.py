"""Tests for invariant application layer (v95.1.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.invariant_application import (
    apply_invariant_overlay,
    evaluate_overlay,
    get_accepted_invariants,
    print_application_report,
    run_invariant_application,
    _apply_bounded_projection_overlay,
    _apply_equivalence_class_overlay,
    _apply_explicit_allowed_state_overlay,
    _apply_geometry_alignment_overlay,
    _apply_local_stability_overlay,
    _best_mode_from_records,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_record(
    dfa_type="chain",
    n=5,
    mode="none",
    comp=0.0,
    stab=0.0,
    gain=0,
    ub=4,
    ua=4,
):
    """Build a single aggregated record."""
    return {
        "dfa_type": dfa_type,
        "n": n,
        "mode": mode,
        "compression_efficiency": comp,
        "stability_efficiency": stab,
        "stability_gain": gain,
        "unique_before": ub,
        "unique_after": ua,
    }


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


def _sample_system_records():
    """Records for a single system with various modes."""
    return [
        _make_record("cycle", 10, "none", 0.0, 0.0, 0, 6, 6),
        _make_record("cycle", 10, "square", 0.2, 0.1, 0, 6, 5),
        _make_record("cycle", 10, "d4", 0.1, 0.3, 1, 6, 4),
        _make_record("cycle", 10, "d4+inv", 0.4, 0.6, 3, 6, 3),
    ]


def _sample_synthesis_report():
    """A synthesis report with accepted invariants."""
    return {
        "structural_diagnostics": {
            "aggregated_records": [],
        },
        "gaps": [],
        "candidates": [],
        "evaluations": [],
        "accepted_invariants": [
            {
                "candidate": {
                    "type": "geometry_alignment_constraint",
                    "rule": {"align_projection": "d4"},
                    "strength": "soft",
                    "gap_type": "mode_disagreement",
                    "dfa_type": "cycle",
                    "n": 10,
                },
                "accepted": True,
                "reason": "improved_stability",
                "improvement": {
                    "before": {"stability_efficiency": 0.3,
                               "compression_efficiency": 0.1,
                               "stability_gain": 0.0},
                    "after": {"stability_efficiency": 0.6,
                              "compression_efficiency": 0.4,
                              "stability_gain": 3.0},
                },
            },
            {
                "candidate": {
                    "type": "local_stability_constraint",
                    "rule": {"limit_transition": ["square"],
                             "prefer_stable_modes": ["d4", "d4+inv"]},
                    "strength": "hard",
                    "gap_type": "unstable_correction_region",
                    "dfa_type": "chain",
                    "n": 5,
                },
                "accepted": True,
                "reason": "improved_stability",
                "improvement": {
                    "before": {"stability_efficiency": 0.2,
                               "compression_efficiency": 0.1,
                               "stability_gain": 0.0},
                    "after": {"stability_efficiency": 0.4,
                              "compression_efficiency": 0.08,
                              "stability_gain": 2.0},
                },
            },
        ],
    }


def _sample_raw_with_gaps():
    """Raw results designed to trigger multiple gap types."""
    return [
        # System 1: cycle n=10 — mode disagreement (spread > 0.4)
        _make_raw_result("cycle", 10, "none", 0.0, 0.0, 0, 6, 6),
        _make_raw_result("cycle", 10, "square", 0.2, 0.1, 0, 6, 5),
        _make_raw_result("cycle", 10, "d4", 0.1, 0.05, -1, 6, 5),
        _make_raw_result("cycle", 10, "d4+inv", 0.4, 0.6, 3, 6, 3),
        # System 2: chain n=5 — weak compression (all < 0.1)
        _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
        _make_raw_result("chain", 5, "square", 0.05, 0.3, 1, 4, 3),
        _make_raw_result("chain", 5, "d4", 0.08, 0.4, 2, 4, 3),
        _make_raw_result("chain", 5, "d4+inv", 0.09, 0.35, 1, 4, 3),
    ]


# ---------------------------------------------------------------------------
# PART 1 — GET ACCEPTED INVARIANTS
# ---------------------------------------------------------------------------


class TestGetAcceptedInvariants:
    """Tests for extracting accepted invariants from synthesis report."""

    def test_extracts_accepted_only(self):
        report = _sample_synthesis_report()
        result = get_accepted_invariants(report)
        assert len(result) == 2
        assert ("cycle", 10) in result
        assert ("chain", 5) in result

    def test_empty_report(self):
        report = {"accepted_invariants": []}
        result = get_accepted_invariants(report)
        assert result == {}

    def test_no_accepted_key(self):
        report = {}
        result = get_accepted_invariants(report)
        assert result == {}

    def test_filters_non_accepted(self):
        report = {
            "accepted_invariants": [
                {
                    "candidate": {
                        "type": "local_stability_constraint",
                        "dfa_type": "chain",
                        "n": 5,
                    },
                    "accepted": False,
                    "reason": "no_improvement",
                },
            ],
        }
        result = get_accepted_invariants(report)
        assert result == {}

    def test_deterministic_key_ordering(self):
        report = {
            "accepted_invariants": [
                {
                    "candidate": {
                        "type": "local_stability_constraint",
                        "dfa_type": "zebra",
                        "n": 3,
                    },
                    "accepted": True,
                    "reason": "improved_stability",
                },
                {
                    "candidate": {
                        "type": "geometry_alignment_constraint",
                        "dfa_type": "alpha",
                        "n": 1,
                    },
                    "accepted": True,
                    "reason": "improved_stability",
                },
            ],
        }
        result = get_accepted_invariants(report)
        keys = list(result.keys())
        assert keys == [("alpha", 1), ("zebra", 3)]


# ---------------------------------------------------------------------------
# PART 2 — OVERLAY APPLICATION
# ---------------------------------------------------------------------------


class TestLocalStabilityOverlay:
    """Tests for local_stability_constraint overlay."""

    def test_filters_bottom_25_percent(self):
        records = _sample_system_records()
        inv = {"type": "local_stability_constraint", "rule": {}}
        result = _apply_local_stability_overlay(records, inv)
        # Bottom 25% is the "none" record with stab=0.0.
        modes = [r["mode"] for r in result]
        assert "none" not in modes

    def test_keeps_at_least_one_record(self):
        records = [_make_record("chain", 5, "none", 0.0, 0.0, 0)]
        inv = {"type": "local_stability_constraint", "rule": {}}
        result = _apply_local_stability_overlay(records, inv)
        assert len(result) == 1

    def test_empty_records(self):
        result = _apply_local_stability_overlay([], {})
        assert result == []

    def test_sorted_by_stability_desc(self):
        records = _sample_system_records()
        inv = {"type": "local_stability_constraint", "rule": {}}
        result = _apply_local_stability_overlay(records, inv)
        stabs = [r["stability_efficiency"] for r in result]
        assert stabs == sorted(stabs, reverse=True)


class TestEquivalenceClassOverlay:
    """Tests for equivalence_class_constraint overlay."""

    def test_sorts_by_compression_desc(self):
        records = _sample_system_records()
        inv = {"type": "equivalence_class_constraint", "rule": {}}
        result = _apply_equivalence_class_overlay(records, inv)
        comps = [r["compression_efficiency"] for r in result]
        assert comps == sorted(comps, reverse=True)

    def test_empty_records(self):
        result = _apply_equivalence_class_overlay([], {})
        assert result == []

    def test_preserves_all_records(self):
        records = _sample_system_records()
        inv = {"type": "equivalence_class_constraint", "rule": {}}
        result = _apply_equivalence_class_overlay(records, inv)
        assert len(result) == len(records)


class TestGeometryAlignmentOverlay:
    """Tests for geometry_alignment_constraint overlay."""

    def test_prefers_aligned_mode(self):
        records = _sample_system_records()
        inv = {
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "d4"},
        }
        result = _apply_geometry_alignment_overlay(records, inv)
        assert result[0]["mode"] == "d4"

    def test_d4_square_default_preference(self):
        records = _sample_system_records()
        inv = {
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "square"},
        }
        result = _apply_geometry_alignment_overlay(records, inv)
        assert result[0]["mode"] == "square"

    def test_no_preferred_mode_falls_back(self):
        records = _sample_system_records()
        inv = {
            "type": "geometry_alignment_constraint",
            "rule": {},
        }
        result = _apply_geometry_alignment_overlay(records, inv)
        # d4 and square should be preferred.
        top_modes = [r["mode"] for r in result[:2]]
        assert "d4" in top_modes or "square" in top_modes


class TestExplicitAllowedStateOverlay:
    """Tests for explicit_allowed_state_constraint overlay."""

    def test_prefers_d4_inv_over_d4(self):
        records = _sample_system_records()
        inv = {"type": "explicit_allowed_state_constraint", "rule": {}}
        result = _apply_explicit_allowed_state_overlay(records, inv)
        assert result[0]["mode"] == "d4+inv"

    def test_d4_second_priority(self):
        records = [
            _make_record("cycle", 10, "d4", 0.1, 0.3, 1),
            _make_record("cycle", 10, "square", 0.2, 0.5, 2),
            _make_record("cycle", 10, "none", 0.0, 0.0, 0),
        ]
        inv = {"type": "explicit_allowed_state_constraint", "rule": {}}
        result = _apply_explicit_allowed_state_overlay(records, inv)
        assert result[0]["mode"] == "d4"


class TestBoundedProjectionOverlay:
    """Tests for bounded_projection_constraint overlay."""

    def test_removes_overcorrected(self):
        records = [
            _make_record("RM", 7, "none", 0.0, 0.0, 0),
            _make_record("RM", 7, "square", 0.7, 0.1, 1),
            _make_record("RM", 7, "d4", 0.3, 0.4, 2),
        ]
        inv = {"type": "bounded_projection_constraint", "rule": {}}
        result = _apply_bounded_projection_overlay(records, inv)
        modes = [r["mode"] for r in result]
        assert "square" not in modes

    def test_keeps_all_if_all_overcorrected(self):
        records = [
            _make_record("RM", 7, "square", 0.7, 0.1, 1),
            _make_record("RM", 7, "d4", 0.8, 0.05, 0),
        ]
        inv = {"type": "bounded_projection_constraint", "rule": {}}
        result = _apply_bounded_projection_overlay(records, inv)
        assert len(result) == 2

    def test_empty_records(self):
        result = _apply_bounded_projection_overlay([], {})
        assert result == []

    def test_sorted_by_stability_desc(self):
        records = [
            _make_record("RM", 7, "none", 0.0, 0.2, 0),
            _make_record("RM", 7, "d4", 0.3, 0.4, 2),
            _make_record("RM", 7, "d4+inv", 0.2, 0.3, 1),
        ]
        inv = {"type": "bounded_projection_constraint", "rule": {}}
        result = _apply_bounded_projection_overlay(records, inv)
        stabs = [r["stability_efficiency"] for r in result]
        assert stabs == sorted(stabs, reverse=True)


# ---------------------------------------------------------------------------
# OVERLAY INTEGRATION
# ---------------------------------------------------------------------------


class TestApplyInvariantOverlay:
    """Tests for the combined overlay application."""

    def test_applies_single_invariant(self):
        records = _sample_system_records()
        invariants = [{
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "d4"},
        }]
        result = apply_invariant_overlay(records, invariants)
        assert result[0]["mode"] == "d4"

    def test_applies_multiple_invariants(self):
        records = _sample_system_records()
        invariants = [
            {
                "type": "local_stability_constraint",
                "rule": {},
            },
            {
                "type": "explicit_allowed_state_constraint",
                "rule": {},
            },
        ]
        result = apply_invariant_overlay(records, invariants)
        assert len(result) > 0
        # d4+inv should be preferred after explicit_allowed_state.
        assert result[0]["mode"] == "d4+inv"

    def test_no_mutation_of_originals(self):
        records = _sample_system_records()
        original = copy.deepcopy(records)
        invariants = [{
            "type": "bounded_projection_constraint",
            "rule": {},
        }]
        apply_invariant_overlay(records, invariants)
        assert records == original

    def test_empty_records(self):
        result = apply_invariant_overlay([], [{"type": "local_stability_constraint"}])
        assert result == []

    def test_empty_invariants(self):
        records = _sample_system_records()
        result = apply_invariant_overlay(records, [])
        assert len(result) == len(records)

    def test_unknown_invariant_type_passthrough(self):
        records = _sample_system_records()
        result = apply_invariant_overlay(records, [{"type": "unknown_type"}])
        assert len(result) == len(records)

    def test_deterministic_output(self):
        records = _sample_system_records()
        invariants = [
            {"type": "local_stability_constraint", "rule": {}},
            {"type": "explicit_allowed_state_constraint", "rule": {}},
        ]
        r1 = apply_invariant_overlay(records, invariants)
        r2 = apply_invariant_overlay(records, invariants)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART 3 — EVALUATION
# ---------------------------------------------------------------------------


class TestBestModeFromRecords:
    """Tests for _best_mode_from_records."""

    def test_selects_highest_stability(self):
        records = _sample_system_records()
        mode, metrics = _best_mode_from_records(records)
        assert mode == "d4+inv"
        assert metrics["stability_efficiency"] == 0.6

    def test_empty_records(self):
        mode, metrics = _best_mode_from_records([])
        assert mode == "none"
        assert metrics["stability_efficiency"] == 0.0

    def test_tiebreak_on_compression(self):
        records = [
            _make_record("chain", 5, "d4", 0.3, 0.5, 1),
            _make_record("chain", 5, "square", 0.5, 0.5, 1),
        ]
        mode, _ = _best_mode_from_records(records)
        assert mode == "square"


class TestEvaluateOverlay:
    """Tests for overlay evaluation."""

    def test_improved_stability(self):
        before = _sample_system_records()
        after = [
            _make_record("cycle", 10, "d4+inv", 0.4, 0.8, 3),
        ]
        result = evaluate_overlay(before, after)
        assert result["improved"] is True
        assert result["reason"] == "improved_stability"

    def test_no_change(self):
        records = _sample_system_records()
        result = evaluate_overlay(records, records)
        assert result["improved"] is False
        assert result["reason"] == "no_change"
        assert result["before_mode"] == result["after_mode"]

    def test_mode_changed_no_improvement(self):
        before = [
            _make_record("cycle", 10, "d4", 0.3, 0.5, 1),
            _make_record("cycle", 10, "square", 0.2, 0.4, 0),
        ]
        after = [
            _make_record("cycle", 10, "square", 0.2, 0.4, 0),
        ]
        result = evaluate_overlay(before, after)
        assert result["improved"] is False
        assert result["reason"] == "mode_changed_no_improvement"

    def test_result_format(self):
        records = _sample_system_records()
        result = evaluate_overlay(records, records)
        assert "before_mode" in result
        assert "after_mode" in result
        assert "improved" in result
        assert "reason" in result
        assert "before_metrics" in result
        assert "after_metrics" in result

    def test_improved_compression(self):
        before = [
            _make_record("cycle", 10, "d4", 0.3, 0.5, 1),
        ]
        after = [
            _make_record("cycle", 10, "square", 0.5, 0.5, 1),
        ]
        result = evaluate_overlay(before, after)
        assert result["improved"] is True
        assert result["reason"] == "improved_compression"


# ---------------------------------------------------------------------------
# PART 4 — FULL PIPELINE
# ---------------------------------------------------------------------------


class TestRunInvariantApplication:
    """Tests for the full invariant application pipeline."""

    def test_with_raw_data(self):
        data = _sample_raw_with_gaps()
        result = run_invariant_application(data)
        assert "accepted_invariants" in result
        assert "applications" in result
        assert isinstance(result["applications"], list)

    def test_with_synthesis_report(self):
        """Test passing a pre-computed synthesis report."""
        raw_data = _sample_raw_with_gaps()
        from qec.analysis.invariant_synthesis import run_invariant_synthesis
        synthesis_report = run_invariant_synthesis(raw_data)
        result = run_invariant_application(synthesis_report)
        assert "accepted_invariants" in result
        assert "applications" in result

    def test_application_format(self):
        data = _sample_raw_with_gaps()
        result = run_invariant_application(data)
        for app in result["applications"]:
            assert "dfa_type" in app
            assert "n" in app
            assert "invariants" in app
            assert "before_mode" in app
            assert "after_mode" in app
            assert "improved" in app
            assert "reason" in app

    def test_deterministic(self):
        data = _sample_raw_with_gaps()
        r1 = run_invariant_application(data)
        r2 = run_invariant_application(data)
        assert r1 == r2

    def test_no_mutation_of_input(self):
        data = _sample_raw_with_gaps()
        original = copy.deepcopy(data)
        run_invariant_application(data)
        assert data == original


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER
# ---------------------------------------------------------------------------


class TestPrintApplicationReport:
    """Tests for the print report layer."""

    def test_basic_format(self):
        report = {
            "accepted_invariants": {},
            "applications": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "invariants": [
                        "geometry_alignment_constraint",
                        "explicit_allowed_state_constraint",
                    ],
                    "before_mode": "square",
                    "after_mode": "d4+inv",
                    "improved": True,
                    "reason": "improved_stability",
                },
            ],
        }
        text = print_application_report(report)
        assert "DFA: cycle (n=10)" in text
        assert "geometry_alignment_constraint" in text
        assert "before: square" in text
        assert "after: d4+inv" in text
        assert "improved: True" in text
        assert "reason: improved_stability" in text

    def test_empty_report(self):
        report = {"accepted_invariants": {}, "applications": []}
        text = print_application_report(report)
        assert "No invariants applied." in text

    def test_summary_counts(self):
        report = {
            "accepted_invariants": {},
            "applications": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "invariants": ["geometry_alignment_constraint"],
                    "before_mode": "square",
                    "after_mode": "d4+inv",
                    "improved": True,
                    "reason": "improved_stability",
                },
                {
                    "dfa_type": "chain",
                    "n": 5,
                    "invariants": ["local_stability_constraint"],
                    "before_mode": "d4",
                    "after_mode": "d4",
                    "improved": False,
                    "reason": "no_change",
                },
            ],
        }
        text = print_application_report(report)
        assert "Total applications: 2" in text
        assert "Improved: 1" in text
        assert "Unchanged: 1" in text

    def test_deterministic_output(self):
        report = {
            "accepted_invariants": {},
            "applications": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "invariants": ["geometry_alignment_constraint"],
                    "before_mode": "square",
                    "after_mode": "d4",
                    "improved": True,
                    "reason": "improved_stability",
                },
            ],
        }
        t1 = print_application_report(report)
        t2 = print_application_report(report)
        assert t1 == t2


# ---------------------------------------------------------------------------
# EDGE CASES & INVARIANTS
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and invariant tests."""

    def test_overlay_does_not_modify_base_records(self):
        """Critical: overlay must never mutate base data."""
        records = _sample_system_records()
        original = copy.deepcopy(records)
        invariants = [
            {"type": "local_stability_constraint", "rule": {}},
            {"type": "bounded_projection_constraint", "rule": {}},
            {"type": "explicit_allowed_state_constraint", "rule": {}},
        ]
        apply_invariant_overlay(records, invariants)
        assert records == original

    def test_stable_ordering_across_calls(self):
        """Output ordering must be deterministic."""
        records = _sample_system_records()
        inv = [{"type": "geometry_alignment_constraint",
                "rule": {"align_projection": "d4"}}]
        results = [apply_invariant_overlay(records, inv) for _ in range(5)]
        for r in results[1:]:
            assert r == results[0]

    def test_single_record_system(self):
        """Systems with one record should pass through."""
        records = [_make_record("chain", 5, "d4", 0.3, 0.5, 1)]
        result = apply_invariant_overlay(
            records,
            [{"type": "local_stability_constraint", "rule": {}}],
        )
        assert len(result) == 1

    def test_all_zero_metrics(self):
        """System where all metrics are zero."""
        records = [
            _make_record("chain", 5, "none", 0.0, 0.0, 0),
            _make_record("chain", 5, "d4", 0.0, 0.0, 0),
        ]
        result = apply_invariant_overlay(
            records,
            [{"type": "local_stability_constraint", "rule": {}}],
        )
        assert len(result) >= 1

    def test_evaluation_with_empty_overlay(self):
        """Evaluating empty overlay returns no_change."""
        records = _sample_system_records()
        result = evaluate_overlay(records, [])
        assert result["improved"] is False

    def test_no_feedback_loops(self):
        """Re-applying overlay converges: f(f(f(x))) == f(f(x))."""
        records = _sample_system_records()
        invariants = [
            {"type": "explicit_allowed_state_constraint", "rule": {}},
        ]
        first_pass = apply_invariant_overlay(records, invariants)
        second_pass = apply_invariant_overlay(first_pass, invariants)
        third_pass = apply_invariant_overlay(second_pass, invariants)
        # After two passes, should be stable (converged).
        assert second_pass == third_pass
