"""Tests for invariant synthesis and validation loop (v95.0.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.invariant_synthesis import (
    INVARIANT_TEMPLATES,
    apply_candidate_proxy,
    evaluate_candidate,
    generate_candidates,
    print_invariant_report,
    run_invariant_synthesis,
    synthesize_for_system,
    _current_best_metrics,
    _extract_proxy_metrics,
    _is_improvement,
)


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


def _sample_raw_overcorrection():
    """Raw results with overcorrection pattern."""
    return [
        _make_raw_result("RM", 7, "none", 0.0, 0.0, 0, 4, 4),
        _make_raw_result("RM", 7, "square", 0.7, 0.1, 1, 4, 2),
        _make_raw_result("RM", 7, "d4", 0.8, 0.05, 0, 4, 1),
        _make_raw_result("RM", 7, "d4+inv", 0.5, 0.4, 2, 4, 2),
    ]


def _sample_raw_invariant_dependency():
    """Raw results with invariant dependency (d4+inv >> d4)."""
    return [
        _make_raw_result("QLDPC", 15, "none", 0.0, 0.0, 0, 8, 8),
        _make_raw_result("QLDPC", 15, "square", 0.3, 0.2, 1, 8, 6),
        _make_raw_result("QLDPC", 15, "d4", 0.4, 0.2, 1, 8, 5),
        _make_raw_result("QLDPC", 15, "d4+inv", 0.5, 0.6, 4, 8, 4),
    ]


def _system_records_mode_disagreement():
    """Aggregated records with mode disagreement (spread > 0.4)."""
    return [
        {
            "dfa_type": "cycle", "n": 10, "mode": "none",
            "compression_efficiency": 0.0, "stability_efficiency": 0.0,
            "stability_gain": 0, "unique_before": 6, "unique_after": 6,
        },
        {
            "dfa_type": "cycle", "n": 10, "mode": "square",
            "compression_efficiency": 0.2, "stability_efficiency": 0.1,
            "stability_gain": 0, "unique_before": 6, "unique_after": 5,
        },
        {
            "dfa_type": "cycle", "n": 10, "mode": "d4",
            "compression_efficiency": 0.1, "stability_efficiency": 0.05,
            "stability_gain": -1, "unique_before": 6, "unique_after": 5,
        },
        {
            "dfa_type": "cycle", "n": 10, "mode": "d4+inv",
            "compression_efficiency": 0.4, "stability_efficiency": 0.6,
            "stability_gain": 3, "unique_before": 6, "unique_after": 3,
        },
    ]


def _gap_mode_disagreement():
    """A mode_disagreement gap."""
    return {
        "dfa_type": "cycle",
        "n": 10,
        "gap_type": "mode_disagreement",
        "evidence": {
            "stability_by_mode": {
                "d4": 0.05,
                "d4+inv": 0.6,
                "none": 0.0,
                "square": 0.1,
            },
            "spread": 0.6,
        },
    }


def _gap_unstable_correction():
    """An unstable_correction_region gap."""
    return {
        "dfa_type": "cycle",
        "n": 10,
        "gap_type": "unstable_correction_region",
        "evidence": {
            "negative_gain_modes": ["d4"],
            "gains": {"d4": -1, "d4+inv": 3, "none": 0, "square": 0},
        },
    }


def _gap_weak_compression():
    """A weak_compression_structure gap."""
    return {
        "dfa_type": "chain",
        "n": 5,
        "gap_type": "weak_compression_structure",
        "evidence": {
            "compression_by_mode": {
                "d4": 0.08,
                "d4+inv": 0.09,
                "none": 0.0,
                "square": 0.05,
            },
            "max_compression": 0.09,
        },
    }


def _gap_invariant_dependency():
    """An invariant_dependency gap."""
    return {
        "dfa_type": "QLDPC",
        "n": 15,
        "gap_type": "invariant_dependency",
        "evidence": {
            "d4_stability": 0.2,
            "d4_inv_stability": 0.6,
            "delta": 0.4,
        },
    }


def _gap_overcorrection():
    """An overcorrection_pattern gap."""
    return {
        "dfa_type": "RM",
        "n": 7,
        "gap_type": "overcorrection_pattern",
        "evidence": {
            "overcorrected_modes": ["d4", "square"],
            "metrics": {
                "d4": {"compression": 0.8, "stability": 0.05},
                "square": {"compression": 0.7, "stability": 0.1},
            },
        },
    }


# ---------------------------------------------------------------------------
# PART 1 — INVARIANT TEMPLATES
# ---------------------------------------------------------------------------


class TestInvariantTemplates:
    """Tests for template registry and generators."""

    def test_all_five_templates_registered(self):
        expected = {
            "local_stability_constraint",
            "equivalence_class_constraint",
            "geometry_alignment_constraint",
            "explicit_allowed_state_constraint",
            "bounded_projection_constraint",
        }
        assert set(INVARIANT_TEMPLATES.keys()) == expected

    def test_all_templates_are_callable(self):
        for key, fn in INVARIANT_TEMPLATES.items():
            assert callable(fn), f"{key} is not callable"

    def test_local_stability_generates_candidates(self):
        gap = _gap_unstable_correction()
        records = _system_records_mode_disagreement()
        candidates = INVARIANT_TEMPLATES["local_stability_constraint"](
            gap, records
        )
        assert len(candidates) >= 1
        assert len(candidates) <= 3
        for c in candidates:
            assert c["type"] == "local_stability_constraint"

    def test_equivalence_class_generates_candidates(self):
        gap = _gap_weak_compression()
        records = [
            {
                "dfa_type": "chain", "n": 5, "mode": "none",
                "compression_efficiency": 0.0, "stability_efficiency": 0.0,
                "stability_gain": 0, "unique_before": 4, "unique_after": 4,
            },
            {
                "dfa_type": "chain", "n": 5, "mode": "d4+inv",
                "compression_efficiency": 0.09, "stability_efficiency": 0.35,
                "stability_gain": 1, "unique_before": 4, "unique_after": 3,
            },
        ]
        candidates = INVARIANT_TEMPLATES["equivalence_class_constraint"](
            gap, records
        )
        assert len(candidates) >= 1
        assert len(candidates) <= 3

    def test_geometry_alignment_generates_candidates(self):
        gap = _gap_mode_disagreement()
        records = _system_records_mode_disagreement()
        candidates = INVARIANT_TEMPLATES["geometry_alignment_constraint"](
            gap, records
        )
        assert len(candidates) >= 1
        assert len(candidates) <= 3

    def test_explicit_allowed_state_generates_candidates(self):
        gap = _gap_invariant_dependency()
        records = _system_records_mode_disagreement()
        candidates = INVARIANT_TEMPLATES["explicit_allowed_state_constraint"](
            gap, records
        )
        assert len(candidates) == 2
        for c in candidates:
            assert c["type"] == "explicit_allowed_state_constraint"

    def test_bounded_projection_generates_candidates(self):
        gap = _gap_overcorrection()
        records = [
            {
                "dfa_type": "RM", "n": 7, "mode": "none",
                "compression_efficiency": 0.0, "stability_efficiency": 0.0,
                "stability_gain": 0, "unique_before": 4, "unique_after": 4,
            },
            {
                "dfa_type": "RM", "n": 7, "mode": "d4",
                "compression_efficiency": 0.8, "stability_efficiency": 0.05,
                "stability_gain": 0, "unique_before": 4, "unique_after": 1,
            },
            {
                "dfa_type": "RM", "n": 7, "mode": "d4+inv",
                "compression_efficiency": 0.5, "stability_efficiency": 0.4,
                "stability_gain": 2, "unique_before": 4, "unique_after": 2,
            },
        ]
        candidates = INVARIANT_TEMPLATES["bounded_projection_constraint"](
            gap, records
        )
        assert len(candidates) >= 1
        assert len(candidates) <= 3


# ---------------------------------------------------------------------------
# PART 2 — CANDIDATE GENERATION
# ---------------------------------------------------------------------------


class TestGenerateCandidates:
    """Tests for generate_candidates()."""

    def test_max_three_candidates(self):
        gap = _gap_unstable_correction()
        records = _system_records_mode_disagreement()
        candidates = generate_candidates(gap, records)
        assert len(candidates) <= 3

    def test_candidates_annotated_with_gap_info(self):
        gap = _gap_mode_disagreement()
        records = _system_records_mode_disagreement()
        candidates = generate_candidates(gap, records)
        for c in candidates:
            assert c["gap_type"] == "mode_disagreement"
            assert c["dfa_type"] == "cycle"
            assert c["n"] == 10

    def test_unknown_gap_type_returns_empty(self):
        gap = {"gap_type": "unknown_gap", "dfa_type": "x", "n": 1}
        candidates = generate_candidates(gap, [])
        assert candidates == []

    def test_deterministic_ordering(self):
        gap = _gap_unstable_correction()
        records = _system_records_mode_disagreement()
        c1 = generate_candidates(gap, records)
        c2 = generate_candidates(gap, records)
        assert c1 == c2

    def test_all_gap_types_produce_candidates(self):
        gaps_and_records = [
            (_gap_unstable_correction(), _system_records_mode_disagreement()),
            (_gap_weak_compression(), [
                {"dfa_type": "chain", "n": 5, "mode": "none",
                 "compression_efficiency": 0.0, "stability_efficiency": 0.0,
                 "stability_gain": 0, "unique_before": 4, "unique_after": 4},
                {"dfa_type": "chain", "n": 5, "mode": "d4+inv",
                 "compression_efficiency": 0.09, "stability_efficiency": 0.35,
                 "stability_gain": 1, "unique_before": 4, "unique_after": 3},
            ]),
            (_gap_mode_disagreement(), _system_records_mode_disagreement()),
            (_gap_invariant_dependency(), _system_records_mode_disagreement()),
            (_gap_overcorrection(), [
                {"dfa_type": "RM", "n": 7, "mode": "d4",
                 "compression_efficiency": 0.8, "stability_efficiency": 0.05,
                 "stability_gain": 0, "unique_before": 4, "unique_after": 1},
            ]),
        ]
        for gap, records in gaps_and_records:
            candidates = generate_candidates(gap, records)
            assert len(candidates) >= 1, f"No candidates for {gap['gap_type']}"

    def test_no_mutation_of_gap(self):
        gap = _gap_unstable_correction()
        original = copy.deepcopy(gap)
        records = _system_records_mode_disagreement()
        generate_candidates(gap, records)
        assert gap == original

    def test_no_mutation_of_records(self):
        gap = _gap_mode_disagreement()
        records = _system_records_mode_disagreement()
        original = copy.deepcopy(records)
        generate_candidates(gap, records)
        assert records == original


# ---------------------------------------------------------------------------
# PART 3 — PROXY EVALUATION
# ---------------------------------------------------------------------------


class TestApplyCandidateProxy:
    """Tests for apply_candidate_proxy()."""

    def test_local_stability_excludes_negative_modes(self):
        candidate = {
            "type": "local_stability_constraint",
            "rule": {"limit_transition": ["d4"], "action": "exclude"},
            "strength": "hard",
        }
        records = _system_records_mode_disagreement()
        metrics = apply_candidate_proxy(candidate, records)
        # d4 excluded, best remaining is d4+inv with stab=0.6.
        assert metrics["stability_efficiency"] == 0.6

    def test_geometry_alignment_selects_aligned_mode(self):
        candidate = {
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "d4+inv"},
            "strength": "hard",
        }
        records = _system_records_mode_disagreement()
        metrics = apply_candidate_proxy(candidate, records)
        assert metrics["stability_efficiency"] == 0.6
        assert metrics["compression_efficiency"] == 0.4

    def test_explicit_allowed_filters_to_allowed(self):
        candidate = {
            "type": "explicit_allowed_state_constraint",
            "rule": {"allowed_states": ["d4+inv"]},
            "strength": "hard",
        }
        records = _system_records_mode_disagreement()
        metrics = apply_candidate_proxy(candidate, records)
        assert metrics["stability_efficiency"] == 0.6

    def test_bounded_projection_prefers_balanced(self):
        candidate = {
            "type": "bounded_projection_constraint",
            "rule": {"prefer_balanced_modes": ["none"]},
            "strength": "soft",
        }
        records = [
            {
                "dfa_type": "RM", "n": 7, "mode": "d4",
                "compression_efficiency": 0.8, "stability_efficiency": 0.05,
                "stability_gain": 0, "unique_before": 4, "unique_after": 1,
            },
            {
                "dfa_type": "RM", "n": 7, "mode": "none",
                "compression_efficiency": 0.0, "stability_efficiency": 0.0,
                "stability_gain": 0, "unique_before": 4, "unique_after": 4,
            },
        ]
        metrics = apply_candidate_proxy(candidate, records)
        # "none" has stab >= comp (0.0 >= 0.0), "d4" doesn't (0.05 < 0.8).
        assert metrics["stability_efficiency"] == 0.0

    def test_empty_records_returns_zeros(self):
        candidate = {
            "type": "local_stability_constraint",
            "rule": {},
            "strength": "hard",
        }
        metrics = apply_candidate_proxy(candidate, [])
        assert metrics["stability_efficiency"] == 0.0
        assert metrics["compression_efficiency"] == 0.0
        assert metrics["stability_gain"] == 0.0

    def test_unknown_type_returns_best_overall(self):
        candidate = {"type": "unknown_type", "rule": {}, "strength": "soft"}
        records = _system_records_mode_disagreement()
        metrics = apply_candidate_proxy(candidate, records)
        # Best overall stability is d4+inv at 0.6.
        assert metrics["stability_efficiency"] == 0.6

    def test_no_mutation_of_records(self):
        candidate = {
            "type": "local_stability_constraint",
            "rule": {"limit_transition": ["d4"]},
            "strength": "hard",
        }
        records = _system_records_mode_disagreement()
        original = copy.deepcopy(records)
        apply_candidate_proxy(candidate, records)
        assert records == original


# ---------------------------------------------------------------------------
# PART 4 — VALIDATION
# ---------------------------------------------------------------------------


class TestIsImprovement:
    """Tests for _is_improvement()."""

    def test_higher_stability_accepted(self):
        before = {"stability_efficiency": 0.3, "compression_efficiency": 0.5}
        after = {"stability_efficiency": 0.6, "compression_efficiency": 0.3}
        accepted, reason = _is_improvement(before, after)
        assert accepted is True
        assert reason == "improved_stability"

    def test_lower_stability_rejected(self):
        before = {"stability_efficiency": 0.6}
        after = {"stability_efficiency": 0.3}
        accepted, reason = _is_improvement(before, after)
        assert accepted is False
        assert reason == "no_improvement"

    def test_tied_stability_higher_compression_accepted(self):
        before = {"stability_efficiency": 0.5, "compression_efficiency": 0.3}
        after = {"stability_efficiency": 0.5, "compression_efficiency": 0.6}
        accepted, reason = _is_improvement(before, after)
        assert accepted is True
        assert reason == "improved_compression"

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
        accepted, reason = _is_improvement(before, after)
        assert accepted is True
        assert reason == "improved_stability_gain"

    def test_identical_metrics_rejected(self):
        m = {"stability_efficiency": 0.5, "compression_efficiency": 0.5}
        accepted, reason = _is_improvement(m, dict(m))
        assert accepted is False
        assert reason == "no_improvement"

    def test_missing_keys_default_to_zero(self):
        accepted, reason = _is_improvement({}, {"stability_efficiency": 0.1})
        assert accepted is True


class TestEvaluateCandidate:
    """Tests for evaluate_candidate()."""

    def test_accepted_candidate(self):
        candidate = {
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "d4+inv"},
            "strength": "hard",
        }
        current = {"stability_efficiency": 0.1, "compression_efficiency": 0.0}
        records = _system_records_mode_disagreement()
        result = evaluate_candidate(candidate, current, records)
        assert result["accepted"] is True
        assert "before" in result["improvement"]
        assert "after" in result["improvement"]

    def test_rejected_candidate(self):
        candidate = {
            "type": "geometry_alignment_constraint",
            "rule": {"align_projection": "none"},
            "strength": "soft",
        }
        # Current best is already 0.6.
        current = {
            "stability_efficiency": 0.6,
            "compression_efficiency": 0.4,
            "stability_gain": 3.0,
        }
        records = _system_records_mode_disagreement()
        result = evaluate_candidate(candidate, current, records)
        assert result["accepted"] is False
        assert result["reason"] == "no_improvement"

    def test_output_structure(self):
        candidate = {
            "type": "local_stability_constraint",
            "rule": {},
            "strength": "hard",
        }
        current = {"stability_efficiency": 0.0}
        records = _system_records_mode_disagreement()
        result = evaluate_candidate(candidate, current, records)
        assert "candidate" in result
        assert "accepted" in result
        assert "reason" in result
        assert "improvement" in result
        assert isinstance(result["accepted"], bool)


# ---------------------------------------------------------------------------
# PART 5 — PER-SYSTEM SYNTHESIS
# ---------------------------------------------------------------------------


class TestSynthesizeForSystem:
    """Tests for synthesize_for_system()."""

    def test_produces_candidates_and_evaluations(self):
        records = _system_records_mode_disagreement()
        gaps = [_gap_mode_disagreement()]
        result = synthesize_for_system(records, gaps)
        assert len(result["candidates"]) >= 1
        assert len(result["evaluations"]) >= 1

    def test_accepted_invariants_subset_of_evaluations(self):
        records = _system_records_mode_disagreement()
        gaps = [_gap_mode_disagreement()]
        result = synthesize_for_system(records, gaps)
        for acc in result["accepted_invariants"]:
            assert acc in result["evaluations"]

    def test_no_gaps_returns_empty(self):
        records = _system_records_mode_disagreement()
        result = synthesize_for_system(records, [])
        assert result["candidates"] == []
        assert result["evaluations"] == []
        assert result["accepted_invariants"] == []

    def test_multiple_gaps_handled(self):
        records = _system_records_mode_disagreement()
        gaps = [_gap_unstable_correction(), _gap_mode_disagreement()]
        result = synthesize_for_system(records, gaps)
        assert len(result["candidates"]) >= 2

    def test_deterministic(self):
        records = _system_records_mode_disagreement()
        gaps = [_gap_mode_disagreement()]
        r1 = synthesize_for_system(records, gaps)
        r2 = synthesize_for_system(records, gaps)
        assert r1 == r2


# ---------------------------------------------------------------------------
# PART 6 — FULL PIPELINE
# ---------------------------------------------------------------------------


class TestRunInvariantSynthesis:
    """Tests for run_invariant_synthesis()."""

    def test_raw_input(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        assert "gaps" in report
        assert "candidates" in report
        assert "evaluations" in report
        assert "accepted_invariants" in report

    def test_deterministic_across_runs(self):
        raw = _sample_raw_with_gaps()
        r1 = run_invariant_synthesis(raw)
        r2 = run_invariant_synthesis(raw)
        assert r1 == r2

    def test_no_mutation_of_input(self):
        raw = _sample_raw_with_gaps()
        original = copy.deepcopy(raw)
        run_invariant_synthesis(raw)
        assert raw == original

    def test_candidates_bounded(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        # Each gap produces at most 3 candidates.
        num_gaps = len(report["gaps"])
        assert len(report["candidates"]) <= num_gaps * 3

    def test_overcorrection_pipeline(self):
        raw = _sample_raw_overcorrection()
        report = run_invariant_synthesis(raw)
        assert "candidates" in report
        assert isinstance(report["candidates"], list)

    def test_invariant_dependency_pipeline(self):
        raw = _sample_raw_invariant_dependency()
        report = run_invariant_synthesis(raw)
        assert "candidates" in report

    def test_structural_diagnostics_included(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        assert "structural_diagnostics" in report


# ---------------------------------------------------------------------------
# PART 7 — PRINT LAYER
# ---------------------------------------------------------------------------


class TestPrintInvariantReport:
    """Tests for print_invariant_report()."""

    def test_produces_string(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        output = print_invariant_report(report)
        assert isinstance(output, str)

    def test_deterministic_output(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        o1 = print_invariant_report(report)
        o2 = print_invariant_report(report)
        assert o1 == o2

    def test_contains_header(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        output = print_invariant_report(report)
        assert "Invariant Synthesis Report" in output

    def test_contains_summary(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        output = print_invariant_report(report)
        assert "Total gaps:" in output
        assert "Total candidates:" in output
        assert "Accepted invariants:" in output

    def test_empty_report(self):
        report = {
            "gaps": [],
            "candidates": [],
            "evaluations": [],
            "accepted_invariants": [],
        }
        output = print_invariant_report(report)
        assert "No gaps detected" in output

    def test_contains_gap_info(self):
        raw = _sample_raw_with_gaps()
        report = run_invariant_synthesis(raw)
        output = print_invariant_report(report)
        if report["gaps"]:
            assert "gap:" in output


# ---------------------------------------------------------------------------
# PART 8 — EDGE CASES AND INVARIANTS
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and invariant checks."""

    def test_extract_proxy_metrics_from_record(self):
        record = {
            "stability_efficiency": 0.5,
            "compression_efficiency": 0.3,
            "stability_gain": 2,
        }
        m = _extract_proxy_metrics(record)
        assert m["stability_efficiency"] == 0.5
        assert m["compression_efficiency"] == 0.3
        assert m["stability_gain"] == 2.0

    def test_extract_proxy_metrics_defaults(self):
        m = _extract_proxy_metrics({})
        assert m["stability_efficiency"] == 0.0
        assert m["compression_efficiency"] == 0.0
        assert m["stability_gain"] == 0.0

    def test_current_best_metrics_empty(self):
        m = _current_best_metrics([])
        assert m["stability_efficiency"] == 0.0

    def test_current_best_metrics_selects_best(self):
        records = _system_records_mode_disagreement()
        m = _current_best_metrics(records)
        assert m["stability_efficiency"] == 0.6

    def test_equivalence_class_fallback(self):
        """Equivalence class generates fallback when no non-zero modes."""
        gap = {
            "gap_type": "weak_compression_structure",
            "dfa_type": "test",
            "n": 1,
            "evidence": {
                "compression_by_mode": {"none": 0.0},
                "max_compression": 0.0,
            },
        }
        records = [
            {"dfa_type": "test", "n": 1, "mode": "none",
             "compression_efficiency": 0.0, "stability_efficiency": 0.0,
             "stability_gain": 0, "unique_before": 2, "unique_after": 2},
        ]
        candidates = generate_candidates(gap, records)
        assert len(candidates) >= 1

    def test_geometry_alignment_fallback(self):
        """Geometry alignment falls back to best mode if d4/square weak."""
        gap = {
            "gap_type": "mode_disagreement",
            "dfa_type": "test",
            "n": 1,
            "evidence": {
                "stability_by_mode": {"none": 0.8, "square": 0.0, "d4": 0.0},
                "spread": 0.8,
            },
        }
        records = []
        candidates = generate_candidates(gap, records)
        assert len(candidates) >= 1
        # Should fall back to "none" as best.
        assert any(
            c["rule"].get("align_projection") == "none" for c in candidates
        )

    def test_bounded_projection_fallback(self):
        """Bounded projection generates fallback when no metrics."""
        gap = {
            "gap_type": "overcorrection_pattern",
            "dfa_type": "test",
            "n": 1,
            "evidence": {
                "overcorrected_modes": [],
                "metrics": {},
            },
        }
        records = []
        candidates = generate_candidates(gap, records)
        assert len(candidates) >= 1
        assert candidates[0]["rule"]["action"] == "default_cap"
