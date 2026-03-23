"""Tests for structural gap detection and invariant opportunity mapping (v94.1.0).

Deterministic, no randomness, no mutation of inputs.
~20 tests covering:
  - gap detection correctness
  - invariant mapping correctness
  - confidence assignment
  - deterministic output
  - no mutation
  - stable ordering
"""

import copy

import pytest

from qec.analysis.structural_diagnostics import (
    detect_structural_gaps,
    map_all_gaps_to_invariants,
    map_gap_to_invariant,
    print_structural_report,
    run_structural_diagnostics,
    _group_by_system,
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


def _make_system_records(entries):
    """Build system_records dict from list of (dfa, n, mode, comp, stab, gain) tuples."""
    records = {}
    for dfa, n, mode, comp, stab, gain in entries:
        key = (dfa, n)
        records.setdefault(key, []).append({
            "dfa_type": dfa,
            "n": n,
            "mode": mode,
            "compression_efficiency": comp,
            "stability_efficiency": stab,
            "stability_gain": gain,
            "unique_before": 4,
            "unique_after": 4,
        })
    return records


def _empty_diagnostics():
    """Minimal diagnostics dict for testing."""
    return {
        "metrics": [],
        "best_modes": [],
        "system_classes": [],
        "issues": [],
        "recommendations": [],
    }


# ---------------------------------------------------------------------------
# PART 1 — GAP DETECTION TESTS
# ---------------------------------------------------------------------------


class TestUnstableCorrectionRegion:
    def test_negative_gain_detected(self):
        system_records = _make_system_records([
            ("chain", 5, "none", 0.0, 0.0, 0),
            ("chain", 5, "d4", 0.5, 0.3, -2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "unstable_correction_region" in gap_types

    def test_no_negative_gain_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "none", 0.0, 0.0, 0),
            ("chain", 5, "d4", 0.5, 0.6, 2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "unstable_correction_region" not in gap_types

    def test_evidence_contains_gains(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.5, 0.3, -1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        unstable = [g for g in gaps if g["gap_type"] == "unstable_correction_region"]
        assert len(unstable) == 1
        assert "gains" in unstable[0]["evidence"]


class TestWeakCompressionStructure:
    def test_all_modes_weak(self):
        system_records = _make_system_records([
            ("cycle", 10, "none", 0.0, 0.0, 0),
            ("cycle", 10, "d4", 0.05, 0.3, 1),
            ("cycle", 10, "d4+inv", 0.09, 0.4, 2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "weak_compression_structure" in gap_types

    def test_one_mode_strong_no_gap(self):
        system_records = _make_system_records([
            ("cycle", 10, "none", 0.0, 0.0, 0),
            ("cycle", 10, "d4", 0.5, 0.3, 1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "weak_compression_structure" not in gap_types


class TestModeDisagreement:
    def test_large_spread_detected(self):
        system_records = _make_system_records([
            ("cycle", 10, "none", 0.0, 0.0, 0),
            ("cycle", 10, "d4", 0.3, 0.1, 1),
            ("cycle", 10, "d4+inv", 0.4, 0.8, 3),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "mode_disagreement" in gap_types

    def test_small_spread_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.3, 0.5, 1),
            ("chain", 5, "d4+inv", 0.4, 0.6, 2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "mode_disagreement" not in gap_types

    def test_single_mode_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.3, 0.5, 1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "mode_disagreement" not in gap_types


class TestInvariantDependency:
    def test_large_delta_detected(self):
        system_records = _make_system_records([
            ("cycle", 10, "d4", 0.3, 0.1, 1),
            ("cycle", 10, "d4+inv", 0.4, 0.8, 3),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "invariant_dependency" in gap_types

    def test_small_delta_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.3, 0.5, 1),
            ("chain", 5, "d4+inv", 0.4, 0.6, 2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "invariant_dependency" not in gap_types

    def test_missing_modes_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "none", 0.0, 0.0, 0),
            ("chain", 5, "square", 0.3, 0.4, 1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "invariant_dependency" not in gap_types


class TestOvercorrectionPattern:
    def test_high_comp_low_stab_detected(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.8, 0.05, 1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "overcorrection_pattern" in gap_types

    def test_balanced_no_gap(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.5, 0.5, 2),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        gap_types = [g["gap_type"] for g in gaps]
        assert "overcorrection_pattern" not in gap_types


# ---------------------------------------------------------------------------
# PART 2 — INVARIANT MAPPING TESTS
# ---------------------------------------------------------------------------


class TestMapGapToInvariant:
    def test_all_mappings_exist(self):
        expected = {
            "unstable_correction_region": "local_stability_constraint",
            "weak_compression_structure": "equivalence_class_constraint",
            "mode_disagreement": "geometry_alignment_constraint",
            "invariant_dependency": "explicit_allowed_state_constraint",
            "overcorrection_pattern": "bounded_projection_constraint",
        }
        for gap_type, expected_inv in expected.items():
            result = map_gap_to_invariant(gap_type)
            assert result is not None
            assert result["suggested_invariant"] == expected_inv

    def test_unknown_gap_returns_none(self):
        assert map_gap_to_invariant("nonexistent_gap") is None

    def test_confidence_high_for_strong_signals(self):
        for gap_type in [
            "unstable_correction_region",
            "mode_disagreement",
            "invariant_dependency",
        ]:
            result = map_gap_to_invariant(gap_type)
            assert result["confidence"] == "high"

    def test_confidence_medium_for_borderline(self):
        for gap_type in [
            "weak_compression_structure",
            "overcorrection_pattern",
        ]:
            result = map_gap_to_invariant(gap_type)
            assert result["confidence"] == "medium"


class TestMapAllGapsToInvariants:
    def test_maps_multiple_gaps(self):
        gaps = [
            {"dfa_type": "chain", "n": 5, "gap_type": "unstable_correction_region", "evidence": {}},
            {"dfa_type": "cycle", "n": 10, "gap_type": "mode_disagreement", "evidence": {}},
        ]
        opps = map_all_gaps_to_invariants(gaps)
        assert len(opps) == 2
        assert opps[0]["suggested_invariant"] == "local_stability_constraint"
        assert opps[1]["suggested_invariant"] == "geometry_alignment_constraint"

    def test_preserves_dfa_info(self):
        gaps = [
            {"dfa_type": "chain", "n": 5, "gap_type": "overcorrection_pattern", "evidence": {}},
        ]
        opps = map_all_gaps_to_invariants(gaps)
        assert opps[0]["dfa_type"] == "chain"
        assert opps[0]["n"] == 5


# ---------------------------------------------------------------------------
# PART 3 — FULL PIPELINE TESTS
# ---------------------------------------------------------------------------


class TestRunStructuralDiagnostics:
    def test_full_pipeline_returns_expected_keys(self):
        data = [
            _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
            _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 4, 2),
            _make_raw_result("cycle", 10, "none", 0.0, 0.0, 0, 6, 6),
            _make_raw_result("cycle", 10, "d4+inv", 0.4, 0.5, 3, 6, 4),
        ]
        report = run_structural_diagnostics(data)
        assert "system_classes" in report
        assert "gaps" in report
        assert "invariant_opportunities" in report

    def test_pipeline_detects_gaps_in_problem_data(self):
        data = [
            _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
            _make_raw_result("chain", 5, "d4", 0.8, 0.05, -1, 4, 1),
        ]
        report = run_structural_diagnostics(data)
        gap_types = [g["gap_type"] for g in report["gaps"]]
        assert "unstable_correction_region" in gap_types
        assert "overcorrection_pattern" in gap_types


# ---------------------------------------------------------------------------
# PART 4 — DETERMINISM & SAFETY TESTS
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_gap_detection_deterministic(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.8, 0.05, -1),
            ("cycle", 10, "d4", 0.05, 0.1, 1),
            ("cycle", 10, "d4+inv", 0.09, 0.8, 3),
        ])
        gaps_a = detect_structural_gaps(system_records, _empty_diagnostics())
        gaps_b = detect_structural_gaps(system_records, _empty_diagnostics())
        assert gaps_a == gaps_b

    def test_full_pipeline_deterministic(self):
        data = [
            _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
            _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 4, 2),
        ]
        report_a = run_structural_diagnostics(data)
        report_b = run_structural_diagnostics(data)
        assert report_a == report_b

    def test_ordering_stable(self):
        system_records = _make_system_records([
            ("cycle", 10, "d4", 0.05, 0.1, -1),
            ("chain", 5, "d4", 0.8, 0.05, -1),
        ])
        gaps = detect_structural_gaps(system_records, _empty_diagnostics())
        # Chain comes before cycle alphabetically.
        dfa_types = [g["dfa_type"] for g in gaps]
        chain_idx = next(i for i, d in enumerate(dfa_types) if d == "chain")
        cycle_idx = next(i for i, d in enumerate(dfa_types) if d == "cycle")
        assert chain_idx < cycle_idx


class TestNoMutation:
    def test_gap_detection_no_input_mutation(self):
        system_records = _make_system_records([
            ("chain", 5, "d4", 0.8, 0.05, -1),
        ])
        original = copy.deepcopy(system_records)
        detect_structural_gaps(system_records, _empty_diagnostics())
        assert system_records == original

    def test_full_pipeline_no_input_mutation(self):
        data = [
            _make_raw_result("chain", 5, "none", 0.0, 0.0, 0, 4, 4),
            _make_raw_result("chain", 5, "d4", 0.5, 0.6, 2, 4, 2),
        ]
        original = copy.deepcopy(data)
        run_structural_diagnostics(data)
        assert data == original


# ---------------------------------------------------------------------------
# PART 5 — PRINT LAYER TESTS
# ---------------------------------------------------------------------------


class TestPrintStructuralReport:
    def test_returns_string(self):
        report = {
            "system_classes": [
                {"dfa_type": "cycle", "n": 10, "system_class": "cycle_like"},
            ],
            "gaps": [
                {"dfa_type": "cycle", "n": 10, "gap_type": "mode_disagreement", "evidence": {}},
            ],
            "invariant_opportunities": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "gap_type": "mode_disagreement",
                    "suggested_invariant": "geometry_alignment_constraint",
                    "confidence": "high",
                },
            ],
        }
        text = print_structural_report(report)
        assert isinstance(text, str)
        assert "cycle" in text
        assert "mode_disagreement" in text
        assert "geometry_alignment_constraint" in text
        assert "high" in text

    def test_empty_report(self):
        report = {"system_classes": [], "gaps": [], "invariant_opportunities": []}
        text = print_structural_report(report)
        assert "No structural gaps detected" in text


# ---------------------------------------------------------------------------
# EDGE CASES
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_system_records(self):
        gaps = detect_structural_gaps({}, _empty_diagnostics())
        assert gaps == []

    def test_empty_gaps_to_invariants(self):
        opps = map_all_gaps_to_invariants([])
        assert opps == []
