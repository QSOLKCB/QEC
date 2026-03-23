"""Tests for law-guided scoring (v97.0.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.law_control import (
    apply_law_prior,
    compute_law_score,
    match_laws,
    print_law_control,
    run_law_control,
    select_with_laws,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_candidate(
    dfa_type="cycle",
    n=10,
    mode="d4",
    stability_efficiency=0.6,
    compression_efficiency=0.3,
    friction_score=1.0,
    regime="adaptive",
    system_class="",
    efficiency=None,
):
    """Build a single candidate record for testing."""
    rec = {
        "dfa_type": dfa_type,
        "n": n,
        "mode": mode,
        "stability_efficiency": stability_efficiency,
        "compression_efficiency": compression_efficiency,
        "friction_score": friction_score,
        "regime": regime,
    }
    if system_class:
        rec["system_class"] = system_class
    if efficiency is not None:
        rec["efficiency"] = efficiency
    else:
        stab = stability_efficiency
        if stab > 0.0:
            rec["efficiency"] = round(stab / (1.0 + friction_score), 6)
        else:
            rec["efficiency"] = 0.0
    return rec


def _make_law(
    law_type="class_invariant_law",
    system_class="cycle",
    mode=None,
    effect="tends to improve correction",
    confidence="strong",
    invariant_type="symmetry",
):
    """Build a single law record for testing."""
    cond = {"system_class": system_class}
    if mode:
        cond["hierarchical_mode"] = mode
    concl = {"invariant_type": invariant_type, "effect": effect}
    return {
        "law_type": law_type,
        "condition": cond,
        "conclusion": concl,
        "evidence": {"support_count": 5, "improved_ratio": 0.8},
        "confidence": confidence,
        "rank": 1,
    }


# ---------------------------------------------------------------------------
# PART 1 — LAW MATCHING TESTS
# ---------------------------------------------------------------------------


class TestMatchLaws:
    """Tests for match_laws."""

    def test_match_by_system_class(self):
        record = _make_candidate(dfa_type="cycle")
        laws = [_make_law(system_class="cycle")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 1
        assert len(result["contradicting"]) == 0

    def test_no_match_different_class(self):
        record = _make_candidate(dfa_type="chain")
        laws = [_make_law(system_class="cycle")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 0
        assert len(result["contradicting"]) == 0

    def test_match_with_mode_filter(self):
        record = _make_candidate(dfa_type="cycle", mode="d4")
        laws = [_make_law(system_class="cycle", mode="d4")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 1

    def test_no_match_wrong_mode(self):
        record = _make_candidate(dfa_type="cycle", mode="d4")
        laws = [_make_law(system_class="cycle", mode="e8_like")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 0

    def test_contradiction_detected(self):
        record = _make_candidate(dfa_type="cycle")
        laws = [_make_law(system_class="cycle", effect="not recommended")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 0
        assert len(result["contradicting"]) == 1

    def test_contradiction_avoid(self):
        record = _make_candidate(dfa_type="cycle")
        laws = [_make_law(system_class="cycle", effect="avoid this mode")]
        result = match_laws(record, laws)
        assert len(result["contradicting"]) == 1

    def test_empty_laws(self):
        record = _make_candidate()
        result = match_laws(record, [])
        assert result == {"matched": [], "contradicting": []}

    def test_multiple_matches(self):
        record = _make_candidate(dfa_type="cycle")
        laws = [
            _make_law(system_class="cycle", invariant_type="symmetry"),
            _make_law(system_class="cycle", invariant_type="parity"),
        ]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 2

    def test_explicit_system_class(self):
        record = _make_candidate(dfa_type="chain", system_class="cycle")
        laws = [_make_law(system_class="cycle")]
        result = match_laws(record, laws)
        assert len(result["matched"]) == 1

    def test_no_mutation_of_inputs(self):
        record = _make_candidate(dfa_type="cycle")
        laws = [_make_law(system_class="cycle")]
        record_copy = copy.deepcopy(record)
        laws_copy = copy.deepcopy(laws)
        match_laws(record, laws)
        assert record == record_copy
        assert laws == laws_copy


# ---------------------------------------------------------------------------
# PART 2 — LAW SCORE TESTS
# ---------------------------------------------------------------------------


class TestComputeLawScore:
    """Tests for compute_law_score."""

    def test_strong_match_scores_two(self):
        matched = [_make_law(confidence="strong")]
        assert compute_law_score(matched, []) == 2

    def test_moderate_match_scores_one(self):
        matched = [_make_law(confidence="moderate")]
        assert compute_law_score(matched, []) == 1

    def test_contradiction_scores_minus_one(self):
        contradicting = [_make_law()]
        assert compute_law_score([], contradicting) == -1

    def test_mixed_scoring(self):
        matched = [
            _make_law(confidence="strong"),
            _make_law(confidence="moderate"),
        ]
        contradicting = [_make_law()]
        # +2 + 1 - 1 = 2
        assert compute_law_score(matched, contradicting) == 2

    def test_empty_inputs(self):
        assert compute_law_score([], []) == 0

    def test_all_contradictions(self):
        contradicting = [_make_law(), _make_law(), _make_law()]
        assert compute_law_score([], contradicting) == -3


# ---------------------------------------------------------------------------
# PART 3 — LAW PRIOR TESTS
# ---------------------------------------------------------------------------


class TestApplyLawPrior:
    """Tests for apply_law_prior."""

    def test_positive_law_score(self):
        record = _make_candidate(efficiency=0.5)
        result = apply_law_prior(record, 2)
        assert result["effective_score"] == round(0.5 + 2 * 0.1, 6)
        assert result["law_score"] == 2

    def test_negative_law_score(self):
        record = _make_candidate(efficiency=0.5)
        result = apply_law_prior(record, -1)
        assert result["effective_score"] == round(0.5 + (-1) * 0.1, 6)

    def test_zero_law_score(self):
        record = _make_candidate(efficiency=0.5)
        result = apply_law_prior(record, 0)
        assert result["effective_score"] == 0.5

    def test_no_mutation(self):
        record = _make_candidate(efficiency=0.5)
        original = copy.deepcopy(record)
        apply_law_prior(record, 2)
        assert record == original

    def test_preserves_original_fields(self):
        record = _make_candidate(dfa_type="cycle", mode="d4", efficiency=0.5)
        result = apply_law_prior(record, 1)
        assert result["dfa_type"] == "cycle"
        assert result["mode"] == "d4"
        assert result["efficiency"] == 0.5


# ---------------------------------------------------------------------------
# PART 4 — DECISION FUNCTION TESTS
# ---------------------------------------------------------------------------


class TestSelectWithLaws:
    """Tests for select_with_laws."""

    def test_best_selected_by_effective_score(self):
        records = [
            _make_candidate(dfa_type="cycle", mode="d4", efficiency=0.3),
            _make_candidate(dfa_type="cycle", mode="square", efficiency=0.5),
        ]
        laws = [_make_law(system_class="cycle", confidence="strong")]
        result = select_with_laws(records, laws)
        # Both match same law (+2), so effective_score = eff + 0.2
        # square: 0.5+0.2=0.7 > d4: 0.3+0.2=0.5
        assert result["best"]["mode"] == "square"

    def test_law_score_can_change_ranking(self):
        records = [
            _make_candidate(
                dfa_type="cycle", mode="d4",
                system_class="cycle", efficiency=0.48,
            ),
            _make_candidate(
                dfa_type="chain", mode="square",
                system_class="chain", efficiency=0.50,
            ),
        ]
        # Law only matches cycle → +2 → effective = 0.48 + 0.2 = 0.68
        # chain has no law match → effective = 0.50
        laws = [_make_law(system_class="cycle", confidence="strong")]
        result = select_with_laws(records, laws)
        assert result["best"]["mode"] == "d4"
        assert result["best"]["law_score"] == 2

    def test_empty_records(self):
        result = select_with_laws([], [])
        assert result["best"] is None
        assert result["trace"] == []

    def test_trace_includes_all(self):
        records = [
            _make_candidate(mode="d4", efficiency=0.3),
            _make_candidate(mode="square", efficiency=0.5),
        ]
        result = select_with_laws(records, [])
        assert len(result["trace"]) == 2

    def test_deterministic_tiebreak(self):
        records = [
            _make_candidate(mode="square", efficiency=0.5),
            _make_candidate(mode="d4", efficiency=0.5),
        ]
        result = select_with_laws(records, [])
        # Same effective_score, same stab/comp/friction → mode alpha
        assert result["best"]["mode"] == "d4"

    def test_friction_tiebreak(self):
        records = [
            _make_candidate(mode="d4", efficiency=0.5, friction_score=2.0),
            _make_candidate(mode="d4x", efficiency=0.5, friction_score=1.0),
        ]
        result = select_with_laws(records, [])
        # Same effective_score → lower friction wins
        assert result["best"]["mode"] == "d4x"


# ---------------------------------------------------------------------------
# PART 5 — PIPELINE TESTS
# ---------------------------------------------------------------------------


class TestRunLawControl:
    """Tests for run_law_control."""

    def test_basic_pipeline(self):
        data = {
            "friction_control": {
                "candidates": [
                    _make_candidate(dfa_type="cycle", n=10, mode="d4"),
                    _make_candidate(dfa_type="cycle", n=10, mode="square",
                                    efficiency=0.4),
                ],
            },
            "law_extraction": {
                "laws": [_make_law(system_class="cycle")],
            },
        }
        result = run_law_control(data)
        assert result["summary"]["total_groups"] == 1
        assert result["groups"][0]["best"] is not None

    def test_groups_by_dfa_and_n(self):
        data = {
            "candidates": [
                _make_candidate(dfa_type="cycle", n=10),
                _make_candidate(dfa_type="cycle", n=20),
                _make_candidate(dfa_type="chain", n=10),
            ],
            "laws": [],
        }
        result = run_law_control(data)
        assert result["summary"]["total_groups"] == 3

    def test_empty_data(self):
        result = run_law_control({})
        assert result["groups"] == []
        assert result["summary"]["total_groups"] == 0

    def test_non_dict_data(self):
        result = run_law_control("invalid")
        assert result["groups"] == []

    def test_candidates_without_efficiency(self):
        cand = {
            "dfa_type": "cycle",
            "n": 10,
            "mode": "d4",
            "stability_efficiency": 0.6,
            "compression_efficiency": 0.3,
            "friction_score": 1.0,
            "regime": "adaptive",
        }
        data = {"candidates": [cand], "laws": []}
        result = run_law_control(data)
        best = result["groups"][0]["best"]
        assert best["efficiency"] == round(0.6 / 2.0, 6)

    def test_no_mutation_of_input(self):
        data = {
            "candidates": [
                _make_candidate(dfa_type="cycle", n=10, mode="d4"),
            ],
            "laws": [_make_law(system_class="cycle")],
        }
        data_copy = copy.deepcopy(data)
        run_law_control(data)
        assert data == data_copy

    def test_deterministic_output(self):
        data = {
            "candidates": [
                _make_candidate(dfa_type="cycle", n=10, mode="d4",
                                efficiency=0.3),
                _make_candidate(dfa_type="cycle", n=10, mode="square",
                                efficiency=0.5),
            ],
            "laws": [_make_law(system_class="cycle")],
        }
        r1 = run_law_control(data)
        r2 = run_law_control(data)
        assert r1 == r2

    def test_law_extraction_nested(self):
        data = {
            "candidates": [_make_candidate(dfa_type="cycle", n=10)],
            "law_extraction": {
                "laws": [_make_law(system_class="cycle")],
            },
        }
        result = run_law_control(data)
        assert result["summary"]["law_count"] == 1

    def test_rulebook_nested(self):
        data = {
            "candidates": [_make_candidate(dfa_type="cycle", n=10)],
            "rulebook": {
                "laws": [_make_law(system_class="cycle")],
            },
        }
        result = run_law_control(data)
        assert result["summary"]["law_count"] == 1


# ---------------------------------------------------------------------------
# PART 6 — PRINT TESTS
# ---------------------------------------------------------------------------


class TestPrintLawControl:
    """Tests for print_law_control."""

    def test_basic_output(self):
        report = {
            "groups": [
                {
                    "dfa_type": "cycle",
                    "n": 10,
                    "best": {
                        "mode": "d4",
                        "efficiency": 0.62,
                        "law_score": 2,
                        "effective_score": 0.82,
                        "matched_laws": 1,
                        "contradicting_laws": 0,
                    },
                    "trace": [],
                },
            ],
            "summary": {
                "total_groups": 1,
                "total_candidates": 2,
                "law_count": 3,
            },
        }
        output = print_law_control(report)
        assert "DFA: cycle (n=10)" in output
        assert "best_mode: d4" in output
        assert "efficiency: 0.62" in output
        assert "law_score: +2" in output
        assert "strong law support" in output

    def test_empty_report(self):
        output = print_law_control({"groups": [], "summary": {}})
        assert "No candidates" in output

    def test_negative_law_score(self):
        report = {
            "groups": [
                {
                    "dfa_type": "chain",
                    "n": None,
                    "best": {
                        "mode": "square",
                        "efficiency": 0.3,
                        "law_score": -2,
                        "effective_score": 0.1,
                        "matched_laws": 0,
                        "contradicting_laws": 2,
                    },
                    "trace": [],
                },
            ],
            "summary": {"total_groups": 1, "total_candidates": 1,
                         "law_count": 2},
        }
        output = print_law_control(report)
        assert "DFA: chain" in output
        assert "law_score: -2" in output
        assert "contradictions" in output

    def test_no_n_in_output(self):
        report = {
            "groups": [
                {
                    "dfa_type": "chain",
                    "n": None,
                    "best": {
                        "mode": "d4",
                        "efficiency": 0.5,
                        "law_score": 0,
                        "effective_score": 0.5,
                        "matched_laws": 0,
                        "contradicting_laws": 0,
                    },
                    "trace": [],
                },
            ],
            "summary": {"total_groups": 1, "total_candidates": 1,
                         "law_count": 0},
        }
        output = print_law_control(report)
        assert "DFA: chain" in output
        assert "(n=" not in output
