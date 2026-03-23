"""Tests for deterministic conjecture engine (v97.1.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.conjecture_engine import (
    attach_tests,
    extract_features,
    generate_conjectures,
    print_conjectures,
    run_conjecture_engine,
    score_conjecture,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_system(
    system_class="cycle",
    mode="d4",
    friction_score=1.0,
    oscillation_ratio=0.2,
    churn_score=0.3,
    stability_efficiency=0.6,
    core_invariants=None,
    law_matches=None,
):
    """Build a single system record for testing."""
    return {
        "system_class": system_class,
        "mode": mode,
        "friction_score": friction_score,
        "oscillation_ratio": oscillation_ratio,
        "churn_score": churn_score,
        "stability_efficiency": stability_efficiency,
        "core_invariants": core_invariants if core_invariants is not None else [],
        "law_matches": law_matches if law_matches is not None else [],
    }


def _make_high_oscillation(**kwargs):
    """System with high oscillation ratio."""
    defaults = {"oscillation_ratio": 0.8, "system_class": "cycle"}
    defaults.update(kwargs)
    return _make_system(**defaults)


def _make_high_churn(**kwargs):
    """System with high churn score."""
    defaults = {"churn_score": 0.9, "system_class": "chain"}
    defaults.update(kwargs)
    return _make_system(**defaults)


def _make_high_friction(**kwargs):
    """System with high friction score."""
    defaults = {"friction_score": 3.0, "system_class": "tree"}
    defaults.update(kwargs)
    return _make_system(**defaults)


def _make_multi_stage_high_friction(**kwargs):
    """Multi-stage system with high friction."""
    defaults = {
        "mode": "d4>e8_like",
        "friction_score": 3.5,
        "system_class": "cycle",
    }
    defaults.update(kwargs)
    return _make_system(**defaults)


# ---------------------------------------------------------------------------
# PART 1 — FEATURE EXTRACTION TESTS
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    """Tests for extract_features."""

    def test_extract_from_list(self):
        data = [_make_system(system_class="cycle", mode="d4")]
        features = extract_features(data)
        assert len(features) == 1
        assert features[0]["system_class"] == "cycle"
        assert features[0]["best_mode"] == "d4"

    def test_extract_from_dict_systems(self):
        data = {"systems": [_make_system()]}
        features = extract_features(data)
        assert len(features) == 1

    def test_extract_from_dict_candidates(self):
        data = {"candidates": [_make_system()]}
        features = extract_features(data)
        assert len(features) == 1

    def test_extract_from_dict_groups(self):
        data = {"groups": [{"best": _make_system(system_class="cycle")}]}
        features = extract_features(data)
        assert len(features) == 1
        assert features[0]["system_class"] == "cycle"

    def test_empty_input(self):
        assert extract_features([]) == []
        assert extract_features({}) == []
        assert extract_features(None) == []

    def test_multi_stage_detection(self):
        data = [_make_system(mode="d4>e8_like")]
        features = extract_features(data)
        assert features[0]["is_multi_stage"] is True

    def test_single_stage_detection(self):
        data = [_make_system(mode="d4")]
        features = extract_features(data)
        assert features[0]["is_multi_stage"] is False

    def test_sorted_output(self):
        data = [
            _make_system(system_class="tree", mode="d4"),
            _make_system(system_class="chain", mode="square"),
            _make_system(system_class="cycle", mode="d4"),
        ]
        features = extract_features(data)
        classes = [f["system_class"] for f in features]
        assert classes == ["chain", "cycle", "tree"]

    def test_no_mutation(self):
        data = [_make_system()]
        data_copy = copy.deepcopy(data)
        extract_features(data)
        assert data == data_copy

    def test_fallback_dfa_type(self):
        data = [{"dfa_type": "chain", "mode": "d4"}]
        features = extract_features(data)
        assert features[0]["system_class"] == "chain"


# ---------------------------------------------------------------------------
# PART 2 — CONJECTURE GENERATION TESTS
# ---------------------------------------------------------------------------


class TestGenerateConjectures:
    """Tests for generate_conjectures."""

    def test_rule_a_oscillation(self):
        features = [_extract(_make_high_oscillation())]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "oscillation_reduction" in types

    def test_rule_a_below_threshold(self):
        features = [_extract(_make_system(oscillation_ratio=0.3))]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "oscillation_reduction" not in types

    def test_rule_b_churn(self):
        features = [_extract(_make_high_churn())]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "churn_reduction" in types

    def test_rule_c_friction(self):
        features = [_extract(_make_high_friction())]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "friction_reduction" in types

    def test_rule_d_invariants(self):
        features = [
            _extract(_make_system(
                system_class="cycle",
                core_invariants=["symmetry"],
            )),
            _extract(_make_system(
                system_class="chain",
                core_invariants=["symmetry"],
            )),
        ]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "invariant_generalization" in types

    def test_rule_d_single_class_no_conjecture(self):
        features = [
            _extract(_make_system(
                system_class="cycle",
                core_invariants=["symmetry"],
            )),
        ]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "invariant_generalization" not in types

    def test_rule_e_hierarchy(self):
        features = [_extract(_make_multi_stage_high_friction())]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "hierarchy_optimization" in types

    def test_rule_e_single_stage_no_conjecture(self):
        features = [_extract(_make_system(mode="d4", friction_score=3.0))]
        conjectures = generate_conjectures(features)
        types = [c["type"] for c in conjectures]
        assert "hierarchy_optimization" not in types

    def test_empty_features(self):
        assert generate_conjectures([]) == []

    def test_conjecture_structure(self):
        features = [_extract(_make_high_oscillation())]
        conjectures = generate_conjectures(features)
        for c in conjectures:
            assert "statement" in c
            assert "type" in c
            assert "conditions" in c
            assert isinstance(c["statement"], str)
            assert isinstance(c["conditions"], dict)

    def test_sorted_output(self):
        features = [_extract(_make_high_oscillation())]
        conjectures = generate_conjectures(features)
        keys = [(c["type"], c["statement"]) for c in conjectures]
        assert keys == sorted(keys)

    def test_no_mutation(self):
        features = [_extract(_make_high_oscillation())]
        features_copy = copy.deepcopy(features)
        generate_conjectures(features)
        assert features == features_copy


# ---------------------------------------------------------------------------
# PART 3 — TEST ATTACHMENT TESTS
# ---------------------------------------------------------------------------


class TestAttachTests:
    """Tests for attach_tests."""

    def test_known_type_gets_template(self):
        conj = {"type": "oscillation_reduction", "statement": "test"}
        result = attach_tests(conj)
        assert "test" in result
        assert result["test"]["metric"] == "oscillation_ratio"

    def test_unknown_type_gets_fallback(self):
        conj = {"type": "unknown_type", "statement": "test"}
        result = attach_tests(conj)
        assert result["test"]["method"] == "manual verification required"

    def test_test_structure(self):
        conj = {"type": "churn_reduction", "statement": "test"}
        result = attach_tests(conj)
        test = result["test"]
        assert "method" in test
        assert "compare" in test
        assert "metric" in test
        assert "expected" in test

    def test_no_mutation(self):
        conj = {"type": "oscillation_reduction", "statement": "test"}
        conj_copy = copy.deepcopy(conj)
        attach_tests(conj)
        assert conj == conj_copy

    def test_preserves_original_fields(self):
        conj = {
            "type": "friction_reduction",
            "statement": "original",
            "conditions": {"a": 1},
        }
        result = attach_tests(conj)
        assert result["statement"] == "original"
        assert result["conditions"] == {"a": 1}


# ---------------------------------------------------------------------------
# PART 4 — SCORING TESTS
# ---------------------------------------------------------------------------


class TestScoreConjecture:
    """Tests for score_conjecture."""

    def test_law_alignment_adds_two(self):
        conj = {"type": "oscillation_reduction", "conditions": {}}
        features = [_extract(_make_system(
            law_matches=["oscillation_law"],
        ))]
        score = score_conjecture(conj, features)
        assert score >= 2

    def test_dynamics_alignment_adds_one(self):
        conj = {
            "type": "oscillation_reduction",
            "conditions": {"system_class": "cycle"},
        }
        features = [_extract(_make_high_oscillation(system_class="cycle"))]
        score = score_conjecture(conj, features)
        assert score >= 1

    def test_no_alignment_zero(self):
        conj = {
            "type": "oscillation_reduction",
            "conditions": {"system_class": "cycle"},
        }
        features = [_extract(_make_system(
            system_class="cycle", oscillation_ratio=0.1,
        ))]
        score = score_conjecture(conj, features)
        assert score == 0

    def test_deterministic(self):
        conj = {
            "type": "churn_reduction",
            "conditions": {"system_class": "chain"},
        }
        features = [_extract(_make_high_churn(system_class="chain"))]
        s1 = score_conjecture(conj, features)
        s2 = score_conjecture(conj, features)
        assert s1 == s2

    def test_returns_int(self):
        conj = {"type": "friction_reduction", "conditions": {}}
        score = score_conjecture(conj, [])
        assert isinstance(score, int)

    def test_no_mutation(self):
        conj = {"type": "oscillation_reduction", "conditions": {}}
        features = [_extract(_make_high_oscillation())]
        conj_copy = copy.deepcopy(conj)
        features_copy = copy.deepcopy(features)
        score_conjecture(conj, features)
        assert conj == conj_copy
        assert features == features_copy


# ---------------------------------------------------------------------------
# PART 5 — PIPELINE TESTS
# ---------------------------------------------------------------------------


class TestRunConjectureEngine:
    """Tests for run_conjecture_engine."""

    def test_basic_pipeline(self):
        data = [_make_high_oscillation()]
        result = run_conjecture_engine(data)
        assert "conjectures" in result
        assert "summary" in result
        assert result["summary"]["total"] > 0

    def test_empty_input(self):
        result = run_conjecture_engine([])
        assert result["conjectures"] == []
        assert result["summary"]["total"] == 0

    def test_conjectures_have_all_fields(self):
        data = [_make_high_oscillation()]
        result = run_conjecture_engine(data)
        for conj in result["conjectures"]:
            assert "statement" in conj
            assert "type" in conj
            assert "conditions" in conj
            assert "test" in conj
            assert "score" in conj
            assert "confidence" in conj
            assert conj["confidence"] in ("high", "medium", "low")

    def test_sorted_by_score_desc(self):
        data = [
            _make_high_oscillation(),
            _make_high_churn(),
            _make_high_friction(),
        ]
        result = run_conjecture_engine(data)
        scores = [c["score"] for c in result["conjectures"]]
        assert scores == sorted(scores, reverse=True)

    def test_summary_counts(self):
        data = [_make_high_oscillation()]
        result = run_conjecture_engine(data)
        s = result["summary"]
        assert s["total"] == s["high"] + s["medium"] + s["low"]

    def test_deterministic_output(self):
        data = [
            _make_high_oscillation(),
            _make_high_churn(),
        ]
        r1 = run_conjecture_engine(data)
        r2 = run_conjecture_engine(data)
        assert r1 == r2

    def test_no_mutation(self):
        data = [_make_high_oscillation()]
        data_copy = copy.deepcopy(data)
        run_conjecture_engine(data)
        assert data == data_copy

    def test_multi_rule_activation(self):
        # System triggers multiple rules.
        data = [_make_system(
            oscillation_ratio=0.8,
            churn_score=0.9,
            friction_score=3.0,
        )]
        result = run_conjecture_engine(data)
        types = {c["type"] for c in result["conjectures"]}
        assert "oscillation_reduction" in types
        assert "churn_reduction" in types
        assert "friction_reduction" in types

    def test_dict_input_with_groups(self):
        data = {
            "groups": [
                {"best": _make_high_oscillation()},
            ],
        }
        result = run_conjecture_engine(data)
        assert result["summary"]["total"] > 0


# ---------------------------------------------------------------------------
# PART 6 — PRINT TESTS
# ---------------------------------------------------------------------------


class TestPrintConjectures:
    """Tests for print_conjectures."""

    def test_basic_output(self):
        report = {
            "conjectures": [
                {
                    "statement": "IF oscillation is high THEN reduce",
                    "type": "oscillation_reduction",
                    "conditions": {},
                    "test": {
                        "method": "run benchmark",
                        "compare": ["square", "d4"],
                        "metric": "oscillation_ratio",
                        "expected": "decrease",
                    },
                    "score": 2,
                    "confidence": "medium",
                },
            ],
            "summary": {"total": 1, "high": 0, "medium": 1, "low": 0},
        }
        output = print_conjectures(report)
        assert "=== Conjectures ===" in output
        assert "[medium]" in output
        assert "oscillation" in output
        assert "metric: oscillation_ratio" in output

    def test_empty_report(self):
        output = print_conjectures({"conjectures": [], "summary": {}})
        assert "No conjectures generated" in output

    def test_summary_in_output(self):
        report = {
            "conjectures": [{
                "statement": "test",
                "type": "test",
                "conditions": {},
                "test": {"method": "m", "compare": [], "metric": "m",
                         "expected": "e"},
                "score": 1,
                "confidence": "medium",
            }],
            "summary": {"total": 1, "high": 0, "medium": 1, "low": 0},
        }
        output = print_conjectures(report)
        assert "total: 1" in output
        assert "medium: 1" in output

    def test_deterministic_output(self):
        report = run_conjecture_engine([_make_high_oscillation()])
        o1 = print_conjectures(report)
        o2 = print_conjectures(report)
        assert o1 == o2


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------


def _extract(record):
    """Shortcut to extract features from a single record."""
    return extract_features([record])[0]
