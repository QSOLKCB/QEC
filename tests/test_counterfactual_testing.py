"""Tests for counterfactual and adversarial testing (v105.2.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.counterfactual_testing import (
    _ADVERSARIAL_STRENGTHS,
    _FRAGILITY_THRESHOLD,
    evaluate_law_violation,
    generate_counterfactuals,
    refine_diagnosis_from_counterfactuals,
    run_law_stress_test,
    select_target_laws,
    update_law_robustness,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_law(name="law_0", stability_score=0.8, count=5, conditions=None):
    law = {
        "name": name,
        "stability_score": stability_score,
        "count": count,
    }
    if conditions is not None:
        law["conditions"] = conditions
    else:
        law["conditions"] = [
            {"metric": "stability", "operator": "gte", "value": 0.5},
        ]
    return law


def _make_influence_map(contested=False):
    if contested:
        nodes = {"n0": {
            "influence_score": 0.5, "instability_pressure": 0.7,
            "control_sensitivity": 0.8, "stability": 0.4,
        }}
    else:
        nodes = {"n0": {
            "influence_score": 0.5, "instability_pressure": 0.2,
            "control_sensitivity": 0.3, "stability": 0.7,
        }}
    return {"nodes": nodes, "summary": {}, "influence_entropy": 0.0}


def _make_diagnosis(modes=None):
    if modes is None:
        modes = [("healthy_convergence", 0.6)]
    return {
        "ranked_diagnoses": [
            {"failure_mode": m, "score": s} for m, s in modes
        ],
    }


# ---------------------------------------------------------------------------
# TARGET LAW SELECTION TESTS
# ---------------------------------------------------------------------------


class TestSelectTargetLaws:

    def test_empty_inputs(self):
        assert select_target_laws({}, []) == []

    def test_selects_fragile_laws(self):
        laws = [
            _make_law("robust", stability_score=0.9),
            _make_law("fragile", stability_score=0.3),
        ]
        result = select_target_laws({}, laws)
        assert "fragile" in result

    def test_selects_emerging_laws(self):
        laws = [
            _make_law("established", stability_score=0.8, count=10),
            _make_law("emerging", stability_score=0.8, count=2),
        ]
        result = select_target_laws({}, laws)
        assert "emerging" in result

    def test_selects_drifting_invariants(self):
        registry = {"drifting": {"break_count": 5, "count": 10}}
        laws = [_make_law("drifting", stability_score=0.8, count=10)]
        result = select_target_laws(registry, laws)
        assert "drifting" in result

    def test_deterministic(self):
        laws = [_make_law(f"law_{i}", stability_score=0.4) for i in range(5)]
        r1 = select_target_laws({}, laws)
        r2 = select_target_laws({}, laws)
        assert r1 == r2

    def test_no_mutation(self):
        laws = [_make_law("fragile", stability_score=0.3)]
        registry = {"fragile": {"break_count": 2, "count": 5}}
        laws_copy = copy.deepcopy(laws)
        reg_copy = copy.deepcopy(registry)
        select_target_laws(registry, laws)
        assert laws == laws_copy
        assert registry == reg_copy


# ---------------------------------------------------------------------------
# COUNTERFACTUAL GENERATION TESTS
# ---------------------------------------------------------------------------


class TestGenerateCounterfactuals:

    def test_empty_law(self):
        result = generate_counterfactuals({}, _make_influence_map())
        assert result == []

    def test_generates_probes_per_condition(self):
        law = _make_law(conditions=[
            {"metric": "stability", "operator": "gte", "value": 0.5},
            {"metric": "convergence", "operator": "gt", "value": 0.3},
        ])
        result = generate_counterfactuals(law, _make_influence_map())
        # Each condition gets len(_ADVERSARIAL_STRENGTHS) violate probes + 1 boundary
        expected = 2 * (len(_ADVERSARIAL_STRENGTHS) + 1)
        assert len(result) == expected

    def test_probe_structure(self):
        law = _make_law()
        result = generate_counterfactuals(law, _make_influence_map())
        for probe in result:
            assert "target_condition" in probe
            assert "perturbation_strength" in probe
            assert "perturbation_direction" in probe
            assert "target_region" in probe
            assert probe["perturbation_direction"] in ("violate", "boundary")

    def test_escalating_strengths(self):
        law = _make_law()
        result = generate_counterfactuals(law, _make_influence_map())
        violate_probes = [p for p in result if p["perturbation_direction"] == "violate"]
        strengths = [p["perturbation_strength"] for p in violate_probes]
        assert strengths == sorted(strengths)  # Should be in escalating order

    def test_region_awareness_contested(self):
        law = _make_law()
        result = generate_counterfactuals(law, _make_influence_map(contested=True))
        for probe in result:
            assert probe["target_region"] == "contested"

    def test_determinism(self):
        law = _make_law()
        imap = _make_influence_map()
        r1 = generate_counterfactuals(law, imap)
        r2 = generate_counterfactuals(law, imap)
        assert r1 == r2


# ---------------------------------------------------------------------------
# LAW VIOLATION EVALUATION TESTS
# ---------------------------------------------------------------------------


class TestEvaluateLawViolation:

    def test_no_violation(self):
        law = _make_law(conditions=[
            {"metric": "stability", "operator": "gte", "value": 0.5},
        ])
        before = {"stability": 0.6}
        after = {"stability": 0.7}
        result = evaluate_law_violation(before, after, law)
        assert result["violated"] is False
        assert result["severity"] == 0.0
        assert result["confidence_shift"] > 0.0

    def test_violation_detected(self):
        law = _make_law(conditions=[
            {"metric": "stability", "operator": "gte", "value": 0.5},
        ])
        before = {"stability": 0.6}
        after = {"stability": 0.3}
        result = evaluate_law_violation(before, after, law)
        assert result["violated"] is True
        assert result["severity"] > 0.0
        assert result["confidence_shift"] < 0.0
        assert len(result["violated_conditions"]) == 1

    def test_all_operators(self):
        ops = [
            ("gt", 0.5, 0.6, False),   # 0.6 > 0.5 → holds
            ("gt", 0.5, 0.4, True),     # 0.4 > 0.5 → violated
            ("gte", 0.5, 0.5, False),
            ("lt", 0.5, 0.3, False),
            ("lt", 0.5, 0.7, True),
            ("lte", 0.5, 0.5, False),
            ("eq", 0.5, 0.5, False),
            ("neq", 0.5, 0.5, True),
        ]
        for op, threshold, after_val, should_violate in ops:
            law = _make_law(conditions=[
                {"metric": "x", "operator": op, "value": threshold},
            ])
            result = evaluate_law_violation({}, {"x": after_val}, law)
            assert result["violated"] == should_violate, (
                f"op={op}, threshold={threshold}, after={after_val}"
            )

    def test_determinism(self):
        law = _make_law()
        before = {"stability": 0.6}
        after = {"stability": 0.3}
        r1 = evaluate_law_violation(before, after, law)
        r2 = evaluate_law_violation(before, after, law)
        assert r1 == r2

    def test_no_mutation(self):
        law = _make_law()
        before = {"stability": 0.6}
        after = {"stability": 0.3}
        law_copy = copy.deepcopy(law)
        before_copy = copy.deepcopy(before)
        after_copy = copy.deepcopy(after)
        evaluate_law_violation(before, after, law)
        assert law == law_copy
        assert before == before_copy
        assert after == after_copy


# ---------------------------------------------------------------------------
# STRESS TEST PIPELINE TESTS
# ---------------------------------------------------------------------------


class TestRunLawStressTest:

    def test_empty_inputs(self):
        result = run_law_stress_test([], {}, [])
        assert result["law_results"] == {}
        assert result["summary"]["total_laws_tested"] == 0

    def test_single_law_no_violation(self):
        law = _make_law(conditions=[
            {"metric": "stability", "operator": "gte", "value": 0.3},
        ])
        runs = [
            {"stability": 0.5},
            {"stability": 0.6},
            {"stability": 0.7},
            {"stability": 0.8},
        ]
        result = run_law_stress_test(runs, {}, [law])
        assert "law_0" in result["law_results"]
        assert result["law_results"]["law_0"]["violations"] == 0

    def test_law_with_violations(self):
        law = _make_law(conditions=[
            {"metric": "stability", "operator": "gte", "value": 0.5},
        ])
        runs = [
            {"stability": 0.6},
            {"stability": 0.3},  # violates
            {"stability": 0.7},
        ]
        result = run_law_stress_test(runs, {}, [law])
        assert result["law_results"]["law_0"]["violations"] > 0

    def test_summary_counts(self):
        laws = [
            _make_law("robust", conditions=[
                {"metric": "stability", "operator": "gte", "value": 0.0},
            ]),
            _make_law("fragile", conditions=[
                {"metric": "stability", "operator": "gte", "value": 0.9},
            ]),
        ]
        runs = [{"stability": 0.5 + i * 0.1} for i in range(5)]
        result = run_law_stress_test(runs, {}, laws)
        assert result["summary"]["total_laws_tested"] == 2

    def test_determinism(self):
        law = _make_law()
        runs = [{"stability": 0.5 + i * 0.1} for i in range(4)]
        r1 = run_law_stress_test(runs, {}, [law])
        r2 = run_law_stress_test(runs, {}, [law])
        assert r1 == r2


# ---------------------------------------------------------------------------
# LAW ROBUSTNESS UPDATE TESTS
# ---------------------------------------------------------------------------


class TestUpdateLawRobustness:

    def test_empty_inputs(self):
        result = update_law_robustness([], {"law_results": {}})
        assert result == []

    def test_updates_stability(self):
        laws = [_make_law("law_0", stability_score=0.8)]
        stress = {"law_results": {
            "law_0": {"violation_rate": 0.5, "violations": 2, "tests": 4,
                       "avg_severity": 0.3, "robust": False},
        }}
        result = update_law_robustness(laws, stress)
        assert len(result) == 1
        assert result[0]["stability_score"] < 0.8

    def test_classification_updated(self):
        laws = [_make_law("law_0", stability_score=0.9)]
        stress = {"law_results": {
            "law_0": {"violation_rate": 0.8, "violations": 4, "tests": 5,
                       "avg_severity": 0.5, "robust": False},
        }}
        result = update_law_robustness(laws, stress)
        assert result[0]["classification"] in ("robust", "moderate", "fragile")

    def test_no_mutation(self):
        laws = [_make_law("law_0")]
        stress = {"law_results": {"law_0": {
            "violation_rate": 0.3, "violations": 1, "tests": 3,
            "avg_severity": 0.2, "robust": False,
        }}}
        laws_copy = copy.deepcopy(laws)
        update_law_robustness(laws, stress)
        assert laws == laws_copy

    def test_determinism(self):
        laws = [_make_law("law_0", stability_score=0.7)]
        stress = {"law_results": {"law_0": {
            "violation_rate": 0.4, "violations": 2, "tests": 5,
            "avg_severity": 0.3, "robust": False,
        }}}
        r1 = update_law_robustness(laws, stress)
        r2 = update_law_robustness(laws, stress)
        assert r1 == r2


# ---------------------------------------------------------------------------
# DIAGNOSIS REFINEMENT TESTS
# ---------------------------------------------------------------------------


class TestRefineDiagnosisFromCounterfactuals:

    def test_empty_inputs(self):
        result = refine_diagnosis_from_counterfactuals({}, [])
        assert "ranked_diagnoses" in result
        assert "counterfactual_insights" in result

    def test_no_violations(self):
        diag = _make_diagnosis([("healthy_convergence", 0.8)])
        results = [{"violated": False, "severity": 0.0}]
        refined = refine_diagnosis_from_counterfactuals(diag, results)
        assert len(refined["counterfactual_insights"]) == 0

    def test_violations_adjust_scores(self):
        diag = _make_diagnosis([
            ("oscillatory_trap", 0.5),
            ("healthy_convergence", 0.6),
        ])
        results = [{"violated": True, "severity": 0.5}]
        refined = refine_diagnosis_from_counterfactuals(diag, results)
        assert len(refined["counterfactual_insights"]) > 0
        # Oscillatory trap score should increase
        trap_entries = [d for d in refined["ranked_diagnoses"]
                        if d["failure_mode"] == "oscillatory_trap"]
        if trap_entries:
            assert trap_entries[0]["score"] >= 0.5

    def test_re_ranked_after_adjustment(self):
        diag = _make_diagnosis([
            ("healthy_convergence", 0.8),
            ("oscillatory_trap", 0.3),
        ])
        results = [
            {"violated": True, "severity": 0.9},
            {"violated": True, "severity": 0.8},
        ]
        refined = refine_diagnosis_from_counterfactuals(diag, results)
        scores = [d["score"] for d in refined["ranked_diagnoses"]]
        assert scores == sorted(scores, reverse=True)

    def test_determinism(self):
        diag = _make_diagnosis([("oscillatory_trap", 0.5)])
        results = [{"violated": True, "severity": 0.4}]
        r1 = refine_diagnosis_from_counterfactuals(diag, results)
        r2 = refine_diagnosis_from_counterfactuals(diag, results)
        assert r1 == r2

    def test_no_mutation(self):
        diag = _make_diagnosis([("oscillatory_trap", 0.5)])
        results = [{"violated": True, "severity": 0.4}]
        diag_copy = copy.deepcopy(diag)
        results_copy = copy.deepcopy(results)
        refine_diagnosis_from_counterfactuals(diag, results)
        assert diag == diag_copy
        assert results == results_copy
