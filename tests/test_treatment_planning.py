"""Tests for treatment_planning.py — v104.5.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.treatment_planning import (
    TREATMENT_STRENGTHS,
    VALID_ACTIONS,
    evaluate_treatments,
    explain_treatment,
    format_treatment_plan,
    generate_treatment_candidates,
    run_treatment_planning,
    score_treatment,
    select_best_treatment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_diagnosis(primary: str, confidence: float = 0.5) -> dict:
    ranked = [(primary, confidence)]
    if primary != "healthy_convergence":
        ranked.append(("slow_convergence", 0.2))
    return {
        "primary_diagnosis": primary,
        "diagnosis_confidence": confidence,
        "ranked": ranked,
    }


def _make_runs() -> list:
    return [
        {
            "strategies": [
                {
                    "name": "s1",
                    "metrics": {
                        "trend": 0.1,
                        "stability": 0.6,
                        "residual_norm": 0.5,
                        "instability_score": 0.3,
                        "barrier_estimate": 0.2,
                        "boundary_distance": 0.4,
                        "spectral_radius_proxy": 0.3,
                        "convergence_signal": 0.5,
                        "control_signal": 0.2,
                        "basin_switch_score": 0.1,
                        "variance_score": 0.1,
                    },
                },
            ],
        },
        {
            "strategies": [
                {
                    "name": "s1",
                    "metrics": {
                        "trend": 0.15,
                        "stability": 0.65,
                        "residual_norm": 0.45,
                        "instability_score": 0.25,
                        "barrier_estimate": 0.2,
                        "boundary_distance": 0.4,
                        "spectral_radius_proxy": 0.28,
                        "convergence_signal": 0.55,
                        "control_signal": 0.2,
                        "basin_switch_score": 0.1,
                        "variance_score": 0.1,
                    },
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Tests: generate_treatment_candidates
# ---------------------------------------------------------------------------


class TestGenerateCandidates:
    def test_oscillatory_trap_candidates(self):
        diag = _make_diagnosis("oscillatory_trap")
        candidates = generate_treatment_candidates(diag)
        # 2 actions x 3 strengths = 6 candidates
        assert len(candidates) == 6

    def test_healthy_convergence_no_candidates(self):
        diag = _make_diagnosis("healthy_convergence")
        candidates = generate_treatment_candidates(diag)
        assert candidates == []

    def test_all_candidates_have_required_keys(self):
        diag = _make_diagnosis("oscillatory_trap")
        for c in generate_treatment_candidates(diag):
            assert "action" in c
            assert "strength" in c
            assert "rationale" in c
            assert c["action"] in VALID_ACTIONS
            assert c["strength"] in TREATMENT_STRENGTHS

    def test_determinism(self):
        diag = _make_diagnosis("basin_switch_instability")
        r1 = generate_treatment_candidates(diag)
        r2 = generate_treatment_candidates(diag)
        assert r1 == r2

    def test_no_mutation_of_input(self):
        diag = _make_diagnosis("oscillatory_trap")
        original = copy.deepcopy(diag)
        generate_treatment_candidates(diag)
        assert diag == original

    def test_revised_diagnosis_used_when_present(self):
        diag = {
            "primary_diagnosis": "oscillatory_trap",
            "revised_diagnosis": "metastable_plateau",
        }
        candidates = generate_treatment_candidates(diag)
        actions = {c["action"] for c in candidates}
        assert "force_transition" in actions

    def test_all_diagnosis_types_produce_candidates(self):
        for mode in ["oscillatory_trap", "metastable_plateau",
                      "basin_switch_instability", "control_overshoot",
                      "underconstrained_dynamics", "slow_convergence"]:
            diag = _make_diagnosis(mode)
            candidates = generate_treatment_candidates(diag)
            assert len(candidates) > 0, f"No candidates for {mode}"


# ---------------------------------------------------------------------------
# Tests: score_treatment
# ---------------------------------------------------------------------------


class TestScoreTreatment:
    def test_score_bounded(self):
        result = {
            "post_metrics": {
                "stability": 0.8,
                "attractor_weight": 0.6,
                "transient_weight": 0.1,
            }
        }
        score = score_treatment(result)
        assert 0.0 <= score <= 1.0

    def test_higher_stability_higher_score(self):
        low = score_treatment({
            "post_metrics": {"stability": 0.2, "attractor_weight": 0.3, "transient_weight": 0.3}
        })
        high = score_treatment({
            "post_metrics": {"stability": 0.9, "attractor_weight": 0.3, "transient_weight": 0.3}
        })
        assert high > low

    def test_lower_transient_higher_score(self):
        high_trans = score_treatment({
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.3, "transient_weight": 0.8}
        })
        low_trans = score_treatment({
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.3, "transient_weight": 0.1}
        })
        assert low_trans > high_trans

    def test_determinism(self):
        result = {
            "post_metrics": {"stability": 0.5, "attractor_weight": 0.4, "transient_weight": 0.2}
        }
        assert score_treatment(result) == score_treatment(result)

    def test_empty_post_metrics(self):
        score = score_treatment({"post_metrics": {}})
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests: select_best_treatment
# ---------------------------------------------------------------------------


class TestSelectBestTreatment:
    def test_selects_highest_score(self):
        results = [
            {"candidate": {"action": "a", "strength": 0.2}, "score": 0.3},
            {"candidate": {"action": "b", "strength": 0.4}, "score": 0.8},
            {"candidate": {"action": "c", "strength": 0.6}, "score": 0.5},
        ]
        best = select_best_treatment(results)
        assert best["score"] == 0.8

    def test_tie_breaks_by_lower_strength(self):
        results = [
            {"candidate": {"action": "a", "strength": 0.6}, "score": 0.5},
            {"candidate": {"action": "a", "strength": 0.2}, "score": 0.5},
        ]
        best = select_best_treatment(results)
        assert best["candidate"]["strength"] == 0.2

    def test_tie_breaks_by_action_name(self):
        results = [
            {"candidate": {"action": "reduce_escape", "strength": 0.2}, "score": 0.5},
            {"candidate": {"action": "boost_stability", "strength": 0.2}, "score": 0.5},
        ]
        best = select_best_treatment(results)
        assert best["candidate"]["action"] == "boost_stability"

    def test_empty_returns_empty(self):
        assert select_best_treatment([]) == {}

    def test_determinism(self):
        results = [
            {"candidate": {"action": "a", "strength": 0.2}, "score": 0.3},
            {"candidate": {"action": "b", "strength": 0.4}, "score": 0.8},
        ]
        assert select_best_treatment(results) == select_best_treatment(results)


# ---------------------------------------------------------------------------
# Tests: explain_treatment
# ---------------------------------------------------------------------------


class TestExplainTreatment:
    def test_explain_returns_string(self):
        best = {
            "candidate": {"action": "boost_stability", "strength": 0.4},
            "score": 0.7,
            "post_metrics": {"stability": 0.8, "attractor_weight": 0.5},
            "aggregate_deltas": {"delta_stability": 0.1},
        }
        diag = _make_diagnosis("oscillatory_trap")
        text = explain_treatment(best, diag)
        assert "boost_stability" in text
        assert "0.4" in text

    def test_explain_empty_best(self):
        text = explain_treatment({}, {})
        assert "No treatment" in text


# ---------------------------------------------------------------------------
# Tests: format_treatment_plan
# ---------------------------------------------------------------------------


class TestFormatTreatmentPlan:
    def test_format_contains_sections(self):
        result = {
            "provocation": {
                "diagnosis": _make_diagnosis("oscillatory_trap"),
            },
            "revised_diagnosis": {
                "revised_diagnosis": "oscillatory_trap",
            },
            "evaluations": [
                {"candidate": {"action": "boost_stability", "strength": 0.4}, "score": 0.7},
            ],
            "best_treatment": {
                "candidate": {"action": "boost_stability", "strength": 0.4},
                "score": 0.7,
            },
            "explanation": "Selected boost_stability(0.4)",
        }
        text = format_treatment_plan(result)
        assert "Treatment Plan" in text
        assert "boost_stability" in text

    def test_format_no_treatment(self):
        result = {
            "provocation": {"diagnosis": _make_diagnosis("healthy_convergence")},
            "revised_diagnosis": {"revised_diagnosis": "healthy_convergence"},
            "evaluations": [],
            "best_treatment": {},
            "explanation": "No treatment needed.",
        }
        text = format_treatment_plan(result)
        assert "No treatment" in text


# ---------------------------------------------------------------------------
# Tests: integration (run_treatment_planning)
# ---------------------------------------------------------------------------


class TestRunTreatmentPlanning:
    def test_full_pipeline_returns_required_keys(self):
        runs = _make_runs()
        result = run_treatment_planning(runs)
        assert "provocation" in result
        assert "revised_diagnosis" in result
        assert "candidates" in result
        assert "evaluations" in result
        assert "best_treatment" in result
        assert "explanation" in result

    def test_full_pipeline_determinism(self):
        runs = _make_runs()
        r1 = run_treatment_planning(runs)
        r2 = run_treatment_planning(runs)
        assert r1 == r2

    def test_no_mutation_of_runs(self):
        runs = _make_runs()
        original = copy.deepcopy(runs)
        run_treatment_planning(runs)
        assert runs == original

    def test_best_treatment_score_bounded(self):
        runs = _make_runs()
        result = run_treatment_planning(runs)
        best = result.get("best_treatment", {})
        if best:
            assert 0.0 <= best["score"] <= 1.0
