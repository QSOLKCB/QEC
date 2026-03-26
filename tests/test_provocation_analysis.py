"""Tests for provocation_analysis.py — v104.5.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.provocation_analysis import (
    RESPONSE_CLASSES,
    apply_baseline_interventions,
    characterize_response,
    format_provocation_analysis,
    get_baseline_interventions,
    revise_diagnosis_from_response,
    run_provocation_analysis,
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
        "features": {},
        "scores": {},
        "explanations": {},
    }


def _make_state_vector(stability: int = 0, phase: int = 0, trend: int = 0) -> dict:
    return {
        "ternary": {
            "stability_state": stability,
            "phase_state": phase,
            "trend_state": trend,
        },
        "membership": {
            "strong_attractor": 0.3,
            "weak_attractor": 0.2,
            "transient": 0.2,
            "basin": 0.3,
        },
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
# Tests: get_baseline_interventions
# ---------------------------------------------------------------------------


class TestGetBaselineInterventions:
    def test_oscillatory_trap_returns_two_interventions(self):
        diag = _make_diagnosis("oscillatory_trap")
        result = get_baseline_interventions(diag)
        assert len(result) == 2
        actions = [r["action"] for r in result]
        assert "reduce_escape" in actions
        assert "boost_stability" in actions

    def test_healthy_convergence_returns_empty(self):
        diag = _make_diagnosis("healthy_convergence")
        result = get_baseline_interventions(diag)
        assert result == []

    def test_metastable_plateau_returns_force_transition(self):
        diag = _make_diagnosis("metastable_plateau")
        result = get_baseline_interventions(diag)
        assert len(result) == 1
        assert result[0]["action"] == "force_transition"

    def test_determinism(self):
        diag = _make_diagnosis("oscillatory_trap")
        r1 = get_baseline_interventions(diag)
        r2 = get_baseline_interventions(diag)
        assert r1 == r2

    def test_no_mutation_of_input(self):
        diag = _make_diagnosis("oscillatory_trap")
        original = copy.deepcopy(diag)
        get_baseline_interventions(diag)
        assert diag == original

    def test_all_strengths_are_low(self):
        for mode in ["oscillatory_trap", "metastable_plateau",
                      "basin_switch_instability", "control_overshoot",
                      "underconstrained_dynamics", "slow_convergence"]:
            diag = _make_diagnosis(mode)
            for iv in get_baseline_interventions(diag):
                assert iv["strength"] == 0.2

    def test_unknown_diagnosis_returns_empty(self):
        diag = {"primary_diagnosis": "nonexistent_mode"}
        result = get_baseline_interventions(diag)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: characterize_response
# ---------------------------------------------------------------------------


class TestCharacterizeResponse:
    def test_improved_classification(self):
        before = {"s1": _make_state_vector(stability=0)}
        after = {"s1": _make_state_vector(stability=1)}
        result = characterize_response(before, after)
        assert result["classification"] == "improved"

    def test_unchanged_classification(self):
        sv = _make_state_vector()
        before = {"s1": sv}
        after = {"s1": copy.deepcopy(sv)}
        result = characterize_response(before, after)
        assert result["classification"] == "unchanged"

    def test_destabilized_classification(self):
        before = {"s1": _make_state_vector(stability=1)}
        after_sv = _make_state_vector(stability=0)
        after_sv["membership"]["transient"] = 0.5
        after = {"s1": after_sv}
        result = characterize_response(before, after)
        assert result["classification"] == "destabilized"

    def test_all_response_classes_valid(self):
        for cls in RESPONSE_CLASSES:
            assert isinstance(cls, str)

    def test_determinism(self):
        before = {"s1": _make_state_vector(stability=0)}
        after = {"s1": _make_state_vector(stability=1)}
        r1 = characterize_response(before, after)
        r2 = characterize_response(before, after)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        before = {"s1": _make_state_vector(stability=0)}
        after = {"s1": _make_state_vector(stability=1)}
        before_orig = copy.deepcopy(before)
        after_orig = copy.deepcopy(after)
        characterize_response(before, after)
        assert before == before_orig
        assert after == after_orig

    def test_deltas_present(self):
        before = {"s1": _make_state_vector(stability=0)}
        after = {"s1": _make_state_vector(stability=1)}
        result = characterize_response(before, after)
        assert "deltas" in result
        assert "stability" in result["deltas"]

    def test_empty_inputs(self):
        result = characterize_response({}, {})
        assert result["classification"] == "unchanged"


# ---------------------------------------------------------------------------
# Tests: revise_diagnosis_from_response
# ---------------------------------------------------------------------------


class TestReviseDiagnosis:
    def test_improved_increases_confidence(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "improved", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert result["confidence_shift"] > 0
        assert result["revised_confidence"] > 0.5

    def test_worsened_elevates_alternative(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "worsened", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert result["confidence_shift"] < 0
        assert result["revised_diagnosis"] == "slow_convergence"

    def test_destabilized_elevates_alternative(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "destabilized", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert result["confidence_shift"] < 0
        assert result["revised_diagnosis"] == "slow_convergence"

    def test_unchanged_no_revision(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "unchanged", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert result["confidence_shift"] == 0.0
        assert result["revised_diagnosis"] == "oscillatory_trap"

    def test_revealed_new_mode(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "revealed_new_mode", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert result["revealed_mode"] != ""

    def test_determinism(self):
        diag = _make_diagnosis("oscillatory_trap", 0.5)
        response = {"classification": "improved", "deltas": {}}
        r1 = revise_diagnosis_from_response(diag, response)
        r2 = revise_diagnosis_from_response(diag, response)
        assert r1 == r2

    def test_confidence_bounded(self):
        diag = _make_diagnosis("oscillatory_trap", 0.99)
        response = {"classification": "improved", "deltas": {}}
        result = revise_diagnosis_from_response(diag, response)
        assert 0.0 <= result["revised_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: format_provocation_analysis
# ---------------------------------------------------------------------------


class TestFormatProvocation:
    def test_format_contains_sections(self):
        result = {
            "diagnosis": _make_diagnosis("oscillatory_trap"),
            "interventions": [{"action": "reduce_escape", "strength": 0.2}],
            "response": {
                "deltas": {"stability": 0.08},
                "classification": "improved",
            },
            "revision": {
                "revised_diagnosis": "oscillatory_trap",
                "confidence_shift": 0.12,
                "revealed_mode": "",
            },
        }
        text = format_provocation_analysis(result)
        assert "Provocation Analysis" in text
        assert "oscillatory_trap" in text
        assert "reduce_escape" in text
        assert "improved" in text

    def test_format_healthy_convergence(self):
        result = {
            "diagnosis": _make_diagnosis("healthy_convergence"),
            "interventions": [],
            "response": {"deltas": {}, "classification": "unchanged"},
            "revision": {
                "revised_diagnosis": "healthy_convergence",
                "confidence_shift": 0.0,
                "revealed_mode": "",
            },
        }
        text = format_provocation_analysis(result)
        assert "healthy convergence" in text.lower()


# ---------------------------------------------------------------------------
# Tests: integration (run_provocation_analysis)
# ---------------------------------------------------------------------------


class TestRunProvocationAnalysis:
    def test_full_pipeline_returns_required_keys(self):
        runs = _make_runs()
        result = run_provocation_analysis(runs)
        assert "diagnosis" in result
        assert "interventions" in result
        assert "response" in result
        assert "revision" in result

    def test_full_pipeline_determinism(self):
        runs = _make_runs()
        r1 = run_provocation_analysis(runs)
        r2 = run_provocation_analysis(runs)
        assert r1 == r2

    def test_no_mutation_of_runs(self):
        runs = _make_runs()
        original = copy.deepcopy(runs)
        run_provocation_analysis(runs)
        assert runs == original
