"""Tests for treatment_invariants.py — v104.5.0."""

from __future__ import annotations

import copy

import pytest

from qec.analysis.treatment_invariants import (
    INVARIANT_TYPES,
    extract_treatment_invariants,
    format_treatment_invariants,
    run_treatment_invariant_analysis,
    score_invariants,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_diagnosis(primary: str = "oscillatory_trap", confidence: float = 0.5) -> dict:
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


def _make_provocation(
    classification: str = "improved",
    d_transient: float = -0.05,
    revised: str = "oscillatory_trap",
) -> dict:
    return {
        "diagnosis": _make_diagnosis("oscillatory_trap"),
        "interventions": [{"action": "reduce_escape", "strength": 0.2}],
        "intervention_result": {
            "before": {
                "s1": {
                    "ternary": {"stability_state": 0, "phase_state": 0, "trend_state": 0},
                    "membership": {
                        "strong_attractor": 0.3,
                        "weak_attractor": 0.2,
                        "transient": 0.2,
                        "basin": 0.3,
                    },
                },
            },
            "after": {
                "s1": {
                    "ternary": {"stability_state": 0, "phase_state": 0, "trend_state": 0},
                    "membership": {
                        "strong_attractor": 0.35,
                        "weak_attractor": 0.25,
                        "transient": 0.15,
                        "basin": 0.25,
                    },
                },
            },
        },
        "response": {
            "deltas": {
                "stability": 0.0,
                "attractor_weight": 0.1,
                "transient_weight": d_transient,
                "phase": 0.0,
            },
            "classification": classification,
            "per_strategy": {},
        },
        "revision": {
            "revised_diagnosis": revised,
            "revised_confidence": 0.62,
            "confidence_shift": 0.12,
            "revealed_mode": "",
            "revision_reason": "baseline improved",
        },
    }


def _make_treatment(
    score: float = 0.7,
    d_stability: float = 0.0,
    d_transient: float = -0.05,
) -> dict:
    return {
        "provocation": _make_provocation(),
        "revised_diagnosis": {
            "primary_diagnosis": "oscillatory_trap",
            "revised_diagnosis": "oscillatory_trap",
            "revised_confidence": 0.62,
            "ranked": [("oscillatory_trap", 0.5)],
        },
        "candidates": [
            {"action": "boost_stability", "strength": 0.4, "rationale": "test"},
        ],
        "evaluations": [
            {
                "candidate": {"action": "boost_stability", "strength": 0.4},
                "score": score,
                "aggregate_deltas": {
                    "delta_stability": d_stability,
                    "delta_attractor_weight": 0.05,
                    "delta_transient_weight": d_transient,
                },
                "post_metrics": {
                    "stability": 0.6,
                    "attractor_weight": 0.5,
                    "transient_weight": 0.15,
                },
            },
        ],
        "best_treatment": {
            "candidate": {"action": "boost_stability", "strength": 0.4},
            "score": score,
            "aggregate_deltas": {
                "delta_stability": d_stability,
                "delta_attractor_weight": 0.05,
                "delta_transient_weight": d_transient,
            },
            "post_metrics": {
                "stability": 0.6,
                "attractor_weight": 0.5,
                "transient_weight": 0.15,
            },
        },
        "explanation": "Selected boost_stability(0.4)",
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
# Tests: extract_treatment_invariants
# ---------------------------------------------------------------------------


class TestExtractInvariants:
    def test_returns_invariants_key(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        assert "invariants" in result

    def test_invariants_are_list(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        assert isinstance(result["invariants"], list)

    def test_only_held_invariants_returned(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        for inv in result["invariants"]:
            assert inv["holds"] is True

    def test_invariant_has_required_keys(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        for inv in result["invariants"]:
            assert "name" in inv
            assert "type" in inv
            assert "description" in inv
            assert "holds" in inv
            assert "support" in inv

    def test_determinism(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        r1 = extract_treatment_invariants(diag, prov, treat)
        r2 = extract_treatment_invariants(diag, prov, treat)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        d_orig = copy.deepcopy(diag)
        p_orig = copy.deepcopy(prov)
        t_orig = copy.deepcopy(treat)
        extract_treatment_invariants(diag, prov, treat)
        assert diag == d_orig
        assert prov == p_orig
        assert treat == t_orig

    def test_diagnosis_persistence_when_same(self):
        diag = _make_diagnosis("oscillatory_trap")
        prov = _make_provocation(revised="oscillatory_trap")
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        names = [inv["name"] for inv in result["invariants"]]
        assert "diagnosis_persistence" in names

    def test_diagnosis_persistence_broken_when_different(self):
        diag = _make_diagnosis("oscillatory_trap")
        prov = _make_provocation(revised="slow_convergence")
        treat = _make_treatment()
        # Also change treatment revised diagnosis.
        treat["revised_diagnosis"]["revised_diagnosis"] = "slow_convergence"
        result = extract_treatment_invariants(diag, prov, treat)
        names = [inv["name"] for inv in result["invariants"]]
        assert "diagnosis_persistence" not in names

    def test_basin_switch_suppression_holds_when_transient_decreases(self):
        diag = _make_diagnosis()
        prov = _make_provocation(classification="improved", d_transient=-0.05)
        treat = _make_treatment(d_transient=-0.03)
        result = extract_treatment_invariants(diag, prov, treat)
        names = [inv["name"] for inv in result["invariants"]]
        assert "basin_switch_suppression" in names

    def test_invariant_types_are_valid(self):
        diag = _make_diagnosis()
        prov = _make_provocation()
        treat = _make_treatment()
        result = extract_treatment_invariants(diag, prov, treat)
        for inv in result["invariants"]:
            assert inv["type"] in INVARIANT_TYPES


# ---------------------------------------------------------------------------
# Tests: score_invariants
# ---------------------------------------------------------------------------


class TestScoreInvariants:
    def test_scored_invariants_present(self):
        raw = {
            "invariants": [
                {
                    "name": "test",
                    "type": "sign",
                    "description": "test",
                    "holds": True,
                    "support": 2,
                    "total": 2,
                },
            ]
        }
        result = score_invariants(raw)
        assert "scored_invariants" in result

    def test_strength_bounded(self):
        raw = {
            "invariants": [
                {
                    "name": "test",
                    "type": "sign",
                    "description": "test",
                    "holds": True,
                    "support": 1,
                    "total": 2,
                },
            ]
        }
        result = score_invariants(raw)
        for inv in result["scored_invariants"]:
            assert 0.0 <= inv["strength"] <= 1.0

    def test_full_support_gives_strength_one(self):
        raw = {
            "invariants": [
                {
                    "name": "test",
                    "type": "sign",
                    "description": "test",
                    "holds": True,
                    "support": 3,
                    "total": 3,
                },
            ]
        }
        result = score_invariants(raw)
        assert result["scored_invariants"][0]["strength"] == 1.0

    def test_sorted_by_strength_desc(self):
        raw = {
            "invariants": [
                {"name": "b", "type": "sign", "description": "", "holds": True, "support": 1, "total": 2},
                {"name": "a", "type": "sign", "description": "", "holds": True, "support": 2, "total": 2},
            ]
        }
        result = score_invariants(raw)
        scored = result["scored_invariants"]
        assert scored[0]["name"] == "a"
        assert scored[1]["name"] == "b"

    def test_determinism(self):
        raw = {
            "invariants": [
                {"name": "test", "type": "sign", "description": "", "holds": True, "support": 1, "total": 2},
            ]
        }
        assert score_invariants(raw) == score_invariants(raw)

    def test_empty_invariants(self):
        result = score_invariants({"invariants": []})
        assert result["scored_invariants"] == []


# ---------------------------------------------------------------------------
# Tests: format_treatment_invariants
# ---------------------------------------------------------------------------


class TestFormatInvariants:
    def test_format_contains_header(self):
        scored = {
            "scored_invariants": [
                {
                    "name": "stability_monotonicity",
                    "type": "geometry",
                    "description": "Stability increased at every step.",
                    "strength": 1.0,
                    "support": 2,
                    "total": 2,
                },
            ]
        }
        text = format_treatment_invariants(scored)
        assert "Treatment Invariants" in text
        assert "Stability Monotonicity" in text
        assert "1.00" in text

    def test_format_no_invariants(self):
        text = format_treatment_invariants({"scored_invariants": []})
        assert "No invariants" in text


# ---------------------------------------------------------------------------
# Tests: integration (run_treatment_invariant_analysis)
# ---------------------------------------------------------------------------


class TestRunTreatmentInvariantAnalysis:
    def test_full_pipeline_returns_required_keys(self):
        runs = _make_runs()
        result = run_treatment_invariant_analysis(runs)
        assert "diagnosis" in result
        assert "provocation" in result
        assert "treatment" in result
        assert "invariants" in result
        assert "scored_invariants" in result

    def test_full_pipeline_determinism(self):
        runs = _make_runs()
        r1 = run_treatment_invariant_analysis(runs)
        r2 = run_treatment_invariant_analysis(runs)
        assert r1 == r2

    def test_no_mutation_of_runs(self):
        runs = _make_runs()
        original = copy.deepcopy(runs)
        run_treatment_invariant_analysis(runs)
        assert runs == original

    def test_only_true_invariants_reported(self):
        runs = _make_runs()
        result = run_treatment_invariant_analysis(runs)
        raw = result.get("invariants", {})
        for inv in raw.get("invariants", []):
            assert inv["holds"] is True

    def test_scored_invariants_strength_bounded(self):
        runs = _make_runs()
        result = run_treatment_invariant_analysis(runs)
        for inv in result.get("scored_invariants", []):
            assert 0.0 <= inv["strength"] <= 1.0
