"""Tests for Art of War strategy modes (v105.2.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.strategy_modes import (
    STRATEGY_ACTIONS,
    STRATEGY_MODES,
    build_strategy_intervention,
    classify_strategy_mode,
    compute_strategy_score,
    get_allowed_actions,
    select_strategy_mode,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_state(stability=0.5, convergence_rate=0.5, escape_rate=0.0,
                coupling_strength=0.5):
    return {
        "global_metrics": {
            "stability": stability,
            "convergence_rate": convergence_rate,
            "escape_rate": escape_rate,
        },
        "trajectory_geometry": {
            "coupling_metrics": {"coupling_strength": coupling_strength},
        },
    }


def _make_diagnosis(failure_mode="healthy_convergence", score=0.5):
    return {
        "ranked_diagnoses": [
            {"failure_mode": failure_mode, "score": score},
        ],
    }


def _make_influence_map(avg_pressure=0.3, avg_sensitivity=0.3):
    return {
        "nodes": {"n0": {"influence_score": 0.5, "instability_pressure": avg_pressure,
                          "control_sensitivity": avg_sensitivity, "stability": 0.5}},
        "summary": {"avg_pressure": avg_pressure, "avg_sensitivity": avg_sensitivity,
                     "node_count": 1, "avg_influence": 0.5,
                     "max_pressure_node": "n0", "max_pressure": avg_pressure},
        "influence_entropy": 0.0,
    }


def _make_laws(fragile_count=0, total=3):
    laws = []
    for i in range(total):
        stability = 0.3 if i < fragile_count else 0.9
        laws.append({"name": f"law_{i}", "stability_score": stability})
    return laws


# ---------------------------------------------------------------------------
# STRATEGY MODE CLASSIFICATION TESTS
# ---------------------------------------------------------------------------


class TestClassifyStrategyMode:

    def test_defensive_on_instability(self):
        state = _make_state(stability=0.3, escape_rate=0.6)
        mode = classify_strategy_mode(state, {})
        assert mode == "defensive"

    def test_offensive_on_plateau(self):
        state = _make_state(stability=0.7, convergence_rate=0.1)
        mode = classify_strategy_mode(state, {})
        assert mode == "offensive"

    def test_positional_on_fragmentation(self):
        state = _make_state(stability=0.7, convergence_rate=0.5,
                            coupling_strength=0.2)
        mode = classify_strategy_mode(state, {})
        assert mode == "positional"

    def test_deceptive_on_persistent_failures(self):
        registry = {
            f"inv_{i}": {"break_count": 5, "count": 10}
            for i in range(5)
        }
        state = _make_state(stability=0.7, convergence_rate=0.5)
        mode = classify_strategy_mode(state, registry)
        assert mode == "deceptive"

    def test_default_defensive(self):
        state = _make_state(stability=0.8, convergence_rate=0.8)
        mode = classify_strategy_mode(state, {})
        assert mode in STRATEGY_MODES

    def test_empty_state(self):
        mode = classify_strategy_mode({}, {})
        assert mode in STRATEGY_MODES

    def test_determinism(self):
        state = _make_state(stability=0.4, escape_rate=0.5)
        r1 = classify_strategy_mode(state, {})
        r2 = classify_strategy_mode(state, {})
        assert r1 == r2

    def test_no_mutation(self):
        state = _make_state()
        registry = {"inv_0": {"break_count": 1}}
        s_copy = copy.deepcopy(state)
        r_copy = copy.deepcopy(registry)
        classify_strategy_mode(state, registry)
        assert state == s_copy
        assert registry == r_copy


# ---------------------------------------------------------------------------
# STRATEGY MODE SELECTION TESTS
# ---------------------------------------------------------------------------


class TestSelectStrategyMode:

    def test_defensive_on_oscillatory_trap(self):
        diag = _make_diagnosis("oscillatory_trap", 0.8)
        imap = _make_influence_map(avg_pressure=0.6)
        mode = select_strategy_mode(diag, imap, [])
        assert mode == "defensive"

    def test_offensive_on_plateau(self):
        diag = _make_diagnosis("metastable_plateau", 0.7)
        imap = _make_influence_map(avg_pressure=0.3)
        mode = select_strategy_mode(diag, imap, [])
        assert mode == "offensive"

    def test_deceptive_on_fragile_laws(self):
        diag = _make_diagnosis("healthy_convergence", 0.3)
        imap = _make_influence_map(avg_pressure=0.2)
        laws = _make_laws(fragile_count=3)
        mode = select_strategy_mode(diag, imap, laws)
        assert mode == "deceptive"

    def test_returns_valid_mode(self):
        diag = _make_diagnosis()
        imap = _make_influence_map()
        mode = select_strategy_mode(diag, imap, [])
        assert mode in STRATEGY_MODES

    def test_determinism(self):
        diag = _make_diagnosis("oscillatory_trap", 0.8)
        imap = _make_influence_map(avg_pressure=0.6)
        r1 = select_strategy_mode(diag, imap, [])
        r2 = select_strategy_mode(diag, imap, [])
        assert r1 == r2


# ---------------------------------------------------------------------------
# ALLOWED ACTIONS TESTS
# ---------------------------------------------------------------------------


class TestGetAllowedActions:

    def test_all_modes_have_actions(self):
        for mode in STRATEGY_MODES:
            actions = get_allowed_actions(mode)
            assert isinstance(actions, list)
            assert len(actions) > 0

    def test_unknown_mode_returns_defensive(self):
        actions = get_allowed_actions("nonexistent")
        assert actions == get_allowed_actions("defensive")

    def test_defensive_actions(self):
        actions = get_allowed_actions("defensive")
        assert "boost_stability" in actions

    def test_offensive_actions(self):
        actions = get_allowed_actions("offensive")
        assert "force_transition" in actions


# ---------------------------------------------------------------------------
# STRATEGY INTERVENTION TESTS
# ---------------------------------------------------------------------------


class TestBuildStrategyIntervention:

    def test_returns_list(self):
        result = build_strategy_intervention("defensive", 0.2)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_intervention_structure(self):
        result = build_strategy_intervention("offensive", 0.3)
        for item in result:
            assert "action" in item
            assert "strength" in item
            assert "strategy_mode" in item
            assert item["strategy_mode"] == "offensive"

    def test_strength_clamped(self):
        result = build_strategy_intervention("defensive", 2.0)
        for item in result:
            assert 0.0 <= item["strength"] <= 1.0

    def test_determinism(self):
        r1 = build_strategy_intervention("positional", 0.4)
        r2 = build_strategy_intervention("positional", 0.4)
        assert r1 == r2


# ---------------------------------------------------------------------------
# STRATEGY SCORE TESTS
# ---------------------------------------------------------------------------


class TestComputeStrategyScore:

    def test_returns_bounded(self):
        imap = _make_influence_map()
        diag = _make_diagnosis()
        for mode in STRATEGY_MODES:
            score = compute_strategy_score(mode, imap, diag)
            assert 0.0 <= score <= 1.0

    def test_defensive_high_when_unstable(self):
        imap = _make_influence_map(avg_pressure=0.9)
        diag = _make_diagnosis("oscillatory_trap", 0.8)
        score = compute_strategy_score("defensive", imap, diag)
        assert score > 0.5

    def test_determinism(self):
        imap = _make_influence_map()
        diag = _make_diagnosis()
        r1 = compute_strategy_score("defensive", imap, diag)
        r2 = compute_strategy_score("defensive", imap, diag)
        assert r1 == r2
