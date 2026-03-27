"""Tests for Wu Wei control: minimal intervention principle (v105.2.0).

Deterministic, no randomness, no mutation of inputs.
"""

import copy

import pytest

from qec.analysis.wu_wei_control import (
    ESCALATION_STRENGTHS,
    build_escalation_ladder,
    compute_intervention_efficiency,
    evaluate_intervention_result,
    select_escalation_level,
    select_minimal_intervention,
)


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


def _make_candidate(action="boost_stability", strength=0.2, expected_improvement=0.05):
    return {
        "action": action,
        "strength": strength,
        "expected_improvement": expected_improvement,
    }


# ---------------------------------------------------------------------------
# MINIMAL INTERVENTION SELECTION TESTS
# ---------------------------------------------------------------------------


class TestSelectMinimalIntervention:

    def test_empty_candidates(self):
        assert select_minimal_intervention([]) == {}

    def test_selects_lowest_strength(self):
        candidates = [
            _make_candidate(strength=0.6, expected_improvement=0.1),
            _make_candidate(strength=0.2, expected_improvement=0.05),
        ]
        result = select_minimal_intervention(candidates)
        assert result["strength"] == 0.2
        assert result["selection_reason"] == "minimal_sufficient"

    def test_fallback_to_best_efficiency(self):
        candidates = [
            _make_candidate(strength=0.4, expected_improvement=0.001),
            _make_candidate(strength=0.2, expected_improvement=0.005),
        ]
        result = select_minimal_intervention(candidates)
        assert result["selection_reason"] == "best_efficiency_fallback"

    def test_determinism(self):
        candidates = [_make_candidate(), _make_candidate(strength=0.4)]
        r1 = select_minimal_intervention(candidates)
        r2 = select_minimal_intervention(candidates)
        assert r1 == r2

    def test_no_mutation(self):
        candidates = [_make_candidate()]
        orig = copy.deepcopy(candidates)
        select_minimal_intervention(candidates)
        assert candidates == orig


# ---------------------------------------------------------------------------
# ESCALATION LADDER TESTS
# ---------------------------------------------------------------------------


class TestBuildEscalationLadder:

    def test_returns_correct_length(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        assert len(ladder) == len(ESCALATION_STRENGTHS)

    def test_strengths_ascending(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        strengths = [s["strength"] for s in ladder]
        assert strengths == sorted(strengths)

    def test_levels_sequential(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        levels = [s["level"] for s in ladder]
        assert levels == list(range(len(ESCALATION_STRENGTHS)))


# ---------------------------------------------------------------------------
# ESCALATION LEVEL SELECTION TESTS
# ---------------------------------------------------------------------------


class TestSelectEscalationLevel:

    def test_empty_ladder(self):
        assert select_escalation_level([]) == {}

    def test_first_attempt_starts_at_level_0(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        result = select_escalation_level(ladder)
        assert result["level"] == 0

    def test_escalates_on_no_improvement(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        result = select_escalation_level(
            ladder,
            previous_strength=0.2,
            previous_improvement=0.0,
        )
        assert result["strength"] > 0.2

    def test_escalates_on_destabilization(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        result = select_escalation_level(
            ladder,
            previous_strength=0.2,
            previous_improvement=0.0,
            previous_stability_change=-0.1,
        )
        assert result["strength"] > 0.2

    def test_no_escalation_when_improving(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        result = select_escalation_level(
            ladder,
            previous_strength=0.2,
            previous_improvement=0.1,
            previous_stability_change=0.1,
        )
        assert result["level"] == 0

    def test_returns_highest_when_all_exhausted(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        result = select_escalation_level(
            ladder,
            previous_strength=0.6,
            previous_improvement=0.0,
        )
        assert result["level"] == len(ESCALATION_STRENGTHS) - 1

    def test_determinism(self):
        ladder = build_escalation_ladder("boost_stability", 0.05)
        r1 = select_escalation_level(ladder, previous_strength=0.2, previous_improvement=0.0)
        r2 = select_escalation_level(ladder, previous_strength=0.2, previous_improvement=0.0)
        assert r1 == r2


# ---------------------------------------------------------------------------
# INTERVENTION EFFICIENCY TESTS
# ---------------------------------------------------------------------------


class TestComputeInterventionEfficiency:

    def test_positive_improvement(self):
        eff = compute_intervention_efficiency(
            {"stability": 0.5}, {"stability": 0.8}, 0.2
        )
        assert eff > 0.0

    def test_no_improvement(self):
        eff = compute_intervention_efficiency(
            {"stability": 0.5}, {"stability": 0.4}, 0.2
        )
        assert eff == 0.0

    def test_zero_strength_returns_zero(self):
        eff = compute_intervention_efficiency(
            {"stability": 0.5}, {"stability": 0.8}, 0.0
        )
        assert eff == 0.0

    def test_negative_strength_returns_zero(self):
        eff = compute_intervention_efficiency(
            {"stability": 0.5}, {"stability": 0.8}, -0.1
        )
        assert eff == 0.0

    def test_determinism(self):
        r1 = compute_intervention_efficiency({"stability": 0.3}, {"stability": 0.7}, 0.4)
        r2 = compute_intervention_efficiency({"stability": 0.3}, {"stability": 0.7}, 0.4)
        assert r1 == r2


# ---------------------------------------------------------------------------
# EVALUATE INTERVENTION RESULT TESTS
# ---------------------------------------------------------------------------


class TestEvaluateInterventionResult:

    def test_effective_intervention(self):
        result = evaluate_intervention_result(
            {"stability": 0.3}, {"stability": 0.6}, {"strength": 0.2}
        )
        assert result["effective"] is True
        assert result["needs_escalation"] is False
        assert result["improvement"] > 0.0

    def test_ineffective_intervention(self):
        result = evaluate_intervention_result(
            {"stability": 0.5}, {"stability": 0.5}, {"strength": 0.2}
        )
        assert result["effective"] is False
        assert result["needs_escalation"] is True

    def test_no_mutation(self):
        before = {"stability": 0.3}
        after = {"stability": 0.6}
        intervention = {"strength": 0.2}
        b_copy, a_copy, i_copy = copy.deepcopy(before), copy.deepcopy(after), copy.deepcopy(intervention)
        evaluate_intervention_result(before, after, intervention)
        assert before == b_copy
        assert after == a_copy
        assert intervention == i_copy
