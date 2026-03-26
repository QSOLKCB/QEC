"""Tests for multi-strategy feedback and global control (v103.2.0).

Verifies:
- deterministic global optimization
- conflict resolution correctness
- convergence detection
- multi-strategy interactions
- no mutation of original data
- global objective evaluation
- joint intervention application
"""

from __future__ import annotations

import copy

import pytest

from qec.analysis.global_control import (
    ACTION_PRIORITY,
    DEFAULT_ATTRACTOR_WEIGHT,
    DEFAULT_MAX_STEPS,
    DEFAULT_STABILITY_WEIGHT,
    DEFAULT_SYNC_WEIGHT,
    DEFAULT_TRANSIENT_WEIGHT,
    GLOBAL_SCORE_DELTA_THRESHOLD,
    ROUND_PRECISION,
    SYNC_DELTA_THRESHOLD,
    apply_global_intervention,
    detect_global_convergence,
    evaluate_global_objective,
    format_global_control_summary,
    generate_global_candidates,
    resolve_conflicts,
    run_global_feedback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_vector(
    trend=0,
    stability=0,
    phase=0,
    membership=None,
):
    """Build a state vector matching multistate output format."""
    if membership is None:
        membership = {
            "strong_attractor": 0.4,
            "weak_attractor": 0.2,
            "basin": 0.2,
            "transient": 0.1,
            "neutral": 0.1,
        }
    return {
        "ternary": {
            "trend_state": trend,
            "stability_state": stability,
            "phase_state": phase,
        },
        "membership": dict(membership),
    }


def _make_multistate(strategy_names=None, **kwargs):
    """Build a multistate dict with default state vectors."""
    if strategy_names is None:
        strategy_names = ["A", "B"]
    return {name: _make_state_vector(**kwargs) for name in strategy_names}


def _make_coupled_summary(strategy_names=None, sync_ratio=0.6):
    """Build a coupled summary dict."""
    if strategy_names is None:
        strategy_names = ["A", "B"]
    summary = {}
    names = sorted(strategy_names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            summary[(a, b)] = {
                "coupling_strength": 0.3,
                "sync_ratio": sync_ratio,
                "sync_classification": "partially_synchronized",
                "overlap": 0.5,
                "alignment": "weakly_aligned",
            }
    return {"coupled_summary": summary}


def _make_runs(strategy_names=None):
    """Build minimal run data."""
    if strategy_names is None:
        strategy_names = ["A", "B"]
    strategies = [
        {
            "name": name,
            "metrics": {
                "trend": 0.0,
                "stability": 0.7,
                "phase": "basin",
            },
        }
        for name in strategy_names
    ]
    return [{"strategies": strategies}]


# ---------------------------------------------------------------------------
# Tests — evaluate_global_objective
# ---------------------------------------------------------------------------


class TestEvaluateGlobalObjective:
    """Tests for evaluate_global_objective."""

    def test_basic_score(self):
        """Score is computed from stability, attractor, transient, sync."""
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        score = evaluate_global_objective(multistate, coupled, objective)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self):
        """Same inputs produce same output."""
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {"w_stability": 0.3, "w_attractor": 0.3}
        s1 = evaluate_global_objective(multistate, coupled, objective)
        s2 = evaluate_global_objective(multistate, coupled, objective)
        assert s1 == s2

    def test_empty_multistate(self):
        """Empty multistate returns 0.0."""
        score = evaluate_global_objective(
            {}, _make_coupled_summary(), {"w_stability": 1.0},
        )
        assert score == 0.0

    def test_high_stability_high_score(self):
        """High stability strategies should produce higher score."""
        high = _make_multistate(stability=1, membership={
            "strong_attractor": 0.6,
            "weak_attractor": 0.2,
            "basin": 0.1,
            "transient": 0.05,
            "neutral": 0.05,
        })
        low = _make_multistate(stability=-1, membership={
            "strong_attractor": 0.1,
            "weak_attractor": 0.1,
            "basin": 0.1,
            "transient": 0.6,
            "neutral": 0.1,
        })
        obj = {"w_stability": 0.3, "w_attractor": 0.3, "w_transient": 0.2, "w_sync": 0.2}
        coupled = _make_coupled_summary()
        score_high = evaluate_global_objective(high, coupled, obj)
        score_low = evaluate_global_objective(low, coupled, obj)
        assert score_high > score_low

    def test_no_mutation(self):
        """Inputs are not mutated."""
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {"w_stability": 0.3}
        ms_copy = copy.deepcopy(multistate)
        coupled_copy = copy.deepcopy(coupled)
        evaluate_global_objective(multistate, coupled, objective)
        assert multistate == ms_copy
        assert coupled == coupled_copy

    def test_default_weights(self):
        """Missing weight keys use defaults."""
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        score = evaluate_global_objective(multistate, coupled, {})
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_bounded(self):
        """Score is always in [0, 1]."""
        multistate = _make_multistate(stability=1, membership={
            "strong_attractor": 1.0,
            "weak_attractor": 0.0,
            "basin": 0.0,
            "transient": 0.0,
            "neutral": 0.0,
        })
        coupled = _make_coupled_summary(sync_ratio=1.0)
        objective = {"w_stability": 1.0, "w_attractor": 1.0, "w_transient": 0.0, "w_sync": 1.0}
        score = evaluate_global_objective(multistate, coupled, objective)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests — generate_global_candidates
# ---------------------------------------------------------------------------


class TestGenerateGlobalCandidates:
    """Tests for generate_global_candidates."""

    def test_default_candidates(self):
        """Generates candidates for all strategies x actions x strengths."""
        candidates = generate_global_candidates(["A", "B"])
        # 2 strategies x 3 actions x 3 strengths = 18
        assert len(candidates) == 18

    def test_sorted_order(self):
        """Candidates are sorted by target, action, strength."""
        candidates = generate_global_candidates(["B", "A"])
        targets = [c["target"] for c in candidates]
        # First 9 should be "A", next 9 should be "B".
        assert targets[:9] == ["A"] * 9
        assert targets[9:] == ["B"] * 9

    def test_custom_actions_strengths(self):
        """Custom actions and strengths work."""
        candidates = generate_global_candidates(
            ["X"], actions=["boost_stability"], strengths=[0.5],
        )
        assert len(candidates) == 1
        assert candidates[0]["target"] == "X"
        assert candidates[0]["action"] == "boost_stability"
        assert candidates[0]["strength"] == 0.5

    def test_deterministic(self):
        """Same inputs produce same output."""
        c1 = generate_global_candidates(["A", "B"])
        c2 = generate_global_candidates(["A", "B"])
        assert c1 == c2

    def test_empty_names(self):
        """Empty strategy list produces empty candidates."""
        assert generate_global_candidates([]) == []


# ---------------------------------------------------------------------------
# Tests — apply_global_intervention
# ---------------------------------------------------------------------------


class TestApplyGlobalIntervention:
    """Tests for apply_global_intervention."""

    def test_basic_application(self):
        """Applying an intervention modifies the targeted strategy."""
        state = _make_multistate()
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]
        result = apply_global_intervention(state, interventions)
        assert "A" in result
        assert "B" in result
        # A should be modified, B unchanged in ternary.
        assert result["B"]["ternary"] == state["B"]["ternary"]

    def test_no_mutation(self):
        """Original state is not mutated."""
        state = _make_multistate()
        state_copy = copy.deepcopy(state)
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.9},
        ]
        apply_global_intervention(state, interventions)
        assert state == state_copy

    def test_deterministic(self):
        """Same inputs produce same output."""
        state = _make_multistate()
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "B", "action": "reduce_escape", "strength": 0.3},
        ]
        r1 = apply_global_intervention(state, interventions)
        r2 = apply_global_intervention(state, interventions)
        assert r1 == r2

    def test_multiple_targets(self):
        """Interventions on different targets are all applied."""
        state = _make_multistate(strategy_names=["A", "B", "C"])
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "B", "action": "reduce_escape", "strength": 0.3},
            {"target": "C", "action": "force_transition", "strength": 0.9},
        ]
        result = apply_global_intervention(state, interventions)
        assert len(result) == 3

    def test_unknown_target_ignored(self):
        """Interventions on unknown targets are skipped."""
        state = _make_multistate()
        interventions = [
            {"target": "UNKNOWN", "action": "boost_stability", "strength": 0.6},
        ]
        result = apply_global_intervention(state, interventions)
        assert result == {
            name: {
                "ternary": dict(state[name]["ternary"]),
                "membership": dict(state[name]["membership"]),
            }
            for name in sorted(state.keys())
        }

    def test_membership_normalized(self):
        """Membership weights sum to ~1.0 after intervention."""
        state = _make_multistate()
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.9},
        ]
        result = apply_global_intervention(state, interventions)
        mem = result["A"]["membership"]
        total = sum(mem.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Tests — resolve_conflicts
# ---------------------------------------------------------------------------


class TestResolveConflicts:
    """Tests for resolve_conflicts."""

    def test_no_conflicts(self):
        """No conflicts when each target has one intervention."""
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "B", "action": "reduce_escape", "strength": 0.3},
        ]
        resolved = resolve_conflicts(interventions)
        assert len(resolved) == 2

    def test_same_target_priority(self):
        """Same target: higher priority action wins."""
        interventions = [
            {"target": "A", "action": "force_transition", "strength": 0.9},
            {"target": "A", "action": "boost_stability", "strength": 0.3},
        ]
        resolved = resolve_conflicts(interventions)
        assert len(resolved) == 1
        # boost_stability has priority 2 > force_transition priority 0.
        assert resolved[0]["action"] == "boost_stability"

    def test_same_target_same_action_highest_strength(self):
        """Same target, same action: highest strength wins."""
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.3},
            {"target": "A", "action": "boost_stability", "strength": 0.9},
        ]
        resolved = resolve_conflicts(interventions)
        assert len(resolved) == 1
        assert resolved[0]["strength"] == 0.9

    def test_priority_order(self):
        """boost_stability > reduce_escape > force_transition."""
        assert ACTION_PRIORITY["boost_stability"] > ACTION_PRIORITY["reduce_escape"]
        assert ACTION_PRIORITY["reduce_escape"] > ACTION_PRIORITY["force_transition"]

    def test_deterministic(self):
        """Same inputs produce same output."""
        interventions = [
            {"target": "A", "action": "force_transition", "strength": 0.9},
            {"target": "A", "action": "boost_stability", "strength": 0.3},
            {"target": "B", "action": "reduce_escape", "strength": 0.6},
        ]
        r1 = resolve_conflicts(interventions)
        r2 = resolve_conflicts(interventions)
        assert r1 == r2

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert resolve_conflicts([]) == []

    def test_no_mutation(self):
        """Inputs are not mutated."""
        interventions = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "A", "action": "reduce_escape", "strength": 0.3},
        ]
        orig = copy.deepcopy(interventions)
        resolve_conflicts(interventions)
        assert interventions == orig

    def test_sorted_output(self):
        """Output is sorted by target name."""
        interventions = [
            {"target": "C", "action": "boost_stability", "strength": 0.6},
            {"target": "A", "action": "reduce_escape", "strength": 0.3},
            {"target": "B", "action": "force_transition", "strength": 0.9},
        ]
        resolved = resolve_conflicts(interventions)
        targets = [r["target"] for r in resolved]
        assert targets == sorted(targets)


# ---------------------------------------------------------------------------
# Tests — detect_global_convergence
# ---------------------------------------------------------------------------


class TestDetectGlobalConvergence:
    """Tests for detect_global_convergence."""

    def test_stable_convergence(self):
        """Small score delta triggers stable convergence."""
        states = [
            {"score": 0.5, "sync": 0.6},
            {"score": 0.5 + GLOBAL_SCORE_DELTA_THRESHOLD / 2, "sync": 0.6},
        ]
        result = detect_global_convergence(states)
        assert result["converged"] is True
        assert result["type"] == "stable"

    def test_no_convergence(self):
        """Large score changes -> no convergence."""
        states = [
            {"score": 0.3, "sync": 0.5},
            {"score": 0.6, "sync": 0.5},
        ]
        result = detect_global_convergence(states)
        assert result["converged"] is False

    def test_cycle_detection(self):
        """Repeated pattern triggers cycle convergence."""
        states = [
            {"score": 0.5, "sync": 0.6},
            {"score": 0.7, "sync": 0.8},
            {"score": 0.5, "sync": 0.6},
        ]
        result = detect_global_convergence(states)
        assert result["converged"] is True
        assert result["type"] == "cycle"

    def test_single_state(self):
        """Single state -> no convergence."""
        result = detect_global_convergence([{"score": 0.5, "sync": 0.6}])
        assert result["converged"] is False
        assert result["type"] == "max_steps"

    def test_empty_states(self):
        """Empty states -> no convergence."""
        result = detect_global_convergence([])
        assert result["converged"] is False

    def test_deterministic(self):
        """Same inputs produce same output."""
        states = [
            {"score": 0.5, "sync": 0.6},
            {"score": 0.5, "sync": 0.6},
        ]
        r1 = detect_global_convergence(states)
        r2 = detect_global_convergence(states)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests — run_global_feedback
# ---------------------------------------------------------------------------


class TestRunGlobalFeedback:
    """Tests for run_global_feedback."""

    def test_basic_run(self):
        """Basic global feedback loop completes."""
        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }
        result = run_global_feedback(
            runs,
            objective,
            max_steps=3,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )

        assert "states" in result
        assert "interventions" in result
        assert "scores" in result
        assert "convergence" in result
        assert "steps_taken" in result
        assert len(result["scores"]) >= 1

    def test_deterministic(self):
        """Same inputs produce same output."""
        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {"w_stability": 0.3, "w_attractor": 0.3}

        r1 = run_global_feedback(
            runs, objective, max_steps=2,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        r2 = run_global_feedback(
            runs, objective, max_steps=2,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        assert r1["scores"] == r2["scores"]
        assert r1["convergence"] == r2["convergence"]

    def test_no_mutation(self):
        """Original inputs are not mutated."""
        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        ms_copy = copy.deepcopy(multistate)
        coupled_copy = copy.deepcopy(coupled)
        objective = {"w_stability": 0.3}

        run_global_feedback(
            runs, objective, max_steps=2,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        assert multistate == ms_copy
        assert coupled == coupled_copy

    def test_scores_list_length(self):
        """Scores list has length = steps_taken + 1 (includes initial)."""
        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {"w_stability": 0.3}

        result = run_global_feedback(
            runs, objective, max_steps=3,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        assert len(result["scores"]) == result["steps_taken"] + 1

    def test_interventions_are_lists(self):
        """Each entry in interventions is a list of dicts."""
        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()
        objective = {"w_stability": 0.3}

        result = run_global_feedback(
            runs, objective, max_steps=2,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        for step_interventions in result["interventions"]:
            assert isinstance(step_interventions, list)
            for intervention in step_interventions:
                assert "target" in intervention
                assert "action" in intervention
                assert "strength" in intervention

    def test_three_strategy_system(self):
        """Global feedback works with three strategies."""
        runs = _make_runs(strategy_names=["A", "B", "C"])
        multistate = _make_multistate(strategy_names=["A", "B", "C"])
        coupled = _make_coupled_summary(strategy_names=["A", "B", "C"])
        objective = {"w_stability": 0.3, "w_attractor": 0.3}

        result = run_global_feedback(
            runs, objective, max_steps=2,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
        )
        assert result["steps_taken"] >= 0


# ---------------------------------------------------------------------------
# Tests — format_global_control_summary
# ---------------------------------------------------------------------------


class TestFormatGlobalControlSummary:
    """Tests for format_global_control_summary."""

    def test_basic_format(self):
        """Output contains expected sections."""
        result = {
            "states": [_make_multistate(), _make_multistate()],
            "interventions": [
                [
                    {"target": "A", "action": "boost_stability", "strength": 0.6},
                    {"target": "B", "action": "reduce_escape", "strength": 0.3},
                ],
            ],
            "scores": [0.5, 0.7],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "steps_taken": 1,
        }
        summary = format_global_control_summary(result)
        assert "=== Global Control ===" in summary
        assert "Step 1:" in summary
        assert "boost_stability" in summary
        assert "reduce_escape" in summary
        assert "Converged: stable" in summary
        assert "Final Score:" in summary

    def test_not_converged(self):
        """Not-converged result shows correct message."""
        result = {
            "states": [_make_multistate()],
            "interventions": [],
            "scores": [0.5],
            "convergence": {"converged": False, "step": 0, "type": "max_steps"},
            "steps_taken": 0,
        }
        summary = format_global_control_summary(result)
        assert "Did not converge" in summary

    def test_empty_result(self):
        """Empty result still produces valid output."""
        result = {
            "states": [],
            "interventions": [],
            "scores": [],
            "convergence": {"converged": False, "step": 0, "type": "max_steps"},
            "steps_taken": 0,
        }
        summary = format_global_control_summary(result)
        assert "=== Global Control ===" in summary


# ---------------------------------------------------------------------------
# Tests — Integration via strategy_adapter
# ---------------------------------------------------------------------------


class TestStrategyAdapterIntegration:
    """Tests for run_global_control_analysis from strategy_adapter."""

    def test_import(self):
        """run_global_control_analysis is importable from strategy_adapter."""
        from qec.analysis.strategy_adapter import run_global_control_analysis
        assert callable(run_global_control_analysis)

    def test_format_import(self):
        """format_global_control_summary is importable from strategy_adapter."""
        from qec.analysis.strategy_adapter import format_global_control_summary
        assert callable(format_global_control_summary)

    def test_basic_pipeline(self):
        """Full pipeline runs without error."""
        from qec.analysis.strategy_adapter import run_global_control_analysis

        runs = _make_runs()
        multistate = _make_multistate()
        coupled = _make_coupled_summary()

        result = run_global_control_analysis(
            runs,
            multistate_result={"multistate": multistate},
            coupled_result=coupled,
            max_steps=2,
        )
        assert "scores" in result
        assert "convergence" in result
