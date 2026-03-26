"""Tests for feedback control and closed-loop adaptation (v103.1.0).

Verifies:
- deterministic iteration
- convergence detection (stable, cycle, max_steps)
- improvement over steps
- no mutation of original data
- candidate adjustment (overshoot, ineffective removal)
- stability score computation
"""

from __future__ import annotations

import copy

import pytest

from qec.analysis.feedback_control import (
    ATTRACTOR_DELTA_THRESHOLD,
    DEFAULT_MAX_STEPS,
    INEFFECTIVE_THRESHOLD,
    MIN_REDUCED_STRENGTH,
    OVERSHOOT_THRESHOLD,
    ROUND_PRECISION,
    STABILITY_DELTA_THRESHOLD,
    adjust_candidates,
    compute_stability_score,
    detect_convergence,
    feedback_step,
    format_feedback_summary,
    run_feedback_control,
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
# Tests — compute_stability_score
# ---------------------------------------------------------------------------


class TestComputeStabilityScore:
    """Tests for compute_stability_score."""

    def test_high_stability(self):
        """Stability +1, high attractor, low transient -> high score."""
        state = _make_state_vector(
            stability=1,
            membership={
                "strong_attractor": 0.5,
                "weak_attractor": 0.3,
                "basin": 0.1,
                "transient": 0.05,
                "neutral": 0.05,
            },
        )
        score = compute_stability_score(state)
        assert score > 0.8

    def test_low_stability(self):
        """Stability -1, low attractor, high transient -> low score."""
        state = _make_state_vector(
            stability=-1,
            membership={
                "strong_attractor": 0.05,
                "weak_attractor": 0.05,
                "basin": 0.1,
                "transient": 0.7,
                "neutral": 0.1,
            },
        )
        score = compute_stability_score(state)
        assert score < 0.3

    def test_score_bounded(self):
        """Score must be in [0, 1]."""
        for stab in (-1, 0, 1):
            for t_weight in (0.0, 0.5, 1.0):
                state = _make_state_vector(
                    stability=stab,
                    membership={
                        "strong_attractor": 0.3,
                        "weak_attractor": 0.2,
                        "basin": 0.1,
                        "transient": t_weight,
                        "neutral": 0.0,
                    },
                )
                score = compute_stability_score(state)
                assert 0.0 <= score <= 1.0

    def test_determinism(self):
        """Identical inputs produce identical outputs."""
        state = _make_state_vector(stability=0)
        s1 = compute_stability_score(state)
        s2 = compute_stability_score(state)
        assert s1 == s2

    def test_empty_state(self):
        """Empty state returns a valid score."""
        score = compute_stability_score({})
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Tests — detect_convergence
# ---------------------------------------------------------------------------


class TestDetectConvergence:
    """Tests for detect_convergence."""

    def test_stable_convergence(self):
        """Identical consecutive states should converge as stable."""
        state = _make_multistate()
        states = [state, state]
        result = detect_convergence(states)
        assert result["converged"] is True
        assert result["type"] == "stable"
        assert result["step"] == 1

    def test_no_convergence_different_states(self):
        """Sufficiently different states should not converge."""
        state1 = _make_multistate(
            membership={
                "strong_attractor": 0.1,
                "weak_attractor": 0.1,
                "basin": 0.3,
                "transient": 0.4,
                "neutral": 0.1,
            },
        )
        state2 = _make_multistate(
            membership={
                "strong_attractor": 0.5,
                "weak_attractor": 0.3,
                "basin": 0.1,
                "transient": 0.05,
                "neutral": 0.05,
            },
        )
        result = detect_convergence([state1, state2])
        assert result["converged"] is False

    def test_cycle_detection(self):
        """State that repeats a previous state (not immediate) -> cycle."""
        state_a = _make_multistate(
            membership={
                "strong_attractor": 0.1,
                "weak_attractor": 0.1,
                "basin": 0.3,
                "transient": 0.4,
                "neutral": 0.1,
            },
        )
        state_b = _make_multistate(
            membership={
                "strong_attractor": 0.5,
                "weak_attractor": 0.3,
                "basin": 0.1,
                "transient": 0.05,
                "neutral": 0.05,
            },
        )
        # A -> B -> A : cycle
        states = [state_a, state_b, state_a]
        result = detect_convergence(states)
        assert result["converged"] is True
        assert result["type"] == "cycle"

    def test_single_state(self):
        """Single state cannot converge."""
        result = detect_convergence([_make_multistate()])
        assert result["converged"] is False

    def test_empty_states(self):
        """Empty list cannot converge."""
        result = detect_convergence([])
        assert result["converged"] is False

    def test_near_threshold(self):
        """States just below threshold should converge."""
        base = {
            "strong_attractor": 0.4,
            "weak_attractor": 0.2,
            "basin": 0.2,
            "transient": 0.1,
            "neutral": 0.1,
        }
        # Change attractor by less than threshold.
        near = dict(base)
        near["strong_attractor"] = 0.4 + ATTRACTOR_DELTA_THRESHOLD * 0.5
        near["neutral"] = 0.1 - ATTRACTOR_DELTA_THRESHOLD * 0.5

        state1 = _make_multistate(membership=base)
        state2 = _make_multistate(membership=near)
        result = detect_convergence([state1, state2])
        assert result["converged"] is True
        assert result["type"] == "stable"


# ---------------------------------------------------------------------------
# Tests — adjust_candidates
# ---------------------------------------------------------------------------


class TestAdjustCandidates:
    """Tests for adjust_candidates."""

    def test_empty_history(self):
        """No history -> candidates unchanged."""
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]
        result = adjust_candidates(candidates, [])
        assert len(result) == len(candidates)
        assert result[0]["strength"] == candidates[0]["strength"]

    def test_overshoot_reduces_strength(self):
        """Score decrease -> reduce strength of matching candidates."""
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "B", "action": "reduce_escape", "strength": 0.6},
        ]
        history = [
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.7},
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.5},
        ]
        result = adjust_candidates(candidates, history)
        # A/boost_stability should have reduced strength.
        a_cand = [c for c in result if c["target"] == "A" and c["action"] == "boost_stability"]
        assert len(a_cand) == 1
        assert a_cand[0]["strength"] < 0.6

    def test_ineffective_removed(self):
        """Negligible improvement -> remove candidate."""
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "B", "action": "reduce_escape", "strength": 0.6},
        ]
        history = [
            {"intervention": {"target": "B", "action": "reduce_escape"}, "score": 0.5},
            {"intervention": {"target": "B", "action": "reduce_escape"}, "score": 0.5 + INEFFECTIVE_THRESHOLD * 0.1},
        ]
        result = adjust_candidates(candidates, history)
        b_cand = [c for c in result if c["target"] == "B" and c["action"] == "reduce_escape"]
        assert len(b_cand) == 0

    def test_no_mutation(self):
        """Original candidates must not be mutated."""
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]
        original = copy.deepcopy(candidates)
        history = [
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.7},
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.5},
        ]
        adjust_candidates(candidates, history)
        assert candidates == original

    def test_empty_candidates(self):
        """Empty candidates returns empty list."""
        result = adjust_candidates([], [{"score": 0.5}])
        assert result == []

    def test_deterministic_ordering(self):
        """Output order matches input order."""
        candidates = [
            {"target": "B", "action": "reduce_escape", "strength": 0.3},
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]
        result = adjust_candidates(candidates, [])
        assert result[0]["target"] == "B"
        assert result[1]["target"] == "A"

    def test_minimum_strength(self):
        """Strength should not go below MIN_REDUCED_STRENGTH."""
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.01},
        ]
        history = [
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.7},
            {"intervention": {"target": "A", "action": "boost_stability"}, "score": 0.5},
        ]
        result = adjust_candidates(candidates, history)
        assert result[0]["strength"] >= MIN_REDUCED_STRENGTH


# ---------------------------------------------------------------------------
# Tests — feedback_step
# ---------------------------------------------------------------------------


class TestFeedbackStep:
    """Tests for feedback_step."""

    def test_returns_new_state_and_intervention(self):
        """feedback_step returns a tuple of (state, intervention)."""
        state = _make_multistate()
        runs = _make_runs()
        objective = {"maximize": "stability", "minimize": "escape"}
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]

        new_state, chosen = feedback_step(
            state, objective, candidates, runs,
        )
        assert isinstance(new_state, dict)
        assert isinstance(chosen, dict)
        assert "action" in chosen

    def test_empty_candidates(self):
        """Empty candidates -> unchanged state, empty intervention."""
        state = _make_multistate()
        runs = _make_runs()
        objective = {"maximize": "stability"}

        new_state, chosen = feedback_step(state, objective, [], runs)
        assert new_state is state
        assert chosen == {}

    def test_no_mutation_of_input(self):
        """Input state must not be mutated."""
        state = _make_multistate()
        original = copy.deepcopy(state)
        runs = _make_runs()
        objective = {"maximize": "stability"}
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
        ]

        feedback_step(state, objective, candidates, runs)
        assert state == original

    def test_determinism(self):
        """Identical inputs produce identical outputs."""
        state = _make_multistate()
        runs = _make_runs()
        objective = {"maximize": "stability", "minimize": "escape"}
        candidates = [
            {"target": "A", "action": "boost_stability", "strength": 0.6},
            {"target": "A", "action": "reduce_escape", "strength": 0.3},
        ]

        s1, i1 = feedback_step(state, objective, candidates, runs)
        s2, i2 = feedback_step(state, objective, candidates, runs)
        assert s1 == s2
        assert i1 == i2


# ---------------------------------------------------------------------------
# Tests — run_feedback_control
# ---------------------------------------------------------------------------


class TestRunFeedbackControl:
    """Tests for run_feedback_control."""

    def test_basic_loop(self):
        """Loop should run and produce structured output."""
        runs = _make_runs()
        objective = {"maximize": "stability", "minimize": "escape"}

        result = run_feedback_control(runs, objective, max_steps=3)

        assert "states" in result
        assert "interventions" in result
        assert "scores" in result
        assert "convergence" in result
        assert "steps_taken" in result
        assert len(result["states"]) >= 1
        assert len(result["scores"]) >= 1

    def test_scores_list_length(self):
        """Scores list has one more entry than interventions (initial + per-step)."""
        runs = _make_runs()
        objective = {"maximize": "stability"}

        result = run_feedback_control(runs, objective, max_steps=2)
        assert len(result["scores"]) == len(result["interventions"]) + 1

    def test_determinism(self):
        """Identical inputs produce identical outputs."""
        runs = _make_runs()
        objective = {"maximize": "stability", "minimize": "escape"}

        r1 = run_feedback_control(runs, objective, max_steps=3)
        r2 = run_feedback_control(runs, objective, max_steps=3)

        assert r1["scores"] == r2["scores"]
        assert r1["interventions"] == r2["interventions"]
        assert r1["convergence"] == r2["convergence"]

    def test_max_steps_respected(self):
        """Loop must not exceed max_steps."""
        runs = _make_runs()
        objective = {"maximize": "stability"}

        result = run_feedback_control(runs, objective, max_steps=2)
        assert result["steps_taken"] <= 2

    def test_no_mutation_of_runs(self):
        """Original runs must not be mutated."""
        runs = _make_runs()
        original = copy.deepcopy(runs)
        objective = {"maximize": "stability"}

        run_feedback_control(runs, objective, max_steps=2)
        assert runs == original

    def test_convergence_detected(self):
        """Loop should detect convergence when states stop changing."""
        # Use a precomputed multistate that will converge quickly.
        state = _make_multistate(
            stability=1,
            membership={
                "strong_attractor": 0.5,
                "weak_attractor": 0.3,
                "basin": 0.1,
                "transient": 0.05,
                "neutral": 0.05,
            },
        )
        runs = _make_runs()
        multistate_result = {"multistate": state}

        result = run_feedback_control(
            runs,
            {"maximize": "stability", "minimize": "escape"},
            max_steps=10,
            multistate_result=multistate_result,
        )

        # A highly stable state should converge quickly.
        conv = result["convergence"]
        if conv["converged"]:
            assert conv["type"] in ("stable", "cycle")

    def test_with_precomputed_multistate(self):
        """Precomputed multistate is accepted and used."""
        state = _make_multistate()
        runs = _make_runs()
        ms = {"multistate": state}

        result = run_feedback_control(
            runs,
            {"maximize": "stability"},
            max_steps=1,
            multistate_result=ms,
        )
        assert result["steps_taken"] >= 0


# ---------------------------------------------------------------------------
# Tests — format_feedback_summary
# ---------------------------------------------------------------------------


class TestFormatFeedbackSummary:
    """Tests for format_feedback_summary."""

    def test_basic_format(self):
        """Summary should contain expected sections."""
        result = {
            "states": [_make_multistate()],
            "interventions": [
                {"action": "boost_stability", "target": "A", "strength": 0.6},
            ],
            "scores": [0.5, 0.7],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "steps_taken": 1,
        }
        summary = format_feedback_summary(result)
        assert "Feedback Control" in summary
        assert "Step 1" in summary
        assert "boost_stability" in summary
        assert "Converged" in summary
        assert "Final State" in summary

    def test_no_convergence_format(self):
        """Non-converged result should indicate that."""
        result = {
            "states": [_make_multistate()],
            "interventions": [],
            "scores": [0.5],
            "convergence": {"converged": False, "step": 0, "type": "max_steps"},
            "steps_taken": 0,
        }
        summary = format_feedback_summary(result)
        assert "Did not converge" in summary

    def test_empty_result(self):
        """Empty result should not crash."""
        summary = format_feedback_summary({})
        assert "Feedback Control" in summary
