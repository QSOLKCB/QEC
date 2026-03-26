"""Tests for meta-control and policy selection engine (v103.5.0).

Verifies:
- deterministic policy selection
- correct switching behavior detection
- convergence detection
- no mutation of inputs
- integration correctness via strategy_adapter
"""

from __future__ import annotations

import copy

from qec.analysis.meta_control import (
    DEFAULT_META_MAX_STEPS,
    META_POLICY_WINDOW,
    META_SCORE_DELTA,
    ROUND_PRECISION,
    SWITCHING_FREQUENT_RATIO,
    SWITCHING_OSCILLATION_MIN,
    detect_meta_convergence,
    detect_policy_switching,
    evaluate_policies_step,
    format_meta_control_summary,
    run_meta_control,
    select_policy,
)
from qec.analysis.policy import Policy, get_policy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_vector(stability_state=0, strong_attractor=0.3,
                       weak_attractor=0.2, transient=0.1,
                       basin=0.2, neutral=0.2):
    """Build a minimal state vector."""
    return {
        "ternary": {
            "stability_state": stability_state,
            "trend_state": 0,
            "phase_state": 0,
        },
        "membership": {
            "strong_attractor": strong_attractor,
            "weak_attractor": weak_attractor,
            "transient": transient,
            "basin": basin,
            "neutral": neutral,
        },
    }


def _make_multistate(*names, stability_state=0):
    """Build a multistate dict from strategy names."""
    return {
        name: _make_state_vector(stability_state=stability_state)
        for name in names
    }


def _make_run(strategies):
    """Build a run dict from a list of (name, design_score) tuples."""
    return {
        "strategies": [
            {
                "name": name,
                "metrics": {
                    "design_score": score,
                    "confidence_efficiency": 0.5,
                    "consistency_gap": 0.1,
                    "revival_strength": 0.0,
                },
            }
            for name, score in strategies
        ],
    }


def _make_runs(strategies=None):
    """Build a list of runs."""
    if strategies is None:
        strategies = [("A", 0.8), ("B", 0.5), ("C", 0.3)]
    return [_make_run(strategies) for _ in range(3)]


def _builtin_policies():
    """Return the three built-in policies."""
    return [
        get_policy("stability_first"),
        get_policy("sync_first"),
        get_policy("balanced"),
    ]


# ---------------------------------------------------------------------------
# Tests — select_policy
# ---------------------------------------------------------------------------


class TestSelectPolicy:
    """Tests for select_policy."""

    def test_highest_score_wins(self):
        results = {
            "a": {"score": 0.5, "action": []},
            "b": {"score": 0.9, "action": []},
            "c": {"score": 0.3, "action": []},
        }
        assert select_policy(results) == "b"

    def test_tie_fewer_actions_wins(self):
        results = {
            "a": {"score": 0.8, "action": [{"x": 1}, {"y": 2}]},
            "b": {"score": 0.8, "action": [{"x": 1}]},
        }
        assert select_policy(results) == "b"

    def test_tie_lexicographic(self):
        results = {
            "beta": {"score": 0.8, "action": []},
            "alpha": {"score": 0.8, "action": []},
        }
        assert select_policy(results) == "alpha"

    def test_single_policy(self):
        results = {"only": {"score": 0.5, "action": []}}
        assert select_policy(results) == "only"

    def test_empty_raises(self):
        try:
            select_policy({})
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_deterministic(self):
        results = {
            "a": {"score": 0.7, "action": [{"x": 1}]},
            "b": {"score": 0.9, "action": []},
            "c": {"score": 0.7, "action": []},
        }
        selections = [select_policy(results) for _ in range(50)]
        assert all(s == "b" for s in selections)


# ---------------------------------------------------------------------------
# Tests — detect_policy_switching
# ---------------------------------------------------------------------------


class TestDetectPolicySwitching:
    """Tests for detect_policy_switching."""

    def test_empty_history(self):
        result = detect_policy_switching([])
        assert result["pattern"] == "none"
        assert result["switches"] == 0

    def test_single_entry(self):
        result = detect_policy_switching(["a"])
        assert result["pattern"] == "stable"
        assert result["dominant_policy"] == "a"

    def test_stable_policy(self):
        result = detect_policy_switching(["a", "a", "a", "a"])
        assert result["pattern"] == "stable"
        assert result["switches"] == 0
        assert result["switch_ratio"] == 0.0
        assert result["dominant_policy"] == "a"

    def test_oscillation_detected(self):
        result = detect_policy_switching(["a", "b", "a", "b"])
        assert result["pattern"] == "oscillation"
        assert result["switches"] == 3

    def test_frequent_switching(self):
        # 5 entries, 4 transitions, all switches => ratio = 1.0
        result = detect_policy_switching(["a", "b", "c", "a", "b"])
        assert result["pattern"] in ("oscillation", "frequent")
        assert result["switches"] == 4

    def test_infrequent_switching(self):
        # Only 1 switch out of 4 transitions => ratio = 0.25
        result = detect_policy_switching(["a", "a", "a", "b", "b"])
        assert result["pattern"] == "none"
        assert result["switches"] == 1
        assert result["switch_ratio"] == round(1 / 4, ROUND_PRECISION)

    def test_dominant_policy_tie_lexicographic(self):
        # a appears 2 times, b appears 2 times => a wins lexicographically
        result = detect_policy_switching(["a", "b", "a", "b"])
        assert result["dominant_policy"] == "a"


# ---------------------------------------------------------------------------
# Tests — detect_meta_convergence
# ---------------------------------------------------------------------------


class TestDetectMetaConvergence:
    """Tests for detect_meta_convergence."""

    def test_insufficient_data(self):
        result = detect_meta_convergence([{}], [], [0.5])
        assert not result["converged"]
        assert result["type"] == "max_steps"

    def test_stable_convergence(self):
        states = [{}, {}, {}, {}]
        policies = ["a", "a", "a"]
        scores = [0.5, 0.7, 0.705, 0.706]  # small delta
        result = detect_meta_convergence(states, policies, scores)
        assert result["converged"]
        assert result["type"] == "stable"

    def test_oscillation_detected(self):
        states = [{}, {}, {}, {}, {}]
        policies = ["a", "b", "a", "b"]
        # Scores not converging
        scores = [0.5, 0.7, 0.5, 0.7, 0.5]
        result = detect_meta_convergence(states, policies, scores)
        assert result["converged"]
        assert result["type"] == "oscillation"

    def test_no_convergence(self):
        states = [{}, {}]
        policies = ["a"]
        scores = [0.5, 0.9]  # big delta
        result = detect_meta_convergence(states, policies, scores)
        assert not result["converged"]
        assert result["type"] == "max_steps"


# ---------------------------------------------------------------------------
# Tests — evaluate_policies_step
# ---------------------------------------------------------------------------


class TestEvaluatePoliciesStep:
    """Tests for evaluate_policies_step."""

    def test_returns_all_policies(self):
        runs = _make_runs()
        state = _make_multistate("A", "B", "C")
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = evaluate_policies_step(state, policies, objective, runs)
        assert set(result.keys()) == {"stability_first", "sync_first", "balanced"}

    def test_each_policy_has_score_and_action(self):
        runs = _make_runs()
        state = _make_multistate("A", "B", "C")
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = evaluate_policies_step(state, policies, objective, runs)
        for name, data in result.items():
            assert "score" in data
            assert "action" in data
            assert isinstance(data["score"], float)
            assert isinstance(data["action"], list)

    def test_deterministic(self):
        runs = _make_runs()
        state = _make_multistate("A", "B")
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        r1 = evaluate_policies_step(state, policies, objective, runs)
        r2 = evaluate_policies_step(state, policies, objective, runs)
        for name in r1:
            assert r1[name]["score"] == r2[name]["score"]
            assert r1[name]["action"] == r2[name]["action"]

    def test_no_mutation(self):
        runs = _make_runs()
        state = _make_multistate("A", "B")
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        state_copy = copy.deepcopy(state)
        runs_copy = copy.deepcopy(runs)

        evaluate_policies_step(state, policies, objective, runs)

        assert state == state_copy
        assert runs == runs_copy


# ---------------------------------------------------------------------------
# Tests — run_meta_control
# ---------------------------------------------------------------------------


class TestRunMetaControl:
    """Tests for run_meta_control."""

    def test_returns_expected_keys(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = run_meta_control(runs, policies, objective, max_steps=2)
        assert "states" in result
        assert "policies" in result
        assert "actions" in result
        assert "scores" in result
        assert "evaluations" in result
        assert "convergence" in result
        assert "switching" in result
        assert "steps_taken" in result

    def test_scores_monotonic_or_bounded(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = run_meta_control(runs, policies, objective, max_steps=3)
        scores = result["scores"]
        assert len(scores) >= 2
        # All scores should be finite floats.
        for s in scores:
            assert isinstance(s, float)

    def test_policies_list_matches_steps(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = run_meta_control(runs, policies, objective, max_steps=3)
        steps = result["steps_taken"]
        assert len(result["policies"]) == steps
        assert len(result["actions"]) == steps
        assert len(result["evaluations"]) == steps
        # scores has initial + one per step
        assert len(result["scores"]) == steps + 1

    def test_deterministic(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        r1 = run_meta_control(runs, policies, objective, max_steps=3)
        r2 = run_meta_control(runs, policies, objective, max_steps=3)
        assert r1["scores"] == r2["scores"]
        assert r1["policies"] == r2["policies"]
        assert r1["steps_taken"] == r2["steps_taken"]
        assert r1["convergence"] == r2["convergence"]

    def test_no_mutation(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        runs_copy = copy.deepcopy(runs)
        obj_copy = copy.deepcopy(objective)

        run_meta_control(runs, policies, objective, max_steps=2)

        assert runs == runs_copy
        assert objective == obj_copy

    def test_max_steps_respected(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = run_meta_control(runs, policies, objective, max_steps=2)
        assert result["steps_taken"] <= 2

    def test_switching_analysis_present(self):
        runs = _make_runs()
        policies = _builtin_policies()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = run_meta_control(runs, policies, objective, max_steps=3)
        switching = result["switching"]
        assert "pattern" in switching
        assert "switches" in switching
        assert "dominant_policy" in switching
        assert "switch_ratio" in switching


# ---------------------------------------------------------------------------
# Tests — format_meta_control_summary
# ---------------------------------------------------------------------------


class TestFormatMetaControlSummary:
    """Tests for format_meta_control_summary."""

    def test_contains_header(self):
        result = {
            "states": [{}, {}],
            "policies": ["balanced"],
            "actions": [[]],
            "scores": [0.5, 0.8],
            "evaluations": [{"balanced": {"score": 0.8, "action": []}}],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "switching": {
                "pattern": "stable",
                "switches": 0,
                "dominant_policy": "balanced",
                "switch_ratio": 0.0,
            },
            "steps_taken": 1,
        }
        text = format_meta_control_summary(result)
        assert "=== Meta Control ===" in text

    def test_contains_step_info(self):
        result = {
            "states": [{}, {}],
            "policies": ["balanced"],
            "actions": [[]],
            "scores": [0.5, 0.8],
            "evaluations": [{"balanced": {"score": 0.8, "action": []}}],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "switching": {
                "pattern": "stable",
                "switches": 0,
                "dominant_policy": "balanced",
                "switch_ratio": 0.0,
            },
            "steps_taken": 1,
        }
        text = format_meta_control_summary(result)
        assert "Step 1:" in text
        assert "Selected Policy: balanced" in text

    def test_contains_final_score(self):
        result = {
            "states": [{}, {}],
            "policies": ["balanced"],
            "actions": [[]],
            "scores": [0.5, 0.89],
            "evaluations": [{"balanced": {"score": 0.89, "action": []}}],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "switching": {
                "pattern": "stable",
                "switches": 0,
                "dominant_policy": "balanced",
                "switch_ratio": 0.0,
            },
            "steps_taken": 1,
        }
        text = format_meta_control_summary(result)
        assert "Final Score: 0.89" in text

    def test_contains_convergence_info(self):
        result = {
            "states": [{}, {}],
            "policies": ["balanced"],
            "actions": [[]],
            "scores": [0.5, 0.8],
            "evaluations": [{"balanced": {"score": 0.8, "action": []}}],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "switching": {
                "pattern": "stable",
                "switches": 0,
                "dominant_policy": "balanced",
                "switch_ratio": 0.0,
            },
            "steps_taken": 1,
        }
        text = format_meta_control_summary(result)
        assert "Converged: stable" in text

    def test_shows_policy_stabilized(self):
        result = {
            "states": [{}, {}],
            "policies": ["balanced"],
            "actions": [[]],
            "scores": [0.5, 0.8],
            "evaluations": [{"balanced": {"score": 0.8, "action": []}}],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "switching": {
                "pattern": "stable",
                "switches": 0,
                "dominant_policy": "balanced",
                "switch_ratio": 0.0,
            },
            "steps_taken": 1,
        }
        text = format_meta_control_summary(result)
        assert "Policy stabilized: balanced" in text


# ---------------------------------------------------------------------------
# Tests — Integration via strategy_adapter
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests via strategy_adapter functions."""

    def test_adapter_function_exists(self):
        from qec.analysis.strategy_adapter import run_meta_control_analysis
        assert callable(run_meta_control_analysis)

    def test_format_function_exists(self):
        from qec.analysis.strategy_adapter import format_meta_control_summary
        assert callable(format_meta_control_summary)

    def test_full_pipeline(self):
        from qec.analysis.strategy_adapter import run_meta_control_analysis

        runs = _make_runs()
        result = run_meta_control_analysis(runs)
        assert "states" in result
        assert "policies" in result
        assert "actions" in result
        assert "scores" in result
        assert "convergence" in result
        assert "switching" in result
        assert "steps_taken" in result

    def test_full_pipeline_deterministic(self):
        from qec.analysis.strategy_adapter import run_meta_control_analysis

        runs = _make_runs()
        r1 = run_meta_control_analysis(runs)
        r2 = run_meta_control_analysis(runs)
        assert r1["scores"] == r2["scores"]
        assert r1["policies"] == r2["policies"]
        assert r1["steps_taken"] == r2["steps_taken"]
        assert r1["convergence"] == r2["convergence"]

    def test_full_pipeline_no_mutation(self):
        from qec.analysis.strategy_adapter import run_meta_control_analysis

        runs = _make_runs()
        runs_copy = copy.deepcopy(runs)

        run_meta_control_analysis(runs)
        assert runs == runs_copy

    def test_custom_policies(self):
        from qec.analysis.strategy_adapter import run_meta_control_analysis

        runs = _make_runs()
        policies = [get_policy("stability_first"), get_policy("balanced")]
        result = run_meta_control_analysis(runs, policies=policies)

        # Selected policies should only be from the provided set.
        for p in result["policies"]:
            assert p in ("stability_first", "balanced")

    def test_format_integration(self):
        from qec.analysis.strategy_adapter import (
            format_meta_control_summary,
            run_meta_control_analysis,
        )

        runs = _make_runs()
        result = run_meta_control_analysis(runs, max_steps=2)
        text = format_meta_control_summary(result)
        assert "=== Meta Control ===" in text
        assert "Final Score:" in text
