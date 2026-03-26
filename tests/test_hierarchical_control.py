"""Tests for hierarchical control and policy routing (v103.3.0).

Verifies:
- deterministic routing for all policy modes
- policy switching correctness (stability, sync, balanced)
- merge logic correctness (dedup, conflict resolution)
- convergence detection (stable, oscillation, max_steps)
- no mutation of inputs
- built-in policy construction
- format output
"""

from __future__ import annotations

import copy

from qec.analysis.hierarchical_control import (
    ACTION_PRIORITY,
    DEFAULT_INSTABILITY_THRESHOLD,
    DEFAULT_MAX_STEPS,
    DEFAULT_SYNC_THRESHOLD,
    MODE_STABILITY_WINDOW,
    POLICY_BALANCED,
    POLICY_STABILITY_FIRST,
    POLICY_SYNC_FIRST,
    ROUND_PRECISION,
    SCORE_DELTA_THRESHOLD,
    detect_hierarchical_convergence,
    evaluate_control_policy,
    format_hierarchical_control_summary,
    get_builtin_policy,
    merge_interventions,
    route_control,
)


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


def _make_global_state(avg_stability=0.5, avg_sync=0.5):
    """Build a global state dict."""
    return {
        "avg_stability": avg_stability,
        "avg_sync": avg_sync,
        "coupled_summary": {},
    }


def _make_intervention(target="A", action="boost_stability", strength=0.6):
    """Build an intervention dict."""
    return {"target": target, "action": action, "strength": strength}


# ---------------------------------------------------------------------------
# Tests — evaluate_control_policy
# ---------------------------------------------------------------------------


class TestEvaluateControlPolicy:
    """Tests for evaluate_control_policy."""

    def test_fixed_local_mode(self):
        policy = {"mode": "local", "priority": "balanced", "thresholds": {}}
        result = evaluate_control_policy({}, {}, policy)
        assert result == "local"

    def test_fixed_global_mode(self):
        policy = {"mode": "global", "priority": "balanced", "thresholds": {}}
        result = evaluate_control_policy({}, {}, policy)
        assert result == "global"

    def test_stability_priority_stable(self):
        """High stability -> local."""
        policy = get_builtin_policy(POLICY_STABILITY_FIRST)
        global_state = _make_global_state(avg_stability=0.8, avg_sync=0.3)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "local"

    def test_stability_priority_unstable(self):
        """Low stability -> global."""
        policy = get_builtin_policy(POLICY_STABILITY_FIRST)
        global_state = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "global"

    def test_sync_priority_high_sync(self):
        """High sync -> local."""
        policy = get_builtin_policy(POLICY_SYNC_FIRST)
        global_state = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "local"

    def test_sync_priority_low_sync(self):
        """Low sync -> global."""
        policy = get_builtin_policy(POLICY_SYNC_FIRST)
        global_state = _make_global_state(avg_stability=0.8, avg_sync=0.3)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "global"

    def test_balanced_both_low(self):
        """Both low -> global."""
        policy = get_builtin_policy(POLICY_BALANCED)
        global_state = _make_global_state(avg_stability=0.3, avg_sync=0.3)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "global"

    def test_balanced_one_low(self):
        """One metric low -> hybrid."""
        policy = get_builtin_policy(POLICY_BALANCED)
        global_state = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "hybrid"

    def test_balanced_both_high(self):
        """Both high -> local."""
        policy = get_builtin_policy(POLICY_BALANCED)
        global_state = _make_global_state(avg_stability=0.8, avg_sync=0.8)
        result = evaluate_control_policy({}, global_state, policy)
        assert result == "local"

    def test_deterministic(self):
        """Same inputs always produce same output."""
        policy = get_builtin_policy(POLICY_BALANCED)
        global_state = _make_global_state(avg_stability=0.45, avg_sync=0.55)
        results = [
            evaluate_control_policy({}, global_state, policy)
            for _ in range(50)
        ]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        """Inputs must not be mutated."""
        state = _make_multistate("A", "B")
        global_state = _make_global_state()
        policy = get_builtin_policy(POLICY_BALANCED)
        state_copy = copy.deepcopy(state)
        global_copy = copy.deepcopy(global_state)
        policy_copy = copy.deepcopy(policy)
        evaluate_control_policy(state, global_state, policy)
        assert state == state_copy
        assert global_state == global_copy
        assert policy == policy_copy


# ---------------------------------------------------------------------------
# Tests — route_control
# ---------------------------------------------------------------------------


class TestRouteControl:
    """Tests for route_control."""

    def test_local_mode(self):
        local = {"interventions": [_make_intervention("A")], "score": 0.8}
        global_ = {"interventions": [_make_intervention("B")], "score": 0.6}
        result = route_control(local, global_, {"mode": "local"})
        assert result["mode"] == "local"
        assert len(result["interventions"]) == 1
        assert result["interventions"][0]["target"] == "A"
        assert result["score"] == 0.8

    def test_global_mode(self):
        local = {"interventions": [_make_intervention("A")], "score": 0.8}
        global_ = {"interventions": [_make_intervention("B")], "score": 0.6}
        result = route_control(local, global_, {"mode": "global"})
        assert result["mode"] == "global"
        assert len(result["interventions"]) == 1
        assert result["interventions"][0]["target"] == "B"
        assert result["score"] == 0.6

    def test_hybrid_merges(self):
        local = {"interventions": [_make_intervention("A")], "score": 0.8}
        global_ = {"interventions": [_make_intervention("B")], "score": 0.6}
        result = route_control(local, global_, {"mode": "hybrid"})
        assert result["mode"] == "hybrid"
        assert len(result["interventions"]) == 2
        targets = {i["target"] for i in result["interventions"]}
        assert targets == {"A", "B"}
        assert result["score"] == 0.8  # max of local and global

    def test_empty_interventions(self):
        local = {"interventions": [], "score": 0.0}
        global_ = {"interventions": [], "score": 0.0}
        result = route_control(local, global_, {"mode": "hybrid"})
        assert result["interventions"] == []

    def test_deterministic(self):
        local = {"interventions": [_make_intervention("A")], "score": 0.7}
        global_ = {"interventions": [_make_intervention("B")], "score": 0.5}
        results = [
            route_control(local, global_, {"mode": "hybrid"})
            for _ in range(50)
        ]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        local = {"interventions": [_make_intervention("A")], "score": 0.8}
        global_ = {"interventions": [_make_intervention("B")], "score": 0.6}
        local_copy = copy.deepcopy(local)
        global_copy = copy.deepcopy(global_)
        route_control(local, global_, {"mode": "hybrid"})
        assert local == local_copy
        assert global_ == global_copy


# ---------------------------------------------------------------------------
# Tests — merge_interventions
# ---------------------------------------------------------------------------


class TestMergeInterventions:
    """Tests for merge_interventions."""

    def test_no_overlap(self):
        local = [_make_intervention("A", "boost_stability", 0.6)]
        global_ = [_make_intervention("B", "reduce_escape", 0.3)]
        result = merge_interventions(local, global_)
        assert len(result) == 2
        assert result[0]["target"] == "A"
        assert result[1]["target"] == "B"

    def test_conflict_same_target_different_action(self):
        """Same target, different actions: higher priority wins."""
        local = [_make_intervention("A", "reduce_escape", 0.9)]
        global_ = [_make_intervention("A", "boost_stability", 0.3)]
        result = merge_interventions(local, global_)
        assert len(result) == 1
        assert result[0]["action"] == "boost_stability"  # higher priority

    def test_conflict_same_target_same_action(self):
        """Same target, same action: higher strength wins."""
        local = [_make_intervention("A", "boost_stability", 0.3)]
        global_ = [_make_intervention("A", "boost_stability", 0.9)]
        result = merge_interventions(local, global_)
        assert len(result) == 1
        assert result[0]["strength"] == 0.9

    def test_empty_inputs(self):
        assert merge_interventions([], []) == []

    def test_one_side_empty(self):
        local = [_make_intervention("A")]
        result = merge_interventions(local, [])
        assert len(result) == 1
        assert result[0]["target"] == "A"

    def test_sorted_by_target(self):
        local = [_make_intervention("C")]
        global_ = [_make_intervention("A"), _make_intervention("B")]
        result = merge_interventions(local, global_)
        targets = [i["target"] for i in result]
        assert targets == ["A", "B", "C"]

    def test_deterministic(self):
        local = [_make_intervention("A", "boost_stability", 0.6)]
        global_ = [_make_intervention("A", "reduce_escape", 0.3)]
        results = [merge_interventions(local, global_) for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        local = [_make_intervention("A")]
        global_ = [_make_intervention("B")]
        local_copy = copy.deepcopy(local)
        global_copy = copy.deepcopy(global_)
        merge_interventions(local, global_)
        assert local == local_copy
        assert global_ == global_copy


# ---------------------------------------------------------------------------
# Tests — detect_hierarchical_convergence
# ---------------------------------------------------------------------------


class TestDetectHierarchicalConvergence:
    """Tests for detect_hierarchical_convergence."""

    def test_too_few_scores(self):
        result = detect_hierarchical_convergence([], [0.5])
        assert not result["converged"]
        assert result["type"] == "max_steps"

    def test_stable_convergence(self):
        states = [
            {"mode": "hybrid"},
            {"mode": "hybrid"},
            {"mode": "hybrid"},
        ]
        scores = [0.5, 0.505, 0.508]  # small deltas
        result = detect_hierarchical_convergence(states, scores)
        assert result["converged"]
        assert result["type"] == "stable"

    def test_no_convergence_large_deltas(self):
        states = [
            {"mode": "local"},
            {"mode": "global"},
        ]
        scores = [0.3, 0.8]
        result = detect_hierarchical_convergence(states, scores)
        assert not result["converged"]
        assert result["type"] == "max_steps"

    def test_oscillation_detection(self):
        states = [
            {"mode": "local"},
            {"mode": "global"},
            {"mode": "local"},
            {"mode": "global"},
        ]
        scores = [0.3, 0.8, 0.3, 0.8]
        result = detect_hierarchical_convergence(states, scores)
        assert result["converged"]
        assert result["type"] == "oscillation"

    def test_deterministic(self):
        states = [{"mode": "hybrid"}, {"mode": "hybrid"}]
        scores = [0.5, 0.505]
        results = [
            detect_hierarchical_convergence(states, scores)
            for _ in range(50)
        ]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        states = [{"mode": "hybrid"}, {"mode": "hybrid"}]
        scores = [0.5, 0.505]
        states_copy = copy.deepcopy(states)
        scores_copy = list(scores)
        detect_hierarchical_convergence(states, scores)
        assert states == states_copy
        assert scores == scores_copy


# ---------------------------------------------------------------------------
# Tests — get_builtin_policy
# ---------------------------------------------------------------------------


class TestGetBuiltinPolicy:
    """Tests for get_builtin_policy."""

    def test_stability_first(self):
        policy = get_builtin_policy(POLICY_STABILITY_FIRST)
        assert policy["mode"] == "hybrid"
        assert policy["priority"] == "stability"
        assert "thresholds" in policy

    def test_sync_first(self):
        policy = get_builtin_policy(POLICY_SYNC_FIRST)
        assert policy["mode"] == "hybrid"
        assert policy["priority"] == "synchronization"

    def test_balanced(self):
        policy = get_builtin_policy(POLICY_BALANCED)
        assert policy["mode"] == "hybrid"
        assert policy["priority"] == "balanced"

    def test_unknown_raises(self):
        try:
            get_builtin_policy("unknown_policy")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_deterministic(self):
        results = [get_builtin_policy(POLICY_BALANCED) for _ in range(50)]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Tests — format_hierarchical_control_summary
# ---------------------------------------------------------------------------


class TestFormatHierarchicalControlSummary:
    """Tests for format_hierarchical_control_summary."""

    def test_basic_format(self):
        result = {
            "states": [
                {"mode": "initial", "multistate": {}},
                {"mode": "hybrid", "multistate": {}},
            ],
            "local_actions": [
                {"interventions": [_make_intervention("A")], "score": 0.7},
            ],
            "global_actions": [
                [_make_intervention("B", "reduce_escape", 0.3)],
            ],
            "final_actions": [
                {
                    "interventions": [
                        _make_intervention("A"),
                        _make_intervention("B", "reduce_escape", 0.3),
                    ],
                    "score": 0.7,
                    "mode": "hybrid",
                },
            ],
            "scores": [0.5, 0.7],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "steps_taken": 1,
        }
        summary = format_hierarchical_control_summary(result)
        assert "=== Hierarchical Control ===" in summary
        assert "Step 1:" in summary
        assert "Local:" in summary
        assert "Global:" in summary
        assert "Final:" in summary
        assert "Mode: hybrid" in summary
        assert "Final Score:" in summary
        assert "Converged: stable" in summary

    def test_empty_result(self):
        result = {
            "states": [],
            "local_actions": [],
            "global_actions": [],
            "final_actions": [],
            "scores": [],
            "convergence": {"converged": False, "step": 0, "type": "max_steps"},
            "steps_taken": 0,
        }
        summary = format_hierarchical_control_summary(result)
        assert "=== Hierarchical Control ===" in summary

    def test_deterministic(self):
        result = {
            "states": [{"mode": "initial"}, {"mode": "hybrid"}],
            "local_actions": [{"interventions": [], "score": 0.0}],
            "global_actions": [[]],
            "final_actions": [{"interventions": [], "score": 0.0, "mode": "hybrid"}],
            "scores": [0.5, 0.5],
            "convergence": {"converged": True, "step": 1, "type": "stable"},
            "steps_taken": 1,
        }
        results = [
            format_hierarchical_control_summary(result) for _ in range(50)
        ]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Tests — Integration via strategy_adapter
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests via strategy_adapter functions."""

    def _make_run(self, strategies):
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

    def test_adapter_function_exists(self):
        """run_hierarchical_control_analysis should be importable."""
        from qec.analysis.strategy_adapter import (
            run_hierarchical_control_analysis,
        )
        assert callable(run_hierarchical_control_analysis)

    def test_format_function_exists(self):
        """format_hierarchical_control_summary should be importable."""
        from qec.analysis.strategy_adapter import (
            format_hierarchical_control_summary,
        )
        assert callable(format_hierarchical_control_summary)

    def test_full_pipeline(self):
        """Full pipeline: runs -> hierarchical control."""
        from qec.analysis.strategy_adapter import (
            run_hierarchical_control_analysis,
        )

        strategies = [("A", 0.8), ("B", 0.5), ("C", 0.3)]
        runs = [self._make_run(strategies) for _ in range(3)]

        result = run_hierarchical_control_analysis(runs)
        assert "states" in result
        assert "local_actions" in result
        assert "global_actions" in result
        assert "final_actions" in result
        assert "scores" in result
        assert "convergence" in result
        assert "steps_taken" in result
        assert len(result["scores"]) >= 1

    def test_full_pipeline_deterministic(self):
        """Pipeline must be deterministic."""
        from qec.analysis.strategy_adapter import (
            run_hierarchical_control_analysis,
        )

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [self._make_run(strategies) for _ in range(3)]

        r1 = run_hierarchical_control_analysis(runs)
        r2 = run_hierarchical_control_analysis(runs)
        assert r1["scores"] == r2["scores"]
        assert r1["steps_taken"] == r2["steps_taken"]
        assert r1["convergence"] == r2["convergence"]

    def test_full_pipeline_no_mutation(self):
        """Pipeline must not mutate inputs."""
        from qec.analysis.strategy_adapter import (
            run_hierarchical_control_analysis,
        )

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [self._make_run(strategies) for _ in range(3)]
        runs_copy = copy.deepcopy(runs)

        run_hierarchical_control_analysis(runs)
        assert runs == runs_copy

    def test_with_custom_policy(self):
        """Pipeline works with custom policies."""
        from qec.analysis.hierarchical_control import get_builtin_policy
        from qec.analysis.strategy_adapter import (
            run_hierarchical_control_analysis,
        )

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [self._make_run(strategies) for _ in range(3)]

        for policy_name in [POLICY_STABILITY_FIRST, POLICY_SYNC_FIRST, POLICY_BALANCED]:
            policy = get_builtin_policy(policy_name)
            result = run_hierarchical_control_analysis(runs, policy=policy)
            assert "scores" in result
            assert "convergence" in result
