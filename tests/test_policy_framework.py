"""Tests for policy framework and experiment registry (v103.4.0).

Verifies:
- Policy object behavior matches old dict logic
- Policy composition is deterministic
- Policy registry works (register, get, list)
- Experiment results are deterministic
- Ranking correctness
- No mutation of inputs
"""

from __future__ import annotations

import copy

from qec.analysis.hierarchical_control import (
    DEFAULT_INSTABILITY_THRESHOLD,
    DEFAULT_SYNC_THRESHOLD,
    POLICY_BALANCED,
    POLICY_STABILITY_FIRST,
    POLICY_SYNC_FIRST,
    evaluate_control_policy,
    get_builtin_policy,
)
from qec.analysis.policy import (
    Policy,
    compose_policies,
    get_policy,
    list_policies,
    register_policy,
)
from qec.analysis.experiment_registry import (
    format_policy_experiment_summary,
    rank_policies,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_global_state(avg_stability=0.5, avg_sync=0.5):
    return {
        "avg_stability": avg_stability,
        "avg_sync": avg_sync,
        "coupled_summary": {},
    }


def _make_run(strategies):
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


# ---------------------------------------------------------------------------
# Tests — Policy class
# ---------------------------------------------------------------------------


class TestPolicy:
    """Tests for the Policy class."""

    def test_construction(self):
        p = Policy("test", "hybrid", "balanced", {"instability": 0.5, "sync": 0.5})
        assert p.name == "test"
        assert p.mode == "hybrid"
        assert p.priority == "balanced"
        assert p.thresholds == {"instability": 0.5, "sync": 0.5}

    def test_to_dict(self):
        p = Policy("test", "hybrid", "stability", {"instability": 0.5, "sync": 0.5})
        d = p.to_dict()
        assert d == {
            "mode": "hybrid",
            "priority": "stability",
            "thresholds": {"instability": 0.5, "sync": 0.5},
        }

    def test_from_dict(self):
        d = {"mode": "hybrid", "priority": "balanced", "thresholds": {"instability": 0.5}}
        p = Policy.from_dict("test", d)
        assert p.name == "test"
        assert p.mode == "hybrid"
        assert p.priority == "balanced"

    def test_equality(self):
        p1 = Policy("a", "hybrid", "balanced", {"instability": 0.5})
        p2 = Policy("a", "hybrid", "balanced", {"instability": 0.5})
        assert p1 == p2

    def test_inequality(self):
        p1 = Policy("a", "hybrid", "balanced", {"instability": 0.5})
        p2 = Policy("b", "hybrid", "balanced", {"instability": 0.5})
        assert p1 != p2

    def test_repr(self):
        p = Policy("test", "hybrid", "balanced", {})
        r = repr(p)
        assert "Policy(" in r
        assert "test" in r


# ---------------------------------------------------------------------------
# Tests — Policy.decide matches evaluate_control_policy
# ---------------------------------------------------------------------------


class TestPolicyDecideMatchesLegacy:
    """Policy.decide must produce the same results as evaluate_control_policy."""

    def test_stability_first_stable(self):
        policy_dict = get_builtin_policy(POLICY_STABILITY_FIRST)
        policy_obj = Policy.from_dict("stability_first", policy_dict)
        gs = _make_global_state(avg_stability=0.8, avg_sync=0.3)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_stability_first_unstable(self):
        policy_dict = get_builtin_policy(POLICY_STABILITY_FIRST)
        policy_obj = Policy.from_dict("stability_first", policy_dict)
        gs = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_sync_first_high(self):
        policy_dict = get_builtin_policy(POLICY_SYNC_FIRST)
        policy_obj = Policy.from_dict("sync_first", policy_dict)
        gs = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_sync_first_low(self):
        policy_dict = get_builtin_policy(POLICY_SYNC_FIRST)
        policy_obj = Policy.from_dict("sync_first", policy_dict)
        gs = _make_global_state(avg_stability=0.8, avg_sync=0.3)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_balanced_both_high(self):
        policy_dict = get_builtin_policy(POLICY_BALANCED)
        policy_obj = Policy.from_dict("balanced", policy_dict)
        gs = _make_global_state(avg_stability=0.8, avg_sync=0.8)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_balanced_both_low(self):
        policy_dict = get_builtin_policy(POLICY_BALANCED)
        policy_obj = Policy.from_dict("balanced", policy_dict)
        gs = _make_global_state(avg_stability=0.3, avg_sync=0.3)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_balanced_one_low(self):
        policy_dict = get_builtin_policy(POLICY_BALANCED)
        policy_obj = Policy.from_dict("balanced", policy_dict)
        gs = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        legacy = evaluate_control_policy({}, gs, policy_dict)
        obj_result = policy_obj.decide({}, gs)
        assert legacy == obj_result

    def test_fixed_local_mode(self):
        p = Policy("local_only", "local", "balanced", {})
        assert p.decide({}, _make_global_state()) == "local"

    def test_fixed_global_mode(self):
        p = Policy("global_only", "global", "balanced", {})
        assert p.decide({}, _make_global_state()) == "global"

    def test_decide_deterministic(self):
        p = Policy.from_dict("balanced", get_builtin_policy(POLICY_BALANCED))
        gs = _make_global_state(avg_stability=0.45, avg_sync=0.55)
        results = [p.decide({}, gs) for _ in range(50)]
        assert all(r == results[0] for r in results)


# ---------------------------------------------------------------------------
# Tests — Policy composition
# ---------------------------------------------------------------------------


class TestComposePolicy:
    """Tests for compose_policies."""

    def test_single_policy(self):
        p = Policy("a", "hybrid", "stability", {"instability": 0.4, "sync": 0.6})
        composed = compose_policies([p])
        assert composed.name == "a"
        assert composed.priority == "stability"
        assert composed.thresholds == {"instability": 0.4, "sync": 0.6}

    def test_two_policies_priority_resolution(self):
        p1 = Policy("stability_first", "hybrid", "stability", {"instability": 0.5, "sync": 0.5})
        p2 = Policy("sync_first", "hybrid", "synchronization", {"instability": 0.5, "sync": 0.5})
        composed = compose_policies([p1, p2])
        assert composed.priority == "stability"  # stability > synchronization

    def test_three_policies_priority_resolution(self):
        p1 = Policy("balanced", "hybrid", "balanced", {"instability": 0.5, "sync": 0.5})
        p2 = Policy("sync_first", "hybrid", "synchronization", {"instability": 0.5, "sync": 0.5})
        p3 = Policy("stability_first", "hybrid", "stability", {"instability": 0.5, "sync": 0.5})
        composed = compose_policies([p1, p2, p3])
        assert composed.priority == "stability"

    def test_mode_agreement(self):
        p1 = Policy("a", "local", "balanced", {})
        p2 = Policy("b", "local", "balanced", {})
        composed = compose_policies([p1, p2])
        assert composed.mode == "local"

    def test_mode_disagreement_becomes_hybrid(self):
        p1 = Policy("a", "local", "balanced", {})
        p2 = Policy("b", "global", "balanced", {})
        composed = compose_policies([p1, p2])
        assert composed.mode == "hybrid"

    def test_threshold_averaging(self):
        p1 = Policy("a", "hybrid", "balanced", {"instability": 0.4, "sync": 0.6})
        p2 = Policy("b", "hybrid", "balanced", {"instability": 0.6, "sync": 0.4})
        composed = compose_policies([p1, p2])
        assert composed.thresholds["instability"] == 0.5
        assert composed.thresholds["sync"] == 0.5

    def test_name_concatenation(self):
        p1 = Policy("a", "hybrid", "balanced", {})
        p2 = Policy("b", "hybrid", "balanced", {})
        composed = compose_policies([p1, p2])
        assert composed.name == "a + b"

    def test_empty_raises(self):
        try:
            compose_policies([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_deterministic(self):
        p1 = Policy("a", "hybrid", "stability", {"instability": 0.4, "sync": 0.6})
        p2 = Policy("b", "hybrid", "synchronization", {"instability": 0.6, "sync": 0.4})
        results = [compose_policies([p1, p2]) for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_no_mutation(self):
        p1 = Policy("a", "hybrid", "stability", {"instability": 0.4, "sync": 0.6})
        p2 = Policy("b", "hybrid", "synchronization", {"instability": 0.6, "sync": 0.4})
        p1_thresholds = dict(p1.thresholds)
        p2_thresholds = dict(p2.thresholds)
        compose_policies([p1, p2])
        assert p1.thresholds == p1_thresholds
        assert p2.thresholds == p2_thresholds

    def test_composed_policy_decide_works(self):
        p1 = get_policy("stability_first")
        p2 = get_policy("sync_first")
        composed = compose_policies([p1, p2])
        # stability wins priority, so composed behaves like stability_first
        gs_stable = _make_global_state(avg_stability=0.8, avg_sync=0.3)
        assert composed.decide({}, gs_stable) == "local"
        gs_unstable = _make_global_state(avg_stability=0.3, avg_sync=0.8)
        assert composed.decide({}, gs_unstable) == "global"


# ---------------------------------------------------------------------------
# Tests — Policy registry
# ---------------------------------------------------------------------------


class TestPolicyRegistry:
    """Tests for the policy registry."""

    def test_builtin_policy_thresholds_match_constants(self):
        for name in ["stability_first", "sync_first", "balanced"]:
            p = get_policy(name)
            assert p.thresholds["instability"] == DEFAULT_INSTABILITY_THRESHOLD
            assert p.thresholds["sync"] == DEFAULT_SYNC_THRESHOLD

    def test_policy_decide_accepts_state_arg(self):
        policy = get_policy("balanced")
        result = policy.decide({}, {})
        assert result in ("local", "global", "hybrid")

    def test_builtin_policies_registered(self):
        names = list_policies()
        assert "stability_first" in names
        assert "sync_first" in names
        assert "balanced" in names

    def test_get_builtin_policy(self):
        p = get_policy("balanced")
        assert p.name == "balanced"
        assert p.mode == "hybrid"
        assert p.priority == "balanced"

    def test_get_unknown_raises(self):
        try:
            get_policy("nonexistent_policy_xyz")
            assert False, "Should have raised KeyError"
        except KeyError:
            pass

    def test_register_custom_policy(self):
        custom = Policy("test_custom_reg", "local", "stability", {"instability": 0.3})
        register_policy(custom)
        retrieved = get_policy("test_custom_reg")
        assert retrieved == custom

    def test_registry_policy_matches_legacy_dict(self):
        """Registry policy must produce the same dict as get_builtin_policy."""
        for name in ["stability_first", "sync_first", "balanced"]:
            legacy = get_builtin_policy(name)
            registry_policy = get_policy(name)
            assert registry_policy.to_dict() == legacy


# ---------------------------------------------------------------------------
# Tests — Ranking
# ---------------------------------------------------------------------------


class TestRankPolicies:
    """Tests for rank_policies."""

    def test_score_descending(self):
        results = {
            "a": {"score": 0.5, "steps": 3, "convergence": "stable"},
            "b": {"score": 0.8, "steps": 3, "convergence": "stable"},
            "c": {"score": 0.3, "steps": 3, "convergence": "stable"},
        }
        ranked = rank_policies(results)
        assert [name for name, _ in ranked] == ["b", "a", "c"]

    def test_steps_ascending_tiebreak(self):
        results = {
            "a": {"score": 0.8, "steps": 5, "convergence": "stable"},
            "b": {"score": 0.8, "steps": 2, "convergence": "stable"},
        }
        ranked = rank_policies(results)
        assert [name for name, _ in ranked] == ["b", "a"]

    def test_empty(self):
        assert rank_policies({}) == []

    def test_deterministic(self):
        results = {
            "a": {"score": 0.5, "steps": 3, "convergence": "stable"},
            "b": {"score": 0.8, "steps": 2, "convergence": "stable"},
        }
        rankings = [rank_policies(results) for _ in range(50)]
        assert all(r == rankings[0] for r in rankings)


# ---------------------------------------------------------------------------
# Tests — Formatting
# ---------------------------------------------------------------------------


class TestFormatPolicyExperiment:
    """Tests for format_policy_experiment_summary."""

    def test_basic_format(self):
        results = {
            "stability_first": {"score": 0.82, "steps": 3, "convergence": "stable"},
            "sync_first": {"score": 0.88, "steps": 4, "convergence": "stable"},
        }
        summary = format_policy_experiment_summary(results)
        assert "=== Policy Experiment ===" in summary
        assert "Policy: sync_first" in summary
        assert "Policy: stability_first" in summary
        assert "Best Policy: sync_first" in summary

    def test_empty_results(self):
        summary = format_policy_experiment_summary({})
        assert "=== Policy Experiment ===" in summary
        assert "No policies evaluated." in summary

    def test_deterministic(self):
        results = {
            "a": {"score": 0.5, "steps": 3, "convergence": "stable"},
        }
        summaries = [format_policy_experiment_summary(results) for _ in range(50)]
        assert all(s == summaries[0] for s in summaries)


# ---------------------------------------------------------------------------
# Tests — Integration via strategy_adapter
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for the policy experiment pipeline."""

    def test_adapter_function_exists(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis
        assert callable(run_policy_experiment_analysis)

    def test_full_pipeline(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis

        strategies = [("A", 0.8), ("B", 0.5), ("C", 0.3)]
        runs = [_make_run(strategies) for _ in range(3)]

        result = run_policy_experiment_analysis(runs)
        assert "results" in result
        assert "ranking" in result
        assert "summary" in result
        assert len(result["results"]) == 3  # three built-in policies

    def test_pipeline_deterministic(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [_make_run(strategies) for _ in range(3)]

        r1 = run_policy_experiment_analysis(runs)
        r2 = run_policy_experiment_analysis(runs)
        assert r1["results"] == r2["results"]
        assert r1["ranking"] == r2["ranking"]

    def test_pipeline_no_mutation(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [_make_run(strategies) for _ in range(3)]
        runs_copy = copy.deepcopy(runs)

        run_policy_experiment_analysis(runs)
        assert runs == runs_copy

    def test_pipeline_with_specific_policies(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [_make_run(strategies) for _ in range(3)]

        policies = [get_policy("stability_first"), get_policy("balanced")]
        result = run_policy_experiment_analysis(runs, policies=policies)
        assert len(result["results"]) == 2
        assert "stability_first" in result["results"]
        assert "balanced" in result["results"]

    def test_summary_contains_best_policy(self):
        from qec.analysis.strategy_adapter import run_policy_experiment_analysis

        strategies = [("A", 0.8), ("B", 0.5)]
        runs = [_make_run(strategies) for _ in range(3)]

        result = run_policy_experiment_analysis(runs)
        assert "Best Policy:" in result["summary"]
