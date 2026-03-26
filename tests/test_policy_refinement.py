"""Tests for policy refinement and threshold optimization (v103.6.0).

Verifies:
- deterministic variant generation
- threshold clamping to [0.0, 1.0]
- improvement monotonicity (or stable selection)
- no mutation of original policy
- integration correctness via strategy_adapter
"""

from __future__ import annotations

import copy

from qec.analysis.policy import Policy, get_policy
from qec.analysis.policy_refinement import (
    REFINEMENT_DELTA,
    REFINEMENT_MAX_ITERS,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    evaluate_policy_variants,
    format_policy_refinement_summary,
    generate_policy_variants,
    refine_policy,
    select_best_variant,
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


def _make_policy(name="test_policy", instability=0.5, sync=0.5):
    """Build a test policy."""
    return Policy(
        name=name,
        mode="hybrid",
        priority="stability",
        thresholds={"instability": instability, "sync": sync},
    )


# ---------------------------------------------------------------------------
# Tests — Variant Generation
# ---------------------------------------------------------------------------


class TestGeneratePolicyVariants:
    """Test deterministic variant generation."""

    def test_generates_nine_variants_for_two_thresholds(self):
        policy = _make_policy()
        variants = generate_policy_variants(policy)
        # 2 thresholds -> 3^2 = 9 variants
        assert len(variants) == 9

    def test_generates_three_variants_for_one_threshold(self):
        policy = Policy(
            name="single",
            mode="hybrid",
            priority="stability",
            thresholds={"instability": 0.5},
        )
        variants = generate_policy_variants(policy)
        assert len(variants) == 3

    def test_generates_one_variant_for_no_thresholds(self):
        policy = Policy(
            name="empty",
            mode="local",
            priority="stability",
            thresholds={},
        )
        variants = generate_policy_variants(policy)
        assert len(variants) == 1
        assert variants[0].name == "empty"

    def test_variants_sorted_by_name(self):
        policy = _make_policy()
        variants = generate_policy_variants(policy)
        names = [v.name for v in variants]
        assert names == sorted(names)

    def test_base_policy_included(self):
        policy = _make_policy(name="base")
        variants = generate_policy_variants(policy)
        # The variant with no offsets should match the base name.
        base_names = [v.name for v in variants if v.name == "base"]
        assert len(base_names) == 1

    def test_preserves_mode_and_priority(self):
        policy = _make_policy()
        variants = generate_policy_variants(policy)
        for v in variants:
            assert v.mode == policy.mode
            assert v.priority == policy.priority

    def test_does_not_mutate_original(self):
        policy = _make_policy()
        original_thresholds = dict(policy.thresholds)
        original_name = policy.name
        generate_policy_variants(policy)
        assert policy.thresholds == original_thresholds
        assert policy.name == original_name

    def test_deterministic_output(self):
        policy = _make_policy()
        v1 = generate_policy_variants(policy)
        v2 = generate_policy_variants(policy)
        assert len(v1) == len(v2)
        for a, b in zip(v1, v2):
            assert a.name == b.name
            assert a.thresholds == b.thresholds

    def test_custom_delta(self):
        policy = _make_policy(instability=0.5, sync=0.5)
        variants = generate_policy_variants(policy, delta=0.2)
        assert len(variants) == 9
        # Check that offsets use custom delta.
        threshold_values = sorted(set(
            v.thresholds["instability"] for v in variants
        ))
        assert threshold_values == [0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# Tests — Threshold Clamping
# ---------------------------------------------------------------------------


class TestThresholdClamping:
    """Test that thresholds are clamped to [0.0, 1.0]."""

    def test_lower_bound_clamping(self):
        policy = _make_policy(instability=0.05, sync=0.05)
        variants = generate_policy_variants(policy, delta=0.1)
        for v in variants:
            for key, val in v.thresholds.items():
                assert val >= THRESHOLD_MIN, (
                    f"{v.name}: {key}={val} below {THRESHOLD_MIN}"
                )

    def test_upper_bound_clamping(self):
        policy = _make_policy(instability=0.95, sync=0.95)
        variants = generate_policy_variants(policy, delta=0.1)
        for v in variants:
            for key, val in v.thresholds.items():
                assert val <= THRESHOLD_MAX, (
                    f"{v.name}: {key}={val} above {THRESHOLD_MAX}"
                )

    def test_exact_boundary_values(self):
        policy = _make_policy(instability=0.0, sync=1.0)
        variants = generate_policy_variants(policy, delta=0.1)
        for v in variants:
            assert v.thresholds["instability"] >= 0.0
            assert v.thresholds["sync"] <= 1.0


# ---------------------------------------------------------------------------
# Tests — Best Variant Selection
# ---------------------------------------------------------------------------


class TestSelectBestVariant:
    """Test deterministic best variant selection."""

    def test_selects_highest_score(self):
        variants = [_make_policy(f"v{i}") for i in range(3)]
        results = {
            "v0": {"score": 0.5, "steps": 3},
            "v1": {"score": 0.8, "steps": 3},
            "v2": {"score": 0.6, "steps": 3},
        }
        best = select_best_variant(results, variants)
        assert best.name == "v1"

    def test_tiebreak_by_steps(self):
        variants = [_make_policy(f"v{i}") for i in range(3)]
        results = {
            "v0": {"score": 0.8, "steps": 5},
            "v1": {"score": 0.8, "steps": 2},
            "v2": {"score": 0.8, "steps": 3},
        }
        best = select_best_variant(results, variants)
        assert best.name == "v1"

    def test_tiebreak_by_name(self):
        variants = [_make_policy(f"v{i}") for i in range(3)]
        results = {
            "v0": {"score": 0.8, "steps": 3},
            "v1": {"score": 0.8, "steps": 3},
            "v2": {"score": 0.8, "steps": 3},
        }
        best = select_best_variant(results, variants)
        assert best.name == "v0"

    def test_empty_results_raises(self):
        try:
            select_best_variant({}, [])
            assert False, "Expected ValueError"
        except ValueError:
            pass

    def test_single_variant(self):
        variants = [_make_policy("only")]
        results = {"only": {"score": 0.5, "steps": 1}}
        best = select_best_variant(results, variants)
        assert best.name == "only"


# ---------------------------------------------------------------------------
# Tests — Evaluation
# ---------------------------------------------------------------------------


class TestEvaluatePolicyVariants:
    """Test variant evaluation via hierarchical control."""

    def test_returns_results_for_all_variants(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        policy = _make_policy()
        variants = generate_policy_variants(policy)

        results = evaluate_policy_variants(runs, variants, {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        })

        assert len(results) == len(variants)
        for name, r in results.items():
            assert "score" in r
            assert "steps" in r

    def test_deterministic_evaluation(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        policy = _make_policy()
        variants = generate_policy_variants(policy)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        r1 = evaluate_policy_variants(runs, variants, objective)
        r2 = evaluate_policy_variants(runs, variants, objective)
        assert r1 == r2

    def test_does_not_mutate_inputs(self):
        runs = [_make_run([("s1", 0.7)])]
        policy = _make_policy()
        variants = generate_policy_variants(policy)
        runs_copy = copy.deepcopy(runs)
        variants_copy = copy.deepcopy(variants)

        evaluate_policy_variants(runs, variants, {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        })

        # Verify runs not mutated.
        assert runs == runs_copy
        # Verify variant names preserved.
        for v, vc in zip(variants, variants_copy):
            assert v.name == vc.name
            assert v.thresholds == vc.thresholds


# ---------------------------------------------------------------------------
# Tests — Refinement Loop
# ---------------------------------------------------------------------------


class TestRefinePolicy:
    """Test the iterative refinement loop."""

    def test_returns_expected_keys(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        policy = _make_policy()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = refine_policy(runs, policy, objective, max_iters=1)

        assert "policies" in result
        assert "scores" in result
        assert "best_policy" in result
        assert "iterations" in result
        assert "improved" in result

    def test_iterations_match_max_iters(self):
        runs = [_make_run([("s1", 0.7)])]
        policy = _make_policy()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        for n in [1, 2, 3]:
            result = refine_policy(runs, policy, objective, max_iters=n)
            assert result["iterations"] == n
            assert len(result["policies"]) == n
            assert len(result["scores"]) == n

    def test_does_not_mutate_base_policy(self):
        runs = [_make_run([("s1", 0.7)])]
        policy = _make_policy()
        original = copy.deepcopy(policy)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        refine_policy(runs, policy, objective, max_iters=2)

        assert policy.name == original.name
        assert policy.mode == original.mode
        assert policy.priority == original.priority
        assert policy.thresholds == original.thresholds

    def test_deterministic_refinement(self):
        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        policy = _make_policy()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        r1 = refine_policy(runs, policy, objective, max_iters=2)
        r2 = refine_policy(runs, policy, objective, max_iters=2)

        assert r1["scores"] == r2["scores"]
        assert r1["best_policy"].thresholds == r2["best_policy"].thresholds

    def test_thresholds_remain_bounded(self):
        runs = [_make_run([("s1", 0.7)])]
        policy = _make_policy(instability=0.95, sync=0.05)
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = refine_policy(runs, policy, objective, max_iters=3)

        best = result["best_policy"]
        for key, val in best.thresholds.items():
            assert THRESHOLD_MIN <= val <= THRESHOLD_MAX, (
                f"Threshold {key}={val} out of bounds"
            )

    def test_best_policy_is_policy_instance(self):
        runs = [_make_run([("s1", 0.7)])]
        policy = _make_policy()
        objective = {
            "w_stability": 0.3,
            "w_attractor": 0.3,
            "w_transient": 0.2,
            "w_sync": 0.2,
        }

        result = refine_policy(runs, policy, objective, max_iters=1)
        assert isinstance(result["best_policy"], Policy)


# ---------------------------------------------------------------------------
# Tests — Formatting
# ---------------------------------------------------------------------------


class TestFormatPolicyRefinementSummary:
    """Test formatting output."""

    def test_contains_header(self):
        result = {
            "policies": [],
            "scores": [],
            "best_policy": _make_policy(),
            "iterations": 0,
            "improved": False,
        }
        summary = format_policy_refinement_summary(result)
        assert "=== Policy Refinement ===" in summary

    def test_shows_iteration_scores(self):
        result = {
            "policies": [_make_policy()],
            "scores": [0.82],
            "best_policy": _make_policy(),
            "iterations": 1,
            "improved": False,
        }
        summary = format_policy_refinement_summary(result)
        assert "Iteration 1" in summary
        assert "0.82" in summary

    def test_shows_improvement_message(self):
        result = {
            "policies": [_make_policy(), _make_policy()],
            "scores": [0.82, 0.86],
            "best_policy": _make_policy(),
            "iterations": 2,
            "improved": True,
        }
        summary = format_policy_refinement_summary(result)
        assert "improved" in summary.lower()

    def test_shows_no_improvement_message(self):
        result = {
            "policies": [_make_policy()],
            "scores": [0.82],
            "best_policy": _make_policy(),
            "iterations": 1,
            "improved": False,
        }
        summary = format_policy_refinement_summary(result)
        assert "did not improve" in summary.lower()

    def test_shows_best_policy_thresholds(self):
        result = {
            "policies": [_make_policy(instability=0.6, sync=0.4)],
            "scores": [0.89],
            "best_policy": _make_policy(instability=0.6, sync=0.4),
            "iterations": 1,
            "improved": True,
        }
        summary = format_policy_refinement_summary(result)
        assert "Best Policy" in summary
        assert "instability_threshold=0.6" in summary
        assert "sync_threshold=0.4" in summary


# ---------------------------------------------------------------------------
# Tests — Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test integration via strategy_adapter."""

    def test_adapter_runs_without_error(self):
        from qec.analysis.strategy_adapter import (
            run_policy_refinement_analysis,
        )

        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        result = run_policy_refinement_analysis(runs, max_iters=1)

        assert "refinements" in result
        assert "best" in result
        assert "summary" in result
        assert isinstance(result["best"], Policy)

    def test_adapter_with_specific_policy(self):
        from qec.analysis.strategy_adapter import (
            run_policy_refinement_analysis,
        )

        runs = [_make_run([("s1", 0.7), ("s2", 0.3)])]
        policy = get_policy("stability_first")
        result = run_policy_refinement_analysis(
            runs, policies=[policy], max_iters=1,
        )

        assert "stability_first" in result["refinements"]

    def test_adapter_format_summary(self):
        from qec.analysis.strategy_adapter import (
            format_policy_refinement_adapter_summary,
            run_policy_refinement_analysis,
        )

        runs = [_make_run([("s1", 0.7)])]
        result = run_policy_refinement_analysis(runs, max_iters=1)
        summary = format_policy_refinement_adapter_summary(result)
        assert "=== Policy Refinement ===" in summary
