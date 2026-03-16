"""
Tests for deterministic ternary decoder rule discovery sandbox.

Covers rule outputs, registry ordering, decoder evaluation reproducibility,
rule comparison stability, and integration with experiment runner.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qec.decoder.ternary.ternary_rule_variants import (
    majority_rule,
    damped_majority_rule,
    conflict_averse_rule,
    parity_pressure_rule,
    RULE_REGISTRY,
)
from src.qec.decoder.ternary.ternary_rule_evaluator import (
    run_decoder_with_rule,
    evaluate_decoder_rule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_parity_matrix() -> np.ndarray:
    """Return a small parity check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 1],
    ], dtype=np.float64)


def _received_vector(n: int = 5) -> np.ndarray:
    """Return a simple deterministic received vector."""
    return np.array([1.0, -1.0, 1.0, 0.0, -1.0], dtype=np.float64)[:n]


# ===========================================================================
# Test rule outputs are deterministic
# ===========================================================================

class TestMajorityRule:
    """Tests for majority_rule."""

    def test_positive_majority(self) -> None:
        msgs = np.array([1, 1, -1], dtype=np.int8)
        assert majority_rule(msgs) == np.int8(1)

    def test_negative_majority(self) -> None:
        msgs = np.array([-1, -1, 1], dtype=np.int8)
        assert majority_rule(msgs) == np.int8(-1)

    def test_tie_resolves_to_zero(self) -> None:
        msgs = np.array([1, -1], dtype=np.int8)
        assert majority_rule(msgs) == np.int8(0)

    def test_all_zeros(self) -> None:
        msgs = np.array([0, 0, 0], dtype=np.int8)
        assert majority_rule(msgs) == np.int8(0)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        result = majority_rule(msgs)
        assert isinstance(result, np.int8)

    def test_deterministic(self) -> None:
        msgs = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        r1 = majority_rule(msgs)
        r2 = majority_rule(msgs)
        assert r1 == r2


class TestDampedMajorityRule:
    """Tests for damped_majority_rule."""

    def test_strong_positive(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        assert damped_majority_rule(msgs) == np.int8(1)

    def test_strong_negative(self) -> None:
        msgs = np.array([-1, -1, -1], dtype=np.int8)
        assert damped_majority_rule(msgs) == np.int8(-1)

    def test_damped_to_zero(self) -> None:
        # sum = 1, abs(1) < 2 → damped to 0
        msgs = np.array([1, 0, 0], dtype=np.int8)
        assert damped_majority_rule(msgs) == np.int8(0)

    def test_threshold_boundary(self) -> None:
        # sum = 2, abs(2) >= 2 → not damped
        msgs = np.array([1, 1, 0], dtype=np.int8)
        assert damped_majority_rule(msgs) == np.int8(1)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1, 1], dtype=np.int8)
        result = damped_majority_rule(msgs)
        assert isinstance(result, np.int8)


class TestConflictAverseRule:
    """Tests for conflict_averse_rule."""

    def test_conflict_returns_zero(self) -> None:
        msgs = np.array([1, -1, 1], dtype=np.int8)
        assert conflict_averse_rule(msgs) == np.int8(0)

    def test_no_conflict_positive(self) -> None:
        msgs = np.array([1, 1, 0], dtype=np.int8)
        assert conflict_averse_rule(msgs) == np.int8(1)

    def test_no_conflict_negative(self) -> None:
        msgs = np.array([-1, -1, 0], dtype=np.int8)
        assert conflict_averse_rule(msgs) == np.int8(-1)

    def test_all_zeros(self) -> None:
        msgs = np.array([0, 0, 0], dtype=np.int8)
        assert conflict_averse_rule(msgs) == np.int8(0)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 0, 0], dtype=np.int8)
        result = conflict_averse_rule(msgs)
        assert isinstance(result, np.int8)


class TestParityPressureRule:
    """Tests for parity_pressure_rule."""

    def test_positive_sum(self) -> None:
        msgs = np.array([1, 1, -1], dtype=np.int8)
        assert parity_pressure_rule(msgs) == np.int8(1)

    def test_negative_sum(self) -> None:
        msgs = np.array([-1, -1, 1], dtype=np.int8)
        assert parity_pressure_rule(msgs) == np.int8(-1)

    def test_zero_sum(self) -> None:
        msgs = np.array([1, -1], dtype=np.int8)
        assert parity_pressure_rule(msgs) == np.int8(0)

    def test_output_dtype(self) -> None:
        msgs = np.array([1, 1], dtype=np.int8)
        result = parity_pressure_rule(msgs)
        assert isinstance(result, np.int8)


# ===========================================================================
# Test registry ordering is deterministic
# ===========================================================================

class TestRuleRegistry:
    """Tests for RULE_REGISTRY determinism and ordering."""

    def test_registry_keys_sorted(self) -> None:
        keys = list(RULE_REGISTRY.keys())
        assert keys == sorted(keys)

    def test_registry_size(self) -> None:
        assert len(RULE_REGISTRY) == 4

    def test_registry_contains_all_rules(self) -> None:
        expected = {"majority", "damped_majority", "conflict_averse", "parity_pressure"}
        assert set(RULE_REGISTRY.keys()) == expected

    def test_registry_ordering_deterministic(self) -> None:
        keys1 = list(RULE_REGISTRY.keys())
        keys2 = list(RULE_REGISTRY.keys())
        assert keys1 == keys2

    def test_registry_functions_callable(self) -> None:
        msgs = np.array([1, -1, 1], dtype=np.int8)
        for name, fn in RULE_REGISTRY.items():
            result = fn(msgs)
            assert isinstance(result, np.int8), f"Rule '{name}' did not return np.int8"


# ===========================================================================
# Test decoder evaluation is reproducible
# ===========================================================================

class TestRunDecoderWithRule:
    """Tests for run_decoder_with_rule."""

    def test_returns_required_keys(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        result = run_decoder_with_rule(H, r, "majority")
        assert "final_messages" in result
        assert "iterations" in result
        assert "converged" in result

    def test_final_messages_dtype(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        result = run_decoder_with_rule(H, r, "majority")
        assert result["final_messages"].dtype == np.int8

    def test_final_messages_shape(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        result = run_decoder_with_rule(H, r, "majority")
        assert result["final_messages"].shape == (H.shape[1],)

    def test_final_messages_ternary_values(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        result = run_decoder_with_rule(H, r, "majority")
        unique = set(result["final_messages"].tolist())
        assert unique.issubset({-1, 0, 1})

    def test_deterministic_across_calls(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        r1 = run_decoder_with_rule(H, r, "majority")
        r2 = run_decoder_with_rule(H, r, "majority")
        assert np.array_equal(r1["final_messages"], r2["final_messages"])
        assert r1["iterations"] == r2["iterations"]
        assert r1["converged"] == r2["converged"]

    def test_invalid_rule_raises(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        with pytest.raises(ValueError, match="Unknown rule"):
            run_decoder_with_rule(H, r, "nonexistent_rule")

    def test_all_rules_run(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        for name in RULE_REGISTRY:
            result = run_decoder_with_rule(H, r, name)
            assert result["final_messages"].dtype == np.int8
            assert isinstance(result["iterations"], (int, np.integer))
            assert isinstance(result["converged"], (bool, np.bool_))

    def test_convergence_within_max_iterations(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        result = run_decoder_with_rule(H, r, "majority", max_iterations=50)
        assert result["iterations"] <= 50


class TestEvaluateDecoderRule:
    """Tests for evaluate_decoder_rule."""

    def test_returns_required_metrics(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        metrics = evaluate_decoder_rule(H, r, "majority")
        assert "stability" in metrics
        assert "entropy" in metrics
        assert "conflict_density" in metrics
        assert "trapping_indicator" in metrics

    def test_metrics_dtype_float64(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        metrics = evaluate_decoder_rule(H, r, "majority")
        for key, val in metrics.items():
            assert isinstance(val, np.float64), f"Metric '{key}' is not np.float64"

    def test_metrics_bounded(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        metrics = evaluate_decoder_rule(H, r, "majority")
        for key, val in metrics.items():
            assert 0.0 <= float(val) <= 1.0, f"Metric '{key}' out of bounds: {val}"

    def test_deterministic_across_calls(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        m1 = evaluate_decoder_rule(H, r, "conflict_averse")
        m2 = evaluate_decoder_rule(H, r, "conflict_averse")
        for key in m1:
            assert m1[key] == m2[key], f"Metric '{key}' not deterministic"

    def test_all_rules_evaluable(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        for name in RULE_REGISTRY:
            metrics = evaluate_decoder_rule(H, r, name)
            assert len(metrics) == 4


# ===========================================================================
# Test rule comparison is stable
# ===========================================================================

class TestRuleComparison:
    """Tests for rule comparison stability."""

    def test_different_rules_may_differ(self) -> None:
        """Rules may produce different metrics — verify structure is consistent."""
        H = _simple_parity_matrix()
        r = _received_vector()
        all_metrics = {}
        for name in sorted(RULE_REGISTRY.keys()):
            all_metrics[name] = evaluate_decoder_rule(H, r, name)
        # All metrics dicts have same keys
        keys_set = {frozenset(m.keys()) for m in all_metrics.values()}
        assert len(keys_set) == 1

    def test_comparison_ordering_deterministic(self) -> None:
        H = _simple_parity_matrix()
        r = _received_vector()
        results1 = []
        results2 = []
        for name in sorted(RULE_REGISTRY.keys()):
            results1.append((name, evaluate_decoder_rule(H, r, name)))
            results2.append((name, evaluate_decoder_rule(H, r, name)))
        for (n1, m1), (n2, m2) in zip(results1, results2):
            assert n1 == n2
            for key in m1:
                assert m1[key] == m2[key]


# ===========================================================================
# Test integration with experiment runner
# ===========================================================================

class TestExperimentIntegration:
    """Tests for integration with discovery_run experiment."""

    def test_discovery_run_with_decoder_rules(self) -> None:
        """Verify that enable_decoder_rule_experiments adds expected keys."""
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_artifact.json",
            enable_decoder_rule_experiments=True,
        )
        results = artifact["results"]
        assert "decoder_rule_metrics" in results
        assert "best_decoder_rule" in results
        assert "rule_stability_scores" in results

    def test_decoder_rule_metrics_structure(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_metrics.json",
            enable_decoder_rule_experiments=True,
        )
        metrics_list = artifact["results"]["decoder_rule_metrics"]
        assert len(metrics_list) == len(RULE_REGISTRY)
        for entry in metrics_list:
            assert "rule_name" in entry
            assert "stability" in entry
            assert "entropy" in entry
            assert "conflict_density" in entry
            assert "trapping_indicator" in entry

    def test_best_decoder_rule_is_valid(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_best.json",
            enable_decoder_rule_experiments=True,
        )
        best = artifact["results"]["best_decoder_rule"]
        assert best in RULE_REGISTRY

    def test_rule_stability_scores_sorted(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_scores.json",
            enable_decoder_rule_experiments=True,
        )
        scores = artifact["results"]["rule_stability_scores"]
        keys = list(scores.keys())
        assert keys == sorted(keys)

    def test_default_disabled_no_keys(self) -> None:
        """When flag is off, no decoder rule keys should appear."""
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        artifact = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_disabled.json",
            enable_decoder_rule_experiments=False,
        )
        results = artifact["results"]
        assert "decoder_rule_metrics" not in results
        assert "best_decoder_rule" not in results
        assert "rule_stability_scores" not in results

    def test_experiment_deterministic(self) -> None:
        from src.qec.experiments.discovery_run import run_discovery_experiment
        spec = {
            "num_variables": 10,
            "num_checks": 5,
            "variable_degree": 3,
            "check_degree": 6,
        }
        a1 = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_det1.json",
            enable_decoder_rule_experiments=True,
        )
        a2 = run_discovery_experiment(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            output_path="/tmp/test_decoder_rules_det2.json",
            enable_decoder_rule_experiments=True,
        )
        assert a1["results"]["decoder_rule_metrics"] == a2["results"]["decoder_rule_metrics"]
        assert a1["results"]["best_decoder_rule"] == a2["results"]["best_decoder_rule"]
        assert a1["results"]["rule_stability_scores"] == a2["results"]["rule_stability_scores"]


# ===========================================================================
# Test API wrappers
# ===========================================================================

class TestAPIWrappers:
    """Tests for analysis API wrappers."""

    def test_run_decoder_with_rule_wrapper(self) -> None:
        from src.qec.analysis.api import run_decoder_with_rule as api_run
        H = _simple_parity_matrix()
        r = _received_vector()
        result = api_run(H, r, "majority")
        assert "final_messages" in result
        assert result["final_messages"].dtype == np.int8

    def test_evaluate_decoder_rule_wrapper(self) -> None:
        from src.qec.analysis.api import evaluate_decoder_rule as api_eval
        H = _simple_parity_matrix()
        r = _received_vector()
        metrics = api_eval(H, r, "majority")
        assert "stability" in metrics
        assert isinstance(metrics["stability"], np.float64)

    def test_list_decoder_rules_wrapper(self) -> None:
        from src.qec.analysis.api import list_decoder_rules as api_list
        rules = api_list()
        assert isinstance(rules, list)
        assert rules == sorted(rules)
        assert len(rules) == 4
