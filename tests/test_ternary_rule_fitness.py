"""Tests for deterministic rule fitness surfaces and ranking.

Verifies:
- Deterministic outputs
- dtype = np.float64
- Ranking stability under ties
- Reproducibility across runs
- No mutation of input data
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.qec.decoder.ternary.ternary_coevolution import evaluate_rule_population
from src.qec.decoder.ternary.ternary_rule_fitness import (
    compute_rule_fitness_metrics,
    rank_rules_by_fitness,
)
from tests.utils import simple_parity_matrix, received_vector


def _make_population_result() -> dict:
    """Run evaluate_rule_population and return results."""
    H = simple_parity_matrix()
    r = received_vector()
    return evaluate_rule_population(H, r, use_extended_rules=True)


# ===========================================================================
# compute_rule_fitness_metrics
# ===========================================================================


class TestComputeRuleFitnessMetrics:
    """Tests for compute_rule_fitness_metrics."""

    def test_returns_dict_with_all_rules(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        rule_names = {e["rule_name"] for e in results["decoder_rule_population"]}
        assert set(metrics.keys()) == rule_names

    def test_keys_sorted_lexicographically(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        keys = list(metrics.keys())
        assert keys == sorted(keys)

    def test_core_metrics_present(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        core_keys = {
            "convergence_rate", "failure_rate",
            "mean_iterations",
        }
        for rule_metrics in metrics.values():
            assert core_keys.issubset(rule_metrics.keys())

    def test_passthrough_metrics_present(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        passthrough_keys = {
            "stability", "entropy",
            "conflict_density", "trapping_indicator",
        }
        for rule_metrics in metrics.values():
            assert passthrough_keys.issubset(rule_metrics.keys())

    def test_all_values_are_float64(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        for rule_metrics in metrics.values():
            for key, val in rule_metrics.items():
                assert isinstance(val, np.float64), (
                    f"Expected np.float64 for '{key}', got {type(val)}"
                )

    def test_convergence_rate_binary(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        for rule_metrics in metrics.values():
            assert rule_metrics["convergence_rate"] in (
                np.float64(0.0), np.float64(1.0),
            )

    def test_failure_rate_complement(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        for rule_metrics in metrics.values():
            assert rule_metrics["failure_rate"] == np.float64(1.0) - rule_metrics["convergence_rate"]

    def test_stability_score_not_present(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        for rule_metrics in metrics.values():
            assert "stability_score" not in rule_metrics

    def test_deterministic(self) -> None:
        results = _make_population_result()
        m1 = compute_rule_fitness_metrics(results)
        m2 = compute_rule_fitness_metrics(results)
        assert list(m1.keys()) == list(m2.keys())
        for rule in m1:
            for key in m1[rule]:
                assert m1[rule][key] == m2[rule][key]

    def test_no_mutation_of_input(self) -> None:
        results = _make_population_result()
        original = copy.deepcopy(results)
        compute_rule_fitness_metrics(results)
        assert results == original


# ===========================================================================
# rank_rules_by_fitness
# ===========================================================================


class TestRankRulesByFitness:
    """Tests for rank_rules_by_fitness."""

    def test_returns_list_of_tuples(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        ranked = rank_rules_by_fitness(metrics)
        assert isinstance(ranked, list)
        for item in ranked:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], dict)

    def test_all_rules_present(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        ranked = rank_rules_by_fitness(metrics)
        ranked_names = {name for name, _ in ranked}
        assert ranked_names == set(metrics.keys())

    def test_ranking_deterministic(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        r1 = rank_rules_by_fitness(metrics)
        r2 = rank_rules_by_fitness(metrics)
        assert [n for n, _ in r1] == [n for n, _ in r2]

    def test_ranking_reproducible_across_runs(self) -> None:
        r1 = rank_rules_by_fitness(compute_rule_fitness_metrics(_make_population_result()))
        r2 = rank_rules_by_fitness(compute_rule_fitness_metrics(_make_population_result()))
        assert [n for n, _ in r1] == [n for n, _ in r2]

    def test_best_rule_has_highest_convergence(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        ranked = rank_rules_by_fitness(metrics)
        best_conv = ranked[0][1]["convergence_rate"]
        max_conv = max(m["convergence_rate"] for m in metrics.values())
        assert best_conv == max_conv

    def test_empty_metrics(self) -> None:
        assert rank_rules_by_fitness({}) == []

    def test_ranking_stability_under_ties(self) -> None:
        """When all metrics are identical, ranking should be lexicographic by name."""
        tied_metrics: dict[str, dict[str, np.float64]] = {
            "rule_b": {
                "convergence_rate": np.float64(1.0),
                "failure_rate": np.float64(0.0),
                "mean_iterations": np.float64(5.0),
                "conflict_density": np.float64(0.1),
            },
            "rule_a": {
                "convergence_rate": np.float64(1.0),
                "failure_rate": np.float64(0.0),
                "mean_iterations": np.float64(5.0),
                "conflict_density": np.float64(0.1),
            },
            "rule_c": {
                "convergence_rate": np.float64(1.0),
                "failure_rate": np.float64(0.0),
                "mean_iterations": np.float64(5.0),
                "conflict_density": np.float64(0.1),
            },
        }
        ranked = rank_rules_by_fitness(tied_metrics)
        names = [n for n, _ in ranked]
        assert names == ["rule_a", "rule_b", "rule_c"]

    def test_no_mutation_of_input(self) -> None:
        results = _make_population_result()
        metrics = compute_rule_fitness_metrics(results)
        original = copy.deepcopy(metrics)
        rank_rules_by_fitness(metrics)
        assert metrics == original
