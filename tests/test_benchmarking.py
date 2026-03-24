"""Tests for v101 benchmarking and convergence validation.

Covers:
- deterministic outputs (identical runs match)
- baseline strategy reproducibility
- convergence detection correctness
- performance metric bounds
- benchmark comparison structure
- no mutation of inputs
"""

from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, List

from qec.analysis.baseline_strategies import (
    fixed_strategy,
    random_strategy_deterministic,
    round_robin_strategy,
)
from qec.analysis.benchmark_comparison import compare_strategies
from qec.analysis.convergence_analysis import (
    compute_convergence_signal,
    detect_convergence,
)
from qec.analysis.performance_metrics import (
    compute_convergence_rate,
    compute_cumulative_score,
    compute_final_performance,
    compute_stability_variance,
)
from qec.experiments.benchmark_runner import run_benchmark


class TestBaselineStrategies(unittest.TestCase):
    """Tests for deterministic baseline strategy functions."""

    def setUp(self) -> None:
        self.strategies = ["s1", "s2", "s3", "s4", "s5", "s6"]

    def test_random_deterministic_reproducibility(self) -> None:
        """Identical seed+step → identical selection."""
        for step in range(20):
            r1 = random_strategy_deterministic(42, self.strategies, step)
            r2 = random_strategy_deterministic(42, self.strategies, step)
            self.assertEqual(r1, r2)

    def test_random_deterministic_different_seeds(self) -> None:
        """Different seeds should (likely) produce different sequences."""
        seq1 = [random_strategy_deterministic(1, self.strategies, i) for i in range(10)]
        seq2 = [random_strategy_deterministic(2, self.strategies, i) for i in range(10)]
        # Not guaranteed to be all different, but sequences should differ
        self.assertNotEqual(seq1, seq2)

    def test_random_deterministic_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            random_strategy_deterministic(42, [], 0)

    def test_random_deterministic_returns_valid(self) -> None:
        for step in range(50):
            result = random_strategy_deterministic(42, self.strategies, step)
            self.assertIn(result, self.strategies)

    def test_fixed_strategy(self) -> None:
        self.assertEqual(fixed_strategy("s3"), "s3")
        self.assertEqual(fixed_strategy("s1"), "s1")

    def test_round_robin_deterministic(self) -> None:
        expected = ["s1", "s2", "s3", "s4", "s5", "s6", "s1", "s2"]
        for step, exp in enumerate(expected):
            self.assertEqual(round_robin_strategy(step, self.strategies), exp)

    def test_round_robin_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            round_robin_strategy(0, [])

    def test_round_robin_reproducibility(self) -> None:
        r1 = [round_robin_strategy(i, self.strategies) for i in range(20)]
        r2 = [round_robin_strategy(i, self.strategies) for i in range(20)]
        self.assertEqual(r1, r2)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for performance metric computations."""

    def test_cumulative_score_basic(self) -> None:
        scores = [1.0, 2.0, 3.0]
        result = compute_cumulative_score(scores)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], 1.5)
        self.assertAlmostEqual(result[2], 2.0)

    def test_cumulative_score_empty(self) -> None:
        self.assertEqual(compute_cumulative_score([]), [])

    def test_convergence_rate_constant(self) -> None:
        """Constant sequence has zero convergence rate."""
        self.assertAlmostEqual(compute_convergence_rate([1.0] * 10), 0.0)

    def test_convergence_rate_increasing(self) -> None:
        scores = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(compute_convergence_rate(scores), 1.0)

    def test_convergence_rate_short(self) -> None:
        self.assertAlmostEqual(compute_convergence_rate([1.0]), 0.0)
        self.assertAlmostEqual(compute_convergence_rate([]), 0.0)

    def test_stability_variance_constant(self) -> None:
        self.assertAlmostEqual(compute_stability_variance([5.0] * 10), 0.0)

    def test_stability_variance_positive(self) -> None:
        self.assertGreater(compute_stability_variance([0.0, 1.0, 0.0, 1.0]), 0.0)

    def test_stability_variance_empty(self) -> None:
        self.assertAlmostEqual(compute_stability_variance([]), 0.0)

    def test_final_performance_basic(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        result = compute_final_performance(scores, window=3)
        self.assertAlmostEqual(result, (0.6 + 0.7 + 0.8) / 3)

    def test_final_performance_empty(self) -> None:
        self.assertAlmostEqual(compute_final_performance([]), 0.0)

    def test_final_performance_default_window(self) -> None:
        scores = [0.1, 0.2, 0.3]
        result = compute_final_performance(scores)
        self.assertAlmostEqual(result, 0.2)  # mean of all 3

    def test_no_mutation(self) -> None:
        """Input lists must not be mutated."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        original = scores.copy()
        compute_cumulative_score(scores)
        compute_convergence_rate(scores)
        compute_stability_variance(scores)
        compute_final_performance(scores)
        self.assertEqual(scores, original)


class TestConvergenceAnalysis(unittest.TestCase):
    """Tests for convergence detection."""

    def test_convergence_detected_constant(self) -> None:
        scores = [0.5] * 10
        result = detect_convergence(scores, window=3)
        self.assertIsNotNone(result)
        self.assertLessEqual(result, 4)

    def test_convergence_not_detected(self) -> None:
        scores = [float(i) for i in range(10)]
        result = detect_convergence(scores, window=3, threshold=0.01)
        self.assertIsNone(result)

    def test_convergence_late(self) -> None:
        scores = [float(i) for i in range(5)] + [5.0] * 10
        result = detect_convergence(scores, window=3, threshold=0.01)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 5)

    def test_convergence_short_input(self) -> None:
        self.assertIsNone(detect_convergence([1.0], window=5))
        self.assertIsNone(detect_convergence([], window=5))

    def test_convergence_signal_constant(self) -> None:
        self.assertAlmostEqual(compute_convergence_signal([1.0] * 10), 1.0)

    def test_convergence_signal_bounded(self) -> None:
        for scores in [
            [0.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, 100.0, 0.0],
            [1.0],
            [],
        ]:
            signal = compute_convergence_signal(scores)
            self.assertGreaterEqual(signal, 0.0)
            self.assertLessEqual(signal, 1.0)

    def test_convergence_signal_empty(self) -> None:
        self.assertAlmostEqual(compute_convergence_signal([]), 1.0)

    def test_convergence_determinism(self) -> None:
        scores = [0.1, 0.5, 0.3, 0.4, 0.41, 0.405, 0.41]
        r1 = detect_convergence(scores, window=3)
        r2 = detect_convergence(scores, window=3)
        self.assertEqual(r1, r2)
        s1 = compute_convergence_signal(scores)
        s2 = compute_convergence_signal(scores)
        self.assertEqual(s1, s2)

    def test_no_mutation(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        original = scores.copy()
        detect_convergence(scores)
        compute_convergence_signal(scores)
        self.assertEqual(scores, original)


class TestBenchmarkComparison(unittest.TestCase):
    """Tests for benchmark comparison engine."""

    def test_compare_strategies_basic(self) -> None:
        results = {
            "qec": {"scores": [0.6, 0.7, 0.8, 0.85, 0.9]},
            "random": {"scores": [0.5, 0.4, 0.6, 0.5, 0.45]},
            "fixed": {"scores": [0.5, 0.5, 0.5, 0.5, 0.5]},
        }
        comparisons = compare_strategies(results)
        self.assertIn("qec_vs_random", comparisons)
        self.assertIn("qec_vs_fixed", comparisons)
        self.assertNotIn("qec_vs_qec", comparisons)

    def test_compare_strategies_has_keys(self) -> None:
        results = {
            "qec": {"scores": [0.8, 0.85]},
            "baseline": {"scores": [0.5, 0.5]},
        }
        comp = compare_strategies(results)["qec_vs_baseline"]
        for key in [
            "performance_ratio", "qec_final", "baseline_final",
            "convergence_signal_diff", "stability_diff",
        ]:
            self.assertIn(key, comp)

    def test_compare_strategies_missing_qec(self) -> None:
        with self.assertRaises(ValueError):
            compare_strategies({"random": {"scores": [0.5]}})

    def test_compare_determinism(self) -> None:
        results = {
            "qec": {"scores": [0.6, 0.7, 0.8]},
            "random": {"scores": [0.5, 0.4, 0.6]},
        }
        c1 = compare_strategies(results)
        c2 = compare_strategies(results)
        self.assertEqual(c1, c2)

    def test_no_mutation(self) -> None:
        results = {
            "qec": {"scores": [0.6, 0.7, 0.8]},
            "random": {"scores": [0.5, 0.4, 0.6]},
        }
        original = copy.deepcopy(results)
        compare_strategies(results)
        self.assertEqual(results, original)


class TestBenchmarkRunner(unittest.TestCase):
    """Tests for the full benchmark runner."""

    def test_run_benchmark_returns_all_keys(self) -> None:
        results = run_benchmark()
        for key in ["qec", "random", "fixed", "round_robin"]:
            self.assertIn(key, results)
            self.assertIn("scores", results[key])
            self.assertIn("regimes", results[key])
            self.assertIn("energies", results[key])
            self.assertIn("strategy_ids", results[key])

    def test_run_benchmark_determinism(self) -> None:
        """Two identical runs produce identical results."""
        r1 = run_benchmark()
        r2 = run_benchmark()
        for key in ["qec", "random", "fixed", "round_robin"]:
            self.assertEqual(r1[key]["scores"], r2[key]["scores"])
            self.assertEqual(r1[key]["regimes"], r2[key]["regimes"])
            self.assertEqual(r1[key]["strategy_ids"], r2[key]["strategy_ids"])

    def test_run_benchmark_scores_bounded(self) -> None:
        results = run_benchmark()
        for key in ["qec", "random", "fixed", "round_robin"]:
            for score in results[key]["scores"]:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_run_benchmark_consistent_lengths(self) -> None:
        results = run_benchmark()
        n = len(results["qec"]["scores"])
        for key in ["random", "fixed", "round_robin"]:
            self.assertEqual(len(results[key]["scores"]), n)

    def test_run_benchmark_no_input_mutation(self) -> None:
        from qec.experiments.metrics_probe import generate_test_inputs
        inputs = generate_test_inputs()
        original = copy.deepcopy(inputs)
        run_benchmark(inputs=inputs)
        self.assertEqual(inputs, original)

    def test_run_benchmark_custom_seed(self) -> None:
        r1 = run_benchmark(config={"seed": 1})
        r2 = run_benchmark(config={"seed": 2})
        # QEC results should be identical (seed only affects random baseline)
        self.assertEqual(r1["qec"]["scores"], r2["qec"]["scores"])
        # Random baseline should differ with different seeds
        self.assertNotEqual(r1["random"]["strategy_ids"], r2["random"]["strategy_ids"])


if __name__ == "__main__":
    unittest.main()
