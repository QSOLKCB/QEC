"""Backward-compatibility verification for dominance_pruning.py (v101.9.0).

This file mirrors the test logic from test_dominance_pruning.py (which
requires pytest) using unittest.TestCase so it can run without pytest.

Purpose: verify that adding the structure_aware keyword argument to
dominates() and pareto_prune() does NOT change behavior when the
parameter is omitted or set to False.

Run with:
    PYTHONPATH=src python -m unittest tests.test_dominance_backward_compat -v
"""

from __future__ import annotations

import copy
import unittest

from qec.analysis.dominance_pruning import (
    DOMINANCE_KEYS,
    dominates,
    pareto_prune,
    pruning_stats,
)


def _make_strategy(name: str, **metric_overrides) -> dict:
    metrics = {
        "design_score": 0.5,
        "confidence_efficiency": 0.5,
        "temporal_stability": 0.5,
        "trust_modulation": 0.5,
    }
    metrics.update(metric_overrides)
    return {"name": name, "metrics": metrics}


# ---------------------------------------------------------------------------
# dominates() backward compatibility
# ---------------------------------------------------------------------------


class TestDominatesBackwardCompat(unittest.TestCase):
    """Exact mirror of pytest-based TestDominates from test_dominance_pruning."""

    def test_strictly_better_dominates(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        self.assertTrue(dominates(a, b))

    def test_better_on_one_equal_on_rest(self):
        a = _make_strategy("a", design_score=0.6)
        b = _make_strategy("b", design_score=0.5)
        self.assertTrue(dominates(a, b))

    def test_identical_does_not_dominate(self):
        a = _make_strategy("a")
        b = _make_strategy("b")
        self.assertFalse(dominates(a, b))

    def test_worse_on_one_does_not_dominate(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.3)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.5)
        self.assertFalse(dominates(a, b))

    def test_symmetry_exclusion(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        self.assertTrue(dominates(a, b))
        self.assertFalse(dominates(b, a))

    def test_missing_metrics_default_to_zero(self):
        a = _make_strategy("a")
        b = {"name": "b", "metrics": {}}
        self.assertTrue(dominates(a, b))

    def test_uses_fixed_key_order(self):
        self.assertEqual(len(DOMINANCE_KEYS), 4)
        self.assertEqual(DOMINANCE_KEYS[0], "design_score")
        self.assertEqual(DOMINANCE_KEYS[1], "confidence_efficiency")
        self.assertEqual(DOMINANCE_KEYS[2], "temporal_stability")
        self.assertEqual(DOMINANCE_KEYS[3], "trust_modulation")

    def test_explicit_false_matches_default(self):
        """structure_aware=False produces identical result to omitting it."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        self.assertEqual(dominates(a, b), dominates(a, b, structure_aware=False))
        self.assertEqual(dominates(b, a), dominates(b, a, structure_aware=False))


# ---------------------------------------------------------------------------
# pareto_prune() backward compatibility
# ---------------------------------------------------------------------------


class TestParetoPruneBackwardCompat(unittest.TestCase):
    """Exact mirror of pytest-based TestParetoPrune from test_dominance_pruning."""

    def test_removes_dominated(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        result = pareto_prune([a, b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "a")

    def test_keeps_identical(self):
        a = _make_strategy("a")
        b = _make_strategy("b")
        result = pareto_prune([a, b])
        self.assertEqual(len(result), 2)

    def test_keeps_pareto_front(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.1)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.9)
        result = pareto_prune([a, b])
        self.assertEqual(len(result), 2)

    def test_deterministic_output(self):
        strategies = [
            _make_strategy("c", design_score=0.9, confidence_efficiency=0.1),
            _make_strategy("a", design_score=0.1, confidence_efficiency=0.9),
            _make_strategy("b", design_score=0.5, confidence_efficiency=0.5),
        ]
        r1 = pareto_prune(strategies)
        r2 = pareto_prune(strategies)
        self.assertEqual(len(r1), len(r2))
        for x, y in zip(r1, r2):
            self.assertEqual(x["name"], y["name"])
            self.assertEqual(x["metrics"], y["metrics"])

    def test_sorted_by_name(self):
        strategies = [
            _make_strategy("z_strategy", design_score=0.9),
            _make_strategy("a_strategy", design_score=0.9),
        ]
        result = pareto_prune(strategies)
        names = [s["name"] for s in result]
        self.assertEqual(names, sorted(names))

    def test_no_mutation(self):
        strategies = [
            _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9),
            _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1),
        ]
        original = copy.deepcopy(strategies)
        pareto_prune(strategies)
        self.assertEqual(strategies, original)

    def test_pruning_reduces_or_equals(self):
        strategies = [
            _make_strategy("a", design_score=0.9),
            _make_strategy("b", design_score=0.5),
            _make_strategy("c", design_score=0.1),
        ]
        result = pareto_prune(strategies)
        self.assertLessEqual(len(result), len(strategies))

    def test_empty_input(self):
        self.assertEqual(pareto_prune([]), [])

    def test_single_strategy(self):
        s = _make_strategy("solo")
        result = pareto_prune([s])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "solo")

    def test_known_synthetic_case(self):
        a = _make_strategy("a", design_score=0.8, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5)
        b = _make_strategy("b", design_score=0.2, confidence_efficiency=0.8,
                           temporal_stability=0.5, trust_modulation=0.5)
        c = _make_strategy("c", design_score=0.3, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5)
        result = pareto_prune([a, b, c])
        names = [s["name"] for s in result]
        self.assertIn("a", names)
        self.assertIn("b", names)
        self.assertNotIn("c", names)
        self.assertEqual(len(result), 2)

    def test_chain_domination(self):
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.5,
                           temporal_stability=0.5, trust_modulation=0.5)
        c = _make_strategy("c", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        result = pareto_prune([c, a, b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "a")

    def test_explicit_false_matches_default(self):
        """structure_aware=False produces identical result to omitting it."""
        strategies = [
            _make_strategy("a", design_score=0.8, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5),
            _make_strategy("b", design_score=0.2, confidence_efficiency=0.8,
                           temporal_stability=0.5, trust_modulation=0.5),
            _make_strategy("c", design_score=0.3, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5),
        ]
        r_default = pareto_prune(strategies)
        r_explicit = pareto_prune(strategies, structure_aware=False)
        self.assertEqual(
            [s["name"] for s in r_default],
            [s["name"] for s in r_explicit],
        )


# ---------------------------------------------------------------------------
# pruning_stats() backward compatibility
# ---------------------------------------------------------------------------


class TestPruningStatsBackwardCompat(unittest.TestCase):
    """Exact mirror of pytest-based TestPruningStats."""

    def test_basic_stats(self):
        original = [_make_strategy("a"), _make_strategy("b"), _make_strategy("c")]
        pruned = [_make_strategy("a")]
        stats = pruning_stats(original, pruned)
        self.assertEqual(stats["pruned_count"], 2)
        self.assertEqual(stats["retained_count"], 1)
        self.assertAlmostEqual(stats["dominance_ratio"], 2.0 / 3.0, places=10)

    def test_no_pruning(self):
        original = [_make_strategy("a"), _make_strategy("b")]
        stats = pruning_stats(original, original)
        self.assertEqual(stats["pruned_count"], 0)
        self.assertEqual(stats["retained_count"], 2)
        self.assertEqual(stats["dominance_ratio"], 0.0)

    def test_empty_input(self):
        stats = pruning_stats([], [])
        self.assertEqual(stats["pruned_count"], 0)
        self.assertEqual(stats["retained_count"], 0)
        self.assertEqual(stats["dominance_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
