"""Tests for v101.9.0 structure-aware dominance logic.

Covers:
- consistency gap computation
- temporal revival detection
- pairwise strategy correlation
- structure-aware dominance conditions
- redundancy pruning
- no mutation of inputs
- determinism guarantees
"""

from __future__ import annotations

import copy
import unittest

from qec.analysis.consistency_metrics import (
    compute_consistency_gap,
    enrich_with_consistency_gap,
)
from qec.analysis.dominance_pruning import dominates, pareto_prune
from qec.analysis.strategy_correlation import (
    compute_strategy_correlation,
    prune_redundant,
)
from qec.analysis.temporal_patterns import detect_revival, enrich_with_revival


# ---------------------------------------------------------------------------
# Consistency gap
# ---------------------------------------------------------------------------


class TestConsistencyGap(unittest.TestCase):
    """Tests for compute_consistency_gap."""

    def test_zero_gap(self):
        metrics = {"design_score": 0.5, "confidence_efficiency": 0.5}
        self.assertEqual(compute_consistency_gap(metrics), 0.0)

    def test_positive_gap(self):
        metrics = {"design_score": 0.8, "confidence_efficiency": 0.3}
        self.assertEqual(compute_consistency_gap(metrics), 0.5)

    def test_negative_difference_clamped(self):
        """If confidence_efficiency > design_score, gap is 0."""
        metrics = {"design_score": 0.2, "confidence_efficiency": 0.9}
        self.assertEqual(compute_consistency_gap(metrics), 0.0)

    def test_gap_clamped_to_one(self):
        metrics = {"design_score": 1.5, "confidence_efficiency": 0.0}
        self.assertEqual(compute_consistency_gap(metrics), 1.0)

    def test_missing_keys_default_zero(self):
        self.assertEqual(compute_consistency_gap({}), 0.0)

    def test_rounding(self):
        metrics = {"design_score": 1.0 / 3.0, "confidence_efficiency": 0.0}
        gap = compute_consistency_gap(metrics)
        # Should be rounded to 12 decimals
        self.assertEqual(gap, round(1.0 / 3.0, 12))

    def test_enrich_does_not_mutate(self):
        strategy = {
            "name": "test",
            "metrics": {"design_score": 0.7, "confidence_efficiency": 0.3},
        }
        original = copy.deepcopy(strategy)
        enriched = enrich_with_consistency_gap(strategy)
        self.assertEqual(strategy, original)
        self.assertIn("consistency_gap", enriched["metrics"])
        self.assertAlmostEqual(enriched["metrics"]["consistency_gap"], 0.4)

    def test_deterministic(self):
        metrics = {"design_score": 0.75, "confidence_efficiency": 0.25}
        r1 = compute_consistency_gap(metrics)
        r2 = compute_consistency_gap(metrics)
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# Revival detection
# ---------------------------------------------------------------------------


class TestRevivalDetection(unittest.TestCase):
    """Tests for detect_revival."""

    def test_clear_revival(self):
        history = [0.8, 0.3, 0.5, 0.7]
        result = detect_revival(history)
        self.assertTrue(result["has_revival"])
        self.assertGreater(result["revival_strength"], 0.0)
        self.assertEqual(result["min_value"], 0.3)
        self.assertEqual(result["recovered_to"], 0.7)

    def test_no_revival_monotone_decrease(self):
        history = [0.9, 0.7, 0.5, 0.3]
        result = detect_revival(history)
        # Min is at end, no recovery
        self.assertFalse(result["has_revival"])
        self.assertEqual(result["revival_strength"], 0.0)

    def test_no_revival_flat(self):
        history = [0.5, 0.5, 0.5]
        result = detect_revival(history)
        self.assertFalse(result["has_revival"])
        self.assertEqual(result["revival_strength"], 0.0)

    def test_single_element(self):
        result = detect_revival([0.5])
        self.assertFalse(result["has_revival"])

    def test_empty_history(self):
        result = detect_revival([])
        self.assertFalse(result["has_revival"])
        self.assertEqual(result["min_value"], 0.0)

    def test_full_recovery(self):
        """Recovery to the max value should give strength 1.0."""
        history = [1.0, 0.0, 1.0]
        result = detect_revival(history)
        self.assertTrue(result["has_revival"])
        self.assertEqual(result["revival_strength"], 1.0)

    def test_deterministic(self):
        history = [0.6, 0.1, 0.4, 0.8]
        r1 = detect_revival(history)
        r2 = detect_revival(history)
        self.assertEqual(r1, r2)

    def test_enrich_does_not_mutate(self):
        strategy = {"name": "s", "metrics": {"design_score": 0.5}}
        original = copy.deepcopy(strategy)
        enriched = enrich_with_revival(strategy, [0.5, 0.1, 0.4])
        self.assertEqual(strategy, original)
        self.assertIn("has_revival", enriched["metrics"])


# ---------------------------------------------------------------------------
# Strategy correlation
# ---------------------------------------------------------------------------


class TestStrategyCorrelation(unittest.TestCase):
    """Tests for compute_strategy_correlation."""

    def test_identical_strategies(self):
        a = {"metrics": {
            "design_score": 0.5, "confidence_efficiency": 0.5,
            "temporal_stability": 0.5, "trust_modulation": 0.5,
        }}
        self.assertEqual(compute_strategy_correlation(a, a), 1.0)

    def test_opposite_strategies(self):
        a = {"metrics": {
            "design_score": 1.0, "confidence_efficiency": 1.0,
            "temporal_stability": 1.0, "trust_modulation": 1.0,
        }}
        b = {"metrics": {
            "design_score": 0.0, "confidence_efficiency": 0.0,
            "temporal_stability": 0.0, "trust_modulation": 0.0,
        }}
        self.assertEqual(compute_strategy_correlation(a, b), 0.0)

    def test_symmetry(self):
        a = {"metrics": {
            "design_score": 0.8, "confidence_efficiency": 0.3,
            "temporal_stability": 0.6, "trust_modulation": 0.9,
        }}
        b = {"metrics": {
            "design_score": 0.5, "confidence_efficiency": 0.7,
            "temporal_stability": 0.4, "trust_modulation": 0.1,
        }}
        self.assertEqual(
            compute_strategy_correlation(a, b),
            compute_strategy_correlation(b, a),
        )

    def test_clamped_to_unit(self):
        a = {"metrics": {
            "design_score": 0.5, "confidence_efficiency": 0.5,
            "temporal_stability": 0.5, "trust_modulation": 0.5,
        }}
        b = {"metrics": {
            "design_score": 0.4, "confidence_efficiency": 0.4,
            "temporal_stability": 0.4, "trust_modulation": 0.4,
        }}
        corr = compute_strategy_correlation(a, b)
        self.assertGreaterEqual(corr, 0.0)
        self.assertLessEqual(corr, 1.0)

    def test_missing_keys_default_zero(self):
        a = {"metrics": {"design_score": 0.5}}
        b = {"metrics": {"design_score": 0.5}}
        corr = compute_strategy_correlation(a, b)
        self.assertGreaterEqual(corr, 0.0)
        self.assertLessEqual(corr, 1.0)

    def test_deterministic(self):
        a = {"metrics": {"design_score": 0.8, "confidence_efficiency": 0.3,
                         "temporal_stability": 0.6, "trust_modulation": 0.9}}
        b = {"metrics": {"design_score": 0.5, "confidence_efficiency": 0.7,
                         "temporal_stability": 0.4, "trust_modulation": 0.1}}
        r1 = compute_strategy_correlation(a, b)
        r2 = compute_strategy_correlation(a, b)
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# Redundancy pruning
# ---------------------------------------------------------------------------


class TestPruneRedundant(unittest.TestCase):
    """Tests for prune_redundant."""

    def test_no_redundancy(self):
        strategies = [
            {"name": "a", "metrics": {
                "design_score": 0.9, "confidence_efficiency": 0.1,
                "temporal_stability": 0.1, "trust_modulation": 0.1,
            }},
            {"name": "b", "metrics": {
                "design_score": 0.1, "confidence_efficiency": 0.9,
                "temporal_stability": 0.9, "trust_modulation": 0.9,
            }},
        ]
        result = prune_redundant(strategies, threshold=0.98)
        self.assertEqual(len(result), 2)

    def test_redundant_pair_keeps_lower_gap(self):
        strategies = [
            {"name": "a", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
                "consistency_gap": 0.1,
            }},
            {"name": "b", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
                "consistency_gap": 0.3,
            }},
        ]
        result = prune_redundant(strategies, threshold=0.98)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "a")

    def test_empty_input(self):
        self.assertEqual(prune_redundant([]), [])

    def test_does_not_mutate(self):
        strategies = [
            {"name": "a", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
            }},
            {"name": "b", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
            }},
        ]
        original = copy.deepcopy(strategies)
        prune_redundant(strategies, threshold=0.98)
        self.assertEqual(strategies, original)

    def test_deterministic(self):
        strategies = [
            {"name": "x", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
                "consistency_gap": 0.2,
            }},
            {"name": "y", "metrics": {
                "design_score": 0.5, "confidence_efficiency": 0.5,
                "temporal_stability": 0.5, "trust_modulation": 0.5,
                "consistency_gap": 0.4,
            }},
        ]
        r1 = prune_redundant(strategies, threshold=0.98)
        r2 = prune_redundant(strategies, threshold=0.98)
        self.assertEqual(
            [s["name"] for s in r1],
            [s["name"] for s in r2],
        )


# ---------------------------------------------------------------------------
# Structure-aware dominance
# ---------------------------------------------------------------------------


class TestStructureAwareDominance(unittest.TestCase):
    """Tests for structure-aware dominance conditions."""

    def _make(self, name, ds, ce, ts, tm, gap=0.0, revival=False, rs=0.0):
        return {
            "name": name,
            "metrics": {
                "design_score": ds,
                "confidence_efficiency": ce,
                "temporal_stability": ts,
                "trust_modulation": tm,
                "consistency_gap": gap,
                "has_revival": revival,
                "revival_strength": rs,
            },
        }

    def test_standard_dominance_still_works(self):
        """Without structure_aware, classic Pareto dominance applies."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.5)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.1)
        self.assertTrue(dominates(a, b, structure_aware=False))

    def test_structure_aware_blocks_high_gap(self):
        """A with higher consistency_gap cannot dominate B."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.5)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.1)
        self.assertFalse(dominates(a, b, structure_aware=True))

    def test_structure_aware_allows_lower_gap(self):
        """A with lower consistency_gap can dominate B."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.1)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.5)
        self.assertTrue(dominates(a, b, structure_aware=True))

    def test_revival_blocks_dominance(self):
        """If both have revival, lower revival_strength blocks dominance."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.1, revival=True, rs=0.2)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.5, revival=True, rs=0.9)
        self.assertFalse(dominates(a, b, structure_aware=True))

    def test_revival_allows_dominance(self):
        """If A has higher revival_strength, dominance can hold."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.1, revival=True, rs=0.9)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.5, revival=True, rs=0.2)
        self.assertTrue(dominates(a, b, structure_aware=True))

    def test_only_one_has_revival(self):
        """Revival constraint only applies when both have revival."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.1, revival=True, rs=0.1)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.5, revival=False, rs=0.9)
        self.assertTrue(dominates(a, b, structure_aware=True))

    def test_pareto_prune_structure_aware(self):
        """Structure-aware pareto_prune retains more strategies."""
        a = self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.5)
        b = self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.1)

        # Without structure-aware: a dominates b
        classic = pareto_prune([a, b], structure_aware=False)
        self.assertEqual(len(classic), 1)

        # With structure-aware: a cannot dominate b (gap constraint)
        aware = pareto_prune([a, b], structure_aware=True)
        self.assertEqual(len(aware), 2)

    def test_pareto_prune_deterministic(self):
        strategies = [
            self._make("c", 0.6, 0.6, 0.6, 0.6, gap=0.3),
            self._make("a", 0.8, 0.8, 0.8, 0.8, gap=0.1),
            self._make("b", 0.5, 0.5, 0.5, 0.5, gap=0.2),
        ]
        r1 = pareto_prune(strategies, structure_aware=True)
        r2 = pareto_prune(strategies, structure_aware=True)
        self.assertEqual(
            [s["name"] for s in r1],
            [s["name"] for s in r2],
        )


if __name__ == "__main__":
    unittest.main()
