"""Tests for deterministic dominance pruning (v101.8.0).

Covers:
- dominates: correct pairwise dominance detection
- pareto_prune: correct filtering, determinism, no mutation
- pruning_stats: correct statistics
- edge cases: identical strategies, empty input, single strategy
"""

from __future__ import annotations

import copy

from qec.analysis.dominance_pruning import (
    DOMINANCE_KEYS,
    dominates,
    pareto_prune,
    pruning_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(name: str, **metric_overrides) -> dict:
    """Build a strategy dict with given metric values."""
    metrics = {
        "design_score": 0.5,
        "confidence_efficiency": 0.5,
        "temporal_stability": 0.5,
        "trust_modulation": 0.5,
    }
    metrics.update(metric_overrides)
    return {"name": name, "metrics": metrics}


# ---------------------------------------------------------------------------
# dominates() tests
# ---------------------------------------------------------------------------

class TestDominates:

    def test_strictly_better_dominates(self):
        """A strategy strictly better on all keys dominates."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        assert dominates(a, b) is True

    def test_better_on_one_equal_on_rest(self):
        """Better on one key, equal on others => dominates."""
        a = _make_strategy("a", design_score=0.6)
        b = _make_strategy("b", design_score=0.5)
        assert dominates(a, b) is True

    def test_identical_does_not_dominate(self):
        """Identical metric values => no domination."""
        a = _make_strategy("a")
        b = _make_strategy("b")
        assert dominates(a, b) is False

    def test_worse_on_one_does_not_dominate(self):
        """Better on some but worse on one => no domination."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.3)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.5)
        assert dominates(a, b) is False

    def test_symmetry_exclusion(self):
        """If A dominates B, B does not dominate A."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        assert dominates(a, b) is True
        assert dominates(b, a) is False

    def test_missing_metrics_default_to_zero(self):
        """Missing metric keys default to 0.0."""
        a = _make_strategy("a")
        b = {"name": "b", "metrics": {}}  # all default to 0.0
        assert dominates(a, b) is True

    def test_uses_fixed_key_order(self):
        """Dominance uses exactly the four fixed keys."""
        assert len(DOMINANCE_KEYS) == 4
        assert DOMINANCE_KEYS[0] == "design_score"
        assert DOMINANCE_KEYS[1] == "confidence_efficiency"
        assert DOMINANCE_KEYS[2] == "temporal_stability"
        assert DOMINANCE_KEYS[3] == "trust_modulation"


# ---------------------------------------------------------------------------
# pareto_prune() tests
# ---------------------------------------------------------------------------

class TestParetoPrune:

    def test_removes_dominated(self):
        """Dominated strategy is removed."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        result = pareto_prune([a, b])
        assert len(result) == 1
        assert result[0]["name"] == "a"

    def test_keeps_identical(self):
        """Identical strategies are all non-dominated."""
        a = _make_strategy("a")
        b = _make_strategy("b")
        result = pareto_prune([a, b])
        assert len(result) == 2

    def test_keeps_pareto_front(self):
        """Non-dominated strategies on different axes are all kept."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.1)
        b = _make_strategy("b", design_score=0.1, confidence_efficiency=0.9)
        result = pareto_prune([a, b])
        assert len(result) == 2

    def test_deterministic_output(self):
        """Two calls produce identical results."""
        strategies = [
            _make_strategy("c", design_score=0.9, confidence_efficiency=0.1),
            _make_strategy("a", design_score=0.1, confidence_efficiency=0.9),
            _make_strategy("b", design_score=0.5, confidence_efficiency=0.5),
        ]
        r1 = pareto_prune(strategies)
        r2 = pareto_prune(strategies)
        assert len(r1) == len(r2)
        for x, y in zip(r1, r2):
            assert x["name"] == y["name"]
            assert x["metrics"] == y["metrics"]

    def test_sorted_by_name(self):
        """Result is sorted by name."""
        strategies = [
            _make_strategy("z_strategy", design_score=0.9),
            _make_strategy("a_strategy", design_score=0.9),
        ]
        result = pareto_prune(strategies)
        names = [s["name"] for s in result]
        assert names == sorted(names)

    def test_no_mutation(self):
        """Input list and strategy dicts are not mutated."""
        strategies = [
            _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9),
            _make_strategy("b", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1),
        ]
        original = copy.deepcopy(strategies)
        pareto_prune(strategies)
        assert strategies == original

    def test_pruning_reduces_or_equals(self):
        """Output size is always <= input size."""
        strategies = [
            _make_strategy("a", design_score=0.9),
            _make_strategy("b", design_score=0.5),
            _make_strategy("c", design_score=0.1),
        ]
        result = pareto_prune(strategies)
        assert len(result) <= len(strategies)

    def test_empty_input(self):
        """Empty input returns empty output."""
        assert pareto_prune([]) == []

    def test_single_strategy(self):
        """Single strategy is always non-dominated."""
        s = _make_strategy("solo")
        result = pareto_prune([s])
        assert len(result) == 1
        assert result[0]["name"] == "solo"

    def test_known_synthetic_case(self):
        """Known case: 3 strategies, 1 dominated, 2 on Pareto front."""
        a = _make_strategy("a", design_score=0.8, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5)
        b = _make_strategy("b", design_score=0.2, confidence_efficiency=0.8,
                           temporal_stability=0.5, trust_modulation=0.5)
        # c is dominated by a (worse design, same everything else)
        c = _make_strategy("c", design_score=0.3, confidence_efficiency=0.2,
                           temporal_stability=0.5, trust_modulation=0.5)
        result = pareto_prune([a, b, c])
        names = [s["name"] for s in result]
        assert "a" in names
        assert "b" in names
        assert "c" not in names
        assert len(result) == 2

    def test_chain_domination(self):
        """A > B > C: only A survives."""
        a = _make_strategy("a", design_score=0.9, confidence_efficiency=0.9,
                           temporal_stability=0.9, trust_modulation=0.9)
        b = _make_strategy("b", design_score=0.5, confidence_efficiency=0.5,
                           temporal_stability=0.5, trust_modulation=0.5)
        c = _make_strategy("c", design_score=0.1, confidence_efficiency=0.1,
                           temporal_stability=0.1, trust_modulation=0.1)
        result = pareto_prune([c, a, b])
        assert len(result) == 1
        assert result[0]["name"] == "a"


# ---------------------------------------------------------------------------
# pruning_stats() tests
# ---------------------------------------------------------------------------

class TestPruningStats:

    def test_basic_stats(self):
        original = [_make_strategy("a"), _make_strategy("b"), _make_strategy("c")]
        pruned = [_make_strategy("a")]
        stats = pruning_stats(original, pruned)
        assert stats["pruned_count"] == 2
        assert stats["retained_count"] == 1
        assert abs(stats["dominance_ratio"] - 2.0 / 3.0) < 1e-10

    def test_no_pruning(self):
        original = [_make_strategy("a"), _make_strategy("b")]
        stats = pruning_stats(original, original)
        assert stats["pruned_count"] == 0
        assert stats["retained_count"] == 2
        assert stats["dominance_ratio"] == 0.0

    def test_empty_input(self):
        stats = pruning_stats([], [])
        assert stats["pruned_count"] == 0
        assert stats["retained_count"] == 0
        assert stats["dominance_ratio"] == 0.0
