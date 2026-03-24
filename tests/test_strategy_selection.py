"""Tests for trust-aware strategy selection (v101.5.0).

Covers:
- score_strategy: determinism, bounds, component contributions
- select_strategy: argmax, deterministic tie-breaking
- rank_strategies: ordering, rank assignment
- strategy_adapter: build_candidate_strategies, run_strategy_selection
- no-mutation guarantees
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from qec.analysis.strategy_selection import (
    rank_strategies,
    score_strategy,
    select_strategy,
)
from qec.analysis.strategy_adapter import (
    build_candidate_strategies,
    format_selection_summary,
    run_strategy_selection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    design_score: float = 0.5,
    confidence_efficiency: float = 0.5,
) -> Dict[str, float]:
    return {
        "design_score": design_score,
        "confidence_efficiency": confidence_efficiency,
    }


def _make_trust(
    stability: float = 0.5,
    global_trust: float = 0.5,
    regime_trust: float = 0.5,
    blended_trust: float | None = None,
) -> Dict[str, float]:
    ts: Dict[str, float] = {
        "stability": stability,
        "global_trust": global_trust,
        "regime_trust": regime_trust,
    }
    if blended_trust is not None:
        ts["blended_trust"] = blended_trust
    return ts


def _make_strategy(
    name: str,
    design_score: float = 0.5,
    confidence_efficiency: float = 0.5,
    stability: float = 0.5,
    blended_trust: float = 0.5,
) -> Dict[str, Any]:
    return {
        "name": name,
        "metrics": _make_metrics(design_score, confidence_efficiency),
        "trust_signals": _make_trust(
            stability=stability, blended_trust=blended_trust,
        ),
    }


def _make_experiment_result() -> Dict[str, Any]:
    return {
        "n_signals": 8,
        "threshold": 0.3,
        "rounds": 3,
        "metrics": {
            "design_score": 0.7,
            "confidence_efficiency": 0.6,
            "neutral_usage": 0.25,
            "agreement_rate": 0.875,
        },
        "final_states": [1, -1, 1, 0, -1, 1, 0, 1],
        "final_confidences": [0.8, 0.7, 0.9, 0.2, 0.6, 0.8, 0.2, 0.7],
    }


# ---------------------------------------------------------------------------
# score_strategy tests
# ---------------------------------------------------------------------------

class TestScoreStrategy:

    def test_determinism(self):
        """Same inputs must produce identical score."""
        m = _make_metrics(0.8, 0.7)
        t = _make_trust(stability=0.9, blended_trust=0.6)
        s1 = score_strategy(m, t)
        s2 = score_strategy(m, t)
        assert s1 == s2

    def test_bounded_zero(self):
        """All-zero inputs produce score in [0, 1]."""
        s = score_strategy(_make_metrics(0.0, 0.0), _make_trust(0.0, 0.0, 0.0, 0.0))
        assert 0.0 <= s <= 1.0

    def test_bounded_one(self):
        """All-one inputs produce score in [0, 1]."""
        s = score_strategy(_make_metrics(1.0, 1.0), _make_trust(1.0, 1.0, 1.0, 1.0))
        assert 0.0 <= s <= 1.0

    def test_score_increases_with_design(self):
        """Higher design_score should increase total score."""
        t = _make_trust(0.5, blended_trust=0.5)
        s_low = score_strategy(_make_metrics(0.2, 0.5), t)
        s_high = score_strategy(_make_metrics(0.9, 0.5), t)
        assert s_high > s_low

    def test_score_increases_with_trust(self):
        """Higher trust should increase total score."""
        m = _make_metrics(0.5, 0.5)
        s_low = score_strategy(m, _make_trust(0.5, blended_trust=0.1))
        s_high = score_strategy(m, _make_trust(0.5, blended_trust=0.9))
        assert s_high > s_low

    def test_score_increases_with_stability(self):
        """Higher stability should increase total score."""
        m = _make_metrics(0.5, 0.5)
        s_low = score_strategy(m, _make_trust(stability=0.1, blended_trust=0.5))
        s_high = score_strategy(m, _make_trust(stability=0.9, blended_trust=0.5))
        assert s_high > s_low

    def test_missing_trust_uses_defaults(self):
        """Missing trust signals should use neutral defaults."""
        m = _make_metrics(0.5, 0.5)
        s = score_strategy(m, {})
        assert 0.0 <= s <= 1.0
        # With neutral defaults (0.5), score should be 0.5
        assert s == 0.5

    def test_no_mutation_of_inputs(self):
        """Input dicts must not be mutated."""
        m = _make_metrics(0.7, 0.6)
        t = _make_trust(0.8, blended_trust=0.7)
        m_copy = copy.deepcopy(m)
        t_copy = copy.deepcopy(t)
        score_strategy(m, t)
        assert m == m_copy
        assert t == t_copy

    def test_rounding_precision(self):
        """Score should be rounded to 12 decimal places."""
        m = _make_metrics(1.0 / 3.0, 2.0 / 7.0)
        t = _make_trust(stability=1.0 / 11.0, blended_trust=3.0 / 13.0)
        s = score_strategy(m, t)
        # Check that it's rounded to 12 decimals
        assert s == round(s, 12)

    def test_blended_trust_preferred_over_global(self):
        """When blended_trust is present, it should be used over global_trust."""
        m = _make_metrics(0.5, 0.5)
        # blended=0.9, global=0.1 — if blended is used, score is higher
        s_blended = score_strategy(m, {"blended_trust": 0.9, "global_trust": 0.1,
                                        "stability": 0.5})
        s_global = score_strategy(m, {"global_trust": 0.1, "stability": 0.5})
        assert s_blended > s_global


# ---------------------------------------------------------------------------
# select_strategy tests
# ---------------------------------------------------------------------------

class TestSelectStrategy:

    def test_selects_highest_score(self):
        """Must select the strategy with the highest score."""
        strategies = [
            _make_strategy("low", design_score=0.2),
            _make_strategy("high", design_score=0.9),
            _make_strategy("mid", design_score=0.5),
        ]
        selected = select_strategy(strategies)
        assert selected["name"] == "high"

    def test_deterministic_tie_break(self):
        """Tied scores must break by name (alphabetical)."""
        strategies = [
            _make_strategy("zebra", design_score=0.5),
            _make_strategy("alpha", design_score=0.5),
            _make_strategy("beta", design_score=0.5),
        ]
        selected = select_strategy(strategies)
        assert selected["name"] == "alpha"

    def test_score_attached(self):
        """Selected strategy must have _score field."""
        strategies = [_make_strategy("only", design_score=0.7)]
        selected = select_strategy(strategies)
        assert "_score" in selected
        assert 0.0 <= selected["_score"] <= 1.0

    def test_empty_raises(self):
        """Empty list must raise ValueError."""
        try:
            select_strategy([])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_no_mutation_of_inputs(self):
        """Input strategies must not be mutated."""
        strategies = [
            _make_strategy("a", design_score=0.3),
            _make_strategy("b", design_score=0.7),
        ]
        original = copy.deepcopy(strategies)
        select_strategy(strategies)
        assert strategies == original

    def test_determinism_repeated_calls(self):
        """Multiple calls with same input must return same result."""
        strategies = [
            _make_strategy("x", design_score=0.6),
            _make_strategy("y", design_score=0.8),
        ]
        r1 = select_strategy(strategies)
        r2 = select_strategy(strategies)
        assert r1["name"] == r2["name"]
        assert r1["_score"] == r2["_score"]

    def test_single_strategy(self):
        """Single strategy must be selected."""
        strategies = [_make_strategy("only")]
        selected = select_strategy(strategies)
        assert selected["name"] == "only"


# ---------------------------------------------------------------------------
# rank_strategies tests
# ---------------------------------------------------------------------------

class TestRankStrategies:

    def test_ordering(self):
        """Strategies must be ordered by descending score."""
        strategies = [
            _make_strategy("low", design_score=0.1),
            _make_strategy("high", design_score=0.9),
            _make_strategy("mid", design_score=0.5),
        ]
        ranked = rank_strategies(strategies)
        scores = [r["_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_assigned(self):
        """Each entry must have _rank (1-based)."""
        strategies = [_make_strategy("a"), _make_strategy("b")]
        ranked = rank_strategies(strategies)
        ranks = [r["_rank"] for r in ranked]
        assert sorted(ranks) == [1, 2]

    def test_empty_input(self):
        """Empty input returns empty list."""
        assert rank_strategies([]) == []

    def test_no_mutation(self):
        """Input must not be mutated."""
        strategies = [_make_strategy("a"), _make_strategy("b")]
        original = copy.deepcopy(strategies)
        rank_strategies(strategies)
        assert strategies == original


# ---------------------------------------------------------------------------
# strategy_adapter tests
# ---------------------------------------------------------------------------

class TestBuildCandidateStrategies:

    def test_default_candidate(self):
        """Without configs, builds single default candidate."""
        result = _make_experiment_result()
        candidates = build_candidate_strategies(result)
        assert len(candidates) == 1
        assert candidates[0]["name"] == "default"
        assert candidates[0]["metrics"]["design_score"] == 0.7

    def test_with_configs(self):
        """With configs, builds one candidate per config."""
        result = _make_experiment_result()
        configs = [
            {"name": "conservative", "threshold": 0.4},
            {"name": "aggressive", "threshold": 0.2},
        ]
        candidates = build_candidate_strategies(result, strategy_configs=configs)
        assert len(candidates) == 2
        names = [c["name"] for c in candidates]
        assert "conservative" in names
        assert "aggressive" in names

    def test_trust_signals_passed_through(self):
        """Trust signals should be included in candidates."""
        result = _make_experiment_result()
        ts = {"stability": 0.9, "global_trust": 0.7}
        candidates = build_candidate_strategies(result, trust_signals=ts)
        assert candidates[0]["trust_signals"]["stability"] == 0.9

    def test_no_mutation(self):
        """Experiment result must not be mutated."""
        result = _make_experiment_result()
        original = copy.deepcopy(result)
        build_candidate_strategies(result)
        assert result == original


class TestRunStrategySelection:

    def test_returns_selected(self):
        """Must return a selected strategy."""
        result = _make_experiment_result()
        sel = run_strategy_selection(result)
        assert "selected" in sel
        assert "candidates" in sel
        assert sel["n_candidates"] >= 1

    def test_determinism(self):
        """Repeated calls produce identical results."""
        result = _make_experiment_result()
        configs = [
            {"name": "a", "threshold": 0.3},
            {"name": "b", "threshold": 0.4},
        ]
        ts = {"stability": 0.7, "global_trust": 0.5}
        s1 = run_strategy_selection(result, ts, configs)
        s2 = run_strategy_selection(result, ts, configs)
        assert s1["selected"]["name"] == s2["selected"]["name"]
        assert s1["selected"]["_score"] == s2["selected"]["_score"]

    def test_with_different_trust(self):
        """Different trust signals should potentially change selection."""
        result = _make_experiment_result()
        configs = [
            {"name": "a", "metrics": {"design_score": 0.5, "confidence_efficiency": 0.5},
             "trust_signals": {"stability": 0.9, "blended_trust": 0.9}},
            {"name": "b", "metrics": {"design_score": 0.5, "confidence_efficiency": 0.5},
             "trust_signals": {"stability": 0.1, "blended_trust": 0.1}},
        ]
        sel = run_strategy_selection(result, strategy_configs=configs)
        assert sel["selected"]["name"] == "a"

    def test_format_summary(self):
        """Format should produce readable output."""
        result = _make_experiment_result()
        sel = run_strategy_selection(result)
        summary = format_selection_summary(sel)
        assert "Strategy Selection" in summary
        assert "Selected" in summary


# ---------------------------------------------------------------------------
# Bounded score edge cases
# ---------------------------------------------------------------------------

class TestBoundedScores:

    def test_extreme_low(self):
        """Negative metric values should be clamped to 0."""
        s = score_strategy(
            {"design_score": -1.0, "confidence_efficiency": -1.0},
            {"stability": -1.0, "blended_trust": -1.0},
        )
        assert s == 0.0

    def test_extreme_high(self):
        """Values above 1.0 should be clamped."""
        s = score_strategy(
            {"design_score": 5.0, "confidence_efficiency": 5.0},
            {"stability": 5.0, "blended_trust": 5.0},
        )
        assert s == 1.0
