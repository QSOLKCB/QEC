"""Tests for the deterministic multi-strategy consensus engine (v97.9.0).

Covers:
- strategy extraction
- thinking hat scoring
- consensus matrix construction
- Byzantine filtering
- consensus selection
- determinism guarantees
- edge cases
"""

import numpy as np
import pytest

from qec.analysis.consensus_engine import (
    BYZANTINE_MIN_CONFIDENCE,
    HAT_NAMES,
    ConsensusStrategy,
    Strategy,
    build_consensus_matrix,
    extract_metrics,
    extract_strategies,
    filter_strategies,
    run_consensus,
    score_strategy,
    score_strategy_detailed,
    select_consensus,
    white_hat,
    red_hat,
    black_hat,
    yellow_hat,
    green_hat,
    blue_hat,
    _clamp,
)
from qec.analysis.law_promotion import Condition, Law


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _make_law(
    law_id: str,
    action: str,
    conditions: list = None,
    confidence: float = 0.8,
    coverage: float = 0.7,
    law_score: float = 0.6,
) -> Law:
    """Create a test law with given parameters."""
    if conditions is None:
        conditions = [Condition("variance", "gt", 0.01)]
    return Law(
        law_id=law_id,
        version=1,
        conditions=conditions,
        action=action,
        evidence=["run_1"],
        scores={
            "confidence": confidence,
            "coverage": coverage,
            "law_score": law_score,
            "consistency": 1.0,
        },
        created_at=0.0,
    )


def _make_state(values: list) -> dict:
    return {"values": np.array(values, dtype=np.float64), "step": 0}


# ---------------------------------------------------------------------------
# TEST: STRATEGY EXTRACTION
# ---------------------------------------------------------------------------


class TestExtractStrategies:
    def test_applicable_laws_are_extracted(self):
        laws = [
            _make_law("law_a", "reduce_oscillation"),
            _make_law("law_b", "stabilize"),
        ]
        metrics = {"variance": 0.5}
        strategies = extract_strategies(laws, metrics)
        assert len(strategies) == 2
        assert "law_a" in strategies
        assert "law_b" in strategies

    def test_inapplicable_laws_are_excluded(self):
        laws = [
            _make_law("law_a", "reduce_oscillation", [Condition("variance", "gt", 10.0)]),
        ]
        metrics = {"variance": 0.5}
        strategies = extract_strategies(laws, metrics)
        assert len(strategies) == 0

    def test_unmapped_actions_are_excluded(self):
        laws = [
            _make_law("law_a", "unknown_action_xyz"),
        ]
        metrics = {"variance": 0.5}
        strategies = extract_strategies(laws, metrics)
        assert len(strategies) == 0

    def test_strategy_has_correct_action_type(self):
        laws = [_make_law("law_a", "hard_correction")]
        metrics = {"variance": 0.5}
        strategies = extract_strategies(laws, metrics)
        assert strategies["law_a"].action_type == "correction_mode"
        assert strategies["law_a"].params == {"mode": "hard"}

    def test_empty_laws(self):
        strategies = extract_strategies([], {"variance": 0.5})
        assert strategies == {}


# ---------------------------------------------------------------------------
# TEST: THINKING HATS
# ---------------------------------------------------------------------------


class TestThinkingHats:
    def _make_strategy(self, confidence=0.8, coverage=0.7, law_score=0.6, n_conditions=1):
        conditions = [Condition("variance", "gt", 0.01)] * n_conditions
        law = _make_law("law_x", "reduce_oscillation", conditions,
                        confidence=confidence, coverage=coverage, law_score=law_score)
        return Strategy("law_x", law, "adjust_damping", {"alpha": 0.5})

    def test_white_hat_scores_in_range(self):
        strat = self._make_strategy()
        result = white_hat(strat, {"variance": 0.5})
        assert 0.0 <= result["score"] <= 1.0

    def test_white_hat_high_confidence(self):
        strat_high = self._make_strategy(confidence=1.0, coverage=1.0)
        strat_low = self._make_strategy(confidence=0.1, coverage=0.1)
        r_high = white_hat(strat_high, {"variance": 0.5})
        r_low = white_hat(strat_low, {"variance": 0.5})
        assert r_high["score"] > r_low["score"]

    def test_red_hat_stable_vs_unstable(self):
        strat = self._make_strategy()
        stable = red_hat(strat, {"variance": 0.0, "delta": 0.0})
        unstable = red_hat(strat, {"variance": 100.0, "delta": 50.0})
        assert stable["score"] > unstable["score"]

    def test_black_hat_high_specificity(self):
        strat_specific = self._make_strategy(n_conditions=5)
        strat_general = self._make_strategy(n_conditions=1)
        r_spec = black_hat(strat_specific, {"variance": 0.5})
        r_gen = black_hat(strat_general, {"variance": 0.5})
        assert r_spec["score"] > r_gen["score"]

    def test_yellow_hat_high_variance_benefit(self):
        strat = self._make_strategy(law_score=0.8)
        high_var = yellow_hat(strat, {"variance": 100.0})
        low_var = yellow_hat(strat, {"variance": 0.0})
        assert high_var["score"] > low_var["score"]

    def test_green_hat_novelty(self):
        strat_novel = self._make_strategy(n_conditions=1, coverage=0.1)
        strat_known = self._make_strategy(n_conditions=5, coverage=0.9)
        r_novel = green_hat(strat_novel, {"variance": 0.5})
        r_known = green_hat(strat_known, {"variance": 0.5})
        assert r_novel["score"] > r_known["score"]

    def test_blue_hat_is_meta_score(self):
        strat = self._make_strategy()
        other_scores = {"white": 0.8, "red": 0.6, "black": 0.7, "yellow": 0.9, "green": 0.5}
        result = blue_hat(strat, {"variance": 0.5}, other_scores)
        assert 0.0 <= result["score"] <= 1.0

    def test_all_hat_scores_in_range(self):
        strat = self._make_strategy()
        scores = score_strategy(strat, {"variance": 0.5, "delta": 0.1})
        for hat in HAT_NAMES:
            assert hat in scores
            assert 0.0 <= scores[hat] <= 1.0


# ---------------------------------------------------------------------------
# TEST: SCORE STRATEGY
# ---------------------------------------------------------------------------


class TestScoreStrategy:
    def test_returns_all_hats(self):
        law = _make_law("law_a", "reduce_oscillation")
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        scores = score_strategy(strat, {"variance": 0.5, "delta": 0.1})
        assert set(scores.keys()) == set(HAT_NAMES)

    def test_detailed_includes_reasons(self):
        law = _make_law("law_a", "reduce_oscillation")
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        details = score_strategy_detailed(strat, {"variance": 0.5})
        for hat in HAT_NAMES:
            assert "score" in details[hat]
            assert "reason" in details[hat]


# ---------------------------------------------------------------------------
# TEST: CONSENSUS MATRIX
# ---------------------------------------------------------------------------


class TestConsensusMatrix:
    def test_self_comparison(self):
        strat = Strategy("law_a", _make_law("law_a", "reduce_oscillation"),
                         "adjust_damping", {"alpha": 0.5})
        scores = {"law_a": score_strategy(strat, {"variance": 0.5})}
        matrix = build_consensus_matrix(["law_a"], scores, {"law_a": strat})
        cell = matrix["law_a"]["law_a"]
        assert cell["agreement"] == 1.0
        assert cell["conflict"] == 0.0
        assert cell["dominance"] == 0.5

    def test_same_action_agreement(self):
        law_a = _make_law("law_a", "reduce_oscillation")
        law_b = _make_law("law_b", "increase_damping")  # also adjust_damping
        strat_a = Strategy("law_a", law_a, "adjust_damping", {"alpha": 0.5})
        strat_b = Strategy("law_b", law_b, "adjust_damping", {"alpha": 0.3})
        strategies = {"law_a": strat_a, "law_b": strat_b}
        scores = {sid: score_strategy(strategies[sid], {"variance": 0.5}) for sid in strategies}
        matrix = build_consensus_matrix(["law_a", "law_b"], scores, strategies)
        assert matrix["law_a"]["law_b"]["agreement"] == 1.0
        assert matrix["law_a"]["law_b"]["conflict"] == 0.0

    def test_different_action_conflict(self):
        law_a = _make_law("law_a", "reduce_oscillation")
        law_b = _make_law("law_b", "stabilize")
        strat_a = Strategy("law_a", law_a, "adjust_damping", {"alpha": 0.5})
        strat_b = Strategy("law_b", law_b, "schedule_update", {"mode": "sequential"})
        strategies = {"law_a": strat_a, "law_b": strat_b}
        scores = {sid: score_strategy(strategies[sid], {"variance": 0.5}) for sid in strategies}
        matrix = build_consensus_matrix(["law_a", "law_b"], scores, strategies)
        assert matrix["law_a"]["law_b"]["conflict"] == 1.0
        assert matrix["law_a"]["law_b"]["agreement"] == 0.0

    def test_dominance_is_fraction(self):
        law_a = _make_law("law_a", "reduce_oscillation", confidence=0.9, law_score=0.9)
        law_b = _make_law("law_b", "stabilize", confidence=0.1, law_score=0.1)
        strat_a = Strategy("law_a", law_a, "adjust_damping", {"alpha": 0.5})
        strat_b = Strategy("law_b", law_b, "schedule_update", {"mode": "sequential"})
        strategies = {"law_a": strat_a, "law_b": strat_b}
        scores = {sid: score_strategy(strategies[sid], {"variance": 0.5}) for sid in strategies}
        matrix = build_consensus_matrix(["law_a", "law_b"], scores, strategies)
        dom = matrix["law_a"]["law_b"]["dominance"]
        assert 0.0 <= dom <= 1.0


# ---------------------------------------------------------------------------
# TEST: BYZANTINE FILTER
# ---------------------------------------------------------------------------


class TestByzantineFilter:
    def test_single_strategy_survives(self):
        law = _make_law("law_a", "reduce_oscillation", confidence=0.8)
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        strategies = {"law_a": strat}
        scores = {"law_a": score_strategy(strat, {"variance": 0.5})}
        matrix = build_consensus_matrix(["law_a"], scores, strategies)
        survivors = filter_strategies(["law_a"], scores, matrix, strategies)
        assert survivors == ["law_a"]

    def test_low_confidence_filtered(self):
        law_good = _make_law("law_a", "reduce_oscillation", confidence=0.8, law_score=0.8)
        law_bad = _make_law("law_b", "stabilize", confidence=0.01, law_score=0.01)
        strat_a = Strategy("law_a", law_good, "adjust_damping", {"alpha": 0.5})
        strat_b = Strategy("law_b", law_bad, "schedule_update", {"mode": "sequential"})
        strategies = {"law_a": strat_a, "law_b": strat_b}
        ids = ["law_a", "law_b"]
        scores = {sid: score_strategy(strategies[sid], {"variance": 0.5}) for sid in ids}
        matrix = build_consensus_matrix(ids, scores, strategies)
        survivors = filter_strategies(ids, scores, matrix, strategies)
        assert "law_b" not in survivors

    def test_empty_input(self):
        survivors = filter_strategies([], {}, {}, {})
        assert survivors == []

    def test_all_filtered_keeps_best(self):
        """If all are filtered, keep the single best."""
        law_a = _make_law("law_a", "reduce_oscillation", confidence=0.05, law_score=0.05)
        law_b = _make_law("law_b", "stabilize", confidence=0.05, law_score=0.04)
        strat_a = Strategy("law_a", law_a, "adjust_damping", {"alpha": 0.5})
        strat_b = Strategy("law_b", law_b, "schedule_update", {"mode": "sequential"})
        strategies = {"law_a": strat_a, "law_b": strat_b}
        ids = ["law_a", "law_b"]
        scores = {sid: score_strategy(strategies[sid], {"variance": 0.5}) for sid in ids}
        matrix = build_consensus_matrix(ids, scores, strategies)
        survivors = filter_strategies(ids, scores, matrix, strategies)
        assert len(survivors) == 1


# ---------------------------------------------------------------------------
# TEST: CONSENSUS SELECTION
# ---------------------------------------------------------------------------


class TestConsensusSelection:
    def test_selects_highest_average(self):
        scores_a = {h: 0.9 for h in HAT_NAMES}
        scores_b = {h: 0.3 for h in HAT_NAMES}
        all_scores = {"law_a": scores_a, "law_b": scores_b}
        # Minimal matrix
        matrix = {
            "law_a": {"law_a": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5},
                       "law_b": {"agreement": 0.0, "conflict": 1.0, "dominance": 1.0}},
            "law_b": {"law_a": {"agreement": 0.0, "conflict": 1.0, "dominance": 0.0},
                       "law_b": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5}},
        }
        winner = select_consensus(["law_a", "law_b"], all_scores, matrix)
        assert winner == "law_a"

    def test_tiebreak_by_agreement(self):
        scores_a = {h: 0.5 for h in HAT_NAMES}
        scores_b = {h: 0.5 for h in HAT_NAMES}
        all_scores = {"law_a": scores_a, "law_b": scores_b}
        # law_a has more agreement
        matrix = {
            "law_a": {"law_a": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5},
                       "law_b": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5}},
            "law_b": {"law_a": {"agreement": 0.0, "conflict": 1.0, "dominance": 0.5},
                       "law_b": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5}},
        }
        winner = select_consensus(["law_a", "law_b"], all_scores, matrix)
        assert winner == "law_a"

    def test_tiebreak_by_lexicographic_id(self):
        scores_a = {h: 0.5 for h in HAT_NAMES}
        scores_b = {h: 0.5 for h in HAT_NAMES}
        all_scores = {"law_a": scores_a, "law_b": scores_b}
        # Same agreement too
        matrix = {
            "law_a": {"law_a": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5},
                       "law_b": {"agreement": 0.5, "conflict": 0.5, "dominance": 0.5}},
            "law_b": {"law_a": {"agreement": 0.5, "conflict": 0.5, "dominance": 0.5},
                       "law_b": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5}},
        }
        winner = select_consensus(["law_a", "law_b"], all_scores, matrix)
        assert winner == "law_a"  # lexicographically first

    def test_single_survivor(self):
        scores = {"law_x": {h: 0.7 for h in HAT_NAMES}}
        matrix = {"law_x": {"law_x": {"agreement": 1.0, "conflict": 0.0, "dominance": 0.5}}}
        winner = select_consensus(["law_x"], scores, matrix)
        assert winner == "law_x"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            select_consensus([], {}, {})


# ---------------------------------------------------------------------------
# TEST: CONSENSUS STRATEGY
# ---------------------------------------------------------------------------


class TestConsensusStrategy:
    def test_apply_adjust_damping(self):
        law = _make_law("law_a", "reduce_oscillation")
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        cs = ConsensusStrategy(strat, {h: 0.8 for h in HAT_NAMES}, ["law_a"])
        state = _make_state([2.0, 4.0, 6.0])
        result = cs.apply(state)
        np.testing.assert_array_almost_equal(result["values"], [1.0, 2.0, 3.0])

    def test_apply_correction_hard(self):
        law = _make_law("law_a", "hard_correction")
        strat = Strategy("law_a", law, "correction_mode", {"mode": "hard"})
        cs = ConsensusStrategy(strat, {}, ["law_a"])
        state = _make_state([-3.0, 0.0, 5.0])
        result = cs.apply(state)
        np.testing.assert_array_equal(result["values"], [-1.0, 0.0, 1.0])

    def test_apply_correction_soft(self):
        law = _make_law("law_a", "soft_correction")
        strat = Strategy("law_a", law, "correction_mode", {"mode": "soft"})
        cs = ConsensusStrategy(strat, {}, ["law_a"])
        state = _make_state([0.0])
        result = cs.apply(state)
        np.testing.assert_array_almost_equal(result["values"], [0.0])

    def test_apply_schedule_sequential(self):
        law = _make_law("law_a", "stabilize")
        strat = Strategy("law_a", law, "schedule_update", {"mode": "sequential"})
        cs = ConsensusStrategy(strat, {}, ["law_a"])
        state = _make_state([1.0, 3.0, 5.0])
        result = cs.apply(state)
        expected = np.cumsum([1.0, 3.0, 5.0]) / np.arange(1, 4)
        np.testing.assert_array_almost_equal(result["values"], expected)

    def test_apply_freeze_nodes(self):
        law = _make_law("law_a", "freeze_unstable")
        strat = Strategy("law_a", law, "freeze_nodes", {"threshold": 0.5})
        cs = ConsensusStrategy(strat, {}, ["law_a"])
        state = _make_state([0.1, 0.9, 0.3])
        result = cs.apply(state)
        np.testing.assert_array_almost_equal(result["values"], [0.0, 0.9, 0.0])

    def test_to_dict(self):
        law = _make_law("law_a", "reduce_oscillation")
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        cs = ConsensusStrategy(strat, {"white": 0.8}, ["law_a", "law_b"])
        d = cs.to_dict()
        assert d["selected_law_id"] == "law_a"
        assert d["action_type"] == "adjust_damping"
        assert d["supporters"] == ["law_a", "law_b"]

    def test_no_mutation_of_input(self):
        law = _make_law("law_a", "reduce_oscillation")
        strat = Strategy("law_a", law, "adjust_damping", {"alpha": 0.5})
        cs = ConsensusStrategy(strat, {}, ["law_a"])
        state = _make_state([2.0, 4.0])
        original = state["values"].copy()
        cs.apply(state)
        np.testing.assert_array_equal(state["values"], original)


# ---------------------------------------------------------------------------
# TEST: EXTRACT METRICS
# ---------------------------------------------------------------------------


class TestExtractMetrics:
    def test_basic_metrics(self):
        state = _make_state([1.0, 2.0, 3.0])
        m = extract_metrics(state)
        assert m["mean"] == pytest.approx(2.0)
        assert m["variance"] == pytest.approx(2.0 / 3.0)
        assert m["delta"] == 0.0


# ---------------------------------------------------------------------------
# TEST: RUN CONSENSUS (INTEGRATION)
# ---------------------------------------------------------------------------


class TestRunConsensus:
    def test_basic_pipeline(self):
        laws = [
            _make_law("law_a", "reduce_oscillation", confidence=0.9, law_score=0.9),
            _make_law("law_b", "stabilize", confidence=0.7, law_score=0.7),
        ]
        state = _make_state([1.0, 2.0, 3.0, 4.0])
        result = run_consensus(laws, state)
        assert result["selected"] is not None
        assert isinstance(result["selected"], ConsensusStrategy)
        assert len(result["all_scores"]) == 2
        assert len(result["reasoning"]) > 0

    def test_no_applicable_laws(self):
        laws = [
            _make_law("law_a", "reduce_oscillation",
                      conditions=[Condition("variance", "gt", 1000.0)]),
        ]
        state = _make_state([1.0, 2.0])
        result = run_consensus(laws, state)
        assert result["selected"] is None

    def test_single_law(self):
        laws = [_make_law("law_a", "reduce_oscillation", confidence=0.8)]
        state = _make_state([1.0, 2.0, 3.0])
        result = run_consensus(laws, state)
        assert result["selected"] is not None
        assert result["selected"].strategy.law_id == "law_a"

    def test_deterministic_repeated_calls(self):
        """Same inputs must produce identical outputs."""
        laws = [
            _make_law("law_a", "reduce_oscillation", confidence=0.9, law_score=0.8),
            _make_law("law_b", "stabilize", confidence=0.7, law_score=0.6),
            _make_law("law_c", "hard_correction", confidence=0.5, law_score=0.4),
        ]
        state = _make_state([1.0, -2.0, 3.0, -4.0, 5.0])
        result1 = run_consensus(laws, state)
        result2 = run_consensus(laws, state)
        assert result1["selected"].strategy.law_id == result2["selected"].strategy.law_id
        assert result1["all_scores"] == result2["all_scores"]

    def test_deterministic_100_runs(self):
        """Verify byte-identical results over 100 runs."""
        laws = [
            _make_law("law_a", "reduce_oscillation", confidence=0.9),
            _make_law("law_b", "stabilize", confidence=0.85),
        ]
        state = _make_state([1.0, 2.0, 3.0])
        baseline = run_consensus(laws, state)
        for _ in range(100):
            result = run_consensus(laws, state)
            assert result["selected"].strategy.law_id == baseline["selected"].strategy.law_id
            for sid in baseline["all_scores"]:
                for hat in HAT_NAMES:
                    assert result["all_scores"][sid][hat] == baseline["all_scores"][sid][hat]

    def test_apply_consensus_result(self):
        laws = [_make_law("law_a", "reduce_oscillation", confidence=0.9)]
        state = _make_state([2.0, 4.0, 6.0])
        result = run_consensus(laws, state)
        new_state = result["selected"].apply(state)
        np.testing.assert_array_almost_equal(new_state["values"], [1.0, 2.0, 3.0])

    def test_reasoning_trace_complete(self):
        laws = [
            _make_law("law_a", "reduce_oscillation"),
            _make_law("law_b", "stabilize"),
        ]
        state = _make_state([1.0, 2.0])
        result = run_consensus(laws, state)
        steps = [r["step"] for r in result["reasoning"]]
        assert "extract_metrics" in steps
        assert "extract_strategies" in steps
        assert "score_strategies" in steps
        assert "build_matrix" in steps
        assert "byzantine_filter" in steps
        assert "select_consensus" in steps

    def test_three_strategies_conflict_resolution(self):
        """Three competing strategies with different action types."""
        laws = [
            _make_law("law_a", "reduce_oscillation", confidence=0.9, law_score=0.9),
            _make_law("law_b", "stabilize", confidence=0.3, law_score=0.3),
            _make_law("law_c", "hard_correction", confidence=0.2, law_score=0.2),
        ]
        state = _make_state([1.0, -1.0, 2.0, -2.0])
        result = run_consensus(laws, state)
        # Highest confidence law should win
        assert result["selected"].strategy.law_id == "law_a"

    def test_consensus_stability_with_varied_states(self):
        """Different states produce valid results each time."""
        laws = [
            _make_law("law_a", "reduce_oscillation", confidence=0.9),
            _make_law("law_b", "stabilize", confidence=0.7),
        ]
        for vals in [[1.0, 3.0], [1.0, 2.0], [5.0, 0.0, 0.0], [100.0, -100.0]]:
            state = _make_state(vals)
            result = run_consensus(laws, state)
            assert result["selected"] is not None
            scores = result["all_scores"]
            for sid in scores:
                for hat in HAT_NAMES:
                    assert 0.0 <= scores[sid][hat] <= 1.0


# ---------------------------------------------------------------------------
# TEST: CLAMP UTILITY
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_zero(self):
        assert _clamp(-1.0) == 0.0

    def test_above_one(self):
        assert _clamp(2.0) == 1.0

    def test_boundary(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0
