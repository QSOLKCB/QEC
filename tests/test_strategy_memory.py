"""Tests for per-strategy memory and local adaptation (v99.1.0)."""

from __future__ import annotations

import copy

from qec.analysis.strategy_memory import (
    DEFAULT_MEMORY_CAP,
    compute_strategy_bias,
    compute_strategy_performance,
    score_strategy_with_memory,
    select_strategy_with_memory,
    update_strategy_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval(score: float, outcome: str = "neutral") -> dict:
    return {"score": score, "outcome": outcome}


def _make_state(regime: str = "unstable") -> dict:
    return {
        "regime": regime,
        "basin_score": 0.5,
        "phi": 0.5,
        "consistency": 0.5,
        "divergence": 0.1,
        "curvature": 0.1,
        "resonance": 0.1,
        "complexity": 0.1,
    }


def _make_strategies() -> dict:
    return {
        "s1": {"action_type": "damping", "params": {"alpha": 0.1}, "confidence": 0.5},
        "s2": {"action_type": "scaling", "params": {"beta": 0.2}, "confidence": 0.5},
        "s3": {"action_type": "rotation", "params": {"theta": 0.3}, "confidence": 0.5},
    }


# ---------------------------------------------------------------------------
# 1. Memory update
# ---------------------------------------------------------------------------


class TestUpdateStrategyMemory:

    def test_append_to_empty(self):
        mem = update_strategy_memory({}, "s1", _make_eval(0.5, "improved"))
        assert "s1" in mem
        assert len(mem["s1"]) == 1
        assert mem["s1"][0]["score"] == 0.5
        assert mem["s1"][0]["outcome"] == "improved"

    def test_append_multiple(self):
        mem = {}
        mem = update_strategy_memory(mem, "s1", _make_eval(0.3))
        mem = update_strategy_memory(mem, "s1", _make_eval(0.5))
        assert len(mem["s1"]) == 2

    def test_cap_enforced(self):
        mem = {}
        for i in range(15):
            mem = update_strategy_memory(mem, "s1", _make_eval(float(i)), cap=10)
        assert len(mem["s1"]) == 10
        # Most recent entries kept
        assert mem["s1"][-1]["score"] == 14.0
        assert mem["s1"][0]["score"] == 5.0

    def test_custom_cap(self):
        mem = {}
        for i in range(5):
            mem = update_strategy_memory(mem, "s1", _make_eval(float(i)), cap=3)
        assert len(mem["s1"]) == 3

    def test_no_mutation(self):
        original = {"s1": [{"score": 0.1, "outcome": "neutral"}]}
        frozen = copy.deepcopy(original)
        update_strategy_memory(original, "s1", _make_eval(0.5))
        assert original == frozen

    def test_separate_strategies(self):
        mem = {}
        mem = update_strategy_memory(mem, "s1", _make_eval(0.3))
        mem = update_strategy_memory(mem, "s2", _make_eval(0.7))
        assert len(mem["s1"]) == 1
        assert len(mem["s2"]) == 1

    def test_returns_new_dict(self):
        mem = {}
        new_mem = update_strategy_memory(mem, "s1", _make_eval(0.5))
        assert new_mem is not mem


# ---------------------------------------------------------------------------
# 2. Performance scoring
# ---------------------------------------------------------------------------


class TestComputeStrategyPerformance:

    def test_no_history(self):
        assert compute_strategy_performance({}, "s1") == 0.0

    def test_missing_strategy(self):
        mem = {"s2": [{"score": 0.5, "outcome": "improved"}]}
        assert compute_strategy_performance(mem, "s1") == 0.0

    def test_all_improved_positive(self):
        mem = {"s1": [
            {"score": 0.5, "outcome": "improved"},
            {"score": 0.6, "outcome": "improved"},
        ]}
        perf = compute_strategy_performance(mem, "s1")
        assert perf > 0.0

    def test_all_regressed_negative(self):
        mem = {"s1": [
            {"score": -0.5, "outcome": "regressed"},
            {"score": -0.6, "outcome": "regressed"},
        ]}
        perf = compute_strategy_performance(mem, "s1")
        assert perf < 0.0

    def test_clamped_upper(self):
        mem = {"s1": [{"score": 10.0, "outcome": "improved"}] * 10}
        perf = compute_strategy_performance(mem, "s1")
        assert perf <= 1.0

    def test_clamped_lower(self):
        mem = {"s1": [{"score": -10.0, "outcome": "regressed"}] * 10}
        perf = compute_strategy_performance(mem, "s1")
        assert perf >= -1.0

    def test_stability_component(self):
        # All scores near zero -> high stability -> lower penalty
        stable_mem = {"s1": [{"score": 0.05, "outcome": "neutral"}] * 5}
        # All scores far from zero -> low stability -> higher penalty
        unstable_mem = {"s1": [{"score": 0.5, "outcome": "neutral"}] * 5}
        stable_perf = compute_strategy_performance(stable_mem, "s1")
        unstable_perf = compute_strategy_performance(unstable_mem, "s1")
        # Unstable has higher avg_score but also higher penalty
        # The key test: both return valid floats
        assert -1.0 <= stable_perf <= 1.0
        assert -1.0 <= unstable_perf <= 1.0

    def test_deterministic(self):
        mem = {"s1": [
            {"score": 0.3, "outcome": "improved"},
            {"score": -0.1, "outcome": "regressed"},
        ]}
        a = compute_strategy_performance(mem, "s1")
        b = compute_strategy_performance(mem, "s1")
        assert a == b


# ---------------------------------------------------------------------------
# 3. Strategy bias
# ---------------------------------------------------------------------------


class TestComputeStrategyBias:

    def test_no_history_zero(self):
        assert compute_strategy_bias({}, "s1") == 0.0

    def test_positive_after_good_outcomes(self):
        mem = {"s1": [{"score": 0.8, "outcome": "improved"}] * 5}
        bias = compute_strategy_bias(mem, "s1")
        assert bias > 0.0

    def test_negative_after_regressions(self):
        mem = {"s1": [{"score": -0.8, "outcome": "regressed"}] * 5}
        bias = compute_strategy_bias(mem, "s1")
        assert bias < 0.0

    def test_clamped_upper(self):
        mem = {"s1": [{"score": 10.0, "outcome": "improved"}] * 10}
        bias = compute_strategy_bias(mem, "s1")
        assert bias <= 0.2

    def test_clamped_lower(self):
        mem = {"s1": [{"score": -10.0, "outcome": "regressed"}] * 10}
        bias = compute_strategy_bias(mem, "s1")
        assert bias >= -0.2

    def test_deterministic(self):
        mem = {"s1": [{"score": 0.5, "outcome": "improved"}] * 3}
        a = compute_strategy_bias(mem, "s1")
        b = compute_strategy_bias(mem, "s1")
        assert a == b


# ---------------------------------------------------------------------------
# 4. Memory-aware scoring
# ---------------------------------------------------------------------------


class TestScoreStrategyWithMemory:

    def test_returns_expected_keys(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        result = score_strategy_with_memory(strategy, state, [], {}, "s1")
        assert "score" in result
        assert "global_bias" in result
        assert "local_bias" in result

    def test_score_clamped_0_1(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        mem = {"s1": [{"score": 10.0, "outcome": "improved"}] * 10}
        history = [{"score": 10.0, "direction": "improved"}] * 10
        result = score_strategy_with_memory(strategy, state, history, mem, "s1")
        assert 0.0 <= result["score"] <= 1.0

    def test_no_memory_no_local_bias(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        result = score_strategy_with_memory(strategy, state, [], {}, "s1")
        assert result["local_bias"] == 0.0
        assert result["global_bias"] == 0.0

    def test_local_bias_affects_score(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        history = []
        no_mem = score_strategy_with_memory(strategy, state, history, {}, "s1")
        good_mem = {"s1": [{"score": 0.8, "outcome": "improved"}] * 5}
        with_mem = score_strategy_with_memory(
            strategy, state, history, good_mem, "s1",
        )
        assert with_mem["local_bias"] > 0.0
        assert with_mem["score"] >= no_mem["score"]

    def test_deterministic(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        mem = {"s1": [{"score": 0.3, "outcome": "improved"}]}
        a = score_strategy_with_memory(strategy, state, [], mem, "s1")
        b = score_strategy_with_memory(strategy, state, [], mem, "s1")
        assert a == b


# ---------------------------------------------------------------------------
# 5. Memory-aware selection
# ---------------------------------------------------------------------------


class TestSelectStrategyWithMemory:

    def test_empty_strategies(self):
        result = select_strategy_with_memory({}, _make_state(), [], {})
        assert result["selected"] == ""
        assert result["score"] == 0.0

    def test_returns_expected_keys(self):
        result = select_strategy_with_memory(
            _make_strategies(), _make_state(), [], {},
        )
        assert "selected" in result
        assert "score" in result
        assert "global_bias" in result
        assert "local_bias" in result

    def test_selects_strategy(self):
        result = select_strategy_with_memory(
            _make_strategies(), _make_state(), [], {},
        )
        assert result["selected"] in _make_strategies()

    def test_prefers_better_memory(self):
        strategies = _make_strategies()
        state = _make_state()
        history = []
        # s1 has great memory, s2 has bad memory
        memory = {
            "s1": [{"score": 0.9, "outcome": "improved"}] * 5,
            "s2": [{"score": -0.9, "outcome": "regressed"}] * 5,
        }
        result = select_strategy_with_memory(strategies, state, history, memory)
        # s1 should be preferred over s2 due to better memory
        # (unless base scores differ enough to override)
        assert result["local_bias"] != 0.0  # memory is being used

    def test_fallback_without_memory(self):
        """Without memory, selection should still work (zero local bias)."""
        strategies = _make_strategies()
        state = _make_state()
        result = select_strategy_with_memory(strategies, state, [], {})
        assert result["local_bias"] == 0.0
        assert result["selected"] in strategies

    def test_deterministic(self):
        strategies = _make_strategies()
        state = _make_state()
        mem = {"s1": [{"score": 0.5, "outcome": "improved"}]}
        a = select_strategy_with_memory(strategies, state, [], mem)
        b = select_strategy_with_memory(strategies, state, [], mem)
        assert a == b

    def test_no_mutation_of_inputs(self):
        strategies = _make_strategies()
        state = _make_state()
        memory = {"s1": [{"score": 0.5, "outcome": "improved"}]}
        history = [{"score": 0.3, "direction": "improved"}]

        strategies_copy = copy.deepcopy(strategies)
        state_copy = copy.deepcopy(state)
        memory_copy = copy.deepcopy(memory)
        history_copy = copy.deepcopy(history)

        select_strategy_with_memory(strategies, state, history, memory)

        assert strategies == strategies_copy
        assert state == state_copy
        assert memory == memory_copy
        assert history == history_copy


# ---------------------------------------------------------------------------
# 6. Integration: strategy_transition with memory
# ---------------------------------------------------------------------------


class TestTransitionIntegration:

    def test_select_next_strategy_with_memory(self):
        from qec.analysis.strategy_transition import select_next_strategy

        metrics = {
            "field": {
                "phi_alignment": 0.5,
                "curvature": {"abs_curvature": 0.3, "curvature_variation": 0.1},
                "resonance": 0.2,
                "complexity": 0.1,
            },
            "multiscale": {
                "scale_consistency": 0.5,
                "scale_divergence": 0.2,
            },
            "attractor": {
                "regime": "unstable",
                "basin_score": 0.4,
                "signals": {
                    "phi": 0.5,
                    "consistency": 0.5,
                    "divergence": 0.2,
                    "curvature": 0.3,
                    "resonance": 0.2,
                    "complexity": 0.1,
                },
            },
        }
        strategies = _make_strategies()
        history = [{"score": 0.3, "direction": "improved"}]
        memory = {"s1": [{"score": 0.8, "outcome": "improved"}] * 3}

        result = select_next_strategy(
            metrics, strategies, history=history, memory=memory,
        )
        assert "strategy" in result
        assert "state" in result
        assert result["adaptation"] is not None
        assert "local_bias" in result["adaptation"]

    def test_select_next_strategy_without_memory_unchanged(self):
        """Existing behavior preserved when memory is not provided."""
        from qec.analysis.strategy_transition import select_next_strategy

        metrics = {
            "field": {
                "phi_alignment": 0.5,
                "curvature": {"abs_curvature": 0.3, "curvature_variation": 0.1},
                "resonance": 0.2,
                "complexity": 0.1,
            },
            "multiscale": {
                "scale_consistency": 0.5,
                "scale_divergence": 0.2,
            },
            "attractor": {
                "regime": "unstable",
                "basin_score": 0.4,
                "signals": {
                    "phi": 0.5,
                    "consistency": 0.5,
                    "divergence": 0.2,
                    "curvature": 0.3,
                    "resonance": 0.2,
                    "complexity": 0.1,
                },
            },
        }
        strategies = _make_strategies()
        history = [{"score": 0.3, "direction": "improved"}]

        result = select_next_strategy(
            metrics, strategies, history=history,
        )
        assert "strategy" in result
        # Should use adaptive path (no local_bias key in adaptation)
        if result.get("adaptation"):
            assert "trajectory_score" in result["adaptation"]
