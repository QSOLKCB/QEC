"""Tests for deterministic transition learning (v99.3.0)."""

from __future__ import annotations

import copy

from qec.analysis.strategy_transition_learning import (
    compute_transition_bias,
    compute_transition_key,
    record_transition_outcome,
    update_transition_memory,
)
from qec.analysis.strategy_memory import (
    compute_attractor_id,
    compute_regime_key,
    score_strategy_with_memory,
    select_strategy_with_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# 1. Transition key determinism
# ---------------------------------------------------------------------------


class TestComputeTransitionKey:

    def test_returns_5_tuple(self):
        key = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        assert isinstance(key, tuple)
        assert len(key) == 5

    def test_deterministic(self):
        a = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        b = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        assert a == b

    def test_hashable(self):
        key = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        d = {key: "test"}
        assert d[key] == "test"

    def test_different_inputs_different_keys(self):
        k1 = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        k2 = compute_transition_key("stable", "basin_2", "s1", "stable", "basin_4")
        assert k1 != k2

    def test_strategy_differentiates(self):
        k1 = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        k2 = compute_transition_key("unstable", "basin_2", "s2", "stable", "basin_4")
        assert k1 != k2

    def test_after_state_differentiates(self):
        k1 = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        k2 = compute_transition_key("unstable", "basin_2", "s1", "oscillatory", "basin_4")
        assert k1 != k2

    def test_coerces_to_string(self):
        key = compute_transition_key("unstable", "basin_2", "s1", "stable", "basin_4")
        assert all(isinstance(k, str) for k in key)

    def test_repeated_100_calls(self):
        """Same inputs produce identical keys across 100 calls."""
        expected = compute_transition_key(
            "unstable", "basin_1", "s1", "stable", "basin_4",
        )
        for _ in range(100):
            result = compute_transition_key(
                "unstable", "basin_1", "s1", "stable", "basin_4",
            )
            assert result == expected


# ---------------------------------------------------------------------------
# 2. Transition memory correctness
# ---------------------------------------------------------------------------


class TestUpdateTransitionMemory:

    def test_first_entry_success(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = update_transition_memory({}, key, 0.5)
        assert key in mem
        assert mem[key]["count"] == 1
        assert mem[key]["mean_delta"] == 0.5
        assert mem[key]["success_rate"] == 1.0  # 0.5 > 0

    def test_first_entry_failure(self):
        key = ("unstable", "basin_2", "s1", "unstable", "basin_1")
        mem = update_transition_memory({}, key, -0.3)
        assert mem[key]["count"] == 1
        assert mem[key]["mean_delta"] == -0.3
        assert mem[key]["success_rate"] == 0.0  # -0.3 <= 0

    def test_first_entry_zero_is_failure(self):
        """improvement_score == 0 is NOT a success (must be > 0)."""
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = update_transition_memory({}, key, 0.0)
        assert mem[key]["success_rate"] == 0.0

    def test_incremental_mean_delta(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = update_transition_memory({}, key, 0.6)
        mem = update_transition_memory(mem, key, 0.4)
        assert mem[key]["count"] == 2
        assert abs(mem[key]["mean_delta"] - 0.5) < 1e-10

    def test_incremental_success_rate(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = update_transition_memory({}, key, 0.5)   # success
        mem = update_transition_memory(mem, key, -0.3)  # failure
        mem = update_transition_memory(mem, key, 0.2)   # success
        assert mem[key]["count"] == 3
        assert abs(mem[key]["success_rate"] - 2.0 / 3.0) < 1e-10

    def test_no_mutation(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        original = {key: {"count": 1, "mean_delta": 0.5, "success_rate": 1.0}}
        frozen = copy.deepcopy(original)
        update_transition_memory(original, key, 0.3)
        assert original == frozen

    def test_returns_new_dict(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {}
        new_mem = update_transition_memory(mem, key, 0.5)
        assert new_mem is not mem

    def test_separate_keys(self):
        k1 = ("unstable", "basin_2", "s1", "stable", "basin_4")
        k2 = ("unstable", "basin_2", "s2", "stable", "basin_4")
        mem = update_transition_memory({}, k1, 0.5)
        mem = update_transition_memory(mem, k2, -0.3)
        assert mem[k1]["success_rate"] == 1.0
        assert mem[k2]["success_rate"] == 0.0


# ---------------------------------------------------------------------------
# 3. Mean delta and success rate correctness
# ---------------------------------------------------------------------------


class TestMeanDeltaSuccessRate:

    def test_mean_delta_five_entries(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        scores = [0.1, 0.3, -0.2, 0.5, -0.1]
        mem = {}
        for s in scores:
            mem = update_transition_memory(mem, key, s)
        expected_mean = sum(scores) / len(scores)
        assert abs(mem[key]["mean_delta"] - expected_mean) < 1e-10

    def test_success_rate_five_entries(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        scores = [0.1, 0.3, -0.2, 0.5, -0.1]
        mem = {}
        for s in scores:
            mem = update_transition_memory(mem, key, s)
        # Successes: 0.1, 0.3, 0.5 (3 out of 5)
        assert abs(mem[key]["success_rate"] - 3.0 / 5.0) < 1e-10

    def test_all_successes(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {}
        for _ in range(10):
            mem = update_transition_memory(mem, key, 0.5)
        assert mem[key]["success_rate"] == 1.0

    def test_all_failures(self):
        key = ("unstable", "basin_2", "s1", "unstable", "basin_0")
        mem = {}
        for _ in range(10):
            mem = update_transition_memory(mem, key, -0.5)
        assert mem[key]["success_rate"] == 0.0
        assert mem[key]["mean_delta"] == -0.5

    def test_mean_delta_single_large(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = update_transition_memory({}, key, 0.99)
        assert abs(mem[key]["mean_delta"] - 0.99) < 1e-10


# ---------------------------------------------------------------------------
# 4. Neutral fallback
# ---------------------------------------------------------------------------


class TestComputeTransitionBias:

    def test_no_history_returns_neutral(self):
        bias = compute_transition_bias({}, "unstable", "basin_2", "s1")
        assert bias == 1.0

    def test_no_matching_strategy_returns_neutral(self):
        key = ("unstable", "basin_2", "s2", "stable", "basin_4")
        mem = {key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.8}}
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        assert bias == 1.0

    def test_no_matching_regime_returns_neutral(self):
        key = ("stable", "basin_4", "s1", "stable", "basin_4")
        mem = {key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.8}}
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        assert bias == 1.0

    def test_all_success_boosts(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {key: {"count": 10, "mean_delta": 0.5, "success_rate": 1.0}}
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        # 0.8 + 0.4 * 1.0 = 1.2
        assert abs(bias - 1.2) < 1e-10

    def test_all_failure_penalizes(self):
        key = ("unstable", "basin_2", "s1", "unstable", "basin_0")
        mem = {key: {"count": 10, "mean_delta": -0.5, "success_rate": 0.0}}
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        # 0.8 + 0.4 * 0.0 = 0.8
        assert abs(bias - 0.8) < 1e-10

    def test_half_success_neutral(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {key: {"count": 10, "mean_delta": 0.0, "success_rate": 0.5}}
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        # 0.8 + 0.4 * 0.5 = 1.0
        assert abs(bias - 1.0) < 1e-10

    def test_aggregates_across_destinations(self):
        """Bias aggregates all transitions from same source regime."""
        k1 = ("unstable", "basin_2", "s1", "stable", "basin_4")
        k2 = ("unstable", "basin_2", "s1", "transitional", "basin_3")
        mem = {
            k1: {"count": 6, "mean_delta": 0.5, "success_rate": 1.0},
            k2: {"count": 4, "mean_delta": -0.2, "success_rate": 0.0},
        }
        bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        # Weighted: (1.0*6 + 0.0*4) / 10 = 0.6
        # Bias: 0.8 + 0.4 * 0.6 = 1.04
        assert abs(bias - 1.04) < 1e-10

    def test_bias_bounds(self):
        """Transition bias is always in [0.8, 1.2] (within float tolerance)."""
        for sr in [0.0, 0.25, 0.5, 0.75, 1.0]:
            key = ("unstable", "basin_2", "s1", "stable", "basin_4")
            mem = {key: {"count": 10, "mean_delta": 0.0, "success_rate": sr}}
            bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
            assert 0.8 - 1e-10 <= bias <= 1.2 + 1e-10

    def test_deterministic(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.6}}
        a = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        b = compute_transition_bias(mem, "unstable", "basin_2", "s1")
        assert a == b


# ---------------------------------------------------------------------------
# 5. Record transition outcome (convenience wrapper)
# ---------------------------------------------------------------------------


class TestRecordTransitionOutcome:

    def test_records_correctly(self):
        mem = record_transition_outcome(
            {}, "unstable", "basin_2", "s1", "stable", "basin_4", 0.5,
        )
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        assert key in mem
        assert mem[key]["count"] == 1
        assert mem[key]["mean_delta"] == 0.5

    def test_no_mutation(self):
        original = {}
        frozen = copy.deepcopy(original)
        record_transition_outcome(
            original, "unstable", "basin_2", "s1", "stable", "basin_4", 0.5,
        )
        assert original == frozen


# ---------------------------------------------------------------------------
# 6. Deterministic selection changes with transition bias
# ---------------------------------------------------------------------------


class TestTransitionBiasInScoring:

    def test_score_includes_transition_bias_key(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=rk, transition_memory={},
        )
        assert "transition_bias" in result

    def test_no_transition_memory_bias_is_neutral(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=rk, transition_memory={},
        )
        assert result["transition_bias"] == 1.0

    def test_without_transition_memory_param_bias_is_neutral(self):
        """When transition_memory is not provided, transition_bias = 1.0."""
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
        )
        assert result["transition_bias"] == 1.0

    def test_good_transition_history_boosts_score(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        regime_mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.5, "metrics": {}}
                for i in range(5)
            ],
        }
        trans_key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        trans_mem = {trans_key: {"count": 10, "mean_delta": 0.5, "success_rate": 1.0}}

        with_trans = score_strategy_with_memory(
            strategy, state, [], regime_mem, "s1",
            regime_key=rk, transition_memory=trans_mem,
        )
        without_trans = score_strategy_with_memory(
            strategy, state, [], regime_mem, "s1",
            regime_key=rk, transition_memory={},
        )
        # Good history (bias=1.2) should boost score
        assert with_trans["score"] >= without_trans["score"]
        assert with_trans["transition_bias"] > 1.0

    def test_bad_transition_history_reduces_score(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        regime_mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.5, "metrics": {}}
                for i in range(5)
            ],
        }
        trans_key = ("unstable", "basin_2", "s1", "unstable", "basin_0")
        trans_mem = {trans_key: {"count": 10, "mean_delta": -0.5, "success_rate": 0.0}}

        with_trans = score_strategy_with_memory(
            strategy, state, [], regime_mem, "s1",
            regime_key=rk, transition_memory=trans_mem,
        )
        without_trans = score_strategy_with_memory(
            strategy, state, [], regime_mem, "s1",
            regime_key=rk, transition_memory={},
        )
        # Bad history (bias=0.8) should reduce score
        assert with_trans["score"] <= without_trans["score"]
        assert with_trans["transition_bias"] < 1.0

    def test_score_clamped_0_1(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        trans_key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        trans_mem = {trans_key: {"count": 10, "mean_delta": 0.9, "success_rate": 1.0}}
        result = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=rk, transition_memory=trans_mem,
        )
        assert 0.0 <= result["score"] <= 1.0


class TestTransitionBiasInSelection:

    def test_select_includes_transition_bias(self):
        strategies = _make_strategies()
        state = _make_state()
        rk = ("unstable", "basin_2")
        result = select_strategy_with_memory(
            strategies, state, [], {}, regime_key=rk, transition_memory={},
        )
        assert "transition_bias" in result

    def test_selection_shifts_with_transition_history(self):
        """Transition history can change which strategy is selected."""
        strategies = _make_strategies()
        state = _make_state()
        rk = ("unstable", "basin_2")

        # s1 has bad transition history, s2 has good transition history
        trans_mem = {
            ("unstable", "basin_2", "s1", "unstable", "basin_0"): {
                "count": 10, "mean_delta": -0.5, "success_rate": 0.0,
            },
            ("unstable", "basin_2", "s2", "stable", "basin_4"): {
                "count": 10, "mean_delta": 0.5, "success_rate": 1.0,
            },
        }

        # Without transition: s1 (damping) scores highest in unstable
        without = select_strategy_with_memory(
            strategies, state, [], {}, regime_key=rk, transition_memory={},
        )

        # With transition: s1 penalized, s2 boosted
        with_trans = select_strategy_with_memory(
            strategies, state, [], {}, regime_key=rk, transition_memory=trans_mem,
        )

        # The selection may or may not change depending on base scores,
        # but scores should differ
        if without["selected"] == "s1" and with_trans["selected"] != "s1":
            # Transition learning changed the outcome
            pass
        # At minimum, verify the mechanism is active
        assert with_trans["transition_bias"] != 1.0 or with_trans["selected"] != "s1"

    def test_empty_strategies(self):
        result = select_strategy_with_memory(
            {}, _make_state(), [], {}, transition_memory={},
        )
        assert result["selected"] == ""
        assert result["transition_bias"] == 1.0


# ---------------------------------------------------------------------------
# 7. Repeated-run determinism
# ---------------------------------------------------------------------------


class TestTransitionLearningDeterminism:

    def test_transition_key_determinism(self):
        for _ in range(100):
            key = compute_transition_key(
                "unstable", "basin_2", "s1", "stable", "basin_4",
            )
            assert key == ("unstable", "basin_2", "s1", "stable", "basin_4")

    def test_memory_update_determinism(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        scores = [0.3, -0.1, 0.5, 0.2, -0.4]

        mem1 = {}
        for s in scores:
            mem1 = update_transition_memory(mem1, key, s)

        mem2 = {}
        for s in scores:
            mem2 = update_transition_memory(mem2, key, s)

        assert mem1 == mem2

    def test_bias_determinism(self):
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.6}}
        results = [
            compute_transition_bias(mem, "unstable", "basin_2", "s1")
            for _ in range(100)
        ]
        assert all(r == results[0] for r in results)

    def test_full_pipeline_determinism(self):
        """Full transition learning pipeline produces identical outputs."""
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        scores = [0.5, -0.2, 0.3, 0.1, -0.1]

        def run_pipeline():
            mem = {}
            for s in scores:
                mem = update_transition_memory(mem, key, s)
            bias = compute_transition_bias(mem, "unstable", "basin_2", "s1")
            return mem, bias

        mem_a, bias_a = run_pipeline()
        mem_b, bias_b = run_pipeline()
        assert mem_a == mem_b
        assert bias_a == bias_b

    def test_scoring_determinism_with_transition(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        trans_key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        trans_mem = {trans_key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.8}}

        a = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=rk, transition_memory=trans_mem,
        )
        b = score_strategy_with_memory(
            strategy, state, [], {}, "s1",
            regime_key=rk, transition_memory=trans_mem,
        )
        assert a == b

    def test_selection_determinism_with_transition(self):
        strategies = _make_strategies()
        state = _make_state()
        rk = ("unstable", "basin_2")
        trans_key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        trans_mem = {trans_key: {"count": 5, "mean_delta": 0.3, "success_rate": 0.8}}

        a = select_strategy_with_memory(
            strategies, state, [], {}, regime_key=rk, transition_memory=trans_mem,
        )
        b = select_strategy_with_memory(
            strategies, state, [], {}, regime_key=rk, transition_memory=trans_mem,
        )
        assert a == b

    def test_no_input_mutation(self):
        """All transition learning operations must not mutate inputs."""
        key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        mem = {key: {"count": 3, "mean_delta": 0.2, "success_rate": 0.67}}
        mem_copy = copy.deepcopy(mem)

        update_transition_memory(mem, key, 0.5)
        compute_transition_bias(mem, "unstable", "basin_2", "s1")

        assert mem == mem_copy


# ---------------------------------------------------------------------------
# 8. Integration with select_next_strategy
# ---------------------------------------------------------------------------


class TestTransitionLearningIntegration:

    def test_select_next_strategy_with_transition_memory(self):
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
                "signals": {},
            },
        }
        strategies = _make_strategies()
        history = [{"score": 0.3, "direction": "improved"}]
        rk = compute_regime_key("unstable", compute_attractor_id(0.4))
        regime_mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.8, "metrics": {}}
                for i in range(3)
            ],
        }
        trans_key = ("unstable", "basin_2", "s1", "stable", "basin_4")
        trans_mem = {trans_key: {"count": 5, "mean_delta": 0.4, "success_rate": 0.8}}

        result = select_next_strategy(
            metrics, strategies,
            history=history, memory=regime_mem,
            transition_memory=trans_mem,
        )
        assert "strategy" in result
        assert "state" in result
        assert result["adaptation"] is not None
        assert "transition_bias" in result["adaptation"]

    def test_select_next_strategy_without_transition_memory_unchanged(self):
        """Existing behavior preserved when transition_memory is not provided."""
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
                "signals": {},
            },
        }
        strategies = _make_strategies()
        history = [{"score": 0.3, "direction": "improved"}]
        rk = compute_regime_key("unstable", compute_attractor_id(0.4))
        regime_mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.8, "metrics": {}}
                for i in range(3)
            ],
        }

        result = select_next_strategy(
            metrics, strategies,
            history=history, memory=regime_mem,
        )
        assert "strategy" in result
        assert result["adaptation"] is not None
        assert "local_bias" in result["adaptation"]
