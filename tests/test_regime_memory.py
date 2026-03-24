"""Tests for regime-aware memory and stability-weighted evaluation (v99.2.0)."""

from __future__ import annotations

import copy

from qec.analysis.strategy_memory import (
    compute_attractor_id,
    compute_regime_aware_score,
    compute_regime_key,
    compute_stability_weight,
    query_regime_memory,
    score_strategy_with_memory,
    select_strategy_with_memory,
    update_regime_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(step: int, score: float, metrics: dict | None = None) -> dict:
    return {
        "step": step,
        "score": score,
        "metrics": metrics or {},
    }


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
# 1. Regime key determinism
# ---------------------------------------------------------------------------


class TestComputeAttractorId:

    def test_deterministic(self):
        a = compute_attractor_id(0.35)
        b = compute_attractor_id(0.35)
        assert a == b

    def test_bucket_boundaries(self):
        assert compute_attractor_id(0.0) == "basin_0"
        assert compute_attractor_id(0.19) == "basin_0"
        assert compute_attractor_id(0.2) == "basin_1"
        assert compute_attractor_id(0.39) == "basin_1"
        assert compute_attractor_id(0.4) == "basin_2"
        assert compute_attractor_id(0.59) == "basin_2"
        assert compute_attractor_id(0.6) == "basin_3"
        assert compute_attractor_id(0.79) == "basin_3"
        assert compute_attractor_id(0.8) == "basin_4"
        assert compute_attractor_id(1.0) == "basin_4"

    def test_returns_string(self):
        assert isinstance(compute_attractor_id(0.5), str)


class TestComputeRegimeKey:

    def test_deterministic(self):
        a = compute_regime_key("unstable", "basin_2")
        b = compute_regime_key("unstable", "basin_2")
        assert a == b

    def test_hashable(self):
        rk = compute_regime_key("stable", "basin_4")
        d = {rk: "test"}
        assert d[rk] == "test"

    def test_is_tuple(self):
        rk = compute_regime_key("oscillatory", "basin_1")
        assert isinstance(rk, tuple)
        assert len(rk) == 2
        assert rk == ("oscillatory", "basin_1")

    def test_different_regimes_different_keys(self):
        a = compute_regime_key("stable", "basin_2")
        b = compute_regime_key("unstable", "basin_2")
        assert a != b


# ---------------------------------------------------------------------------
# 2. Memory indexing correctness
# ---------------------------------------------------------------------------


class TestUpdateRegimeMemory:

    def test_append_to_empty(self):
        rk = ("unstable", "basin_2")
        mem = update_regime_memory({}, rk, "s1", _make_event(0, 0.5))
        key = (rk, "s1")
        assert key in mem
        assert len(mem[key]) == 1
        assert mem[key][0]["step"] == 0
        assert mem[key][0]["score"] == 0.5

    def test_append_multiple(self):
        rk = ("unstable", "basin_2")
        mem = {}
        mem = update_regime_memory(mem, rk, "s1", _make_event(0, 0.3))
        mem = update_regime_memory(mem, rk, "s1", _make_event(1, 0.5))
        assert len(mem[(rk, "s1")]) == 2

    def test_cap_enforced(self):
        rk = ("unstable", "basin_2")
        mem = {}
        for i in range(15):
            mem = update_regime_memory(
                mem, rk, "s1", _make_event(i, float(i)), cap=10,
            )
        assert len(mem[(rk, "s1")]) == 10
        assert mem[(rk, "s1")][-1]["score"] == 14.0
        assert mem[(rk, "s1")][0]["score"] == 5.0

    def test_no_mutation(self):
        rk = ("unstable", "basin_2")
        original = {(rk, "s1"): [{"step": 0, "score": 0.1, "metrics": {}}]}
        frozen = copy.deepcopy(original)
        update_regime_memory(original, rk, "s1", _make_event(1, 0.5))
        assert original == frozen

    def test_separate_regime_keys(self):
        rk1 = ("unstable", "basin_1")
        rk2 = ("stable", "basin_4")
        mem = {}
        mem = update_regime_memory(mem, rk1, "s1", _make_event(0, 0.3))
        mem = update_regime_memory(mem, rk2, "s1", _make_event(1, 0.7))
        assert len(mem[(rk1, "s1")]) == 1
        assert len(mem[(rk2, "s1")]) == 1

    def test_event_structure(self):
        rk = ("stable", "basin_3")
        mem = update_regime_memory(
            {}, rk, "s1",
            _make_event(5, 0.8, {"basin_score": 0.7}),
        )
        event = mem[(rk, "s1")][0]
        assert event["step"] == 5
        assert event["score"] == 0.8
        assert event["metrics"]["basin_score"] == 0.7


# ---------------------------------------------------------------------------
# 3. Fallback behavior
# ---------------------------------------------------------------------------


class TestQueryRegimeMemory:

    def test_exact_match(self):
        rk = ("unstable", "basin_2")
        mem = {(rk, "s1"): [{"step": 0, "score": 0.5, "metrics": {}}]}
        result = query_regime_memory(mem, rk, "s1")
        assert len(result) == 1
        assert result[0]["score"] == 0.5

    def test_fallback_across_regimes(self):
        rk_old = ("unstable", "basin_1")
        rk_new = ("stable", "basin_4")
        mem = {
            (rk_old, "s1"): [{"step": 0, "score": 0.3, "metrics": {}}],
        }
        # Query with different regime key -> should fallback
        result = query_regime_memory(mem, rk_new, "s1")
        assert len(result) == 1
        assert result[0]["score"] == 0.3

    def test_fallback_aggregates_multiple(self):
        rk1 = ("unstable", "basin_1")
        rk2 = ("oscillatory", "basin_2")
        rk_query = ("stable", "basin_4")
        mem = {
            (rk1, "s1"): [{"step": 0, "score": 0.3, "metrics": {}}],
            (rk2, "s1"): [{"step": 1, "score": 0.7, "metrics": {}}],
        }
        result = query_regime_memory(mem, rk_query, "s1")
        assert len(result) == 2

    def test_empty_when_no_data(self):
        rk = ("unstable", "basin_2")
        result = query_regime_memory({}, rk, "s1")
        assert result == []

    def test_no_cross_strategy_contamination(self):
        rk = ("unstable", "basin_2")
        mem = {
            (rk, "s2"): [{"step": 0, "score": 0.9, "metrics": {}}],
        }
        result = query_regime_memory(mem, rk, "s1")
        assert result == []

    def test_returns_copies(self):
        rk = ("unstable", "basin_2")
        records = [{"step": 0, "score": 0.5, "metrics": {}}]
        mem = {(rk, "s1"): records}
        result = query_regime_memory(mem, rk, "s1")
        assert result is not records


# ---------------------------------------------------------------------------
# 4. Stability weighting correctness
# ---------------------------------------------------------------------------


class TestComputeStabilityWeight:

    def test_empty_returns_one(self):
        assert compute_stability_weight([]) == 1.0

    def test_single_returns_one(self):
        assert compute_stability_weight([0.5]) == 1.0

    def test_identical_scores_returns_one(self):
        assert compute_stability_weight([0.5, 0.5, 0.5]) == 1.0

    def test_high_variance_low_weight(self):
        # Scores with high variance: [-1, 1, -1, 1]
        # variance = 1.0, stability = 1/(1+1) = 0.5
        w = compute_stability_weight([-1.0, 1.0, -1.0, 1.0])
        assert abs(w - 0.5) < 1e-10

    def test_low_variance_high_weight(self):
        w_low = compute_stability_weight([0.5, 0.51, 0.49, 0.5])
        w_high = compute_stability_weight([0.0, 1.0, 0.0, 1.0])
        assert w_low > w_high

    def test_always_positive(self):
        w = compute_stability_weight([-10.0, 10.0, -10.0, 10.0])
        assert w > 0.0

    def test_at_most_one(self):
        w = compute_stability_weight([0.3, 0.3, 0.3])
        assert w <= 1.0

    def test_deterministic(self):
        scores = [0.1, 0.5, -0.2, 0.8]
        a = compute_stability_weight(scores)
        b = compute_stability_weight(scores)
        assert a == b


class TestComputeRegimeAwareScore:

    def test_empty_memory(self):
        rk = ("unstable", "basin_2")
        result = compute_regime_aware_score({}, rk, "s1")
        assert result["mean_score"] == 0.0
        assert result["stability_weight"] == 1.0
        assert result["final_score"] == 0.0
        assert result["n_events"] == 0

    def test_single_event(self):
        rk = ("unstable", "basin_2")
        mem = {(rk, "s1"): [{"step": 0, "score": 0.6, "metrics": {}}]}
        result = compute_regime_aware_score(mem, rk, "s1")
        assert result["mean_score"] == 0.6
        assert result["stability_weight"] == 1.0
        assert result["final_score"] == 0.6
        assert result["n_events"] == 1

    def test_stable_scores_high_weight(self):
        rk = ("stable", "basin_4")
        mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.5, "metrics": {}}
                for i in range(5)
            ],
        }
        result = compute_regime_aware_score(mem, rk, "s1")
        assert result["stability_weight"] == 1.0
        assert result["final_score"] == 0.5

    def test_unstable_scores_penalized(self):
        rk = ("unstable", "basin_1")
        mem = {
            (rk, "s1"): [
                {"step": 0, "score": 1.0, "metrics": {}},
                {"step": 1, "score": -1.0, "metrics": {}},
                {"step": 2, "score": 1.0, "metrics": {}},
                {"step": 3, "score": -1.0, "metrics": {}},
            ],
        }
        result = compute_regime_aware_score(mem, rk, "s1")
        assert result["mean_score"] == 0.0
        assert result["stability_weight"] < 1.0

    def test_deterministic(self):
        rk = ("unstable", "basin_2")
        mem = {
            (rk, "s1"): [
                {"step": 0, "score": 0.3, "metrics": {}},
                {"step": 1, "score": 0.7, "metrics": {}},
            ],
        }
        a = compute_regime_aware_score(mem, rk, "s1")
        b = compute_regime_aware_score(mem, rk, "s1")
        assert a == b

    def test_uses_fallback(self):
        rk_old = ("unstable", "basin_1")
        rk_new = ("stable", "basin_4")
        mem = {
            (rk_old, "s1"): [{"step": 0, "score": 0.8, "metrics": {}}],
        }
        result = compute_regime_aware_score(mem, rk_new, "s1")
        assert result["n_events"] == 1
        assert result["mean_score"] == 0.8


# ---------------------------------------------------------------------------
# 5. Deterministic outputs across runs
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_regime_key_determinism_repeated(self):
        """Same inputs produce same regime keys across 100 calls."""
        for _ in range(100):
            aid = compute_attractor_id(0.55)
            rk = compute_regime_key("unstable", aid)
            assert aid == "basin_2"
            assert rk == ("unstable", "basin_2")

    def test_memory_update_determinism(self):
        rk = ("oscillatory", "basin_2")
        mem = {}
        for i in range(5):
            mem = update_regime_memory(mem, rk, "s1", _make_event(i, 0.1 * i))
        mem2 = {}
        for i in range(5):
            mem2 = update_regime_memory(mem2, rk, "s1", _make_event(i, 0.1 * i))
        assert mem == mem2

    def test_query_determinism(self):
        rk1 = ("unstable", "basin_1")
        rk2 = ("oscillatory", "basin_2")
        rk_query = ("stable", "basin_4")
        mem = {
            (rk1, "s1"): [{"step": 0, "score": 0.3, "metrics": {}}],
            (rk2, "s1"): [{"step": 1, "score": 0.7, "metrics": {}}],
        }
        a = query_regime_memory(mem, rk_query, "s1")
        b = query_regime_memory(mem, rk_query, "s1")
        assert a == b

    def test_stability_weight_determinism(self):
        scores = [0.1, -0.3, 0.5, 0.2, -0.1]
        a = compute_stability_weight(scores)
        b = compute_stability_weight(scores)
        assert a == b

    def test_full_pipeline_determinism(self):
        """Full regime-aware pipeline produces identical outputs."""
        rk = compute_regime_key("unstable", compute_attractor_id(0.35))
        mem = {}
        for i in range(5):
            mem = update_regime_memory(
                mem, rk, "s1", _make_event(i, 0.1 * (i + 1)),
            )
        score_a = compute_regime_aware_score(mem, rk, "s1")

        mem2 = {}
        for i in range(5):
            mem2 = update_regime_memory(
                mem2, rk, "s1", _make_event(i, 0.1 * (i + 1)),
            )
        score_b = compute_regime_aware_score(mem2, rk, "s1")

        assert score_a == score_b

    def test_no_input_mutation(self):
        rk = ("unstable", "basin_2")
        mem = {(rk, "s1"): [{"step": 0, "score": 0.5, "metrics": {}}]}
        mem_copy = copy.deepcopy(mem)

        update_regime_memory(mem, rk, "s1", _make_event(1, 0.8))
        query_regime_memory(mem, rk, "s1")
        compute_regime_aware_score(mem, rk, "s1")

        assert mem == mem_copy


# ---------------------------------------------------------------------------
# 6. Integration: regime-aware scoring in strategy selection
# ---------------------------------------------------------------------------


class TestRegimeAwareSelection:

    def test_score_with_regime_key(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        rk = ("unstable", "basin_2")
        mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.8, "metrics": {}}
                for i in range(5)
            ],
        }
        result = score_strategy_with_memory(
            strategy, state, [], mem, "s1", regime_key=rk,
        )
        assert "score" in result
        assert "global_bias" in result
        assert "local_bias" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_score_without_regime_key_backward_compat(self):
        strategy = {"action_type": "damping", "params": {"alpha": 0.1}}
        state = _make_state()
        # Flat memory (old format)
        mem = {"s1": [{"score": 0.8, "outcome": "improved"}] * 3}
        result = score_strategy_with_memory(
            strategy, state, [], mem, "s1",
        )
        assert "score" in result
        assert result["local_bias"] != 0.0

    def test_select_with_regime_key(self):
        strategies = _make_strategies()
        state = _make_state()
        rk = ("unstable", "basin_2")
        mem = {
            (rk, "s1"): [
                {"step": i, "score": 0.9, "metrics": {}}
                for i in range(5)
            ],
        }
        result = select_strategy_with_memory(
            strategies, state, [], mem, regime_key=rk,
        )
        assert result["selected"] in strategies
        assert "score" in result

    def test_select_without_regime_key_backward_compat(self):
        strategies = _make_strategies()
        state = _make_state()
        result = select_strategy_with_memory(
            strategies, state, [], {},
        )
        assert result["selected"] in strategies

    def test_transition_integration_with_regime_memory(self):
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
        memory = {
            (rk, "s1"): [
                {"step": i, "score": 0.8, "metrics": {}}
                for i in range(3)
            ],
        }

        result = select_next_strategy(
            metrics, strategies, history=history, memory=memory,
        )
        assert "strategy" in result
        assert "state" in result
        assert result["adaptation"] is not None
