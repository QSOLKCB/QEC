"""Tests for the quaternary bosonic decoder pipeline.

Covers: quantization, message passing, pipeline, determinism,
no mutation, bounded outputs, dual-system generation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from qec.analysis.quaternary_quantization import (
    QUATERNARY_STATES,
    quantize_quaternary,
    quaternary_stats,
)
from qec.analysis.quaternary_message_passing import (
    aggregate_messages,
    make_message,
    run_message_passing_round,
)
from qec.experiments.quaternary_bosonic_decoder import (
    format_summary,
    run_quaternary_bosonic_experiment,
)
from qec.analysis.strategy_generation import generate_strategies
from qec.analysis.strategy_adapter import (
    format_comparison_summary,
    run_dual_generation_pipeline,
    run_generation_selection_pipeline,
)


# ─── quaternary_quantization ──────────────────────────────────────

class TestQuantizeQuaternary:
    def test_basic(self):
        vals = np.array([-0.9, -0.4, 0.3, 0.8])
        out = quantize_quaternary(vals)
        np.testing.assert_array_equal(out, [-1.0, -0.5, 0.5, 1.0])

    def test_exact_states(self):
        vals = np.array([-1.0, -0.5, 0.5, 1.0])
        out = quantize_quaternary(vals)
        np.testing.assert_array_equal(out, vals)

    def test_only_four_states(self):
        vals = np.linspace(-1.0, 1.0, 100)
        out = quantize_quaternary(vals)
        unique = set(out.tolist())
        assert unique.issubset(set(QUATERNARY_STATES))

    def test_midpoint_boundary(self):
        # At exact midpoint between -0.5 and 0.5 (i.e. 0.0),
        # argmin picks the first match → -0.5
        vals = np.array([0.0])
        out = quantize_quaternary(vals)
        assert out[0] in QUATERNARY_STATES

    def test_empty(self):
        out = quantize_quaternary(np.array([]))
        assert out.size == 0

    def test_determinism(self):
        vals = np.array([0.3, -0.7, 0.1, -0.2])
        a = quantize_quaternary(vals)
        b = quantize_quaternary(vals)
        np.testing.assert_array_equal(a, b)

    def test_no_mutation(self):
        vals = np.array([0.3, -0.7, 0.1])
        original = vals.copy()
        quantize_quaternary(vals)
        np.testing.assert_array_equal(vals, original)


class TestQuaternaryStats:
    def test_basic(self):
        q = np.array([1.0, -1.0, 0.5, -0.5, 0.5])
        s = quaternary_stats(q)
        assert s["n_strong_positive"] == 1
        assert s["n_strong_negative"] == 1
        assert s["n_soft_positive"] == 2
        assert s["n_soft_negative"] == 1
        assert abs(s["soft_fraction"] - 0.6) < 1e-10


# ─── quaternary_message_passing ───────────────────────────────────

class TestQuaternaryMakeMessage:
    def test_valid(self):
        m = make_message(0.5, 0.7)
        assert m == {"state": 0.5, "confidence": 0.7}

    def test_all_states(self):
        for s in QUATERNARY_STATES:
            m = make_message(s, 0.5)
            assert m["state"] == s

    def test_invalid_state(self):
        with pytest.raises(ValueError):
            make_message(0.0, 0.5)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            make_message(0.5, 1.5)


class TestQuaternaryAggregateMessages:
    def test_empty(self):
        result = aggregate_messages([])
        assert result["state"] in QUATERNARY_STATES
        assert result["confidence"] == 0.0

    def test_unanimous_positive(self):
        msgs = [make_message(1.0, 0.9), make_message(0.5, 0.8)]
        agg = aggregate_messages(msgs)
        assert agg["state"] > 0
        assert agg["state"] in QUATERNARY_STATES

    def test_unanimous_negative(self):
        msgs = [make_message(-1.0, 0.9), make_message(-0.5, 0.8)]
        agg = aggregate_messages(msgs)
        assert agg["state"] < 0
        assert agg["state"] in QUATERNARY_STATES

    def test_determinism(self):
        msgs = [
            make_message(1.0, 0.7),
            make_message(-0.5, 0.3),
            make_message(0.5, 0.5),
        ]
        a = aggregate_messages(msgs)
        b = aggregate_messages(msgs)
        assert a == b

    def test_confidence_bounded(self):
        msgs = [make_message(1.0, 1.0), make_message(-1.0, 0.0)]
        agg = aggregate_messages(msgs)
        assert 0.0 <= agg["confidence"] <= 1.0


class TestQuaternaryMessagePassingRound:
    def test_basic_chain(self):
        states = np.array([1.0, -0.5, 0.5], dtype=np.float64)
        confs = np.array([0.8, 0.5, 0.7])
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
        new_s, new_c = run_message_passing_round(states, confs, adj)
        assert new_s.shape == (3,)
        assert new_c.shape == (3,)
        for val in new_s:
            assert val in QUATERNARY_STATES

    def test_isolated_nodes(self):
        states = np.array([1.0, -1.0], dtype=np.float64)
        confs = np.array([0.9, 0.7])
        adj = np.zeros((2, 2), dtype=np.int8)
        new_s, new_c = run_message_passing_round(states, confs, adj)
        np.testing.assert_array_equal(new_s, states)
        np.testing.assert_array_equal(new_c, confs)

    def test_no_mutation(self):
        states = np.array([1.0, -0.5, 0.5], dtype=np.float64)
        confs = np.array([0.8, 0.5, 0.7])
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
        orig_s = states.copy()
        orig_c = confs.copy()
        run_message_passing_round(states, confs, adj)
        np.testing.assert_array_equal(states, orig_s)
        np.testing.assert_array_equal(confs, orig_c)


# ─── quaternary_bosonic_decoder ───────────────────────────────────

class TestQuaternaryExperiment:
    def test_basic_run(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9])
        result = run_quaternary_bosonic_experiment(raw, rounds=3)
        assert result["n_signals"] == 6
        assert result["rounds"] == 3
        assert result["state_system"] == "quaternary"
        assert len(result["round_diagnostics"]) == 3
        assert "metrics" in result
        assert 0.0 <= result["metrics"]["design_score"] <= 1.0

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            run_quaternary_bosonic_experiment(np.array([1.0]), rounds=2)
        with pytest.raises(ValueError):
            run_quaternary_bosonic_experiment(np.array([1.0]), rounds=6)

    def test_determinism(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4])
        a = run_quaternary_bosonic_experiment(raw, rounds=3)
        b = run_quaternary_bosonic_experiment(raw, rounds=3)
        assert a["final_states"] == b["final_states"]
        assert a["final_confidences"] == b["final_confidences"]
        assert a["metrics"] == b["metrics"]

    def test_five_rounds(self):
        raw = np.array([0.2, -0.6, 0.3, 0.7, -0.4])
        result = run_quaternary_bosonic_experiment(raw, rounds=5)
        assert len(result["round_diagnostics"]) == 5

    def test_final_states_are_quaternary(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9])
        result = run_quaternary_bosonic_experiment(raw, rounds=3)
        for s in result["final_states"]:
            assert s in QUATERNARY_STATES

    def test_format_summary(self):
        raw = np.array([0.1, -0.8, 0.5])
        result = run_quaternary_bosonic_experiment(raw, rounds=3)
        summary = format_summary(result)
        assert "Quaternary Bosonic Decoder" in summary
        assert "Design score" in summary

    def test_no_input_mutation(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0])
        original = raw.copy()
        run_quaternary_bosonic_experiment(raw, rounds=3)
        np.testing.assert_array_equal(raw, original)

    def test_metrics_bounded(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9])
        result = run_quaternary_bosonic_experiment(raw, rounds=3)
        for v in result["metrics"].values():
            assert 0.0 <= v <= 1.0


# ─── dual-system strategy generation ─────────────────────────────

def _make_base_strategy():
    return {
        "config": {
            "threshold": 0.3,
            "rounds": 3,
            "confidence_scale": 1.0,
        },
        "metrics": {
            "design_score": 0.7,
            "confidence_efficiency": 0.6,
        },
    }


class TestDualGeneration:
    def test_54_strategies_when_enabled(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        assert len(result) == 54

    def test_27_without_quaternary(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=False)
        assert len(result) == 27

    def test_correct_split(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        ternary = [s for s in result if s["state_system"] == "ternary"]
        quaternary = [s for s in result if s["state_system"] == "quaternary"]
        assert len(ternary) == 27
        assert len(quaternary) == 27

    def test_deterministic_naming(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        names = [s["name"] for s in result]
        assert names == sorted(names)
        assert len(names) == len(set(names))

    def test_ternary_names_prefix(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        ternary = [s for s in result if s["state_system"] == "ternary"]
        for s in ternary:
            assert s["name"].startswith("ternary__")

    def test_quaternary_names_prefix(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        quaternary = [s for s in result if s["state_system"] == "quaternary"]
        for s in quaternary:
            assert s["name"].startswith("quaternary__")

    def test_state_system_in_config(self):
        base = _make_base_strategy()
        result = generate_strategies(base, include_quaternary=True)
        for s in result:
            assert s["config"]["state_system"] == s["state_system"]

    def test_no_mutation(self):
        base = _make_base_strategy()
        original = copy.deepcopy(base)
        generate_strategies(base, include_quaternary=True)
        assert base == original

    def test_deterministic_output(self):
        base = _make_base_strategy()
        r1 = generate_strategies(base, include_quaternary=True)
        r2 = generate_strategies(base, include_quaternary=True)
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a["name"] == b["name"]
            assert a["config"] == b["config"]
            assert a["state_system"] == b["state_system"]


class TestGenerationPipelineWithQuaternary:
    def test_pipeline_54_candidates(self):
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(
            base, include_quaternary=True,
        )
        assert len(result["candidates"]) == 54
        assert len(result["ranked"]) == 54

    def test_pipeline_selected_has_score(self):
        base = _make_base_strategy()
        result = run_generation_selection_pipeline(
            base, include_quaternary=True,
        )
        assert "_score" in result["selected"]
        assert 0.0 <= result["selected"]["_score"] <= 1.0

    def test_pipeline_determinism(self):
        base = _make_base_strategy()
        ts = {"stability": 0.8, "global_trust": 0.6}
        r1 = run_generation_selection_pipeline(
            base, trust_signals=ts, include_quaternary=True,
        )
        r2 = run_generation_selection_pipeline(
            base, trust_signals=ts, include_quaternary=True,
        )
        assert r1["selected"]["name"] == r2["selected"]["name"]
        assert r1["selected"]["_score"] == r2["selected"]["_score"]


class TestDualGenerationPipeline:
    def test_basic_run(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4])
        base = {
            "config": {"threshold": 0.3, "rounds": 3, "confidence_scale": 1.0},
            "metrics": {"design_score": 0.7, "confidence_efficiency": 0.6},
        }
        result = run_dual_generation_pipeline(
            base, raw_signals=raw,
            trust_signals={"stability": 0.8, "global_trust": 0.6},
        )
        assert len(result["candidates"]) == 54
        assert result["n_ternary"] == 27
        assert result["n_quaternary"] == 27
        assert result["best_ternary"] is not None
        assert result["best_quaternary"] is not None
        assert "selected" in result

    def test_determinism(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4])
        base = {
            "config": {"threshold": 0.3, "rounds": 3, "confidence_scale": 1.0},
            "metrics": {"design_score": 0.7, "confidence_efficiency": 0.6},
        }
        ts = {"stability": 0.8, "global_trust": 0.6}
        r1 = run_dual_generation_pipeline(base, raw, ts)
        r2 = run_dual_generation_pipeline(base, raw, ts)
        assert r1["selected"]["name"] == r2["selected"]["name"]
        assert r1["best_ternary"]["_score"] == r2["best_ternary"]["_score"]
        assert r1["best_quaternary"]["_score"] == r2["best_quaternary"]["_score"]

    def test_format_comparison(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4])
        base = {
            "config": {"threshold": 0.3, "rounds": 3, "confidence_scale": 1.0},
            "metrics": {"design_score": 0.7, "confidence_efficiency": 0.6},
        }
        result = run_dual_generation_pipeline(
            base, raw_signals=raw,
            trust_signals={"stability": 0.8, "global_trust": 0.6},
        )
        summary = format_comparison_summary(result)
        assert "State System Comparison" in summary
        assert "Ternary best:" in summary
        assert "Quaternary best:" in summary
        assert "Winner:" in summary
        assert "54" in summary
