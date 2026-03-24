"""Tests for the ternary bosonic decoder experimental layer.

Covers: determinism, bounds, no mutation, edge cases for all modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.analysis.bosonic_interface import normalize_to_bipolar, prepare_bosonic_input
from qec.analysis.decoder_design_baselines import (
    hard_threshold_decoder,
    run_baselines,
    signed_soft_decoder,
)
from qec.analysis.decoder_design_metrics import (
    agreement_rate,
    compute_all_metrics,
    confidence_efficiency,
    design_score,
    neutral_usage,
)
from qec.analysis.ternary_message_passing import (
    aggregate_messages,
    make_message,
    run_message_passing_round,
)
from qec.analysis.ternary_quantization import quantize_ternary, ternary_stats
from qec.experiments.concatenated_bosonic_decoder import (
    format_summary,
    run_concatenated_bosonic_experiment,
)


# ─── bosonic_interface ───────────────────────────────────────────────

class TestNormalizeToBipolar:
    def test_basic_normalization(self):
        vals = np.array([0.0, 5.0, 10.0])
        out = normalize_to_bipolar(vals)
        np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])

    def test_constant_input(self):
        vals = np.array([3.0, 3.0, 3.0])
        out = normalize_to_bipolar(vals)
        np.testing.assert_array_equal(out, [0.0, 0.0, 0.0])

    def test_empty_input(self):
        out = normalize_to_bipolar(np.array([]))
        assert out.size == 0

    def test_bounds(self):
        vals = np.array([-100.0, 0.0, 50.0, 200.0])
        out = normalize_to_bipolar(vals)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_determinism(self):
        vals = np.array([1.5, -2.3, 0.7, 4.1])
        a = normalize_to_bipolar(vals)
        b = normalize_to_bipolar(vals)
        np.testing.assert_array_equal(a, b)

    def test_no_mutation(self):
        vals = np.array([1.0, 2.0, 3.0])
        original = vals.copy()
        normalize_to_bipolar(vals)
        np.testing.assert_array_equal(vals, original)


class TestPrepareBosonic:
    def test_packet_structure(self):
        raw = np.array([1.0, 2.0, 3.0])
        pkt = prepare_bosonic_input(raw)
        assert "normalized" in pkt
        assert pkt["length"] == 3
        assert pkt["input_range"] == (1.0, 3.0)

    def test_empty(self):
        pkt = prepare_bosonic_input(np.array([]))
        assert pkt["length"] == 0


# ─── ternary_quantization ───────────────────────────────────────────

class TestQuantizeTernary:
    def test_basic(self):
        vals = np.array([-0.5, -0.1, 0.0, 0.2, 0.8])
        out = quantize_ternary(vals, threshold=0.3)
        np.testing.assert_array_equal(out, [-1, 0, 0, 0, 1])

    def test_boundary_values(self):
        vals = np.array([-0.3, 0.3])
        out = quantize_ternary(vals, threshold=0.3)
        np.testing.assert_array_equal(out, [-1, 1])

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            quantize_ternary(np.array([0.0]), threshold=0.0)
        with pytest.raises(ValueError):
            quantize_ternary(np.array([0.0]), threshold=1.5)

    def test_all_neutral(self):
        vals = np.array([0.0, 0.1, -0.1])
        out = quantize_ternary(vals, threshold=0.5)
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_determinism(self):
        vals = np.array([0.5, -0.7, 0.1])
        a = quantize_ternary(vals)
        b = quantize_ternary(vals)
        np.testing.assert_array_equal(a, b)


class TestTernaryStats:
    def test_basic(self):
        t = np.array([1, 0, -1, 0, 1], dtype=np.int8)
        s = ternary_stats(t)
        assert s["n_positive"] == 2
        assert s["n_negative"] == 1
        assert s["n_neutral"] == 2
        assert abs(s["neutral_fraction"] - 0.4) < 1e-10


# ─── ternary_message_passing ────────────────────────────────────────

class TestMakeMessage:
    def test_valid(self):
        m = make_message(1, 0.5)
        assert m == {"state": 1, "confidence": 0.5}

    def test_invalid_state(self):
        with pytest.raises(ValueError):
            make_message(2, 0.5)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            make_message(0, 1.5)


class TestAggregateMessages:
    def test_empty(self):
        assert aggregate_messages([]) == {"state": 0, "confidence": 0.0}

    def test_unanimous_positive(self):
        msgs = [make_message(1, 0.9), make_message(1, 0.8)]
        agg = aggregate_messages(msgs)
        assert agg["state"] == 1
        assert agg["confidence"] > 0.0

    def test_tie_resolves_neutral(self):
        msgs = [make_message(1, 0.5), make_message(-1, 0.5)]
        agg = aggregate_messages(msgs)
        assert agg["state"] == 0

    def test_determinism(self):
        msgs = [make_message(1, 0.7), make_message(-1, 0.3), make_message(0, 0.5)]
        a = aggregate_messages(msgs)
        b = aggregate_messages(msgs)
        assert a == b


class TestMessagePassingRound:
    def test_basic_chain(self):
        states = np.array([1, 0, -1], dtype=np.int8)
        confs = np.array([0.8, 0.5, 0.9])
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
        new_s, new_c = run_message_passing_round(states, confs, adj)
        assert new_s.shape == (3,)
        assert new_c.shape == (3,)
        assert set(new_s.tolist()).issubset({-1, 0, 1})

    def test_isolated_nodes(self):
        states = np.array([1, -1], dtype=np.int8)
        confs = np.array([0.9, 0.7])
        adj = np.zeros((2, 2), dtype=np.int8)
        new_s, new_c = run_message_passing_round(states, confs, adj)
        np.testing.assert_array_equal(new_s, states)
        np.testing.assert_array_equal(new_c, confs)

    def test_no_mutation(self):
        states = np.array([1, 0, -1], dtype=np.int8)
        confs = np.array([0.8, 0.5, 0.9])
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.int8)
        orig_s = states.copy()
        orig_c = confs.copy()
        run_message_passing_round(states, confs, adj)
        np.testing.assert_array_equal(states, orig_s)
        np.testing.assert_array_equal(confs, orig_c)


# ─── decoder_design_baselines ───────────────────────────────────────

class TestBaselines:
    def test_hard_threshold(self):
        vals = np.array([-1.0, 0.0, 0.5, -0.3])
        out = hard_threshold_decoder(vals)
        np.testing.assert_array_equal(out, [-1, -1, 1, -1])

    def test_signed_soft(self):
        vals = np.array([-2.0, 0.0, 3.0])
        out = signed_soft_decoder(vals)
        np.testing.assert_allclose(out, [-2.0, 0.0, 3.0], atol=1e-10)
        # sign(0) = +1, so 0 * +1 = 0
        assert out[1] == 0.0

    def test_run_baselines(self):
        vals = np.array([0.5, -0.3])
        result = run_baselines(vals)
        assert "hard_threshold" in result
        assert "signed_soft" in result

    def test_determinism(self):
        vals = np.array([0.1, -0.5, 0.8])
        a = run_baselines(vals)
        b = run_baselines(vals)
        np.testing.assert_array_equal(a["hard_threshold"], b["hard_threshold"])


# ─── decoder_design_metrics ─────────────────────────────────────────

class TestMetrics:
    def test_neutral_usage(self):
        t = np.array([0, 0, 1, -1], dtype=np.int8)
        assert abs(neutral_usage(t) - 0.5) < 1e-10

    def test_neutral_usage_empty(self):
        assert neutral_usage(np.array([], dtype=np.int8)) == 0.0

    def test_confidence_efficiency(self):
        c = np.array([0.5, 0.8, 1.0])
        assert abs(confidence_efficiency(c) - 0.7666666666666667) < 1e-10

    def test_confidence_efficiency_empty(self):
        assert confidence_efficiency(np.array([])) == 0.0

    def test_agreement_rate(self):
        t = np.array([1, -1, 0], dtype=np.int8)
        b = np.array([1, -1, 1])
        assert agreement_rate(t, b) == 1.0  # all agree (neutral abstains)

    def test_design_score_bounded(self):
        t = np.array([1, 0, -1], dtype=np.int8)
        c = np.array([0.9, 0.3, 0.8])
        b = np.array([1, 1, -1])
        s = design_score(t, c, b)
        assert 0.0 <= s <= 1.0

    def test_compute_all(self):
        t = np.array([1, 0, -1], dtype=np.int8)
        c = np.array([0.9, 0.3, 0.8])
        b = np.array([1, 1, -1])
        m = compute_all_metrics(t, c, b)
        assert set(m.keys()) == {
            "neutral_usage", "confidence_efficiency",
            "agreement_rate", "design_score",
        }
        for v in m.values():
            assert 0.0 <= v <= 1.0

    def test_determinism(self):
        t = np.array([1, 0, -1], dtype=np.int8)
        c = np.array([0.9, 0.3, 0.8])
        b = np.array([1, 1, -1])
        a = compute_all_metrics(t, c, b)
        bb = compute_all_metrics(t, c, b)
        assert a == bb


# ─── concatenated_bosonic_decoder ────────────────────────────────────

class TestConcatenatedExperiment:
    def test_basic_run(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9])
        result = run_concatenated_bosonic_experiment(raw, rounds=3)
        assert result["n_signals"] == 6
        assert result["rounds"] == 3
        assert len(result["round_diagnostics"]) == 3
        assert "metrics" in result
        assert 0.0 <= result["metrics"]["design_score"] <= 1.0

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            run_concatenated_bosonic_experiment(np.array([1.0]), rounds=2)
        with pytest.raises(ValueError):
            run_concatenated_bosonic_experiment(np.array([1.0]), rounds=6)

    def test_determinism(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0, -0.3, 0.9, -0.1, 0.4])
        a = run_concatenated_bosonic_experiment(raw, threshold=0.3, rounds=3)
        b = run_concatenated_bosonic_experiment(raw, threshold=0.3, rounds=3)
        assert a["final_states"] == b["final_states"]
        assert a["final_confidences"] == b["final_confidences"]
        assert a["metrics"] == b["metrics"]

    def test_five_rounds(self):
        raw = np.array([0.2, -0.6, 0.3, 0.7, -0.4])
        result = run_concatenated_bosonic_experiment(raw, rounds=5)
        assert len(result["round_diagnostics"]) == 5

    def test_custom_adjacency(self):
        raw = np.array([0.5, -0.5, 0.5])
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int8)
        result = run_concatenated_bosonic_experiment(raw, adjacency=adj, rounds=3)
        assert result["n_signals"] == 3

    def test_format_summary(self):
        raw = np.array([0.1, -0.8, 0.5])
        result = run_concatenated_bosonic_experiment(raw, rounds=3)
        summary = format_summary(result)
        assert "Ternary Bosonic Decoder" in summary
        assert "Design score" in summary

    def test_no_input_mutation(self):
        raw = np.array([0.1, -0.8, 0.5, 0.0])
        original = raw.copy()
        run_concatenated_bosonic_experiment(raw, rounds=3)
        np.testing.assert_array_equal(raw, original)
