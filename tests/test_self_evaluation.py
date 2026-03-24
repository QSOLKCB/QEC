"""Tests for benchmark-aware self-evaluation — v101.1.0.

Covers:
- deterministic outputs
- bounds [0, 1]
- neutral fallback
- confidence higher when QEC outperforms baselines
- confidence modulation in [0.9, 1.1]
- no mutation of inputs

All tests are deterministic.  No randomness, no network, no GPU.
"""

from __future__ import annotations

import copy

import pytest

from qec.analysis.self_evaluation import (
    compute_benchmark_confidence,
    compute_confidence_modulation,
    compute_relative_advantage,
    compute_self_evaluation_signal,
)
from qec.analysis.benchmark_comparison import summarize_baseline_finals


# ---------------------------------------------------------------------------
# compute_relative_advantage
# ---------------------------------------------------------------------------

class TestComputeRelativeAdvantage:
    """Tests for compute_relative_advantage."""

    def test_qec_better_than_baseline(self) -> None:
        result = compute_relative_advantage(0.8, 0.5)
        assert 0.0 <= result <= 1.0
        assert result > 0.0

    def test_qec_equal_to_baseline(self) -> None:
        result = compute_relative_advantage(0.5, 0.5)
        assert result == 0.0

    def test_qec_worse_than_baseline(self) -> None:
        result = compute_relative_advantage(0.3, 0.7)
        assert result == 0.0

    def test_both_zero(self) -> None:
        result = compute_relative_advantage(0.0, 0.0)
        assert result == 0.0

    def test_bounded_output(self) -> None:
        # Even with extreme values, output stays in [0, 1]
        result = compute_relative_advantage(100.0, 0.0)
        assert 0.0 <= result <= 1.0

    def test_deterministic(self) -> None:
        r1 = compute_relative_advantage(0.75, 0.4)
        r2 = compute_relative_advantage(0.75, 0.4)
        assert r1 == r2

    def test_negative_scores(self) -> None:
        result = compute_relative_advantage(-0.1, -0.5)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_benchmark_confidence
# ---------------------------------------------------------------------------

class TestComputeBenchmarkConfidence:
    """Tests for compute_benchmark_confidence."""

    def test_no_baselines_returns_zero(self) -> None:
        assert compute_benchmark_confidence(0.8, {}) == 0.0

    def test_qec_outperforms_all(self) -> None:
        result = compute_benchmark_confidence(
            0.9, {"a": 0.3, "b": 0.4, "c": 0.2},
        )
        assert 0.0 < result <= 1.0

    def test_qec_underperforms_all(self) -> None:
        result = compute_benchmark_confidence(
            0.1, {"a": 0.5, "b": 0.6},
        )
        assert result == 0.0

    def test_bounded(self) -> None:
        result = compute_benchmark_confidence(
            1.0, {"a": 0.0, "b": 0.0},
        )
        assert 0.0 <= result <= 1.0

    def test_deterministic(self) -> None:
        baselines = {"x": 0.3, "y": 0.5}
        r1 = compute_benchmark_confidence(0.7, baselines)
        r2 = compute_benchmark_confidence(0.7, baselines)
        assert r1 == r2

    def test_higher_when_outperforming(self) -> None:
        """Confidence should be higher when QEC strongly outperforms."""
        c_strong = compute_benchmark_confidence(0.9, {"a": 0.1})
        c_weak = compute_benchmark_confidence(0.6, {"a": 0.5})
        assert c_strong > c_weak


# ---------------------------------------------------------------------------
# compute_confidence_modulation
# ---------------------------------------------------------------------------

class TestComputeConfidenceModulation:
    """Tests for compute_confidence_modulation."""

    def test_zero_confidence(self) -> None:
        result = compute_confidence_modulation(0.0)
        assert result == pytest.approx(0.9)

    def test_full_confidence(self) -> None:
        result = compute_confidence_modulation(1.0)
        assert result == pytest.approx(1.1)

    def test_half_confidence_neutral(self) -> None:
        result = compute_confidence_modulation(0.5)
        assert result == pytest.approx(1.0)

    def test_range_lower_bound(self) -> None:
        assert compute_confidence_modulation(0.0) >= 0.9

    def test_range_upper_bound(self) -> None:
        assert compute_confidence_modulation(1.0) <= 1.1

    def test_clamped_below(self) -> None:
        result = compute_confidence_modulation(-1.0)
        assert result >= 0.9

    def test_clamped_above(self) -> None:
        result = compute_confidence_modulation(2.0)
        assert result <= 1.1

    def test_deterministic(self) -> None:
        r1 = compute_confidence_modulation(0.73)
        r2 = compute_confidence_modulation(0.73)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_self_evaluation_signal
# ---------------------------------------------------------------------------

class TestComputeSelfEvaluationSignal:
    """Tests for compute_self_evaluation_signal."""

    def test_basic_output_keys(self) -> None:
        result = compute_self_evaluation_signal(
            {"final_score": 0.8},
            {"baseline_a": {"final_score": 0.5}},
        )
        assert "relative_advantage" in result
        assert "benchmark_confidence" in result
        assert "margin_over_baseline" in result
        assert "confidence_modulation" in result

    def test_all_outputs_bounded(self) -> None:
        result = compute_self_evaluation_signal(
            {"final_score": 0.9},
            {"a": {"final_score": 0.2}, "b": {"final_score": 0.4}},
        )
        assert 0.0 <= result["relative_advantage"] <= 1.0
        assert 0.0 <= result["benchmark_confidence"] <= 1.0
        assert 0.0 <= result["margin_over_baseline"] <= 1.0
        assert 0.9 <= result["confidence_modulation"] <= 1.1

    def test_neutral_fallback_no_baselines(self) -> None:
        result = compute_self_evaluation_signal(
            {"final_score": 0.5},
            {},
        )
        assert result["relative_advantage"] == 0.0
        assert result["benchmark_confidence"] == 0.0
        # margin is qec_final - best_baseline; with no baselines, best=0.0
        assert 0.0 <= result["margin_over_baseline"] <= 1.0
        # No baselines → confidence 0 → modulation 0.9
        assert result["confidence_modulation"] == pytest.approx(0.9)

    def test_no_mutation_of_inputs(self) -> None:
        qec = {"final_score": 0.8}
        baselines = {"a": {"final_score": 0.3}, "b": {"final_score": 0.5}}
        qec_copy = copy.deepcopy(qec)
        baselines_copy = copy.deepcopy(baselines)

        compute_self_evaluation_signal(qec, baselines)

        assert qec == qec_copy
        assert baselines == baselines_copy

    def test_deterministic(self) -> None:
        qec = {"final_score": 0.7}
        baselines = {"a": {"final_score": 0.4}, "b": {"final_score": 0.6}}
        r1 = compute_self_evaluation_signal(qec, baselines)
        r2 = compute_self_evaluation_signal(qec, baselines)
        assert r1 == r2

    def test_missing_final_score_defaults_zero(self) -> None:
        result = compute_self_evaluation_signal(
            {},
            {"a": {}},
        )
        assert 0.0 <= result["relative_advantage"] <= 1.0
        assert 0.0 <= result["benchmark_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# summarize_baseline_finals
# ---------------------------------------------------------------------------

class TestSummarizeBaselineFinals:
    """Tests for benchmark_comparison.summarize_baseline_finals."""

    def test_extracts_scores(self) -> None:
        results = {
            "a": {"final_score": 0.5},
            "b": {"final_score": 0.7},
        }
        finals = summarize_baseline_finals(results)
        assert finals == {"a": 0.5, "b": 0.7}

    def test_skips_missing_score(self) -> None:
        results = {
            "a": {"final_score": 0.5},
            "b": {"other_key": 0.7},
        }
        finals = summarize_baseline_finals(results)
        assert finals == {"a": 0.5}

    def test_skips_non_dict_entry(self) -> None:
        results = {"a": {"final_score": 0.5}, "b": "invalid"}
        finals = summarize_baseline_finals(results)
        assert finals == {"a": 0.5}

    def test_empty_input(self) -> None:
        assert summarize_baseline_finals({}) == {}

    def test_no_mutation(self) -> None:
        results = {"a": {"final_score": 0.5}}
        results_copy = copy.deepcopy(results)
        summarize_baseline_finals(results)
        assert results == results_copy

    def test_deterministic(self) -> None:
        results = {"c": {"final_score": 0.3}, "a": {"final_score": 0.9}}
        r1 = summarize_baseline_finals(results)
        r2 = summarize_baseline_finals(results)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Integration: confidence_modulation in compute_robust_score
# ---------------------------------------------------------------------------

class TestConfidenceModulationInRobustScore:
    """Verify confidence_modulation integrates with compute_robust_score."""

    def test_default_neutral(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        # Default confidence_modulation=1.0 should not change result
        score_without = compute_robust_score(0.5)
        score_with = compute_robust_score(0.5, confidence_modulation=1.0)
        assert score_without == score_with

    def test_modulation_increases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, confidence_modulation=1.0)
        score_boosted = compute_robust_score(0.5, confidence_modulation=1.1)
        assert score_boosted > score_neutral

    def test_modulation_decreases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, confidence_modulation=1.0)
        score_reduced = compute_robust_score(0.5, confidence_modulation=0.9)
        assert score_reduced < score_neutral

    def test_result_still_bounded(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score = compute_robust_score(
            1.0, 1.0, 1.2, 1.2, 1.5, 1.0, 1.1, 1.1,
        )
        assert 0.0 <= score <= 1.0
