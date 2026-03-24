"""Tests for temporal confidence tracking — v101.2.0.

Covers:
- history update correctness and bounded length
- variance correctness
- stability in [0, 1]
- trend in [-1, 1]
- trust in [0, 1]
- trust_modulation in [0.9, 1.1]
- deterministic repeated runs
- no mutation of inputs

All tests are deterministic.  No randomness, no network, no GPU.
"""

from __future__ import annotations

import copy

import pytest

from qec.analysis.temporal_confidence import (
    compute_confidence_stability,
    compute_confidence_trend,
    compute_confidence_variance,
    compute_trust_modulation,
    compute_trust_signal,
    update_confidence_history,
)
from qec.analysis.self_evaluation import compute_temporal_self_evaluation


# ---------------------------------------------------------------------------
# update_confidence_history
# ---------------------------------------------------------------------------


class TestUpdateConfidenceHistory:
    """Tests for update_confidence_history."""

    def test_append_to_empty(self) -> None:
        result = update_confidence_history([], 0.5)
        assert result == [0.5]

    def test_append_preserves_order(self) -> None:
        result = update_confidence_history([0.1, 0.2], 0.3)
        assert result == [0.1, 0.2, 0.3]

    def test_bounded_length(self) -> None:
        history = [float(i) / 10.0 for i in range(10)]
        result = update_confidence_history(history, 0.99, max_len=10)
        assert len(result) == 10
        assert result[-1] == 0.99
        # First element should be dropped
        assert result[0] == 0.1

    def test_bounded_length_small(self) -> None:
        result = update_confidence_history([0.1, 0.2, 0.3], 0.4, max_len=3)
        assert len(result) == 3
        assert result == [0.2, 0.3, 0.4]

    def test_no_mutation_of_input(self) -> None:
        original = [0.1, 0.2, 0.3]
        original_copy = copy.deepcopy(original)
        update_confidence_history(original, 0.4)
        assert original == original_copy

    def test_deterministic(self) -> None:
        r1 = update_confidence_history([0.1, 0.2], 0.3)
        r2 = update_confidence_history([0.1, 0.2], 0.3)
        assert r1 == r2

    def test_max_len_one(self) -> None:
        result = update_confidence_history([0.1, 0.2], 0.3, max_len=1)
        assert result == [0.3]


# ---------------------------------------------------------------------------
# compute_confidence_variance
# ---------------------------------------------------------------------------


class TestComputeConfidenceVariance:
    """Tests for compute_confidence_variance."""

    def test_empty_returns_zero(self) -> None:
        assert compute_confidence_variance([]) == 0.0

    def test_single_returns_zero(self) -> None:
        assert compute_confidence_variance([0.5]) == 0.0

    def test_identical_values_zero_variance(self) -> None:
        assert compute_confidence_variance([0.5, 0.5, 0.5]) == 0.0

    def test_known_variance(self) -> None:
        # [0, 1] -> mean=0.5, variance = ((0.5)^2 + (0.5)^2) / 2 = 0.25
        result = compute_confidence_variance([0.0, 1.0])
        assert result == pytest.approx(0.25)

    def test_non_negative(self) -> None:
        result = compute_confidence_variance([0.1, 0.9, 0.3, 0.7])
        assert result >= 0.0

    def test_deterministic(self) -> None:
        history = [0.1, 0.3, 0.5, 0.7]
        r1 = compute_confidence_variance(history)
        r2 = compute_confidence_variance(history)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_confidence_stability
# ---------------------------------------------------------------------------


class TestComputeConfidenceStability:
    """Tests for compute_confidence_stability."""

    def test_empty_returns_one(self) -> None:
        assert compute_confidence_stability([]) == 1.0

    def test_single_returns_one(self) -> None:
        assert compute_confidence_stability([0.5]) == 1.0

    def test_identical_values_max_stability(self) -> None:
        assert compute_confidence_stability([0.5, 0.5, 0.5]) == 1.0

    def test_bounded_zero_one(self) -> None:
        result = compute_confidence_stability([0.0, 1.0, 0.0, 1.0])
        assert 0.0 <= result <= 1.0

    def test_high_variance_low_stability(self) -> None:
        # Wide spread -> higher variance -> lower stability
        stable = compute_confidence_stability([0.5, 0.5, 0.5])
        volatile = compute_confidence_stability([0.0, 1.0, 0.0, 1.0])
        assert stable > volatile

    def test_deterministic(self) -> None:
        history = [0.2, 0.4, 0.6]
        r1 = compute_confidence_stability(history)
        r2 = compute_confidence_stability(history)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_confidence_trend
# ---------------------------------------------------------------------------


class TestComputeConfidenceTrend:
    """Tests for compute_confidence_trend."""

    def test_empty_returns_zero(self) -> None:
        assert compute_confidence_trend([]) == 0.0

    def test_single_returns_zero(self) -> None:
        assert compute_confidence_trend([0.5]) == 0.0

    def test_increasing_positive(self) -> None:
        result = compute_confidence_trend([0.2, 0.4, 0.6])
        assert result > 0.0

    def test_decreasing_negative(self) -> None:
        result = compute_confidence_trend([0.8, 0.5, 0.2])
        assert result < 0.0

    def test_flat_zero(self) -> None:
        result = compute_confidence_trend([0.5, 0.5, 0.5])
        assert result == 0.0

    def test_bounded_minus_one_one(self) -> None:
        # Extreme case
        result = compute_confidence_trend([0.0, 1.0])
        assert -1.0 <= result <= 1.0

    def test_clamped_large_difference(self) -> None:
        # Even if we pass values outside [0,1], trend is clamped
        result = compute_confidence_trend([-5.0, 10.0])
        assert result == 1.0

    def test_deterministic(self) -> None:
        history = [0.3, 0.5, 0.7]
        r1 = compute_confidence_trend(history)
        r2 = compute_confidence_trend(history)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_trust_signal
# ---------------------------------------------------------------------------


class TestComputeTrustSignal:
    """Tests for compute_trust_signal."""

    def test_bounded_zero_one(self) -> None:
        for s in [0.0, 0.5, 1.0]:
            for t in [-1.0, 0.0, 1.0]:
                result = compute_trust_signal(s, t)
                assert 0.0 <= result <= 1.0, f"s={s}, t={t}, result={result}"

    def test_max_trust(self) -> None:
        # stability=1.0, trend=1.0 -> trend_factor=1.0 -> trust=1.0
        assert compute_trust_signal(1.0, 1.0) == pytest.approx(1.0)

    def test_min_trust(self) -> None:
        # stability=1.0, trend=-1.0 -> trend_factor=0.0 -> trust=0.0
        assert compute_trust_signal(1.0, -1.0) == pytest.approx(0.0)

    def test_zero_stability(self) -> None:
        # stability=0 -> trust=0 regardless of trend
        assert compute_trust_signal(0.0, 1.0) == pytest.approx(0.0)

    def test_neutral_trend(self) -> None:
        # stability=1.0, trend=0.0 -> trend_factor=0.5 -> trust=0.5
        assert compute_trust_signal(1.0, 0.0) == pytest.approx(0.5)

    def test_high_stability_positive_trend_high_trust(self) -> None:
        high = compute_trust_signal(0.9, 0.5)
        low = compute_trust_signal(0.9, -0.5)
        assert high > low

    def test_deterministic(self) -> None:
        r1 = compute_trust_signal(0.8, 0.3)
        r2 = compute_trust_signal(0.8, 0.3)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_trust_modulation
# ---------------------------------------------------------------------------


class TestComputeTrustModulation:
    """Tests for compute_trust_modulation."""

    def test_zero_trust(self) -> None:
        assert compute_trust_modulation(0.0) == pytest.approx(0.9)

    def test_full_trust(self) -> None:
        assert compute_trust_modulation(1.0) == pytest.approx(1.1)

    def test_half_trust_neutral(self) -> None:
        assert compute_trust_modulation(0.5) == pytest.approx(1.0)

    def test_range_lower_bound(self) -> None:
        assert compute_trust_modulation(0.0) >= 0.9

    def test_range_upper_bound(self) -> None:
        assert compute_trust_modulation(1.0) <= 1.1

    def test_clamped_below(self) -> None:
        result = compute_trust_modulation(-1.0)
        assert result >= 0.9

    def test_clamped_above(self) -> None:
        result = compute_trust_modulation(2.0)
        assert result <= 1.1

    def test_deterministic(self) -> None:
        r1 = compute_trust_modulation(0.73)
        r2 = compute_trust_modulation(0.73)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_temporal_self_evaluation
# ---------------------------------------------------------------------------


class TestComputeTemporalSelfEvaluation:
    """Tests for compute_temporal_self_evaluation."""

    def test_basic_output_keys(self) -> None:
        result = compute_temporal_self_evaluation([0.3, 0.4, 0.5], 0.6)
        assert "stability" in result
        assert "trend" in result
        assert "trust" in result
        assert "trust_modulation" in result

    def test_all_outputs_bounded(self) -> None:
        result = compute_temporal_self_evaluation([0.1, 0.5, 0.9], 0.7)
        assert 0.0 <= result["stability"] <= 1.0
        assert -1.0 <= result["trend"] <= 1.0
        assert 0.0 <= result["trust"] <= 1.0
        assert 0.9 <= result["trust_modulation"] <= 1.1

    def test_empty_history(self) -> None:
        result = compute_temporal_self_evaluation([], 0.5)
        assert 0.0 <= result["stability"] <= 1.0
        assert 0.0 <= result["trust"] <= 1.0

    def test_no_mutation_of_history(self) -> None:
        history = [0.2, 0.4, 0.6]
        history_copy = copy.deepcopy(history)
        compute_temporal_self_evaluation(history, 0.8)
        assert history == history_copy

    def test_deterministic(self) -> None:
        r1 = compute_temporal_self_evaluation([0.3, 0.5], 0.7)
        r2 = compute_temporal_self_evaluation([0.3, 0.5], 0.7)
        assert r1 == r2

    def test_increasing_history_positive_trend(self) -> None:
        result = compute_temporal_self_evaluation([0.1, 0.2, 0.3], 0.4)
        assert result["trend"] > 0.0

    def test_stable_history_high_stability(self) -> None:
        result = compute_temporal_self_evaluation([0.5, 0.5, 0.5], 0.5)
        assert result["stability"] == 1.0


# ---------------------------------------------------------------------------
# Integration: trust_modulation in compute_robust_score
# ---------------------------------------------------------------------------


class TestTrustModulationInRobustScore:
    """Verify trust_modulation integrates with compute_robust_score."""

    def test_default_neutral(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_without = compute_robust_score(0.5)
        score_with = compute_robust_score(0.5, trust_modulation=1.0)
        assert score_without == score_with

    def test_modulation_increases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, trust_modulation=1.0)
        score_boosted = compute_robust_score(0.5, trust_modulation=1.1)
        assert score_boosted > score_neutral

    def test_modulation_decreases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, trust_modulation=1.0)
        score_reduced = compute_robust_score(0.5, trust_modulation=0.9)
        assert score_reduced < score_neutral

    def test_result_still_bounded(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score = compute_robust_score(
            1.0, 1.0, 1.2, 1.2, 1.5, 1.0, 1.1, 1.1, 1.1,
        )
        assert 0.0 <= score <= 1.0
