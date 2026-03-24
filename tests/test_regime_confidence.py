"""Tests for regime-aware confidence tracking — v101.3.0.

Covers:
- deterministic memory updates
- FIFO behavior
- per-regime isolation
- trust correctness
- blending correctness
- modulation bounds
- no mutation of inputs
- regime self-evaluation integration

All tests are deterministic.  No randomness, no network, no GPU.
"""

from __future__ import annotations

import copy

import pytest

from qec.analysis.regime_confidence import (
    blend_trust_signals,
    compute_regime_trust,
    compute_regime_trust_modulation,
    update_regime_confidence_history,
)
from qec.analysis.self_evaluation import compute_regime_self_evaluation


# ---------------------------------------------------------------------------
# update_regime_confidence_history
# ---------------------------------------------------------------------------


class TestUpdateRegimeConfidenceHistory:
    """Tests for update_regime_confidence_history."""

    def test_append_to_empty_memory(self) -> None:
        result = update_regime_confidence_history({}, ("stable", "basin_1"), 0.5)
        assert result == {("stable", "basin_1"): [0.5]}

    def test_append_preserves_order(self) -> None:
        memory = {("stable", "basin_1"): [0.1, 0.2]}
        result = update_regime_confidence_history(memory, ("stable", "basin_1"), 0.3)
        assert result[("stable", "basin_1")] == [0.1, 0.2, 0.3]

    def test_bounded_length(self) -> None:
        history = [float(i) / 10.0 for i in range(10)]
        memory = {("stable", "basin_1"): history}
        result = update_regime_confidence_history(
            memory, ("stable", "basin_1"), 0.99, max_len=10,
        )
        assert len(result[("stable", "basin_1")]) == 10
        assert result[("stable", "basin_1")][-1] == 0.99
        assert result[("stable", "basin_1")][0] == 0.1

    def test_bounded_length_small(self) -> None:
        memory = {("stable", "basin_1"): [0.1, 0.2, 0.3]}
        result = update_regime_confidence_history(
            memory, ("stable", "basin_1"), 0.4, max_len=3,
        )
        assert result[("stable", "basin_1")] == [0.2, 0.3, 0.4]

    def test_per_regime_isolation(self) -> None:
        memory = {("stable", "basin_1"): [0.5]}
        result = update_regime_confidence_history(
            memory, ("unstable", "basin_2"), 0.7,
        )
        assert result[("stable", "basin_1")] == [0.5]
        assert result[("unstable", "basin_2")] == [0.7]

    def test_no_mutation_of_input(self) -> None:
        original = {("stable", "basin_1"): [0.1, 0.2]}
        original_copy = copy.deepcopy(original)
        update_regime_confidence_history(original, ("stable", "basin_1"), 0.3)
        assert original == original_copy

    def test_no_mutation_of_inner_list(self) -> None:
        inner = [0.1, 0.2]
        memory = {("stable", "basin_1"): inner}
        result = update_regime_confidence_history(memory, ("stable", "basin_1"), 0.3)
        assert inner == [0.1, 0.2]  # original inner list unchanged
        assert result[("stable", "basin_1")] == [0.1, 0.2, 0.3]

    def test_deterministic(self) -> None:
        memory = {("stable", "basin_1"): [0.1]}
        r1 = update_regime_confidence_history(memory, ("stable", "basin_1"), 0.5)
        r2 = update_regime_confidence_history(memory, ("stable", "basin_1"), 0.5)
        assert r1 == r2

    def test_multiple_regimes_independent(self) -> None:
        memory: dict = {}
        memory = update_regime_confidence_history(memory, ("stable", "b1"), 0.3)
        memory = update_regime_confidence_history(memory, ("unstable", "b2"), 0.7)
        memory = update_regime_confidence_history(memory, ("stable", "b1"), 0.4)
        assert memory[("stable", "b1")] == [0.3, 0.4]
        assert memory[("unstable", "b2")] == [0.7]


# ---------------------------------------------------------------------------
# compute_regime_trust
# ---------------------------------------------------------------------------


class TestComputeRegimeTrust:
    """Tests for compute_regime_trust."""

    def test_output_keys(self) -> None:
        result = compute_regime_trust([0.3, 0.4, 0.5])
        assert "stability" in result
        assert "trend" in result
        assert "trust" in result

    def test_all_outputs_bounded(self) -> None:
        result = compute_regime_trust([0.1, 0.5, 0.9])
        assert 0.0 <= result["stability"] <= 1.0
        assert -1.0 <= result["trend"] <= 1.0
        assert 0.0 <= result["trust"] <= 1.0

    def test_empty_history(self) -> None:
        result = compute_regime_trust([])
        assert result["stability"] == 1.0
        assert result["trend"] == 0.0
        assert 0.0 <= result["trust"] <= 1.0

    def test_increasing_history_positive_trend(self) -> None:
        result = compute_regime_trust([0.1, 0.2, 0.3, 0.4])
        assert result["trend"] > 0.0

    def test_stable_history_high_stability(self) -> None:
        result = compute_regime_trust([0.5, 0.5, 0.5])
        assert result["stability"] == 1.0

    def test_deterministic(self) -> None:
        r1 = compute_regime_trust([0.3, 0.5, 0.7])
        r2 = compute_regime_trust([0.3, 0.5, 0.7])
        assert r1 == r2


# ---------------------------------------------------------------------------
# blend_trust_signals
# ---------------------------------------------------------------------------


class TestBlendTrustSignals:
    """Tests for blend_trust_signals."""

    def test_equal_weight(self) -> None:
        result = blend_trust_signals(0.8, 0.4, alpha=0.5)
        assert result == pytest.approx(0.6)

    def test_full_local(self) -> None:
        result = blend_trust_signals(0.8, 0.4, alpha=1.0)
        assert result == pytest.approx(0.4)

    def test_full_global(self) -> None:
        result = blend_trust_signals(0.8, 0.4, alpha=0.0)
        assert result == pytest.approx(0.8)

    def test_bounded_zero_one(self) -> None:
        for g in [0.0, 0.5, 1.0]:
            for l in [0.0, 0.5, 1.0]:
                for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    result = blend_trust_signals(g, l, alpha=a)
                    assert 0.0 <= result <= 1.0

    def test_clamped_inputs(self) -> None:
        result = blend_trust_signals(-0.5, 1.5, alpha=0.5)
        assert 0.0 <= result <= 1.0

    def test_deterministic(self) -> None:
        r1 = blend_trust_signals(0.7, 0.3, alpha=0.6)
        r2 = blend_trust_signals(0.7, 0.3, alpha=0.6)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_regime_trust_modulation
# ---------------------------------------------------------------------------


class TestComputeRegimeTrustModulation:
    """Tests for compute_regime_trust_modulation."""

    def test_zero_trust(self) -> None:
        assert compute_regime_trust_modulation(0.0) == pytest.approx(0.9)

    def test_full_trust(self) -> None:
        assert compute_regime_trust_modulation(1.0) == pytest.approx(1.1)

    def test_half_trust_neutral(self) -> None:
        assert compute_regime_trust_modulation(0.5) == pytest.approx(1.0)

    def test_range_lower_bound(self) -> None:
        assert compute_regime_trust_modulation(0.0) >= 0.9

    def test_range_upper_bound(self) -> None:
        assert compute_regime_trust_modulation(1.0) <= 1.1

    def test_clamped_below(self) -> None:
        result = compute_regime_trust_modulation(-1.0)
        assert result >= 0.9

    def test_clamped_above(self) -> None:
        result = compute_regime_trust_modulation(2.0)
        assert result <= 1.1

    def test_deterministic(self) -> None:
        r1 = compute_regime_trust_modulation(0.73)
        r2 = compute_regime_trust_modulation(0.73)
        assert r1 == r2


# ---------------------------------------------------------------------------
# compute_regime_self_evaluation
# ---------------------------------------------------------------------------


class TestComputeRegimeSelfEvaluation:
    """Tests for compute_regime_self_evaluation."""

    def test_output_keys(self) -> None:
        memory = {("stable", "basin_1"): [0.3, 0.5, 0.7]}
        result = compute_regime_self_evaluation(
            ("stable", "basin_1"), memory, 0.8,
        )
        assert "local_trust" in result
        assert "global_trust" in result
        assert "blended_trust" in result
        assert "regime_trust_modulation" in result

    def test_all_outputs_bounded(self) -> None:
        memory = {("stable", "basin_1"): [0.1, 0.5, 0.9]}
        result = compute_regime_self_evaluation(
            ("stable", "basin_1"), memory, 0.6,
        )
        assert 0.0 <= result["local_trust"] <= 1.0
        assert 0.0 <= result["global_trust"] <= 1.0
        assert 0.0 <= result["blended_trust"] <= 1.0
        assert 0.9 <= result["regime_trust_modulation"] <= 1.1

    def test_empty_memory_for_regime(self) -> None:
        result = compute_regime_self_evaluation(
            ("unknown", "basin_x"), {}, 0.5,
        )
        assert 0.0 <= result["local_trust"] <= 1.0
        assert 0.0 <= result["blended_trust"] <= 1.0
        assert 0.9 <= result["regime_trust_modulation"] <= 1.1

    def test_no_mutation_of_memory(self) -> None:
        memory = {("stable", "basin_1"): [0.3, 0.5]}
        memory_copy = copy.deepcopy(memory)
        compute_regime_self_evaluation(("stable", "basin_1"), memory, 0.7)
        assert memory == memory_copy

    def test_deterministic(self) -> None:
        memory = {("stable", "basin_1"): [0.2, 0.4, 0.6]}
        r1 = compute_regime_self_evaluation(("stable", "basin_1"), memory, 0.8)
        r2 = compute_regime_self_evaluation(("stable", "basin_1"), memory, 0.8)
        assert r1 == r2

    def test_global_trust_passthrough(self) -> None:
        memory = {("stable", "basin_1"): [0.5]}
        result = compute_regime_self_evaluation(
            ("stable", "basin_1"), memory, 0.88,
        )
        assert result["global_trust"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Integration: regime_trust_modulation in compute_robust_score
# ---------------------------------------------------------------------------


class TestRegimeTrustModulationInRobustScore:
    """Verify regime_trust_modulation integrates with compute_robust_score."""

    def test_default_neutral(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_without = compute_robust_score(0.5)
        score_with = compute_robust_score(0.5, regime_trust_modulation=1.0)
        assert score_without == score_with

    def test_modulation_increases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, regime_trust_modulation=1.0)
        score_boosted = compute_robust_score(0.5, regime_trust_modulation=1.1)
        assert score_boosted > score_neutral

    def test_modulation_decreases_score(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score_neutral = compute_robust_score(0.5, regime_trust_modulation=1.0)
        score_reduced = compute_robust_score(0.5, regime_trust_modulation=0.9)
        assert score_reduced < score_neutral

    def test_result_still_bounded(self) -> None:
        from qec.analysis.policy_signal_robustness import compute_robust_score
        score = compute_robust_score(
            1.0, 1.0, 1.2, 1.2, 1.5, 1.0, 1.1, 1.1, 1.1, 1.1,
        )
        assert 0.0 <= score <= 1.0
