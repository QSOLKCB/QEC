"""Tests for policy consistency, cycle detection & signal robustness (v99.8.0)."""

from __future__ import annotations

import copy

from qec.analysis.policy_signal_robustness import (
    clamp_modulation,
    compute_adaptive_threshold,
    compute_cycle_penalty,
    compute_geometric_mean_modulation,
    compute_oscillation_threshold,
    compute_phase_threshold,
    compute_robust_score,
    compute_robustness_diagnostics,
    detect_cycle,
    normalize_signals,
)


# ---------------------------------------------------------------------------
# 1. Cycle Detection
# ---------------------------------------------------------------------------


class TestDetectCycle:

    def test_empty_history(self):
        assert detect_cycle([]) is False

    def test_short_history(self):
        assert detect_cycle(["A", "B"]) is False

    def test_stable_repetition_not_flagged(self):
        """All-same entries are stable, not oscillatory."""
        assert detect_cycle(["A", "A", "A", "A", "A"]) is False

    def test_period_2_cycle(self):
        assert detect_cycle(["A", "B", "A", "B"]) is True

    def test_period_2_cycle_long(self):
        assert detect_cycle(["A", "B", "A", "B", "A", "B"]) is True

    def test_period_3_cycle(self):
        assert detect_cycle(["A", "B", "C", "A", "B", "C"], window=6) is True

    def test_no_cycle_varied(self):
        assert detect_cycle(["A", "B", "C", "D", "E"]) is False

    def test_window_limits_scope(self):
        # The cycle pattern is outside the window
        history = ["A", "B", "A", "B", "C", "C", "C"]
        assert detect_cycle(history, window=3) is False

    def test_window_captures_cycle(self):
        history = ["X", "A", "B", "A", "B"]
        assert detect_cycle(history, window=4) is True

    def test_deterministic(self):
        history = ["A", "B", "A", "B", "A"]
        a = detect_cycle(history)
        b = detect_cycle(history)
        assert a == b


class TestComputeCyclePenalty:

    def test_no_cycle_returns_one(self):
        assert compute_cycle_penalty(["A", "A", "A"]) == 1.0

    def test_cycle_returns_penalty(self):
        penalty = compute_cycle_penalty(["A", "B", "A", "B"])
        assert 0.8 <= penalty < 1.0

    def test_penalty_bounded(self):
        penalty = compute_cycle_penalty(["A", "B", "C", "A", "B", "C"])
        assert 0.8 <= penalty <= 1.0

    def test_empty_returns_one(self):
        assert compute_cycle_penalty([]) == 1.0

    def test_deterministic(self):
        history = ["A", "B", "A", "B"]
        a = compute_cycle_penalty(history)
        b = compute_cycle_penalty(history)
        assert a == b

    def test_no_mutation(self):
        history = ["A", "B", "A", "B"]
        original = list(history)
        compute_cycle_penalty(history)
        assert history == original


# ---------------------------------------------------------------------------
# 2. Geometric Mean Modulation
# ---------------------------------------------------------------------------


class TestGeometricMeanModulation:

    def test_missing_signal_returns_neutral(self):
        assert compute_geometric_mean_modulation(energy=0.5) == 1.0
        assert compute_geometric_mean_modulation(energy=0.5, phase=0.5) == 1.0
        assert compute_geometric_mean_modulation() == 1.0

    def test_all_perfect_signals(self):
        """energy=0, phase=1, coherence=1, alignment=1 → max modulation."""
        mod = compute_geometric_mean_modulation(
            energy=0.0, phase=1.0, coherence=1.0, alignment=1.0,
        )
        assert abs(mod - 1.5) < 1e-9

    def test_all_worst_signals(self):
        """energy=1, phase=0 → product=0 → modulation=0.5."""
        mod = compute_geometric_mean_modulation(
            energy=1.0, phase=0.0, coherence=0.0, alignment=0.0,
        )
        assert abs(mod - 0.5) < 1e-9

    def test_avoids_collapse(self):
        """Geometric mean should give better range than raw product."""
        # Raw product: 0.5 * 0.5 * 0.5 * 0.5 = 0.0625
        # Geometric mean: 0.0625 ** 0.25 = 0.5
        mod = compute_geometric_mean_modulation(
            energy=0.5, phase=0.5, coherence=0.5, alignment=0.5,
        )
        raw_product = 0.5 * 0.5 * 0.5 * 0.5
        geo_mean = raw_product ** 0.25
        expected = 0.5 + geo_mean
        assert abs(mod - expected) < 1e-9
        assert mod > 0.5 + raw_product  # geometric mean > raw product for small values

    def test_bounded(self):
        """Output always in [0.5, 1.5]."""
        for e in (0.0, 0.5, 1.0):
            for p in (0.0, 0.5, 1.0):
                for c in (0.0, 0.5, 1.0):
                    for a in (0.0, 0.5, 1.0):
                        mod = compute_geometric_mean_modulation(
                            energy=e, phase=p, coherence=c, alignment=a,
                        )
                        assert 0.5 <= mod <= 1.5, f"Out of bounds: {mod}"

    def test_deterministic(self):
        a = compute_geometric_mean_modulation(0.3, 0.7, 0.8, 0.6)
        b = compute_geometric_mean_modulation(0.3, 0.7, 0.8, 0.6)
        assert a == b


# ---------------------------------------------------------------------------
# 3. Signal Normalization
# ---------------------------------------------------------------------------


class TestNormalizeSignals:

    def test_basic_normalization(self):
        signals = {"phase": 0.2, "coherence": 0.8, "alignment": 0.5}
        result = normalize_signals(signals)
        for k in ("phase", "coherence", "alignment"):
            assert 0.0 <= result[k] <= 1.0

    def test_preserves_other_keys(self):
        signals = {"phase": 0.2, "coherence": 0.8, "alignment": 0.5, "energy": 0.9}
        result = normalize_signals(signals)
        assert result["energy"] == 0.9

    def test_single_key_no_change(self):
        signals = {"phase": 0.5}
        result = normalize_signals(signals)
        assert result["phase"] == 0.5

    def test_all_identical_signals(self):
        signals = {"phase": 0.5, "coherence": 0.5, "alignment": 0.5}
        result = normalize_signals(signals)
        for k in ("phase", "coherence", "alignment"):
            assert result[k] == 0.5

    def test_custom_keys(self):
        signals = {"x": 0.1, "y": 0.9, "z": 0.5}
        result = normalize_signals(signals, keys=["x", "y"])
        assert result["x"] == 0.0
        assert result["y"] == 1.0
        assert result["z"] == 0.5  # unchanged

    def test_no_mutation(self):
        signals = {"phase": 0.2, "coherence": 0.8, "alignment": 0.5}
        original = dict(signals)
        normalize_signals(signals)
        assert signals == original

    def test_output_bounded(self):
        signals = {"phase": -1.0, "coherence": 5.0, "alignment": 0.5}
        result = normalize_signals(signals)
        for k in ("phase", "coherence", "alignment"):
            assert 0.0 <= result[k] <= 1.0

    def test_deterministic(self):
        signals = {"phase": 0.3, "coherence": 0.7, "alignment": 0.5}
        a = normalize_signals(signals)
        b = normalize_signals(signals)
        assert a == b


# ---------------------------------------------------------------------------
# 4. Adaptive Thresholds
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:

    def test_insufficient_data_returns_fallback(self):
        assert compute_adaptive_threshold([0.1, 0.2], 75.0, 0.7) == 0.7

    def test_sufficient_data_returns_percentile(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold = compute_adaptive_threshold(values, 50.0, 0.5)
        # 50th percentile of [0.1..1.0] should be around 0.55
        assert 0.4 <= threshold <= 0.7

    def test_0th_percentile(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        threshold = compute_adaptive_threshold(values, 0.0, 0.5)
        assert abs(threshold - 0.1) < 1e-9

    def test_100th_percentile(self):
        values = [0.1, 0.5, 0.9, 0.3, 0.7]
        threshold = compute_adaptive_threshold(values, 100.0, 0.5)
        assert abs(threshold - 0.9) < 1e-9

    def test_deterministic(self):
        values = [0.3, 0.1, 0.7, 0.5, 0.2, 0.8]
        a = compute_adaptive_threshold(values, 75.0, 0.7)
        b = compute_adaptive_threshold(values, 75.0, 0.7)
        assert a == b

    def test_custom_min_samples(self):
        values = [0.1, 0.2, 0.3]
        # min_samples=3 → should compute
        threshold = compute_adaptive_threshold(values, 50.0, 0.5, min_samples=3)
        assert threshold != 0.5  # should not be fallback

    def test_no_mutation(self):
        values = [0.5, 0.3, 0.8, 0.1, 0.6]
        original = list(values)
        compute_adaptive_threshold(values, 75.0, 0.7)
        assert values == original


class TestOscillationThreshold:

    def test_fallback(self):
        assert compute_oscillation_threshold([0.1, 0.2]) == 0.7

    def test_adaptive(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold = compute_oscillation_threshold(values)
        assert threshold != 0.7  # adaptive, not fallback


class TestPhaseThreshold:

    def test_fallback(self):
        assert compute_phase_threshold([0.1, 0.2]) == 0.3

    def test_adaptive(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        threshold = compute_phase_threshold(values)
        assert threshold != 0.3  # adaptive, not fallback


# ---------------------------------------------------------------------------
# 5. Modulation Stability Clamp
# ---------------------------------------------------------------------------


class TestClampModulation:

    def test_within_range_unchanged(self):
        assert clamp_modulation(1.0) == 1.0
        assert clamp_modulation(0.8) == 0.8

    def test_clamp_high(self):
        assert clamp_modulation(2.0) == 1.4

    def test_clamp_low(self):
        assert clamp_modulation(0.3) == 0.6

    def test_boundary_values(self):
        assert clamp_modulation(0.6) == 0.6
        assert clamp_modulation(1.4) == 1.4

    def test_custom_bounds(self):
        result = clamp_modulation(0.4, inner_low=0.5, inner_high=1.5)
        assert result == 0.5


# ---------------------------------------------------------------------------
# 6. Integrated Scoring
# ---------------------------------------------------------------------------


class TestComputeRobustScore:

    def test_all_neutral(self):
        score = compute_robust_score(0.5)
        assert abs(score - 0.5) < 1e-9

    def test_with_all_factors(self):
        score = compute_robust_score(
            base_score=0.8,
            stability_weight=0.9,
            transition_bias=1.1,
            multi_step_factor=1.05,
            adaptation_modulation=1.0,
            cycle_penalty=0.95,
        )
        expected = 0.8 * 0.9 * 1.1 * 1.05 * 1.0 * 0.95
        assert abs(score - max(0.0, min(1.0, expected))) < 1e-9

    def test_clamped_to_unit(self):
        score = compute_robust_score(1.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        assert score == 1.0

    def test_clamped_non_negative(self):
        score = compute_robust_score(-1.0)
        assert score == 0.0

    def test_cycle_penalty_reduces_score(self):
        base = compute_robust_score(0.5)
        with_penalty = compute_robust_score(0.5, cycle_penalty=0.8)
        assert with_penalty < base

    def test_modulation_affects_score(self):
        neutral = compute_robust_score(0.5, adaptation_modulation=1.0)
        boosted = compute_robust_score(0.5, adaptation_modulation=1.3)
        assert boosted > neutral

    def test_deterministic(self):
        a = compute_robust_score(0.7, 0.9, 1.1, 1.05, 1.2, 0.9)
        b = compute_robust_score(0.7, 0.9, 1.1, 1.05, 1.2, 0.9)
        assert a == b

    def test_no_regressions_vs_old_formula(self):
        """The old formula without modulation/penalty should be equivalent."""
        old = 0.6 * 0.95 * 1.1 * 1.05
        new = compute_robust_score(
            0.6, 0.95, 1.1, 1.05,
            adaptation_modulation=1.0,
            cycle_penalty=1.0,
        )
        assert abs(new - max(0.0, min(1.0, old))) < 1e-9


# ---------------------------------------------------------------------------
# 7. Robustness Diagnostics
# ---------------------------------------------------------------------------


class TestComputeRobustnessDiagnostics:

    def test_returns_expected_keys(self):
        result = compute_robustness_diagnostics(["A", "B", "C"])
        assert "cycle_detected" in result
        assert "cycle_penalty" in result
        assert "modulation_raw" in result
        assert "modulation_adjusted" in result

    def test_no_cycle_diagnostics(self):
        result = compute_robustness_diagnostics(["A", "A", "A"])
        assert result["cycle_detected"] is False
        assert result["cycle_penalty"] == 1.0

    def test_cycle_diagnostics(self):
        result = compute_robustness_diagnostics(["A", "B", "A", "B"])
        assert result["cycle_detected"] is True
        assert result["cycle_penalty"] < 1.0

    def test_modulation_with_signals(self):
        result = compute_robustness_diagnostics(
            ["A", "B", "C"],
            energy=0.3, phase=0.8, coherence=0.7, alignment=0.9,
        )
        assert result["modulation_raw"] != 1.0  # signals provided
        assert 0.5 <= result["modulation_adjusted"] <= 1.5

    def test_modulation_without_signals(self):
        result = compute_robustness_diagnostics(["A", "B", "C"])
        assert result["modulation_raw"] == 1.0  # neutral fallback
        assert result["modulation_adjusted"] == 1.0

    def test_deterministic(self):
        a = compute_robustness_diagnostics(
            ["A", "B", "A", "B"], energy=0.5, phase=0.6,
            coherence=0.7, alignment=0.8,
        )
        b = compute_robustness_diagnostics(
            ["A", "B", "A", "B"], energy=0.5, phase=0.6,
            coherence=0.7, alignment=0.8,
        )
        assert a == b


# ---------------------------------------------------------------------------
# 8. Determinism stress test
# ---------------------------------------------------------------------------


class TestDeterministicRepeatability:

    def test_full_pipeline_100_runs(self):
        """Run full pipeline 100 times, verify identical results."""
        history = ["A", "B", "A", "B", "C"]
        signals = {"phase": 0.3, "coherence": 0.7, "alignment": 0.5}

        first_cycle = detect_cycle(history)
        first_penalty = compute_cycle_penalty(history)
        first_mod = compute_geometric_mean_modulation(0.4, 0.6, 0.8, 0.7)
        first_norm = normalize_signals(signals)
        first_thresh = compute_adaptive_threshold(
            [0.1, 0.3, 0.5, 0.7, 0.9], 75.0, 0.7,
        )
        first_score = compute_robust_score(0.6, 0.9, 1.1, 1.0, first_mod, first_penalty)
        first_diag = compute_robustness_diagnostics(
            history, energy=0.4, phase=0.6, coherence=0.8, alignment=0.7,
        )

        for _ in range(99):
            assert detect_cycle(history) == first_cycle
            assert compute_cycle_penalty(history) == first_penalty
            assert compute_geometric_mean_modulation(0.4, 0.6, 0.8, 0.7) == first_mod
            assert normalize_signals(signals) == first_norm
            assert compute_adaptive_threshold(
                [0.1, 0.3, 0.5, 0.7, 0.9], 75.0, 0.7,
            ) == first_thresh
            assert compute_robust_score(
                0.6, 0.9, 1.1, 1.0, first_mod, first_penalty,
            ) == first_score
            assert compute_robustness_diagnostics(
                history, energy=0.4, phase=0.6, coherence=0.8, alignment=0.7,
            ) == first_diag
