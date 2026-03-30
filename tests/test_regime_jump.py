"""Tests for regime jump and parity coherence analysis."""

from qec.analysis.regime_jump import (
    classify_regime_behavior,
    compute_coherence_length,
    compute_regime_jump,
    run_regime_jump_analysis,
)


class TestComputeRegimeJump:
    def test_no_jump(self):
        result = compute_regime_jump("stable", "stable")
        assert result["jump_detected"] is False
        assert result["jump_type"] == "none"

    def test_rising_jump(self):
        result = compute_regime_jump("stable", "rising")
        assert result["jump_detected"] is True
        assert result["jump_type"] == "stable_to_rising"

    def test_falling_to_stable(self):
        result = compute_regime_jump("falling", "stable")
        assert result["jump_detected"] is True
        assert result["jump_type"] == "falling_to_stable"

    def test_rising_to_falling(self):
        result = compute_regime_jump("rising", "falling")
        assert result["jump_detected"] is True
        assert result["jump_type"] == "rising_to_falling"


class TestComputeCoherenceLength:
    def test_empty_history(self):
        assert compute_coherence_length([]) == 0

    def test_all_same(self):
        assert compute_coherence_length(["rising", "rising", "rising"]) == 3

    def test_mixed_trailing(self):
        assert compute_coherence_length(["stable", "rising", "rising", "rising"]) == 3

    def test_single_element(self):
        assert compute_coherence_length(["stable"]) == 1

    def test_no_trailing_run(self):
        assert compute_coherence_length(["stable", "rising", "stable"]) == 1


class TestClassifyRegimeBehavior:
    def test_transition(self):
        assert classify_regime_behavior(True, 0) == "transition"

    def test_transition_overrides_coherence(self):
        assert classify_regime_behavior(True, 5) == "transition"

    def test_locked(self):
        assert classify_regime_behavior(False, 3) == "locked"

    def test_locked_high(self):
        assert classify_regime_behavior(False, 10) == "locked"

    def test_oscillatory(self):
        assert classify_regime_behavior(False, 1) == "oscillatory"

    def test_oscillatory_two(self):
        assert classify_regime_behavior(False, 2) == "oscillatory"


class TestRunRegimeJumpAnalysis:
    def test_no_jump_locked(self):
        result = run_regime_jump_analysis(
            "rising", "rising", ["rising", "rising", "rising"]
        )
        assert result["jump_detected"] is False
        assert result["jump_type"] == "none"
        assert result["coherence_length"] == 3
        assert result["regime_behavior"] == "locked"

    def test_jump_transition(self):
        result = run_regime_jump_analysis(
            "stable", "rising", ["stable", "stable", "rising"]
        )
        assert result["jump_detected"] is True
        assert result["jump_type"] == "stable_to_rising"
        assert result["coherence_length"] == 1
        assert result["regime_behavior"] == "transition"

    def test_oscillatory(self):
        result = run_regime_jump_analysis(
            "stable", "stable", ["stable", "rising", "stable"]
        )
        assert result["jump_detected"] is False
        assert result["jump_type"] == "none"
        assert result["coherence_length"] == 1
        assert result["regime_behavior"] == "oscillatory"

    def test_determinism(self):
        """Repeated runs must produce identical results."""
        args = ("stable", "rising", ["stable", "stable", "rising"])
        results = [run_regime_jump_analysis(*args) for _ in range(10)]
        assert all(r == results[0] for r in results)
