"""Tests for regime history window analysis."""

from src.qec.analysis.regime_history import (
    classify_history_behavior,
    compute_persistence_ratio,
    compute_state_counts,
    compute_transition_count,
    run_regime_history_analysis,
)


class TestEmpty:
    def test_state_counts_empty(self):
        assert compute_state_counts([]) == {}

    def test_persistence_ratio_empty(self):
        assert compute_persistence_ratio([], "stable") == 0.0

    def test_transition_count_empty(self):
        assert compute_transition_count([]) == 0

    def test_classify_empty(self):
        assert classify_history_behavior(0, 0.0) == "stable_window"

    def test_run_analysis_empty(self):
        result = run_regime_history_analysis([])
        assert result["state_counts"] == {}
        assert result["oscillation_ratio"] == 0.0
        assert result["transition_count"] == 0
        assert result["history_behavior"] == "stable_window"


class TestStableWindow:
    def test_stable_counts(self):
        h = ["stable", "stable", "stable"]
        assert compute_state_counts(h) == {"stable": 3}

    def test_stable_transitions(self):
        assert compute_transition_count(["stable", "stable", "stable"]) == 0

    def test_stable_behavior(self):
        result = run_regime_history_analysis(["stable", "stable", "stable"])
        assert result["transition_count"] == 0
        assert result["history_behavior"] == "stable_window"


class TestPersistentOscillation:
    def test_oscillation_ratio(self):
        h = ["oscillatory", "oscillatory", "stable", "oscillatory"]
        ratio = compute_persistence_ratio(h, "oscillatory")
        assert ratio >= 0.5

    def test_oscillation_behavior(self):
        h = ["oscillatory", "oscillatory", "stable", "oscillatory"]
        result = run_regime_history_analysis(h)
        assert result["history_behavior"] == "persistent_oscillation"


class TestUnstableTransitions:
    def test_transition_count(self):
        h = ["stable", "rising", "falling", "stable"]
        assert compute_transition_count(h) == 3

    def test_unstable_behavior(self):
        h = ["stable", "rising", "falling", "stable"]
        result = run_regime_history_analysis(h)
        assert result["history_behavior"] == "unstable"


class TestDeterminism:
    def test_repeated_runs_identical(self):
        h = ["stable", "rising", "oscillatory", "rising", "stable"]
        results = [run_regime_history_analysis(h) for _ in range(10)]
        for r in results[1:]:
            assert r == results[0]

    def test_state_counts_key_order(self):
        h = ["rising", "stable", "oscillatory"]
        counts = compute_state_counts(h)
        assert list(counts.keys()) == ["oscillatory", "rising", "stable"]
