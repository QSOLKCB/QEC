"""Tests for v102.2.0 trajectory analysis: history, metrics, regime classification.

Verifies:
- deterministic outputs
- correct variance/stability computation
- correct regime classification
- no mutation of inputs
"""

from __future__ import annotations

import copy

from qec.analysis.regime_classification import classify_regime
from qec.analysis.strategy_adapter import (
    format_trajectory_summary,
    run_trajectory_analysis,
)
from qec.analysis.strategy_history import build_strategy_history
from qec.analysis.trajectory_metrics import compute_trajectory_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy(name: str, design_score: float, **extra_metrics: float) -> dict:
    metrics = {"design_score": design_score}
    metrics.update(extra_metrics)
    return {"name": name, "metrics": metrics}


def _make_runs(*score_sequences: list[float]) -> list[dict]:
    """Build runs where each run has one strategy 'A' with given scores.

    score_sequences[i] is a list of scores for each strategy in run i.
    For simplicity, creates strategies named 'A', 'B', ... for each entry.
    """
    runs = []
    for scores in score_sequences:
        strategies = []
        for idx, score in enumerate(scores):
            name = chr(ord("A") + idx)
            strategies.append(_make_strategy(name, score))
        runs.append({"strategies": strategies})
    return runs


# ---------------------------------------------------------------------------
# build_strategy_history
# ---------------------------------------------------------------------------


class TestBuildStrategyHistory:

    def test_basic_history(self):
        runs = [
            {"strategies": [_make_strategy("X", 0.5), _make_strategy("Y", 0.6)]},
            {"strategies": [_make_strategy("X", 0.7), _make_strategy("Y", 0.8)]},
        ]
        h = build_strategy_history(runs)
        assert sorted(h.keys()) == ["X", "Y"]
        assert h["X"]["design_score"] == [0.5, 0.7]
        assert h["Y"]["design_score"] == [0.6, 0.8]

    def test_missing_strategy_skipped(self):
        runs = [
            {"strategies": [_make_strategy("A", 0.5)]},
            {"strategies": [_make_strategy("B", 0.6)]},
        ]
        h = build_strategy_history(runs)
        assert h["A"]["design_score"] == [0.5]
        assert h["B"]["design_score"] == [0.6]

    def test_empty_runs(self):
        h = build_strategy_history([])
        assert h == {}

    def test_deterministic_ordering(self):
        runs = [
            {"strategies": [
                _make_strategy("Z", 0.1),
                _make_strategy("A", 0.2),
                _make_strategy("M", 0.3),
            ]},
        ]
        h = build_strategy_history(runs)
        assert list(h.keys()) == ["A", "M", "Z"]

    def test_no_mutation(self):
        runs = [
            {"strategies": [_make_strategy("A", 0.5)]},
        ]
        original = copy.deepcopy(runs)
        build_strategy_history(runs)
        assert runs == original

    def test_deterministic_repeated_calls(self):
        runs = [
            {"strategies": [_make_strategy("A", 0.5), _make_strategy("B", 0.6)]},
            {"strategies": [_make_strategy("A", 0.7)]},
        ]
        h1 = build_strategy_history(runs)
        h2 = build_strategy_history(runs)
        assert h1 == h2

    def test_extra_metrics_tracked(self):
        runs = [
            {"strategies": [
                _make_strategy("A", 0.5, confidence_efficiency=0.8, consistency_gap=0.1),
            ]},
        ]
        h = build_strategy_history(runs)
        assert h["A"]["confidence_efficiency"] == [0.8]
        assert h["A"]["consistency_gap"] == [0.1]


# ---------------------------------------------------------------------------
# compute_trajectory_metrics
# ---------------------------------------------------------------------------


class TestTrajectoryMetrics:

    def test_stable_strategy(self):
        history = {"S": {"design_score": [0.5, 0.5, 0.5]}}
        m = compute_trajectory_metrics(history)
        assert m["S"]["mean_score"] == 0.5
        assert m["S"]["variance_score"] == 0.0
        assert m["S"]["stability"] == 1.0
        assert m["S"]["trend"] == 0.0
        assert m["S"]["oscillation"] == 0

    def test_trending_strategy(self):
        history = {"T": {"design_score": [0.1, 0.3, 0.5]}}
        m = compute_trajectory_metrics(history)
        assert m["T"]["mean_score"] == 0.3
        assert m["T"]["trend"] == 0.4
        assert m["T"]["oscillation"] == 0

    def test_oscillating_strategy(self):
        history = {"O": {"design_score": [0.5, 0.8, 0.3, 0.7, 0.2]}}
        m = compute_trajectory_metrics(history)
        # Deltas: +0.3, -0.5, +0.4, -0.5 → sign changes at positions 0-1, 1-2, 2-3 = 3
        assert m["O"]["oscillation"] == 3

    def test_single_value(self):
        history = {"X": {"design_score": [0.5]}}
        m = compute_trajectory_metrics(history)
        assert m["X"]["mean_score"] == 0.5
        assert m["X"]["variance_score"] == 0.0
        assert m["X"]["stability"] == 1.0
        assert m["X"]["trend"] == 0.0
        assert m["X"]["oscillation"] == 0

    def test_empty_scores(self):
        history = {"E": {"design_score": []}}
        m = compute_trajectory_metrics(history)
        assert m["E"]["mean_score"] == 0.0
        assert m["E"]["variance_score"] == 0.0
        assert m["E"]["stability"] == 1.0
        assert m["E"]["trend"] == 0.0
        assert m["E"]["oscillation"] == 0

    def test_variance_computation(self):
        # Values: [0.2, 0.8], mean=0.5, variance=((0.3)^2 + (0.3)^2)/2 = 0.09
        history = {"V": {"design_score": [0.2, 0.8]}}
        m = compute_trajectory_metrics(history)
        assert abs(m["V"]["variance_score"] - 0.09) < 1e-10
        expected_stability = round(1.0 / (1.0 + 0.09), 12)
        assert m["V"]["stability"] == expected_stability

    def test_no_mutation(self):
        history = {"A": {"design_score": [0.5, 0.7]}}
        original = copy.deepcopy(history)
        compute_trajectory_metrics(history)
        assert history == original

    def test_deterministic(self):
        history = {"A": {"design_score": [0.1, 0.5, 0.3]}}
        m1 = compute_trajectory_metrics(history)
        m2 = compute_trajectory_metrics(history)
        assert m1 == m2

    def test_flat_sequence_zero_oscillation(self):
        """Flat sequences must never register oscillation."""
        history = {"F": {"design_score": [0.5, 0.5, 0.5, 0.5, 0.5]}}
        m = compute_trajectory_metrics(history)
        assert m["F"]["oscillation"] == 0

    def test_zero_delta_filtered_from_oscillation(self):
        """Zero deltas between non-zero deltas should not cause extra sign changes."""
        # Scores: 0.3, 0.5, 0.5, 0.3
        # Deltas: +0.2, 0.0, -0.2
        # Non-zero deltas: +0.2, -0.2 → 1 sign change
        history = {"Z": {"design_score": [0.3, 0.5, 0.5, 0.3]}}
        m = compute_trajectory_metrics(history)
        assert m["Z"]["oscillation"] == 1

    def test_two_identical_values(self):
        """Two identical values: no trend, no oscillation, zero variance."""
        history = {"D": {"design_score": [0.7, 0.7]}}
        m = compute_trajectory_metrics(history)
        assert m["D"]["mean_score"] == 0.7
        assert m["D"]["variance_score"] == 0.0
        assert m["D"]["stability"] == 1.0
        assert m["D"]["trend"] == 0.0
        assert m["D"]["oscillation"] == 0

    def test_plateau_with_single_change(self):
        """Plateau broken by a single rise should not oscillate."""
        # Scores: 0.5, 0.5, 0.5, 0.7
        # Deltas: 0.0, 0.0, +0.2
        # Non-zero deltas: [+0.2] → length < 2 → oscillation = 0
        history = {"P": {"design_score": [0.5, 0.5, 0.5, 0.7]}}
        m = compute_trajectory_metrics(history)
        assert m["P"]["oscillation"] == 0


# ---------------------------------------------------------------------------
# classify_regime
# ---------------------------------------------------------------------------


class TestRegimeClassification:

    def test_stable_regime(self):
        metrics = {"S": {"stability": 0.95, "trend": 0.01, "oscillation": 0}}
        r = classify_regime(metrics)
        assert r["S"] == "stable"

    def test_improving_regime(self):
        metrics = {"I": {"stability": 0.8, "trend": 0.1, "oscillation": 0}}
        r = classify_regime(metrics)
        assert r["I"] == "improving"

    def test_declining_regime(self):
        metrics = {"D": {"stability": 0.8, "trend": -0.1, "oscillation": 0}}
        r = classify_regime(metrics)
        assert r["D"] == "declining"

    def test_oscillatory_regime(self):
        metrics = {"O": {"stability": 0.5, "trend": 0.0, "oscillation": 3}}
        r = classify_regime(metrics)
        assert r["O"] == "oscillatory"

    def test_transitional_regime(self):
        metrics = {"T": {"stability": 0.6, "trend": 0.02, "oscillation": 1}}
        r = classify_regime(metrics)
        assert r["T"] == "transitional"

    def test_stable_takes_priority_over_oscillatory(self):
        # Stable check comes first
        metrics = {"X": {"stability": 0.95, "trend": 0.01, "oscillation": 5}}
        r = classify_regime(metrics)
        assert r["X"] == "stable"

    def test_improving_takes_priority_over_oscillatory(self):
        metrics = {"X": {"stability": 0.8, "trend": 0.1, "oscillation": 3}}
        r = classify_regime(metrics)
        assert r["X"] == "improving"

    def test_deterministic(self):
        metrics = {
            "A": {"stability": 0.95, "trend": 0.0, "oscillation": 0},
            "B": {"stability": 0.5, "trend": -0.2, "oscillation": 4},
        }
        r1 = classify_regime(metrics)
        r2 = classify_regime(metrics)
        assert r1 == r2

    def test_no_mutation(self):
        metrics = {"A": {"stability": 0.95, "trend": 0.0, "oscillation": 0}}
        original = copy.deepcopy(metrics)
        classify_regime(metrics)
        assert metrics == original


# ---------------------------------------------------------------------------
# Integration: run_trajectory_analysis
# ---------------------------------------------------------------------------


class TestRunTrajectoryAnalysis:

    def test_full_pipeline(self):
        runs = [
            {"strategies": [
                _make_strategy("A", 0.5),
                _make_strategy("B", 0.8),
            ]},
            {"strategies": [
                _make_strategy("A", 0.5),
                _make_strategy("B", 0.7),
            ]},
            {"strategies": [
                _make_strategy("A", 0.5),
                _make_strategy("B", 0.9),
            ]},
        ]
        result = run_trajectory_analysis(runs)

        assert "history" in result
        assert "trajectory_metrics" in result
        assert "regimes" in result

        # A is stable (constant 0.5)
        assert result["regimes"]["A"] == "stable"

        # B has trajectory: [0.8, 0.7, 0.9]
        tm = result["trajectory_metrics"]["B"]
        assert tm["trend"] == round(0.9 - 0.8, 12)
        assert tm["oscillation"] == 1  # deltas: -0.1, +0.2 → 1 sign change

    def test_deterministic(self):
        runs = [
            {"strategies": [_make_strategy("X", 0.5)]},
            {"strategies": [_make_strategy("X", 0.6)]},
        ]
        r1 = run_trajectory_analysis(runs)
        r2 = run_trajectory_analysis(runs)
        assert r1 == r2

    def test_no_mutation(self):
        runs = [
            {"strategies": [_make_strategy("A", 0.5)]},
        ]
        original = copy.deepcopy(runs)
        run_trajectory_analysis(runs)
        assert runs == original


# ---------------------------------------------------------------------------
# format_trajectory_summary
# ---------------------------------------------------------------------------


class TestFormatTrajectorySummary:

    def test_format_output(self):
        result = {
            "trajectory_metrics": {
                "A": {
                    "mean_score": 0.5,
                    "variance_score": 0.0,
                    "stability": 1.0,
                    "trend": 0.0,
                    "oscillation": 0,
                },
            },
            "regimes": {"A": "stable"},
        }
        text = format_trajectory_summary(result)
        assert "=== Trajectory Summary ===" in text
        assert "Strategy: A" in text
        assert "Mean: 0.500000" in text
        assert "Stability: 1.000000" in text
        assert "Regime: stable" in text

    def test_empty_result(self):
        result = {"trajectory_metrics": {}, "regimes": {}}
        text = format_trajectory_summary(result)
        assert "=== Trajectory Summary ===" in text
