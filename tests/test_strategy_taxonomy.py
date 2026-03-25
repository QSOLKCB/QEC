"""Tests for strategy taxonomy classification (v102.3.0).

Verifies:
- deterministic classification
- correct type assignment for each taxonomy type
- confidence bounds [0, 1]
- edge cases (flat strategies, noisy improvers, unstable declines)
- integration via run_taxonomy_analysis
"""

from __future__ import annotations

import copy

from qec.analysis.strategy_taxonomy import classify_strategy_type


# ---------------------------------------------------------------------------
# Helper to build metrics/regimes for a single strategy
# ---------------------------------------------------------------------------

def _single(
    name: str,
    stability: float,
    trend: float,
    oscillation: int,
    regime: str,
):
    metrics = {
        name: {
            "mean_score": 0.5,
            "variance_score": 0.0,
            "stability": stability,
            "trend": trend,
            "oscillation": oscillation,
        },
    }
    regimes = {name: regime}
    return metrics, regimes


# ---------------------------------------------------------------------------
# Type assignment tests
# ---------------------------------------------------------------------------


class TestClassifyStrategyType:
    """Correct type assignment for each taxonomy type."""

    def test_stable_core(self):
        metrics, regimes = _single("s", 0.95, 0.01, 0, "stable")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "stable_core"

    def test_steady_improver(self):
        metrics, regimes = _single("s", 0.85, 0.1, 0, "improving")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "steady_improver"

    def test_volatile_improver(self):
        metrics, regimes = _single("s", 0.75, 0.1, 1, "improving")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "volatile_improver"

    def test_oscillatory(self):
        metrics, regimes = _single("s", 0.5, 0.0, 3, "oscillatory")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "oscillatory"

    def test_degrading(self):
        metrics, regimes = _single("s", 0.8, -0.1, 0, "declining")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "degrading"

    def test_unstable_decliner(self):
        metrics, regimes = _single("s", 0.5, -0.2, 0, "declining")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "unstable_decliner"

    def test_transitional(self):
        metrics, regimes = _single("s", 0.6, 0.02, 0, "transitional")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["type"] == "transitional"


# ---------------------------------------------------------------------------
# Confidence score tests
# ---------------------------------------------------------------------------


class TestConfidence:
    """Confidence = min(1.0, stability + abs(trend)), rounded to 12 decimals."""

    def test_confidence_basic(self):
        metrics, regimes = _single("s", 0.8, 0.1, 0, "improving")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["confidence"] == round(min(1.0, 0.8 + 0.1), 12)

    def test_confidence_capped_at_one(self):
        metrics, regimes = _single("s", 0.95, 0.1, 0, "stable")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["confidence"] == 1.0

    def test_confidence_with_negative_trend(self):
        metrics, regimes = _single("s", 0.5, -0.2, 0, "declining")
        result = classify_strategy_type(metrics, regimes)
        assert result["s"]["confidence"] == round(min(1.0, 0.5 + 0.2), 12)

    def test_confidence_bounds(self):
        """Confidence must always be in [0, 1]."""
        for stab in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for trend in [-0.5, -0.1, 0.0, 0.1, 0.5]:
                metrics, regimes = _single("s", stab, trend, 0, "transitional")
                result = classify_strategy_type(metrics, regimes)
                c = result["s"]["confidence"]
                assert 0.0 <= c <= 1.0, f"confidence={c} for stab={stab}, trend={trend}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases from the task specification."""

    def test_flat_strategy_is_stable_core(self):
        """Flat strategies (zero trend, high stability) → stable_core."""
        metrics, regimes = _single("flat", 0.99, 0.0, 0, "stable")
        result = classify_strategy_type(metrics, regimes)
        assert result["flat"]["type"] == "stable_core"

    def test_noisy_improver_is_volatile_improver(self):
        """Noisy improvers (improving regime + oscillation) → volatile_improver."""
        metrics, regimes = _single("noisy", 0.75, 0.08, 2, "improving")
        result = classify_strategy_type(metrics, regimes)
        assert result["noisy"]["type"] == "volatile_improver"

    def test_unstable_decline(self):
        """Unstable declines (declining + low stability) → unstable_decliner."""
        metrics, regimes = _single("bad", 0.4, -0.3, 1, "declining")
        result = classify_strategy_type(metrics, regimes)
        assert result["bad"]["type"] == "unstable_decliner"

    def test_missing_regime_defaults_to_transitional(self):
        """Strategy not in regimes dict → transitional."""
        metrics = {
            "orphan": {
                "mean_score": 0.5,
                "variance_score": 0.0,
                "stability": 0.5,
                "trend": 0.0,
                "oscillation": 0,
            },
        }
        regimes = {}  # no entry for "orphan"
        result = classify_strategy_type(metrics, regimes)
        assert result["orphan"]["type"] == "transitional"

    def test_every_strategy_gets_a_type(self):
        """All strategies in metrics dict get a type assigned."""
        names = ["a", "b", "c", "d"]
        metrics = {}
        regimes = {}
        for i, name in enumerate(names):
            metrics[name] = {
                "mean_score": 0.5,
                "variance_score": 0.0,
                "stability": 0.5 + i * 0.1,
                "trend": 0.0,
                "oscillation": 0,
            }
            regimes[name] = "transitional"
        result = classify_strategy_type(metrics, regimes)
        assert set(result.keys()) == set(names)
        for name in names:
            assert "type" in result[name]
            assert "confidence" in result[name]


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Classification must be deterministic and side-effect free."""

    def test_deterministic_repeated_calls(self):
        metrics, regimes = _single("s", 0.85, 0.1, 0, "improving")
        r1 = classify_strategy_type(metrics, regimes)
        r2 = classify_strategy_type(metrics, regimes)
        assert r1 == r2

    def test_no_mutation_of_inputs(self):
        metrics, regimes = _single("s", 0.85, 0.1, 0, "improving")
        metrics_copy = copy.deepcopy(metrics)
        regimes_copy = copy.deepcopy(regimes)
        classify_strategy_type(metrics, regimes)
        assert metrics == metrics_copy
        assert regimes == regimes_copy

    def test_sorted_output_order(self):
        """Output keys are sorted alphabetically."""
        metrics = {}
        regimes = {}
        for name in ["z_strat", "a_strat", "m_strat"]:
            metrics[name] = {
                "mean_score": 0.5,
                "variance_score": 0.0,
                "stability": 0.95,
                "trend": 0.0,
                "oscillation": 0,
            }
            regimes[name] = "stable"
        result = classify_strategy_type(metrics, regimes)
        assert list(result.keys()) == ["a_strat", "m_strat", "z_strat"]


# ---------------------------------------------------------------------------
# Integration: run_taxonomy_analysis
# ---------------------------------------------------------------------------


class TestRunTaxonomyAnalysis:
    """Integration test for the full taxonomy pipeline."""

    def test_full_pipeline(self):
        from qec.analysis.strategy_adapter import run_taxonomy_analysis

        runs = [
            {
                "strategies": [
                    {"name": "alpha", "metrics": {"design_score": 0.8}},
                    {"name": "beta", "metrics": {"design_score": 0.5}},
                ],
            },
            {
                "strategies": [
                    {"name": "alpha", "metrics": {"design_score": 0.8}},
                    {"name": "beta", "metrics": {"design_score": 0.5}},
                ],
            },
        ]
        result = run_taxonomy_analysis(runs)

        assert "history" in result
        assert "trajectory_metrics" in result
        assert "regimes" in result
        assert "taxonomy" in result

        for name in ["alpha", "beta"]:
            assert name in result["taxonomy"]
            assert "type" in result["taxonomy"][name]
            assert "confidence" in result["taxonomy"][name]

    def test_deterministic(self):
        from qec.analysis.strategy_adapter import run_taxonomy_analysis

        runs = [
            {
                "strategies": [
                    {"name": "x", "metrics": {"design_score": 0.6}},
                ],
            },
            {
                "strategies": [
                    {"name": "x", "metrics": {"design_score": 0.7}},
                ],
            },
        ]
        r1 = run_taxonomy_analysis(runs)
        r2 = run_taxonomy_analysis(runs)
        assert r1 == r2

    def test_no_mutation(self):
        from qec.analysis.strategy_adapter import run_taxonomy_analysis

        runs = [
            {
                "strategies": [
                    {"name": "x", "metrics": {"design_score": 0.6}},
                ],
            },
        ]
        runs_copy = copy.deepcopy(runs)
        run_taxonomy_analysis(runs)
        assert runs == runs_copy


# ---------------------------------------------------------------------------
# Format tests
# ---------------------------------------------------------------------------


class TestFormatTaxonomySummary:
    """Tests for format_taxonomy_summary."""

    def test_format_output(self):
        from qec.analysis.strategy_adapter import format_taxonomy_summary

        result = {
            "taxonomy": {
                "alpha": {"type": "stable_core", "confidence": 0.95},
                "beta": {"type": "degrading", "confidence": 0.7},
            },
        }
        text = format_taxonomy_summary(result)
        assert "=== Strategy Taxonomy ===" in text
        assert "Strategy: alpha" in text
        assert "Type: stable_core" in text
        assert "Confidence: 0.95" in text
        assert "Strategy: beta" in text
        assert "Type: degrading" in text

    def test_empty_result(self):
        from qec.analysis.strategy_adapter import format_taxonomy_summary

        text = format_taxonomy_summary({})
        assert "=== Strategy Taxonomy ===" in text
