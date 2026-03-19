"""Tests for sonification interpretation layer (v72.4.0)."""

import copy
import math

from src.qec.experiments.sonification_interpretation import (
    interpret_sonification_comparison,
)


def _make_comparison(**overrides):
    """Build a valid comparison dict with sensible defaults."""
    base = {
        "baseline_silence_fidelity": 1.0,
        "multidim_silence_fidelity": 1.0,
        "leakage_rate": 0.0,
        "baseline_energy": 0.5,
        "multidim_ch0_energy": 0.4,
        "multidim_ch1_energy": 0.6,
        "channel_correlation": 0.1,
        "baseline_variance": 0.02,
        "multidim_variance": 0.03,
    }
    base.update(overrides)
    return base


class TestDeterminism:
    def test_same_input_same_output(self):
        comp = _make_comparison()
        r1 = interpret_sonification_comparison(comp)
        r2 = interpret_sonification_comparison(comp)
        assert r1 == r2

    def test_determinism_across_calls(self):
        for _ in range(5):
            comp = _make_comparison()
            r = interpret_sonification_comparison(comp)
            assert r["composite_score"] == r["composite_score"]


class TestNoMutation:
    def test_input_not_mutated(self):
        comp = _make_comparison()
        original = copy.deepcopy(comp)
        interpret_sonification_comparison(comp)
        assert comp == original


class TestScoreBounds:
    def test_all_scores_in_unit_interval(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        for name, val in result["scores"].items():
            assert 0.0 <= val <= 1.0, f"score {name} = {val} out of bounds"

    def test_composite_in_unit_interval(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        assert 0.0 <= result["composite_score"] <= 1.0

    def test_extreme_values_still_bounded(self):
        comp = _make_comparison(
            baseline_energy=1e-10,
            multidim_ch0_energy=100.0,
            multidim_ch1_energy=100.0,
            channel_correlation=-1.0,
            baseline_variance=0.0,
            multidim_variance=1e6,
        )
        result = interpret_sonification_comparison(comp)
        for name, val in result["scores"].items():
            assert 0.0 <= val <= 1.0, f"score {name} = {val} out of bounds"
        assert 0.0 <= result["composite_score"] <= 1.0


class TestVerdictInvalid:
    def test_silence_break_baseline(self):
        comp = _make_comparison(baseline_silence_fidelity=0.99)
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "invalid"

    def test_silence_break_multidim(self):
        comp = _make_comparison(multidim_silence_fidelity=0.95)
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "invalid"

    def test_leakage_nonzero(self):
        comp = _make_comparison(leakage_rate=0.01)
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "invalid"


class TestVerdictMultidimImproves:
    def test_low_correlation_positive_variance(self):
        comp = _make_comparison(
            channel_correlation=0.1,
            baseline_variance=0.01,
            multidim_variance=0.05,
        )
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "multidim_improves_structure"


class TestVerdictRedundant:
    def test_high_correlation(self):
        comp = _make_comparison(
            channel_correlation=0.95,
            baseline_variance=0.02,
            multidim_variance=0.03,
        )
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "channels_redundant"


class TestVerdictBaselineMoreStable:
    def test_lower_multidim_variance(self):
        comp = _make_comparison(
            channel_correlation=0.5,
            baseline_variance=0.05,
            multidim_variance=0.01,
        )
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "baseline_more_stable"


class TestVerdictTradeoff:
    def test_tradeoff_case(self):
        # correlation_score = 1 - 0.5 = 0.5 (not > 0.5, not < 0.2)
        # variance_score = 0 (delta = 0)
        comp = _make_comparison(
            channel_correlation=0.5,
            baseline_variance=0.02,
            multidim_variance=0.02,
        )
        result = interpret_sonification_comparison(comp)
        assert result["verdict"] == "tradeoff"


class TestZeroEdgeCases:
    def test_zero_baseline_energy(self):
        comp = _make_comparison(baseline_energy=0.0)
        result = interpret_sonification_comparison(comp)
        assert result["derived"]["energy_ratio"] == 1.0
        assert math.isfinite(result["composite_score"])

    def test_zero_variance(self):
        comp = _make_comparison(baseline_variance=0.0, multidim_variance=0.0)
        result = interpret_sonification_comparison(comp)
        assert result["derived"]["variance_delta"] == 0.0
        assert result["scores"]["variance"] == 0.0
        assert math.isfinite(result["composite_score"])


class TestFiniteness:
    def test_all_outputs_finite(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        for val in result["scores"].values():
            assert math.isfinite(val)
        for val in result["derived"].values():
            assert math.isfinite(val)
        assert math.isfinite(result["composite_score"])

    def test_finite_with_zero_inputs(self):
        comp = _make_comparison(
            baseline_energy=0.0,
            multidim_ch0_energy=0.0,
            multidim_ch1_energy=0.0,
            channel_correlation=0.0,
            baseline_variance=0.0,
            multidim_variance=0.0,
        )
        result = interpret_sonification_comparison(comp)
        for val in result["scores"].values():
            assert math.isfinite(val)
        for val in result["derived"].values():
            assert math.isfinite(val)
        assert math.isfinite(result["composite_score"])


class TestOutputStructure:
    def test_has_required_keys(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        assert "scores" in result
        assert "derived" in result
        assert "composite_score" in result
        assert "verdict" in result

    def test_scores_keys(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        expected = {"silence", "leakage", "correlation", "variance", "energy"}
        assert set(result["scores"].keys()) == expected

    def test_derived_keys(self):
        comp = _make_comparison()
        result = interpret_sonification_comparison(comp)
        expected = {"energy_ratio", "variance_delta", "abs_correlation"}
        assert set(result["derived"].keys()) == expected


class TestDerivedMetrics:
    def test_energy_ratio(self):
        comp = _make_comparison(
            baseline_energy=1.0,
            multidim_ch0_energy=0.8,
            multidim_ch1_energy=1.2,
        )
        result = interpret_sonification_comparison(comp)
        assert result["derived"]["energy_ratio"] == 1.0

    def test_variance_delta(self):
        comp = _make_comparison(baseline_variance=0.1, multidim_variance=0.3)
        result = interpret_sonification_comparison(comp)
        assert abs(result["derived"]["variance_delta"] - 0.2) < 1e-12

    def test_abs_correlation(self):
        comp = _make_comparison(channel_correlation=-0.7)
        result = interpret_sonification_comparison(comp)
        assert abs(result["derived"]["abs_correlation"] - 0.7) < 1e-12
