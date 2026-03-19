"""Tests for sonification_regression v73.1.0."""

import copy
import json
import math
import os
import tempfile

from qec.experiments.sonification_regression import compare_sonification_runs


def _make_run(mean=0.5, variance=0.01, stability=0.99, improves=5, total=10):
    """Helper to build a minimal multirun output dict."""
    remainder = total - improves
    return {
        "global_mean_score": mean,
        "global_variance": variance,
        "stability_score": stability,
        "verdict_totals": {
            "invalid": 0,
            "multidim_improves_structure": improves,
            "channels_redundant": remainder,
            "baseline_more_stable": 0,
            "tradeoff": 0,
        },
    }


class TestDeterminism:
    """Identical inputs must produce identical outputs."""

    def test_determinism(self):
        a = _make_run(mean=0.4, stability=0.8)
        b = _make_run(mean=0.6, stability=0.9)
        r1 = compare_sonification_runs(a, b)
        r2 = compare_sonification_runs(a, b)
        assert r1 == r2

    def test_determinism_multiple_calls(self):
        a = _make_run()
        b = _make_run(mean=0.6)
        results = [compare_sonification_runs(a, b) for _ in range(5)]
        assert all(r == results[0] for r in results)


class TestNoMutation:
    """Inputs must not be modified."""

    def test_run_a_not_mutated(self):
        a = _make_run()
        b = _make_run(mean=0.7)
        a_copy = copy.deepcopy(a)
        compare_sonification_runs(a, b)
        assert a == a_copy

    def test_run_b_not_mutated(self):
        a = _make_run()
        b = _make_run(mean=0.7)
        b_copy = copy.deepcopy(b)
        compare_sonification_runs(a, b)
        assert b == b_copy


class TestDeltaCalculation:
    """Deltas must be run_b - run_a."""

    def test_positive_mean_delta(self):
        a = _make_run(mean=0.4)
        b = _make_run(mean=0.6)
        result = compare_sonification_runs(a, b)
        assert abs(result["deltas"]["mean"] - 0.2) < 1e-12

    def test_negative_mean_delta(self):
        a = _make_run(mean=0.6)
        b = _make_run(mean=0.4)
        result = compare_sonification_runs(a, b)
        assert abs(result["deltas"]["mean"] - (-0.2)) < 1e-12

    def test_zero_mean_delta(self):
        a = _make_run(mean=0.5)
        b = _make_run(mean=0.5)
        result = compare_sonification_runs(a, b)
        assert result["deltas"]["mean"] == 0.0

    def test_variance_delta(self):
        a = _make_run(variance=0.01)
        b = _make_run(variance=0.02)
        result = compare_sonification_runs(a, b)
        assert abs(result["deltas"]["variance"] - 0.01) < 1e-12

    def test_stability_delta(self):
        a = _make_run(stability=0.9)
        b = _make_run(stability=0.95)
        result = compare_sonification_runs(a, b)
        assert abs(result["deltas"]["stability"] - 0.05) < 1e-12


class TestClassification:
    """Classification must follow the spec rules."""

    def test_improved(self):
        a = _make_run(mean=0.4, stability=0.8)
        b = _make_run(mean=0.6, stability=0.9)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "improved"

    def test_regressed(self):
        a = _make_run(mean=0.6, stability=0.9)
        b = _make_run(mean=0.4, stability=0.8)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "regressed"

    def test_equivalent(self):
        a = _make_run(mean=0.5, stability=0.9)
        b = _make_run(mean=0.5, stability=0.9)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "equivalent"

    def test_equivalent_within_tolerance(self):
        a = _make_run(mean=0.5, stability=0.9)
        b = _make_run(mean=0.5 + 1e-7, stability=0.95)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "equivalent"

    def test_mixed_mean_up_stability_down(self):
        a = _make_run(mean=0.4, stability=0.9)
        b = _make_run(mean=0.6, stability=0.8)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "mixed"

    def test_mixed_mean_down_stability_up(self):
        a = _make_run(mean=0.6, stability=0.8)
        b = _make_run(mean=0.4, stability=0.9)
        result = compare_sonification_runs(a, b)
        assert result["classification"] == "mixed"


class TestVerdictShift:
    """Verdict shift ratios must be computed correctly."""

    def test_improve_ratios(self):
        a = _make_run(improves=3, total=10)
        b = _make_run(improves=7, total=10)
        result = compare_sonification_runs(a, b)
        assert abs(result["verdict_shift"]["run_a_improve_ratio"] - 0.3) < 1e-12
        assert abs(result["verdict_shift"]["run_b_improve_ratio"] - 0.7) < 1e-12

    def test_zero_totals(self):
        a = _make_run(improves=0, total=0)
        b = _make_run(improves=0, total=0)
        result = compare_sonification_runs(a, b)
        assert result["verdict_shift"]["run_a_improve_ratio"] == 0.0
        assert result["verdict_shift"]["run_b_improve_ratio"] == 0.0

    def test_zero_total_one_side(self):
        a = _make_run(improves=0, total=0)
        b = _make_run(improves=5, total=10)
        result = compare_sonification_runs(a, b)
        assert result["verdict_shift"]["run_a_improve_ratio"] == 0.0
        assert abs(result["verdict_shift"]["run_b_improve_ratio"] - 0.5) < 1e-12


class TestFiniteOutputs:
    """All numeric outputs must be finite."""

    def test_all_finite(self):
        a = _make_run()
        b = _make_run(mean=0.7)
        result = compare_sonification_runs(a, b)
        assert math.isfinite(result["deltas"]["mean"])
        assert math.isfinite(result["deltas"]["variance"])
        assert math.isfinite(result["deltas"]["stability"])
        assert math.isfinite(result["verdict_shift"]["run_a_improve_ratio"])
        assert math.isfinite(result["verdict_shift"]["run_b_improve_ratio"])

    def test_zero_inputs_finite(self):
        a = _make_run(mean=0.0, variance=0.0, stability=0.0, improves=0, total=0)
        b = _make_run(mean=0.0, variance=0.0, stability=0.0, improves=0, total=0)
        result = compare_sonification_runs(a, b)
        for v in result["deltas"].values():
            assert math.isfinite(v)
        for v in result["verdict_shift"].values():
            assert math.isfinite(v)


class TestOutputFile:
    """Optional JSON output must be written correctly."""

    def test_writes_json(self):
        a = _make_run(mean=0.4, stability=0.8)
        b = _make_run(mean=0.6, stability=0.9)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = compare_sonification_runs(a, b, output_dir=tmpdir)
            path = os.path.join(tmpdir, "regression_summary.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == result

    def test_no_file_without_output_dir(self):
        a = _make_run()
        b = _make_run()
        result = compare_sonification_runs(a, b)
        assert isinstance(result, dict)


class TestOutputStructure:
    """Output must contain all required keys."""

    def test_top_level_keys(self):
        a = _make_run()
        b = _make_run()
        result = compare_sonification_runs(a, b)
        assert "deltas" in result
        assert "verdict_shift" in result
        assert "classification" in result

    def test_deltas_keys(self):
        a = _make_run()
        b = _make_run()
        result = compare_sonification_runs(a, b)
        assert "mean" in result["deltas"]
        assert "variance" in result["deltas"]
        assert "stability" in result["deltas"]

    def test_verdict_shift_keys(self):
        a = _make_run()
        b = _make_run()
        result = compare_sonification_runs(a, b)
        assert "run_a_improve_ratio" in result["verdict_shift"]
        assert "run_b_improve_ratio" in result["verdict_shift"]
