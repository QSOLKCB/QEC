"""Tests for sonification comparison experiment (v72.3.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import numpy as np
import pytest

from qec.experiments.sonification_comparison import run_sonification_comparison


def _make_input() -> dict:
    """Standard test input dict."""
    return {
        "columns": [0, 1, 2, 3],
        "errorRate": 0.05,
        "complexity": 0.5,
        "invariants": [(1.0, 1.5), (3.0, 3.5)],
    }


class TestDeterminism:
    """Same input must produce identical metrics."""

    def test_metrics_identical_across_runs(self):
        inp = _make_input()
        m1 = run_sonification_comparison(inp)
        m2 = run_sonification_comparison(inp)
        for key in m1:
            if key == "input_summary":
                assert m1[key] == m2[key]
            elif isinstance(m1[key], float):
                assert m1[key] == m2[key], f"Mismatch on {key}"
            else:
                assert m1[key] == m2[key]

    def test_determinism_with_different_params(self):
        inp = {
            "columns": [5, 10, 15],
            "errorRate": 0.1,
            "complexity": 0.8,
            "invariants": [(0.0, 0.5)],
        }
        m1 = run_sonification_comparison(inp)
        m2 = run_sonification_comparison(inp)
        assert m1 == m2


class TestNoMutation:
    """Input dict must not be modified."""

    def test_input_unchanged(self):
        inp = _make_input()
        original = copy.deepcopy(inp)
        run_sonification_comparison(inp)
        assert inp == original

    def test_nested_invariants_unchanged(self):
        inp = _make_input()
        original_invariants = copy.deepcopy(inp["invariants"])
        run_sonification_comparison(inp)
        assert inp["invariants"] == original_invariants


class TestSilenceFidelity:
    """Invariant windows must produce perfect silence."""

    def test_baseline_silence_is_one(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["baseline_silence_fidelity"] == 1.0

    def test_multidim_silence_is_one(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["multidim_silence_fidelity"] == 1.0

    def test_multidim_ch0_silence_is_one(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["multidim_ch0_silence_fidelity"] == 1.0

    def test_multidim_ch1_silence_is_one(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["multidim_ch1_silence_fidelity"] == 1.0


class TestLeakage:
    """No signal leakage into invariant windows."""

    def test_leakage_rate_zero(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["leakage_rate"] == 0.0


class TestOutputStructure:
    """Metrics dict must have the required keys and types."""

    def test_required_keys_present(self):
        metrics = run_sonification_comparison(_make_input())
        required = [
            "version",
            "baseline_silence_fidelity",
            "multidim_silence_fidelity",
            "leakage_rate",
            "baseline_energy",
            "multidim_ch0_energy",
            "multidim_ch1_energy",
            "channel_correlation",
            "baseline_variance",
            "multidim_variance",
            "num_samples",
            "num_invariant_samples",
            "input_summary",
        ]
        for key in required:
            assert key in metrics, f"Missing key: {key}"

    def test_version_string(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["version"] == "v72.3.0"

    def test_input_summary_structure(self):
        metrics = run_sonification_comparison(_make_input())
        summary = metrics["input_summary"]
        assert summary["n_columns"] == 4
        assert summary["error_rate"] == 0.05
        assert summary["complexity"] == 0.5
        assert summary["n_invariants"] == 2


class TestCorrelationBounds:
    """Channel correlation must be in [-1, 1]."""

    def test_correlation_bounded(self):
        metrics = run_sonification_comparison(_make_input())
        assert -1.0 <= metrics["channel_correlation"] <= 1.0

    def test_correlation_bounded_high_complexity(self):
        inp = _make_input()
        inp["complexity"] = 0.95
        metrics = run_sonification_comparison(inp)
        assert -1.0 <= metrics["channel_correlation"] <= 1.0

    def test_correlation_bounded_low_complexity(self):
        inp = _make_input()
        inp["complexity"] = 0.05
        metrics = run_sonification_comparison(inp)
        assert -1.0 <= metrics["channel_correlation"] <= 1.0


class TestMetricsFinite:
    """All numeric metrics must be finite (no NaN or inf)."""

    def test_no_nan_or_inf(self):
        metrics = run_sonification_comparison(_make_input())
        for key, val in metrics.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key} is not finite: {val}"

    def test_energy_nonnegative(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["baseline_energy"] >= 0.0
        assert metrics["multidim_ch0_energy"] >= 0.0
        assert metrics["multidim_ch1_energy"] >= 0.0

    def test_variance_nonnegative(self):
        metrics = run_sonification_comparison(_make_input())
        assert metrics["baseline_variance"] >= 0.0
        assert metrics["multidim_variance"] >= 0.0


class TestArtifactOutput:
    """Optional file outputs must be written correctly."""

    def test_writes_json_and_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_sonification_comparison(_make_input(), output_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "baseline.wav"))
            assert os.path.exists(os.path.join(tmpdir, "multidim.wav"))
            json_path = os.path.join(tmpdir, "comparison.json")
            assert os.path.exists(json_path)
            with open(json_path) as f:
                loaded = json.load(f)
            assert loaded["version"] == "v72.3.0"

    def test_json_matches_returned_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = run_sonification_comparison(_make_input(), output_dir=tmpdir)
            with open(os.path.join(tmpdir, "comparison.json")) as f:
                loaded = json.load(f)
            for key in metrics:
                assert key in loaded


class TestEdgeCases:
    """Edge case inputs."""

    def test_no_invariants(self):
        inp = _make_input()
        inp["invariants"] = []
        metrics = run_sonification_comparison(inp)
        assert metrics["baseline_silence_fidelity"] == 1.0
        assert metrics["leakage_rate"] == 0.0

    def test_single_column(self):
        inp = {
            "columns": [7],
            "errorRate": 0.01,
            "complexity": 0.3,
            "invariants": [(2.0, 2.5)],
        }
        metrics = run_sonification_comparison(inp)
        assert math.isfinite(metrics["channel_correlation"])

    def test_no_output_dir(self):
        metrics = run_sonification_comparison(_make_input())
        assert isinstance(metrics, dict)

    def test_zero_complexity(self):
        inp = _make_input()
        inp["complexity"] = 0.0
        metrics = run_sonification_comparison(inp)
        assert math.isfinite(metrics["baseline_energy"])
