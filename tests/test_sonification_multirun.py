"""Tests for sonification multi-run engine (v73.0.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import numpy as np
import pytest

from qec.experiments.sonification_multirun import run_sonification_multirun


def _make_input(columns=None, error_rate=0.05, complexity=0.5) -> dict:
    """Standard test input dict."""
    return {
        "columns": columns if columns is not None else [0, 1, 2, 3],
        "errorRate": error_rate,
        "complexity": complexity,
        "invariants": [(1.0, 1.5), (3.0, 3.5)],
    }


def _make_batch(n=3, offset=0) -> list[dict]:
    """Create a batch of *n* distinct inputs."""
    items = []
    for i in range(n):
        items.append(_make_input(
            columns=[i + offset, i + offset + 1, i + offset + 2],
            error_rate=0.01 * (i + 1),
            complexity=0.2 * (i + 1),
        ))
    return items


def _make_datasets(n_runs=3, batch_size=3) -> list[list[dict]]:
    """Create *n_runs* datasets, each with *batch_size* inputs."""
    return [_make_batch(batch_size, offset=i * 10) for i in range(n_runs)]


class TestDeterminism:
    """Same inputs must produce identical outputs."""

    def test_multirun_determinism(self):
        datasets = _make_datasets(3, 3)
        r1 = run_sonification_multirun(datasets)
        r2 = run_sonification_multirun(datasets)
        assert r1["n_runs"] == r2["n_runs"]
        assert r1["global_mean_score"] == r2["global_mean_score"]
        assert r1["global_variance"] == r2["global_variance"]
        assert r1["stability_score"] == r2["stability_score"]
        assert r1["verdict_totals"] == r2["verdict_totals"]
        assert r1["best_run_index"] == r2["best_run_index"]
        for s1, s2 in zip(r1["runs"], r2["runs"]):
            assert s1["n_samples"] == s2["n_samples"]
            assert s1["mean_score"] == s2["mean_score"]

    def test_single_run_determinism(self):
        datasets = [_make_batch(2)]
        r1 = run_sonification_multirun(datasets)
        r2 = run_sonification_multirun(datasets)
        assert r1 == r2


class TestNoMutation:
    """Input datasets must not be modified."""

    def test_input_datasets_unchanged(self):
        datasets = _make_datasets(2, 2)
        original = copy.deepcopy(datasets)
        run_sonification_multirun(datasets)
        assert datasets == original

    def test_inner_lists_unchanged(self):
        datasets = _make_datasets(2, 3)
        originals = [copy.deepcopy(ds) for ds in datasets]
        run_sonification_multirun(datasets)
        for ds, orig in zip(datasets, originals):
            assert ds == orig

    def test_inner_elements_unchanged(self):
        datasets = _make_datasets(2, 2)
        originals = [[copy.deepcopy(item) for item in ds] for ds in datasets]
        run_sonification_multirun(datasets)
        for ds, orig_list in zip(datasets, originals):
            for item, orig in zip(ds, orig_list):
                assert item == orig


class TestCorrectAggregation:
    """Cross-run metrics must be correctly computed."""

    def test_global_mean(self):
        datasets = _make_datasets(3, 2)
        result = run_sonification_multirun(datasets)
        mean_scores = [r["mean_score"] for r in result["runs"]]
        expected = sum(mean_scores) / len(mean_scores)
        assert math.isclose(result["global_mean_score"], expected, rel_tol=1e-12)

    def test_global_variance(self):
        datasets = _make_datasets(3, 2)
        result = run_sonification_multirun(datasets)
        mean_scores = [r["mean_score"] for r in result["runs"]]
        expected = float(np.var(mean_scores))
        assert math.isclose(result["global_variance"], expected, rel_tol=1e-12)

    def test_stability_score(self):
        datasets = _make_datasets(3, 2)
        result = run_sonification_multirun(datasets)
        expected = 1.0 / (1.0 + result["global_variance"])
        assert math.isclose(result["stability_score"], expected, rel_tol=1e-12)

    def test_verdict_totals(self):
        datasets = _make_datasets(3, 3)
        result = run_sonification_multirun(datasets)
        expected = {
            "invalid": 0,
            "multidim_improves_structure": 0,
            "channels_redundant": 0,
            "baseline_more_stable": 0,
            "tradeoff": 0,
        }
        for r in result["runs"]:
            for key in expected:
                expected[key] += r["verdict_counts"].get(key, 0)
        assert result["verdict_totals"] == expected

    def test_verdict_totals_sum(self):
        datasets = _make_datasets(3, 3)
        result = run_sonification_multirun(datasets)
        total_samples = sum(r["n_samples"] for r in result["runs"])
        total_verdicts = sum(result["verdict_totals"].values())
        assert total_verdicts == total_samples

    def test_best_run_index(self):
        datasets = _make_datasets(4, 2)
        result = run_sonification_multirun(datasets)
        mean_scores = [r["mean_score"] for r in result["runs"]]
        assert result["best_run_index"] == int(np.argmax(mean_scores))

    def test_n_runs(self):
        for n in [1, 2, 5]:
            datasets = _make_datasets(n, 2)
            result = run_sonification_multirun(datasets)
            assert result["n_runs"] == n
            assert len(result["runs"]) == n


class TestEmptyInput:
    """Empty datasets list must return a safe default structure."""

    def test_empty_datasets(self):
        result = run_sonification_multirun([])
        assert result["n_runs"] == 0
        assert result["global_mean_score"] == 0.0
        assert result["global_variance"] == 0.0
        assert result["stability_score"] == 1.0
        assert result["best_run_index"] == -1
        assert result["runs"] == []
        assert sum(result["verdict_totals"].values()) == 0

    def test_single_empty_batch(self):
        result = run_sonification_multirun([[]])
        assert result["n_runs"] == 1
        assert result["global_mean_score"] == 0.0
        assert result["runs"][0]["n_samples"] == 0


class TestSingleRun:
    """Single-run datasets must work correctly."""

    def test_single_run(self):
        datasets = [_make_batch(3)]
        result = run_sonification_multirun(datasets)
        assert result["n_runs"] == 1
        assert result["global_variance"] == 0.0
        assert result["stability_score"] == 1.0
        assert result["best_run_index"] == 0
        assert math.isclose(
            result["global_mean_score"],
            result["runs"][0]["mean_score"],
            rel_tol=1e-12,
        )


class TestFiniteOutputs:
    """All numeric outputs must be finite."""

    def test_all_finite(self):
        datasets = _make_datasets(3, 3)
        result = run_sonification_multirun(datasets)
        assert math.isfinite(result["global_mean_score"])
        assert math.isfinite(result["global_variance"])
        assert math.isfinite(result["stability_score"])
        for r in result["runs"]:
            assert math.isfinite(r["mean_score"])


class TestOutputDir:
    """Optional output directory must produce multirun_summary.json."""

    def test_writes_summary(self):
        datasets = _make_datasets(2, 2)
        with tempfile.TemporaryDirectory() as td:
            result = run_sonification_multirun(datasets, output_dir=td)
            path = os.path.join(td, "multirun_summary.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["n_runs"] == result["n_runs"]
            assert loaded["global_mean_score"] == result["global_mean_score"]

    def test_empty_with_output_dir(self):
        with tempfile.TemporaryDirectory() as td:
            run_sonification_multirun([], output_dir=td)
            path = os.path.join(td, "multirun_summary.json")
            assert os.path.isfile(path)


class TestRunStructure:
    """Each run must contain expected batch summary keys."""

    def test_run_keys(self):
        datasets = _make_datasets(2, 2)
        result = run_sonification_multirun(datasets)
        for r in result["runs"]:
            assert "n_samples" in r
            assert "mean_score" in r
            assert "verdict_counts" in r
            assert "best_index" in r
            assert "worst_index" in r
            assert "samples" in r
