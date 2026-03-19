"""Tests for sonification batch runner (v72.5.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile

import numpy as np
import pytest

from qec.experiments.sonification_batch import run_sonification_batch


def _make_input(columns=None, error_rate=0.05, complexity=0.5) -> dict:
    """Standard test input dict."""
    return {
        "columns": columns if columns is not None else [0, 1, 2, 3],
        "errorRate": error_rate,
        "complexity": complexity,
        "invariants": [(1.0, 1.5), (3.0, 3.5)],
    }


def _make_batch(n=3) -> list[dict]:
    """Create a batch of *n* distinct inputs."""
    items = []
    for i in range(n):
        items.append(_make_input(
            columns=[i, i + 1, i + 2],
            error_rate=0.01 * (i + 1),
            complexity=0.2 * (i + 1),
        ))
    return items


class TestDeterminism:
    """Same inputs must produce identical outputs."""

    def test_batch_determinism(self):
        batch = _make_batch(3)
        r1 = run_sonification_batch(batch)
        r2 = run_sonification_batch(batch)
        assert r1["n_samples"] == r2["n_samples"]
        assert r1["mean_score"] == r2["mean_score"]
        assert r1["verdict_counts"] == r2["verdict_counts"]
        assert r1["best_index"] == r2["best_index"]
        assert r1["worst_index"] == r2["worst_index"]
        for s1, s2 in zip(r1["samples"], r2["samples"]):
            assert s1["comparison"] == s2["comparison"]
            assert s1["interpretation"] == s2["interpretation"]

    def test_single_element_determinism(self):
        batch = [_make_input()]
        r1 = run_sonification_batch(batch)
        r2 = run_sonification_batch(batch)
        assert r1 == r2


class TestNoMutation:
    """Input list and elements must not be modified."""

    def test_input_list_unchanged(self):
        batch = _make_batch(3)
        original = copy.deepcopy(batch)
        run_sonification_batch(batch)
        assert batch == original

    def test_input_elements_unchanged(self):
        batch = _make_batch(2)
        originals = [copy.deepcopy(item) for item in batch]
        run_sonification_batch(batch)
        for item, orig in zip(batch, originals):
            assert item == orig


class TestSampleCount:
    """n_samples must match input length."""

    def test_sample_count_matches(self):
        for n in [1, 2, 5]:
            batch = _make_batch(n)
            result = run_sonification_batch(batch)
            assert result["n_samples"] == n
            assert len(result["samples"]) == n


class TestMeanScore:
    """Mean score must equal the average of composite scores."""

    def test_mean_score_correct(self):
        batch = _make_batch(3)
        result = run_sonification_batch(batch)
        scores = [
            s["interpretation"]["composite_score"]
            for s in result["samples"]
        ]
        expected = sum(scores) / len(scores)
        assert math.isclose(result["mean_score"], expected, rel_tol=1e-12)


class TestVerdictCounts:
    """Verdict counts must match per-sample verdicts."""

    def test_verdict_counts_correct(self):
        batch = _make_batch(4)
        result = run_sonification_batch(batch)
        expected_counts = {
            "invalid": 0,
            "multidim_improves_structure": 0,
            "channels_redundant": 0,
            "baseline_more_stable": 0,
            "tradeoff": 0,
        }
        for s in result["samples"]:
            v = s["interpretation"]["verdict"]
            if v in expected_counts:
                expected_counts[v] += 1
        assert result["verdict_counts"] == expected_counts

    def test_verdict_counts_sum_to_n(self):
        batch = _make_batch(5)
        result = run_sonification_batch(batch)
        total = sum(result["verdict_counts"].values())
        assert total == result["n_samples"]


class TestBestWorstIndex:
    """Best and worst indices must point to correct samples."""

    def test_best_index(self):
        batch = _make_batch(4)
        result = run_sonification_batch(batch)
        scores = [
            s["interpretation"]["composite_score"]
            for s in result["samples"]
        ]
        assert result["best_index"] == int(np.argmax(scores))

    def test_worst_index(self):
        batch = _make_batch(4)
        result = run_sonification_batch(batch)
        scores = [
            s["interpretation"]["composite_score"]
            for s in result["samples"]
        ]
        assert result["worst_index"] == int(np.argmin(scores))


class TestEmptyInput:
    """Empty list must return a safe default structure."""

    def test_empty_batch(self):
        result = run_sonification_batch([])
        assert result["n_samples"] == 0
        assert result["mean_score"] == 0.0
        assert result["best_index"] == -1
        assert result["worst_index"] == -1
        assert result["samples"] == []
        assert sum(result["verdict_counts"].values()) == 0


class TestSingleElement:
    """Single-element batch must work correctly."""

    def test_single_element(self):
        batch = [_make_input()]
        result = run_sonification_batch(batch)
        assert result["n_samples"] == 1
        assert len(result["samples"]) == 1
        assert result["best_index"] == 0
        assert result["worst_index"] == 0
        score = result["samples"][0]["interpretation"]["composite_score"]
        assert math.isclose(result["mean_score"], score, rel_tol=1e-12)


class TestFiniteOutputs:
    """All numeric outputs must be finite."""

    def test_all_finite(self):
        batch = _make_batch(3)
        result = run_sonification_batch(batch)
        assert math.isfinite(result["mean_score"])
        for s in result["samples"]:
            assert math.isfinite(s["interpretation"]["composite_score"])
            for v in s["interpretation"]["scores"].values():
                assert math.isfinite(v)


class TestOutputDir:
    """Optional output directory must produce batch_summary.json."""

    def test_writes_summary(self):
        batch = _make_batch(2)
        with tempfile.TemporaryDirectory() as td:
            result = run_sonification_batch(batch, output_dir=td)
            path = os.path.join(td, "batch_summary.json")
            assert os.path.isfile(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["n_samples"] == result["n_samples"]
            assert loaded["mean_score"] == result["mean_score"]

    def test_no_wav_files(self):
        batch = _make_batch(2)
        with tempfile.TemporaryDirectory() as td:
            run_sonification_batch(batch, output_dir=td)
            files = os.listdir(td)
            wav_files = [f for f in files if f.endswith(".wav")]
            assert wav_files == []


class TestSampleStructure:
    """Each sample must contain comparison and interpretation."""

    def test_sample_keys(self):
        batch = _make_batch(2)
        result = run_sonification_batch(batch)
        for s in result["samples"]:
            assert "comparison" in s
            assert "interpretation" in s
            assert "composite_score" in s["interpretation"]
            assert "verdict" in s["interpretation"]
