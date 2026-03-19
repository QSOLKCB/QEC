"""Tests for sonic comparison and transition analysis (v74.1.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import wave

import numpy as np
import pytest

from qec.experiments.sonic_comparison import (
    _fft_overlap_score,
    classify_comparison,
    compare_sonic_features,
)
from qec.experiments.sonic_transition_analysis import (
    analyze_sonic_sequence,
    run_sequence_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, *, sr: int = 44100, duration: float = 0.5,
              freq: float = 440.0, amplitude: float = 0.5) -> str:
    """Create a deterministic mono 16-bit WAV test file."""
    n_samples = int(sr * duration)
    t = np.arange(n_samples, dtype=np.float64) / sr
    signal = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)
    pcm = (signal * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return path


def _make_features(*, energy: float = 0.3, centroid: float = 1000.0,
                   spread: float = 500.0, zcr: float = 0.1,
                   peak_freq: float = 440.0, peak_mag: float = 100.0,
                   n_peaks: int = 5) -> dict:
    """Build a minimal analysis feature dict for testing."""
    peaks = []
    for i in range(n_peaks):
        peaks.append({
            "frequency_hz": peak_freq + i * 100.0,
            "magnitude": peak_mag / (i + 1),
        })
    return {
        "duration_seconds": 1.0,
        "sample_rate": 44100,
        "n_samples": 44100,
        "rms_energy": energy,
        "peak_amplitude": energy * 2,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "fft_top_peaks": peaks,
        "source_file": "test.wav",
    }


# ---------------------------------------------------------------------------
# compare_sonic_features
# ---------------------------------------------------------------------------

class TestCompareSonicFeatures:
    """Tests for pairwise comparison."""

    def test_identical_features_give_zero_deltas(self):
        a = _make_features()
        result = compare_sonic_features(a, a)
        assert result["energy_delta"] == 0.0
        assert result["centroid_delta"] == 0.0
        assert result["spread_delta"] == 0.0
        assert result["zcr_delta"] == 0.0

    def test_fft_similarity_identical(self):
        a = _make_features()
        result = compare_sonic_features(a, a)
        assert result["fft_similarity"] == pytest.approx(1.0)

    def test_deltas_sign(self):
        a = _make_features(energy=0.2, centroid=800.0)
        b = _make_features(energy=0.5, centroid=1200.0)
        result = compare_sonic_features(a, b)
        assert result["energy_delta"] > 0
        assert result["centroid_delta"] > 0

    def test_no_mutation_of_inputs(self):
        a = _make_features()
        b = _make_features(energy=0.5)
        a_copy = copy.deepcopy(a)
        b_copy = copy.deepcopy(b)
        compare_sonic_features(a, b)
        assert a == a_copy
        assert b == b_copy

    def test_determinism(self):
        a = _make_features()
        b = _make_features(energy=0.4, centroid=1500.0)
        r1 = compare_sonic_features(a, b)
        r2 = compare_sonic_features(a, b)
        assert r1 == r2

    def test_all_outputs_finite(self):
        a = _make_features()
        b = _make_features(energy=0.0, centroid=0.0, spread=0.0, zcr=0.0)
        result = compare_sonic_features(a, b)
        for v in result.values():
            assert math.isfinite(v)

    def test_output_keys(self):
        a = _make_features()
        result = compare_sonic_features(a, a)
        expected_keys = {"energy_delta", "centroid_delta", "spread_delta",
                         "zcr_delta", "fft_similarity"}
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# FFT Overlap Score
# ---------------------------------------------------------------------------

class TestFFTOverlapScore:
    """Tests for the FFT similarity metric."""

    def test_identical_peaks(self):
        peaks = [{"frequency_hz": 440.0, "magnitude": 100.0}]
        assert _fft_overlap_score(peaks, peaks) == pytest.approx(1.0)

    def test_empty_peaks(self):
        assert _fft_overlap_score([], []) == 0.0
        assert _fft_overlap_score([{"frequency_hz": 1.0, "magnitude": 1.0}], []) == 0.0
        assert _fft_overlap_score([], [{"frequency_hz": 1.0, "magnitude": 1.0}]) == 0.0

    def test_no_overlap(self):
        a = [{"frequency_hz": 100.0, "magnitude": 50.0}]
        b = [{"frequency_hz": 10000.0, "magnitude": 50.0}]
        assert _fft_overlap_score(a, b) == 0.0

    def test_score_between_zero_and_one(self):
        a = [{"frequency_hz": 440.0, "magnitude": 100.0},
             {"frequency_hz": 880.0, "magnitude": 50.0}]
        b = [{"frequency_hz": 445.0, "magnitude": 80.0},
             {"frequency_hz": 1320.0, "magnitude": 60.0}]
        score = _fft_overlap_score(a, b)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassifyComparison:
    """Tests for transition classification."""

    def test_stable(self):
        metrics = {
            "energy_delta": 0.0,
            "centroid_delta": 0.0,
            "spread_delta": 0.0,
            "zcr_delta": 0.0,
            "fft_similarity": 0.95,
        }
        assert classify_comparison(metrics) == "stable"

    def test_collapse(self):
        metrics = {
            "energy_delta": -0.1,
            "centroid_delta": -500.0,
            "spread_delta": -300.0,
            "zcr_delta": -0.05,
            "fft_similarity": 0.1,
        }
        assert classify_comparison(metrics) == "collapse"

    def test_recovery(self):
        metrics = {
            "energy_delta": 0.1,
            "centroid_delta": 200.0,
            "spread_delta": 100.0,
            "zcr_delta": 0.01,
            "fft_similarity": 0.5,
        }
        assert classify_comparison(metrics) == "recovery"

    def test_transition(self):
        metrics = {
            "energy_delta": 0.01,
            "centroid_delta": 500.0,
            "spread_delta": 300.0,
            "zcr_delta": 0.05,
            "fft_similarity": 0.4,
        }
        assert classify_comparison(metrics) == "transition"

    def test_divergent(self):
        metrics = {
            "energy_delta": 0.01,
            "centroid_delta": 300.0,
            "spread_delta": 50.0,
            "zcr_delta": 0.005,
            "fft_similarity": 0.8,
        }
        assert classify_comparison(metrics) == "divergent"

    def test_classification_returns_valid_string(self):
        a = _make_features()
        b = _make_features(energy=0.5)
        metrics = compare_sonic_features(a, b)
        label = classify_comparison(metrics)
        assert label in {"stable", "divergent", "transition", "collapse", "recovery"}


# ---------------------------------------------------------------------------
# Sequential Analysis
# ---------------------------------------------------------------------------

class TestAnalyzeSonicSequence:
    """Tests for sequential state-transition analysis."""

    def test_empty_input(self):
        result = analyze_sonic_sequence([])
        assert result["n_states"] == 0
        assert result["transitions"] == []

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as td:
            wav = _make_wav(os.path.join(td, "state1.wav"))
            result = analyze_sonic_sequence(
                [wav], output_dir=os.path.join(td, "out"))
            assert result["n_states"] == 1
            assert result["transitions"] == []

    def test_two_files(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = _make_wav(os.path.join(td, "s1.wav"), freq=440.0)
            w2 = _make_wav(os.path.join(td, "s2.wav"), freq=880.0)
            result = analyze_sonic_sequence(
                [w1, w2], output_dir=os.path.join(td, "out"))
            assert result["n_states"] == 2
            assert len(result["transitions"]) == 1
            tr = result["transitions"][0]
            assert "metrics" in tr
            assert "classification" in tr
            assert tr["from_index"] == 0
            assert tr["to_index"] == 1

    def test_three_files_produce_two_transitions(self):
        with tempfile.TemporaryDirectory() as td:
            wavs = [
                _make_wav(os.path.join(td, f"s{i}.wav"), freq=200.0 + i * 300.0)
                for i in range(3)
            ]
            result = analyze_sonic_sequence(
                wavs, output_dir=os.path.join(td, "out"))
            assert result["n_states"] == 3
            assert len(result["transitions"]) == 2

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = _make_wav(os.path.join(td, "a.wav"), freq=440.0)
            w2 = _make_wav(os.path.join(td, "b.wav"), freq=880.0)
            r1 = analyze_sonic_sequence(
                [w1, w2], output_dir=os.path.join(td, "out1"))
            r2 = analyze_sonic_sequence(
                [w1, w2], output_dir=os.path.join(td, "out2"))
            assert r1 == r2

    def test_transition_structure(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = _make_wav(os.path.join(td, "x.wav"), freq=440.0)
            w2 = _make_wav(os.path.join(td, "y.wav"), freq=1200.0, amplitude=0.1)
            result = analyze_sonic_sequence(
                [w1, w2], output_dir=os.path.join(td, "out"))
            tr = result["transitions"][0]
            required_metric_keys = {"energy_delta", "centroid_delta",
                                    "spread_delta", "zcr_delta",
                                    "fft_similarity"}
            assert set(tr["metrics"].keys()) == required_metric_keys
            assert tr["classification"] in {
                "stable", "divergent", "transition", "collapse", "recovery"}


# ---------------------------------------------------------------------------
# run_sequence_analysis (artifact writer)
# ---------------------------------------------------------------------------

class TestRunSequenceAnalysis:
    """Tests for the artifact-writing wrapper."""

    def test_writes_json_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = _make_wav(os.path.join(td, "a.wav"))
            w2 = _make_wav(os.path.join(td, "b.wav"), freq=880.0)
            out_dir = os.path.join(td, "artifacts")
            result = run_sequence_analysis([w1, w2], output_dir=out_dir)
            json_path = os.path.join(out_dir, "sequence_analysis.json")
            assert os.path.isfile(json_path)
            with open(json_path) as f:
                loaded = json.load(f)
            assert loaded == result

    def test_all_values_finite(self):
        with tempfile.TemporaryDirectory() as td:
            w1 = _make_wav(os.path.join(td, "a.wav"))
            w2 = _make_wav(os.path.join(td, "b.wav"), freq=880.0)
            result = run_sequence_analysis(
                [w1, w2], output_dir=os.path.join(td, "out"))
            for tr in result["transitions"]:
                for v in tr["metrics"].values():
                    assert math.isfinite(v)
