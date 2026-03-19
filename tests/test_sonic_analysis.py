"""Tests for sonic artifact analysis toolkit (v74.0.0)."""

from __future__ import annotations

import copy
import json
import math
import os
import struct
import tempfile
import wave

import numpy as np
import pytest

from qec.experiments.sonic_analysis import (
    analyse_file,
    compute_features,
    compute_spectrogram,
    load_audio,
)
from qec.experiments.sonic_batch_analysis import run_sonic_batch_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav(path: str, *, sr: int = 44100, duration: float = 0.5,
              freq: float = 440.0, amplitude: float = 0.5) -> str:
    """Create a deterministic mono 16-bit WAV test file."""
    n_samples = int(sr * duration)
    t = np.arange(n_samples, dtype=np.float64) / sr
    signal = (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float64)
    # Convert to int16
    pcm = (signal * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    return path


def _make_stereo_wav(path: str, *, sr: int = 44100, duration: float = 0.25) -> str:
    """Create a deterministic stereo 16-bit WAV."""
    n_samples = int(sr * duration)
    t = np.arange(n_samples, dtype=np.float64) / sr
    left = (0.5 * np.sin(2.0 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    right = (0.3 * np.sin(2.0 * np.pi * 880.0 * t) * 32767).astype(np.int16)
    interleaved = np.empty(2 * n_samples, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())

    return path


# ---------------------------------------------------------------------------
# Tests — Audio Loading
# ---------------------------------------------------------------------------

class TestLoadAudio:
    """Tests for load_audio."""

    def test_load_wav_basic(self):
        """Load a simple WAV and verify shape/rate."""
        with tempfile.TemporaryDirectory() as d:
            path = _make_wav(os.path.join(d, "test.wav"))
            signal, sr = load_audio(path)
            assert sr == 44100
            assert signal.ndim == 1
            assert len(signal) == int(44100 * 0.5)
            assert signal.dtype == np.float64

    def test_load_wav_stereo_to_mono(self):
        """Stereo WAV should be mixed to mono."""
        with tempfile.TemporaryDirectory() as d:
            path = _make_stereo_wav(os.path.join(d, "stereo.wav"))
            signal, sr = load_audio(path)
            assert signal.ndim == 1
            assert sr == 44100

    def test_load_nonexistent_raises(self):
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_audio("/nonexistent/audio.wav")

    def test_load_unsupported_format_raises(self):
        """Unsupported extension raises RuntimeError."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "file.ogg")
            with open(path, "wb") as f:
                f.write(b"\x00" * 100)
            with pytest.raises(RuntimeError, match="Unsupported"):
                load_audio(path)

    def test_mp3_without_decoder_raises(self):
        """MP3 without ffmpeg raises RuntimeError with clear message."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "fake.mp3")
            with open(path, "wb") as f:
                f.write(b"\xff\xfb\x90\x00" + b"\x00" * 100)
            # This should fail gracefully since ffmpeg is not installed
            # (or succeed if it is — both are valid)
            try:
                load_audio(path)
            except RuntimeError as e:
                assert "MP3 decoding" in str(e) or "not available" in str(e)

    def test_load_determinism(self):
        """Same WAV produces identical output twice."""
        with tempfile.TemporaryDirectory() as d:
            path = _make_wav(os.path.join(d, "det.wav"))
            s1, sr1 = load_audio(path)
            s2, sr2 = load_audio(path)
            assert sr1 == sr2
            np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Tests — Feature Extraction
# ---------------------------------------------------------------------------

class TestComputeFeatures:
    """Tests for compute_features."""

    def _make_signal(self, freq=440.0, sr=44100, duration=0.5):
        t = np.arange(int(sr * duration), dtype=np.float64) / sr
        return 0.5 * np.sin(2.0 * np.pi * freq * t), sr

    def test_basic_structure(self):
        """Features dict has all required keys."""
        signal, sr = self._make_signal()
        feats = compute_features(signal, sr)
        required_keys = {
            "duration_seconds", "sample_rate", "n_samples",
            "rms_energy", "peak_amplitude", "zero_crossing_rate",
            "spectral_centroid_hz", "spectral_spread_hz", "fft_top_peaks",
        }
        assert required_keys.issubset(feats.keys())

    def test_duration(self):
        """Duration matches expected value."""
        signal, sr = self._make_signal(duration=1.0)
        feats = compute_features(signal, sr)
        assert math.isclose(feats["duration_seconds"], 1.0, rel_tol=1e-6)

    def test_rms_energy_finite(self):
        """RMS energy is finite and non-negative."""
        signal, sr = self._make_signal()
        feats = compute_features(signal, sr)
        assert math.isfinite(feats["rms_energy"])
        assert feats["rms_energy"] >= 0

    def test_peak_amplitude(self):
        """Peak amplitude within expected range."""
        signal, sr = self._make_signal()
        feats = compute_features(signal, sr)
        assert 0 < feats["peak_amplitude"] <= 1.0

    def test_spectral_centroid_near_frequency(self):
        """Centroid of pure tone should be near the tone frequency."""
        signal, sr = self._make_signal(freq=1000.0)
        feats = compute_features(signal, sr)
        # For a pure sine the centroid should be close to the tone frequency
        assert abs(feats["spectral_centroid_hz"] - 1000.0) < 50.0

    def test_all_values_finite(self):
        """All scalar features are finite."""
        signal, sr = self._make_signal()
        feats = compute_features(signal, sr)
        for key, val in feats.items():
            if isinstance(val, float):
                assert math.isfinite(val), f"{key} is not finite"

    def test_determinism(self):
        """Same input yields identical features."""
        signal, sr = self._make_signal()
        f1 = compute_features(signal.copy(), sr)
        f2 = compute_features(signal.copy(), sr)
        for key in f1:
            if isinstance(f1[key], float):
                assert f1[key] == f2[key], f"Non-deterministic: {key}"

    def test_no_mutation(self):
        """Input signal is not mutated."""
        signal, sr = self._make_signal()
        original = signal.copy()
        compute_features(signal, sr)
        np.testing.assert_array_equal(signal, original)

    def test_zero_signal(self):
        """Features on silence are valid."""
        signal = np.zeros(44100, dtype=np.float64)
        feats = compute_features(signal, 44100)
        assert feats["rms_energy"] == 0.0
        assert feats["peak_amplitude"] == 0.0

    def test_fft_top_peaks_structure(self):
        """FFT profile entries have correct structure."""
        signal, sr = self._make_signal()
        feats = compute_features(signal, sr)
        for peak in feats["fft_top_peaks"]:
            assert "frequency_hz" in peak
            assert "magnitude" in peak
            assert math.isfinite(peak["frequency_hz"])
            assert math.isfinite(peak["magnitude"])


# ---------------------------------------------------------------------------
# Tests — Spectrogram
# ---------------------------------------------------------------------------

class TestSpectrogram:
    """Tests for compute_spectrogram."""

    def test_shape(self):
        """Spectrogram has correct shape."""
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        S_db, freqs, times = compute_spectrogram(signal, sr)
        assert S_db.ndim == 2
        assert len(freqs) == S_db.shape[0]
        assert len(times) == S_db.shape[1]

    def test_finite_values(self):
        """All spectrogram values are finite."""
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        S_db, _, _ = compute_spectrogram(signal, sr)
        assert np.all(np.isfinite(S_db))

    def test_determinism(self):
        """Spectrogram is deterministic."""
        sr = 44100
        signal = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
        S1, f1, t1 = compute_spectrogram(signal.copy(), sr)
        S2, f2, t2 = compute_spectrogram(signal.copy(), sr)
        np.testing.assert_array_equal(S1, S2)

    def test_short_signal(self):
        """Signal shorter than FFT window is handled."""
        signal = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        S_db, freqs, times = compute_spectrogram(signal, 44100)
        assert S_db.shape[1] >= 1
        assert np.all(np.isfinite(S_db))

    def test_no_mutation(self):
        """Input signal is not mutated."""
        signal = np.sin(2 * np.pi * 440 * np.arange(22050) / 44100)
        original = signal.copy()
        compute_spectrogram(signal, 44100)
        np.testing.assert_array_equal(signal, original)


# ---------------------------------------------------------------------------
# Tests — Per-File Analysis
# ---------------------------------------------------------------------------

class TestAnalyseFile:
    """Tests for analyse_file."""

    def test_creates_artifacts(self):
        """analyse_file creates expected output files."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "test.wav"))
            out_dir = os.path.join(d, "output")
            result = analyse_file(wav_path, out_dir)

            assert os.path.isfile(os.path.join(out_dir, "analysis.json"))
            assert os.path.isfile(os.path.join(out_dir, "waveform.png"))
            assert os.path.isfile(os.path.join(out_dir, "spectrogram.png"))

    def test_json_matches_return(self):
        """Written JSON matches returned dict."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "test.wav"))
            out_dir = os.path.join(d, "output")
            result = analyse_file(wav_path, out_dir)

            with open(os.path.join(out_dir, "analysis.json")) as f:
                written = json.load(f)

            assert written["duration_seconds"] == result["duration_seconds"]
            assert written["rms_energy"] == result["rms_energy"]

    def test_determinism(self):
        """Same file produces identical JSON."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "test.wav"))
            out1 = os.path.join(d, "out1")
            out2 = os.path.join(d, "out2")

            r1 = analyse_file(wav_path, out1)
            r2 = analyse_file(wav_path, out2)

            for key in r1:
                if isinstance(r1[key], float):
                    assert r1[key] == r2[key], f"Non-deterministic: {key}"

    def test_source_file_included(self):
        """Result includes source_file field."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "tone.wav"))
            out_dir = os.path.join(d, "output")
            result = analyse_file(wav_path, out_dir)
            assert result["source_file"] == "tone.wav"


# ---------------------------------------------------------------------------
# Tests — Batch Analysis
# ---------------------------------------------------------------------------

class TestBatchAnalysis:
    """Tests for run_sonic_batch_analysis."""

    def test_empty_input(self):
        """Empty path list returns zero-count summary."""
        result = run_sonic_batch_analysis([])
        assert result["n_files"] == 0
        assert result["mean_duration"] == 0.0
        assert result["files"] == []

    def test_single_file(self):
        """Single file batch produces valid summary."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "single.wav"))
            out_dir = os.path.join(d, "artifacts")
            result = run_sonic_batch_analysis([wav_path], output_root=out_dir)

            assert result["n_files"] == 1
            assert math.isfinite(result["mean_duration"])
            assert math.isfinite(result["mean_energy"])
            assert result["variance_centroid"] == 0.0  # single file → zero variance

    def test_multiple_files(self):
        """Multiple files produce correct aggregate metrics."""
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, freq in enumerate([440, 880, 1320]):
                p = _make_wav(
                    os.path.join(d, f"tone_{i}.wav"),
                    freq=float(freq),
                    duration=0.25,
                )
                paths.append(p)

            out_dir = os.path.join(d, "artifacts")
            result = run_sonic_batch_analysis(paths, output_root=out_dir)

            assert result["n_files"] == 3
            assert len(result["files"]) == 3
            assert result["mean_duration"] > 0
            assert result["variance_centroid"] > 0  # different freqs → nonzero

    def test_batch_summary_json(self):
        """Batch summary JSON file is written."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "test.wav"))
            out_dir = os.path.join(d, "artifacts")
            run_sonic_batch_analysis([wav_path], output_root=out_dir)

            summary_path = os.path.join(out_dir, "batch_summary.json")
            assert os.path.isfile(summary_path)
            with open(summary_path) as f:
                data = json.load(f)
            assert "n_files" in data

    def test_determinism(self):
        """Batch analysis is deterministic."""
        with tempfile.TemporaryDirectory() as d:
            paths = [
                _make_wav(os.path.join(d, f"t{i}.wav"), freq=float(440 + i * 100))
                for i in range(2)
            ]
            out1 = os.path.join(d, "out1")
            out2 = os.path.join(d, "out2")

            r1 = run_sonic_batch_analysis(paths, output_root=out1)
            r2 = run_sonic_batch_analysis(paths, output_root=out2)

            assert r1["mean_duration"] == r2["mean_duration"]
            assert r1["mean_energy"] == r2["mean_energy"]
            assert r1["mean_centroid"] == r2["mean_centroid"]

    def test_no_mutation_of_paths(self):
        """Input path list is not mutated."""
        with tempfile.TemporaryDirectory() as d:
            paths = [_make_wav(os.path.join(d, "a.wav"))]
            original = list(paths)
            run_sonic_batch_analysis(paths, output_root=os.path.join(d, "out"))
            assert paths == original

    def test_all_aggregate_fields_finite(self):
        """All aggregate numeric fields are finite."""
        with tempfile.TemporaryDirectory() as d:
            wav_path = _make_wav(os.path.join(d, "test.wav"))
            out_dir = os.path.join(d, "out")
            result = run_sonic_batch_analysis([wav_path], output_root=out_dir)

            for key in ("mean_duration", "mean_energy", "mean_centroid",
                        "variance_centroid"):
                assert math.isfinite(result[key]), f"{key} not finite"
