"""Tests for deterministic sonification of hierarchical correction (v96.1.0).

Covers:
  - deterministic waveform generation
  - same input → identical samples
  - frequency mapping correctness
  - invariant harmonics applied correctly
  - wav file creation validity
  - no mutation of inputs
  - stable output across repeated runs
  - amplitude/sustain mapping
  - comparison mode
  - print layer
"""

from __future__ import annotations

import math
import os
import struct
import tempfile
import wave

import pytest

from qec.analysis.sonification import (
    AMPLITUDE,
    BASE_FREQ,
    DURATION_PER_STAGE,
    INVARIANT_HARMONICS,
    SAMPLE_RATE,
    amplitude_from_projection,
    apply_invariant_harmonics,
    generate_tone,
    map_mode_to_freq,
    print_sonification_summary,
    run_sonification,
    sonify_comparison,
    sonify_stages,
    sustain_from_stability,
    write_wav,
    _normalize,
)


# ---------------------------------------------------------------------------
# Test 1 — Frequency mapping: single stages
# ---------------------------------------------------------------------------

class TestMapModeToFreq:
    def test_square(self):
        assert map_mode_to_freq("square") == BASE_FREQ

    def test_d4(self):
        assert map_mode_to_freq("d4") == pytest.approx(BASE_FREQ * 4.0 / 3.0)

    def test_e8_like(self):
        assert map_mode_to_freq("e8_like") == pytest.approx(BASE_FREQ * 3.0 / 2.0)

    def test_multi_stage_sum(self):
        """Multi-stage mode sums component frequencies."""
        expected = BASE_FREQ + BASE_FREQ * 4.0 / 3.0 + BASE_FREQ * 3.0 / 2.0
        assert map_mode_to_freq("square>d4>e8_like") == pytest.approx(expected)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown correction stage"):
            map_mode_to_freq("invalid_mode")


# ---------------------------------------------------------------------------
# Test 2 — Tone generation determinism
# ---------------------------------------------------------------------------

class TestGenerateTone:
    def test_deterministic(self):
        """Same inputs → identical samples."""
        t1 = generate_tone(440.0, 0.1, 0.5)
        t2 = generate_tone(440.0, 0.1, 0.5)
        assert t1 == t2

    def test_length(self):
        """Duration and sample rate determine sample count."""
        samples = generate_tone(220.0, 0.25, 0.3)
        expected_len = int(SAMPLE_RATE * 0.25)
        assert len(samples) == expected_len

    def test_amplitude_bound(self):
        """All samples within [-amplitude, amplitude]."""
        samples = generate_tone(440.0, 0.1, 0.5)
        for s in samples:
            assert -0.5 <= s <= 0.5 + 1e-10

    def test_zero_duration(self):
        samples = generate_tone(440.0, 0.0, 0.5)
        assert samples == []

    def test_sine_shape(self):
        """First sample is zero (sin(0) = 0)."""
        samples = generate_tone(440.0, 0.01, 1.0)
        assert abs(samples[0]) < 1e-10


# ---------------------------------------------------------------------------
# Test 3 — Amplitude from projection
# ---------------------------------------------------------------------------

class TestAmplitudeFromProjection:
    def test_zero_distance(self):
        """Zero projection → minimum amplitude (0.05)."""
        assert amplitude_from_projection(0.0) == pytest.approx(0.05)

    def test_large_distance(self):
        """Large projection → near 1.0."""
        amp = amplitude_from_projection(10.0)
        assert amp > 0.95
        assert amp <= 1.0

    def test_monotonic(self):
        """Amplitude increases with projection distance."""
        a1 = amplitude_from_projection(0.1)
        a2 = amplitude_from_projection(1.0)
        a3 = amplitude_from_projection(5.0)
        assert a1 < a2 < a3

    def test_deterministic(self):
        assert amplitude_from_projection(0.7) == amplitude_from_projection(0.7)


# ---------------------------------------------------------------------------
# Test 4 — Sustain from stability
# ---------------------------------------------------------------------------

class TestSustainFromStability:
    def test_zero_stability(self):
        """Low stability → 0.3x duration."""
        assert sustain_from_stability(0.0) == pytest.approx(0.3)

    def test_full_stability(self):
        """Full stability → 2.0x duration."""
        assert sustain_from_stability(1.0) == pytest.approx(2.0)

    def test_clamped_above(self):
        """Values above 1.0 are clamped."""
        assert sustain_from_stability(5.0) == pytest.approx(2.0)

    def test_clamped_below(self):
        """Values below 0.0 are clamped."""
        assert sustain_from_stability(-1.0) == pytest.approx(0.3)

    def test_deterministic(self):
        assert sustain_from_stability(0.5) == sustain_from_stability(0.5)


# ---------------------------------------------------------------------------
# Test 5 — Invariant harmonics
# ---------------------------------------------------------------------------

class TestInvariantHarmonics:
    def test_no_invariants(self):
        """No invariants → only base frequency."""
        freqs = apply_invariant_harmonics(220.0, [])
        assert freqs == [220.0]

    def test_known_invariant(self):
        """Known invariant adds one harmonic."""
        freqs = apply_invariant_harmonics(
            220.0, ["local_stability_constraint"],
        )
        assert len(freqs) == 2
        assert freqs[0] == 220.0
        assert freqs[1] == pytest.approx(220.0 * 1.25)

    def test_multiple_invariants(self):
        """Multiple invariants add multiple harmonics."""
        freqs = apply_invariant_harmonics(
            220.0,
            ["local_stability_constraint", "bounded_projection_constraint"],
        )
        assert len(freqs) == 3
        assert freqs[0] == 220.0

    def test_unknown_invariant_ignored(self):
        """Unknown invariant names are silently ignored."""
        freqs = apply_invariant_harmonics(220.0, ["nonexistent_constraint"])
        assert freqs == [220.0]

    def test_deterministic(self):
        a = apply_invariant_harmonics(220.0, ["local_stability_constraint"])
        b = apply_invariant_harmonics(220.0, ["local_stability_constraint"])
        assert a == b


# ---------------------------------------------------------------------------
# Test 6 — Stage sonification
# ---------------------------------------------------------------------------

class TestSonifyStages:
    def test_single_stage(self):
        """Single stage produces non-empty waveform."""
        samples = sonify_stages(["square"], [0.5], 0.5, [])
        assert len(samples) > 0

    def test_multi_stage(self):
        """Multi-stage produces longer waveform than single stage."""
        s1 = sonify_stages(["square"], [0.5], 0.5, [])
        s3 = sonify_stages(["square", "d4", "e8_like"], [0.5, 0.3, 0.2], 0.5, [])
        assert len(s3) > len(s1)

    def test_deterministic(self):
        """Identical inputs → identical output."""
        a = sonify_stages(["square", "d4"], [0.5, 0.3], 0.7, [])
        b = sonify_stages(["square", "d4"], [0.5, 0.3], 0.7, [])
        assert a == b

    def test_with_invariants(self):
        """Adding invariants changes the waveform."""
        without = sonify_stages(["square"], [0.5], 0.5, [])
        with_inv = sonify_stages(
            ["square"], [0.5], 0.5, ["local_stability_constraint"],
        )
        assert without != with_inv

    def test_no_mutation(self):
        """Input lists are not mutated."""
        stages = ["square", "d4"]
        dists = [0.5, 0.3]
        invs = ["local_stability_constraint"]
        stages_copy = list(stages)
        dists_copy = list(dists)
        invs_copy = list(invs)

        sonify_stages(stages, dists, 0.5, invs)

        assert stages == stages_copy
        assert dists == dists_copy
        assert invs == invs_copy

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="unknown correction stage"):
            sonify_stages(["nonexistent"], [0.5], 0.5, [])


# ---------------------------------------------------------------------------
# Test 7 — WAV file creation
# ---------------------------------------------------------------------------

class TestWriteWav:
    def test_creates_file(self, tmp_path):
        """write_wav creates a valid WAV file."""
        samples = generate_tone(440.0, 0.1, 0.5)
        fpath = str(tmp_path / "test.wav")
        result = write_wav(fpath, samples)
        assert os.path.isfile(result)

    def test_wav_valid_header(self, tmp_path):
        """Created WAV has correct format parameters."""
        samples = generate_tone(440.0, 0.1, 0.5)
        fpath = str(tmp_path / "test.wav")
        write_wav(fpath, samples)

        with wave.open(fpath, "r") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE
            assert wf.getnframes() == len(samples)

    def test_wav_deterministic(self, tmp_path):
        """Same samples → identical file bytes."""
        samples = generate_tone(440.0, 0.05, 0.3)
        f1 = str(tmp_path / "a.wav")
        f2 = str(tmp_path / "b.wav")
        write_wav(f1, samples)
        write_wav(f2, samples)

        with open(f1, "rb") as a, open(f2, "rb") as b:
            assert a.read() == b.read()


# ---------------------------------------------------------------------------
# Test 8 — Normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_empty(self):
        assert _normalize([]) == []

    def test_already_normalized(self):
        result = _normalize([0.5, -0.5, 0.0])
        peak = max(abs(s) for s in result)
        assert peak == pytest.approx(1.0)

    def test_all_zero(self):
        result = _normalize([0.0, 0.0, 0.0])
        assert result == [0.0, 0.0, 0.0]

    def test_deterministic(self):
        a = _normalize([0.3, -0.7, 0.1])
        b = _normalize([0.3, -0.7, 0.1])
        assert a == b


# ---------------------------------------------------------------------------
# Test 9 — Full pipeline
# ---------------------------------------------------------------------------

class TestRunSonification:
    def test_produces_files(self, tmp_path):
        """Pipeline produces WAV files and returns reports."""
        data = [
            {
                "mode": "square>d4",
                "stages": ["square", "d4"],
                "projection_distances": [0.5, 0.3],
                "stability_efficiency": 0.7,
                "invariants": [],
                "dfa_type": "chain",
                "n": 10,
            },
        ]
        reports = run_sonification(data, output_dir=str(tmp_path))
        assert len(reports) == 1
        assert reports[0]["dfa_type"] == "chain"
        assert reports[0]["n"] == 10
        assert reports[0]["duration"] > 0
        assert os.path.isfile(os.path.join(str(tmp_path), reports[0]["file"]))

    def test_deterministic_pipeline(self, tmp_path):
        """Same input → identical reports and files."""
        data = [
            {
                "mode": "square",
                "stages": ["square"],
                "projection_distances": [0.4],
                "stability_efficiency": 0.6,
                "invariants": ["local_stability_constraint"],
                "dfa_type": "cycle",
                "n": 5,
            },
        ]
        d1 = tmp_path / "run1"
        d2 = tmp_path / "run2"
        d1.mkdir()
        d2.mkdir()

        r1 = run_sonification(data, output_dir=str(d1))
        r2 = run_sonification(data, output_dir=str(d2))

        assert r1 == r2

        # Check file-level byte identity.
        f1 = d1 / r1[0]["file"]
        f2 = d2 / r2[0]["file"]
        assert f1.read_bytes() == f2.read_bytes()

    def test_multiple_systems(self, tmp_path):
        """Pipeline handles multiple input systems."""
        data = [
            {
                "mode": "square",
                "stages": ["square"],
                "projection_distances": [0.2],
                "stability_efficiency": 0.5,
                "dfa_type": "chain",
                "n": 5,
            },
            {
                "mode": "d4",
                "stages": ["d4"],
                "projection_distances": [0.8],
                "stability_efficiency": 0.9,
                "dfa_type": "cycle",
                "n": 8,
            },
        ]
        reports = run_sonification(data, output_dir=str(tmp_path))
        assert len(reports) == 2


# ---------------------------------------------------------------------------
# Test 10 — Comparison mode
# ---------------------------------------------------------------------------

class TestSonifyComparison:
    def test_creates_stereo_wav(self, tmp_path):
        """Comparison produces a valid stereo WAV."""
        before = {
            "mode": "square",
            "stages": ["square"],
            "projection_distances": [0.3],
            "stability_efficiency": 0.5,
            "invariants": [],
        }
        after = {
            "mode": "square>d4>e8_like",
            "stages": ["square", "d4", "e8_like"],
            "projection_distances": [0.3, 0.5, 0.2],
            "stability_efficiency": 0.8,
            "invariants": ["local_stability_constraint"],
        }
        fpath = str(tmp_path / "cmp.wav")
        result = sonify_comparison(before, after, fpath)
        assert os.path.isfile(result)

        with wave.open(result, "r") as wf:
            assert wf.getnchannels() == 2
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SAMPLE_RATE

    def test_comparison_deterministic(self, tmp_path):
        """Same inputs → identical comparison file."""
        before = {"mode": "square", "stages": ["square"],
                  "projection_distances": [0.3], "stability_efficiency": 0.5}
        after = {"mode": "d4", "stages": ["d4"],
                 "projection_distances": [0.6], "stability_efficiency": 0.9}
        f1 = str(tmp_path / "a.wav")
        f2 = str(tmp_path / "b.wav")
        sonify_comparison(before, after, f1)
        sonify_comparison(before, after, f2)
        with open(f1, "rb") as a, open(f2, "rb") as b:
            assert a.read() == b.read()


# ---------------------------------------------------------------------------
# Test 11 — Print summary
# ---------------------------------------------------------------------------

class TestPrintSonificationSummary:
    def test_format(self):
        report = [
            {
                "dfa_type": "cycle",
                "n": 10,
                "file": "qec_cycle_10.wav",
                "duration": 1.5,
            },
        ]
        text = print_sonification_summary(report)
        assert "cycle" in text
        assert "n=10" in text
        assert "qec_cycle_10.wav" in text
        assert "1.5s" in text

    def test_empty_report(self):
        text = print_sonification_summary([])
        assert "Sonification Summary" in text
