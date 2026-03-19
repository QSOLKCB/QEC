"""Tests for invariant sonification engine (v74.6.0)."""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest

from qec.experiments.invariant_sonification import (
    _amplitude_from_stability,
    _envelope,
    _generate_feature_tone,
    _harmonic_mix,
    _modulation_rate,
    generate_invariant_signal,
    generate_sequence_sound,
    write_wav,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(
    *,
    stability_score: float = 0.5,
    phase: str = "stable_region",
    strong: list | None = None,
    weak: list | None = None,
    non: list | None = None,
    energy_drift: float = 0.01,
    centroid_drift: float = 0.02,
    spread_drift: float = 0.001,
    zcr_drift: float = 0.1,
) -> dict:
    """Build a minimal invariant analysis dict for testing."""
    return {
        "stability_score": stability_score,
        "phase": phase,
        "invariants": {
            "strong_invariants": strong if strong is not None else ["spread"],
            "weak_invariants": weak if weak is not None else ["energy"],
            "non_invariants": non if non is not None else ["centroid", "zcr"],
        },
        "feature_ranking": [
            ("spread", 0.001),
            ("energy", 0.01),
            ("centroid", 0.02),
            ("zcr", 0.1),
        ],
        "most_stable": "spread",
        "most_sensitive": "zcr",
        "mean_drift": {
            "energy": energy_drift,
            "centroid": centroid_drift,
            "spread": spread_drift,
            "zcr": zcr_drift,
        },
    }


# ---------------------------------------------------------------------------
# Amplitude mapping
# ---------------------------------------------------------------------------

class TestAmplitudeFromStability:
    """Tests for stability → amplitude mapping."""

    def test_zero_stability_gives_max_amplitude(self):
        assert _amplitude_from_stability(0.0) == 1.0

    def test_high_stability_reduces_amplitude(self):
        amp = _amplitude_from_stability(9.0)
        assert amp == pytest.approx(0.1)

    def test_always_positive(self):
        for s in [0.0, 0.5, 1.0, 10.0, 100.0]:
            assert _amplitude_from_stability(s) > 0.0


# ---------------------------------------------------------------------------
# Modulation rate
# ---------------------------------------------------------------------------

class TestModulationRate:
    """Tests for drift → modulation rate mapping."""

    def test_zero_drift_gives_zero_modulation(self):
        assert _modulation_rate(0.0) == 0.0

    def test_max_drift_caps_at_20(self):
        assert _modulation_rate(5.0) == 20.0

    def test_proportional_for_small_drift(self):
        rate = _modulation_rate(0.5)
        assert rate == pytest.approx(10.0)

    def test_negative_drift_uses_abs(self):
        assert _modulation_rate(-0.25) == _modulation_rate(0.25)


# ---------------------------------------------------------------------------
# Harmonic mix
# ---------------------------------------------------------------------------

class TestHarmonicMix:
    """Tests for invariant strength → harmonic structure."""

    def test_strong_invariant_pure_sine(self):
        inv = {"strong_invariants": ["energy"]}
        assert _harmonic_mix("energy", inv) == [1.0, 0.0, 0.0]

    def test_weak_invariant_small_harmonic(self):
        inv = {"weak_invariants": ["centroid"]}
        h = _harmonic_mix("centroid", inv)
        assert h[0] == 1.0
        assert h[1] > 0.0
        assert h[2] == 0.0

    def test_non_invariant_multi_harmonic(self):
        inv = {"strong_invariants": [], "weak_invariants": []}
        h = _harmonic_mix("zcr", inv)
        assert h[0] == 1.0
        assert h[1] > 0.0
        assert h[2] > 0.0


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class TestEnvelope:
    """Tests for phase → envelope shape."""

    def test_sustained_all_ones(self):
        env = _envelope(100, "stable_region")
        np.testing.assert_array_equal(env, np.ones(100))

    def test_pulsing_range(self):
        env = _envelope(44100, "near_boundary")
        assert np.min(env) >= 0.0
        assert np.max(env) <= 1.0

    def test_decaying_starts_high_ends_low(self):
        env = _envelope(44100, "unstable_region", sr=44100)
        assert env[0] > env[-1]

    def test_irregular_bounded(self):
        env = _envelope(44100, "chaotic_transition")
        assert np.min(env) >= 0.0
        assert np.max(env) <= 1.0

    def test_unknown_phase_defaults_to_sustained(self):
        env = _envelope(100, "unknown_phase")
        np.testing.assert_array_equal(env, np.ones(100))


# ---------------------------------------------------------------------------
# Determinism — generate_invariant_signal
# ---------------------------------------------------------------------------

class TestGenerateInvariantSignal:
    """Tests for the main signal generator."""

    def test_determinism(self):
        analysis = _make_analysis()
        s1 = generate_invariant_signal(analysis, duration=0.5, sr=8000)
        s2 = generate_invariant_signal(analysis, duration=0.5, sr=8000)
        np.testing.assert_array_equal(s1, s2)

    def test_no_mutation(self):
        analysis = _make_analysis()
        original = copy.deepcopy(analysis)
        generate_invariant_signal(analysis, duration=0.5, sr=8000)
        assert analysis == original

    def test_correct_shape(self):
        signal = generate_invariant_signal(_make_analysis(), duration=1.0, sr=8000)
        assert signal.shape == (8000,)

    def test_finite_values(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.5, sr=8000)
        assert np.all(np.isfinite(signal))

    def test_amplitude_bounds(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.5, sr=8000)
        assert np.max(np.abs(signal)) <= 1.0

    def test_zero_duration_empty(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.0, sr=8000)
        assert len(signal) == 0

    def test_different_phases_differ(self):
        a1 = _make_analysis(phase="stable_region")
        a2 = _make_analysis(phase="chaotic_transition")
        s1 = generate_invariant_signal(a1, duration=0.5, sr=8000)
        s2 = generate_invariant_signal(a2, duration=0.5, sr=8000)
        assert not np.array_equal(s1, s2)

    def test_different_stability_differ(self):
        a1 = _make_analysis(stability_score=0.0)
        a2 = _make_analysis(stability_score=5.0)
        s1 = generate_invariant_signal(a1, duration=0.5, sr=8000)
        s2 = generate_invariant_signal(a2, duration=0.5, sr=8000)
        assert not np.array_equal(s1, s2)


# ---------------------------------------------------------------------------
# Sequence mode
# ---------------------------------------------------------------------------

class TestGenerateSequenceSound:
    """Tests for multi-frame sequence generation."""

    def test_empty_input(self):
        signal = generate_sequence_sound([])
        assert len(signal) == 0

    def test_single_frame_matches(self):
        a = _make_analysis()
        single = generate_invariant_signal(a, duration=1.0, sr=8000)
        seq = generate_sequence_sound([a], duration_per_frame=1.0, sr=8000)
        np.testing.assert_array_equal(single, seq)

    def test_multi_frame_length(self):
        analyses = [_make_analysis(), _make_analysis(stability_score=2.0)]
        seq = generate_sequence_sound(analyses, duration_per_frame=0.5, sr=8000)
        assert len(seq) == 2 * 4000

    def test_determinism(self):
        analyses = [_make_analysis(), _make_analysis(phase="near_boundary")]
        s1 = generate_sequence_sound(analyses, duration_per_frame=0.5, sr=8000)
        s2 = generate_sequence_sound(analyses, duration_per_frame=0.5, sr=8000)
        np.testing.assert_array_equal(s1, s2)


# ---------------------------------------------------------------------------
# WAV output
# ---------------------------------------------------------------------------

class TestWriteWav:
    """Tests for WAV file writing."""

    def test_creates_file(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.1, sr=8000)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.wav")
            result = write_wav(signal, path, sr=8000)
            assert os.path.exists(result)

    def test_file_nonzero_size(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.1, sr=8000)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.wav")
            write_wav(signal, path, sr=8000)
            assert os.path.getsize(path) > 0

    def test_creates_parent_dirs(self):
        signal = generate_invariant_signal(_make_analysis(), duration=0.1, sr=8000)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "dir", "test.wav")
            write_wav(signal, path, sr=8000)
            assert os.path.exists(path)
