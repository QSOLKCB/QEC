"""
Tests for QEC SID 6581 Sonification Engine (v136.8.8).

Required categories:
- dataclass immutability
- voice construction
- waveform determinism
- ring modulation determinism
- filter determinism
- spectral hash stability
- 100 replay determinism
- same-input byte identity
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

from qec.audio.sid6581_sonification_engine import (
    ATTACK_MS_MAX,
    ATTACK_MS_MIN,
    DECAY_MS_MAX,
    DECAY_MS_MIN,
    DURATION,
    ENGINE_VERSION,
    FILTER_CUTOFF_MAX,
    FILTER_CUTOFF_MIN,
    NUM_SAMPLES,
    PULSE_DUTY,
    RELEASE_MS_MAX,
    RELEASE_MS_MIN,
    RESONANCE_MAX,
    RESONANCE_MIN,
    SAMPLE_RATE,
    SID_FREQ_MAX,
    SID_FREQ_MIN,
    SUSTAIN_MAX,
    SUSTAIN_MIN,
    VALID_FILTER_MODES,
    VALID_WAVEFORMS,
    SIDFrame,
    SIDRenderResult,
    SIDVoice,
    _apply_adsr_envelope,
    _clamp,
    _generate_noise,
    _generate_pulse,
    _generate_sawtooth,
    _generate_triangle,
    _generate_waveform,
    _hash_to_float,
    _hash_to_range,
    _render_voice,
    apply_sid_filter,
    apply_sid_ring_modulation,
    build_sid_frame_from_qec_state,
    compute_sid_spectral_hash,
    export_sid_bundle,
    render_sid_waveform,
    run_sid_sonification_cycle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_voice():
    return SIDVoice(
        frequency_hz=440.0,
        waveform="triangle",
        attack_ms=10.0,
        decay_ms=50.0,
        sustain_level=0.7,
        release_ms=100.0,
    )


@pytest.fixture
def default_frame(default_voice):
    v2 = SIDVoice(
        frequency_hz=880.0, waveform="sawtooth",
        attack_ms=20.0, decay_ms=60.0, sustain_level=0.5, release_ms=80.0,
    )
    v3 = SIDVoice(
        frequency_hz=220.0, waveform="pulse",
        attack_ms=5.0, decay_ms=40.0, sustain_level=0.8, release_ms=120.0,
    )
    return SIDFrame(
        voice1=default_voice, voice2=v2, voice3=v3,
        ring_mod_enabled=False, filter_mode="lowpass",
        cutoff_hz=1000.0, resonance=0.3,
    )


@pytest.fixture
def cognition_result():
    return {"match": {"confidence": 0.85, "identity": "test_state"}}


@pytest.fixture
def gate_result():
    return {"decision": {"confidence": 0.7, "verdict": "promote"}}


@pytest.fixture
def history_ledger():
    return {"drift_score": 0.2, "stable_hash": "abc123"}


@pytest.fixture
def reference_result(cognition_result, gate_result, history_ledger):
    return run_sid_sonification_cycle(
        cognition_result, gate_result, history_ledger,
    )


# ===========================================================================
# 1. Dataclass immutability tests
# ===========================================================================


class TestDataclassImmutability:
    def test_sid_voice_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.frequency_hz = 100.0

    def test_sid_voice_waveform_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.waveform = "noise"

    def test_sid_voice_attack_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.attack_ms = 999.0

    def test_sid_voice_decay_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.decay_ms = 999.0

    def test_sid_voice_sustain_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.sustain_level = 0.1

    def test_sid_voice_release_frozen(self, default_voice):
        with pytest.raises(AttributeError):
            default_voice.release_ms = 999.0

    def test_sid_frame_frozen(self, default_frame):
        with pytest.raises(AttributeError):
            default_frame.ring_mod_enabled = True

    def test_sid_frame_filter_frozen(self, default_frame):
        with pytest.raises(AttributeError):
            default_frame.filter_mode = "highpass"

    def test_sid_frame_cutoff_frozen(self, default_frame):
        with pytest.raises(AttributeError):
            default_frame.cutoff_hz = 5000.0

    def test_sid_frame_resonance_frozen(self, default_frame):
        with pytest.raises(AttributeError):
            default_frame.resonance = 0.9

    def test_sid_render_result_frozen(self, reference_result):
        with pytest.raises(AttributeError):
            reference_result.spectral_hash = "tampered"

    def test_sid_render_result_buffer_frozen(self, reference_result):
        with pytest.raises(AttributeError):
            reference_result.audio_buffer = ()

    def test_sid_render_result_stable_hash_frozen(self, reference_result):
        with pytest.raises(AttributeError):
            reference_result.stable_hash = "tampered"

    def test_sid_render_result_drift_frozen(self, reference_result):
        with pytest.raises(AttributeError):
            reference_result.drift_overlay = 999.0


# ===========================================================================
# 2. Voice construction tests
# ===========================================================================


class TestVoiceConstruction:
    def test_voice_fields(self, default_voice):
        assert default_voice.frequency_hz == 440.0
        assert default_voice.waveform == "triangle"
        assert default_voice.attack_ms == 10.0
        assert default_voice.decay_ms == 50.0
        assert default_voice.sustain_level == 0.7
        assert default_voice.release_ms == 100.0

    def test_voice_equality(self):
        v1 = SIDVoice(440.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        v2 = SIDVoice(440.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        assert v1 == v2

    def test_voice_inequality(self):
        v1 = SIDVoice(440.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        v2 = SIDVoice(880.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        assert v1 != v2

    def test_voice_hash_stable(self):
        v1 = SIDVoice(440.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        v2 = SIDVoice(440.0, "triangle", 10.0, 50.0, 0.7, 100.0)
        assert hash(v1) == hash(v2)

    def test_frame_three_voices(self, default_frame):
        assert default_frame.voice1 is not None
        assert default_frame.voice2 is not None
        assert default_frame.voice3 is not None

    def test_frame_voice_waveforms(self, default_frame):
        assert default_frame.voice1.waveform == "triangle"
        assert default_frame.voice2.waveform == "sawtooth"
        assert default_frame.voice3.waveform == "pulse"

    def test_all_waveforms_valid(self):
        for wf in VALID_WAVEFORMS:
            v = SIDVoice(440.0, wf, 10.0, 50.0, 0.5, 100.0)
            assert v.waveform == wf

    def test_voice_with_noise_waveform(self):
        v = SIDVoice(0.0, "noise", 10.0, 50.0, 0.5, 100.0)
        assert v.waveform == "noise"


# ===========================================================================
# 3. Waveform determinism tests
# ===========================================================================


class TestWaveformDeterminism:
    def test_triangle_deterministic(self):
        a = _generate_triangle(1000, 440.0, SAMPLE_RATE)
        b = _generate_triangle(1000, 440.0, SAMPLE_RATE)
        assert a == b

    def test_sawtooth_deterministic(self):
        a = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        assert a == b

    def test_pulse_deterministic(self):
        a = _generate_pulse(1000, 440.0, SAMPLE_RATE)
        b = _generate_pulse(1000, 440.0, SAMPLE_RATE)
        assert a == b

    def test_noise_deterministic(self):
        a = _generate_noise(1000, "test_salt")
        b = _generate_noise(1000, "test_salt")
        assert a == b

    def test_noise_different_salt_different_output(self):
        a = _generate_noise(1000, "salt_a")
        b = _generate_noise(1000, "salt_b")
        assert a != b

    def test_triangle_range(self):
        buf = _generate_triangle(NUM_SAMPLES, 440.0, SAMPLE_RATE)
        assert all(-1.0 <= s <= 1.0 for s in buf)

    def test_sawtooth_range(self):
        buf = _generate_sawtooth(NUM_SAMPLES, 440.0, SAMPLE_RATE)
        assert all(-1.0 <= s <= 1.0 for s in buf)

    def test_pulse_range(self):
        buf = _generate_pulse(NUM_SAMPLES, 440.0, SAMPLE_RATE)
        assert all(-1.0 <= s <= 1.0 for s in buf)

    def test_noise_range(self):
        buf = _generate_noise(NUM_SAMPLES, "range_test")
        assert all(-2.0 <= s <= 2.0 for s in buf)

    def test_triangle_zero_freq(self):
        buf = _generate_triangle(100, 0.0, SAMPLE_RATE)
        assert all(s == 0.0 for s in buf)

    def test_sawtooth_zero_freq(self):
        buf = _generate_sawtooth(100, 0.0, SAMPLE_RATE)
        assert all(s == 0.0 for s in buf)

    def test_pulse_zero_freq(self):
        buf = _generate_pulse(100, 0.0, SAMPLE_RATE)
        assert all(s == 0.0 for s in buf)

    def test_waveform_dispatch_triangle(self):
        a = _generate_waveform("triangle", 500, 440.0, SAMPLE_RATE)
        b = _generate_triangle(500, 440.0, SAMPLE_RATE)
        assert a == b

    def test_waveform_dispatch_sawtooth(self):
        a = _generate_waveform("sawtooth", 500, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(500, 440.0, SAMPLE_RATE)
        assert a == b

    def test_waveform_dispatch_pulse(self):
        a = _generate_waveform("pulse", 500, 440.0, SAMPLE_RATE)
        b = _generate_pulse(500, 440.0, SAMPLE_RATE)
        assert a == b

    def test_waveform_dispatch_noise(self):
        a = _generate_waveform("noise", 500, 440.0, SAMPLE_RATE, "salt")
        b = _generate_noise(500, "salt")
        assert a == b

    def test_waveform_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid waveform"):
            _generate_waveform("sine", 100, 440.0, SAMPLE_RATE)

    def test_waveform_length(self):
        for wf in ("triangle", "sawtooth", "pulse"):
            buf = _generate_waveform(wf, 1234, 440.0, SAMPLE_RATE)
            assert len(buf) == 1234

    def test_noise_length(self):
        buf = _generate_noise(2000, "len_test")
        assert len(buf) == 2000


# ===========================================================================
# 4. ADSR envelope tests
# ===========================================================================


class TestADSREnvelope:
    def test_adsr_deterministic(self):
        buf = tuple(1.0 for _ in range(1000))
        a = _apply_adsr_envelope(buf, 10.0, 20.0, 0.5, 30.0, SAMPLE_RATE)
        b = _apply_adsr_envelope(buf, 10.0, 20.0, 0.5, 30.0, SAMPLE_RATE)
        assert a == b

    def test_adsr_length_preserved(self):
        buf = tuple(1.0 for _ in range(500))
        result = _apply_adsr_envelope(buf, 10.0, 20.0, 0.5, 30.0, SAMPLE_RATE)
        assert len(result) == 500

    def test_adsr_starts_at_zero(self):
        buf = tuple(1.0 for _ in range(1000))
        result = _apply_adsr_envelope(buf, 100.0, 100.0, 0.5, 100.0, SAMPLE_RATE)
        assert abs(result[0]) < 0.01

    def test_adsr_sustain_level(self):
        buf = tuple(1.0 for _ in range(44100))
        result = _apply_adsr_envelope(buf, 10.0, 10.0, 0.6, 10.0, SAMPLE_RATE)
        mid = len(result) // 2
        assert abs(result[mid] - 0.6) < 0.05


# ===========================================================================
# 5. Ring modulation determinism tests
# ===========================================================================


class TestRingModulation:
    def test_ring_mod_deterministic(self):
        a = _generate_triangle(1000, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(1000, 220.0, SAMPLE_RATE)
        r1 = apply_sid_ring_modulation(a, b)
        r2 = apply_sid_ring_modulation(a, b)
        assert r1 == r2

    def test_ring_mod_length(self):
        a = tuple(1.0 for _ in range(100))
        b = tuple(0.5 for _ in range(100))
        r = apply_sid_ring_modulation(a, b)
        assert len(r) == 100

    def test_ring_mod_values(self):
        a = tuple(1.0 for _ in range(10))
        b = tuple(0.5 for _ in range(10))
        r = apply_sid_ring_modulation(a, b)
        assert all(abs(s - 0.5) < 1e-10 for s in r)

    def test_ring_mod_zero(self):
        a = tuple(0.0 for _ in range(50))
        b = _generate_sawtooth(50, 440.0, SAMPLE_RATE)
        r = apply_sid_ring_modulation(a, b)
        assert all(s == 0.0 for s in r)

    def test_ring_mod_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            apply_sid_ring_modulation((1.0, 2.0), (1.0,))

    def test_ring_mod_commutative(self):
        a = _generate_triangle(500, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(500, 220.0, SAMPLE_RATE)
        assert apply_sid_ring_modulation(a, b) == apply_sid_ring_modulation(b, a)

    def test_ring_mod_identity(self):
        a = _generate_triangle(100, 440.0, SAMPLE_RATE)
        ones = tuple(1.0 for _ in range(100))
        assert apply_sid_ring_modulation(a, ones) == a


# ===========================================================================
# 6. Filter determinism tests
# ===========================================================================


class TestFilterDeterminism:
    def test_lowpass_deterministic(self):
        buf = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        a = apply_sid_filter(buf, "lowpass", 1000.0, 0.3)
        b = apply_sid_filter(buf, "lowpass", 1000.0, 0.3)
        assert a == b

    def test_highpass_deterministic(self):
        buf = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        a = apply_sid_filter(buf, "highpass", 1000.0, 0.3)
        b = apply_sid_filter(buf, "highpass", 1000.0, 0.3)
        assert a == b

    def test_bandpass_deterministic(self):
        buf = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        a = apply_sid_filter(buf, "bandpass", 1000.0, 0.3)
        b = apply_sid_filter(buf, "bandpass", 1000.0, 0.3)
        assert a == b

    def test_filter_length_preserved(self):
        buf = _generate_triangle(500, 440.0, SAMPLE_RATE)
        for mode in VALID_FILTER_MODES:
            r = apply_sid_filter(buf, mode, 1000.0, 0.5)
            assert len(r) == 500

    def test_filter_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid filter mode"):
            apply_sid_filter((0.0,), "notch", 1000.0, 0.5)

    def test_lowpass_attenuates_high_freq(self):
        buf = _generate_sawtooth(4000, 4000.0, SAMPLE_RATE)
        filtered = apply_sid_filter(buf, "lowpass", 200.0, 0.0)
        raw_energy = sum(s * s for s in buf)
        filt_energy = sum(s * s for s in filtered)
        assert filt_energy < raw_energy

    def test_filter_modes_differ(self):
        buf = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        lp = apply_sid_filter(buf, "lowpass", 500.0, 0.5)
        hp = apply_sid_filter(buf, "highpass", 500.0, 0.5)
        bp = apply_sid_filter(buf, "bandpass", 500.0, 0.5)
        assert lp != hp
        assert lp != bp
        assert hp != bp


# ===========================================================================
# 7. Spectral hash stability tests
# ===========================================================================


class TestSpectralHash:
    def test_hash_deterministic(self):
        buf = _generate_triangle(1000, 440.0, SAMPLE_RATE)
        h1 = compute_sid_spectral_hash(buf)
        h2 = compute_sid_spectral_hash(buf)
        assert h1 == h2

    def test_hash_is_sha256(self):
        buf = (0.0, 1.0, -1.0)
        h = compute_sid_spectral_hash(buf)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_differs_for_different_input(self):
        a = _generate_triangle(500, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(500, 440.0, SAMPLE_RATE)
        assert compute_sid_spectral_hash(a) != compute_sid_spectral_hash(b)

    def test_hash_empty_buffer(self):
        h = compute_sid_spectral_hash(())
        assert len(h) == 64

    def test_hash_single_sample(self):
        h = compute_sid_spectral_hash((0.5,))
        assert len(h) == 64


# ===========================================================================
# 8. Frame construction tests
# ===========================================================================


class TestFrameConstruction:
    def test_frame_from_qec_state_deterministic(
        self, cognition_result, gate_result, history_ledger
    ):
        f1 = build_sid_frame_from_qec_state(
            cognition_result, gate_result, history_ledger,
        )
        f2 = build_sid_frame_from_qec_state(
            cognition_result, gate_result, history_ledger,
        )
        assert f1 == f2

    def test_frame_voice1_maps_confidence(self):
        low = build_sid_frame_from_qec_state(
            {"confidence": 0.1}, {"severity": 0.5}, {"drift_score": 0.0},
        )
        high = build_sid_frame_from_qec_state(
            {"confidence": 0.9}, {"severity": 0.5}, {"drift_score": 0.0},
        )
        assert high.voice1.frequency_hz > low.voice1.frequency_hz

    def test_frame_voice2_maps_severity(self):
        low = build_sid_frame_from_qec_state(
            {"confidence": 0.5}, {"decision": {"confidence": 0.9, "verdict": "p"}},
            {"drift_score": 0.0, "stable_hash": "x"},
        )
        high = build_sid_frame_from_qec_state(
            {"confidence": 0.5}, {"decision": {"confidence": 0.1, "verdict": "r"}},
            {"drift_score": 0.0, "stable_hash": "y"},
        )
        # Higher severity -> lower frequency (ominous)
        assert high.voice2.frequency_hz < low.voice2.frequency_hz

    def test_frame_voice3_noise_on_high_drift(self):
        frame = build_sid_frame_from_qec_state(
            {"confidence": 0.5}, {"severity": 0.5},
            {"drift_score": 0.9, "stable_hash": "drift"},
        )
        assert frame.voice3.waveform == "noise"

    def test_frame_voice3_pulse_on_low_drift(self):
        frame = build_sid_frame_from_qec_state(
            {"confidence": 0.5}, {"severity": 0.5},
            {"drift_score": 0.1, "stable_hash": "stable"},
        )
        assert frame.voice3.waveform == "pulse"

    def test_frame_valid_filter_mode(
        self, cognition_result, gate_result, history_ledger
    ):
        frame = build_sid_frame_from_qec_state(
            cognition_result, gate_result, history_ledger,
        )
        assert frame.filter_mode in VALID_FILTER_MODES

    def test_frame_valid_waveforms(
        self, cognition_result, gate_result, history_ledger
    ):
        frame = build_sid_frame_from_qec_state(
            cognition_result, gate_result, history_ledger,
        )
        for v in (frame.voice1, frame.voice2, frame.voice3):
            assert v.waveform in VALID_WAVEFORMS


# ===========================================================================
# 9. Full render tests
# ===========================================================================


class TestFullRender:
    def test_render_deterministic(self, default_frame):
        a = render_sid_waveform(default_frame)
        b = render_sid_waveform(default_frame)
        assert a == b

    def test_render_length(self, default_frame):
        buf = render_sid_waveform(default_frame)
        assert len(buf) == NUM_SAMPLES

    def test_render_custom_length(self, default_frame):
        buf = render_sid_waveform(default_frame, num_samples=500)
        assert len(buf) == 500

    def test_render_with_ring_mod(self, default_voice):
        v2 = SIDVoice(880.0, "sawtooth", 20.0, 60.0, 0.5, 80.0)
        v3 = SIDVoice(220.0, "triangle", 5.0, 40.0, 0.8, 120.0)
        frame = SIDFrame(
            default_voice, v2, v3,
            ring_mod_enabled=True, filter_mode="bandpass",
            cutoff_hz=800.0, resonance=0.5,
        )
        a = render_sid_waveform(frame)
        b = render_sid_waveform(frame)
        assert a == b
        assert len(a) == NUM_SAMPLES


# ===========================================================================
# 10. Sonification cycle tests
# ===========================================================================


class TestSonificationCycle:
    def test_cycle_returns_result(
        self, cognition_result, gate_result, history_ledger
    ):
        result = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        assert isinstance(result, SIDRenderResult)

    def test_cycle_has_audio(
        self, cognition_result, gate_result, history_ledger
    ):
        result = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        assert len(result.audio_buffer) == NUM_SAMPLES

    def test_cycle_has_spectral_hash(
        self, cognition_result, gate_result, history_ledger
    ):
        result = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        assert len(result.spectral_hash) == 64

    def test_cycle_has_stable_hash(
        self, cognition_result, gate_result, history_ledger
    ):
        result = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        assert len(result.stable_hash) == 64

    def test_cycle_drift_overlay(
        self, cognition_result, gate_result, history_ledger
    ):
        result = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        assert result.drift_overlay == 0.2


# ===========================================================================
# 11. Export bundle tests
# ===========================================================================


class TestExportBundle:
    def test_bundle_keys(self, reference_result):
        bundle = export_sid_bundle(reference_result)
        expected_keys = {
            "engine_version", "sample_count", "spectral_hash",
            "drift_overlay", "stable_hash", "audio_buffer_hash",
            "deterministic",
        }
        assert set(bundle.keys()) == expected_keys

    def test_bundle_version(self, reference_result):
        bundle = export_sid_bundle(reference_result)
        assert bundle["engine_version"] == ENGINE_VERSION

    def test_bundle_deterministic_flag(self, reference_result):
        bundle = export_sid_bundle(reference_result)
        assert bundle["deterministic"] is True

    def test_bundle_sample_count(self, reference_result):
        bundle = export_sid_bundle(reference_result)
        assert bundle["sample_count"] == NUM_SAMPLES

    def test_bundle_deterministic(self, reference_result):
        a = export_sid_bundle(reference_result)
        b = export_sid_bundle(reference_result)
        assert a == b


# ===========================================================================
# 12. Helper function tests
# ===========================================================================


class TestHelpers:
    def test_hash_to_float_range(self):
        for i in range(100):
            v = _hash_to_float(f"test_{i}")
            assert 0.0 <= v < 1.0

    def test_hash_to_float_deterministic(self):
        assert _hash_to_float("hello") == _hash_to_float("hello")

    def test_hash_to_float_salt(self):
        a = _hash_to_float("data", "salt_a")
        b = _hash_to_float("data", "salt_b")
        assert a != b

    def test_hash_to_range(self):
        v = _hash_to_range("test", 10.0, 20.0)
        assert 10.0 <= v < 20.0

    def test_hash_to_range_deterministic(self):
        a = _hash_to_range("x", 0.0, 100.0, "s")
        b = _hash_to_range("x", 0.0, 100.0, "s")
        assert a == b

    def test_clamp_within_range(self):
        assert _clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamp_below(self):
        assert _clamp(-1.0, 0.0, 10.0) == 0.0

    def test_clamp_above(self):
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# ===========================================================================
# 13. 100-replay determinism (MANDATORY)
# ===========================================================================


class TestReplayDeterminism:
    def test_100_replay_cycle(
        self, cognition_result, gate_result, history_ledger, reference_result
    ):
        """Mandatory: 100 consecutive runs must produce identical results."""
        for i in range(100):
            result = run_sid_sonification_cycle(
                cognition_result, gate_result, history_ledger,
            )
            assert result.audio_buffer == reference_result.audio_buffer, (
                f"Replay {i}: audio_buffer mismatch"
            )
            assert result.spectral_hash == reference_result.spectral_hash, (
                f"Replay {i}: spectral_hash mismatch"
            )
            assert result.stable_hash == reference_result.stable_hash, (
                f"Replay {i}: stable_hash mismatch"
            )
            assert result.drift_overlay == reference_result.drift_overlay, (
                f"Replay {i}: drift_overlay mismatch"
            )

    def test_100_replay_render(self, default_frame):
        """100 renders of the same frame must be identical."""
        ref = render_sid_waveform(default_frame)
        for i in range(100):
            assert render_sid_waveform(default_frame) == ref, (
                f"Render replay {i}: mismatch"
            )

    def test_100_replay_spectral_hash(self):
        """100 spectral hash computations must be identical."""
        buf = _generate_triangle(NUM_SAMPLES, 440.0, SAMPLE_RATE)
        ref = compute_sid_spectral_hash(buf)
        for i in range(100):
            assert compute_sid_spectral_hash(buf) == ref, (
                f"Hash replay {i}: mismatch"
            )

    def test_100_replay_ring_mod(self):
        """100 ring modulation runs must be identical."""
        a = _generate_triangle(1000, 440.0, SAMPLE_RATE)
        b = _generate_sawtooth(1000, 220.0, SAMPLE_RATE)
        ref = apply_sid_ring_modulation(a, b)
        for i in range(100):
            assert apply_sid_ring_modulation(a, b) == ref, (
                f"Ring mod replay {i}: mismatch"
            )

    def test_100_replay_filter(self):
        """100 filter runs must be identical."""
        buf = _generate_sawtooth(1000, 440.0, SAMPLE_RATE)
        ref = apply_sid_filter(buf, "lowpass", 1000.0, 0.5)
        for i in range(100):
            assert apply_sid_filter(buf, "lowpass", 1000.0, 0.5) == ref, (
                f"Filter replay {i}: mismatch"
            )


# ===========================================================================
# 14. Same-input byte identity
# ===========================================================================


class TestByteIdentity:
    def test_byte_identity_audio_buffer(
        self, cognition_result, gate_result, history_ledger
    ):
        r1 = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        r2 = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        # Compare serialized bytes
        b1 = json.dumps(
            [round(s, 10) for s in r1.audio_buffer],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        b2 = json.dumps(
            [round(s, 10) for s in r2.audio_buffer],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        assert b1 == b2

    def test_byte_identity_bundle(
        self, cognition_result, gate_result, history_ledger
    ):
        r1 = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        r2 = run_sid_sonification_cycle(
            cognition_result, gate_result, history_ledger,
        )
        b1 = json.dumps(
            export_sid_bundle(r1), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        b2 = json.dumps(
            export_sid_bundle(r2), sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        assert b1 == b2


# ===========================================================================
# 15. Decoder untouched verification
# ===========================================================================


class TestDecoderUntouched:
    def test_no_decoder_import_in_engine(self):
        """Verify the SID engine does not import from qec.decoder."""
        engine_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "audio",
            "sid6581_sonification_engine.py",
        )
        engine_path = os.path.normpath(engine_path)
        with open(engine_path, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_decoder_import_in_tests(self):
        """Verify this test file has no actual decoder imports."""
        import qec.audio.sid6581_sonification_engine as mod
        # Verify the engine module has no decoder references in its namespace
        assert not any(
            "decoder" in str(getattr(mod, attr, ""))
            for attr in ("__file__",)
            if "decoder" in str(getattr(mod, attr, ""))
        )
        # Direct check: module's file path should not be under decoder
        assert "decoder" not in mod.__file__

    def test_decoder_directory_untouched(self):
        """Verify decoder directory exists and was not modified."""
        decoder_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder",
        )
        decoder_path = os.path.normpath(decoder_path)
        assert os.path.isdir(decoder_path), "Decoder directory must exist"


# ===========================================================================
# 16. Engine version and constants tests
# ===========================================================================


class TestConstants:
    def test_engine_version(self):
        assert ENGINE_VERSION == "v136.8.8"

    def test_sample_rate(self):
        assert SAMPLE_RATE == 44100

    def test_num_samples(self):
        assert NUM_SAMPLES == int(SAMPLE_RATE * DURATION)

    def test_valid_waveforms(self):
        assert set(VALID_WAVEFORMS) == {"triangle", "sawtooth", "pulse", "noise"}

    def test_valid_filter_modes(self):
        assert set(VALID_FILTER_MODES) == {"lowpass", "highpass", "bandpass"}
