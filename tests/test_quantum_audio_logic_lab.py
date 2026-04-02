"""Tests for quantum_audio_logic_lab — deterministic spectral analysis."""

import hashlib
import math
import os
import tempfile

import pytest

import numpy as np

from qec.audio.quantum_audio_logic_lab import (
    ComparativeAnalysisResult,
    QuantumAudioLogicReport,
    _coherence_score,
    _cluster_tightness,
    _compute_byte_pseudospectrum,
    _compute_waveform_spectrum,
    _deterministic_clusters,
    _parse_mp3_metadata,
    _stability_label,
    _try_decode_audio,
    _welch_psd,
    analyze_quantum_audio_file,
    compare_reports,
)

# ---------------------------------------------------------------------------
# Paths to real artifacts (skip if not present)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v1).mp3")
_V2_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v2).mp3")

_HAS_ARTIFACTS = os.path.isfile(_V1_PATH) and os.path.isfile(_V2_PATH)


# ---------------------------------------------------------------------------
# Unit tests — coherence law
# ---------------------------------------------------------------------------

class TestCoherenceLaw:
    def test_zero_inputs(self):
        assert _coherence_score(0, 0, 0, 0) == 0.0

    def test_full_resonance(self):
        score = _coherence_score(1.0, 1.0, 1.0, 0.0)
        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_clamped_upper(self):
        score = _coherence_score(1.0, 1.0, 1.0, 0.0)
        assert score <= 1.0

    def test_clamped_lower(self):
        score = _coherence_score(0.0, 0.0, 0.0, 1.0)
        assert score >= 0.0

    def test_entropy_penalises(self):
        s_low_h = _coherence_score(0.5, 0.5, 0.5, 0.0)
        s_high_h = _coherence_score(0.5, 0.5, 0.5, 1.0)
        assert s_low_h > s_high_h


# ---------------------------------------------------------------------------
# Unit tests — stability label
# ---------------------------------------------------------------------------

class TestStabilityLabel:
    def test_high(self):
        assert _stability_label(0.85) == "highly_coherent"

    def test_moderate(self):
        assert _stability_label(0.50) == "moderately_coherent"

    def test_weak(self):
        assert _stability_label(0.30) == "weakly_coherent"

    def test_incoherent(self):
        assert _stability_label(0.10) == "incoherent"


# ---------------------------------------------------------------------------
# Unit tests — cluster tightness
# ---------------------------------------------------------------------------

class TestClusterTightness:
    def test_empty(self):
        assert _cluster_tightness(()) == 0.0

    def test_single_point(self):
        assert _cluster_tightness(((1.0, 2.0),)) == 0.0

    def test_symmetric(self):
        pts = ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0), (2.0, 2.0))
        t = _cluster_tightness(pts)
        assert t == pytest.approx(math.sqrt(2.0), rel=1e-6)


# ---------------------------------------------------------------------------
# Unit tests — deterministic clusters
# ---------------------------------------------------------------------------

class TestDeterministicClusters:
    def test_reproducibility(self):
        a = _deterministic_clusters(10.0, 0.5, 0.5, "abc123")
        b = _deterministic_clusters(10.0, 0.5, 0.5, "abc123")
        assert a == b

    def test_different_hash_gives_different(self):
        a = _deterministic_clusters(10.0, 0.5, 0.5, "abc")
        b = _deterministic_clusters(10.0, 0.5, 0.5, "xyz")
        assert a != b

    def test_tuple_type(self):
        pts = _deterministic_clusters(10.0, 0.5, 0.5, "h")
        assert isinstance(pts, tuple)
        for p in pts:
            assert isinstance(p, tuple)
            assert len(p) == 2


# ---------------------------------------------------------------------------
# Unit tests — MP3 metadata parser
# ---------------------------------------------------------------------------

class TestMp3MetadataParser:
    @pytest.mark.skipif(not _HAS_ARTIFACTS, reason="MP3 artifacts not present")
    def test_v1_metadata(self):
        with open(_V1_PATH, "rb") as f:
            data = f.read()
        meta = _parse_mp3_metadata(data)
        assert meta is not None
        assert meta.sample_rate == 48000
        assert meta.channels == 2
        assert meta.duration_seconds == pytest.approx(148.848, abs=0.5)
        assert meta.frame_count > 0

    @pytest.mark.skipif(not _HAS_ARTIFACTS, reason="MP3 artifacts not present")
    def test_v2_metadata(self):
        with open(_V2_PATH, "rb") as f:
            data = f.read()
        meta = _parse_mp3_metadata(data)
        assert meta is not None
        assert meta.sample_rate == 48000
        assert meta.channels == 2
        assert meta.duration_seconds == pytest.approx(179.976, abs=0.5)

    def test_non_mp3_returns_none(self):
        meta = _parse_mp3_metadata(b"\x00\x01\x02\x03" * 256)
        assert meta is None


# ---------------------------------------------------------------------------
# Unit tests — decode path cascade
# ---------------------------------------------------------------------------

class TestDecodeCascade:
    def test_try_decode_returns_none_for_mp3_without_decoder(self):
        """MP3 decode fails gracefully when no decoder is installed."""
        if not _HAS_ARTIFACTS:
            pytest.skip("MP3 artifacts not present")
        result = _try_decode_audio(_V1_PATH)
        # If no decoder is available, result is None (fallback path)
        # If a decoder IS available, result is a tuple
        if result is None:
            assert True  # expected fallback
        else:
            samples, sr, name = result
            assert sr > 0
            assert len(samples) > 0

    def test_byte_pseudospectrum_produces_signal(self):
        raw = b"\x00\x01\x02\x03" * 2048
        samples, sr = _compute_byte_pseudospectrum(raw, 44100, 0)
        assert len(samples) > 0
        assert sr == 44100

    def test_byte_pseudospectrum_respects_offset(self):
        raw = b"\xff" * 100 + b"\x00\x01\x02\x03" * 1024
        samples_with_offset, _ = _compute_byte_pseudospectrum(raw, 44100, 100)
        samples_no_offset, _ = _compute_byte_pseudospectrum(raw, 44100, 0)
        # Different offset → different signals
        assert len(samples_with_offset) != len(samples_no_offset)


# ---------------------------------------------------------------------------
# Unit tests — spectral computation
# ---------------------------------------------------------------------------

class TestWaveformSpectrum:
    def test_spectral_features_keys(self):
        samples = np.sin(2 * np.pi * 432 * np.arange(44100) / 44100)
        result = _compute_waveform_spectrum(samples, 44100)
        expected_keys = {
            "dominant_freq", "centroid", "entropy", "max_entropy",
            "harmonic_density", "subharmonic_ratio", "resonance",
        }
        assert set(result.keys()) == expected_keys

    def test_pure_432hz_tone(self):
        """A pure 432 Hz sine wave should have high 432 Hz resonance."""
        sr = 44100
        t = np.arange(sr * 2) / sr  # 2 seconds
        samples = np.sin(2 * np.pi * 432 * t)
        result = _compute_waveform_spectrum(samples, sr)
        assert result["resonance"] > 0.1
        assert result["dominant_freq"] == pytest.approx(432.0, abs=20.0)


# ---------------------------------------------------------------------------
# Unit tests — _welch_psd edge cases
# ---------------------------------------------------------------------------

class TestWelchPsdEdgeCases:
    def test_empty_samples(self):
        """Empty input returns single-element zero spectrum."""
        freqs, psd = _welch_psd(np.array([]), 44100)
        assert len(freqs) == 1
        assert len(psd) == 1
        assert freqs[0] == 0.0
        assert psd[0] == 0.0

    def test_single_sample(self):
        """Single sample does not raise."""
        freqs, psd = _welch_psd(np.array([1.0]), 44100)
        assert len(freqs) >= 1
        assert len(psd) >= 1

    def test_short_samples(self):
        """Very short input (< nperseg) works without error."""
        samples = np.array([0.1, -0.2, 0.3, -0.4])
        freqs, psd = _welch_psd(samples, 44100)
        assert len(freqs) > 0
        assert len(psd) > 0


# ---------------------------------------------------------------------------
# Unit tests — MP3 3-frame validation
# ---------------------------------------------------------------------------

class TestMp3ThreeFrameValidation:
    def test_two_frame_sequence_rejected(self):
        """A byte stream with only 2 valid consecutive frames is rejected."""
        # Build 2 valid MPEG1 Layer III frames (sync=0xFFE, v=1, l=III,
        # bitrate=128k, sr=44100, no padding, stereo)
        # Header: 0xFFFB9004  (sync + v1 + lIII + 128k + 44100 + no-pad)
        import struct
        hdr = struct.pack(">I", 0xFFFB9004)
        frame_len = 144 * 128000 // 44100  # 417 bytes
        frame = hdr + b"\x00" * (frame_len - 4)
        # Two frames then garbage
        data = frame * 2 + b"\x00" * 100
        meta = _parse_mp3_metadata(data)
        # Only 2 consecutive frames — 3-frame check should fail
        assert meta is None

    def test_three_frame_sequence_accepted(self):
        """A byte stream with 3+ valid consecutive frames is accepted."""
        import struct
        hdr = struct.pack(">I", 0xFFFB9004)
        frame_len = 144 * 128000 // 44100
        frame = hdr + b"\x00" * (frame_len - 4)
        data = frame * 5
        meta = _parse_mp3_metadata(data)
        assert meta is not None
        assert meta.sample_rate == 44100
        assert meta.frame_count >= 3


# ---------------------------------------------------------------------------
# Synthetic file analysis — determinism
# ---------------------------------------------------------------------------

class TestSyntheticAnalysis:
    def test_deterministic_replay(self):
        """Same synthetic bytes → identical report."""
        data = b"\x00\x01\x02\x03" * 2048
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            r1 = analyze_quantum_audio_file(path)
            r2 = analyze_quantum_audio_file(path)
            assert r1 == r2
        finally:
            os.unlink(path)

    def test_report_is_frozen(self):
        data = b"\xff\xfe" * 1024
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            r = analyze_quantum_audio_file(path)
            with pytest.raises(AttributeError):
                r.filename = "hack"  # type: ignore[misc]
        finally:
            os.unlink(path)

    def test_report_fields_present(self):
        data = bytes(range(256)) * 64
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            r = analyze_quantum_audio_file(path)
            assert isinstance(r.filename, str)
            assert isinstance(r.file_sha256, str)
            assert r.file_size_bytes == len(data)
            assert isinstance(r.mapping_2d, tuple)
            assert isinstance(r.cluster_points, tuple)
            assert isinstance(r.decode_path, str)
            assert 0.0 <= r.coherence_score <= 1.0
        finally:
            os.unlink(path)

    def test_decode_path_label(self):
        data = b"\xab\xcd" * 2048
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            r = analyze_quantum_audio_file(path)
            assert r.decode_path == "byte_pseudospectrum"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Real artifact tests (skipped if files absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_ARTIFACTS, reason="MP3 artifacts not present")
class TestRealArtifacts:
    def test_v1_analysis(self):
        r = analyze_quantum_audio_file(_V1_PATH)
        assert r.file_size_bytes == 2394436
        assert r.filename == "Quantum Coherence Threshold (v1).mp3"
        assert r.file_sha256 == "a1416e98f8449f50152e438900b563218f565c1d0355f69d3c91b6a1f6835b39"
        assert r.sample_rate == 48000
        assert r.duration_seconds == pytest.approx(148.848, abs=0.5)
        assert r.spectral_entropy == pytest.approx(10.992865, abs=1e-3)

    def test_v2_analysis(self):
        r = analyze_quantum_audio_file(_V2_PATH)
        assert r.file_size_bytes == 2892484
        assert r.filename == "Quantum Coherence Threshold (v2).mp3"
        assert r.file_sha256 == "65849d89b7807cea23fe16689959b9966c54d7c9b326feaa99c0432ba3067629"
        assert r.sample_rate == 48000
        assert r.duration_seconds == pytest.approx(179.976, abs=0.5)
        assert r.spectral_entropy == pytest.approx(10.993405, abs=1e-3)

    def test_v2_higher_coherence(self):
        r1 = analyze_quantum_audio_file(_V1_PATH)
        r2 = analyze_quantum_audio_file(_V2_PATH)
        assert r2.coherence_score > r1.coherence_score

    def test_comparison(self):
        r1 = analyze_quantum_audio_file(_V1_PATH)
        r2 = analyze_quantum_audio_file(_V2_PATH)
        comp = compare_reports(r1, r2)
        assert isinstance(comp, ComparativeAnalysisResult)
        assert comp.best_version == "Quantum Coherence Threshold (v2).mp3"
        assert comp.coherence_delta > 0
        assert comp.cluster_tightness_delta < 0  # v2 tighter

    def test_deterministic_replay_v1(self):
        a = analyze_quantum_audio_file(_V1_PATH)
        b = analyze_quantum_audio_file(_V1_PATH)
        assert a == b

    def test_no_filename_inference(self):
        """Verify that analysis is content-based, not name-based."""
        r1 = analyze_quantum_audio_file(_V1_PATH)
        r2 = analyze_quantum_audio_file(_V2_PATH)
        assert r1.file_sha256 != r2.file_sha256
        assert r1.dominant_frequency_hz != r2.dominant_frequency_hz or \
               r1.spectral_entropy != r2.spectral_entropy

    def test_decode_path_is_pseudospectrum_or_waveform(self):
        r = analyze_quantum_audio_file(_V1_PATH)
        assert r.decode_path in ("byte_pseudospectrum",) or \
               r.decode_path.startswith("waveform:")
