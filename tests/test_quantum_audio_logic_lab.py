"""Tests for quantum_audio_logic_lab — deterministic spectral analysis."""

import hashlib
import math
import os
import tempfile

import pytest

import numpy as np

from qec.audio.quantum_audio_logic_lab import (
    AudioStackHealthReport,
    ComparativeAnalysisResult,
    CrossEraComparisonReport,
    LegacyAudioSequenceReport,
    LegacySequenceTransitionReport,
    QuantumAudioLogicReport,
    _coherence_score,
    _cluster_tightness,
    _compute_byte_pseudospectrum,
    _compute_waveform_spectrum,
    _deterministic_clusters,
    _get_dependency_tuples,
    _LEGACY_STATE_FILES,
    _LEGACY_STATE_NAMES,
    _LEGACY_TRANSITION_CLASSES,
    _load_samples,
    _parse_mp3_metadata,
    _psd_similarity,
    _stability_label,
    _try_decode_audio,
    _welch_psd,
    analyze_legacy_audio_sequence,
    analyze_quantum_audio_file,
    compare_legacy_vs_v1366,
    compare_reports,
    verify_audio_stack_health,
)

# ---------------------------------------------------------------------------
# Paths to real artifacts (skip if not present)
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V1_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v1).mp3")
_V2_PATH = os.path.join(_REPO_ROOT, "Quantum Coherence Threshold (v2).mp3")

_HAS_ARTIFACTS = os.path.isfile(_V1_PATH) and os.path.isfile(_V2_PATH)

_SONIC_DIR = os.path.join(_REPO_ROOT, "artifacts", "sonic")
_BASELINE_JSON = os.path.join(_SONIC_DIR, "sequence_analysis.json")

_LEGACY_PATHS = [
    os.path.join(_REPO_ROOT, fname) for fname in _LEGACY_STATE_FILES
]
_HAS_LEGACY = all(os.path.isfile(p) for p in _LEGACY_PATHS)


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


# ---------------------------------------------------------------------------
# v136.6.1 — PSD similarity
# ---------------------------------------------------------------------------

class TestPsdSimilarity:
    def test_identical_signals(self):
        """Identical signals yield similarity ~1.0."""
        s = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
        sim = _psd_similarity(s, 44100, s, 44100)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_different_signals(self):
        """Different signals yield similarity < 1.0."""
        sa = np.sin(2 * np.pi * 200 * np.arange(44100) / 44100)
        sb = np.sin(2 * np.pi * 4000 * np.arange(44100) / 44100)
        sim = _psd_similarity(sa, 44100, sb, 44100)
        assert sim < 1.0

    def test_empty_signal(self):
        sim = _psd_similarity(np.array([]), 44100, np.array([1.0]), 44100)
        assert sim == 0.0

    def test_deterministic(self):
        s = np.random.RandomState(42).randn(8192)
        a = _psd_similarity(s, 44100, s, 44100)
        b = _psd_similarity(s, 44100, s, 44100)
        assert a == b


# ---------------------------------------------------------------------------
# v136.6.1 — Load samples helper
# ---------------------------------------------------------------------------

class TestLoadSamples:
    def test_synthetic_file(self):
        data = b"\x00\x01\x02\x03" * 2048
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            samples, sr = _load_samples(path)
            assert len(samples) > 0
            assert sr > 0
        finally:
            os.unlink(path)

    def test_deterministic_load(self):
        data = bytes(range(256)) * 32
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            s1, sr1 = _load_samples(path)
            s2, sr2 = _load_samples(path)
            assert sr1 == sr2
            assert np.array_equal(s1, s2)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# v136.6.1 — Dependency hardening tests
# ---------------------------------------------------------------------------

class TestAudioStackHealth:
    def test_report_type(self):
        report = verify_audio_stack_health()
        assert isinstance(report, AudioStackHealthReport)

    def test_report_is_frozen(self):
        report = verify_audio_stack_health()
        with pytest.raises(AttributeError):
            report.numpy_version = "hack"  # type: ignore[misc]

    def test_numpy_available(self):
        report = verify_audio_stack_health()
        assert report.numpy_version != "not_installed"

    def test_scipy_available(self):
        report = verify_audio_stack_health()
        assert report.scipy_version != "not_installed"
        assert report.scipy_signal_available is True

    def test_pseudo_spectrum_always_ready(self):
        report = verify_audio_stack_health()
        assert report.pseudo_spectrum_ready is True

    def test_operating_mode_valid(self):
        report = verify_audio_stack_health()
        assert report.operating_mode in (
            "pseudo_spectrum", "true_waveform_decode",
        )

    def test_active_decode_path_not_empty(self):
        report = verify_audio_stack_health()
        assert len(report.active_decode_path) > 0

    def test_deterministic_replay(self):
        a = verify_audio_stack_health()
        b = verify_audio_stack_health()
        assert a == b


class TestDependencyTuples:
    def test_returns_tuple(self):
        deps = _get_dependency_tuples()
        assert isinstance(deps, tuple)

    def test_sorted(self):
        deps = _get_dependency_tuples()
        names = [d[0] for d in deps]
        assert names == sorted(names)

    def test_contains_numpy(self):
        deps = _get_dependency_tuples()
        names = [d[0] for d in deps]
        assert "numpy" in names

    def test_contains_scipy(self):
        deps = _get_dependency_tuples()
        names = [d[0] for d in deps]
        assert "scipy" in names


# ---------------------------------------------------------------------------
# v136.6.1 — Legacy constants
# ---------------------------------------------------------------------------

class TestLegacyConstants:
    def test_state_count(self):
        assert len(_LEGACY_STATE_NAMES) == 5

    def test_file_count(self):
        assert len(_LEGACY_STATE_FILES) == 5

    def test_transition_count(self):
        assert len(_LEGACY_TRANSITION_CLASSES) == 4

    def test_state_chain_order(self):
        assert _LEGACY_STATE_NAMES == (
            "Stable", "Instability", "Transition", "Collapse", "Recovery",
        )


# ---------------------------------------------------------------------------
# v136.6.1 — Legacy sequence regression (real artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LEGACY, reason="Legacy MP3 artifacts not present")
class TestLegacySequenceRegression:
    def test_analyse_returns_report(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert isinstance(report, LegacyAudioSequenceReport)

    def test_n_states(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert report.n_states == 5

    def test_transition_count(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert len(report.transition_reports) == 4

    def test_transition_classifications(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        classes = tuple(t.classification for t in report.transition_reports)
        assert classes == ("divergent", "transition", "collapse", "recovery")

    def test_legacy_similarity_bounded(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert 0.0 <= report.legacy_similarity_score <= 1.0

    def test_stability_verdict_valid(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert report.stability_verdict in (
            "stable_regression", "partial_regression", "degraded_regression",
        )

    def test_best_recovery_match_is_filename(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        valid_names = set(_LEGACY_STATE_FILES)
        assert report.best_recovery_match in valid_names

    def test_frozen(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        with pytest.raises(AttributeError):
            report.n_states = 99  # type: ignore[misc]

    def test_deterministic_replay(self):
        a = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        b = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        assert a == b

    def test_transition_psd_similarity_bounded(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        for t in report.transition_reports:
            assert 0.0 <= t.psd_similarity <= 1.0

    def test_dependency_health_present(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        assert len(report.dependency_health) > 0

    def test_without_baseline(self):
        report = analyze_legacy_audio_sequence(_REPO_ROOT, None)
        assert report.n_states == 5
        # Without baseline, centroid_drift should be 0
        for t in report.transition_reports:
            assert t.centroid_drift == 0.0


# ---------------------------------------------------------------------------
# v136.6.1 — JSON baseline comparison
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LEGACY, reason="Legacy MP3 artifacts not present")
class TestJsonBaselineComparison:
    def test_baseline_json_loads(self):
        import json
        with open(_BASELINE_JSON) as f:
            data = json.load(f)
        assert data["n_states"] == 5
        assert len(data["transitions"]) == 4

    def test_centroid_drift_finite(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        for t in report.transition_reports:
            assert math.isfinite(t.centroid_drift)

    def test_legacy_centroid_delta_nonzero(self):
        report = analyze_legacy_audio_sequence(
            _REPO_ROOT, _BASELINE_JSON,
        )
        # At least one legacy centroid delta should be non-zero
        any_nonzero = any(
            t.legacy_centroid_delta != 0.0
            for t in report.transition_reports
        )
        assert any_nonzero


# ---------------------------------------------------------------------------
# v136.6.1 — Cross-era comparison (real artifacts)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_HAS_LEGACY and _HAS_ARTIFACTS),
    reason="MP3 artifacts not present",
)
class TestCrossEraComparison:
    def test_returns_report(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        result = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        assert isinstance(result, CrossEraComparisonReport)

    def test_best_analogs_are_filenames(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        result = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        valid = set(_LEGACY_STATE_FILES)
        assert result.best_legacy_analog_v1 in valid
        assert result.best_legacy_analog_v2 in valid

    def test_similarity_bounded(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        result = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        assert 0.0 <= result.collapse_similarity <= 1.0
        assert 0.0 <= result.recovery_similarity <= 1.0
        assert 0.0 <= result.topology_alignment_score <= 1.0

    def test_frozen(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        result = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        with pytest.raises(AttributeError):
            result.collapse_similarity = 0.0  # type: ignore[misc]

    def test_deterministic(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        a = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        b = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        assert a == b

    def test_v2_recovery_flag_is_bool(self):
        legacy = analyze_legacy_audio_sequence(_REPO_ROOT, _BASELINE_JSON)
        result = compare_legacy_vs_v1366(legacy, _V1_PATH, _V2_PATH)
        assert isinstance(result.v2_resembles_recovery, bool)


# ---------------------------------------------------------------------------
# v136.6.1 — Decode path tests
# ---------------------------------------------------------------------------

class TestDecodePathHardening:
    def test_pseudospectrum_for_raw_bytes(self):
        data = bytes(range(256)) * 64
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            r = analyze_quantum_audio_file(path)
            assert r.decode_path == "byte_pseudospectrum"
        finally:
            os.unlink(path)

    @pytest.mark.skipif(not _HAS_ARTIFACTS, reason="MP3 artifacts not present")
    def test_mp3_decode_path_label(self):
        r = analyze_quantum_audio_file(_V1_PATH)
        # Either pseudospectrum or waveform — both valid
        assert "pseudospectrum" in r.decode_path or "waveform" in r.decode_path
