"""Tests for quantum_audio_logic_lab — deterministic spectral analysis."""

import hashlib
import math
import os
import tempfile

import pytest

from qec.audio.quantum_audio_logic_lab import (
    ComparativeAnalysisResult,
    QuantumAudioLogicReport,
    _coherence_score,
    _cluster_tightness,
    _deterministic_clusters,
    _stability_label,
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
            assert r.sample_rate == 44100
            assert isinstance(r.mapping_2d, tuple)
            assert isinstance(r.cluster_points, tuple)
            assert 0.0 <= r.coherence_score <= 1.0
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
        assert r.spectral_entropy == pytest.approx(10.992885, abs=1e-4)

    def test_v2_analysis(self):
        r = analyze_quantum_audio_file(_V2_PATH)
        assert r.file_size_bytes == 2892484
        assert r.filename == "Quantum Coherence Threshold (v2).mp3"
        assert r.file_sha256 == "65849d89b7807cea23fe16689959b9966c54d7c9b326feaa99c0432ba3067629"
        assert r.spectral_entropy == pytest.approx(10.993427, abs=1e-4)

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
        # Different files must produce different sha256 hashes
        assert r1.file_sha256 != r2.file_sha256
        # And different spectral values
        assert r1.dominant_frequency_hz != r2.dominant_frequency_hz
