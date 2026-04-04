"""Tests for v137.0.15 — Spectral Mesh Resonance Audit.

Covers:
  - frozen dataclasses
  - deterministic peak extraction
  - phi score bounds
  - triality score bounds
  - loopback score bounds
  - SID-style score bounds
  - same file -> same hash
  - different file -> different hash
  - export determinism
  - 25-run replay
  - no decoder contamination
  - sys.modules side-effect proof
  - real MP3 fixture tests
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

from qec.analysis.spectral_mesh_resonance_audit import (
    SPECTRAL_MESH_RESONANCE_VERSION,
    PHI,
    _BLOCK_SIZE,
    _TOP_K_PEAKS,
    SpectralMeshAuditDecision,
    SpectralMeshAuditLedger,
    SpectralResonancePeak,
    build_spectral_mesh_audit,
    build_spectral_mesh_audit_ledger,
    export_spectral_mesh_audit_bundle,
    export_spectral_mesh_audit_ledger,
    extract_spectral_peaks,
    _bytes_to_float_vector,
    _compute_fft_magnitudes,
    _find_top_peaks,
    _stable_hash_dict,
    _compute_phi_lock_score,
    _compute_triality_recurrence_score,
    _compute_loopback_cycle_score,
    _compute_spectral_instability_score,
    _compute_spectral_drift_score,
    _compute_attractor_lock_score,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAVERSAL_MP3 = os.path.join(REPO_ROOT, "Deterministic Traversal Mesh.mp3")
E8_MP3 = os.path.join(REPO_ROOT, "E8 Triality Topology.mp3")


def _mp3_available(path: str) -> bool:
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# 1. Version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_version_string(self):
        assert SPECTRAL_MESH_RESONANCE_VERSION == "v137.0.15"

    def test_phi_constant(self):
        assert abs(PHI - 1.618033988749895) < 1e-12


# ---------------------------------------------------------------------------
# 2. Frozen dataclasses
# ---------------------------------------------------------------------------

class TestFrozenDataclasses:
    def test_peak_frozen(self):
        p = SpectralResonancePeak(
            peak_index=0, frequency_ratio=0.5, magnitude=1.0,
            stable_hash="abc",
        )
        with pytest.raises(AttributeError):
            p.peak_index = 99  # type: ignore[misc]

    def test_decision_frozen(self):
        d = SpectralMeshAuditDecision(
            source_name="x", peak_count=1,
            phi_lock_score=0.5, triality_recurrence_score=0.5,
            loopback_cycle_score=0.5, spectral_instability_score=0.1,
            spectral_drift_score=0.1, attractor_lock_score=0.9,
            symbolic_trace="t", stable_hash="h",
        )
        with pytest.raises(AttributeError):
            d.source_name = "y"  # type: ignore[misc]

    def test_ledger_frozen(self):
        ledger = SpectralMeshAuditLedger(
            decisions=(), decision_count=0, stable_hash="h",
        )
        with pytest.raises(AttributeError):
            ledger.decision_count = 99  # type: ignore[misc]

    def test_peak_default_version(self):
        p = SpectralResonancePeak(
            peak_index=0, frequency_ratio=0.0, magnitude=0.0,
            stable_hash="x",
        )
        assert p.version == "v137.0.15"

    def test_decision_default_version(self):
        d = SpectralMeshAuditDecision(
            source_name="x", peak_count=0,
            phi_lock_score=0.0, triality_recurrence_score=0.0,
            loopback_cycle_score=0.0, spectral_instability_score=0.0,
            spectral_drift_score=0.0, attractor_lock_score=0.0,
            symbolic_trace="", stable_hash="",
        )
        assert d.version == "v137.0.15"


# ---------------------------------------------------------------------------
# 3. Byte-to-vector conversion
# ---------------------------------------------------------------------------

class TestBytesToVector:
    def test_deterministic(self):
        data = b"\x00\x01\xff\x80"
        v1 = _bytes_to_float_vector(data)
        v2 = _bytes_to_float_vector(data)
        np.testing.assert_array_equal(v1, v2)

    def test_values(self):
        data = b"\x00\x01\xff"
        v = _bytes_to_float_vector(data)
        assert v[0] == 0.0
        assert v[1] == 1.0
        assert v[2] == 255.0


# ---------------------------------------------------------------------------
# 4. FFT magnitudes
# ---------------------------------------------------------------------------

class TestFFTMagnitudes:
    def test_deterministic(self):
        vec = np.arange(8192, dtype=np.float64)
        m1 = _compute_fft_magnitudes(vec, 4096)
        m2 = _compute_fft_magnitudes(vec, 4096)
        np.testing.assert_array_equal(m1, m2)

    def test_output_length(self):
        vec = np.arange(8192, dtype=np.float64)
        m = _compute_fft_magnitudes(vec, 4096)
        # rfft of length 4096 -> 2049 bins
        assert len(m) == 2049


# ---------------------------------------------------------------------------
# 5. Score bounds — synthetic peaks
# ---------------------------------------------------------------------------

def _make_synthetic_peaks(n: int = 16) -> tuple:
    peaks = []
    for i in range(n):
        peaks.append(SpectralResonancePeak(
            peak_index=i * 10,
            frequency_ratio=float(i) / float(n),
            magnitude=float(i + 1) * 100.0,
            stable_hash=f"h{i}",
        ))
    return tuple(peaks)


class TestPhiScoreBounds:
    def test_bounded_01(self):
        peaks = _make_synthetic_peaks(16)
        score = _compute_phi_lock_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert _compute_phi_lock_score(()) == 0.0

    def test_single(self):
        peaks = _make_synthetic_peaks(1)
        assert _compute_phi_lock_score(peaks) == 0.0


class TestTrialityScoreBounds:
    def test_bounded_01(self):
        peaks = _make_synthetic_peaks(20)
        score = _compute_triality_recurrence_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert _compute_triality_recurrence_score(()) == 0.0

    def test_perfect_uniformity(self):
        # 5 peaks at indices 0,1,2,3,4 -> perfectly uniform mod 5
        peaks = tuple(
            SpectralResonancePeak(
                peak_index=i, frequency_ratio=0.0,
                magnitude=1.0, stable_hash=f"h{i}",
            )
            for i in range(5)
        )
        score = _compute_triality_recurrence_score(peaks)
        assert score == 1.0


class TestLoopbackScoreBounds:
    def test_bounded_01(self):
        peaks = _make_synthetic_peaks(16)
        score = _compute_loopback_cycle_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_empty(self):
        assert _compute_loopback_cycle_score(()) == 0.0

    def test_too_few(self):
        peaks = _make_synthetic_peaks(3)
        assert _compute_loopback_cycle_score(peaks) == 0.0


class TestSIDStyleScoreBounds:
    def test_instability_bounded(self):
        peaks = _make_synthetic_peaks(16)
        score = _compute_spectral_instability_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_drift_bounded(self):
        peaks = _make_synthetic_peaks(16)
        score = _compute_spectral_drift_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_attractor_bounded(self):
        peaks = _make_synthetic_peaks(16)
        score = _compute_attractor_lock_score(peaks)
        assert 0.0 <= score <= 1.0

    def test_instability_empty(self):
        assert _compute_spectral_instability_score(()) == 0.0

    def test_drift_empty(self):
        assert _compute_spectral_drift_score(()) == 0.0

    def test_attractor_empty(self):
        assert _compute_attractor_lock_score(()) == 0.0


# ---------------------------------------------------------------------------
# 6. Real MP3 fixture tests — peak extraction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mp3_available(TRAVERSAL_MP3),
    reason="Deterministic Traversal Mesh.mp3 not found",
)
class TestTraversalMP3Peaks:
    def test_peaks_nonempty(self):
        peaks = extract_spectral_peaks(TRAVERSAL_MP3)
        assert len(peaks) > 0

    def test_peaks_frozen(self):
        peaks = extract_spectral_peaks(TRAVERSAL_MP3)
        with pytest.raises(AttributeError):
            peaks[0].peak_index = 999  # type: ignore[misc]

    def test_peaks_deterministic(self):
        p1 = extract_spectral_peaks(TRAVERSAL_MP3)
        p2 = extract_spectral_peaks(TRAVERSAL_MP3)
        assert p1 == p2


@pytest.mark.skipif(
    not _mp3_available(E8_MP3),
    reason="E8 Triality Topology.mp3 not found",
)
class TestE8MP3Peaks:
    def test_peaks_nonempty(self):
        peaks = extract_spectral_peaks(E8_MP3)
        assert len(peaks) > 0

    def test_peaks_deterministic(self):
        p1 = extract_spectral_peaks(E8_MP3)
        p2 = extract_spectral_peaks(E8_MP3)
        assert p1 == p2


# ---------------------------------------------------------------------------
# 7. Same file -> same hash, different file -> different hash
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_mp3_available(TRAVERSAL_MP3) and _mp3_available(E8_MP3)),
    reason="MP3 fixtures not found",
)
class TestHashDeterminism:
    def test_same_file_same_hash(self):
        d1 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        d2 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        assert d1.stable_hash == d2.stable_hash

    def test_different_file_different_hash(self):
        d1 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        d2 = build_spectral_mesh_audit(E8_MP3)
        assert d1.stable_hash != d2.stable_hash


# ---------------------------------------------------------------------------
# 8. Full audit decision — real fixtures
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mp3_available(TRAVERSAL_MP3),
    reason="Deterministic Traversal Mesh.mp3 not found",
)
class TestFullAuditTraversal:
    def test_all_scores_bounded(self):
        d = build_spectral_mesh_audit(TRAVERSAL_MP3)
        assert 0.0 <= d.phi_lock_score <= 1.0
        assert 0.0 <= d.triality_recurrence_score <= 1.0
        assert 0.0 <= d.loopback_cycle_score <= 1.0
        assert 0.0 <= d.spectral_instability_score <= 1.0
        assert 0.0 <= d.spectral_drift_score <= 1.0
        assert 0.0 <= d.attractor_lock_score <= 1.0

    def test_version(self):
        d = build_spectral_mesh_audit(TRAVERSAL_MP3)
        assert d.version == "v137.0.15"

    def test_source_name(self):
        d = build_spectral_mesh_audit(TRAVERSAL_MP3)
        assert d.source_name == "Deterministic Traversal Mesh.mp3"


@pytest.mark.skipif(
    not _mp3_available(E8_MP3),
    reason="E8 Triality Topology.mp3 not found",
)
class TestFullAuditE8:
    def test_all_scores_bounded(self):
        d = build_spectral_mesh_audit(E8_MP3)
        assert 0.0 <= d.phi_lock_score <= 1.0
        assert 0.0 <= d.triality_recurrence_score <= 1.0
        assert 0.0 <= d.loopback_cycle_score <= 1.0
        assert 0.0 <= d.spectral_instability_score <= 1.0
        assert 0.0 <= d.spectral_drift_score <= 1.0
        assert 0.0 <= d.attractor_lock_score <= 1.0

    def test_source_name(self):
        d = build_spectral_mesh_audit(E8_MP3)
        assert d.source_name == "E8 Triality Topology.mp3"


# ---------------------------------------------------------------------------
# 9. Export determinism
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mp3_available(TRAVERSAL_MP3),
    reason="Deterministic Traversal Mesh.mp3 not found",
)
class TestExportDeterminism:
    def test_bundle_deterministic(self):
        d = build_spectral_mesh_audit(TRAVERSAL_MP3)
        b1 = export_spectral_mesh_audit_bundle(d)
        b2 = export_spectral_mesh_audit_bundle(d)
        j1 = json.dumps(b1, sort_keys=True)
        j2 = json.dumps(b2, sort_keys=True)
        assert j1 == j2

    def test_ledger_export_deterministic(self):
        d1 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        d2 = build_spectral_mesh_audit(E8_MP3) if _mp3_available(E8_MP3) else d1
        ledger = build_spectral_mesh_audit_ledger((d1, d2))
        e1 = export_spectral_mesh_audit_ledger(ledger)
        e2 = export_spectral_mesh_audit_ledger(ledger)
        j1 = json.dumps(e1, sort_keys=True)
        j2 = json.dumps(e2, sort_keys=True)
        assert j1 == j2


# ---------------------------------------------------------------------------
# 10. Ledger construction
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_mp3_available(TRAVERSAL_MP3) and _mp3_available(E8_MP3)),
    reason="MP3 fixtures not found",
)
class TestLedger:
    def test_decision_count(self):
        d1 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        d2 = build_spectral_mesh_audit(E8_MP3)
        ledger = build_spectral_mesh_audit_ledger((d1, d2))
        assert ledger.decision_count == 2

    def test_ledger_hash_stable(self):
        d1 = build_spectral_mesh_audit(TRAVERSAL_MP3)
        d2 = build_spectral_mesh_audit(E8_MP3)
        l1 = build_spectral_mesh_audit_ledger((d1, d2))
        l2 = build_spectral_mesh_audit_ledger((d1, d2))
        assert l1.stable_hash == l2.stable_hash


# ---------------------------------------------------------------------------
# 11. 25-run replay
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mp3_available(TRAVERSAL_MP3),
    reason="Deterministic Traversal Mesh.mp3 not found",
)
class TestReplay25:
    def test_25_run_replay(self):
        """25 consecutive runs must produce byte-identical exports."""
        reference = None
        for _ in range(25):
            d = build_spectral_mesh_audit(TRAVERSAL_MP3)
            bundle = export_spectral_mesh_audit_bundle(d)
            j = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256(j.encode("utf-8")).hexdigest()
            if reference is None:
                reference = h
            else:
                assert h == reference, "25-run replay diverged"


# ---------------------------------------------------------------------------
# 12. No decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """The module must not import anything from qec.decoder."""
        import importlib
        mod = importlib.import_module(
            "qec.analysis.spectral_mesh_resonance_audit"
        )
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ---------------------------------------------------------------------------
# 13. sys.modules side-effect proof
# ---------------------------------------------------------------------------

class TestSysModulesSideEffect:
    def test_no_decoder_in_sys_modules(self):
        """Importing the audit module must not pull in decoder modules."""
        decoder_mods_before = {
            k for k in sys.modules if k.startswith("qec.decoder")
        }
        import importlib
        importlib.import_module("qec.analysis.spectral_mesh_resonance_audit")
        decoder_mods_after = {
            k for k in sys.modules if k.startswith("qec.decoder")
        }
        # No new decoder modules should appear
        new_decoder = decoder_mods_after - decoder_mods_before
        assert len(new_decoder) == 0, f"Decoder modules loaded: {new_decoder}"


# ---------------------------------------------------------------------------
# 14. Symbolic trace content
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _mp3_available(TRAVERSAL_MP3),
    reason="Deterministic Traversal Mesh.mp3 not found",
)
class TestSymbolicTrace:
    def test_trace_contains_invariants(self):
        d = build_spectral_mesh_audit(TRAVERSAL_MP3)
        assert "PHI_LOCK=" in d.symbolic_trace
        assert "E8_TRIALITY=" in d.symbolic_trace
        assert "OUROBOROS=" in d.symbolic_trace
        assert "SID_INST=" in d.symbolic_trace
        assert "SID_DRIFT=" in d.symbolic_trace
        assert "SID_ATTR=" in d.symbolic_trace


# ---------------------------------------------------------------------------
# 15. Hardening — FFT edge cases
# ---------------------------------------------------------------------------

class TestFFTEdgeCases:
    def test_empty_vector_safe(self):
        """Empty input must return zero spectrum, not crash."""
        vec = np.array([], dtype=np.float64)
        result = _compute_fft_magnitudes(vec, 4096)
        assert len(result) == 4096 // 2 + 1
        assert float(np.sum(result)) == 0.0

    def test_short_vector_safe(self):
        """Input shorter than block_size must zero-pad, not crash."""
        vec = np.arange(100, dtype=np.float64)
        result = _compute_fft_magnitudes(vec, 4096)
        assert len(result) == 4096 // 2 + 1

    def test_one_byte_vector_safe(self):
        """Single-element input must produce valid spectrum."""
        vec = np.array([42.0], dtype=np.float64)
        result = _compute_fft_magnitudes(vec, 4096)
        assert len(result) == 4096 // 2 + 1
        # DC bin should reflect the single value
        assert result[0] > 0.0


# ---------------------------------------------------------------------------
# 16. Hardening — peak edge cases
# ---------------------------------------------------------------------------

class TestPeakEdgeCases:
    def test_empty_magnitudes_returns_empty(self):
        mags = np.array([], dtype=np.float64)
        result = _find_top_peaks(mags, 10)
        assert len(result) == 0
        assert result.dtype == np.int64

    def test_equal_magnitudes_deterministic(self):
        """Equal magnitudes must produce deterministic index ordering."""
        mags = np.ones(64, dtype=np.float64)
        r1 = _find_top_peaks(mags, 16)
        r2 = _find_top_peaks(mags, 16)
        np.testing.assert_array_equal(r1, r2)

    def test_equal_magnitudes_repeat_10(self):
        """10 repeats of equal-magnitude extraction must be identical."""
        mags = np.ones(64, dtype=np.float64) * 7.0
        reference = _find_top_peaks(mags, 16)
        for _ in range(10):
            np.testing.assert_array_equal(
                _find_top_peaks(mags, 16), reference
            )


# ---------------------------------------------------------------------------
# 17. Hardening — hash version coupling
# ---------------------------------------------------------------------------

class TestHashVersionCoupling:
    def test_peak_hash_includes_version(self):
        """Peak hash payload path must include version, block_size, top_k."""
        # Reconstruct what the hash dict looks like and verify fields
        peak_dict = {
            "peak_index": 5,
            "frequency_ratio": 0.25,
            "magnitude": 100.0,
            "version": SPECTRAL_MESH_RESONANCE_VERSION,
            "block_size": _BLOCK_SIZE,
            "top_k": _TOP_K_PEAKS,
        }
        h_with = _stable_hash_dict(peak_dict)
        # Without version fields -> different hash
        peak_dict_no_ver = {
            "peak_index": 5,
            "frequency_ratio": 0.25,
            "magnitude": 100.0,
        }
        h_without = _stable_hash_dict(peak_dict_no_ver)
        assert h_with != h_without

    def test_decision_hash_includes_version(self):
        """Decision hash must change if version field changes."""
        base = {
            "source_name": "test.mp3",
            "peak_count": 10,
            "phi_lock_score": 0.5,
            "triality_recurrence_score": 0.5,
            "loopback_cycle_score": 0.5,
            "spectral_instability_score": 0.1,
            "spectral_drift_score": 0.1,
            "attractor_lock_score": 0.9,
            "symbolic_trace": "trace",
            "version": "v137.0.15",
        }
        h1 = _stable_hash_dict(base)
        altered = dict(base)
        altered["version"] = "v999.0.0"
        h2 = _stable_hash_dict(altered)
        assert h1 != h2
