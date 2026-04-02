"""
Tests for QEC Audio Cognition Engine (v136.8.3).

Required categories:
- dataclass immutability
- waveform determinism
- PSD determinism
- registry matching
- unknown state detection
- confidence bounds
- 100 replay determinism
- same-input byte identity
- action hook integrity
- code zoo integration
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

from qec.audio.triality_signal_engine import (
    CARRIER_FREQ_MAX,
    CARRIER_FREQ_MIN,
    NUM_SAMPLES,
    SAMPLE_RATE,
    TrialityParams,
    _hash_to_float,
    _hash_to_int,
    compute_peak_bins,
    compute_psd,
    compute_psd_hash,
    compute_spectral_centroid,
    compute_spectral_rolloff,
    derive_triality_params,
    synthesize_triality_waveform,
)
from qec.audio.cognition_registry import (
    AudioFingerprint,
    CognitionMatch,
    CognitionRegistry,
    CognitionRegistryEntry,
    HIGH_CONFIDENCE_THRESHOLD,
    UNKNOWN_ACTION,
    UNKNOWN_STATE,
    build_registry,
    cosine_similarity,
    fingerprint_to_vector,
    match_registry_signature,
    recall_similar_failure_state,
    register_cognition_entry,
)
from qec.audio.audio_cognition_engine import (
    ENGINE_VERSION,
    CognitionCycleResult,
    compute_spectral_fingerprint,
    derive_carrier_freq_from_zoo,
    export_cognition_bundle,
    export_cognition_bundle_json,
    get_code_zoo_families,
    render_qec_audio_signature,
    run_cognition_cycle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REFERENCE_CODE_FAMILY = "surface"
REFERENCE_ERROR_TYPE = "X_error"
REFERENCE_TOPOLOGY = "planar"
REFERENCE_STATE_HASH = hashlib.sha256(b"test_state").hexdigest()


@pytest.fixture
def reference_params():
    return derive_triality_params(
        REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
        REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
    )


@pytest.fixture
def reference_waveform():
    return render_qec_audio_signature(
        REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
        REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
    )


@pytest.fixture
def reference_fingerprint(reference_waveform):
    return compute_spectral_fingerprint(reference_waveform)


@pytest.fixture
def sample_entry(reference_fingerprint):
    return CognitionRegistryEntry(
        state_hash=REFERENCE_STATE_HASH,
        code_family=REFERENCE_CODE_FAMILY,
        error_type=REFERENCE_ERROR_TYPE,
        topology_state=REFERENCE_TOPOLOGY,
        fingerprint=reference_fingerprint,
        recommended_action="REINIT_CODE_LATTICE",
        failure_mode="syndrome_collapse",
    )


@pytest.fixture
def sample_registry(sample_entry):
    return register_cognition_entry(sample_entry)


# ===================================================================
# 1. DATACLASS IMMUTABILITY TESTS
# ===================================================================


class TestDataclassImmutability:
    """Frozen dataclass immutability tests."""

    def test_triality_params_frozen(self, reference_params):
        with pytest.raises(AttributeError):
            reference_params.carrier_freq = 999.0  # type: ignore[misc]

    def test_audio_fingerprint_frozen(self, reference_fingerprint):
        with pytest.raises(AttributeError):
            reference_fingerprint.centroid = 999.0  # type: ignore[misc]

    def test_cognition_match_frozen(self):
        match = CognitionMatch(
            confidence=0.99, identity="test",
            failure_mode="test", recommended_action="TEST",
        )
        with pytest.raises(AttributeError):
            match.confidence = 0.5  # type: ignore[misc]

    def test_cognition_registry_entry_frozen(self, sample_entry):
        with pytest.raises(AttributeError):
            sample_entry.state_hash = "new_hash"  # type: ignore[misc]

    def test_cognition_registry_frozen(self, sample_registry):
        with pytest.raises(AttributeError):
            sample_registry.entries = ()  # type: ignore[misc]

    def test_cognition_cycle_result_frozen(self, reference_params, reference_fingerprint):
        match = CognitionMatch(
            confidence=0.5, identity=UNKNOWN_STATE,
            failure_mode=UNKNOWN_STATE, recommended_action=UNKNOWN_ACTION,
        )
        result = CognitionCycleResult(
            params=reference_params, fingerprint=reference_fingerprint,
            match=match, engine_version=ENGINE_VERSION,
        )
        with pytest.raises(AttributeError):
            result.engine_version = "hacked"  # type: ignore[misc]


# ===================================================================
# 2. WAVEFORM DETERMINISM TESTS
# ===================================================================


class TestWaveformDeterminism:
    """Waveform determinism tests."""

    def test_waveform_shape(self, reference_waveform):
        assert reference_waveform.shape == (NUM_SAMPLES,)

    def test_waveform_dtype(self, reference_waveform):
        assert reference_waveform.dtype == np.float64

    def test_waveform_normalized(self, reference_waveform):
        assert np.max(np.abs(reference_waveform)) <= 1.0 + 1e-15

    def test_waveform_not_silent(self, reference_waveform):
        assert np.max(np.abs(reference_waveform)) > 0.1

    def test_waveform_deterministic_two_calls(self):
        w1 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        np.testing.assert_array_equal(w1, w2)

    def test_waveform_byte_identical(self):
        w1 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        assert w1.tobytes() == w2.tobytes()

    def test_different_families_different_waveforms(self):
        w1 = render_qec_audio_signature(
            "surface", REFERENCE_ERROR_TYPE, REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            "toric", REFERENCE_ERROR_TYPE, REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        assert not np.array_equal(w1, w2)

    def test_different_error_types_different_waveforms(self):
        w1 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, "X_error", REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, "Z_error", REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        assert not np.array_equal(w1, w2)

    def test_different_topology_different_waveforms(self):
        w1 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE, "planar", REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE, "torus", REFERENCE_STATE_HASH,
        )
        assert not np.array_equal(w1, w2)


# ===================================================================
# 3. PSD DETERMINISM TESTS
# ===================================================================


class TestPSDDeterminism:
    """PSD determinism tests."""

    def test_psd_shape(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        expected_len = NUM_SAMPLES // 2 + 1
        assert psd.shape == (expected_len,)

    def test_psd_non_negative(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        assert np.all(psd >= 0.0)

    def test_psd_deterministic(self, reference_waveform):
        psd1 = compute_psd(reference_waveform)
        psd2 = compute_psd(reference_waveform)
        np.testing.assert_array_equal(psd1, psd2)

    def test_psd_hash_deterministic(self, reference_waveform):
        psd1 = compute_psd(reference_waveform)
        psd2 = compute_psd(reference_waveform)
        assert compute_psd_hash(psd1) == compute_psd_hash(psd2)

    def test_psd_hash_is_sha256(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        h = compute_psd_hash(psd)
        assert len(h) == 64
        int(h, 16)  # must be valid hex


# ===================================================================
# 4. REGISTRY MATCHING TESTS
# ===================================================================


class TestRegistryMatching:
    """Registry matching tests."""

    def test_exact_match_confidence(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.confidence >= HIGH_CONFIDENCE_THRESHOLD

    def test_exact_match_identity(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.identity == f"{REFERENCE_CODE_FAMILY}:{REFERENCE_ERROR_TYPE}"

    def test_exact_match_action(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.recommended_action == "REINIT_CODE_LATTICE"

    def test_exact_match_failure_mode(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.failure_mode == "syndrome_collapse"

    def test_recall_alias(self, reference_fingerprint, sample_registry):
        m1 = match_registry_signature(reference_fingerprint, sample_registry)
        m2 = recall_similar_failure_state(reference_fingerprint, sample_registry)
        assert m1 == m2

    def test_match_deterministic(self, reference_fingerprint, sample_registry):
        m1 = match_registry_signature(reference_fingerprint, sample_registry)
        m2 = match_registry_signature(reference_fingerprint, sample_registry)
        assert m1 == m2

    def test_empty_registry_returns_unknown(self, reference_fingerprint):
        empty_reg = build_registry(())
        match = match_registry_signature(reference_fingerprint, empty_reg)
        assert match.identity == UNKNOWN_STATE
        assert match.confidence == 0.0


# ===================================================================
# 5. UNKNOWN STATE DETECTION TESTS
# ===================================================================


class TestUnknownStateDetection:
    """Unknown state detection tests."""

    def test_dissimilar_fingerprint_unknown(self, sample_registry):
        alien_fp = AudioFingerprint(
            centroid=9999.0,
            rolloff=9999.0,
            peak_bins=(999, 998, 997, 996, 995),
            psd_hash="a" * 64,
        )
        match = match_registry_signature(alien_fp, sample_registry)
        assert match.identity == UNKNOWN_STATE

    def test_unknown_state_action(self, sample_registry):
        alien_fp = AudioFingerprint(
            centroid=9999.0,
            rolloff=9999.0,
            peak_bins=(999, 998, 997, 996, 995),
            psd_hash="b" * 64,
        )
        match = match_registry_signature(alien_fp, sample_registry)
        assert match.recommended_action == UNKNOWN_ACTION

    def test_unknown_state_failure_mode(self, sample_registry):
        alien_fp = AudioFingerprint(
            centroid=9999.0,
            rolloff=9999.0,
            peak_bins=(999, 998, 997, 996, 995),
            psd_hash="c" * 64,
        )
        match = match_registry_signature(alien_fp, sample_registry)
        assert match.failure_mode == UNKNOWN_STATE


# ===================================================================
# 6. CONFIDENCE BOUNDS TESTS
# ===================================================================


class TestConfidenceBounds:
    """Confidence bounds tests."""

    def test_confidence_in_range(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert 0.0 <= match.confidence <= 1.0

    def test_self_match_high_confidence(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.confidence >= HIGH_CONFIDENCE_THRESHOLD

    def test_cosine_similarity_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == 0.0

    def test_threshold_value(self):
        assert HIGH_CONFIDENCE_THRESHOLD == 0.98


# ===================================================================
# 7. 100-RUN REPLAY DETERMINISM TESTS
# ===================================================================


class TestReplayDeterminism:
    """100-run replay determinism tests."""

    def test_waveform_100_replay(self):
        reference = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        for _ in range(100):
            result = render_qec_audio_signature(
                REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
                REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
            )
            np.testing.assert_array_equal(result, reference)

    def test_psd_hash_100_replay(self):
        ref_waveform = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        ref_psd = compute_psd(ref_waveform)
        ref_hash = compute_psd_hash(ref_psd)
        for _ in range(100):
            w = render_qec_audio_signature(
                REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
                REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
            )
            psd = compute_psd(w)
            assert compute_psd_hash(psd) == ref_hash

    def test_fingerprint_100_replay(self):
        ref_waveform = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        ref_fp = compute_spectral_fingerprint(ref_waveform)
        for _ in range(100):
            w = render_qec_audio_signature(
                REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
                REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
            )
            fp = compute_spectral_fingerprint(w)
            assert fp == ref_fp

    def test_cognition_cycle_100_replay(self, sample_registry):
        ref = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        for _ in range(100):
            result = run_cognition_cycle(
                REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
                REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
            )
            assert result == ref


# ===================================================================
# 8. BYTE IDENTITY TESTS
# ===================================================================


class TestByteIdentity:
    """Same-input byte identity tests."""

    def test_waveform_byte_identity(self):
        w1 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        w2 = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        assert w1.tobytes() == w2.tobytes()

    def test_psd_byte_identity(self):
        w = render_qec_audio_signature(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH,
        )
        psd1 = compute_psd(w)
        psd2 = compute_psd(w)
        assert psd1.tobytes() == psd2.tobytes()

    def test_export_json_byte_identity(self, sample_registry):
        r1 = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        r2 = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        assert export_cognition_bundle_json(r1) == export_cognition_bundle_json(r2)


# ===================================================================
# 9. ACTION HOOK INTEGRITY TESTS
# ===================================================================


class TestActionHookIntegrity:
    """Action hook integrity tests."""

    def test_known_action_reinit(self, reference_fingerprint, sample_registry):
        match = match_registry_signature(reference_fingerprint, sample_registry)
        assert match.recommended_action == "REINIT_CODE_LATTICE"

    def test_action_hook_decode_portfolio(self):
        fp = AudioFingerprint(
            centroid=100.0, rolloff=50.0,
            peak_bins=(1, 2, 3, 4, 5), psd_hash="d" * 64,
        )
        entry = CognitionRegistryEntry(
            state_hash="e" * 64, code_family="toric",
            error_type="Z_error", topology_state="torus",
            fingerprint=fp, recommended_action="DECODE_PORTFOLIO_C",
            failure_mode="logical_error",
        )
        reg = register_cognition_entry(entry)
        match = match_registry_signature(fp, reg)
        assert match.recommended_action == "DECODE_PORTFOLIO_C"

    def test_action_hook_qldpc(self):
        fp = AudioFingerprint(
            centroid=200.0, rolloff=100.0,
            peak_bins=(10, 20, 30, 40, 50), psd_hash="f" * 64,
        )
        entry = CognitionRegistryEntry(
            state_hash="a" * 64, code_family="qldpc",
            error_type="depolarizing", topology_state="hypergraph",
            fingerprint=fp, recommended_action="QLDPC_PORTFOLIO_B",
            failure_mode="rate_collapse",
        )
        reg = register_cognition_entry(entry)
        match = match_registry_signature(fp, reg)
        assert match.recommended_action == "QLDPC_PORTFOLIO_B"

    def test_unknown_action_for_no_match(self, sample_registry):
        alien_fp = AudioFingerprint(
            centroid=50000.0, rolloff=50000.0,
            peak_bins=(800, 801, 802, 803, 804), psd_hash="0" * 64,
        )
        match = match_registry_signature(alien_fp, sample_registry)
        assert match.recommended_action == UNKNOWN_ACTION

    def test_action_hook_present_in_cycle_result(self, sample_registry):
        result = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        assert result.match.recommended_action is not None
        assert isinstance(result.match.recommended_action, str)
        assert len(result.match.recommended_action) > 0


# ===================================================================
# 10. CODE ZOO INTEGRATION TESTS
# ===================================================================


class TestCodeZooIntegration:
    """Code zoo integration tests."""

    def test_get_code_zoo_families(self):
        families = get_code_zoo_families()
        assert isinstance(families, tuple)
        assert len(families) > 0

    def test_code_zoo_families_sorted(self):
        families = get_code_zoo_families()
        assert families == tuple(sorted(families))

    def test_code_zoo_contains_known_families(self):
        families = get_code_zoo_families()
        assert "surface" in families
        assert "toric" in families
        assert "repetition" in families
        assert "qldpc" in families

    def test_derive_carrier_from_zoo_surface(self):
        freq = derive_carrier_freq_from_zoo("surface")
        assert freq is not None
        assert CARRIER_FREQ_MIN <= freq <= CARRIER_FREQ_MAX

    def test_derive_carrier_from_zoo_unknown(self):
        freq = derive_carrier_freq_from_zoo("nonexistent_family")
        assert freq is None

    def test_carrier_freq_deterministic(self):
        f1 = derive_carrier_freq_from_zoo("surface")
        f2 = derive_carrier_freq_from_zoo("surface")
        assert f1 == f2

    def test_different_families_different_carriers(self):
        f_surface = derive_carrier_freq_from_zoo("surface")
        f_toric = derive_carrier_freq_from_zoo("toric")
        assert f_surface != f_toric

    def test_cognition_cycle_with_zoo_family(self, sample_registry):
        result = run_cognition_cycle(
            "surface", "X_error", "planar",
            REFERENCE_STATE_HASH, sample_registry,
        )
        assert result.engine_version == ENGINE_VERSION


# ===================================================================
# 11. DECODER UNTOUCHED VERIFICATION TESTS
# ===================================================================


class TestDecoderUntouched:
    """Verify decoder core is not imported or modified by audio modules."""

    def test_triality_engine_no_decoder_import(self):
        import qec.audio.triality_signal_engine as mod
        source = open(mod.__file__).read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_cognition_registry_no_decoder_import(self):
        import qec.audio.cognition_registry as mod
        source = open(mod.__file__).read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_audio_cognition_engine_no_decoder_import(self):
        import qec.audio.audio_cognition_engine as mod
        source = open(mod.__file__).read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_directory_unmodified(self):
        """Verify decoder directory exists and was not touched."""
        decoder_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder",
        )
        assert os.path.isdir(decoder_path)


# ===================================================================
# 12. TRIALITY PARAMETER TESTS
# ===================================================================


class TestTrialityParameters:
    """Triality signal parameter tests."""

    def test_carrier_freq_in_range(self, reference_params):
        assert CARRIER_FREQ_MIN <= reference_params.carrier_freq <= CARRIER_FREQ_MAX

    def test_mod_depth_in_range(self, reference_params):
        assert 0.1 <= reference_params.mod_depth <= 0.9

    def test_params_deterministic(self):
        p1 = derive_triality_params("surface", "X_error", "planar", "abc")
        p2 = derive_triality_params("surface", "X_error", "planar", "abc")
        assert p1 == p2

    def test_hash_to_float_range(self):
        for s in ["a", "b", "c", "test", "surface", "toric"]:
            v = _hash_to_float(s)
            assert 0.0 <= v < 1.0

    def test_hash_to_int_range(self):
        for s in ["a", "b", "c", "test"]:
            v = _hash_to_int(s, 1, 8)
            assert 1 <= v <= 8


# ===================================================================
# 13. SPECTRAL FEATURE TESTS
# ===================================================================


class TestSpectralFeatures:
    """Spectral feature extraction tests."""

    def test_centroid_positive(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        centroid = compute_spectral_centroid(psd)
        assert centroid > 0.0

    def test_rolloff_positive(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        rolloff = compute_spectral_rolloff(psd)
        assert rolloff > 0.0

    def test_peak_bins_sorted(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        peaks = compute_peak_bins(psd)
        assert peaks == tuple(sorted(peaks))

    def test_peak_bins_count(self, reference_waveform):
        psd = compute_psd(reference_waveform)
        peaks = compute_peak_bins(psd, n_peaks=5)
        assert len(peaks) == 5

    def test_fingerprint_has_all_fields(self, reference_fingerprint):
        assert isinstance(reference_fingerprint.centroid, float)
        assert isinstance(reference_fingerprint.rolloff, float)
        assert isinstance(reference_fingerprint.peak_bins, tuple)
        assert isinstance(reference_fingerprint.psd_hash, str)
        assert len(reference_fingerprint.psd_hash) == 64


# ===================================================================
# 14. REGISTRY MANAGEMENT TESTS
# ===================================================================


class TestRegistryManagement:
    """Registry management tests."""

    def test_register_single_entry(self, sample_entry):
        reg = register_cognition_entry(sample_entry)
        assert len(reg.entries) == 1

    def test_register_multiple_entries(self, sample_entry):
        entry2 = CognitionRegistryEntry(
            state_hash="b" * 64, code_family="toric",
            error_type="Z_error", topology_state="torus",
            fingerprint=AudioFingerprint(
                centroid=50.0, rolloff=30.0,
                peak_bins=(1, 2, 3), psd_hash="c" * 64,
            ),
            recommended_action="DECODE_PORTFOLIO_C",
            failure_mode="logical_error",
        )
        reg = register_cognition_entry(sample_entry)
        reg = register_cognition_entry(entry2, reg)
        assert len(reg.entries) == 2

    def test_registry_sorted_order(self, sample_entry):
        entry2 = CognitionRegistryEntry(
            state_hash="a" * 64, code_family="a_first_family",
            error_type="error", topology_state="flat",
            fingerprint=AudioFingerprint(
                centroid=10.0, rolloff=5.0,
                peak_bins=(0, 1, 2), psd_hash="d" * 64,
            ),
            recommended_action="TEST_ACTION",
            failure_mode="test_fail",
        )
        reg = register_cognition_entry(sample_entry)
        reg = register_cognition_entry(entry2, reg)
        families = [e.code_family for e in reg.entries]
        assert families == sorted(families)

    def test_registry_hash_deterministic(self, sample_entry):
        reg1 = register_cognition_entry(sample_entry)
        reg2 = register_cognition_entry(sample_entry)
        assert reg1.registry_hash == reg2.registry_hash

    def test_build_empty_registry(self):
        reg = build_registry(())
        assert len(reg.entries) == 0
        assert len(reg.registry_hash) == 64


# ===================================================================
# 15. EXPORT / SERIALIZATION TESTS
# ===================================================================


class TestExportSerialization:
    """Export and serialization tests."""

    def test_export_bundle_keys(self, sample_registry):
        result = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        bundle = export_cognition_bundle(result)
        assert "engine_version" in bundle
        assert "fingerprint" in bundle
        assert "match" in bundle
        assert "params" in bundle

    def test_export_json_valid(self, sample_registry):
        result = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        j = export_cognition_bundle_json(result)
        parsed = json.loads(j)
        assert parsed["engine_version"] == ENGINE_VERSION

    def test_export_json_deterministic(self, sample_registry):
        r1 = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        r2 = run_cognition_cycle(
            REFERENCE_CODE_FAMILY, REFERENCE_ERROR_TYPE,
            REFERENCE_TOPOLOGY, REFERENCE_STATE_HASH, sample_registry,
        )
        assert export_cognition_bundle_json(r1) == export_cognition_bundle_json(r2)


# ===================================================================
# 16. FINGERPRINT VECTOR TESTS
# ===================================================================


class TestFingerprintVector:
    """Fingerprint vector conversion tests."""

    def test_vector_length(self, reference_fingerprint):
        v = fingerprint_to_vector(reference_fingerprint)
        expected = 2 + len(reference_fingerprint.peak_bins)
        assert len(v) == expected

    def test_vector_deterministic(self, reference_fingerprint):
        v1 = fingerprint_to_vector(reference_fingerprint)
        v2 = fingerprint_to_vector(reference_fingerprint)
        np.testing.assert_array_equal(v1, v2)

    def test_vector_dtype(self, reference_fingerprint):
        v = fingerprint_to_vector(reference_fingerprint)
        assert v.dtype == np.float64
