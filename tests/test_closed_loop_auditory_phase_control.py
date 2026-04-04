"""Tests for Closed-Loop Auditory + Phase Control — v137.0.3.

Covers:
  1. Deterministic frequency mapping
  2. Bit-depth mapping
  3. Symbolic trace equality
  4. Export equality
  5. Frozen immutability
  6. 100-run replay determinism
  7. No decoder contamination
  8. Stable ledger hashing
  9. Invalid phase_bin validation
  10. Risk band classification
  11. Ledger construction
  12. Canonical JSON stability
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from qec.analysis.closed_loop_auditory_phase_control import (
    AUDITORY_PHASE_VERSION,
    AuditoryPhaseLedger,
    AuditoryPhaseSignature,
    _BIT_DEPTH_MAP,
    _FREQUENCY_MAP,
    _classify_risk,
    build_auditory_phase_ledger,
    export_auditory_phase_bundle,
    export_auditory_phase_ledger,
    observe_auditory_phase_control,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(
    phase_bin: tuple[int, int] = (2, 3),
    drift: float = 0.5,
    risk: float = 0.5,
    route: str = "RECOVERY",
) -> AuditoryPhaseSignature:
    """Create a standard test signature."""
    return observe_auditory_phase_control(phase_bin, drift, risk, route)


# ---------------------------------------------------------------------------
# 1. Deterministic frequency mapping
# ---------------------------------------------------------------------------

class TestFrequencyMapping:
    """Verify risk → frequency mapping is deterministic."""

    @pytest.mark.parametrize("risk,expected_hz", [
        (0.0, 220.0),
        (0.1, 220.0),
        (0.2, 440.0),
        (0.3, 440.0),
        (0.4, 880.0),
        (0.5, 880.0),
        (0.6, 1760.0),
        (0.7, 1760.0),
        (0.8, 3520.0),
        (0.99, 3520.0),
    ])
    def test_frequency_for_risk(self, risk: float, expected_hz: float) -> None:
        sig = _make_sig(risk=risk)
        assert sig.instability_frequency_hz == expected_hz

    def test_all_bands_covered(self) -> None:
        """Every band has a frequency entry."""
        for band in ("LOW", "WATCH", "WARNING", "CRITICAL", "COLLAPSE"):
            assert band in _FREQUENCY_MAP


# ---------------------------------------------------------------------------
# 2. Bit-depth mapping
# ---------------------------------------------------------------------------

class TestBitDepthMapping:
    """Verify risk → bit-depth mapping."""

    @pytest.mark.parametrize("risk,expected_depth", [
        (0.0, 24),
        (0.2, 16),
        (0.4, 12),
        (0.6, 8),
        (0.8, 4),
    ])
    def test_bit_depth_for_risk(self, risk: float, expected_depth: int) -> None:
        sig = _make_sig(risk=risk)
        assert sig.bit_depth_level == expected_depth

    def test_all_bands_have_depth(self) -> None:
        for band in ("LOW", "WATCH", "WARNING", "CRITICAL", "COLLAPSE"):
            assert band in _BIT_DEPTH_MAP


# ---------------------------------------------------------------------------
# 3. Symbolic trace equality
# ---------------------------------------------------------------------------

class TestSymbolicTrace:
    """Verify trace string format and determinism."""

    def test_trace_format(self) -> None:
        sig = _make_sig(phase_bin=(2, 3), risk=0.5, route="RECOVERY")
        assert sig.audio_symbolic_trace == "PB(2,3)-F880-B12-RECOVERY"

    def test_trace_low(self) -> None:
        sig = _make_sig(phase_bin=(0, 1), risk=0.0, route="IDLE")
        assert sig.audio_symbolic_trace == "PB(0,1)-F220-B24-IDLE"

    def test_trace_collapse(self) -> None:
        sig = _make_sig(phase_bin=(7, 9), risk=0.9, route="HALT")
        assert sig.audio_symbolic_trace == "PB(7,9)-F3520-B4-HALT"

    def test_trace_determinism(self) -> None:
        s1 = _make_sig()
        s2 = _make_sig()
        assert s1.audio_symbolic_trace == s2.audio_symbolic_trace


# ---------------------------------------------------------------------------
# 4. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Verify export produces identical dicts for identical inputs."""

    def test_signature_export_determinism(self) -> None:
        sig = _make_sig()
        e1 = export_auditory_phase_bundle(sig)
        e2 = export_auditory_phase_bundle(sig)
        assert e1 == e2

    def test_signature_export_json_stability(self) -> None:
        sig = _make_sig()
        j1 = json.dumps(export_auditory_phase_bundle(sig), sort_keys=True)
        j2 = json.dumps(export_auditory_phase_bundle(sig), sort_keys=True)
        assert j1 == j2

    def test_export_contains_layer(self) -> None:
        bundle = export_auditory_phase_bundle(_make_sig())
        assert bundle["layer"] == "closed_loop_auditory_phase_control"

    def test_export_contains_version(self) -> None:
        bundle = export_auditory_phase_bundle(_make_sig())
        assert bundle["version"] == AUDITORY_PHASE_VERSION

    def test_ledger_export_determinism(self) -> None:
        sigs = [_make_sig(risk=r) for r in (0.1, 0.3, 0.5)]
        ledger = build_auditory_phase_ledger(sigs)
        e1 = export_auditory_phase_ledger(ledger)
        e2 = export_auditory_phase_ledger(ledger)
        assert e1 == e2

    def test_ledger_export_json_stability(self) -> None:
        sigs = [_make_sig(risk=r) for r in (0.1, 0.3, 0.5)]
        ledger = build_auditory_phase_ledger(sigs)
        j1 = json.dumps(export_auditory_phase_ledger(ledger), sort_keys=True)
        j2 = json.dumps(export_auditory_phase_ledger(ledger), sort_keys=True)
        assert j1 == j2


# ---------------------------------------------------------------------------
# 5. Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Frozen dataclasses must reject mutation."""

    def test_signature_frozen(self) -> None:
        sig = _make_sig()
        with pytest.raises(AttributeError):
            sig.amplitude_band = "LOW"  # type: ignore[misc]

    def test_ledger_frozen(self) -> None:
        ledger = build_auditory_phase_ledger([_make_sig()])
        with pytest.raises(AttributeError):
            ledger.signature_count = 999  # type: ignore[misc]

    def test_signature_fields_immutable(self) -> None:
        sig = _make_sig()
        with pytest.raises(AttributeError):
            sig.stable_hash = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. 100-run replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    """100-run identical-input replay must produce identical output."""

    def test_100_run_signature_replay(self) -> None:
        reference = _make_sig(phase_bin=(4, 5), risk=0.65, route="RESTORE")
        for _ in range(100):
            sig = _make_sig(phase_bin=(4, 5), risk=0.65, route="RESTORE")
            assert sig == reference
            assert sig.stable_hash == reference.stable_hash

    def test_100_run_ledger_replay(self) -> None:
        sigs = [_make_sig(risk=r) for r in (0.0, 0.3, 0.6, 0.9)]
        reference = build_auditory_phase_ledger(sigs)
        for _ in range(100):
            sigs_i = [_make_sig(risk=r) for r in (0.0, 0.3, 0.6, 0.9)]
            ledger_i = build_auditory_phase_ledger(sigs_i)
            assert ledger_i == reference
            assert ledger_i.stable_hash == reference.stable_hash

    def test_100_run_export_replay(self) -> None:
        sig = _make_sig()
        reference = json.dumps(export_auditory_phase_bundle(sig), sort_keys=True)
        for _ in range(100):
            result = json.dumps(export_auditory_phase_bundle(sig), sort_keys=True)
            assert result == reference


# ---------------------------------------------------------------------------
# 7. No decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    """Module must not import from decoder layer."""

    def test_no_decoder_import(self) -> None:
        import qec.analysis.closed_loop_auditory_phase_control as mod
        source = open(mod.__file__, "r").read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_channel_import(self) -> None:
        import qec.analysis.closed_loop_auditory_phase_control as mod
        source = open(mod.__file__, "r").read()
        assert "qec.channel" not in source


# ---------------------------------------------------------------------------
# 8. Stable ledger hashing
# ---------------------------------------------------------------------------

class TestLedgerHashing:
    """Ledger hash must be stable and order-sensitive."""

    def test_ledger_hash_stable(self) -> None:
        sigs = [_make_sig(risk=r) for r in (0.1, 0.5, 0.9)]
        l1 = build_auditory_phase_ledger(sigs)
        l2 = build_auditory_phase_ledger(sigs)
        assert l1.stable_hash == l2.stable_hash

    def test_ledger_hash_order_sensitive(self) -> None:
        s_low = _make_sig(risk=0.1)
        s_high = _make_sig(risk=0.9)
        l1 = build_auditory_phase_ledger([s_low, s_high])
        l2 = build_auditory_phase_ledger([s_high, s_low])
        assert l1.stable_hash != l2.stable_hash

    def test_ledger_count_matches(self) -> None:
        sigs = [_make_sig(risk=r) for r in (0.0, 0.2, 0.4)]
        ledger = build_auditory_phase_ledger(sigs)
        assert ledger.signature_count == 3

    def test_empty_ledger(self) -> None:
        ledger = build_auditory_phase_ledger([])
        assert ledger.signature_count == 0
        assert ledger.signatures == ()
        assert isinstance(ledger.stable_hash, str)
        assert len(ledger.stable_hash) == 64


# ---------------------------------------------------------------------------
# 9. Invalid phase_bin validation
# ---------------------------------------------------------------------------

class TestInvalidPhaseBin:
    """Invalid phase_bin_index must raise TypeError."""

    def test_not_tuple(self) -> None:
        with pytest.raises(TypeError):
            observe_auditory_phase_control([2, 3], 0.5, 0.5, "R")  # type: ignore[arg-type]

    def test_wrong_length(self) -> None:
        with pytest.raises(TypeError):
            observe_auditory_phase_control((1,), 0.5, 0.5, "R")

    def test_non_int_elements(self) -> None:
        with pytest.raises(TypeError):
            observe_auditory_phase_control((1.0, 2.0), 0.5, 0.5, "R")  # type: ignore[arg-type]

    def test_three_elements(self) -> None:
        with pytest.raises(TypeError):
            observe_auditory_phase_control((1, 2, 3), 0.5, 0.5, "R")  # type: ignore[arg-type]

    def test_negative_risk(self) -> None:
        with pytest.raises(ValueError):
            observe_auditory_phase_control((0, 0), 0.0, -0.1, "R")

    def test_non_numeric_risk(self) -> None:
        with pytest.raises(TypeError):
            observe_auditory_phase_control((0, 0), 0.0, "high", "R")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. Risk band classification
# ---------------------------------------------------------------------------

class TestRiskClassification:
    """Risk-band boundaries must be precise."""

    @pytest.mark.parametrize("score,expected", [
        (0.0, "LOW"),
        (0.19, "LOW"),
        (0.2, "WATCH"),
        (0.39, "WATCH"),
        (0.4, "WARNING"),
        (0.59, "WARNING"),
        (0.6, "CRITICAL"),
        (0.79, "CRITICAL"),
        (0.8, "COLLAPSE"),
        (1.0, "COLLAPSE"),
        (5.0, "COLLAPSE"),
    ])
    def test_band_boundary(self, score: float, expected: str) -> None:
        assert _classify_risk(score) == expected


# ---------------------------------------------------------------------------
# 11. Version stamp
# ---------------------------------------------------------------------------

class TestVersionStamp:
    """Signatures carry correct version."""

    def test_version(self) -> None:
        sig = _make_sig()
        assert sig.version == "v137.0.3"

    def test_version_constant(self) -> None:
        assert AUDITORY_PHASE_VERSION == "v137.0.3"


# ---------------------------------------------------------------------------
# 12. Signature hash correctness
# ---------------------------------------------------------------------------

class TestSignatureHash:
    """Stable hash must be a valid 64-char hex SHA-256."""

    def test_hash_length(self) -> None:
        sig = _make_sig()
        assert len(sig.stable_hash) == 64

    def test_hash_hex(self) -> None:
        sig = _make_sig()
        int(sig.stable_hash, 16)  # must not raise

    def test_different_inputs_different_hash(self) -> None:
        s1 = _make_sig(risk=0.1)
        s2 = _make_sig(risk=0.9)
        assert s1.stable_hash != s2.stable_hash
