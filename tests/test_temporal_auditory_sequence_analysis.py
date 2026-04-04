"""Tests for v137.0.4 — Temporal Auditory Sequence Analysis.

Target: 40–50 tests covering oscillation classification, escalation,
de-escalation, recurrence, hashing, export, replay, immutability,
and decoder non-contamination.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from qec.analysis.closed_loop_auditory_phase_control import (
    AuditoryPhaseSignature,
    observe_auditory_phase_control,
)
from qec.analysis.temporal_auditory_sequence_analysis import (
    OSCILLATION_ALTERNATING,
    OSCILLATION_COLLAPSE_LOOP,
    OSCILLATION_CYCLIC,
    OSCILLATION_ESCALATING,
    OSCILLATION_STATIC,
    TEMPORAL_AUDITORY_SEQUENCE_VERSION,
    TemporalAuditorySequenceDecision,
    TemporalAuditorySequenceLedger,
    analyze_auditory_sequence,
    build_temporal_auditory_ledger,
    export_temporal_auditory_bundle,
    export_temporal_auditory_ledger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(risk: float, route: str = "RECOVERY") -> AuditoryPhaseSignature:
    """Create a signature via the v137.0.3 API."""
    return observe_auditory_phase_control(
        phase_bin_index=(2, 3),
        spectral_drift=0.50,
        risk_score=risk,
        governed_route=route,
    )


def _make_sig_band(band: str) -> AuditoryPhaseSignature:
    """Create a signature with a specific amplitude band."""
    risk_map = {"LOW": 0.1, "WATCH": 0.3, "WARNING": 0.5, "CRITICAL": 0.7, "COLLAPSE": 0.9}
    return _make_sig(risk_map[band])


# ---------------------------------------------------------------------------
# 1. Static sequence detection
# ---------------------------------------------------------------------------

class TestStaticSequence:
    def test_single_element_is_static(self):
        sig = _make_sig(0.1)
        decision = analyze_auditory_sequence([sig])
        assert decision.oscillation_pattern == OSCILLATION_STATIC

    def test_all_same_band_is_static(self):
        sigs = [_make_sig(0.1), _make_sig(0.15), _make_sig(0.05)]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_STATIC

    def test_static_sequence_length(self):
        sigs = [_make_sig(0.1)] * 5
        decision = analyze_auditory_sequence(sigs)
        assert decision.sequence_length == 5


# ---------------------------------------------------------------------------
# 2. ABA cyclic detection
# ---------------------------------------------------------------------------

class TestCyclicSequence:
    def test_aba_is_cyclic(self):
        sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("LOW")]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_CYCLIC

    def test_abca_is_cyclic(self):
        sigs = [
            _make_sig_band("LOW"),
            _make_sig_band("WATCH"),
            _make_sig_band("WARNING"),
            _make_sig_band("LOW"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_CYCLIC

    def test_two_different_bands_escalating(self):
        """Two bands in escalation order is ESCALATING, not CYCLIC."""
        sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH")]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ESCALATING


# ---------------------------------------------------------------------------
# 3. ABAB alternating detection
# ---------------------------------------------------------------------------

class TestAlternatingSequence:
    def test_abab_is_alternating(self):
        sigs = [
            _make_sig_band("LOW"), _make_sig_band("WATCH"),
            _make_sig_band("LOW"), _make_sig_band("WATCH"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ALTERNATING

    def test_aba_three_elements_is_cyclic(self):
        """ABA with 3 elements is CYCLIC (returns to A), not ALTERNATING."""
        sigs = [
            _make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("LOW"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_CYCLIC

    def test_ababab_is_alternating(self):
        sigs = [
            _make_sig_band("CRITICAL"), _make_sig_band("WARNING"),
            _make_sig_band("CRITICAL"), _make_sig_band("WARNING"),
            _make_sig_band("CRITICAL"), _make_sig_band("WARNING"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ALTERNATING


# ---------------------------------------------------------------------------
# 4. Escalating sequence detection
# ---------------------------------------------------------------------------

class TestEscalatingSequence:
    def test_low_to_critical_is_escalating(self):
        sigs = [
            _make_sig_band("LOW"),
            _make_sig_band("WATCH"),
            _make_sig_band("WARNING"),
            _make_sig_band("CRITICAL"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ESCALATING

    def test_full_escalation_to_collapse(self):
        sigs = [
            _make_sig_band("LOW"),
            _make_sig_band("WATCH"),
            _make_sig_band("WARNING"),
            _make_sig_band("CRITICAL"),
            _make_sig_band("COLLAPSE"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ESCALATING

    def test_partial_escalation(self):
        sigs = [_make_sig_band("WATCH"), _make_sig_band("WARNING")]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_ESCALATING


# ---------------------------------------------------------------------------
# 5. Collapse loop detection
# ---------------------------------------------------------------------------

class TestCollapseLoop:
    def test_double_collapse_is_collapse_loop(self):
        sigs = [
            _make_sig_band("COLLAPSE"),
            _make_sig_band("LOW"),
            _make_sig_band("COLLAPSE"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_COLLAPSE_LOOP

    def test_collapse_with_recovery_loop(self):
        sigs = [
            _make_sig_band("WARNING"),
            _make_sig_band("COLLAPSE"),
            _make_sig_band("WATCH"),
            _make_sig_band("COLLAPSE"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern == OSCILLATION_COLLAPSE_LOOP

    def test_single_collapse_not_loop(self):
        sigs = [_make_sig_band("LOW"), _make_sig_band("COLLAPSE")]
        decision = analyze_auditory_sequence(sigs)
        assert decision.oscillation_pattern != OSCILLATION_COLLAPSE_LOOP


# ---------------------------------------------------------------------------
# 6. Recurrence bounds
# ---------------------------------------------------------------------------

class TestRecurrenceBounds:
    def test_recurrence_zero_for_unique_transitions(self):
        sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("WARNING")]
        decision = analyze_auditory_sequence(sigs)
        assert decision.recurrence_score == 0.0

    def test_recurrence_positive_for_repeated_transitions(self):
        sigs = [
            _make_sig_band("LOW"), _make_sig_band("WATCH"),
            _make_sig_band("LOW"), _make_sig_band("WATCH"),
        ]
        decision = analyze_auditory_sequence(sigs)
        assert decision.recurrence_score > 0.0

    def test_recurrence_bounded_zero_one(self):
        sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH")] * 50
        decision = analyze_auditory_sequence(sigs)
        assert 0.0 <= decision.recurrence_score <= 1.0

    def test_recurrence_single_element_zero(self):
        decision = analyze_auditory_sequence([_make_sig(0.1)])
        assert decision.recurrence_score == 0.0


# ---------------------------------------------------------------------------
# 7. Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    def test_decision_is_frozen(self):
        decision = analyze_auditory_sequence([_make_sig(0.1)])
        with pytest.raises(AttributeError):
            decision.oscillation_pattern = "HACKED"  # type: ignore[misc]

    def test_ledger_is_frozen(self):
        decision = analyze_auditory_sequence([_make_sig(0.1)])
        ledger = build_temporal_auditory_ledger([decision])
        with pytest.raises(AttributeError):
            ledger.decision_count = 999  # type: ignore[misc]

    def test_decision_hash_is_string(self):
        decision = analyze_auditory_sequence([_make_sig(0.1)])
        assert isinstance(decision.stable_hash, str)
        assert len(decision.stable_hash) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# 8. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    def test_export_bundle_deterministic(self):
        sig = _make_sig(0.1)
        d = analyze_auditory_sequence([sig])
        e1 = export_temporal_auditory_bundle(d)
        e2 = export_temporal_auditory_bundle(d)
        assert e1 == e2

    def test_export_ledger_deterministic(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        ledger = build_temporal_auditory_ledger([d])
        e1 = export_temporal_auditory_ledger(ledger)
        e2 = export_temporal_auditory_ledger(ledger)
        assert e1 == e2

    def test_export_bundle_contains_layer(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        bundle = export_temporal_auditory_bundle(d)
        assert bundle["layer"] == "temporal_auditory_sequence_analysis"

    def test_export_ledger_contains_version(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        ledger = build_temporal_auditory_ledger([d])
        exported = export_temporal_auditory_ledger(ledger)
        assert exported["version"] == TEMPORAL_AUDITORY_SEQUENCE_VERSION

    def test_export_bundle_json_serializable(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        bundle = export_temporal_auditory_bundle(d)
        serialized = json.dumps(bundle, sort_keys=True)
        assert isinstance(serialized, str)

    def test_export_ledger_json_serializable(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        ledger = build_temporal_auditory_ledger([d])
        exported = export_temporal_auditory_ledger(ledger)
        serialized = json.dumps(exported, sort_keys=True)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# 9. Stable hashing
# ---------------------------------------------------------------------------

class TestStableHashing:
    def test_decision_hash_stable(self):
        sig = _make_sig(0.1)
        d1 = analyze_auditory_sequence([sig])
        d2 = analyze_auditory_sequence([sig])
        assert d1.stable_hash == d2.stable_hash

    def test_different_input_different_hash(self):
        d1 = analyze_auditory_sequence([_make_sig(0.1)])
        d2 = analyze_auditory_sequence([_make_sig(0.7)])
        assert d1.stable_hash != d2.stable_hash

    def test_ledger_hash_stable(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        l1 = build_temporal_auditory_ledger([d])
        l2 = build_temporal_auditory_ledger([d])
        assert l1.stable_hash == l2.stable_hash

    def test_ledger_hash_changes_with_different_decisions(self):
        d1 = analyze_auditory_sequence([_make_sig(0.1)])
        d2 = analyze_auditory_sequence([_make_sig(0.7)])
        l1 = build_temporal_auditory_ledger([d1])
        l2 = build_temporal_auditory_ledger([d2])
        assert l1.stable_hash != l2.stable_hash


# ---------------------------------------------------------------------------
# 10. 100-run replay
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    def test_100_run_replay_decision(self):
        sigs = [
            _make_sig_band("LOW"),
            _make_sig_band("WATCH"),
            _make_sig_band("WARNING"),
            _make_sig_band("CRITICAL"),
        ]
        reference = analyze_auditory_sequence(sigs)
        for _ in range(100):
            result = analyze_auditory_sequence(sigs)
            assert result.stable_hash == reference.stable_hash
            assert result.oscillation_pattern == reference.oscillation_pattern
            assert result.recurrence_score == reference.recurrence_score
            assert result.escalation_signal == reference.escalation_signal
            assert result.deescalation_signal == reference.deescalation_signal

    def test_100_run_replay_ledger(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        ref_ledger = build_temporal_auditory_ledger([d])
        for _ in range(100):
            ledger = build_temporal_auditory_ledger([d])
            assert ledger.stable_hash == ref_ledger.stable_hash

    def test_100_run_replay_export(self):
        d = analyze_auditory_sequence([_make_sig(0.5)])
        ref = export_temporal_auditory_bundle(d)
        ref_json = json.dumps(ref, sort_keys=True, separators=(",", ":"))
        for _ in range(100):
            export = export_temporal_auditory_bundle(d)
            export_json = json.dumps(export, sort_keys=True, separators=(",", ":"))
            assert export_json == ref_json


# ---------------------------------------------------------------------------
# 11. No decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """temporal_auditory_sequence_analysis must not import decoder."""
        import qec.analysis.temporal_auditory_sequence_analysis as mod
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_modules_untouched(self):
        """Verify decoder modules are not in our transitive imports."""
        import qec.analysis.temporal_auditory_sequence_analysis as mod  # noqa: F811
        imported = set(sys.modules.keys())
        decoder_modules = [k for k in imported if "qec.decoder" in k]
        assert decoder_modules == [], f"Decoder contamination: {decoder_modules}"


# ---------------------------------------------------------------------------
# 12. Invalid input rejection
# ---------------------------------------------------------------------------

class TestInvalidInputRejection:
    def test_empty_sequence_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            analyze_auditory_sequence([])

    def test_non_iterable_raises_type_error(self):
        with pytest.raises(TypeError):
            analyze_auditory_sequence(42)  # type: ignore[arg-type]

    def test_wrong_type_in_sequence_raises_type_error(self):
        with pytest.raises(TypeError, match="must be AuditoryPhaseSignature"):
            analyze_auditory_sequence(["not_a_signature"])  # type: ignore[list-item]

    def test_ledger_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditorySequenceDecision"):
            build_temporal_auditory_ledger(["not_a_decision"])  # type: ignore[list-item]

    def test_none_in_sequence_raises_type_error(self):
        with pytest.raises(TypeError, match="must be AuditoryPhaseSignature"):
            analyze_auditory_sequence([None])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 13. Escalation signal correctness
# ---------------------------------------------------------------------------

class TestEscalationSignals:
    def test_collapse_gives_critical_signal(self):
        d = analyze_auditory_sequence([_make_sig_band("COLLAPSE")])
        assert d.escalation_signal == "CRITICAL"

    def test_critical_gives_escalate_signal(self):
        d = analyze_auditory_sequence([_make_sig_band("CRITICAL")])
        assert d.escalation_signal == "ESCALATE"

    def test_warning_gives_watch_signal(self):
        d = analyze_auditory_sequence([_make_sig_band("WARNING")])
        assert d.escalation_signal == "WATCH"

    def test_low_gives_none_signal(self):
        d = analyze_auditory_sequence([_make_sig_band("LOW")])
        assert d.escalation_signal == "NONE"


# ---------------------------------------------------------------------------
# 14. De-escalation signal correctness
# ---------------------------------------------------------------------------

class TestDeescalationSignals:
    def test_deescalation_on_drop(self):
        sigs = [_make_sig_band("CRITICAL"), _make_sig_band("WARNING")]
        d = analyze_auditory_sequence(sigs)
        assert d.deescalation_signal == "RELAX"

    def test_deescalation_to_low_is_recover(self):
        sigs = [_make_sig_band("WATCH"), _make_sig_band("LOW")]
        d = analyze_auditory_sequence(sigs)
        assert d.deescalation_signal == "RECOVER"

    def test_no_deescalation_on_escalation(self):
        sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH")]
        d = analyze_auditory_sequence(sigs)
        assert d.deescalation_signal == "NONE"

    def test_no_deescalation_single_element(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        assert d.deescalation_signal == "NONE"


# ---------------------------------------------------------------------------
# 15. Temporal symbolic trace
# ---------------------------------------------------------------------------

class TestTemporalSymbolicTrace:
    def test_single_trace(self):
        sig = _make_sig(0.1)
        d = analyze_auditory_sequence([sig])
        assert d.temporal_symbolic_trace == sig.audio_symbolic_trace

    def test_multi_trace_arrow_joined(self):
        sigs = [_make_sig(0.1), _make_sig(0.7)]
        d = analyze_auditory_sequence(sigs)
        expected = f"{sigs[0].audio_symbolic_trace} -> {sigs[1].audio_symbolic_trace}"
        assert d.temporal_symbolic_trace == expected

    def test_trace_contains_all_signatures(self):
        sigs = [_make_sig(0.1), _make_sig(0.3), _make_sig(0.5)]
        d = analyze_auditory_sequence(sigs)
        for s in sigs:
            assert s.audio_symbolic_trace in d.temporal_symbolic_trace


# ---------------------------------------------------------------------------
# 16. Version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_decision_version(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        assert d.version == "v137.0.4"

    def test_version_constant(self):
        assert TEMPORAL_AUDITORY_SEQUENCE_VERSION == "v137.0.4"


# ---------------------------------------------------------------------------
# 17. Ledger construction
# ---------------------------------------------------------------------------

class TestLedgerConstruction:
    def test_ledger_decision_count(self):
        d1 = analyze_auditory_sequence([_make_sig(0.1)])
        d2 = analyze_auditory_sequence([_make_sig(0.7)])
        ledger = build_temporal_auditory_ledger([d1, d2])
        assert ledger.decision_count == 2

    def test_ledger_decisions_are_tuple(self):
        d = analyze_auditory_sequence([_make_sig(0.1)])
        ledger = build_temporal_auditory_ledger([d])
        assert isinstance(ledger.decisions, tuple)

    def test_empty_ledger(self):
        ledger = build_temporal_auditory_ledger([])
        assert ledger.decision_count == 0
        assert ledger.decisions == ()
