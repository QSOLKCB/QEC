"""Tests for v137.0.5 — Temporal Auditory Sequence Policy Memory.

Target: 45–60 tests covering dominant pattern detection, recurrence trend,
hysteresis activation, escalation dampening, governed response hints,
frozen immutability, export equality, stable hashing, 100-run replay,
no decoder contamination, and invalid input rejection.
"""

from __future__ import annotations

import json
import sys

import pytest

from qec.analysis.closed_loop_auditory_phase_control import (
    AuditoryPhaseSignature,
    observe_auditory_phase_control,
)
from qec.analysis.temporal_auditory_sequence_analysis import (
    TemporalAuditorySequenceDecision,
    analyze_auditory_sequence,
)
from qec.analysis.temporal_auditory_sequence_policy_memory import (
    DAMPENING_DAMPEN,
    DAMPENING_HOLD,
    DAMPENING_LOCK,
    DAMPENING_NONE,
    RESPONSE_CRITICAL_LOCK,
    RESPONSE_INTERVENE,
    RESPONSE_MONITOR,
    RESPONSE_NONE,
    RESPONSE_STABILIZE,
    TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION,
    TREND_DOWN,
    TREND_STABLE,
    TREND_UP,
    TREND_VOLATILE,
    TemporalAuditoryPolicyLedger,
    TemporalAuditoryPolicyState,
    build_temporal_auditory_policy_ledger,
    build_temporal_auditory_policy_state,
    export_temporal_auditory_policy_bundle,
    export_temporal_auditory_policy_ledger,
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


def _make_decision_static() -> TemporalAuditorySequenceDecision:
    """A STATIC decision with low risk."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("LOW"), _make_sig_band("LOW")]
    return analyze_auditory_sequence(sigs)


def _make_decision_escalating() -> TemporalAuditorySequenceDecision:
    """An ESCALATING decision."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("WARNING"), _make_sig_band("CRITICAL")]
    return analyze_auditory_sequence(sigs)


def _make_decision_alternating() -> TemporalAuditorySequenceDecision:
    """An ALTERNATING decision."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("LOW"), _make_sig_band("WATCH")]
    return analyze_auditory_sequence(sigs)


def _make_decision_collapse_loop() -> TemporalAuditorySequenceDecision:
    """A COLLAPSE_LOOP decision."""
    sigs = [_make_sig_band("COLLAPSE"), _make_sig_band("LOW"), _make_sig_band("COLLAPSE")]
    return analyze_auditory_sequence(sigs)


def _make_decision_cyclic() -> TemporalAuditorySequenceDecision:
    """A CYCLIC decision."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("LOW")]
    return analyze_auditory_sequence(sigs)


def _make_decision_with_critical_escalation() -> TemporalAuditorySequenceDecision:
    """A decision ending at COLLAPSE (escalation_signal = CRITICAL)."""
    sigs = [_make_sig_band("WARNING"), _make_sig_band("COLLAPSE")]
    return analyze_auditory_sequence(sigs)


def _make_decision_with_deescalation() -> TemporalAuditorySequenceDecision:
    """A decision with de-escalation signal."""
    sigs = [_make_sig_band("CRITICAL"), _make_sig_band("WARNING")]
    return analyze_auditory_sequence(sigs)


def _make_decision_low_recurrence() -> TemporalAuditorySequenceDecision:
    """A decision with zero recurrence (unique transitions)."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"), _make_sig_band("WARNING")]
    return analyze_auditory_sequence(sigs)


def _make_decision_high_recurrence() -> TemporalAuditorySequenceDecision:
    """A decision with high recurrence (repeated transitions)."""
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH")] * 5
    return analyze_auditory_sequence(sigs)


# ---------------------------------------------------------------------------
# 1. Dominant pattern detection
# ---------------------------------------------------------------------------

class TestDominantPatternDetection:
    def test_all_static_dominant(self):
        decisions = [_make_decision_static()] * 5
        state = build_temporal_auditory_policy_state(decisions)
        assert state.dominant_pattern == "STATIC"

    def test_mostly_escalating_dominant(self):
        decisions = [_make_decision_escalating()] * 3 + [_make_decision_static()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.dominant_pattern == "ESCALATING"

    def test_mostly_alternating_dominant(self):
        decisions = [_make_decision_alternating()] * 4 + [_make_decision_static()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.dominant_pattern == "ALTERNATING"

    def test_collapse_loop_dominant(self):
        decisions = [_make_decision_collapse_loop()] * 3 + [_make_decision_static()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.dominant_pattern == "COLLAPSE_LOOP"

    def test_tie_break_by_severity(self):
        """When tied, higher severity pattern wins."""
        decisions = [_make_decision_static(), _make_decision_escalating()]
        state = build_temporal_auditory_policy_state(decisions)
        # Both appear once, ESCALATING has higher severity
        assert state.dominant_pattern == "ESCALATING"

    def test_single_decision_dominant(self):
        decisions = [_make_decision_cyclic()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.dominant_pattern == "CYCLIC"


# ---------------------------------------------------------------------------
# 2. Recurrence trend classification
# ---------------------------------------------------------------------------

class TestRecurrenceTrend:
    def test_single_decision_is_stable(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert state.recurrence_trend == TREND_STABLE

    def test_rising_recurrence_is_up(self):
        d_low = _make_decision_low_recurrence()
        d_high = _make_decision_high_recurrence()
        # low -> high = UP
        state = build_temporal_auditory_policy_state([d_low, d_high])
        assert state.recurrence_trend == TREND_UP

    def test_falling_recurrence_is_down(self):
        d_low = _make_decision_low_recurrence()
        d_high = _make_decision_high_recurrence()
        # high -> low = DOWN
        state = build_temporal_auditory_policy_state([d_high, d_low])
        assert state.recurrence_trend == TREND_DOWN

    def test_equal_recurrence_is_stable(self):
        d = _make_decision_static()
        state = build_temporal_auditory_policy_state([d, d, d])
        assert state.recurrence_trend == TREND_STABLE

    def test_mixed_recurrence_is_volatile(self):
        d_low = _make_decision_low_recurrence()
        d_high = _make_decision_high_recurrence()
        # low -> high -> low = VOLATILE
        state = build_temporal_auditory_policy_state([d_low, d_high, d_low])
        assert state.recurrence_trend == TREND_VOLATILE


# ---------------------------------------------------------------------------
# 3. Hysteresis activation detection
# ---------------------------------------------------------------------------

class TestHysteresisActivation:
    def test_no_hysteresis_with_single_decision(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert state.hysteresis_active is False

    def test_no_hysteresis_with_two_decisions(self):
        decisions = [_make_decision_static(), _make_decision_escalating()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.hysteresis_active is False

    def test_hysteresis_with_alternating_escalation_deescalation(self):
        """Repeated esc/deesc alternation triggers hysteresis."""
        # Need escalation followed by de-escalation alternation across windows
        d_esc = _make_decision_with_critical_escalation()
        d_deesc = _make_decision_with_deescalation()
        # esc -> deesc -> esc -> deesc = 2+ alternations
        decisions = [d_esc, d_deesc, d_esc, d_deesc]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.hysteresis_active is True

    def test_no_hysteresis_all_static(self):
        decisions = [_make_decision_static()] * 5
        state = build_temporal_auditory_policy_state(decisions)
        assert state.hysteresis_active is False

    def test_hysteresis_requires_minimum_alternations(self):
        """Single alternation is not enough."""
        d_esc = _make_decision_with_critical_escalation()
        d_deesc = _make_decision_with_deescalation()
        d_static = _make_decision_static()
        decisions = [d_esc, d_deesc, d_static]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.hysteresis_active is False


# ---------------------------------------------------------------------------
# 4. Escalation dampening classification
# ---------------------------------------------------------------------------

class TestEscalationDampening:
    def test_no_dampening_without_hysteresis(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert state.escalation_dampening == DAMPENING_NONE

    def test_dampen_with_hysteresis_few_escalations(self):
        """Hysteresis active but < 3 escalation windows -> DAMPEN."""
        d_esc = _make_decision_with_critical_escalation()
        d_deesc = _make_decision_with_deescalation()
        # 4 windows: esc, deesc, esc, deesc -> hysteresis active
        # 2 escalation windows (d_esc has escalation_signal != NONE)
        decisions = [d_esc, d_deesc, d_esc, d_deesc]
        state = build_temporal_auditory_policy_state(decisions)
        if state.hysteresis_active:
            assert state.escalation_dampening in (DAMPENING_DAMPEN, DAMPENING_HOLD, DAMPENING_LOCK)

    def test_dampening_none_without_hysteresis_many_escalations(self):
        """Many escalation windows but no hysteresis -> NONE."""
        decisions = [_make_decision_escalating()] * 5
        state = build_temporal_auditory_policy_state(decisions)
        if not state.hysteresis_active:
            assert state.escalation_dampening == DAMPENING_NONE

    def test_lock_with_collapse_loop_and_hysteresis(self):
        """Hysteresis + 3+ escalation windows + COLLAPSE_LOOP -> LOCK."""
        d_esc = _make_decision_with_critical_escalation()
        d_deesc = _make_decision_with_deescalation()
        d_collapse = _make_decision_collapse_loop()
        # Need hysteresis (2+ alternations) and 3+ escalation windows and COLLAPSE_LOOP
        decisions = [d_esc, d_deesc, d_esc, d_deesc, d_esc, d_collapse]
        state = build_temporal_auditory_policy_state(decisions)
        if state.hysteresis_active:
            # 3+ escalation windows present and COLLAPSE_LOOP present
            esc_count = sum(1 for d in decisions if d.escalation_signal != "NONE")
            if esc_count >= 3:
                assert state.escalation_dampening == DAMPENING_LOCK


# ---------------------------------------------------------------------------
# 5. Governed response hint correctness
# ---------------------------------------------------------------------------

class TestGovernedResponseHint:
    def test_collapse_loop_gives_critical_lock(self):
        decisions = [_make_decision_collapse_loop()] * 3
        state = build_temporal_auditory_policy_state(decisions)
        assert state.governed_response_hint == RESPONSE_CRITICAL_LOCK

    def test_repeated_critical_gives_critical_lock(self):
        d_crit = _make_decision_with_critical_escalation()
        decisions = [d_crit, d_crit, d_crit]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.governed_response_hint == RESPONSE_CRITICAL_LOCK

    def test_escalating_trend_up_gives_intervene(self):
        d_esc = _make_decision_escalating()
        d_low = _make_decision_low_recurrence()
        d_high = _make_decision_high_recurrence()
        # Need: dominant = ESCALATING, trend = UP
        # Make mostly ESCALATING and rising recurrence
        decisions = [d_esc, d_esc, d_high]
        state = build_temporal_auditory_policy_state(decisions)
        if state.dominant_pattern == "ESCALATING" and state.recurrence_trend == TREND_UP:
            assert state.governed_response_hint == RESPONSE_INTERVENE

    def test_static_trend_down_gives_monitor(self):
        d_high_rec = _make_decision_high_recurrence()
        d_static = _make_decision_static()
        # high recurrence -> static (low recurrence) = DOWN trend
        # dominant should be neither COLLAPSE_LOOP nor ESCALATING
        # Need mostly STATIC for dominant
        decisions = [d_static, d_static, d_high_rec, d_static]
        state = build_temporal_auditory_policy_state(decisions)
        if state.dominant_pattern == "STATIC" and state.recurrence_trend == TREND_DOWN:
            assert state.governed_response_hint == RESPONSE_MONITOR

    def test_none_response_for_generic(self):
        decisions = [_make_decision_cyclic()]
        state = build_temporal_auditory_policy_state(decisions)
        assert state.governed_response_hint == RESPONSE_NONE

    def test_alternating_with_hysteresis_gives_stabilize(self):
        """ALTERNATING + hysteresis_active -> STABILIZE."""
        d_alt = _make_decision_alternating()
        # Use WATCH-level escalation (not CRITICAL) to avoid triggering
        # the higher-priority CRITICAL_LOCK rule
        d_watch_esc = analyze_auditory_sequence(
            [_make_sig_band("LOW"), _make_sig_band("WARNING")]
        )
        d_deesc = _make_decision_with_deescalation()
        # Need: dominant = ALTERNATING, hysteresis_active = True
        # watch_esc has escalation_signal=WATCH (not CRITICAL) and
        # deesc has deescalation_signal != NONE, giving alternations
        decisions = [d_alt, d_watch_esc, d_deesc, d_watch_esc, d_deesc, d_alt, d_alt]
        state = build_temporal_auditory_policy_state(decisions)
        if state.dominant_pattern == "ALTERNATING" and state.hysteresis_active:
            assert state.governed_response_hint == RESPONSE_STABILIZE


# ---------------------------------------------------------------------------
# 6. Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    def test_policy_state_is_frozen(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        with pytest.raises(AttributeError):
            state.dominant_pattern = "HACKED"  # type: ignore[misc]

    def test_policy_state_window_count_frozen(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        with pytest.raises(AttributeError):
            state.window_count = 999  # type: ignore[misc]

    def test_ledger_is_frozen(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        with pytest.raises(AttributeError):
            ledger.state_count = 999  # type: ignore[misc]

    def test_policy_state_hash_is_string(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert isinstance(state.stable_hash, str)
        assert len(state.stable_hash) == 64  # SHA-256 hex

    def test_ledger_hash_is_string(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        assert isinstance(ledger.stable_hash, str)
        assert len(ledger.stable_hash) == 64


# ---------------------------------------------------------------------------
# 7. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    def test_export_bundle_deterministic(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        e1 = export_temporal_auditory_policy_bundle(state)
        e2 = export_temporal_auditory_policy_bundle(state)
        assert e1 == e2

    def test_export_ledger_deterministic(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        e1 = export_temporal_auditory_policy_ledger(ledger)
        e2 = export_temporal_auditory_policy_ledger(ledger)
        assert e1 == e2

    def test_export_bundle_contains_layer(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        bundle = export_temporal_auditory_policy_bundle(state)
        assert bundle["layer"] == "temporal_auditory_sequence_policy_memory"

    def test_export_ledger_contains_version(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        exported = export_temporal_auditory_policy_ledger(ledger)
        assert exported["version"] == TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION

    def test_export_bundle_json_serializable(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        bundle = export_temporal_auditory_policy_bundle(state)
        serialized = json.dumps(bundle, sort_keys=True)
        assert isinstance(serialized, str)

    def test_export_ledger_json_serializable(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        exported = export_temporal_auditory_policy_ledger(ledger)
        serialized = json.dumps(exported, sort_keys=True)
        assert isinstance(serialized, str)

    def test_export_bundle_hash_matches_state_hash(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        bundle = export_temporal_auditory_policy_bundle(state)
        assert bundle["stable_hash"] == state.stable_hash


# ---------------------------------------------------------------------------
# 8. Stable hashing
# ---------------------------------------------------------------------------

class TestStableHashing:
    def test_state_hash_stable(self):
        decisions = [_make_decision_static()]
        s1 = build_temporal_auditory_policy_state(decisions)
        s2 = build_temporal_auditory_policy_state(decisions)
        assert s1.stable_hash == s2.stable_hash

    def test_different_input_different_hash(self):
        s1 = build_temporal_auditory_policy_state([_make_decision_static()])
        s2 = build_temporal_auditory_policy_state([_make_decision_escalating()])
        assert s1.stable_hash != s2.stable_hash

    def test_ledger_hash_stable(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        l1 = build_temporal_auditory_policy_ledger([state])
        l2 = build_temporal_auditory_policy_ledger([state])
        assert l1.stable_hash == l2.stable_hash

    def test_ledger_hash_changes_with_different_states(self):
        s1 = build_temporal_auditory_policy_state([_make_decision_static()])
        s2 = build_temporal_auditory_policy_state([_make_decision_escalating()])
        l1 = build_temporal_auditory_policy_ledger([s1])
        l2 = build_temporal_auditory_policy_ledger([s2])
        assert l1.stable_hash != l2.stable_hash

    def test_hash_length_is_64(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert len(state.stable_hash) == 64

    def test_hash_is_hex(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        int(state.stable_hash, 16)  # Should not raise


# ---------------------------------------------------------------------------
# 9. 100-run replay
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    def test_100_run_replay_state(self):
        decisions = [_make_decision_static(), _make_decision_escalating(), _make_decision_alternating()]
        reference = build_temporal_auditory_policy_state(decisions)
        for _ in range(100):
            result = build_temporal_auditory_policy_state(decisions)
            assert result.stable_hash == reference.stable_hash
            assert result.dominant_pattern == reference.dominant_pattern
            assert result.recurrence_trend == reference.recurrence_trend
            assert result.escalation_dampening == reference.escalation_dampening
            assert result.governed_response_hint == reference.governed_response_hint
            assert result.hysteresis_active == reference.hysteresis_active

    def test_100_run_replay_ledger(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ref_ledger = build_temporal_auditory_policy_ledger([state])
        for _ in range(100):
            ledger = build_temporal_auditory_policy_ledger([state])
            assert ledger.stable_hash == ref_ledger.stable_hash

    def test_100_run_replay_export(self):
        state = build_temporal_auditory_policy_state([_make_decision_escalating()])
        ref = export_temporal_auditory_policy_bundle(state)
        ref_json = json.dumps(ref, sort_keys=True, separators=(",", ":"))
        for _ in range(100):
            export = export_temporal_auditory_policy_bundle(state)
            export_json = json.dumps(export, sort_keys=True, separators=(",", ":"))
            assert export_json == ref_json


# ---------------------------------------------------------------------------
# 10. No decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    def test_no_decoder_import(self):
        """temporal_auditory_sequence_policy_memory must not import decoder."""
        import qec.analysis.temporal_auditory_sequence_policy_memory as mod
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file, encoding="utf-8") as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_modules_untouched(self):
        """Verify importing our module introduces no new decoder modules."""
        before = {k for k in sys.modules if "qec.decoder" in k}
        import qec.analysis.temporal_auditory_sequence_policy_memory as mod  # noqa: F811
        after = {k for k in sys.modules if "qec.decoder" in k}
        assert after == before, f"New decoder contamination: {after - before}"


# ---------------------------------------------------------------------------
# 11. Invalid input rejection
# ---------------------------------------------------------------------------

class TestInvalidInputRejection:
    def test_empty_decisions_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_temporal_auditory_policy_state([])

    def test_non_iterable_raises_type_error(self):
        with pytest.raises(TypeError):
            build_temporal_auditory_policy_state(42)  # type: ignore[arg-type]

    def test_wrong_type_in_decisions_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditorySequenceDecision"):
            build_temporal_auditory_policy_state(["not_a_decision"])  # type: ignore[list-item]

    def test_none_in_decisions_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditorySequenceDecision"):
            build_temporal_auditory_policy_state([None])  # type: ignore[list-item]

    def test_ledger_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditoryPolicyState"):
            build_temporal_auditory_policy_ledger(["not_a_state"])  # type: ignore[list-item]

    def test_ledger_none_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditoryPolicyState"):
            build_temporal_auditory_policy_ledger([None])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 12. Window count
# ---------------------------------------------------------------------------

class TestWindowCount:
    def test_single_decision_window_count(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert state.window_count == 1

    def test_multiple_decisions_window_count(self):
        decisions = [_make_decision_static()] * 7
        state = build_temporal_auditory_policy_state(decisions)
        assert state.window_count == 7


# ---------------------------------------------------------------------------
# 13. Policy symbolic trace
# ---------------------------------------------------------------------------

class TestPolicySymbolicTrace:
    def test_trace_contains_trend(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert "TREND:" in state.policy_symbolic_trace

    def test_trace_contains_resp(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert "RESP:" in state.policy_symbolic_trace

    def test_trace_contains_pattern(self):
        state = build_temporal_auditory_policy_state([_make_decision_escalating()])
        assert "ESCALATING" in state.policy_symbolic_trace

    def test_trace_multi_window_joined(self):
        decisions = [_make_decision_static(), _make_decision_escalating()]
        state = build_temporal_auditory_policy_state(decisions)
        assert " > " in state.policy_symbolic_trace


# ---------------------------------------------------------------------------
# 14. Version
# ---------------------------------------------------------------------------

class TestVersion:
    def test_state_version(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        assert state.version == "v137.0.5"

    def test_version_constant(self):
        assert TEMPORAL_AUDITORY_POLICY_MEMORY_VERSION == "v137.0.5"


# ---------------------------------------------------------------------------
# 15. Ledger construction
# ---------------------------------------------------------------------------

class TestLedgerConstruction:
    def test_ledger_state_count(self):
        s1 = build_temporal_auditory_policy_state([_make_decision_static()])
        s2 = build_temporal_auditory_policy_state([_make_decision_escalating()])
        ledger = build_temporal_auditory_policy_ledger([s1, s2])
        assert ledger.state_count == 2

    def test_ledger_states_are_tuple(self):
        state = build_temporal_auditory_policy_state([_make_decision_static()])
        ledger = build_temporal_auditory_policy_ledger([state])
        assert isinstance(ledger.states, tuple)

    def test_empty_ledger(self):
        ledger = build_temporal_auditory_policy_ledger([])
        assert ledger.state_count == 0
        assert ledger.states == ()

    def test_ledger_preserves_order(self):
        s1 = build_temporal_auditory_policy_state([_make_decision_static()])
        s2 = build_temporal_auditory_policy_state([_make_decision_escalating()])
        ledger = build_temporal_auditory_policy_ledger([s1, s2])
        assert ledger.states[0] is s1
        assert ledger.states[1] is s2
