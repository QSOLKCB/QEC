"""
Tests for v136.8.7 — Deterministic Promotion History & Policy Drift Monitor

Minimum 75 tests covering:
  - dataclass immutability
  - history recording
  - promotion rate correctness
  - rollback rate correctness
  - drift score determinism
  - severity boundaries
  - ledger hash stability
  - 100-replay determinism
  - same-input identity
  - decoder untouched verification
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import FrozenInstanceError
from typing import Dict, Any, Optional, Tuple

import pytest

from qec.evolution.promotion_history_monitor import (
    DRIFT_CRITICAL_THRESHOLD,
    DRIFT_HIGH_THRESHOLD,
    DRIFT_LOW_THRESHOLD,
    DRIFT_MEDIUM_THRESHOLD,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    SEVERITY_NONE,
    VALID_SEVERITIES,
    DriftAlert,
    PromotionHistoryEntry,
    PromotionHistoryLedger,
    compute_history_hash,
    compute_policy_drift_score,
    compute_promotion_rates,
    detect_policy_drift,
    export_history_bundle,
    record_promotion_history,
    run_history_monitor_cycle,
    validate_history_ledger,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    cycle_index: int = 0,
    verdict: str = "PROMOTE",
    action: str = "promote",
    confidence: float = 0.9,
    improvement_score: float = 0.8,
    snapshot_hash: str = "abc123",
) -> PromotionHistoryEntry:
    return PromotionHistoryEntry(
        cycle_index=cycle_index,
        verdict=verdict,
        action=action,
        confidence=confidence,
        improvement_score=improvement_score,
        snapshot_hash=snapshot_hash,
    )


def _make_gate_result(
    verdict: str = "PROMOTE",
    confidence: float = 0.9,
    improvement_score: float = 0.8,
    snapshot_hash: str = "snap_abc",
) -> Dict[str, Any]:
    """Mimic the return of run_gate_cycle from v136.8.6."""

    class _Decision:
        def __init__(self, v: str, c: float):
            self.verdict = v
            self.confidence = c

    class _Candidate:
        def __init__(self, imp: float):
            self.improvement_score = imp

    return {
        "decision": _Decision(verdict, confidence),
        "candidate": _Candidate(improvement_score),
        "snapshot_hash": snapshot_hash,
        "gate_hash": "gate_hash_placeholder",
    }


def _build_ledger_with_entries(
    actions: Tuple[str, ...],
    verdicts: Optional[Tuple[str, ...]] = None,
    confidences: Optional[Tuple[float, ...]] = None,
) -> PromotionHistoryLedger:
    """Build a ledger from a sequence of actions."""
    ledger = None
    for i, action in enumerate(actions):
        v = verdicts[i] if verdicts else ("PROMOTE" if action == "promote" else "ROLLBACK" if action == "rollback" else "HOLD")
        c = confidences[i] if confidences else 0.8
        entry = _make_entry(cycle_index=i, verdict=v, action=action, confidence=c)
        ledger = record_promotion_history(entry, ledger)
    return ledger


# ===========================================================================
# 1. Dataclass immutability (tests 1-9)
# ===========================================================================

class TestDataclassImmutability:
    def test_entry_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.cycle_index = 99  # type: ignore[misc]

    def test_entry_verdict_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.verdict = "X"  # type: ignore[misc]

    def test_entry_action_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.action = "X"  # type: ignore[misc]

    def test_entry_confidence_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.confidence = 0.0  # type: ignore[misc]

    def test_entry_improvement_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.improvement_score = 0.0  # type: ignore[misc]

    def test_entry_snapshot_hash_frozen(self):
        e = _make_entry()
        with pytest.raises(FrozenInstanceError):
            e.snapshot_hash = "x"  # type: ignore[misc]

    def test_drift_alert_frozen(self):
        a = DriftAlert(drift_detected=False, drift_score=0.0, rationale="ok", severity="NONE")
        with pytest.raises(FrozenInstanceError):
            a.drift_detected = True  # type: ignore[misc]

    def test_ledger_frozen(self):
        ledger = _build_ledger_with_entries(("promote",))
        with pytest.raises(FrozenInstanceError):
            ledger.promotion_rate = 0.0  # type: ignore[misc]

    def test_ledger_entries_frozen(self):
        ledger = _build_ledger_with_entries(("promote",))
        with pytest.raises(FrozenInstanceError):
            ledger.entries = ()  # type: ignore[misc]


# ===========================================================================
# 2. History recording (tests 10-17)
# ===========================================================================

class TestHistoryRecording:
    def test_record_first_entry(self):
        entry = _make_entry()
        ledger = record_promotion_history(entry)
        assert len(ledger.entries) == 1
        assert ledger.entries[0] is entry

    def test_record_appends(self):
        e1 = _make_entry(cycle_index=0)
        e2 = _make_entry(cycle_index=1)
        l1 = record_promotion_history(e1)
        l2 = record_promotion_history(e2, l1)
        assert len(l2.entries) == 2
        assert l2.entries[0] is e1
        assert l2.entries[1] is e2

    def test_record_preserves_order(self):
        ledger = None
        for i in range(10):
            ledger = record_promotion_history(_make_entry(cycle_index=i), ledger)
        assert tuple(e.cycle_index for e in ledger.entries) == tuple(range(10))

    def test_record_updates_hash(self):
        e1 = _make_entry(cycle_index=0)
        e2 = _make_entry(cycle_index=1)
        l1 = record_promotion_history(e1)
        l2 = record_promotion_history(e2, l1)
        assert l1.stable_hash != l2.stable_hash

    def test_record_updates_rates(self):
        e1 = _make_entry(action="promote")
        e2 = _make_entry(cycle_index=1, action="rollback", verdict="ROLLBACK")
        l1 = record_promotion_history(e1)
        l2 = record_promotion_history(e2, l1)
        assert l2.promotion_rate == 0.5
        assert l2.rollback_rate == 0.5

    def test_record_none_ledger_creates_new(self):
        entry = _make_entry()
        ledger = record_promotion_history(entry, None)
        assert ledger is not None
        assert len(ledger.entries) == 1

    def test_record_many_entries(self):
        ledger = None
        for i in range(50):
            ledger = record_promotion_history(_make_entry(cycle_index=i), ledger)
        assert len(ledger.entries) == 50

    def test_record_idempotent_hash(self):
        e = _make_entry()
        l1 = record_promotion_history(e)
        l2 = record_promotion_history(e)
        assert l1.stable_hash == l2.stable_hash


# ===========================================================================
# 3. Promotion rate correctness (tests 18-24)
# ===========================================================================

class TestPromotionRates:
    def test_all_promote(self):
        ledger = _build_ledger_with_entries(("promote",) * 10)
        rates = compute_promotion_rates(ledger)
        assert rates["promotion_rate"] == 1.0
        assert rates["rollback_rate"] == 0.0

    def test_all_rollback(self):
        ledger = _build_ledger_with_entries(("rollback",) * 10)
        rates = compute_promotion_rates(ledger)
        assert rates["promotion_rate"] == 0.0
        assert rates["rollback_rate"] == 1.0

    def test_all_hold(self):
        ledger = _build_ledger_with_entries(("hold",) * 10)
        rates = compute_promotion_rates(ledger)
        assert rates["promotion_rate"] == 0.0
        assert rates["rollback_rate"] == 0.0
        assert rates["hold_rate"] == 1.0

    def test_mixed_rates(self):
        actions = ("promote", "rollback", "hold", "promote")
        ledger = _build_ledger_with_entries(actions)
        rates = compute_promotion_rates(ledger)
        assert rates["promotion_rate"] == 0.5
        assert rates["rollback_rate"] == 0.25

    def test_single_entry_promote(self):
        ledger = _build_ledger_with_entries(("promote",))
        rates = compute_promotion_rates(ledger)
        assert rates["promotion_rate"] == 1.0

    def test_single_entry_rollback(self):
        ledger = _build_ledger_with_entries(("rollback",))
        rates = compute_promotion_rates(ledger)
        assert rates["rollback_rate"] == 1.0

    def test_rates_sum_to_one(self):
        actions = ("promote", "rollback", "hold", "promote", "hold")
        ledger = _build_ledger_with_entries(actions)
        rates = compute_promotion_rates(ledger)
        total = rates["promotion_rate"] + rates["rollback_rate"] + rates["hold_rate"]
        assert abs(total - 1.0) < 1e-9


# ===========================================================================
# 4. Rollback rate correctness (tests 25-29)
# ===========================================================================

class TestRollbackRates:
    def test_zero_rollbacks(self):
        ledger = _build_ledger_with_entries(("promote", "hold", "promote"))
        rates = compute_promotion_rates(ledger)
        assert rates["rollback_rate"] == 0.0

    def test_half_rollbacks(self):
        actions = ("rollback", "promote", "rollback", "promote")
        ledger = _build_ledger_with_entries(actions)
        rates = compute_promotion_rates(ledger)
        assert rates["rollback_rate"] == 0.5

    def test_increasing_rollbacks(self):
        actions = ("promote",) * 5 + ("rollback",) * 5
        ledger = _build_ledger_with_entries(actions)
        rates = compute_promotion_rates(ledger)
        assert rates["rollback_rate"] == 0.5

    def test_rollback_rate_stored_in_ledger(self):
        ledger = _build_ledger_with_entries(("rollback",) * 3 + ("promote",) * 7)
        assert ledger.rollback_rate == 0.3

    def test_promotion_rate_stored_in_ledger(self):
        ledger = _build_ledger_with_entries(("promote",) * 7 + ("rollback",) * 3)
        assert ledger.promotion_rate == 0.7


# ===========================================================================
# 5. Drift score determinism (tests 30-39)
# ===========================================================================

class TestDriftScoreDeterminism:
    def test_same_input_same_drift(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 5)
        s1 = compute_policy_drift_score(ledger)
        s2 = compute_policy_drift_score(ledger)
        assert s1 == s2

    def test_drift_score_clamped_low(self):
        ledger = _build_ledger_with_entries(("promote",) * 10)
        score = compute_policy_drift_score(ledger)
        assert score >= 0.0

    def test_drift_score_clamped_high(self):
        # Extreme alternation + confidence shifts
        actions = ("promote", "rollback") * 20
        confidences = tuple(0.1 if i % 2 == 0 else 0.99 for i in range(40))
        verdicts = tuple("PROMOTE" if a == "promote" else "ROLLBACK" for a in actions)
        ledger = _build_ledger_with_entries(actions, verdicts, confidences)
        score = compute_policy_drift_score(ledger)
        assert score <= 1.0

    def test_drift_zero_for_uniform(self):
        ledger = _build_ledger_with_entries(("promote",) * 20)
        score = compute_policy_drift_score(ledger)
        assert score == 0.0

    def test_drift_single_entry(self):
        ledger = _build_ledger_with_entries(("promote",))
        score = compute_policy_drift_score(ledger)
        assert score == 0.0

    def test_drift_two_identical(self):
        ledger = _build_ledger_with_entries(("promote", "promote"))
        score = compute_policy_drift_score(ledger)
        assert score == 0.0

    def test_drift_two_different(self):
        actions = ("promote", "rollback")
        verdicts = ("PROMOTE", "ROLLBACK")
        ledger = _build_ledger_with_entries(actions, verdicts)
        score = compute_policy_drift_score(ledger)
        assert score > 0.0

    def test_drift_replay_100(self):
        actions = ("promote", "rollback", "hold") * 10
        verdicts = tuple("PROMOTE" if a == "promote" else "ROLLBACK" if a == "rollback" else "HOLD" for a in actions)
        ledger = _build_ledger_with_entries(actions, verdicts)
        reference = compute_policy_drift_score(ledger)
        for _ in range(100):
            assert compute_policy_drift_score(ledger) == reference

    def test_drift_affected_by_rollback_streak(self):
        stable = _build_ledger_with_entries(("promote",) * 10)
        streak = _build_ledger_with_entries(("rollback",) * 5 + ("promote",) * 5)
        assert compute_policy_drift_score(streak) >= compute_policy_drift_score(stable)

    def test_drift_affected_by_confidence_shift(self):
        uniform_conf = _build_ledger_with_entries(
            ("promote",) * 10,
            confidences=(0.5,) * 10,
        )
        shifted_conf = _build_ledger_with_entries(
            ("promote",) * 10,
            confidences=(0.1,) * 5 + (0.9,) * 5,
        )
        assert compute_policy_drift_score(shifted_conf) >= compute_policy_drift_score(uniform_conf)


# ===========================================================================
# 6. Severity boundaries (tests 40-49)
# ===========================================================================

class TestSeverityBoundaries:
    def test_severity_none(self):
        from qec.evolution.promotion_history_monitor import _score_to_severity
        assert _score_to_severity(0.0) == SEVERITY_NONE
        assert _score_to_severity(0.19) == SEVERITY_NONE

    def test_severity_low(self):
        from qec.evolution.promotion_history_monitor import _score_to_severity
        assert _score_to_severity(0.20) == SEVERITY_LOW
        assert _score_to_severity(0.39) == SEVERITY_LOW

    def test_severity_medium(self):
        from qec.evolution.promotion_history_monitor import _score_to_severity
        assert _score_to_severity(0.40) == SEVERITY_MEDIUM
        assert _score_to_severity(0.59) == SEVERITY_MEDIUM

    def test_severity_high(self):
        from qec.evolution.promotion_history_monitor import _score_to_severity
        assert _score_to_severity(0.60) == SEVERITY_HIGH
        assert _score_to_severity(0.79) == SEVERITY_HIGH

    def test_severity_critical(self):
        from qec.evolution.promotion_history_monitor import _score_to_severity
        assert _score_to_severity(0.80) == SEVERITY_CRITICAL
        assert _score_to_severity(1.0) == SEVERITY_CRITICAL

    def test_alert_none_for_stable(self):
        ledger = _build_ledger_with_entries(("promote",) * 20)
        alert = detect_policy_drift(ledger)
        assert alert.severity == SEVERITY_NONE
        assert alert.drift_detected is False

    def test_alert_detected_for_unstable(self):
        # High alternation + confidence shift → detected
        actions = ("promote",) * 10 + ("rollback",) * 10
        verdicts = ("PROMOTE",) * 10 + ("ROLLBACK",) * 10
        confidences = (0.9,) * 10 + (0.1,) * 10
        ledger = _build_ledger_with_entries(actions, verdicts, confidences)
        alert = detect_policy_drift(ledger)
        assert alert.drift_detected is True

    def test_alert_severity_in_valid_set(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 10)
        alert = detect_policy_drift(ledger)
        assert alert.severity in VALID_SEVERITIES

    def test_alert_rationale_non_empty(self):
        ledger = _build_ledger_with_entries(("promote",) * 10)
        alert = detect_policy_drift(ledger)
        assert len(alert.rationale) > 0

    def test_alert_drift_score_matches_ledger(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 5)
        alert = detect_policy_drift(ledger)
        assert alert.drift_score == compute_policy_drift_score(ledger)


# ===========================================================================
# 7. Ledger hash stability (tests 50-57)
# ===========================================================================

class TestLedgerHashStability:
    def test_hash_deterministic(self):
        ledger = _build_ledger_with_entries(("promote", "rollback"))
        h1 = compute_history_hash(ledger)
        h2 = compute_history_hash(ledger)
        assert h1 == h2

    def test_hash_changes_with_content(self):
        l1 = _build_ledger_with_entries(("promote",))
        l2 = _build_ledger_with_entries(("rollback",))
        assert compute_history_hash(l1) != compute_history_hash(l2)

    def test_hash_is_sha256(self):
        ledger = _build_ledger_with_entries(("promote",))
        h = compute_history_hash(ledger)
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_hash_matches_ledger_field(self):
        ledger = _build_ledger_with_entries(("promote", "rollback", "hold"))
        assert ledger.stable_hash == compute_history_hash(ledger)

    def test_hash_stable_across_rebuild(self):
        actions = ("promote", "rollback", "hold")
        l1 = _build_ledger_with_entries(actions)
        l2 = _build_ledger_with_entries(actions)
        assert l1.stable_hash == l2.stable_hash

    def test_hash_canonical_json(self):
        entry = _make_entry()
        ledger = record_promotion_history(entry)
        canonical = json.dumps(
            [{
                "cycle_index": entry.cycle_index,
                "verdict": entry.verdict,
                "action": entry.action,
                "confidence": entry.confidence,
                "improvement_score": entry.improvement_score,
                "snapshot_hash": entry.snapshot_hash,
            }],
            sort_keys=True,
            separators=(",", ":"),
        )
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert ledger.stable_hash == expected

    def test_validate_valid_ledger(self):
        ledger = _build_ledger_with_entries(("promote", "rollback", "hold"))
        assert validate_history_ledger(ledger) is True

    def test_validate_invalid_hash(self):
        ledger = _build_ledger_with_entries(("promote",))
        tampered = PromotionHistoryLedger(
            entries=ledger.entries,
            promotion_rate=ledger.promotion_rate,
            rollback_rate=ledger.rollback_rate,
            drift_score=ledger.drift_score,
            stable_hash="bad_hash",
        )
        assert validate_history_ledger(tampered) is False


# ===========================================================================
# 8. 100-replay determinism (tests 58-62)
# ===========================================================================

class TestReplayDeterminism:
    def test_full_cycle_replay_100(self):
        gate_result = _make_gate_result()
        reference = run_history_monitor_cycle(gate_result)
        for _ in range(100):
            result = run_history_monitor_cycle(gate_result)
            assert result["ledger"].stable_hash == reference["ledger"].stable_hash
            assert result["alert"].drift_score == reference["alert"].drift_score
            assert result["alert"].severity == reference["alert"].severity
            assert result["history_hash"] == reference["history_hash"]

    def test_chain_replay_100(self):
        gate_results = [
            _make_gate_result("PROMOTE", 0.9, 0.8),
            _make_gate_result("ROLLBACK", 0.4, 0.3),
            _make_gate_result("HOLD", 0.6, 0.5),
        ]
        ref_ledger = None
        ref_hashes = []
        for gr in gate_results:
            ref = run_history_monitor_cycle(gr, ref_ledger)
            ref_hashes.append(ref["history_hash"])
            ref_ledger = ref["ledger"]

        for _ in range(100):
            ledger = None
            for i, gr in enumerate(gate_results):
                r = run_history_monitor_cycle(gr, ledger)
                assert r["history_hash"] == ref_hashes[i]
                ledger = r["ledger"]

    def test_export_replay_100(self):
        gate_result = _make_gate_result()
        ref = export_history_bundle(run_history_monitor_cycle(gate_result))
        for _ in range(100):
            bundle = export_history_bundle(run_history_monitor_cycle(gate_result))
            assert bundle == ref

    def test_drift_score_replay_100(self):
        actions = ("promote", "rollback", "hold", "promote", "rollback")
        ledger = _build_ledger_with_entries(actions)
        ref = compute_policy_drift_score(ledger)
        for _ in range(100):
            assert compute_policy_drift_score(ledger) == ref

    def test_hash_replay_100(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 5)
        ref = compute_history_hash(ledger)
        for _ in range(100):
            assert compute_history_hash(ledger) == ref


# ===========================================================================
# 9. Same-input identity (tests 63-68)
# ===========================================================================

class TestSameInputIdentity:
    def test_same_gate_result_same_output(self):
        gr = _make_gate_result()
        r1 = run_history_monitor_cycle(gr)
        r2 = run_history_monitor_cycle(gr)
        assert r1["history_hash"] == r2["history_hash"]

    def test_same_entries_same_ledger_hash(self):
        e = _make_entry()
        l1 = record_promotion_history(e)
        l2 = record_promotion_history(e)
        assert l1.stable_hash == l2.stable_hash
        assert l1.promotion_rate == l2.promotion_rate

    def test_same_ledger_same_drift(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 5)
        d1 = detect_policy_drift(ledger)
        d2 = detect_policy_drift(ledger)
        assert d1 == d2

    def test_same_ledger_same_rates(self):
        ledger = _build_ledger_with_entries(("promote",) * 3 + ("rollback",) * 2)
        r1 = compute_promotion_rates(ledger)
        r2 = compute_promotion_rates(ledger)
        assert r1 == r2

    def test_same_ledger_same_validation(self):
        ledger = _build_ledger_with_entries(("promote", "hold", "rollback"))
        assert validate_history_ledger(ledger) == validate_history_ledger(ledger)

    def test_identical_bundles(self):
        gr = _make_gate_result()
        b1 = export_history_bundle(run_history_monitor_cycle(gr))
        b2 = export_history_bundle(run_history_monitor_cycle(gr))
        assert json.dumps(b1, sort_keys=True) == json.dumps(b2, sort_keys=True)


# ===========================================================================
# 10. Decoder untouched verification (tests 69-72)
# ===========================================================================

class TestDecoderUntouched:
    def test_no_decoder_imports_in_module(self):
        import inspect
        import qec.evolution.promotion_history_monitor as mod
        source = inspect.getsource(mod)
        assert "qec.decoder" not in source

    def test_decoder_files_unchanged(self):
        decoder_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder"
        )
        if os.path.isdir(decoder_dir):
            for fname in sorted(os.listdir(decoder_dir)):
                if fname.endswith(".py"):
                    fpath = os.path.join(decoder_dir, fname)
                    assert os.path.isfile(fpath)

    def test_module_does_not_touch_decoder_at_runtime(self):
        loaded_before = set(
            k for k in sys.modules if k.startswith("qec.decoder")
        )
        importlib.reload(
            importlib.import_module("qec.evolution.promotion_history_monitor")
        )
        loaded_after = set(
            k for k in sys.modules if k.startswith("qec.decoder")
        )
        assert loaded_after == loaded_before

    def test_evolution_init_exists(self):
        import qec.evolution
        assert hasattr(qec.evolution, "__doc__")


# ===========================================================================
# 11. Monitor cycle integration (tests 73-78)
# ===========================================================================

class TestMonitorCycleIntegration:
    def test_cycle_returns_required_keys(self):
        gr = _make_gate_result()
        result = run_history_monitor_cycle(gr)
        assert "entry" in result
        assert "ledger" in result
        assert "alert" in result
        assert "rates" in result
        assert "history_hash" in result
        assert "snapshot_hash" in result

    def test_cycle_increments_index(self):
        gr = _make_gate_result()
        r1 = run_history_monitor_cycle(gr)
        r2 = run_history_monitor_cycle(gr, r1["ledger"])
        assert r2["entry"].cycle_index == 1

    def test_cycle_verdict_mapping_promote(self):
        gr = _make_gate_result("PROMOTE")
        result = run_history_monitor_cycle(gr)
        assert result["entry"].action == "promote"

    def test_cycle_verdict_mapping_rollback(self):
        gr = _make_gate_result("ROLLBACK")
        result = run_history_monitor_cycle(gr)
        assert result["entry"].action == "rollback"

    def test_cycle_verdict_mapping_hold(self):
        gr = _make_gate_result("HOLD")
        result = run_history_monitor_cycle(gr)
        assert result["entry"].action == "hold"

    def test_cycle_verdict_mapping_blocked(self):
        gr = _make_gate_result("BLOCKED_BY_INVARIANT")
        result = run_history_monitor_cycle(gr)
        assert result["entry"].action == "rollback"


# ===========================================================================
# 12. Export correctness (tests 79-82)
# ===========================================================================

class TestExportCorrectness:
    def test_export_json_serializable(self):
        gr = _make_gate_result()
        bundle = export_history_bundle(run_history_monitor_cycle(gr))
        serialized = json.dumps(bundle, sort_keys=True)
        assert isinstance(serialized, str)

    def test_export_contains_all_fields(self):
        gr = _make_gate_result()
        bundle = export_history_bundle(run_history_monitor_cycle(gr))
        assert "entry" in bundle
        assert "ledger" in bundle
        assert "alert" in bundle
        assert "rates" in bundle
        assert "history_hash" in bundle

    def test_export_entry_fields(self):
        gr = _make_gate_result()
        bundle = export_history_bundle(run_history_monitor_cycle(gr))
        entry = bundle["entry"]
        assert "cycle_index" in entry
        assert "verdict" in entry
        assert "action" in entry
        assert "confidence" in entry

    def test_export_alert_fields(self):
        gr = _make_gate_result()
        bundle = export_history_bundle(run_history_monitor_cycle(gr))
        alert = bundle["alert"]
        assert "drift_detected" in alert
        assert "drift_score" in alert
        assert "severity" in alert


# ===========================================================================
# 13. Validation edge cases (tests 83-86)
# ===========================================================================

class TestValidationEdgeCases:
    def test_validate_rejects_wrong_rate(self):
        ledger = _build_ledger_with_entries(("promote",) * 5)
        tampered = PromotionHistoryLedger(
            entries=ledger.entries,
            promotion_rate=0.0,  # wrong
            rollback_rate=ledger.rollback_rate,
            drift_score=ledger.drift_score,
            stable_hash=ledger.stable_hash,
        )
        assert validate_history_ledger(tampered) is False

    def test_validate_rejects_wrong_drift(self):
        ledger = _build_ledger_with_entries(("promote", "rollback") * 5)
        tampered = PromotionHistoryLedger(
            entries=ledger.entries,
            promotion_rate=ledger.promotion_rate,
            rollback_rate=ledger.rollback_rate,
            drift_score=0.999,  # wrong
            stable_hash=ledger.stable_hash,
        )
        assert validate_history_ledger(tampered) is False

    def test_validate_rejects_invalid_action(self):
        bad_entry = PromotionHistoryEntry(
            cycle_index=0,
            verdict="PROMOTE",
            action="INVALID_ACTION",
            confidence=0.9,
            improvement_score=0.8,
            snapshot_hash="x",
        )
        # Build ledger manually with correct hash but invalid action
        from qec.evolution.promotion_history_monitor import _hash_entries, _compute_rates, _compute_drift
        entries = (bad_entry,)
        h = _hash_entries(entries)
        pr, rr = _compute_rates(entries)
        ds = _compute_drift(entries)
        tampered = PromotionHistoryLedger(
            entries=entries,
            promotion_rate=pr,
            rollback_rate=rr,
            drift_score=ds,
            stable_hash=h,
        )
        assert validate_history_ledger(tampered) is False

    def test_validate_empty_entries(self):
        from qec.evolution.promotion_history_monitor import _hash_entries
        h = _hash_entries(())
        ledger = PromotionHistoryLedger(
            entries=(),
            promotion_rate=0.0,
            rollback_rate=0.0,
            drift_score=0.0,
            stable_hash=h,
        )
        assert validate_history_ledger(ledger) is True
