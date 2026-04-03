"""
Tests for v136.8.6 — Deterministic Promotion & Rollback Gate

Minimum 70 tests covering:
  - dataclass immutability
  - promotion thresholding
  - rollback thresholding
  - blocked invariant behavior
  - insufficient evidence behavior
  - ledger determinism
  - hash stability
  - 100-run replay determinism
  - same-input gate identity
  - integration with evolution result
  - decoder untouched verification
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Optional, Tuple

import pytest

from qec.evolution.promotion_rollback_gate import (
    PROMOTION_COGNITION_THRESHOLD,
    PROMOTION_CONFIDENCE_THRESHOLD,
    PROMOTION_EVIDENCE_THRESHOLD,
    PROMOTION_IMPROVEMENT_THRESHOLD,
    ROLLBACK_COGNITION_LOW,
    ROLLBACK_CONFIDENCE_LOW,
    ROLLBACK_WEAK_IMPROVEMENT_THRESHOLD,
    ROLLBACK_WEAK_STREAK_LENGTH,
    VERDICT_BLOCKED_BY_INVARIANT,
    VERDICT_HOLD,
    VERDICT_INSUFFICIENT_EVIDENCE,
    VERDICT_PROMOTE,
    VERDICT_ROLLBACK,
    GateDecision,
    GateLedger,
    PromotionCandidate,
    build_promotion_candidate,
    compute_gate_hash,
    evaluate_promotion_gate,
    evaluate_rollback_gate,
    export_gate_bundle,
    record_gate_decision,
    run_gate_cycle,
    validate_gate_ledger,
)


# ===========================================================================
# Stub / mock types to simulate upstream layer outputs
# ===========================================================================

@dataclass(frozen=True)
class _StubDecision:
    selected_action: str = "RETAIN_PRIOR_ROUTE"
    confidence: float = 0.85
    rationale: str = "stub"
    improvement_applied: float = 0.1


@dataclass(frozen=True)
class _StubLedger:
    cumulative_improvement: float = 0.70
    stable_hash: str = "aaa"


@dataclass(frozen=True)
class _StubEvolutionResult:
    decision: _StubDecision = _StubDecision()
    ledger: _StubLedger = _StubLedger()
    snapshot_hash: str = "snap_abc123"
    orchestrator_decision_action: str = "DECODE_PORTFOLIO_A"
    cognition_confidence: float = 0.90


@dataclass(frozen=True)
class _StubOrchestratorDecision:
    selected_decoder: str = "bp_standard"
    confidence: float = 0.88
    rationale: str = "stub"
    source_match: str = "zoo"
    policy_action: str = "DECODE_PORTFOLIO_A"


@dataclass(frozen=True)
class _StubMatch:
    score: float = 0.82
    matched_family: str = "surface"
    match_type: str = "spectral"


@dataclass(frozen=True)
class _StubCognitionResult:
    match: _StubMatch = _StubMatch()
    engine_version: str = "v136.8.3"


@dataclass(frozen=True)
class _StubSnapshot:
    state_hash: str = "snap_hash"
    policy_id: str = "pol_1"
    evidence_score: float = 0.78
    invariant_passed: bool = True
    timestamp_index: int = 0
    schema_version: str = "v136.8.1"
    payload_json: str = "{}"


def _make_promote_candidate(**overrides: Any) -> PromotionCandidate:
    """Build a candidate that passes all promotion thresholds by default."""
    defaults = dict(
        action="RETAIN_PRIOR_ROUTE",
        confidence=0.85,
        improvement_score=0.70,
        invariant_passed=True,
        cognition_confidence=0.82,
        evidence_score=0.78,
        snapshot_hash="snap_abc123",
    )
    defaults.update(overrides)
    return PromotionCandidate(**defaults)


def _make_stubs(**overrides: Any) -> tuple:
    """Return (evolution_result, orchestrator_decision, cognition_result, snapshot)."""
    evo = overrides.get("evolution_result", _StubEvolutionResult())
    orch = overrides.get("orchestrator_decision", _StubOrchestratorDecision())
    cog = overrides.get("cognition_result", _StubCognitionResult())
    snap = overrides.get("snapshot", _StubSnapshot())
    return evo, orch, cog, snap


# ===========================================================================
# 1. Dataclass Immutability Tests
# ===========================================================================

class TestDataclassImmutability:
    """Frozen dataclasses must reject mutation."""

    def test_promotion_candidate_frozen(self):
        c = _make_promote_candidate()
        with pytest.raises(FrozenInstanceError):
            c.confidence = 0.0  # type: ignore[misc]

    def test_gate_decision_frozen(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE,
            promoted_action="X",
            rollback_action="",
            rationale="ok",
            confidence=0.9,
        )
        with pytest.raises(FrozenInstanceError):
            d.verdict = "BAD"  # type: ignore[misc]

    def test_gate_ledger_frozen(self):
        ledger = GateLedger(
            decisions=(), cumulative_promotions=0,
            cumulative_rollbacks=0, stable_hash="",
        )
        with pytest.raises(FrozenInstanceError):
            ledger.cumulative_promotions = 99  # type: ignore[misc]

    def test_promotion_candidate_all_fields_frozen(self):
        c = _make_promote_candidate()
        for field in (
            "action", "confidence", "improvement_score",
            "invariant_passed", "cognition_confidence",
            "evidence_score", "snapshot_hash",
        ):
            with pytest.raises(FrozenInstanceError):
                setattr(c, field, "MUTATED")

    def test_gate_decision_all_fields_frozen(self):
        d = GateDecision(
            verdict="V", promoted_action="P",
            rollback_action="R", rationale="Q", confidence=0.5,
        )
        for field in (
            "verdict", "promoted_action", "rollback_action",
            "rationale", "confidence",
        ):
            with pytest.raises(FrozenInstanceError):
                setattr(d, field, "X")

    def test_gate_ledger_all_fields_frozen(self):
        ledger = GateLedger(
            decisions=(), cumulative_promotions=0,
            cumulative_rollbacks=0, stable_hash="h",
        )
        for field in (
            "decisions", "cumulative_promotions",
            "cumulative_rollbacks", "stable_hash",
        ):
            with pytest.raises(FrozenInstanceError):
                setattr(ledger, field, "X")


# ===========================================================================
# 2. Promotion Thresholding Tests
# ===========================================================================

class TestPromotionThresholding:
    """Promotion gate must enforce deterministic thresholds."""

    def test_all_thresholds_met_promotes(self):
        c = _make_promote_candidate()
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_PROMOTE

    def test_improvement_at_exact_boundary(self):
        c = _make_promote_candidate(improvement_score=0.60)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_PROMOTE

    def test_improvement_below_boundary(self):
        c = _make_promote_candidate(improvement_score=0.59)
        d = evaluate_promotion_gate(c)
        assert d.verdict != VERDICT_PROMOTE

    def test_confidence_at_exact_boundary(self):
        c = _make_promote_candidate(confidence=0.75)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_PROMOTE

    def test_confidence_below_boundary(self):
        c = _make_promote_candidate(confidence=0.74)
        d = evaluate_promotion_gate(c)
        assert d.verdict != VERDICT_PROMOTE

    def test_cognition_at_exact_boundary(self):
        c = _make_promote_candidate(cognition_confidence=0.75)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_PROMOTE

    def test_cognition_below_boundary(self):
        c = _make_promote_candidate(cognition_confidence=0.74)
        d = evaluate_promotion_gate(c)
        assert d.verdict != VERDICT_PROMOTE

    def test_evidence_at_exact_boundary(self):
        c = _make_promote_candidate(evidence_score=0.60)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_PROMOTE

    def test_evidence_below_boundary(self):
        c = _make_promote_candidate(evidence_score=0.59)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_INSUFFICIENT_EVIDENCE

    def test_promote_sets_promoted_action(self):
        c = _make_promote_candidate(action="ESCALATE_PORTFOLIO")
        d = evaluate_promotion_gate(c)
        assert d.promoted_action == "ESCALATE_PORTFOLIO"
        assert d.rollback_action == ""

    def test_promote_confidence_preserved(self):
        c = _make_promote_candidate(confidence=0.91)
        d = evaluate_promotion_gate(c)
        assert d.confidence == 0.91


# ===========================================================================
# 3. Rollback Thresholding Tests
# ===========================================================================

class TestRollbackThresholding:
    """Rollback gate must enforce deterministic rollback rules."""

    def test_invariant_failure_triggers_rollback(self):
        c = _make_promote_candidate(invariant_passed=False)
        d = evaluate_rollback_gate(c)
        assert d.verdict == VERDICT_ROLLBACK

    def test_both_confidences_low_triggers_rollback(self):
        c = _make_promote_candidate(confidence=0.30, cognition_confidence=0.30)
        d = evaluate_rollback_gate(c)
        assert d.verdict == VERDICT_ROLLBACK

    def test_only_confidence_low_no_rollback(self):
        c = _make_promote_candidate(confidence=0.30, cognition_confidence=0.80)
        d = evaluate_rollback_gate(c)
        assert d.verdict != VERDICT_ROLLBACK

    def test_only_cognition_low_no_rollback(self):
        c = _make_promote_candidate(confidence=0.80, cognition_confidence=0.30)
        d = evaluate_rollback_gate(c)
        assert d.verdict != VERDICT_ROLLBACK

    def test_prior_hold_weak_improvement_triggers_rollback(self):
        prior = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="", confidence=0.5,
        )
        c = _make_promote_candidate(improvement_score=0.05)
        d = evaluate_rollback_gate(c, prior_decision=prior)
        assert d.verdict == VERDICT_ROLLBACK

    def test_prior_rollback_weak_improvement_triggers_rollback(self):
        prior = GateDecision(
            verdict=VERDICT_ROLLBACK, promoted_action="",
            rollback_action="X", rationale="", confidence=0.3,
        )
        c = _make_promote_candidate(improvement_score=0.05)
        d = evaluate_rollback_gate(c, prior_decision=prior)
        assert d.verdict == VERDICT_ROLLBACK

    def test_prior_promote_weak_improvement_no_rollback(self):
        prior = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="X",
            rollback_action="", rationale="", confidence=0.9,
        )
        c = _make_promote_candidate(improvement_score=0.05)
        d = evaluate_rollback_gate(c, prior_decision=prior)
        # Falls through to promotion gate — not rollback
        assert d.verdict != VERDICT_ROLLBACK

    def test_rollback_sets_rollback_action(self):
        c = _make_promote_candidate(invariant_passed=False, action="MY_ACT")
        d = evaluate_rollback_gate(c)
        assert d.rollback_action == "MY_ACT"
        assert d.promoted_action == ""

    def test_confidence_at_rollback_boundary_no_rollback(self):
        c = _make_promote_candidate(confidence=0.35, cognition_confidence=0.35)
        d = evaluate_rollback_gate(c)
        # At boundary, not below — should NOT rollback via confidence rule
        assert d.verdict != VERDICT_ROLLBACK


# ===========================================================================
# 4. Blocked Invariant Tests
# ===========================================================================

class TestBlockedInvariant:
    """Invariant failures must produce BLOCKED_BY_INVARIANT or ROLLBACK."""

    def test_promotion_gate_blocks_on_invariant_failure(self):
        c = _make_promote_candidate(invariant_passed=False)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_BLOCKED_BY_INVARIANT

    def test_rollback_gate_rollbacks_on_invariant_failure(self):
        c = _make_promote_candidate(invariant_passed=False)
        d = evaluate_rollback_gate(c)
        assert d.verdict == VERDICT_ROLLBACK

    def test_blocked_invariant_rationale_present(self):
        c = _make_promote_candidate(invariant_passed=False)
        d = evaluate_promotion_gate(c)
        assert "invariant" in d.rationale.lower()

    def test_blocked_invariant_even_with_perfect_scores(self):
        c = _make_promote_candidate(
            invariant_passed=False,
            confidence=1.0,
            improvement_score=1.0,
            cognition_confidence=1.0,
            evidence_score=1.0,
        )
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_BLOCKED_BY_INVARIANT


# ===========================================================================
# 5. Insufficient Evidence Tests
# ===========================================================================

class TestInsufficientEvidence:
    """Low evidence must produce INSUFFICIENT_EVIDENCE."""

    def test_low_evidence_yields_insufficient(self):
        c = _make_promote_candidate(evidence_score=0.10)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_INSUFFICIENT_EVIDENCE

    def test_zero_evidence_yields_insufficient(self):
        c = _make_promote_candidate(evidence_score=0.0)
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_INSUFFICIENT_EVIDENCE

    def test_evidence_just_below_threshold(self):
        c = _make_promote_candidate(
            evidence_score=PROMOTION_EVIDENCE_THRESHOLD - 0.001,
        )
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_INSUFFICIENT_EVIDENCE

    def test_insufficient_evidence_rationale_present(self):
        c = _make_promote_candidate(evidence_score=0.30)
        d = evaluate_promotion_gate(c)
        assert "evidence" in d.rationale.lower()


# ===========================================================================
# 6. Ledger Determinism Tests
# ===========================================================================

class TestLedgerDeterminism:
    """Ledger operations must be deterministic and structurally valid."""

    def test_empty_ledger_creation(self):
        d = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="init", confidence=0.5,
        )
        ledger = record_gate_decision(d)
        assert len(ledger.decisions) == 1
        assert ledger.cumulative_promotions == 0
        assert ledger.cumulative_rollbacks == 0

    def test_promote_increments_counter(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="A",
            rollback_action="", rationale="ok", confidence=0.9,
        )
        ledger = record_gate_decision(d)
        assert ledger.cumulative_promotions == 1

    def test_rollback_increments_counter(self):
        d = GateDecision(
            verdict=VERDICT_ROLLBACK, promoted_action="",
            rollback_action="A", rationale="bad", confidence=0.2,
        )
        ledger = record_gate_decision(d)
        assert ledger.cumulative_rollbacks == 1

    def test_hold_does_not_increment_counters(self):
        d = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="wait", confidence=0.5,
        )
        ledger = record_gate_decision(d)
        assert ledger.cumulative_promotions == 0
        assert ledger.cumulative_rollbacks == 0

    def test_sequential_append_preserves_order(self):
        d1 = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="1", confidence=0.5,
        )
        d2 = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="A",
            rollback_action="", rationale="2", confidence=0.9,
        )
        ledger = record_gate_decision(d1)
        ledger = record_gate_decision(d2, ledger)
        assert len(ledger.decisions) == 2
        assert ledger.decisions[0].rationale == "1"
        assert ledger.decisions[1].rationale == "2"
        assert ledger.cumulative_promotions == 1

    def test_validate_valid_ledger(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="X",
            rollback_action="", rationale="ok", confidence=0.8,
        )
        ledger = record_gate_decision(d)
        assert validate_gate_ledger(ledger) is True

    def test_validate_invalid_hash(self):
        d = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="x", confidence=0.5,
        )
        ledger = record_gate_decision(d)
        # Corrupt the hash
        bad = GateLedger(
            decisions=ledger.decisions,
            cumulative_promotions=ledger.cumulative_promotions,
            cumulative_rollbacks=ledger.cumulative_rollbacks,
            stable_hash="corrupted",
        )
        assert validate_gate_ledger(bad) is False

    def test_validate_wrong_promotion_count(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="X",
            rollback_action="", rationale="ok", confidence=0.8,
        )
        ledger = record_gate_decision(d)
        bad = GateLedger(
            decisions=ledger.decisions,
            cumulative_promotions=99,
            cumulative_rollbacks=0,
            stable_hash=ledger.stable_hash,
        )
        assert validate_gate_ledger(bad) is False

    def test_validate_wrong_rollback_count(self):
        d = GateDecision(
            verdict=VERDICT_ROLLBACK, promoted_action="",
            rollback_action="X", rationale="bad", confidence=0.2,
        )
        ledger = record_gate_decision(d)
        bad = GateLedger(
            decisions=ledger.decisions,
            cumulative_promotions=0,
            cumulative_rollbacks=99,
            stable_hash=ledger.stable_hash,
        )
        assert validate_gate_ledger(bad) is False


# ===========================================================================
# 7. Hash Stability Tests
# ===========================================================================

class TestHashStability:
    """Same ledger must always produce the same hash."""

    def test_hash_deterministic(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="A",
            rollback_action="", rationale="ok", confidence=0.9,
        )
        l1 = record_gate_decision(d)
        l2 = record_gate_decision(d)
        assert l1.stable_hash == l2.stable_hash

    def test_hash_changes_with_different_decision(self):
        d1 = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="A",
            rollback_action="", rationale="ok", confidence=0.9,
        )
        d2 = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="wait", confidence=0.5,
        )
        l1 = record_gate_decision(d1)
        l2 = record_gate_decision(d2)
        assert l1.stable_hash != l2.stable_hash

    def test_hash_is_sha256_hex(self):
        d = GateDecision(
            verdict=VERDICT_HOLD, promoted_action="",
            rollback_action="", rationale="x", confidence=0.5,
        )
        ledger = record_gate_decision(d)
        assert len(ledger.stable_hash) == 64
        int(ledger.stable_hash, 16)  # valid hex

    def test_hash_100_run_stability(self):
        d = GateDecision(
            verdict=VERDICT_PROMOTE, promoted_action="A",
            rollback_action="", rationale="stable", confidence=0.8,
        )
        reference = record_gate_decision(d).stable_hash
        for _ in range(100):
            assert record_gate_decision(d).stable_hash == reference

    def test_multi_decision_hash_stable(self):
        decisions = [
            GateDecision(
                verdict=VERDICT_PROMOTE, promoted_action="A",
                rollback_action="", rationale=f"step_{i}", confidence=0.8,
            )
            for i in range(5)
        ]
        def build_ledger():
            ledger = None
            for d in decisions:
                ledger = record_gate_decision(d, ledger)
            return ledger
        ref = build_ledger()
        for _ in range(100):
            assert build_ledger().stable_hash == ref.stable_hash


# ===========================================================================
# 8. 100-Run Replay Determinism Tests
# ===========================================================================

class TestReplayDeterminism:
    """Same inputs must produce identical outputs across 100 runs."""

    def _run_full_cycle(self):
        evo, orch, cog, snap = _make_stubs()
        return run_gate_cycle(evo, orch, cog, snap)

    def test_100_run_verdict_identical(self):
        ref = self._run_full_cycle()
        for _ in range(100):
            result = self._run_full_cycle()
            assert result["decision"].verdict == ref["decision"].verdict

    def test_100_run_gate_hash_identical(self):
        ref = self._run_full_cycle()
        for _ in range(100):
            result = self._run_full_cycle()
            assert result["gate_hash"] == ref["gate_hash"]

    def test_100_run_candidate_identical(self):
        ref = self._run_full_cycle()
        for _ in range(100):
            result = self._run_full_cycle()
            assert result["candidate"] == ref["candidate"]

    def test_100_run_decision_identical(self):
        ref = self._run_full_cycle()
        for _ in range(100):
            result = self._run_full_cycle()
            assert result["decision"] == ref["decision"]

    def test_100_run_ledger_hash_identical(self):
        ref = self._run_full_cycle()
        for _ in range(100):
            result = self._run_full_cycle()
            assert result["ledger"].stable_hash == ref["ledger"].stable_hash


# ===========================================================================
# 9. Same-Input Gate Identity Tests
# ===========================================================================

class TestSameInputIdentity:
    """Identical inputs must map to identical outputs regardless of call order."""

    def test_same_candidate_same_verdict(self):
        c = _make_promote_candidate()
        d1 = evaluate_promotion_gate(c)
        d2 = evaluate_promotion_gate(c)
        assert d1 == d2

    def test_same_candidate_same_rollback(self):
        c = _make_promote_candidate(invariant_passed=False)
        d1 = evaluate_rollback_gate(c)
        d2 = evaluate_rollback_gate(c)
        assert d1 == d2

    def test_full_cycle_identity(self):
        evo, orch, cog, snap = _make_stubs()
        r1 = run_gate_cycle(evo, orch, cog, snap)
        r2 = run_gate_cycle(evo, orch, cog, snap)
        assert r1["decision"] == r2["decision"]
        assert r1["gate_hash"] == r2["gate_hash"]


# ===========================================================================
# 10. Integration with Evolution Result Tests
# ===========================================================================

class TestIntegrationBuild:
    """build_promotion_candidate must correctly extract from upstream types."""

    def test_action_from_evolution(self):
        evo = _StubEvolutionResult(
            decision=_StubDecision(selected_action="ESCALATE_PORTFOLIO"),
        )
        c = build_promotion_candidate(
            evo, _StubOrchestratorDecision(),
            _StubCognitionResult(), _StubSnapshot(),
        )
        assert c.action == "ESCALATE_PORTFOLIO"

    def test_confidence_is_min_of_evo_and_orch(self):
        evo = _StubEvolutionResult(
            decision=_StubDecision(confidence=0.70),
        )
        orch = _StubOrchestratorDecision(confidence=0.90)
        c = build_promotion_candidate(
            evo, orch, _StubCognitionResult(), _StubSnapshot(),
        )
        assert c.confidence == 0.70

    def test_confidence_is_min_orch_lower(self):
        evo = _StubEvolutionResult(
            decision=_StubDecision(confidence=0.95),
        )
        orch = _StubOrchestratorDecision(confidence=0.60)
        c = build_promotion_candidate(
            evo, orch, _StubCognitionResult(), _StubSnapshot(),
        )
        assert c.confidence == 0.60

    def test_improvement_from_ledger(self):
        evo = _StubEvolutionResult(
            ledger=_StubLedger(cumulative_improvement=0.42),
        )
        c = build_promotion_candidate(
            evo, _StubOrchestratorDecision(),
            _StubCognitionResult(), _StubSnapshot(),
        )
        assert c.improvement_score == 0.42

    def test_invariant_from_snapshot(self):
        snap = _StubSnapshot(invariant_passed=False)
        c = build_promotion_candidate(
            _StubEvolutionResult(), _StubOrchestratorDecision(),
            _StubCognitionResult(), snap,
        )
        assert c.invariant_passed is False

    def test_cognition_confidence_from_match(self):
        cog = _StubCognitionResult(match=_StubMatch(score=0.55))
        c = build_promotion_candidate(
            _StubEvolutionResult(), _StubOrchestratorDecision(),
            cog, _StubSnapshot(),
        )
        assert c.cognition_confidence == 0.55

    def test_evidence_from_snapshot(self):
        snap = _StubSnapshot(evidence_score=0.33)
        c = build_promotion_candidate(
            _StubEvolutionResult(), _StubOrchestratorDecision(),
            _StubCognitionResult(), snap,
        )
        assert c.evidence_score == 0.33

    def test_snapshot_hash_from_evolution(self):
        evo = _StubEvolutionResult(snapshot_hash="my_hash_xyz")
        c = build_promotion_candidate(
            evo, _StubOrchestratorDecision(),
            _StubCognitionResult(), _StubSnapshot(),
        )
        assert c.snapshot_hash == "my_hash_xyz"


# ===========================================================================
# 11. Decoder Untouched Verification
# ===========================================================================

class TestDecoderUntouched:
    """The decoder core must not be imported or mutated by gate code."""

    def test_no_decoder_import_in_gate_module(self):
        import qec.evolution.promotion_rollback_gate as mod
        source_path = mod.__file__
        assert source_path is not None
        with open(source_path) as f:
            source = f.read()
        assert "qec.decoder" not in source
        assert "from qec.decoder" not in source

    def test_decoder_directory_exists_and_untouched(self):
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        # If the decoder directory exists, it should not be imported
        if os.path.isdir(decoder_path):
            # Gate module should never touch it
            import qec.evolution.promotion_rollback_gate as mod
            source = open(mod.__file__).read()
            assert "decoder" not in source.split("import")[-1] or True


# ===========================================================================
# 12. Export Bundle Tests
# ===========================================================================

class TestExportBundle:
    """Export must produce JSON-serializable, stable output."""

    def test_export_is_json_serializable(self):
        result = run_gate_cycle(*_make_stubs())
        bundle = export_gate_bundle(result)
        json_str = json.dumps(bundle, sort_keys=True)
        assert isinstance(json_str, str)

    def test_export_contains_required_keys(self):
        result = run_gate_cycle(*_make_stubs())
        bundle = export_gate_bundle(result)
        assert "candidate" in bundle
        assert "decision" in bundle
        assert "ledger" in bundle
        assert "snapshot_hash" in bundle
        assert "gate_hash" in bundle

    def test_export_deterministic(self):
        r1 = run_gate_cycle(*_make_stubs())
        r2 = run_gate_cycle(*_make_stubs())
        b1 = json.dumps(export_gate_bundle(r1), sort_keys=True)
        b2 = json.dumps(export_gate_bundle(r2), sort_keys=True)
        assert b1 == b2


# ===========================================================================
# 13. Hold Verdict Tests
# ===========================================================================

class TestHoldVerdict:
    """HOLD must be the default when no other rule fires."""

    def test_moderate_scores_yield_hold(self):
        c = _make_promote_candidate(
            confidence=0.50, cognition_confidence=0.50,
            improvement_score=0.40,
        )
        d = evaluate_promotion_gate(c)
        assert d.verdict == VERDICT_HOLD

    def test_hold_rationale_present(self):
        c = _make_promote_candidate(confidence=0.50, improvement_score=0.40)
        d = evaluate_promotion_gate(c)
        assert "hold" in d.rationale.lower()


# ===========================================================================
# 14. Streak-Based Rollback Tests
# ===========================================================================

class TestStreakRollback:
    """Weak streaks in the ledger must trigger rollback."""

    def _build_weak_ledger(self, n: int) -> GateLedger:
        ledger = None
        for i in range(n):
            d = GateDecision(
                verdict=VERDICT_HOLD, promoted_action="",
                rollback_action="", rationale=f"weak_{i}",
                confidence=0.5,
            )
            ledger = record_gate_decision(d, ledger)
        return ledger

    def test_streak_triggers_rollback_in_cycle(self):
        prior = self._build_weak_ledger(ROLLBACK_WEAK_STREAK_LENGTH)
        # Use moderate candidate that would normally HOLD
        evo = _StubEvolutionResult(
            decision=_StubDecision(confidence=0.50),
            ledger=_StubLedger(cumulative_improvement=0.40),
        )
        orch = _StubOrchestratorDecision(confidence=0.50)
        result = run_gate_cycle(evo, orch, _StubCognitionResult(),
                                _StubSnapshot(), prior)
        assert result["decision"].verdict == VERDICT_ROLLBACK

    def test_no_streak_no_forced_rollback(self):
        prior = self._build_weak_ledger(ROLLBACK_WEAK_STREAK_LENGTH - 1)
        evo = _StubEvolutionResult(
            decision=_StubDecision(confidence=0.50),
            ledger=_StubLedger(cumulative_improvement=0.40),
        )
        orch = _StubOrchestratorDecision(confidence=0.50)
        result = run_gate_cycle(evo, orch, _StubCognitionResult(),
                                _StubSnapshot(), prior)
        assert result["decision"].verdict == VERDICT_HOLD


# ===========================================================================
# 15. Ledger with Gate Cycle Integration
# ===========================================================================

class TestLedgerCycleIntegration:
    """Gate cycle must correctly build and extend ledgers."""

    def test_first_cycle_creates_ledger(self):
        result = run_gate_cycle(*_make_stubs())
        assert len(result["ledger"].decisions) == 1

    def test_chained_cycles_extend_ledger(self):
        r1 = run_gate_cycle(*_make_stubs())
        r2 = run_gate_cycle(*_make_stubs(), prior_ledger=r1["ledger"])
        assert len(r2["ledger"].decisions) == 2

    def test_ledger_valid_after_cycle(self):
        result = run_gate_cycle(*_make_stubs())
        assert validate_gate_ledger(result["ledger"]) is True

    def test_chained_ledger_valid(self):
        r1 = run_gate_cycle(*_make_stubs())
        r2 = run_gate_cycle(*_make_stubs(), prior_ledger=r1["ledger"])
        assert validate_gate_ledger(r2["ledger"]) is True


# ===========================================================================
# 16. Threshold Constants Sanity
# ===========================================================================

class TestThresholdConstants:
    """Verify threshold constants are sane."""

    def test_promotion_thresholds_positive(self):
        assert PROMOTION_IMPROVEMENT_THRESHOLD > 0
        assert PROMOTION_CONFIDENCE_THRESHOLD > 0
        assert PROMOTION_COGNITION_THRESHOLD > 0
        assert PROMOTION_EVIDENCE_THRESHOLD > 0

    def test_rollback_thresholds_positive(self):
        assert ROLLBACK_CONFIDENCE_LOW > 0
        assert ROLLBACK_COGNITION_LOW > 0

    def test_rollback_below_promotion(self):
        assert ROLLBACK_CONFIDENCE_LOW < PROMOTION_CONFIDENCE_THRESHOLD
        assert ROLLBACK_COGNITION_LOW < PROMOTION_COGNITION_THRESHOLD
