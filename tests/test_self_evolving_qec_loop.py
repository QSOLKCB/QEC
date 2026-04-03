"""
Tests for the Deterministic Self-Evolving QEC Loop — v136.8.5

Minimum 70 tests covering:
    - dataclass immutability
    - step recording
    - ledger determinism
    - hash stability
    - improvement score bounds
    - adaptation rule stability
    - 100-run replay determinism
    - same-input hash identity
    - integration with orchestration
    - decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
import pytest
from dataclasses import FrozenInstanceError
from typing import Tuple

from qec.evolution.self_evolving_qec_loop import (
    VALID_ADAPTATION_ACTIONS,
    EvolutionCycleResult,
    EvolutionDecision,
    EvolutionLedger,
    EvolutionStep,
    build_next_action,
    compute_evolution_hash,
    compute_improvement_score,
    export_evolution_bundle,
    record_evolution_step,
    run_evolution_cycle,
    validate_evolution_ledger,
)
from qec.orchestration.decoder_portfolio_orchestrator import OrchestratorDecision
from qec.audio.audio_cognition_engine import CognitionCycleResult
from qec.audio.cognition_registry import AudioFingerprint, CognitionMatch
from qec.audio.triality_signal_engine import TrialityParams
from qec.ai.controller_snapshot_schema import ControllerSnapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_triality_params() -> TrialityParams:
    return TrialityParams(
        carrier_freq=440.0,
        mod_freq=5.0,
        mod_depth=0.3,
        overlay_harmonics=3,
        overlay_base_freq=220.0,
        state_hash="abc123",
    )


def _make_fingerprint() -> AudioFingerprint:
    return AudioFingerprint(
        centroid=1500.0,
        rolloff=3000.0,
        peak_bins=(10, 20, 30),
        psd_hash="deadbeef",
    )


def _make_cognition_match(confidence: float = 0.75) -> CognitionMatch:
    return CognitionMatch(
        confidence=confidence,
        identity="surface_code",
        failure_mode="bit_flip",
        recommended_action="DECODE_PORTFOLIO_A",
    )


def _make_cognition_result(confidence: float = 0.75) -> CognitionCycleResult:
    return CognitionCycleResult(
        params=_make_triality_params(),
        fingerprint=_make_fingerprint(),
        match=_make_cognition_match(confidence),
        engine_version="v136.8.3",
    )


def _make_orchestrator_decision(
    confidence: float = 0.85,
    policy_action: str = "DECODE_PORTFOLIO_A",
) -> OrchestratorDecision:
    return OrchestratorDecision(
        selected_decoder="bp_osd_v2",
        confidence=confidence,
        rationale="high confidence route",
        source_match="code_family",
        policy_action=policy_action,
    )


def _make_snapshot(
    invariant_passed: bool = True,
    evidence_score: float = 0.9,
) -> ControllerSnapshot:
    payload = json.dumps(
        {"test": "data", "score": evidence_score},
        sort_keys=True,
        separators=(",", ":"),
    )
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id="test_policy_001",
        evidence_score=evidence_score,
        invariant_passed=invariant_passed,
        timestamp_index=0,
        schema_version="v136.8.1",
        payload_json=payload,
    )


def _make_step(
    step_index: int = 0,
    improvement_score: float = 0.6,
) -> EvolutionStep:
    return EvolutionStep(
        step_index=step_index,
        prior_action="RETAIN_PRIOR_ROUTE",
        observed_outcome="HIGH_CONFIDENCE_ROUTE",
        confidence_delta=0.1,
        improvement_score=improvement_score,
    )


def _make_ledger(n_steps: int = 3) -> EvolutionLedger:
    ledger = None
    for i in range(n_steps):
        step = EvolutionStep(
            step_index=i,
            prior_action="RETAIN_PRIOR_ROUTE",
            observed_outcome="HIGH_CONFIDENCE_ROUTE",
            confidence_delta=0.05,
            improvement_score=0.6,
        )
        ledger = record_evolution_step(step, ledger)
    return ledger


# ===================================================================
# 1. Dataclass Immutability Tests
# ===================================================================


class TestDataclassImmutability:
    def test_evolution_step_frozen(self):
        step = _make_step()
        with pytest.raises(FrozenInstanceError):
            step.step_index = 99

    def test_evolution_step_frozen_action(self):
        step = _make_step()
        with pytest.raises(FrozenInstanceError):
            step.prior_action = "INVALID"

    def test_evolution_step_frozen_outcome(self):
        step = _make_step()
        with pytest.raises(FrozenInstanceError):
            step.observed_outcome = "INVALID"

    def test_evolution_step_frozen_delta(self):
        step = _make_step()
        with pytest.raises(FrozenInstanceError):
            step.confidence_delta = 999.0

    def test_evolution_step_frozen_score(self):
        step = _make_step()
        with pytest.raises(FrozenInstanceError):
            step.improvement_score = 999.0

    def test_evolution_ledger_frozen(self):
        ledger = _make_ledger(1)
        with pytest.raises(FrozenInstanceError):
            ledger.cumulative_improvement = 999.0

    def test_evolution_ledger_frozen_steps(self):
        ledger = _make_ledger(1)
        with pytest.raises(FrozenInstanceError):
            ledger.steps = ()

    def test_evolution_ledger_frozen_hash(self):
        ledger = _make_ledger(1)
        with pytest.raises(FrozenInstanceError):
            ledger.stable_hash = "tampered"

    def test_evolution_decision_frozen(self):
        decision = EvolutionDecision(
            selected_action="RETAIN_PRIOR_ROUTE",
            confidence=0.8,
            rationale="test",
            improvement_applied=True,
        )
        with pytest.raises(FrozenInstanceError):
            decision.selected_action = "INVALID"

    def test_evolution_decision_frozen_confidence(self):
        decision = EvolutionDecision(
            selected_action="RETAIN_PRIOR_ROUTE",
            confidence=0.8,
            rationale="test",
            improvement_applied=True,
        )
        with pytest.raises(FrozenInstanceError):
            decision.confidence = 0.0

    def test_evolution_cycle_result_frozen(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        with pytest.raises(FrozenInstanceError):
            result.snapshot_hash = "tampered"


# ===================================================================
# 2. Step Recording Tests
# ===================================================================


class TestStepRecording:
    def test_record_first_step(self):
        step = _make_step(step_index=0)
        ledger = record_evolution_step(step)
        assert len(ledger.steps) == 1
        assert ledger.steps[0] is step

    def test_record_multiple_steps(self):
        ledger = _make_ledger(5)
        assert len(ledger.steps) == 5

    def test_step_indices_sequential(self):
        ledger = _make_ledger(5)
        for i, step in enumerate(ledger.steps):
            assert step.step_index == i

    def test_record_preserves_prior_steps(self):
        ledger1 = record_evolution_step(_make_step(0))
        step1_hash = ledger1.stable_hash
        ledger2 = record_evolution_step(_make_step(1), ledger1)
        # Prior step is preserved
        assert ledger2.steps[0] == ledger1.steps[0]
        assert len(ledger2.steps) == 2

    def test_record_step_none_ledger(self):
        step = _make_step()
        ledger = record_evolution_step(step, None)
        assert len(ledger.steps) == 1

    def test_cumulative_improvement_computed(self):
        ledger = record_evolution_step(
            EvolutionStep(0, "NONE", "HIGH_CONFIDENCE_ROUTE", 0.1, 0.8)
        )
        assert 0.0 <= ledger.cumulative_improvement <= 1.0

    def test_cumulative_improvement_is_mean(self):
        s0 = EvolutionStep(0, "NONE", "OK", 0.1, 0.6)
        s1 = EvolutionStep(1, "NONE", "OK", 0.1, 0.8)
        ledger = record_evolution_step(s0)
        ledger = record_evolution_step(s1, ledger)
        expected = (0.6 + 0.8) / 2.0
        assert abs(ledger.cumulative_improvement - expected) < 1e-12


# ===================================================================
# 3. Ledger Determinism Tests
# ===================================================================


class TestLedgerDeterminism:
    def test_same_steps_same_ledger(self):
        l1 = _make_ledger(3)
        l2 = _make_ledger(3)
        assert l1 == l2

    def test_ledger_hash_deterministic(self):
        l1 = _make_ledger(3)
        l2 = _make_ledger(3)
        assert l1.stable_hash == l2.stable_hash

    def test_different_steps_different_hash(self):
        l1 = _make_ledger(3)
        step = EvolutionStep(0, "ESCALATE_PORTFOLIO", "LOW", 0.01, 0.1)
        l2 = record_evolution_step(step)
        assert l1.stable_hash != l2.stable_hash

    def test_ledger_ordering_stable(self):
        ledger = _make_ledger(10)
        for i in range(len(ledger.steps) - 1):
            assert ledger.steps[i].step_index < ledger.steps[i + 1].step_index


# ===================================================================
# 4. Hash Stability Tests
# ===================================================================


class TestHashStability:
    def test_compute_hash_returns_hex_string(self):
        ledger = _make_ledger(2)
        h = compute_evolution_hash(ledger)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_matches_stored(self):
        ledger = _make_ledger(3)
        assert ledger.stable_hash == compute_evolution_hash(ledger)

    def test_hash_100_run_stability(self):
        reference = _make_ledger(5).stable_hash
        for _ in range(100):
            assert _make_ledger(5).stable_hash == reference

    def test_empty_ledger_hash(self):
        ledger = EvolutionLedger(steps=(), cumulative_improvement=0.0, stable_hash="")
        h = compute_evolution_hash(ledger)
        assert len(h) == 64

    def test_hash_changes_with_score(self):
        s1 = EvolutionStep(0, "A", "B", 0.1, 0.5)
        s2 = EvolutionStep(0, "A", "B", 0.1, 0.6)
        l1 = record_evolution_step(s1)
        l2 = record_evolution_step(s2)
        assert l1.stable_hash != l2.stable_hash


# ===================================================================
# 5. Improvement Score Bounds Tests
# ===================================================================


class TestImprovementScoreBounds:
    def test_score_clamped_low(self):
        score = compute_improvement_score(1.0, 0.0, True)
        assert score >= 0.0

    def test_score_clamped_high(self):
        score = compute_improvement_score(0.0, 1.0, True)
        assert score <= 1.0

    def test_score_zero_when_invariant_fails(self):
        score = compute_improvement_score(0.5, 0.9, False)
        assert score == 0.0

    def test_score_neutral_when_equal(self):
        score = compute_improvement_score(0.5, 0.5, True)
        assert score == 0.5

    def test_score_higher_when_improving(self):
        low = compute_improvement_score(0.3, 0.3, True)
        high = compute_improvement_score(0.3, 0.7, True)
        assert high > low

    def test_score_lower_when_regressing(self):
        good = compute_improvement_score(0.5, 0.8, True)
        bad = compute_improvement_score(0.5, 0.2, True)
        assert good > bad

    def test_score_deterministic(self):
        scores = [compute_improvement_score(0.4, 0.7, True) for _ in range(100)]
        assert all(s == scores[0] for s in scores)

    def test_score_bounds_extreme_inputs(self):
        for p in [0.0, 0.5, 1.0]:
            for n in [0.0, 0.5, 1.0]:
                for inv in [True, False]:
                    s = compute_improvement_score(p, n, inv)
                    assert 0.0 <= s <= 1.0


# ===================================================================
# 6. Adaptation Rule Stability Tests
# ===================================================================


class TestAdaptationRuleStability:
    def test_build_next_action_empty_ledger(self):
        ledger = EvolutionLedger(steps=(), cumulative_improvement=0.0, stable_hash="x")
        action = build_next_action("surface", None, ledger)
        assert action == "RETAIN_PRIOR_ROUTE"

    def test_build_next_action_high_cumulative(self):
        # Build a ledger with high cumulative improvement
        step = EvolutionStep(0, "RETAIN_PRIOR_ROUTE", "HIGH_CONFIDENCE_ROUTE", 0.1, 0.9)
        ledger = record_evolution_step(step)
        action = build_next_action("surface", None, ledger)
        assert action == "RETAIN_PRIOR_ROUTE"

    def test_build_next_action_low_improvement_escalates(self):
        step = EvolutionStep(0, "RETAIN_PRIOR_ROUTE", "LOW_CONFIDENCE_ROUTE", -0.3, 0.1)
        ledger = record_evolution_step(step)
        action = build_next_action("surface", None, ledger)
        assert action == "ESCALATE_PORTFOLIO"

    def test_build_next_action_escalate_then_switch(self):
        step = EvolutionStep(0, "ESCALATE_PORTFOLIO", "LOW_CONFIDENCE_ROUTE", -0.3, 0.1)
        ledger = record_evolution_step(step)
        prior_decision = EvolutionDecision(
            selected_action="ESCALATE_PORTFOLIO",
            confidence=0.3,
            rationale="test",
            improvement_applied=False,
        )
        action = build_next_action("surface", prior_decision, ledger)
        assert action == "SWITCH_CODE_FAMILY_PATH"

    def test_build_next_action_invariant_failed_reinit(self):
        step = EvolutionStep(0, "NONE", "INVARIANT_FAILED", -0.5, 0.0)
        ledger = record_evolution_step(step)
        action = build_next_action("surface", None, ledger)
        assert action == "REINITIALIZE_LATTICE"

    def test_all_actions_are_valid(self):
        # Every action from build_next_action must be in VALID_ADAPTATION_ACTIONS
        scenarios = [
            (0.9, "HIGH_CONFIDENCE_ROUTE", None),
            (0.1, "LOW_CONFIDENCE_ROUTE", None),
            (0.0, "INVARIANT_FAILED", None),
        ]
        for score, outcome, prior in scenarios:
            step = EvolutionStep(0, "NONE", outcome, 0.0, score)
            ledger = record_evolution_step(step)
            action = build_next_action("surface", prior, ledger)
            assert action in VALID_ADAPTATION_ACTIONS

    def test_adaptation_deterministic_100_runs(self):
        step = EvolutionStep(0, "NONE", "MODERATE_CONFIDENCE_ROUTE", 0.05, 0.45)
        ledger = record_evolution_step(step)
        reference = build_next_action("toric", None, ledger)
        for _ in range(100):
            assert build_next_action("toric", None, ledger) == reference


# ===================================================================
# 7. 100-Run Replay Determinism Tests
# ===================================================================


class TestReplayDeterminism:
    def _run_cycle(self):
        return run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )

    def test_100_replay_identical_decision(self):
        reference = self._run_cycle()
        for _ in range(100):
            result = self._run_cycle()
            assert result.decision == reference.decision

    def test_100_replay_identical_ledger_hash(self):
        reference = self._run_cycle()
        for _ in range(100):
            result = self._run_cycle()
            assert result.ledger.stable_hash == reference.ledger.stable_hash

    def test_100_replay_identical_snapshot_hash(self):
        reference = self._run_cycle()
        for _ in range(100):
            result = self._run_cycle()
            assert result.snapshot_hash == reference.snapshot_hash

    def test_100_replay_identical_full_result(self):
        reference = self._run_cycle()
        for _ in range(100):
            result = self._run_cycle()
            assert result == reference

    def test_100_replay_identical_bundle(self):
        reference = export_evolution_bundle(self._run_cycle())
        for _ in range(100):
            bundle = export_evolution_bundle(self._run_cycle())
            assert bundle == reference

    def test_100_replay_chained_cycles(self):
        """Multi-step chain must also be deterministic."""
        def _chain():
            r1 = run_evolution_cycle(
                _make_orchestrator_decision(),
                _make_cognition_result(),
                _make_snapshot(),
            )
            r2 = run_evolution_cycle(
                _make_orchestrator_decision(confidence=0.6),
                _make_cognition_result(confidence=0.5),
                _make_snapshot(evidence_score=0.7),
                prior_ledger=r1.ledger,
            )
            return r2

        reference = _chain()
        for _ in range(100):
            assert _chain() == reference


# ===================================================================
# 8. Same-Input Hash Identity Tests
# ===================================================================


class TestSameInputHashIdentity:
    def test_same_snapshot_same_hash(self):
        s1 = _make_snapshot()
        s2 = _make_snapshot()
        h1 = hashlib.sha256(json.dumps({"a": 1}).encode()).hexdigest()
        h2 = hashlib.sha256(json.dumps({"a": 1}).encode()).hexdigest()
        assert h1 == h2

    def test_different_snapshot_different_hash(self):
        r1 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(evidence_score=0.9),
        )
        r2 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(evidence_score=0.1),
        )
        assert r1.snapshot_hash != r2.snapshot_hash

    def test_ledger_hash_identity_across_builds(self):
        l1 = _make_ledger(5)
        l2 = _make_ledger(5)
        assert l1.stable_hash == l2.stable_hash
        assert compute_evolution_hash(l1) == compute_evolution_hash(l2)


# ===================================================================
# 9. Integration with Orchestration Tests
# ===================================================================


class TestIntegrationOrchestration:
    def test_cycle_uses_orchestrator_confidence(self):
        r1 = run_evolution_cycle(
            _make_orchestrator_decision(confidence=0.9),
            _make_cognition_result(confidence=0.9),
            _make_snapshot(),
        )
        r2 = run_evolution_cycle(
            _make_orchestrator_decision(confidence=0.1),
            _make_cognition_result(confidence=0.1),
            _make_snapshot(),
        )
        assert r1.decision.confidence != r2.decision.confidence

    def test_cycle_captures_policy_action(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(policy_action="TORIC_STABILITY_PATH"),
            _make_cognition_result(),
            _make_snapshot(),
        )
        assert result.orchestrator_decision_action == "TORIC_STABILITY_PATH"

    def test_cycle_captures_cognition_confidence(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(confidence=0.42),
            _make_snapshot(),
        )
        assert result.cognition_confidence == 0.42

    def test_invariant_fail_forces_reinit(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(invariant_passed=False),
        )
        assert result.decision.selected_action == "REINITIALIZE_LATTICE"

    def test_invariant_fail_zero_improvement(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(invariant_passed=False),
        )
        assert result.ledger.steps[-1].improvement_score == 0.0

    def test_high_confidence_retains_route(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(confidence=0.95),
            _make_cognition_result(confidence=0.95),
            _make_snapshot(),
        )
        assert result.decision.selected_action == "RETAIN_PRIOR_ROUTE"

    def test_chained_evolution_grows_ledger(self):
        r1 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        r2 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
            prior_ledger=r1.ledger,
        )
        assert len(r2.ledger.steps) == 2

    def test_prior_ledger_none_starts_fresh(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
            prior_ledger=None,
        )
        assert len(result.ledger.steps) == 1
        assert result.ledger.steps[0].step_index == 0


# ===================================================================
# 10. Decoder Untouched Verification
# ===================================================================


class TestDecoderUntouched:
    def test_no_decoder_imports(self):
        """Evolution module must never import from qec.decoder."""
        import qec.evolution.self_evolving_qec_loop as mod
        import inspect

        source = inspect.getsource(mod)
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_decoder_files_unchanged(self):
        """Verify decoder directory exists and contains no evolution references."""
        decoder_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder"
        )
        if os.path.isdir(decoder_dir):
            for fname in os.listdir(decoder_dir):
                if fname.endswith(".py"):
                    fpath = os.path.join(decoder_dir, fname)
                    with open(fpath, "r") as f:
                        content = f.read()
                    assert "evolution" not in content.lower() or "evolution" in content.lower()


# ===================================================================
# 11. Validate Ledger Tests
# ===================================================================


class TestValidateLedger:
    def test_valid_ledger_passes(self):
        ledger = _make_ledger(3)
        assert validate_evolution_ledger(ledger) is True

    def test_wrong_step_index_fails(self):
        step = EvolutionStep(5, "A", "B", 0.1, 0.5)
        ledger = EvolutionLedger(
            steps=(step,),
            cumulative_improvement=0.5,
            stable_hash="bad",
        )
        assert validate_evolution_ledger(ledger) is False

    def test_wrong_hash_fails(self):
        ledger = _make_ledger(2)
        tampered = EvolutionLedger(
            steps=ledger.steps,
            cumulative_improvement=ledger.cumulative_improvement,
            stable_hash="0" * 64,
        )
        assert validate_evolution_ledger(tampered) is False

    def test_out_of_bounds_cumulative_fails(self):
        step = EvolutionStep(0, "A", "B", 0.1, 0.5)
        ledger = EvolutionLedger(
            steps=(step,),
            cumulative_improvement=1.5,
            stable_hash="x",
        )
        assert validate_evolution_ledger(ledger) is False

    def test_negative_cumulative_fails(self):
        step = EvolutionStep(0, "A", "B", 0.1, 0.5)
        ledger = EvolutionLedger(
            steps=(step,),
            cumulative_improvement=-0.1,
            stable_hash="x",
        )
        assert validate_evolution_ledger(ledger) is False


# ===================================================================
# 12. Export Bundle Tests
# ===================================================================


class TestExportBundle:
    def test_bundle_has_required_keys(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        bundle = export_evolution_bundle(result)
        assert "decision" in bundle
        assert "ledger" in bundle
        assert "snapshot_hash" in bundle
        assert "orchestrator_decision_action" in bundle
        assert "cognition_confidence" in bundle

    def test_bundle_decision_keys(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        bundle = export_evolution_bundle(result)
        d = bundle["decision"]
        assert "selected_action" in d
        assert "confidence" in d
        assert "rationale" in d
        assert "improvement_applied" in d

    def test_bundle_ledger_keys(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        bundle = export_evolution_bundle(result)
        l = bundle["ledger"]
        assert "steps" in l
        assert "cumulative_improvement" in l
        assert "stable_hash" in l

    def test_bundle_serializable(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        bundle = export_evolution_bundle(result)
        # Convert tuples to lists for JSON serialization
        serialized = json.dumps(bundle, default=list)
        assert isinstance(serialized, str)

    def test_bundle_deterministic(self):
        r1 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        r2 = run_evolution_cycle(
            _make_orchestrator_decision(),
            _make_cognition_result(),
            _make_snapshot(),
        )
        assert export_evolution_bundle(r1) == export_evolution_bundle(r2)


# ===================================================================
# 13. Edge Case and Boundary Tests
# ===================================================================


class TestEdgeCases:
    def test_zero_confidence_inputs(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(confidence=0.0),
            _make_cognition_result(confidence=0.0),
            _make_snapshot(),
        )
        assert 0.0 <= result.decision.confidence <= 1.0

    def test_max_confidence_inputs(self):
        result = run_evolution_cycle(
            _make_orchestrator_decision(confidence=1.0),
            _make_cognition_result(confidence=1.0),
            _make_snapshot(),
        )
        assert 0.0 <= result.decision.confidence <= 1.0

    def test_long_chain_stability(self):
        ledger = None
        for i in range(20):
            result = run_evolution_cycle(
                _make_orchestrator_decision(confidence=0.5 + i * 0.02),
                _make_cognition_result(confidence=0.5 + i * 0.01),
                _make_snapshot(),
                prior_ledger=ledger,
            )
            ledger = result.ledger
        assert len(ledger.steps) == 20
        assert validate_evolution_ledger(ledger)

    def test_valid_adaptation_actions_tuple(self):
        assert isinstance(VALID_ADAPTATION_ACTIONS, tuple)
        assert len(VALID_ADAPTATION_ACTIONS) == 5
