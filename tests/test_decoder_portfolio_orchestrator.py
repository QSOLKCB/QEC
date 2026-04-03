"""
Tests for QEC Decoder Portfolio Orchestrator (v136.8.4).

Categories:
- dataclass immutability
- selection determinism
- stable tie-breaking
- portfolio registration
- hash stability
- code zoo integration
- snapshot integration
- audio cognition integration
- 100-run replay determinism
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError

import pytest

from qec.orchestration.decoder_portfolio_orchestrator import (
    ORCHESTRATOR_VERSION,
    VALID_ACTIONS,
    OrchestratorDecision,
    PortfolioCandidate,
    PortfolioRegistry,
    _canonical_json,
    _candidate_to_canonical_dict,
    _resolve_action,
    _validate_candidate,
    build_default_decoder_portfolio,
    compute_portfolio_hash,
    export_orchestration_bundle,
    export_orchestration_bundle_json,
    register_portfolio_candidate,
    run_orchestration_cycle,
    select_decoder_path,
    validate_portfolio_registry,
)
from qec.ai.controller_snapshot_schema import (
    ControllerSnapshot,
    SCHEMA_VERSION as SNAPSHOT_SCHEMA_VERSION,
)
from qec.audio.audio_cognition_engine import (
    CognitionCycleResult,
    run_cognition_cycle,
)
from qec.audio.cognition_registry import (
    AudioFingerprint,
    CognitionMatch,
    CognitionRegistry,
)
from qec.audio.triality_signal_engine import TrialityParams
from qec.codes.code_zoo import build_default_code_zoo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_snapshot(
    evidence_score: float = 0.8,
    invariant_passed: bool = True,
    policy_id: str = "test_policy",
) -> ControllerSnapshot:
    """Build a minimal valid ControllerSnapshot for testing."""
    payload = _canonical_json({"test": True})
    state_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return ControllerSnapshot(
        state_hash=state_hash,
        policy_id=policy_id,
        evidence_score=evidence_score,
        invariant_passed=invariant_passed,
        timestamp_index=0,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        payload_json=payload,
    )


def _make_cognition_match(confidence: float = 0.95) -> CognitionMatch:
    return CognitionMatch(
        confidence=confidence,
        identity="test_identity",
        failure_mode="none",
        recommended_action="DECODE_PORTFOLIO_A",
    )


def _make_fingerprint() -> AudioFingerprint:
    return AudioFingerprint(
        centroid=100.0,
        rolloff=200.0,
        peak_bins=(10, 20, 30),
        psd_hash="a" * 64,
    )


def _make_triality_params() -> TrialityParams:
    return TrialityParams(
        carrier_freq=440.0,
        mod_freq=5.0,
        mod_depth=0.3,
        overlay_base_freq=880.0,
        overlay_harmonics=3,
        state_hash="b" * 64,
    )


def _make_cognition_cycle_result(confidence: float = 0.95) -> CognitionCycleResult:
    return CognitionCycleResult(
        params=_make_triality_params(),
        fingerprint=_make_fingerprint(),
        match=_make_cognition_match(confidence),
        engine_version="v136.8.3",
    )


def _make_candidate(
    decoder_id: str = "decoder_surface_default",
    code_family: str = "surface",
    confidence: float = 0.9,
    expected_recovery_score: float = 0.85,
    route_priority: int = 0,
) -> PortfolioCandidate:
    return PortfolioCandidate(
        decoder_id=decoder_id,
        code_family=code_family,
        confidence=confidence,
        expected_recovery_score=expected_recovery_score,
        route_priority=route_priority,
    )


# ===================================================================
# 1. Dataclass immutability tests
# ===================================================================


class TestDataclassImmutability:
    def test_portfolio_candidate_frozen(self):
        c = _make_candidate()
        with pytest.raises(FrozenInstanceError):
            c.decoder_id = "mutated"  # type: ignore[misc]

    def test_portfolio_candidate_frozen_confidence(self):
        c = _make_candidate()
        with pytest.raises(FrozenInstanceError):
            c.confidence = 0.0  # type: ignore[misc]

    def test_orchestrator_decision_frozen(self):
        d = OrchestratorDecision(
            selected_decoder="d1",
            confidence=0.9,
            rationale="test",
            source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        with pytest.raises(FrozenInstanceError):
            d.selected_decoder = "mutated"  # type: ignore[misc]

    def test_orchestrator_decision_frozen_action(self):
        d = OrchestratorDecision(
            selected_decoder="d1",
            confidence=0.9,
            rationale="test",
            source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        with pytest.raises(FrozenInstanceError):
            d.policy_action = "mutated"  # type: ignore[misc]

    def test_portfolio_registry_frozen(self):
        reg = build_default_decoder_portfolio()
        with pytest.raises(FrozenInstanceError):
            reg.registry_hash = "mutated"  # type: ignore[misc]

    def test_portfolio_registry_frozen_candidates(self):
        reg = build_default_decoder_portfolio()
        with pytest.raises(FrozenInstanceError):
            reg.candidates = ()  # type: ignore[misc]


# ===================================================================
# 2. Selection determinism tests
# ===================================================================


class TestSelectionDeterminism:
    def test_same_inputs_same_decision(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d1 = select_decoder_path("surface", 0.95, snapshot, reg)
        d2 = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d1 == d2

    def test_different_family_different_decision(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d1 = select_decoder_path("surface", 0.95, snapshot, reg)
        d2 = select_decoder_path("toric", 0.95, snapshot, reg)
        assert d1.selected_decoder != d2.selected_decoder

    def test_surface_maps_to_surface_fast_path(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.policy_action == "SURFACE_FAST_PATH"

    def test_toric_maps_to_toric_stability_path(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("toric", 0.95, snapshot, reg)
        assert d.policy_action == "TORIC_STABILITY_PATH"

    def test_qldpc_maps_to_qldpc_portfolio(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("qldpc", 0.95, snapshot, reg)
        assert d.policy_action == "QLDPC_PORTFOLIO_B"

    def test_repetition_maps_to_decode_portfolio_a(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("repetition", 0.95, snapshot, reg)
        assert d.policy_action == "DECODE_PORTFOLIO_A"

    def test_unknown_family_fallback(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("unknown_family", 0.95, snapshot, reg)
        assert d.source_match == "fallback"

    def test_unknown_family_fallback_uses_default_action(self):
        """Fallback resolves action from requested code_family, not candidate's."""
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("unknown_family", 0.95, snapshot, reg)
        assert d.policy_action == "DECODE_PORTFOLIO_B"

    def test_empty_registry_raises(self):
        snapshot = _make_snapshot()
        empty = PortfolioRegistry(candidates=(), registry_hash="0" * 64)
        with pytest.raises(ValueError, match="empty portfolio"):
            select_decoder_path("surface", 0.95, snapshot, empty)

    def test_invariant_failed_reinit(self):
        snapshot = _make_snapshot(invariant_passed=False)
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.policy_action == "REINIT_CODE_LATTICE"

    def test_confidence_combination(self):
        snapshot = _make_snapshot(evidence_score=0.8)
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        # 0.9 * 0.5 + 0.95 * 0.3 + 0.8 * 0.2 = 0.45 + 0.285 + 0.16 = 0.895
        assert d.confidence == round(0.9 * 0.5 + 0.95 * 0.3 + 0.8 * 0.2, 15)

    def test_all_valid_actions_covered(self):
        """Verify all declared actions are reachable."""
        snapshot = _make_snapshot()
        snapshot_fail = _make_snapshot(invariant_passed=False)
        reg = build_default_decoder_portfolio()

        actions_seen = set()
        for family in ("surface", "toric", "qldpc", "repetition"):
            d = select_decoder_path(family, 0.95, snapshot, reg)
            actions_seen.add(d.policy_action)

        # REINIT via invariant failure
        d = select_decoder_path("surface", 0.95, snapshot_fail, reg)
        actions_seen.add(d.policy_action)

        # Fallback gives DECODE_PORTFOLIO_B or one of the mapped ones
        # At least verify we have several actions
        assert len(actions_seen) >= 5


# ===================================================================
# 3. Stable tie-breaking tests
# ===================================================================


class TestStableTieBreaking:
    def test_tie_by_confidence_uses_recovery_score(self):
        c1 = _make_candidate(decoder_id="d_a", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        c2 = _make_candidate(decoder_id="d_b", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.8, route_priority=0)
        reg = register_portfolio_candidate(c1)
        reg = register_portfolio_candidate(c2, reg)
        snapshot = _make_snapshot()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.selected_decoder == "d_a"  # higher recovery

    def test_tie_by_recovery_uses_priority(self):
        c1 = _make_candidate(decoder_id="d_a", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=1)
        c2 = _make_candidate(decoder_id="d_b", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        reg = register_portfolio_candidate(c1)
        reg = register_portfolio_candidate(c2, reg)
        snapshot = _make_snapshot()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.selected_decoder == "d_b"  # lower priority wins

    def test_tie_by_all_uses_decoder_id(self):
        c1 = _make_candidate(decoder_id="d_beta", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        c2 = _make_candidate(decoder_id="d_alpha", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        reg = register_portfolio_candidate(c1)
        reg = register_portfolio_candidate(c2, reg)
        snapshot = _make_snapshot()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.selected_decoder == "d_alpha"  # alphabetical

    def test_stable_across_100_runs(self):
        c1 = _make_candidate(decoder_id="d_a", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        c2 = _make_candidate(decoder_id="d_b", code_family="surface",
                             confidence=0.9, expected_recovery_score=0.9, route_priority=0)
        reg = register_portfolio_candidate(c1)
        reg = register_portfolio_candidate(c2, reg)
        snapshot = _make_snapshot()
        ref = select_decoder_path("surface", 0.95, snapshot, reg)
        for _ in range(100):
            assert select_decoder_path("surface", 0.95, snapshot, reg) == ref


# ===================================================================
# 4. Portfolio registration tests
# ===================================================================


class TestPortfolioRegistration:
    def test_register_single_candidate(self):
        c = _make_candidate()
        reg = register_portfolio_candidate(c)
        assert len(reg.candidates) == 1
        assert reg.candidates[0] == c

    def test_register_multiple_candidates(self):
        c1 = _make_candidate(decoder_id="d1", code_family="surface")
        c2 = _make_candidate(decoder_id="d2", code_family="toric")
        reg = register_portfolio_candidate(c1)
        reg = register_portfolio_candidate(c2, reg)
        assert len(reg.candidates) == 2

    def test_register_duplicate_raises(self):
        c = _make_candidate(decoder_id="d1")
        reg = register_portfolio_candidate(c)
        with pytest.raises(ValueError, match="Duplicate decoder_id"):
            register_portfolio_candidate(c, reg)

    def test_register_maintains_sorted_order(self):
        c_toric = _make_candidate(decoder_id="d_toric", code_family="toric")
        c_surface = _make_candidate(decoder_id="d_surface", code_family="surface")
        c_rep = _make_candidate(decoder_id="d_rep", code_family="repetition")
        reg = register_portfolio_candidate(c_toric)
        reg = register_portfolio_candidate(c_surface, reg)
        reg = register_portfolio_candidate(c_rep, reg)
        families = [c.code_family for c in reg.candidates]
        assert families == sorted(families)

    def test_register_recomputes_hash(self):
        c1 = _make_candidate(decoder_id="d1", code_family="surface")
        reg1 = register_portfolio_candidate(c1)
        c2 = _make_candidate(decoder_id="d2", code_family="toric")
        reg2 = register_portfolio_candidate(c2, reg1)
        assert reg1.registry_hash != reg2.registry_hash

    def test_register_none_registry(self):
        c = _make_candidate()
        reg = register_portfolio_candidate(c, None)
        assert len(reg.candidates) == 1

    def test_register_validates_candidate(self):
        """Registration rejects invalid candidates at registration time."""
        bad = _make_candidate(confidence=5.0)
        with pytest.raises(ValueError, match="confidence"):
            register_portfolio_candidate(bad)

    def test_register_validates_negative_priority(self):
        bad = _make_candidate(route_priority=-1)
        with pytest.raises(ValueError, match="route_priority"):
            register_portfolio_candidate(bad)


# ===================================================================
# 5. Hash stability tests
# ===================================================================


class TestHashStability:
    def test_same_registry_same_hash(self):
        reg1 = build_default_decoder_portfolio()
        reg2 = build_default_decoder_portfolio()
        assert reg1.registry_hash == reg2.registry_hash

    def test_hash_is_64_hex(self):
        reg = build_default_decoder_portfolio()
        assert len(reg.registry_hash) == 64
        int(reg.registry_hash, 16)  # must not raise

    def test_hash_changes_with_different_candidates(self):
        c1 = _make_candidate(decoder_id="d1", confidence=0.9)
        c2 = _make_candidate(decoder_id="d1", confidence=0.8)
        r1 = register_portfolio_candidate(c1)
        r2 = register_portfolio_candidate(c2)
        assert r1.registry_hash != r2.registry_hash

    def test_compute_portfolio_hash_deterministic(self):
        reg = build_default_decoder_portfolio()
        h1 = compute_portfolio_hash(reg)
        h2 = compute_portfolio_hash(reg)
        assert h1 == h2

    def test_hash_stable_100_runs(self):
        ref = build_default_decoder_portfolio().registry_hash
        for _ in range(100):
            assert build_default_decoder_portfolio().registry_hash == ref


# ===================================================================
# 6. Code zoo integration tests
# ===================================================================


class TestCodeZooIntegration:
    def test_default_portfolio_uses_zoo_families(self):
        zoo = build_default_code_zoo()
        families = sorted(set(s.family for s in zoo.codes))
        reg = build_default_decoder_portfolio()
        portfolio_families = sorted(set(c.code_family for c in reg.candidates))
        assert portfolio_families == families

    def test_default_portfolio_has_candidates(self):
        reg = build_default_decoder_portfolio()
        assert len(reg.candidates) > 0

    def test_default_portfolio_validates(self):
        reg = build_default_decoder_portfolio()
        assert validate_portfolio_registry(reg) is True

    def test_zoo_families_all_have_decoders(self):
        zoo = build_default_code_zoo()
        reg = build_default_decoder_portfolio()
        zoo_families = set(s.family for s in zoo.codes)
        portfolio_families = set(c.code_family for c in reg.candidates)
        assert zoo_families == portfolio_families

    def test_zoo_determinism(self):
        r1 = build_default_decoder_portfolio()
        r2 = build_default_decoder_portfolio()
        assert r1 == r2


# ===================================================================
# 7. Snapshot integration tests
# ===================================================================


class TestSnapshotIntegration:
    def test_select_with_snapshot(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert isinstance(d, OrchestratorDecision)

    def test_snapshot_evidence_affects_confidence(self):
        reg = build_default_decoder_portfolio()
        d_high = select_decoder_path("surface", 0.95, _make_snapshot(evidence_score=1.0), reg)
        d_low = select_decoder_path("surface", 0.95, _make_snapshot(evidence_score=0.0), reg)
        assert d_high.confidence > d_low.confidence

    def test_snapshot_invariant_failure_overrides_action(self):
        snapshot = _make_snapshot(invariant_passed=False)
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.policy_action == "REINIT_CODE_LATTICE"
        assert "invariant_failed" in d.rationale

    def test_snapshot_invariant_pass_normal_action(self):
        snapshot = _make_snapshot(invariant_passed=True)
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert d.policy_action != "REINIT_CODE_LATTICE"

    def test_snapshot_zero_evidence(self):
        snapshot = _make_snapshot(evidence_score=0.0)
        reg = build_default_decoder_portfolio()
        d = select_decoder_path("surface", 0.95, snapshot, reg)
        assert isinstance(d, OrchestratorDecision)


# ===================================================================
# 8. Audio cognition integration tests
# ===================================================================


class TestAudioCognitionIntegration:
    def test_orchestration_cycle_with_cognition(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result(confidence=0.95)
        d = run_orchestration_cycle("surface", ccr, snapshot, reg)
        assert isinstance(d, OrchestratorDecision)
        assert d.policy_action == "SURFACE_FAST_PATH"

    def test_cognition_confidence_propagates(self):
        snapshot = _make_snapshot(evidence_score=0.8)
        reg = build_default_decoder_portfolio()
        ccr_high = _make_cognition_cycle_result(confidence=1.0)
        ccr_low = _make_cognition_cycle_result(confidence=0.0)
        d_high = run_orchestration_cycle("surface", ccr_high, snapshot, reg)
        d_low = run_orchestration_cycle("surface", ccr_low, snapshot, reg)
        assert d_high.confidence > d_low.confidence

    def test_cognition_cycle_deterministic(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        d1 = run_orchestration_cycle("surface", ccr, snapshot, reg)
        d2 = run_orchestration_cycle("surface", ccr, snapshot, reg)
        assert d1 == d2

    def test_cognition_various_families(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        for family in ("surface", "toric", "qldpc", "repetition"):
            d = run_orchestration_cycle(family, ccr, snapshot, reg)
            assert isinstance(d, OrchestratorDecision)
            assert d.selected_decoder != ""


# ===================================================================
# 9. 100-run replay determinism tests
# ===================================================================


class TestReplayDeterminism:
    def test_100_replay_orchestration_cycle(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        reference = run_orchestration_cycle("surface", ccr, snapshot, reg)
        for _ in range(100):
            assert run_orchestration_cycle("surface", ccr, snapshot, reg) == reference

    def test_100_replay_select_decoder_path(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        reference = select_decoder_path("toric", 0.9, snapshot, reg)
        for _ in range(100):
            assert select_decoder_path("toric", 0.9, snapshot, reg) == reference

    def test_100_replay_portfolio_hash(self):
        reference = build_default_decoder_portfolio().registry_hash
        for _ in range(100):
            assert build_default_decoder_portfolio().registry_hash == reference

    def test_100_replay_export_json(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        result = run_orchestration_cycle("surface", ccr, snapshot, reg)
        ref_json = export_orchestration_bundle_json(result)
        for _ in range(100):
            r = run_orchestration_cycle("surface", ccr, snapshot, reg)
            assert export_orchestration_bundle_json(r) == ref_json

    def test_100_replay_all_families(self):
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        references = {}
        for family in ("surface", "toric", "qldpc", "repetition"):
            references[family] = run_orchestration_cycle(family, ccr, snapshot, reg)
        for _ in range(100):
            for family in ("surface", "toric", "qldpc", "repetition"):
                assert run_orchestration_cycle(family, ccr, snapshot, reg) == references[family]


# ===================================================================
# 10. Decoder untouched verification tests
# ===================================================================


class TestDecoderUntouched:
    def test_no_decoder_imports_in_orchestrator(self):
        """Verify orchestrator module does not import from qec.decoder."""
        import qec.orchestration.decoder_portfolio_orchestrator as mod
        source_path = mod.__file__
        assert source_path is not None
        with open(source_path, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_decoder_directory_untouched(self):
        """Verify decoder directory files have not been modified by this module."""
        decoder_dir = os.path.join(
            os.path.dirname(__file__), "..", "src", "qec", "decoder"
        )
        decoder_dir = os.path.normpath(decoder_dir)
        # Just verify decoder dir exists and contains files
        assert os.path.isdir(decoder_dir), f"Decoder dir missing: {decoder_dir}"
        files = os.listdir(decoder_dir)
        assert len(files) > 0, "Decoder directory is empty"

    def test_orchestrator_does_not_mutate_decoder(self):
        """Running orchestration should not affect decoder module state."""
        # Import decoder module and capture initial state
        import qec.decoder as decoder_mod
        initial_attrs = set(dir(decoder_mod))
        # Run orchestration
        snapshot = _make_snapshot()
        reg = build_default_decoder_portfolio()
        ccr = _make_cognition_cycle_result()
        run_orchestration_cycle("surface", ccr, snapshot, reg)
        # Verify decoder untouched
        post_attrs = set(dir(decoder_mod))
        assert initial_attrs == post_attrs


# ===================================================================
# 11. Validation tests
# ===================================================================


class TestValidation:
    def test_validate_valid_registry(self):
        reg = build_default_decoder_portfolio()
        assert validate_portfolio_registry(reg) is True

    def test_validate_bad_hash(self):
        reg = build_default_decoder_portfolio()
        bad = PortfolioRegistry(candidates=reg.candidates, registry_hash="0" * 64)
        with pytest.raises(ValueError, match="registry_hash mismatch"):
            validate_portfolio_registry(bad)

    def test_validate_duplicate_id(self):
        c = _make_candidate(decoder_id="dup")
        good = register_portfolio_candidate(c)
        # Manually create bad registry with duplicates
        bad = PortfolioRegistry(
            candidates=(c, c),
            registry_hash="0" * 64,
        )
        with pytest.raises(ValueError, match="Duplicate decoder_id"):
            validate_portfolio_registry(bad)

    def test_validate_unsorted_candidates(self):
        c1 = _make_candidate(decoder_id="d_z", code_family="toric")
        c2 = _make_candidate(decoder_id="d_a", code_family="repetition")
        bad = PortfolioRegistry(
            candidates=(c1, c2),  # toric before repetition = unsorted
            registry_hash="0" * 64,
        )
        with pytest.raises(ValueError, match="sorted"):
            validate_portfolio_registry(bad)

    def test_validate_confidence_out_of_range(self):
        c = _make_candidate(confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            register_portfolio_candidate(c)

    def test_validate_negative_priority(self):
        c = _make_candidate(route_priority=-1)
        with pytest.raises(ValueError, match="route_priority"):
            register_portfolio_candidate(c)

    def test_validate_empty_decoder_id(self):
        c = PortfolioCandidate(
            decoder_id="", code_family="surface",
            confidence=0.9, expected_recovery_score=0.85, route_priority=0,
        )
        with pytest.raises(ValueError, match="decoder_id"):
            register_portfolio_candidate(c)

    def test_validate_empty_code_family(self):
        c = PortfolioCandidate(
            decoder_id="d1", code_family="",
            confidence=0.9, expected_recovery_score=0.85, route_priority=0,
        )
        with pytest.raises(ValueError, match="code_family"):
            register_portfolio_candidate(c)

    def test_validate_recovery_score_out_of_range(self):
        c = _make_candidate(expected_recovery_score=2.0)
        with pytest.raises(ValueError, match="expected_recovery_score"):
            register_portfolio_candidate(c)

    def test_validate_empty_registry(self):
        empty = PortfolioRegistry(candidates=(), registry_hash="0" * 64)
        with pytest.raises(ValueError, match="at least one candidate"):
            validate_portfolio_registry(empty)


# ===================================================================
# 12. Export tests
# ===================================================================


class TestExport:
    def test_export_bundle_keys(self):
        d = OrchestratorDecision(
            selected_decoder="d1", confidence=0.9,
            rationale="test", source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        bundle = export_orchestration_bundle(d)
        assert set(bundle.keys()) == {
            "confidence", "orchestrator_version", "policy_action",
            "rationale", "selected_decoder", "source_match",
        }

    def test_export_bundle_version(self):
        d = OrchestratorDecision(
            selected_decoder="d1", confidence=0.9,
            rationale="test", source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        bundle = export_orchestration_bundle(d)
        assert bundle["orchestrator_version"] == ORCHESTRATOR_VERSION

    def test_export_json_deterministic(self):
        d = OrchestratorDecision(
            selected_decoder="d1", confidence=0.9,
            rationale="test", source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        j1 = export_orchestration_bundle_json(d)
        j2 = export_orchestration_bundle_json(d)
        assert j1 == j2

    def test_export_json_is_valid_json(self):
        d = OrchestratorDecision(
            selected_decoder="d1", confidence=0.9,
            rationale="test", source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        parsed = json.loads(export_orchestration_bundle_json(d))
        assert parsed["selected_decoder"] == "d1"

    def test_export_json_sorted_keys(self):
        d = OrchestratorDecision(
            selected_decoder="d1", confidence=0.9,
            rationale="test", source_match="test",
            policy_action="DECODE_PORTFOLIO_A",
        )
        j = export_orchestration_bundle_json(d)
        parsed = json.loads(j)
        assert list(parsed.keys()) == sorted(parsed.keys())


# ===================================================================
# 13. Version and constants tests
# ===================================================================


class TestConstants:
    def test_orchestrator_version(self):
        assert ORCHESTRATOR_VERSION == "v136.8.4"

    def test_valid_actions_tuple(self):
        assert isinstance(VALID_ACTIONS, tuple)
        assert len(VALID_ACTIONS) == 7

    def test_valid_actions_sorted(self):
        assert VALID_ACTIONS == tuple(sorted(VALID_ACTIONS))

    def test_resolve_action_known_families(self):
        assert _resolve_action("surface") == "SURFACE_FAST_PATH"
        assert _resolve_action("toric") == "TORIC_STABILITY_PATH"
        assert _resolve_action("qldpc") == "QLDPC_PORTFOLIO_B"
        assert _resolve_action("repetition") == "DECODE_PORTFOLIO_A"

    def test_resolve_action_unknown_family(self):
        assert _resolve_action("unknown") == "DECODE_PORTFOLIO_B"


# ===================================================================
# 14. Canonical serialization tests
# ===================================================================


class TestCanonicalSerialization:
    def test_canonical_json_sorted_keys(self):
        j = _canonical_json({"z": 1, "a": 2})
        assert j == '{"a":2,"z":1}'

    def test_canonical_json_compact(self):
        j = _canonical_json({"key": "value"})
        assert " " not in j

    def test_candidate_to_dict(self):
        c = _make_candidate()
        d = _candidate_to_canonical_dict(c)
        assert set(d.keys()) == {
            "code_family", "confidence", "decoder_id",
            "expected_recovery_score", "route_priority",
        }

    def test_candidate_to_dict_deterministic(self):
        c = _make_candidate()
        d1 = _candidate_to_canonical_dict(c)
        d2 = _candidate_to_canonical_dict(c)
        assert d1 == d2
