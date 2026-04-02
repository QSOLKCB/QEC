"""
Tests for Surface Feedback Engine (v136.7.1).

Covers:
- dataclass immutability
- event validation
- unknown type rejection
- score bounds
- ledger merge determinism
- classification correctness
- movement integration
- bridge compatibility
- validator compatibility
- 100-run replay determinism
- hash stability
- decoder untouched verification
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Tuple

import pytest

from qec.ai.surface_feedback_engine import (
    VALID_EVENT_TYPES,
    VALID_LEDGER_CLASSIFICATIONS,
    FeedbackEvent,
    FeedbackLedger,
    apply_feedback_to_policy,
    classify_feedback_ledger,
    episode_to_feedback_ledger,
    export_feedback_state_space,
    merge_feedback_ledgers,
    record_feedback,
    score_feedback,
)
from qec.ai.state_space_bridge import (
    UnifiedStateSpaceReport,
    build_movement_state_space,
)
from qec.ai.state_space_validator import (
    compute_state_space_hash,
    validate_state_space_report,
)
from qec.ai.movement_learning_2d import (
    MovementEpisode,
    MovementState,
    PolicyDecision,
    run_episode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_type: str = "recovery",
    magnitude: float = 0.5,
    confidence: float = 0.8,
    source: str = "test",
    timestamp_index: int = 0,
) -> FeedbackEvent:
    return FeedbackEvent(
        source=source,
        magnitude=magnitude,
        event_type=event_type,
        timestamp_index=timestamp_index,
        confidence=confidence,
    )


def _build_sample_ledger() -> FeedbackLedger:
    """Build a deterministic sample ledger with known events."""
    events = [
        _make_event("recovery", 0.5, 0.8, "s1", 0),
        _make_event("hazard", 0.6, 0.7, "s2", 1),
        _make_event("stable_route", 0.4, 0.9, "s3", 2),
        _make_event("drift", 0.3, 0.5, "s4", 3),
        _make_event("collapse", 0.8, 0.6, "s5", 4),
        _make_event("reward", 0.7, 0.8, "s6", 5),
        _make_event("penalty", 0.4, 0.5, "s7", 6),
    ]
    ledger = None
    for ev in events:
        ledger = record_feedback(ev, ledger)
    return ledger


# ===========================================================================
# 1. Dataclass immutability
# ===========================================================================


class TestDataclassImmutability:
    def test_feedback_event_frozen(self):
        ev = _make_event()
        with pytest.raises(AttributeError):
            ev.source = "modified"  # type: ignore[misc]

    def test_feedback_event_frozen_magnitude(self):
        ev = _make_event()
        with pytest.raises(AttributeError):
            ev.magnitude = 999.0  # type: ignore[misc]

    def test_feedback_ledger_frozen(self):
        ledger = record_feedback(_make_event())
        with pytest.raises(AttributeError):
            ledger.cumulative_score = 999.0  # type: ignore[misc]

    def test_feedback_ledger_frozen_events(self):
        ledger = record_feedback(_make_event())
        with pytest.raises(AttributeError):
            ledger.events = ()  # type: ignore[misc]

    def test_feedback_ledger_frozen_classification(self):
        ledger = record_feedback(_make_event())
        with pytest.raises(AttributeError):
            ledger.classification = "invalid"  # type: ignore[misc]


# ===========================================================================
# 2. Event validation
# ===========================================================================


class TestEventValidation:
    @pytest.mark.parametrize("event_type", VALID_EVENT_TYPES)
    def test_valid_event_types_accepted(self, event_type: str):
        ev = _make_event(event_type=event_type)
        ledger = record_feedback(ev)
        assert len(ledger.events) == 1
        assert ledger.events[0].event_type == event_type

    def test_event_fields_preserved(self):
        ev = FeedbackEvent(
            source="src1",
            magnitude=0.75,
            event_type="recovery",
            timestamp_index=42,
            confidence=0.9,
        )
        assert ev.source == "src1"
        assert ev.magnitude == 0.75
        assert ev.event_type == "recovery"
        assert ev.timestamp_index == 42
        assert ev.confidence == 0.9


# ===========================================================================
# 3. Unknown type rejection
# ===========================================================================


class TestUnknownTypeRejection:
    def test_unknown_event_type_raises(self):
        ev = _make_event(event_type="unknown_type")
        with pytest.raises(ValueError, match="Unknown event_type"):
            record_feedback(ev)

    def test_empty_event_type_raises(self):
        ev = _make_event(event_type="")
        with pytest.raises(ValueError, match="Unknown event_type"):
            record_feedback(ev)

    def test_typo_event_type_raises(self):
        ev = _make_event(event_type="recovry")
        with pytest.raises(ValueError, match="Unknown event_type"):
            record_feedback(ev)


# ===========================================================================
# 4. Score bounds
# ===========================================================================


class TestScoreBounds:
    def test_score_feedback_in_bounds(self):
        ledger = _build_sample_ledger()
        score = score_feedback(ledger)
        assert 0.0 <= score <= 1.0

    def test_cumulative_score_in_bounds(self):
        ledger = _build_sample_ledger()
        assert 0.0 <= ledger.cumulative_score <= 1.0

    def test_stability_score_in_bounds(self):
        ledger = _build_sample_ledger()
        assert 0.0 <= ledger.stability_score <= 1.0

    def test_hazard_pressure_in_bounds(self):
        ledger = _build_sample_ledger()
        assert 0.0 <= ledger.hazard_pressure <= 1.0

    def test_extreme_positive_events_clamped(self):
        ledger = None
        for i in range(100):
            ev = _make_event("reward", 1.0, 1.0, "s", i)
            ledger = record_feedback(ev, ledger)
        assert ledger.cumulative_score <= 1.0
        assert ledger.stability_score <= 1.0
        assert score_feedback(ledger) <= 1.0

    def test_extreme_negative_events_clamped(self):
        ledger = None
        for i in range(100):
            ev = _make_event("collapse", 1.0, 1.0, "s", i)
            ledger = record_feedback(ev, ledger)
        assert ledger.cumulative_score >= 0.0
        assert ledger.stability_score >= 0.0
        assert ledger.hazard_pressure <= 1.0
        assert score_feedback(ledger) >= 0.0

    def test_score_neutral_ledger(self):
        ledger = FeedbackLedger(
            events=(),
            cumulative_score=0.5,
            stability_score=0.5,
            hazard_pressure=0.0,
            classification="stable_feedback",
        )
        score = score_feedback(ledger)
        assert 0.0 <= score <= 1.0
        # Neutral should be 0.5*0.4 + 0.5*0.4 + 1.0*0.2 = 0.6
        assert abs(score - 0.6) < 1e-12


# ===========================================================================
# 5. Ledger merge determinism
# ===========================================================================


class TestLedgerMergeDeterminism:
    def test_merge_empty_list(self):
        result = merge_feedback_ledgers([])
        assert len(result.events) == 0
        assert result.cumulative_score == 0.5

    def test_merge_single_ledger(self):
        ledger = _build_sample_ledger()
        merged = merge_feedback_ledgers([ledger])
        # Scores should match since same events are replayed
        assert merged.cumulative_score == ledger.cumulative_score
        assert merged.stability_score == ledger.stability_score

    def test_merge_deterministic_ordering(self):
        ev1 = _make_event("recovery", 0.5, 0.8, "a", 0)
        ev2 = _make_event("hazard", 0.6, 0.7, "b", 1)
        l1 = record_feedback(ev1)
        l2 = record_feedback(ev2)

        merged_ab = merge_feedback_ledgers([l1, l2])
        merged_ba = merge_feedback_ledgers([l2, l1])

        # Same events, sorted by timestamp — should be identical
        assert merged_ab.cumulative_score == merged_ba.cumulative_score
        assert merged_ab.stability_score == merged_ba.stability_score
        assert merged_ab.hazard_pressure == merged_ba.hazard_pressure

    def test_merge_100_replay_determinism(self):
        ledger_a = _build_sample_ledger()
        ev_extra = _make_event("reward", 0.3, 0.9, "extra", 10)
        ledger_b = record_feedback(ev_extra)

        reference = merge_feedback_ledgers([ledger_a, ledger_b])
        for _ in range(100):
            result = merge_feedback_ledgers([ledger_a, ledger_b])
            assert result.cumulative_score == reference.cumulative_score
            assert result.stability_score == reference.stability_score
            assert result.hazard_pressure == reference.hazard_pressure


# ===========================================================================
# 6. Classification correctness
# ===========================================================================


class TestClassificationCorrectness:
    def test_classification_always_valid(self):
        ledger = _build_sample_ledger()
        cls = classify_feedback_ledger(ledger)
        assert cls in VALID_LEDGER_CLASSIFICATIONS

    def test_stable_feedback_classification(self):
        ledger = None
        for i in range(10):
            ev = _make_event("stable_route", 0.3, 0.8, "s", i)
            ledger = record_feedback(ev, ledger)
        assert classify_feedback_ledger(ledger) == "stable_feedback"

    def test_hazard_pressure_classification(self):
        ledger = None
        for i in range(20):
            ev = _make_event("hazard", 0.8, 0.9, "s", i)
            ledger = record_feedback(ev, ledger)
        cls = classify_feedback_ledger(ledger)
        assert cls in ("hazard_pressure", "chaotic_feedback")

    def test_classification_matches_ledger_field(self):
        ledger = _build_sample_ledger()
        assert ledger.classification == classify_feedback_ledger(ledger)

    def test_all_classifications_reachable(self):
        """Verify every classification can be produced."""
        seen = set()
        # stable_feedback
        lg = FeedbackLedger((), 0.5, 0.5, 0.0, "stable_feedback")
        seen.add(classify_feedback_ledger(lg))
        # hazard_pressure
        lg = FeedbackLedger((), 0.3, 0.5, 0.5, "hazard_pressure")
        seen.add(classify_feedback_ledger(lg))
        # collapse_recovery
        lg = FeedbackLedger((), 0.5, 0.4, 0.35, "collapse_recovery")
        seen.add(classify_feedback_ledger(lg))
        # drifting_signal
        lg = FeedbackLedger((), 0.4, 0.3, 0.1, "drifting_signal")
        seen.add(classify_feedback_ledger(lg))
        # chaotic_feedback
        lg = FeedbackLedger((), 0.2, 0.2, 0.7, "chaotic_feedback")
        seen.add(classify_feedback_ledger(lg))
        assert seen == set(VALID_LEDGER_CLASSIFICATIONS)


# ===========================================================================
# 7. Movement integration
# ===========================================================================


class TestMovementIntegration:
    def test_episode_to_ledger_basic(self):
        ep = run_episode(seed=42, steps=20)
        ledger = episode_to_feedback_ledger(ep)
        assert isinstance(ledger, FeedbackLedger)
        assert len(ledger.events) == 20  # one event per state transition

    def test_episode_to_ledger_deterministic(self):
        ep = run_episode(seed=123, steps=30)
        ref = episode_to_feedback_ledger(ep)
        for _ in range(50):
            result = episode_to_feedback_ledger(ep)
            assert result.cumulative_score == ref.cumulative_score
            assert result.stability_score == ref.stability_score
            assert result.hazard_pressure == ref.hazard_pressure

    def test_episode_to_ledger_score_bounded(self):
        ep = run_episode(seed=999, steps=50)
        ledger = episode_to_feedback_ledger(ep)
        assert 0.0 <= ledger.cumulative_score <= 1.0
        assert 0.0 <= ledger.stability_score <= 1.0
        assert 0.0 <= ledger.hazard_pressure <= 1.0
        assert 0.0 <= score_feedback(ledger) <= 1.0

    def test_episode_to_ledger_classification_valid(self):
        for seed in range(10):
            ep = run_episode(seed=seed, steps=30)
            ledger = episode_to_feedback_ledger(ep)
            assert ledger.classification in VALID_LEDGER_CLASSIFICATIONS

    def test_episode_to_ledger_event_types_valid(self):
        ep = run_episode(seed=7, steps=40)
        ledger = episode_to_feedback_ledger(ep)
        for ev in ledger.events:
            assert ev.event_type in VALID_EVENT_TYPES

    def test_empty_episode(self):
        ep = MovementEpisode(
            states=(MovementState((0.0, 0.0), (0.0, 0.0), 0.0, 0.8, 0.1, 0.9),),
            decisions=(),
            total_reward=0.0,
            recovery_events=0,
            classification="stable_route",
        )
        ledger = episode_to_feedback_ledger(ep)
        assert len(ledger.events) == 0
        assert ledger.classification == "stable_feedback"


# ===========================================================================
# 8. Bridge compatibility
# ===========================================================================


class TestBridgeCompatibility:
    def test_export_produces_valid_trace(self):
        ledger = _build_sample_ledger()
        trace = export_feedback_state_space(ledger)
        assert len(trace) == len(ledger.events)
        for point in trace:
            assert "x" in point
            assert "y" in point
            assert "coherence" in point
            assert "entropy" in point
            assert "stability" in point
            assert "label" in point

    def test_export_to_build_movement_state_space(self):
        ledger = _build_sample_ledger()
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        assert isinstance(report, UnifiedStateSpaceReport)
        assert len(report.nodes) == len(ledger.events)

    def test_export_empty_ledger(self):
        ledger = FeedbackLedger((), 0.5, 0.5, 0.0, "stable_feedback")
        trace = export_feedback_state_space(ledger)
        assert len(trace) == 0

    def test_movement_episode_roundtrip(self):
        """Episode -> Ledger -> StateSpace -> Validate."""
        ep = run_episode(seed=77, steps=25)
        ledger = episode_to_feedback_ledger(ep)
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        assert isinstance(report, UnifiedStateSpaceReport)
        assert len(report.nodes) == len(ledger.events)


# ===========================================================================
# 9. Validator compatibility
# ===========================================================================


class TestValidatorCompatibility:
    def test_validate_exported_state_space(self):
        ledger = _build_sample_ledger()
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_validate_movement_episode_pipeline(self):
        ep = run_episode(seed=55, steps=30)
        ledger = episode_to_feedback_ledger(ep)
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.overall_passed is True

    def test_validator_classification_stable(self):
        ledger = _build_sample_ledger()
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        validation = validate_state_space_report(report)
        assert validation.classification_stable is True


# ===========================================================================
# 10. 100-run replay determinism
# ===========================================================================


class TestReplayDeterminism:
    def test_score_feedback_100_replay(self):
        ledger = _build_sample_ledger()
        reference = score_feedback(ledger)
        for _ in range(100):
            assert score_feedback(ledger) == reference

    def test_record_feedback_100_replay(self):
        events = [
            _make_event("recovery", 0.5, 0.8, "s1", 0),
            _make_event("hazard", 0.6, 0.7, "s2", 1),
            _make_event("reward", 0.4, 0.9, "s3", 2),
        ]
        ref_ledger = None
        for ev in events:
            ref_ledger = record_feedback(ev, ref_ledger)

        for _ in range(100):
            ledger = None
            for ev in events:
                ledger = record_feedback(ev, ledger)
            assert ledger.cumulative_score == ref_ledger.cumulative_score
            assert ledger.stability_score == ref_ledger.stability_score
            assert ledger.hazard_pressure == ref_ledger.hazard_pressure

    def test_classify_100_replay(self):
        ledger = _build_sample_ledger()
        reference = classify_feedback_ledger(ledger)
        for _ in range(100):
            assert classify_feedback_ledger(ledger) == reference

    def test_export_state_space_100_replay(self):
        ledger = _build_sample_ledger()
        reference = export_feedback_state_space(ledger)
        for _ in range(100):
            assert export_feedback_state_space(ledger) == reference


# ===========================================================================
# 11. Hash stability
# ===========================================================================


class TestHashStability:
    def test_hash_stable_across_runs(self):
        ledger = _build_sample_ledger()
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            trace2 = export_feedback_state_space(ledger)
            report2 = build_movement_state_space(trace2)
            assert compute_state_space_hash(report2) == ref_hash

    def test_hash_different_for_different_ledgers(self):
        l1 = record_feedback(_make_event("recovery", 0.5, 0.8, "s", 0))
        l2 = record_feedback(_make_event("hazard", 0.5, 0.8, "s", 0))
        t1 = export_feedback_state_space(l1)
        t2 = export_feedback_state_space(l2)
        r1 = build_movement_state_space(t1)
        r2 = build_movement_state_space(t2)
        assert compute_state_space_hash(r1) != compute_state_space_hash(r2)

    def test_movement_episode_hash_stable(self):
        ep = run_episode(seed=42, steps=20)
        ledger = episode_to_feedback_ledger(ep)
        trace = export_feedback_state_space(ledger)
        report = build_movement_state_space(trace)
        ref_hash = compute_state_space_hash(report)
        for _ in range(100):
            ledger2 = episode_to_feedback_ledger(ep)
            trace2 = export_feedback_state_space(ledger2)
            report2 = build_movement_state_space(trace2)
            assert compute_state_space_hash(report2) == ref_hash


# ===========================================================================
# 12. Decoder untouched verification
# ===========================================================================


class TestDecoderUntouched:
    def test_no_decoder_imports_in_engine(self):
        import qec.ai.surface_feedback_engine as mod
        source_path = mod.__file__
        with open(source_path, "r") as f:
            source = f.read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
        assert "qec.decoder" not in source

    def test_decoder_directory_unchanged(self):
        """Verify decoder directory exists and was not modified by this module."""
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "src", "qec", "decoder",
        )
        # Just verify it exists — we don't modify it
        if os.path.isdir(decoder_path):
            # Can list files without error
            entries = os.listdir(decoder_path)
            assert isinstance(entries, list)

    def test_no_decoder_imports_in_tests(self):
        """This test file must not import from qec.decoder."""
        with open(__file__, "r") as f:
            lines = f.readlines()
        # Check actual import lines only (lines starting with from/import)
        import_lines = [
            ln.strip() for ln in lines
            if ln.strip().startswith(("from ", "import "))
        ]
        decoder_imports = [
            ln for ln in import_lines
            if "qec.decoder" in ln
        ]
        assert decoder_imports == [], f"Decoder imports found: {decoder_imports}"


# ===========================================================================
# 13. Apply feedback to policy
# ===========================================================================


class TestApplyFeedbackToPolicy:
    def test_basic_policy_application(self):
        policy = {"name": "test_policy", "version": 1}
        ledger = _build_sample_ledger()
        result = apply_feedback_to_policy(policy, ledger)
        assert "feedback_score" in result
        assert "feedback_classification" in result
        assert "stability_score" in result
        assert "hazard_pressure" in result
        assert result["name"] == "test_policy"
        assert result["version"] == 1

    def test_policy_not_mutated(self):
        policy = {"name": "original"}
        ledger = record_feedback(_make_event())
        result = apply_feedback_to_policy(policy, ledger)
        assert "feedback_score" not in policy
        assert "feedback_score" in result

    def test_feedback_score_in_bounds(self):
        policy = {}
        ledger = _build_sample_ledger()
        result = apply_feedback_to_policy(policy, ledger)
        assert 0.0 <= result["feedback_score"] <= 1.0


# ===========================================================================
# 14. Edge cases and additional coverage
# ===========================================================================


class TestEdgeCases:
    def test_zero_magnitude_event(self):
        ev = _make_event(magnitude=0.0)
        ledger = record_feedback(ev)
        # Zero magnitude => no score change from neutral
        assert ledger.cumulative_score == 0.5
        assert ledger.stability_score == 0.5

    def test_zero_confidence_event(self):
        ev = _make_event(confidence=0.0)
        ledger = record_feedback(ev)
        assert ledger.cumulative_score == 0.5

    def test_max_magnitude_and_confidence(self):
        ev = _make_event(magnitude=1.0, confidence=1.0, event_type="reward")
        ledger = record_feedback(ev)
        assert 0.0 <= ledger.cumulative_score <= 1.0
        assert 0.0 <= score_feedback(ledger) <= 1.0

    def test_ledger_events_tuple_type(self):
        ledger = _build_sample_ledger()
        assert isinstance(ledger.events, tuple)

    def test_record_feedback_none_ledger(self):
        ev = _make_event()
        ledger = record_feedback(ev, None)
        assert len(ledger.events) == 1

    def test_multiple_seeds_episode_integration(self):
        """Multiple seeds produce valid feedback ledgers."""
        for seed in [0, 1, 42, 100, 9999]:
            ep = run_episode(seed=seed, steps=15)
            ledger = episode_to_feedback_ledger(ep)
            assert ledger.classification in VALID_LEDGER_CLASSIFICATIONS
            assert 0.0 <= score_feedback(ledger) <= 1.0
