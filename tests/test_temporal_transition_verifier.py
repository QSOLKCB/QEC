"""Tests for v132.0.0 temporal transition verifier."""

from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.temporal_transition_verifier import (
    TransitionHistory,
    run_temporal_transition_verifier,
    validate_transition_sequence,
)


class TestTransitionHistory:
    def test_frozen_immutability(self):
        history = TransitionHistory(transitions=("normal", "recovery"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            history.transitions = ("normal",)


class TestValidateTransitionSequence:
    def test_valid_legal_sequence(self):
        history = TransitionHistory(
            transitions=("normal", "recovery", "normal", "recovery", "safe_mode", "safe_mode")
        )
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is True
        assert result["illegal_transition_detected"] is False
        assert result["safe_mode_violation"] is False
        assert result["oscillation_detected"] is False
        assert result["verification_score"] == 0.0
        assert result["verification_label"] == "safe"

    def test_illegal_transition(self):
        history = TransitionHistory(transitions=("normal", "safe_mode"))
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is False
        assert result["illegal_transition_detected"] is True
        assert result["safe_mode_violation"] is False
        assert result["oscillation_detected"] is False
        assert result["verification_score"] == 1.0
        assert result["verification_label"] == "critical"

    def test_safe_mode_absorbing_legality(self):
        history = TransitionHistory(transitions=("recovery", "safe_mode", "safe_mode", "safe_mode"))
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is True
        assert result["illegal_transition_detected"] is False
        assert result["safe_mode_violation"] is False
        assert result["verification_score"] == 0.0
        assert result["verification_label"] == "safe"

    def test_safe_mode_violation(self):
        history = TransitionHistory(transitions=("recovery", "safe_mode", "recovery"))
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is False
        assert result["illegal_transition_detected"] is True
        assert result["safe_mode_violation"] is True
        assert result["verification_score"] == 1.0
        assert result["verification_label"] == "critical"

    def test_escalation_lock_absorbing_legality(self):
        history = TransitionHistory(
            transitions=("normal", "recovery", "escalation_lock", "escalation_lock")
        )
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is True
        assert result["illegal_transition_detected"] is False
        assert result["safe_mode_violation"] is False
        assert result["oscillation_detected"] is False
        assert result["verification_score"] == 0.0
        assert result["verification_label"] == "safe"

    def test_oscillation_detection(self):
        history = TransitionHistory(
            transitions=("normal", "recovery", "normal", "recovery", "normal")
        )
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is True
        assert result["illegal_transition_detected"] is False
        assert result["safe_mode_violation"] is False
        assert result["oscillation_detected"] is True
        assert result["verification_score"] == 0.5
        assert result["verification_label"] == "warning"

    def test_empty_history_behavior(self):
        history = TransitionHistory(transitions=())
        result = validate_transition_sequence(history)

        assert result["sequence_valid"] is True
        assert result["illegal_transition_detected"] is False
        assert result["safe_mode_violation"] is False
        assert result["oscillation_detected"] is False
        assert result["verification_score"] == 0.0
        assert result["verification_label"] == "safe"


class TestTemporalTransitionVerifier:
    def test_deterministic_repeatability(self):
        history = TransitionHistory(
            transitions=("normal", "recovery", "normal", "recovery", "normal")
        )

        def build_result() -> dict:
            return run_temporal_transition_verifier(history)

        assert build_result() == build_result()

    def test_exact_schema_stability(self):
        history = TransitionHistory(transitions=("normal", "recovery", "normal"))
        result = run_temporal_transition_verifier(history)

        assert tuple(result.keys()) == (
            "history",
            "verification",
            "temporal_ready",
        )
        assert result["history"] == history
        assert tuple(result["verification"].keys()) == (
            "sequence_valid",
            "illegal_transition_detected",
            "safe_mode_violation",
            "oscillation_detected",
            "verification_score",
            "verification_label",
        )
        assert result["temporal_ready"] is True
