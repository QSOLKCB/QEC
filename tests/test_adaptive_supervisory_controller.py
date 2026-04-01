"""Tests for v131.0.0 adaptive supervisory controller."""

from __future__ import annotations

import dataclasses
import math

import pytest

from qec.analysis.adaptive_supervisory_controller import (
    MODE_ESCALATION_LOCK,
    MODE_NORMAL,
    MODE_RECOVERY,
    MODE_SAFE_MODE,
    SupervisorState,
    evaluate_supervisory_transition,
    normalize_supervisory_score,
    run_adaptive_supervisory_controller,
)


class TestSupervisorState:
    def test_frozen_immutability(self):
        state = SupervisorState(
            current_mode=MODE_NORMAL,
            recovery_attempts=0,
            safe_mode_latched=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.current_mode = MODE_RECOVERY


class TestNormalizeSupervisoryScore:
    def test_nan_inf_normalization(self):
        assert normalize_supervisory_score(math.nan) == 0.0
        assert normalize_supervisory_score(math.inf) == 1.0
        assert normalize_supervisory_score(-math.inf) == 0.0

    def test_finite_clamping(self):
        assert normalize_supervisory_score(-0.2) == 0.0
        assert normalize_supervisory_score(1.2) == 1.0
        assert normalize_supervisory_score(0.4) == pytest.approx(0.4)


class TestEvaluateSupervisoryTransition:
    def test_normal_remain(self):
        state = SupervisorState(MODE_NORMAL, recovery_attempts=0, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, False, False, 0.1) == MODE_NORMAL

    def test_fault_to_recovery(self):
        state = SupervisorState(MODE_NORMAL, recovery_attempts=0, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, False, 0.1) == MODE_RECOVERY

    def test_recovery_success_to_normal(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, True, 0.8) == MODE_NORMAL

    def test_recovery_fail_safe_precedence_over_success(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, True, 0.95) == MODE_SAFE_MODE

    def test_recovery_risk_to_safe_mode(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, False, 0.95) == MODE_SAFE_MODE

    def test_recovery_risk_boundary_below_threshold_remains_recovery(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, False, 0.94) == MODE_RECOVERY

    def test_recovery_risk_boundary_at_one_triggers_safe_mode(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, False, 1.0) == MODE_SAFE_MODE

    def test_recovery_attempts_to_escalation_lock(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=3, safe_mode_latched=False)
        assert (
            evaluate_supervisory_transition(state, True, False, 0.2)
            == MODE_ESCALATION_LOCK
        )

    def test_recovery_attempts_boundary_below_threshold_remains_recovery(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, True, False, 0.2) == MODE_RECOVERY

    def test_recovery_attempts_boundary_above_threshold_escalates(self):
        state = SupervisorState(MODE_RECOVERY, recovery_attempts=4, safe_mode_latched=False)
        assert (
            evaluate_supervisory_transition(state, True, False, 0.2)
            == MODE_ESCALATION_LOCK
        )

    def test_absorbing_safe_mode(self):
        state = SupervisorState(MODE_SAFE_MODE, recovery_attempts=1, safe_mode_latched=False)
        assert evaluate_supervisory_transition(state, False, True, 0.1) == MODE_SAFE_MODE

    def test_absorbing_escalation_lock(self):
        state = SupervisorState(
            MODE_ESCALATION_LOCK,
            recovery_attempts=1,
            safe_mode_latched=False,
        )
        assert (
            evaluate_supervisory_transition(state, True, True, 0.0)
            == MODE_ESCALATION_LOCK
        )


class TestAdaptiveSupervisoryController:
    def test_deterministic_repeatability(self):
        def build_result():
            state = SupervisorState(MODE_RECOVERY, recovery_attempts=2, safe_mode_latched=False)
            return run_adaptive_supervisory_controller(
                state=state,
                fault_detected=True,
                recovery_success=False,
                risk_score=0.95,
            )

        assert build_result() == build_result()

    def test_exact_schema_stability(self):
        state = SupervisorState(MODE_NORMAL, recovery_attempts=0, safe_mode_latched=False)
        result = run_adaptive_supervisory_controller(
            state=state,
            fault_detected=True,
            recovery_success=False,
            risk_score=0.2,
        )

        assert tuple(result.keys()) == (
            "previous_mode",
            "next_mode",
            "mode_transition",
            "fail_safe_triggered",
            "supervisory_ready",
        )
        assert result["previous_mode"] == MODE_NORMAL
        assert result["next_mode"] == MODE_RECOVERY
        assert result["mode_transition"] == "normal->recovery"
        assert result["fail_safe_triggered"] is False
        assert result["supervisory_ready"] is True
