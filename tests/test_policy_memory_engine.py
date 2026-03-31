"""Tests for v129.0.0 policy memory & escalation history engine."""

from __future__ import annotations

import dataclasses

import pytest

from qec.analysis.policy_memory_engine import (
    PolicyMemoryState,
    compute_memory_risk,
    run_policy_memory_engine,
    update_policy_memory,
)


class TestPolicyMemoryState:
    def test_frozen_immutability(self):
        state = PolicyMemoryState(history=("observe",), escalation_count=0, fail_safe_count=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.escalation_count = 1


class TestUpdatePolicyMemory:
    def test_bounded_history_eviction(self):
        state = PolicyMemoryState(history=("a", "b", "c"), escalation_count=0, fail_safe_count=0)
        updated = update_policy_memory(state, "d", max_history=3)
        assert updated.history == ("b", "c", "d")

    def test_escalation_counting(self):
        state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
        for label in ("observe", "stabilize", "explore", "intervene", "fail_safe"):
            state = update_policy_memory(state, label)
        assert state.escalation_count == 3

    def test_fail_safe_counting(self):
        state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
        for label in ("observe", "fail_safe", "intervene", "fail_safe"):
            state = update_policy_memory(state, label)
        assert state.fail_safe_count == 2

    def test_max_history_boundary(self):
        state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
        state = update_policy_memory(state, "a", max_history=1)
        state = update_policy_memory(state, "b", max_history=1)
        assert state.history == ("b",)

    def test_invalid_max_history_raises(self):
        state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
        with pytest.raises(ValueError):
            update_policy_memory(state, "observe", max_history=0)


class TestComputeMemoryRisk:
    def test_persistent_escalation_detection(self):
        state = PolicyMemoryState(
            history=("stabilize", "observe", "intervene"),
            escalation_count=3,
            fail_safe_count=0,
        )
        risk = compute_memory_risk(state)
        assert risk["persistent_escalation"] is True
        assert risk["memory_risk_score"] == 0.5

    def test_fail_safe_streak_detection(self):
        state = PolicyMemoryState(
            history=("observe", "fail_safe", "fail_safe"),
            escalation_count=2,
            fail_safe_count=2,
        )
        risk = compute_memory_risk(state)
        assert risk["fail_safe_streak"] is True
        assert risk["memory_risk_score"] == 1.0

    def test_safe_warning_critical_labels(self):
        safe = compute_memory_risk(
            PolicyMemoryState(history=("observe",), escalation_count=0, fail_safe_count=0)
        )
        warning = compute_memory_risk(
            PolicyMemoryState(history=("observe",), escalation_count=3, fail_safe_count=0)
        )
        critical = compute_memory_risk(
            PolicyMemoryState(
                history=("intervene", "fail_safe", "fail_safe"),
                escalation_count=3,
                fail_safe_count=2,
            )
        )

        assert safe["memory_risk_label"] == "safe"
        assert warning["memory_risk_label"] == "warning"
        assert critical["memory_risk_label"] == "critical"


class TestPolicyMemoryEngine:
    def test_deterministic_repeatability(self):
        def build_result():
            state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
            result = run_policy_memory_engine(state, "observe")
            result = run_policy_memory_engine(result["memory_state"], "intervene")
            result = run_policy_memory_engine(result["memory_state"], "fail_safe")
            return result

        assert build_result() == build_result()

    def test_schema_stability(self):
        state = PolicyMemoryState(history=(), escalation_count=0, fail_safe_count=0)
        result = run_policy_memory_engine(state, "observe")

        assert tuple(result.keys()) == ("memory_state", "memory_analysis", "memory_ready")
        assert isinstance(result["memory_state"], PolicyMemoryState)
        assert tuple(result["memory_analysis"].keys()) == (
            "persistent_escalation",
            "fail_safe_streak",
            "memory_risk_score",
            "memory_risk_label",
        )
        assert result["memory_ready"] is True
