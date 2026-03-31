"""v129.0.0 — Deterministic policy memory and escalation history engine.

This module provides a bounded temporal memory layer for policy orchestration.
All operations are deterministic, side-effect free, and schema-stable.
"""

from __future__ import annotations

from dataclasses import dataclass


ESCALATION_LABELS = ("stabilize", "intervene", "fail_safe")


@dataclass(frozen=True)
class PolicyMemoryState:
    """Immutable policy memory state.

    Attributes
    ----------
    history : tuple[str, ...]
        Rolling history of most recent policy labels in deterministic order.
    escalation_count : int
        Total count of escalation-class policy labels seen so far.
    fail_safe_count : int
        Total count of fail-safe labels seen so far.
    """

    history: tuple[str, ...]
    escalation_count: int
    fail_safe_count: int


def update_policy_memory(
    state: PolicyMemoryState,
    policy_label: str,
    max_history: int = 8,
) -> PolicyMemoryState:
    """Return updated immutable policy memory state.

    The newest ``policy_label`` is appended, history is bounded to
    ``max_history``, and escalation counters are incremented deterministically.
    """
    if max_history < 1:
        raise ValueError("max_history must be >= 1")

    history = tuple(state.history) + (policy_label,)
    if len(history) > max_history:
        history = history[-max_history:]

    is_escalation = policy_label in ESCALATION_LABELS
    is_fail_safe = policy_label == "fail_safe"

    return PolicyMemoryState(
        history=history,
        escalation_count=state.escalation_count + int(is_escalation),
        fail_safe_count=state.fail_safe_count + int(is_fail_safe),
    )


def compute_memory_risk(state: PolicyMemoryState) -> dict:
    """Compute deterministic memory risk analysis from state."""
    persistent_escalation = state.escalation_count >= 3
    fail_safe_streak = (
        len(state.history) >= 2
        and state.history[-1] == "fail_safe"
        and state.history[-2] == "fail_safe"
    )

    if fail_safe_streak:
        memory_risk_score = 1.0
        memory_risk_label = "critical"
    elif persistent_escalation:
        memory_risk_score = 0.5
        memory_risk_label = "warning"
    else:
        memory_risk_score = 0.0
        memory_risk_label = "safe"

    return {
        "persistent_escalation": persistent_escalation,
        "fail_safe_streak": fail_safe_streak,
        "memory_risk_score": memory_risk_score,
        "memory_risk_label": memory_risk_label,
    }


def run_policy_memory_engine(
    state: PolicyMemoryState,
    policy_label: str,
) -> dict:
    """Run one deterministic policy memory engine step."""
    memory_state = update_policy_memory(state, policy_label)
    memory_analysis = compute_memory_risk(memory_state)
    return {
        "memory_state": memory_state,
        "memory_analysis": memory_analysis,
        "memory_ready": True,
    }


__all__ = [
    "PolicyMemoryState",
    "compute_memory_risk",
    "run_policy_memory_engine",
    "update_policy_memory",
]
