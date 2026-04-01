"""v112.1.0 — Deterministic finite-state control layer over correction fields."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.cellular_correction_field import run_cellular_correction_field
from qec.analysis.correction_dispatch import (
    ACTION_BOUNDARY_INTERVENE,
    ACTION_HOLD_STATE,
    ACTION_LOCAL_STABILIZE,
    ACTION_SPECTRAL_REBALANCE,
)
from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS

STATE_IDLE = "idle_state"
STATE_STABILIZATION = "stabilization_state"
STATE_REBALANCE = "rebalance_state"
STATE_INTERVENTION = "intervention_state"

CONTROL_LOOP_STABLE = "stable_loop"
CONTROL_LOOP_ADAPTIVE = "adaptive_loop"
CONTROL_LOOP_INTERVENTION = "intervention_loop"

HIGH_STABILITY_MIN = 0.8
DRIFT_RECOVERY_MAX = 0.05
LOCAL_STABILITY_RECOVERY_MIN = 0.95

CONTROLLER_STATE_BY_ACTION = {
    ACTION_HOLD_STATE: STATE_IDLE,
    ACTION_LOCAL_STABILIZE: STATE_STABILIZATION,
    ACTION_SPECTRAL_REBALANCE: STATE_REBALANCE,
    ACTION_BOUNDARY_INTERVENE: STATE_INTERVENTION,
}


def run_finite_state_controller(
    chain_length: int = 9,
    chain_state: Sequence[float] | None = None,
    automata_steps: int = 3,
    controller_cycles: int = 3,
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    return_controller_trace: bool = False,
) -> dict[str, Any]:
    """Run deterministic finite-state control cycles over cellular correction field outputs."""
    resolved_controller_cycles = max(0, int(controller_cycles))
    transition_count = 0
    trace: list[str] = []

    current_chain_length = int(chain_length)
    current_chain_state = chain_state
    latest_field_result: dict[str, Any] | None = None

    for cycle_index in range(resolved_controller_cycles):
        field_result = run_cellular_correction_field(
            chain_length=current_chain_length,
            chain_state=current_chain_state,
            chain_lengths=chain_lengths,
            threshold_values=threshold_values,
            perturbation_values=perturbation_values,
            diffusion_steps=diffusion_steps,
            automata_steps=automata_steps,
            return_trace=True,
        )
        previous_state = trace[-1] if trace else None
        controller_state = _infer_controller_state(
            field_result,
            previous_state=previous_state,
        )

        if cycle_index > 0 and controller_state != trace[-1]:
            transition_count += 1
        trace.append(controller_state)

        current_chain_length = int(field_result["chain_length"])
        current_chain_state = tuple(float(value) for value in field_result["corrected_field"])
        latest_field_result = field_result

    if latest_field_result is None:
        raise RuntimeError("latest_field_result must not be None after controller execution")

    stability_score = _controller_stability_score(
        transition_count=transition_count,
        controller_cycles=resolved_controller_cycles,
    )
    final_state = trace[-1]

    result: dict[str, Any] = {
        "chain_length": int(latest_field_result["chain_length"]),
        "field_result": latest_field_result,
        "controller_state": final_state,
        "state_transition_count": int(transition_count),
        "controller_stability_score": stability_score,
        "control_loop_class": _control_loop_class(
            controller_state=final_state,
            controller_trace=trace,
            controller_stability_score=stability_score,
        ),
    }
    if return_controller_trace:
        result["controller_trace"] = trace

    return result


def _infer_controller_state(
    field_result: dict[str, Any],
    *,
    previous_state: str | None,
) -> str:
    dispatch_result = field_result.get("dispatch_result", {})
    correction_action = str(dispatch_result.get("correction_action", ACTION_HOLD_STATE))

    field_drift_score = float(field_result.get("field_drift_score", 0.0))
    local_stability_score = float(field_result.get("local_stability_score", 1.0))

    if (
        previous_state is not None
        and previous_state != STATE_IDLE
        and correction_action == ACTION_HOLD_STATE
        and field_drift_score <= DRIFT_RECOVERY_MAX
        and local_stability_score >= LOCAL_STABILITY_RECOVERY_MIN
    ):
        return STATE_IDLE

    if correction_action not in CONTROLLER_STATE_BY_ACTION:
        raise KeyError(f"unknown correction_action for controller state: {correction_action}")
    return CONTROLLER_STATE_BY_ACTION[correction_action]


def _controller_stability_score(*, transition_count: int, controller_cycles: int) -> float:
    denominator = max(1, int(controller_cycles) - 1)
    score = 1.0 - (float(transition_count) / float(denominator))
    return _clamp01(score)


def _control_loop_class(
    *,
    controller_state: str,
    controller_trace: list[str],
    controller_stability_score: float,
) -> str:
    if STATE_INTERVENTION in controller_trace or controller_state == STATE_INTERVENTION:
        return CONTROL_LOOP_INTERVENTION
    if controller_state == STATE_IDLE and controller_stability_score >= HIGH_STABILITY_MIN:
        return CONTROL_LOOP_STABLE
    return CONTROL_LOOP_ADAPTIVE


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_finite_state_controller"]
