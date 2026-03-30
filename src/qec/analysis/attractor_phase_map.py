"""v114.0.0 — Deterministic attractor phase-map diagnostics for ternary lattice control."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS
from qec.analysis.ternary_lattice_controller import (
    BOUNDARY_MODE_FIXED,
    run_ternary_lattice_controller,
)

ATTRACTOR_STATE_FIXED_POINT = "fixed_point"
ATTRACTOR_STATE_PERIOD_TWO = "period_two"
ATTRACTOR_STATE_DRIFTING_PHASE = "drifting_phase"
ATTRACTOR_STATE_INTERVENTION_PHASE = "intervention_phase"

PHASE_CLASS_STABLE = "stable_phase"
PHASE_CLASS_OSCILLATORY = "oscillatory_phase"
PHASE_CLASS_DRIFTING = "drifting_phase"
PHASE_CLASS_CRITICAL = "critical_phase"

CONTROLLER_STATE_INTERVENTION = "intervention_state"
RECENT_STATE_WINDOW = 4
STABILITY_WINDOW_TRANSITIONS = 3


def run_attractor_phase_map(
    chain_length: int = 9,
    chain_state: Sequence[float] | None = None,
    controller_cycles: int = 3,
    lattice_cycles: int = 6,
    lattice_boundary_mode: Literal["fixed", "reflective", "periodic"] = BOUNDARY_MODE_FIXED,
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    return_phase_trace: bool = False,
) -> dict[str, Any]:
    """Run deterministic attractor/phase diagnostics over ternary lattice traces."""
    lattice_result = run_ternary_lattice_controller(
        chain_length=chain_length,
        chain_state=chain_state,
        controller_cycles=controller_cycles,
        lattice_cycles=lattice_cycles,
        chain_lengths=chain_lengths,
        threshold_values=threshold_values,
        perturbation_values=perturbation_values,
        diffusion_steps=diffusion_steps,
        lattice_boundary_mode=lattice_boundary_mode,
        return_lattice_trace=True,
    )

    lattice_trace = [tuple(state) for state in lattice_result.get("lattice_trace", ())]
    attractor_state, attractor_cycle_length = _detect_attractor(
        lattice_trace=lattice_trace,
        controller_state=str(lattice_result["controller_result"]["controller_state"]),
    )
    phase_stability_score = _phase_stability_score(lattice_trace)

    result: dict[str, Any] = {
        "chain_length": int(lattice_result["chain_length"]),
        "lattice_result": lattice_result,
        "attractor_state": attractor_state,
        "attractor_cycle_length": int(attractor_cycle_length),
        "phase_stability_score": phase_stability_score,
        "phase_class": _phase_class(attractor_state),
    }

    if return_phase_trace:
        result["phase_trace"] = lattice_trace

    return result


def _detect_attractor(lattice_trace: list[tuple[int, ...]], controller_state: str) -> tuple[str, int]:
    if controller_state == CONTROLLER_STATE_INTERVENTION:
        return ATTRACTOR_STATE_INTERVENTION_PHASE, 0

    if _is_fixed_point(lattice_trace):
        return ATTRACTOR_STATE_FIXED_POINT, 1

    if _is_period_two(lattice_trace):
        return ATTRACTOR_STATE_PERIOD_TWO, 2

    return ATTRACTOR_STATE_DRIFTING_PHASE, 0


def _is_fixed_point(lattice_trace: list[tuple[int, ...]]) -> bool:
    if len(lattice_trace) < 2:
        return False
    recent = lattice_trace[-min(RECENT_STATE_WINDOW, len(lattice_trace)) :]
    baseline = recent[-1]
    return all(state == baseline for state in recent[:-1])


def _is_period_two(lattice_trace: list[tuple[int, ...]]) -> bool:
    if len(lattice_trace) < 4:
        return False
    recent = lattice_trace[-min(RECENT_STATE_WINDOW, len(lattice_trace)) :]
    a = recent[-1]
    b = recent[-2]
    if a == b:
        return False
    for index, state in enumerate(recent):
        expected = a if ((len(recent) - 1 - index) % 2 == 0) else b
        if state != expected:
            return False
    return True


def _phase_stability_score(lattice_trace: list[tuple[int, ...]]) -> float:
    if len(lattice_trace) <= 1:
        return 1.0

    transition_count = min(STABILITY_WINDOW_TRANSITIONS, len(lattice_trace) - 1)
    previous_states = lattice_trace[-(transition_count + 1) : -1]
    next_states = lattice_trace[-transition_count:]
    state_lengths = [len(state) for state in previous_states + next_states]
    if len(set(state_lengths)) != 1:
        raise ValueError("lattice trace states must have equal length")
    state_length = state_lengths[0]
    total_positions = transition_count * state_length
    if total_positions <= 0:
        return 1.0

    changed_positions = 0
    for previous_state, next_state in zip(previous_states, next_states):
        changed_positions += sum(
            1 for previous_value, next_value in zip(previous_state, next_state) if previous_value != next_value
        )

    changed_fraction = float(changed_positions) / float(total_positions)
    return _clamp01(1.0 - changed_fraction)


def _phase_class(attractor_state: str) -> str:
    if attractor_state == ATTRACTOR_STATE_FIXED_POINT:
        return PHASE_CLASS_STABLE
    if attractor_state == ATTRACTOR_STATE_PERIOD_TWO:
        return PHASE_CLASS_OSCILLATORY
    if attractor_state == ATTRACTOR_STATE_INTERVENTION_PHASE:
        return PHASE_CLASS_CRITICAL
    return PHASE_CLASS_DRIFTING


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_attractor_phase_map"]
