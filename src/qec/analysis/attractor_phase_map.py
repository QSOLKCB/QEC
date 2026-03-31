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
    phase_transition_index = _phase_transition_index(lattice_trace)
    attractor_entry_cycle = _attractor_entry_cycle(lattice_trace, attractor_state)

    result: dict[str, Any] = {
        "chain_length": int(lattice_result["chain_length"]),
        "lattice_result": lattice_result,
        "attractor_state": attractor_state,
        "attractor_cycle_length": int(attractor_cycle_length),
        "phase_stability_score": phase_stability_score,
        "phase_class": _phase_class(attractor_state),
        "phase_transition_index": phase_transition_index,
        "attractor_entry_cycle": attractor_entry_cycle,
        "transition_sharpness_score": _transition_sharpness_score(lattice_trace, attractor_state, attractor_entry_cycle),
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


def _phase_transition_index(lattice_trace: list[tuple[int, ...]]) -> int:
    seen_states: dict[tuple[int, ...], int] = {}
    for index, state in enumerate(lattice_trace):
        if state in seen_states:
            return int(index)
        seen_states[state] = index
    return -1


def _attractor_entry_cycle(lattice_trace: list[tuple[int, ...]], attractor_state: str) -> int:
    if attractor_state == ATTRACTOR_STATE_FIXED_POINT:
        if not lattice_trace:
            return -1
        final_state = lattice_trace[-1]
        for index, state in enumerate(lattice_trace):
            if state == final_state and all(candidate == final_state for candidate in lattice_trace[index:]):
                return int(index)
        return -1

    if attractor_state == ATTRACTOR_STATE_PERIOD_TWO:
        if len(lattice_trace) < 2:
            return -1
        final_a = lattice_trace[-2]
        final_b = lattice_trace[-1]
        if final_a == final_b:
            return -1
        for start in range(len(lattice_trace) - 1):
            is_match = True
            for offset, state in enumerate(lattice_trace[start:]):
                expected = final_a if (offset % 2 == 0) else final_b
                if state != expected:
                    is_match = False
                    break
            if is_match:
                return int(start)
    return -1


def _transition_sharpness_score(
    lattice_trace: list[tuple[int, ...]],
    attractor_state: str,
    entry_cycle: int | None = None,
) -> float:
    if entry_cycle is None:
        entry_cycle = _attractor_entry_cycle(lattice_trace, attractor_state)
    if entry_cycle < 0:
        return 0.0

    changed_fraction_total = _changed_fraction(lattice_trace, 0, len(lattice_trace) - 1)
    if changed_fraction_total <= 0.0:
        return 1.0

    changed_fraction_before_entry = _changed_fraction(lattice_trace, 0, entry_cycle - 1)
    return _clamp01(1.0 - (changed_fraction_before_entry / changed_fraction_total))


def _changed_fraction(lattice_trace: list[tuple[int, ...]], start_cycle: int, end_cycle: int) -> float:
    if end_cycle <= start_cycle:
        return 0.0
    transitions = [
        (lattice_trace[index], lattice_trace[index + 1])
        for index in range(start_cycle, min(end_cycle, len(lattice_trace) - 1))
    ]
    if not transitions:
        return 0.0

    state_lengths = [len(state) for pair in transitions for state in pair]
    if len(set(state_lengths)) != 1:
        raise ValueError("lattice trace states must have equal length")
    state_length = state_lengths[0]
    total_positions = len(transitions) * state_length
    if total_positions <= 0:
        return 0.0

    changed_positions = 0
    for previous_state, next_state in transitions:
        changed_positions += sum(
            1 for previous_value, next_value in zip(previous_state, next_state) if previous_value != next_value
        )
    return float(changed_positions) / float(total_positions)


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
