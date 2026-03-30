"""v113.1.0 — Deterministic ternary lattice controller over finite-state control."""

from __future__ import annotations

from typing import Any, Sequence

from qec.analysis.finite_state_controller import (
    CONTROL_LOOP_STABLE,
    STATE_IDLE,
    STATE_INTERVENTION,
    STATE_REBALANCE,
    STATE_STABILIZATION,
    run_finite_state_controller,
)
from qec.analysis.minimal_chain_experiments import DIFFUSION_STEPS

LATTICE_CLASS_STABLE = "stable_lattice"
LATTICE_CLASS_ADAPTIVE = "adaptive_lattice"
LATTICE_CLASS_INTERVENTION = "intervention_lattice"

HIGH_STABILITY_MIN = 0.8


def run_ternary_lattice_controller(
    chain_length: int = 9,
    chain_state: Sequence[float] | None = None,
    controller_cycles: int = 3,
    lattice_cycles: int = 3,
    chain_lengths: Sequence[int] | None = None,
    threshold_values: Sequence[float] | None = None,
    perturbation_values: Sequence[float] | None = None,
    diffusion_steps: int = DIFFUSION_STEPS,
    lattice_boundary_mode: str = "fixed",
    return_lattice_trace: bool = False,
) -> dict[str, Any]:
    """Run deterministic ternary lattice control cycles over finite-state controller outputs."""
    resolved_chain_length = max(1, int(chain_length))
    resolved_controller_cycles = int(controller_cycles)
    resolved_lattice_boundary_mode = str(lattice_boundary_mode)

    if resolved_controller_cycles <= 0:
        controller_result = {
            "chain_length": resolved_chain_length,
            "field_result": {},
            "controller_state": STATE_IDLE,
            "state_transition_count": 0,
            "controller_stability_score": 1.0,
            "control_loop_class": CONTROL_LOOP_STABLE,
        }
    else:
        controller_result = run_finite_state_controller(
            chain_length=chain_length,
            chain_state=chain_state,
            controller_cycles=controller_cycles,
            chain_lengths=chain_lengths,
            threshold_values=threshold_values,
            perturbation_values=perturbation_values,
            diffusion_steps=diffusion_steps,
        )

    resolved_chain_length = int(controller_result["chain_length"])
    controller_state = str(controller_result["controller_state"])
    resolved_lattice_cycles = max(0, int(lattice_cycles))

    lattice_state = _map_controller_state_to_lattice(
        controller_state=controller_state,
        chain_length=resolved_chain_length,
    )

    transition_count = 0
    lattice_trace: list[tuple[int, ...]] = [lattice_state]

    for _ in range(resolved_lattice_cycles):
        next_state = _evolve_lattice_state(
            lattice_state,
            lattice_boundary_mode=resolved_lattice_boundary_mode,
        )
        transition_count += _count_transitions(lattice_state, next_state)
        lattice_state = next_state
        lattice_trace.append(lattice_state)

    stability_score = _lattice_stability_score(
        transition_count=transition_count,
        chain_length=resolved_chain_length,
        lattice_cycles=resolved_lattice_cycles,
    )

    result: dict[str, Any] = {
        "chain_length": resolved_chain_length,
        "controller_result": controller_result,
        "ternary_lattice_state": lattice_state,
        "lattice_transition_count": int(transition_count),
        "lattice_stability_score": stability_score,
        "lattice_control_class": _lattice_control_class(
            controller_state=controller_state,
            lattice_stability_score=stability_score,
        ),
    }
    if return_lattice_trace:
        result["lattice_trace"] = lattice_trace

    return result


def _map_controller_state_to_lattice(*, controller_state: str, chain_length: int) -> tuple[int, ...]:
    resolved_chain_length = max(1, int(chain_length))
    lattice = [0] * resolved_chain_length

    if controller_state == STATE_IDLE:
        return tuple(lattice)

    if controller_state == STATE_STABILIZATION:
        center = resolved_chain_length // 2
        for index in (center - 1, center, center + 1):
            if 0 <= index < resolved_chain_length:
                lattice[index] = 1
        return tuple(lattice)

    if controller_state == STATE_REBALANCE:
        midpoint = (resolved_chain_length - 1) / 2.0
        for index in range(resolved_chain_length):
            if float(index) < midpoint:
                lattice[index] = -1
            elif float(index) > midpoint:
                lattice[index] = 1
            else:
                lattice[index] = 0
        return tuple(lattice)

    if controller_state == STATE_INTERVENTION:
        lattice[0] = -1
        lattice[-1] = 1
        return tuple(lattice)

    raise KeyError(f"unknown controller_state for lattice initialization: {controller_state}")


def _evolve_lattice_state(
    lattice_state: tuple[int, ...],
    *,
    lattice_boundary_mode: str,
) -> tuple[int, ...]:
    evolved: list[int] = []
    length = len(lattice_state)

    for index in range(length):
        neighborhood = [
            lattice_state[index],
            _neighbor_value(
                lattice_state=lattice_state,
                index=index - 1,
                lattice_boundary_mode=lattice_boundary_mode,
            ),
            _neighbor_value(
                lattice_state=lattice_state,
                index=index + 1,
                lattice_boundary_mode=lattice_boundary_mode,
            ),
        ]
        mean_value = float(sum(neighborhood)) / float(len(neighborhood))
        evolved.append(_sign_to_ternary(mean_value))

    return tuple(evolved)


def _neighbor_value(*, lattice_state: tuple[int, ...], index: int, lattice_boundary_mode: str) -> int:
    length = len(lattice_state)
    if 0 <= index < length:
        return int(lattice_state[index])

    if lattice_boundary_mode == "fixed":
        return 0

    if lattice_boundary_mode == "reflective":
        if index < 0:
            return int(lattice_state[0])
        return int(lattice_state[-1])

    if lattice_boundary_mode == "periodic":
        return int(lattice_state[index % length])

    raise ValueError("lattice_boundary_mode must be one of: fixed, reflective, periodic")


def _count_transitions(previous_state: tuple[int, ...], next_state: tuple[int, ...]) -> int:
    if len(previous_state) != len(next_state):
        raise ValueError("lattice states must have equal length")
    return sum(1 for previous, current in zip(previous_state, next_state) if previous != current)


def _lattice_stability_score(*, transition_count: int, chain_length: int, lattice_cycles: int) -> float:
    if int(lattice_cycles) <= 0:
        return 1.0
    denominator = int(chain_length) * int(lattice_cycles)
    changed_fraction = float(transition_count) / float(denominator)
    return _clamp01(1.0 - changed_fraction)


def _lattice_control_class(*, controller_state: str, lattice_stability_score: float) -> str:
    if controller_state == STATE_INTERVENTION:
        return LATTICE_CLASS_INTERVENTION
    if lattice_stability_score >= HIGH_STABILITY_MIN:
        return LATTICE_CLASS_STABLE
    return LATTICE_CLASS_ADAPTIVE


def _sign_to_ternary(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = ["run_ternary_lattice_controller"]
