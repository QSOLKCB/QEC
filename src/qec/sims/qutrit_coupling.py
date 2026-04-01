# SPDX-License-Identifier: MIT
"""Deterministic qutrit-field coupling law — v133.2.0.

Provides bidirectional coupling: qutrit channel states influence
field amplitude evolution via bounded, deterministic multipliers.

Coupling rule:
    state 0 -> neutral   (multiplier 1.000)
    state 1 -> amplify   (multiplier 1.001)
    state 2 -> damp      (multiplier 0.998)

All operations are replay-safe with byte-identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from qec.sims.universe_kernel import UniverseState

# Deterministic coupling multipliers — frozen mapping.
_COUPLING_MULTIPLIERS: Tuple[float, ...] = (1.000, 1.001, 0.998)
_VALID_QUTRIT_STATES = frozenset({0, 1, 2})


def _validate_qutrit_states(qutrit_states: Sequence[int]) -> None:
    """Fail fast if any qutrit state is not in {0, 1, 2}."""
    for q in qutrit_states:
        if q not in _VALID_QUTRIT_STATES:
            raise ValueError(
                "qutrit_states must contain only values {0, 1, 2}"
            )


def _lane_multipliers(
    num_lanes: int,
    qutrit_states: Sequence[int],
) -> Tuple[float, ...]:
    """Compute per-lane coupling multipliers with cyclic qutrit repeat.

    Single source of truth for coupling multiplier resolution.
    Validates qutrit states and maps them to multipliers.

    Parameters
    ----------
    num_lanes : int
        Number of field lanes.
    qutrit_states : Sequence[int]
        Qutrit channel states (0, 1, or 2).

    Returns
    -------
    Tuple[float, ...]
        Per-lane multipliers.
    """
    n_qutrits = len(qutrit_states)
    if num_lanes == 0 or n_qutrits == 0:
        return ()
    _validate_qutrit_states(qutrit_states)
    return tuple(
        _COUPLING_MULTIPLIERS[qutrit_states[i % n_qutrits]]
        for i in range(num_lanes)
    )


def apply_qutrit_coupling(
    field_amplitudes: Sequence[float],
    qutrit_states: Sequence[int],
) -> Tuple[float, ...]:
    """Apply qutrit coupling multipliers to field amplitudes.

    Each field lane is multiplied by the coupling factor determined
    by its corresponding qutrit state.  If qutrit_states is shorter
    than field_amplitudes, qutrit states are repeated cyclically.

    Parameters
    ----------
    field_amplitudes : Sequence[float]
        Current field values per lane.
    qutrit_states : Sequence[int]
        Qutrit channel states (0, 1, or 2).

    Returns
    -------
    Tuple[float, ...]
        New field amplitudes after coupling.

    Raises
    ------
    ValueError
        If any qutrit state is not in {0, 1, 2}.
    """
    n_fields = len(field_amplitudes)
    if n_fields == 0:
        return ()
    n_qutrits = len(qutrit_states)
    if n_qutrits == 0:
        return tuple(field_amplitudes)
    multipliers = _lane_multipliers(n_fields, qutrit_states)
    return tuple(
        field_amplitudes[i] * multipliers[i]
        for i in range(n_fields)
    )


def evolve_universe_coupled(state: UniverseState) -> UniverseState:
    """Evolve with decay then qutrit coupling.

    Steps:
        1. Apply field decay (0.999 multiplier per lane)
        2. Apply qutrit coupling multipliers
        3. Increment timestep

    This is additive to the base evolve_universe — it does not
    replace or modify the original evolution semantics.

    Parameters
    ----------
    state : UniverseState
        Current universe snapshot.

    Returns
    -------
    UniverseState
        New immutable state after coupled evolution step.
    """
    # Step 1: deterministic decay
    decayed = tuple(f * 0.999 for f in state.field_amplitudes)
    # Step 2: qutrit coupling
    coupled = apply_qutrit_coupling(decayed, state.qutrit_states)
    return UniverseState(
        field_amplitudes=coupled,
        qutrit_states=state.qutrit_states,
        timestep=state.timestep + 1,
        law_name=state.law_name,
    )


@dataclass(frozen=True)
class CouplingObservation:
    """Immutable observation of qutrit-field coupling state."""

    mean_coupling_gain: float
    amplified_lanes: int
    damped_lanes: int
    timestep: int


def observe_coupling(state: UniverseState) -> CouplingObservation:
    """Observe the coupling characteristics of a universe state.

    Computes per-lane coupling multipliers from the qutrit states
    and reports aggregate statistics.

    Parameters
    ----------
    state : UniverseState
        The universe state to observe.

    Returns
    -------
    CouplingObservation
        Frozen observation with coupling statistics.
    """
    n_fields = len(state.field_amplitudes)
    n_qutrits = len(state.qutrit_states)
    if n_fields == 0 or n_qutrits == 0:
        return CouplingObservation(
            mean_coupling_gain=1.0,
            amplified_lanes=0,
            damped_lanes=0,
            timestep=state.timestep,
        )
    multipliers = _lane_multipliers(n_fields, state.qutrit_states)
    mean_gain = sum(multipliers) / len(multipliers)
    amplified = sum(1 for m in multipliers if m > 1.0)
    damped = sum(1 for m in multipliers if m < 1.0)
    return CouplingObservation(
        mean_coupling_gain=mean_gain,
        amplified_lanes=amplified,
        damped_lanes=damped,
        timestep=state.timestep,
    )
