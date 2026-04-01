# SPDX-License-Identifier: MIT
"""Comparative universe experiment framework — v133.4.0.

Runs multiple universes under different law sets from identical
initial conditions and compares divergence.

Supported configurations:
    - lawful: standard coupled evolution (decay + qutrit coupling)
    - anti-law: inverted qutrit semantics with amplifying decay

All operations are deterministic and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.universe_kernel import UniverseState
from qec.sims.observable_probe import observe_universe

# Standard coupled evolution — reuse existing law.
from qec.sims.qutrit_coupling import evolve_universe_coupled

# Anti-law qutrit multipliers: inverted semantics.
#   state 0 -> 1.000  (neutral)
#   state 1 -> 0.998  (damp, inverted from amplify)
#   state 2 -> 1.001  (amplify, inverted from damp)
_ANTI_COUPLING_MULTIPLIERS: Tuple[float, ...] = (1.000, 0.998, 1.001)

# Anti-law decay: amplifying instead of decaying.
_ANTI_DECAY: float = 1.001


def _apply_anti_coupling(
    field_amplitudes: Tuple[float, ...],
    qutrit_states: Tuple[int, ...],
) -> Tuple[float, ...]:
    """Apply anti-law qutrit coupling to field amplitudes.

    Uses inverted multiplier mapping and cyclic qutrit repeat.
    """
    n_fields = len(field_amplitudes)
    n_qutrits = len(qutrit_states)
    if n_fields == 0 or n_qutrits == 0:
        return field_amplitudes
    return tuple(
        field_amplitudes[i] * _ANTI_COUPLING_MULTIPLIERS[qutrit_states[i % n_qutrits]]
        for i in range(n_fields)
    )


def _evolve_anti_universe(state: UniverseState) -> UniverseState:
    """Evolve one step under anti-law rules.

    Steps:
        1. Apply anti-decay (1.001 multiplier per lane)
        2. Apply anti-law qutrit coupling (inverted multipliers)
        3. Increment timestep
    """
    decayed = tuple(f * _ANTI_DECAY for f in state.field_amplitudes)
    coupled = _apply_anti_coupling(decayed, state.qutrit_states)
    return UniverseState(
        field_amplitudes=coupled,
        qutrit_states=state.qutrit_states,
        timestep=state.timestep + 1,
        law_name="anti-law",
    )


def run_lawful_universe(
    initial_state: UniverseState,
    steps: int,
) -> Tuple[UniverseState, ...]:
    """Run lawful coupled evolution for the given number of steps.

    Uses the standard evolve_universe_coupled law.

    Parameters
    ----------
    initial_state : UniverseState
        Starting state (included as first element of history).
    steps : int
        Number of evolution steps to run.

    Returns
    -------
    Tuple[UniverseState, ...]
        Full state history including initial state (length = steps + 1).
    """
    history = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = evolve_universe_coupled(state)
        history.append(state)
    return tuple(history)


def run_anti_universe(
    initial_state: UniverseState,
    steps: int,
) -> Tuple[UniverseState, ...]:
    """Run anti-law evolution for the given number of steps.

    Anti-law rules invert qutrit semantics and use amplifying decay.

    Parameters
    ----------
    initial_state : UniverseState
        Starting state (included as first element of history).
    steps : int
        Number of evolution steps to run.

    Returns
    -------
    Tuple[UniverseState, ...]
        Full state history including initial state (length = steps + 1).
    """
    history = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = _evolve_anti_universe(state)
        history.append(state)
    return tuple(history)


@dataclass(frozen=True)
class UniverseComparison:
    """Frozen comparison result between lawful and anti-law universes."""

    lawful_final_energy: float
    anti_final_energy: float
    divergence_score: float
    energy_ratio: float
    steps: int


def compare_universes(
    initial_state: UniverseState,
    steps: int = 100,
) -> UniverseComparison:
    """Run lawful and anti-law universes and compare divergence.

    Both universes start from identical initial conditions.

    Parameters
    ----------
    initial_state : UniverseState
        Shared starting state for both universes.
    steps : int
        Number of evolution steps (default 100).

    Returns
    -------
    UniverseComparison
        Frozen comparison with divergence metrics.
    """
    lawful_history = run_lawful_universe(initial_state, steps)
    anti_history = run_anti_universe(initial_state, steps)

    lawful_obs = observe_universe(lawful_history[-1])
    anti_obs = observe_universe(anti_history[-1])

    lawful_energy = lawful_obs.mean_field_energy
    anti_energy = anti_obs.mean_field_energy

    divergence_score = abs(lawful_energy - anti_energy)

    if lawful_energy == 0.0:
        energy_ratio = 0.0
    else:
        energy_ratio = anti_energy / lawful_energy

    return UniverseComparison(
        lawful_final_energy=lawful_energy,
        anti_final_energy=anti_energy,
        divergence_score=divergence_score,
        energy_ratio=energy_ratio,
        steps=steps,
    )
