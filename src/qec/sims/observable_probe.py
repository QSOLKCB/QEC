# SPDX-License-Identifier: MIT
"""Observable probes for the micro-universe kernel.

All observations are deterministic pure functions of state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.universe_kernel import UniverseState


@dataclass(frozen=True)
class UniverseObservation:
    """Immutable observation of a universe state."""

    mean_field_energy: float
    active_qutrit_count: int
    stability_score: float
    timestep: int


def observe_universe(state: UniverseState) -> UniverseObservation:
    """Compute deterministic observables from a universe state.

    Parameters
    ----------
    state : UniverseState
        The universe snapshot to observe.

    Returns
    -------
    UniverseObservation
        Frozen observation with computed metrics.
    """
    n = len(state.field_amplitudes)
    if n == 0:
        mean_field_energy = 0.0
    else:
        mean_field_energy = sum(x * x for x in state.field_amplitudes) / n

    active_qutrit_count = sum(1 for q in state.qutrit_states if q != 0)

    stability_score = mean_field_energy / (1 + active_qutrit_count)

    return UniverseObservation(
        mean_field_energy=mean_field_energy,
        active_qutrit_count=active_qutrit_count,
        stability_score=stability_score,
        timestep=state.timestep,
    )
