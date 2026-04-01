# SPDX-License-Identifier: MIT
"""Deterministic micro-universe simulation kernel.

Provides frozen immutable state objects and a pure evolution function.
All operations are replay-safe with byte-identical outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.simulation.export_codec import with_computed_trace_hash
from qec.simulation.export_schema import (
    ExportMetadata,
    SimulationExport,
)

EXPORT_SCHEMA_VERSION = "132.5.0"
CREATED_BY_RELEASE = "133.0.0"


@dataclass(frozen=True)
class UniverseState:
    """Immutable snapshot of a micro-universe patch.

    All collection fields use tuples to enforce immutability
    and deterministic equality.
    """

    field_amplitudes: Tuple[float, ...]
    qutrit_states: Tuple[int, ...]
    timestep: int
    law_name: str


def evolve_universe(state: UniverseState) -> UniverseState:
    """Evolve the universe state by one deterministic timestep.

    Pure function. No randomness, no global state, no mutation.

    Evolution law (minimal deterministic):
        new_field[i] = old_field[i] * 0.999
        new_qutrit[i] = old_qutrit[i]
        timestep increments by 1

    Parameters
    ----------
    state : UniverseState
        Current universe snapshot.

    Returns
    -------
    UniverseState
        New immutable state after one evolution step.
    """
    new_fields = tuple(f * 0.999 for f in state.field_amplitudes)
    return UniverseState(
        field_amplitudes=new_fields,
        qutrit_states=state.qutrit_states,
        timestep=state.timestep + 1,
        law_name=state.law_name,
    )


def to_simulation_export(state: UniverseState) -> SimulationExport:
    """Convert a UniverseState to a v132.5 SimulationExport.

    Consumes the existing export bridge schema without modifying it.
    Produces a finalized export with a computed trace hash.

    Parameters
    ----------
    state : UniverseState
        The universe state to export.

    Returns
    -------
    SimulationExport
        A finalized, hash-stamped export artifact.
    """
    control_trace = tuple(
        f"field[{i}]={v:.17g}" for i, v in enumerate(state.field_amplitudes)
    )
    dwell_events = (state.timestep,)
    export = SimulationExport(
        control_trace=control_trace,
        dwell_events=dwell_events,
        fail_safe_events=(),
        transition_events=(),
        metadata=ExportMetadata(
            schema_version=EXPORT_SCHEMA_VERSION,
            trace_hash="",
            created_by_release=CREATED_BY_RELEASE,
        ),
    )
    return with_computed_trace_hash(export)
