# SPDX-License-Identifier: MIT
"""Simulation Adapter Interface Layer — v133.3.0.

Deterministic adapter interface for exporting simulation states
into external simulator-friendly formats (QuTiP, Qiskit, generic).

No external simulator dependencies are imported.
This is pure interface infrastructure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from qec.sims.universe_kernel import UniverseState

_SCHEMA_VERSION = "133.3.0"

_VALID_BACKENDS = frozenset({"generic", "qutip", "qiskit"})


def _validate_backend_target(backend_target: str) -> str:
    """Validate and return a canonical backend target string.

    Raises:
        ValueError: If backend_target is not supported.
    """
    if backend_target not in _VALID_BACKENDS:
        raise ValueError(f"unsupported backend_target: {backend_target!r}")
    return backend_target


@dataclass(frozen=True)
class SimulationAdapterPayload:
    """Immutable payload for adapter export.

    All collection fields are tuples for deterministic equality
    and replay safety.
    """

    state_vector: Tuple[float, ...]
    qutrit_register: Tuple[int, ...]
    timestep: int
    backend_target: str
    schema_version: str


def to_adapter_payload(
    state: UniverseState,
    backend_target: str = "generic",
) -> SimulationAdapterPayload:
    """Export a UniverseState into a SimulationAdapterPayload.

    Args:
        state: The universe state snapshot to export.
        backend_target: One of "generic", "qutip", "qiskit".

    Returns:
        Frozen adapter payload with tuple-only collections.

    Raises:
        ValueError: If backend_target is not recognised.
    """
    _validate_backend_target(backend_target)
    return SimulationAdapterPayload(
        state_vector=tuple(state.field_amplitudes),
        qutrit_register=tuple(state.qutrit_states),
        timestep=state.timestep,
        backend_target=backend_target,
        schema_version=_SCHEMA_VERSION,
    )


def normalize_for_backend(
    payload: SimulationAdapterPayload,
) -> SimulationAdapterPayload:
    """Apply backend-specific deterministic normalization.

    Rules:
        generic  — return unchanged.
        qutip    — L2-normalize state_vector to unit norm.
        qiskit   — clip qutrit_register values to {0, 1, 2}.

    Returns:
        A new frozen payload (original is never mutated).

    Raises:
        ValueError: If backend_target is not recognised.
    """
    backend = _validate_backend_target(payload.backend_target)

    if backend == "generic":
        return payload

    if backend == "qutip":
        norm = math.sqrt(sum(x * x for x in payload.state_vector))
        if norm == 0.0:
            normalized = payload.state_vector
        else:
            normalized = tuple(x / norm for x in payload.state_vector)
        return replace(payload, state_vector=normalized)

    if backend == "qiskit":
        clipped = tuple(max(0, min(2, v)) for v in payload.qutrit_register)
        return replace(payload, qutrit_register=clipped)

    raise ValueError(f"unsupported backend_target: {backend!r}")  # pragma: no cover
