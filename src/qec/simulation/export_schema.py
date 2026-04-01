# SPDX-License-Identifier: MIT
"""Frozen immutable export schema for deterministic simulation traces.

All dataclasses are frozen to guarantee immutability and byte-identical replay.
These schemas define the export contract consumed by future v133 simulation
harnesses (QuTiP, Qiskit, bosonic workflows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Sentinel used during trace hash computation to break circularity.
TRACE_HASH_SENTINEL = ""


@dataclass(frozen=True)
class TransitionEvent:
    """A single supervisory state transition."""

    from_state: str
    to_state: str
    timestamp_ns: int
    reason: str


@dataclass(frozen=True)
class FailSafeEvent:
    """A fail-safe activation event."""

    timestamp_ns: int
    trigger: str
    resolved: bool


@dataclass(frozen=True)
class ExportMetadata:
    """Versioned metadata for an export artifact."""

    schema_version: str
    trace_hash: str
    created_by_release: str


@dataclass(frozen=True)
class SimulationExport:
    """Top-level deterministic simulation export object.

    All collection fields use tuples (not lists) to enforce immutability.
    """

    control_trace: Tuple[str, ...]
    dwell_events: Tuple[int, ...]
    fail_safe_events: Tuple[FailSafeEvent, ...]
    transition_events: Tuple[TransitionEvent, ...]
    metadata: ExportMetadata
