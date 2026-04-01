# SPDX-License-Identifier: MIT
"""Deterministic JSON codec for simulation export artifacts.

Guarantees:
- Canonical key ordering (sort_keys=True)
- Stable JSON output — same object always produces identical bytes
- Explicit tuple reconstruction on load
- No mutation of input objects
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from qec.simulation.export_schema import (
    TRACE_HASH_SENTINEL,
    ExportMetadata,
    FailSafeEvent,
    SimulationExport,
    TransitionEvent,
)


class ExportSchemaError(Exception):
    """Raised when JSON payload does not conform to the export schema."""


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _transition_to_dict(event: TransitionEvent) -> Dict[str, Any]:
    return {
        "from_state": event.from_state,
        "to_state": event.to_state,
        "timestamp_ns": event.timestamp_ns,
        "reason": event.reason,
    }


def _failsafe_to_dict(event: FailSafeEvent) -> Dict[str, Any]:
    return {
        "timestamp_ns": event.timestamp_ns,
        "trigger": event.trigger,
        "resolved": event.resolved,
    }


def _metadata_to_dict(meta: ExportMetadata) -> Dict[str, Any]:
    return {
        "schema_version": meta.schema_version,
        "trace_hash": meta.trace_hash,
        "created_by_release": meta.created_by_release,
    }


def _export_to_dict(export: SimulationExport) -> Dict[str, Any]:
    return {
        "control_trace": list(export.control_trace),
        "dwell_events": list(export.dwell_events),
        "fail_safe_events": [_failsafe_to_dict(e) for e in export.fail_safe_events],
        "transition_events": [_transition_to_dict(e) for e in export.transition_events],
        "metadata": _metadata_to_dict(export.metadata),
    }


def export_to_json(export: SimulationExport) -> str:
    """Serialize a SimulationExport to canonical, deterministic JSON."""
    return json.dumps(
        _export_to_dict(export),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def _dict_to_transition(d: Dict[str, Any]) -> TransitionEvent:
    return TransitionEvent(
        from_state=d["from_state"],
        to_state=d["to_state"],
        timestamp_ns=d["timestamp_ns"],
        reason=d["reason"],
    )


def _dict_to_failsafe(d: Dict[str, Any]) -> FailSafeEvent:
    return FailSafeEvent(
        timestamp_ns=d["timestamp_ns"],
        trigger=d["trigger"],
        resolved=d["resolved"],
    )


def _dict_to_metadata(d: Dict[str, Any]) -> ExportMetadata:
    return ExportMetadata(
        schema_version=d["schema_version"],
        trace_hash=d["trace_hash"],
        created_by_release=d["created_by_release"],
    )


_EXPORT_REQUIRED_KEYS = frozenset({
    "control_trace", "dwell_events", "fail_safe_events",
    "transition_events", "metadata",
})
_METADATA_REQUIRED_KEYS = frozenset({
    "schema_version", "trace_hash", "created_by_release",
})
_TRANSITION_REQUIRED_KEYS = frozenset({
    "from_state", "to_state", "timestamp_ns", "reason",
})
_FAILSAFE_REQUIRED_KEYS = frozenset({
    "timestamp_ns", "trigger", "resolved",
})


def _validate_keys(d: Any, required: frozenset, context: str) -> None:
    if not isinstance(d, dict):
        raise ExportSchemaError(f"{context}: expected dict, got {type(d).__name__}")
    missing = required - d.keys()
    if missing:
        raise ExportSchemaError(f"{context}: missing keys {sorted(missing)}")


def load_from_json(text: str) -> SimulationExport:
    """Deserialize canonical JSON back to a SimulationExport.

    Lists are explicitly reconstructed as tuples.
    Raises ExportSchemaError on malformed or missing fields.
    """
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExportSchemaError(f"invalid JSON: {exc}") from exc
    _validate_keys(d, _EXPORT_REQUIRED_KEYS, "SimulationExport")
    _validate_keys(d["metadata"], _METADATA_REQUIRED_KEYS, "ExportMetadata")
    for i, evt in enumerate(d.get("transition_events", ())):
        _validate_keys(evt, _TRANSITION_REQUIRED_KEYS, f"TransitionEvent[{i}]")
    for i, evt in enumerate(d.get("fail_safe_events", ())):
        _validate_keys(evt, _FAILSAFE_REQUIRED_KEYS, f"FailSafeEvent[{i}]")
    return SimulationExport(
        control_trace=tuple(d["control_trace"]),
        dwell_events=tuple(d["dwell_events"]),
        fail_safe_events=tuple(_dict_to_failsafe(e) for e in d["fail_safe_events"]),
        transition_events=tuple(_dict_to_transition(e) for e in d["transition_events"]),
        metadata=_dict_to_metadata(d["metadata"]),
    )


# ---------------------------------------------------------------------------
# Trace hashing
# ---------------------------------------------------------------------------


def compute_trace_hash(export: SimulationExport) -> str:
    """Compute a deterministic SHA-256 hash of the export's canonical JSON.

    To avoid circularity, ``trace_hash`` is normalized to
    :data:`TRACE_HASH_SENTINEL` before hashing.  This makes the hash
    independent of any previously stored ``trace_hash`` value and ensures
    idempotency: calling this function on a finalized export produces the
    same result as calling it on the pre-hash version.
    """
    normalized = SimulationExport(
        control_trace=export.control_trace,
        dwell_events=export.dwell_events,
        fail_safe_events=export.fail_safe_events,
        transition_events=export.transition_events,
        metadata=ExportMetadata(
            schema_version=export.metadata.schema_version,
            trace_hash=TRACE_HASH_SENTINEL,
            created_by_release=export.metadata.created_by_release,
        ),
    )
    canonical = export_to_json(normalized)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def with_computed_trace_hash(export: SimulationExport) -> SimulationExport:
    """Return a new export with ``metadata.trace_hash`` set to the computed hash.

    This is the canonical way to finalize an export artifact.
    """
    trace_hash = compute_trace_hash(export)
    return SimulationExport(
        control_trace=export.control_trace,
        dwell_events=export.dwell_events,
        fail_safe_events=export.fail_safe_events,
        transition_events=export.transition_events,
        metadata=ExportMetadata(
            schema_version=export.metadata.schema_version,
            trace_hash=trace_hash,
            created_by_release=export.metadata.created_by_release,
        ),
    )


# ---------------------------------------------------------------------------
# Replay validation
# ---------------------------------------------------------------------------


def validate_export_replay(export: SimulationExport) -> bool:
    """Verify byte-identical round-trip: export -> json -> load -> json.

    Returns True if the two JSON serializations are identical.
    """
    json_a = export_to_json(export)
    reconstructed = load_from_json(json_a)
    json_b = export_to_json(reconstructed)
    return json_a == json_b
