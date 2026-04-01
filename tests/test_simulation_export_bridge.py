# SPDX-License-Identifier: MIT
"""Deterministic tests for the simulation export bridge (v132.5.0)."""

from __future__ import annotations

import pytest

from qec.simulation.export_schema import (
    ExportMetadata,
    FailSafeEvent,
    SimulationExport,
    TransitionEvent,
)
from qec.simulation.export_codec import (
    ExportSchemaError,
    compute_trace_hash,
    export_to_json,
    load_from_json,
    validate_export_replay,
    with_computed_trace_hash,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata() -> ExportMetadata:
    return ExportMetadata(
        schema_version="132.5.0",
        trace_hash="placeholder",
        created_by_release="v132.5.0",
    )


def _make_transition() -> TransitionEvent:
    return TransitionEvent(
        from_state="idle",
        to_state="active",
        timestamp_ns=1000,
        reason="threshold_exceeded",
    )


def _make_failsafe() -> FailSafeEvent:
    return FailSafeEvent(
        timestamp_ns=2000,
        trigger="overload",
        resolved=True,
    )


def _make_export() -> SimulationExport:
    return SimulationExport(
        control_trace=("init", "run", "halt"),
        dwell_events=(100, 200, 300),
        fail_safe_events=(_make_failsafe(),),
        transition_events=(_make_transition(),),
        metadata=_make_metadata(),
    )


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------


class TestFrozenImmutability:
    def test_simulation_export_is_frozen(self) -> None:
        export = _make_export()
        with pytest.raises(AttributeError):
            export.control_trace = ("modified",)  # type: ignore[misc]

    def test_metadata_is_frozen(self) -> None:
        meta = _make_metadata()
        with pytest.raises(AttributeError):
            meta.schema_version = "999"  # type: ignore[misc]

    def test_transition_event_is_frozen(self) -> None:
        event = _make_transition()
        with pytest.raises(AttributeError):
            event.from_state = "other"  # type: ignore[misc]

    def test_failsafe_event_is_frozen(self) -> None:
        event = _make_failsafe()
        with pytest.raises(AttributeError):
            event.timestamp_ns = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Round-trip JSON equality
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_export_round_trip(self) -> None:
        original = _make_export()
        json_text = export_to_json(original)
        restored = load_from_json(json_text)
        assert original == restored

    def test_tuple_reconstruction(self) -> None:
        original = _make_export()
        restored = load_from_json(export_to_json(original))
        assert isinstance(restored.control_trace, tuple)
        assert isinstance(restored.dwell_events, tuple)
        assert isinstance(restored.fail_safe_events, tuple)
        assert isinstance(restored.transition_events, tuple)

    def test_schema_version_preserved(self) -> None:
        original = _make_export()
        restored = load_from_json(export_to_json(original))
        assert restored.metadata.schema_version == "132.5.0"
        assert restored.metadata.created_by_release == "v132.5.0"


# ---------------------------------------------------------------------------
# Byte-identical replay
# ---------------------------------------------------------------------------


class TestByteIdenticalReplay:
    def test_validate_export_replay(self) -> None:
        export = _make_export()
        assert validate_export_replay(export) is True

    def test_repeated_serialization_identical(self) -> None:
        export = _make_export()
        json_a = export_to_json(export)
        json_b = export_to_json(export)
        assert json_a == json_b

    def test_deterministic_repeated_export(self) -> None:
        """Serialize 100 times — all outputs must be identical."""
        export = _make_export()
        first = export_to_json(export)
        for _ in range(100):
            assert export_to_json(export) == first


# ---------------------------------------------------------------------------
# Stable hashing
# ---------------------------------------------------------------------------


class TestStableHashing:
    def test_hash_deterministic(self) -> None:
        export = _make_export()
        h1 = compute_trace_hash(export)
        h2 = compute_trace_hash(export)
        assert h1 == h2

    def test_hash_changes_with_data(self) -> None:
        export_a = _make_export()
        export_b = SimulationExport(
            control_trace=("different",),
            dwell_events=(999,),
            fail_safe_events=(),
            transition_events=(),
            metadata=_make_metadata(),
        )
        assert compute_trace_hash(export_a) != compute_trace_hash(export_b)

    def test_hash_is_sha256_hex(self) -> None:
        h = compute_trace_hash(_make_export())
        assert len(h) == 64
        int(h, 16)  # valid hex

    def test_hash_idempotent_after_finalization(self) -> None:
        """Hash must be the same before and after inserting it into metadata."""
        export = _make_export()
        finalized = with_computed_trace_hash(export)
        assert compute_trace_hash(finalized) == finalized.metadata.trace_hash

    def test_hash_independent_of_placeholder(self) -> None:
        """Different placeholder values must produce the same hash."""
        meta_a = ExportMetadata(
            schema_version="132.5.0",
            trace_hash="placeholder_a",
            created_by_release="v132.5.0",
        )
        meta_b = ExportMetadata(
            schema_version="132.5.0",
            trace_hash="placeholder_b",
            created_by_release="v132.5.0",
        )
        export_a = SimulationExport(
            control_trace=("x",), dwell_events=(1,),
            fail_safe_events=(), transition_events=(),
            metadata=meta_a,
        )
        export_b = SimulationExport(
            control_trace=("x",), dwell_events=(1,),
            fail_safe_events=(), transition_events=(),
            metadata=meta_b,
        )
        assert compute_trace_hash(export_a) == compute_trace_hash(export_b)


# ---------------------------------------------------------------------------
# Empty export
# ---------------------------------------------------------------------------


class TestWithComputedTraceHash:
    def test_finalized_export_has_valid_hash(self) -> None:
        finalized = with_computed_trace_hash(_make_export())
        assert finalized.metadata.trace_hash != "placeholder"
        assert len(finalized.metadata.trace_hash) == 64

    def test_finalized_round_trips(self) -> None:
        finalized = with_computed_trace_hash(_make_export())
        assert validate_export_replay(finalized) is True

    def test_double_finalize_idempotent(self) -> None:
        once = with_computed_trace_hash(_make_export())
        twice = with_computed_trace_hash(once)
        assert once == twice


class TestSchemaValidation:
    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ExportSchemaError, match="invalid JSON"):
            load_from_json("not json")

    def test_missing_top_level_key_raises(self) -> None:
        with pytest.raises(ExportSchemaError, match="missing keys"):
            load_from_json('{"control_trace":[]}')

    def test_missing_metadata_key_raises(self) -> None:
        payload = export_to_json(_make_export())
        import json
        d = json.loads(payload)
        del d["metadata"]["schema_version"]
        with pytest.raises(ExportSchemaError, match="ExportMetadata.*missing keys"):
            load_from_json(json.dumps(d))

    def test_non_dict_metadata_raises(self) -> None:
        payload = export_to_json(_make_export())
        import json
        d = json.loads(payload)
        d["metadata"] = "not_a_dict"
        with pytest.raises(ExportSchemaError, match="expected dict"):
            load_from_json(json.dumps(d))


class TestEdgeCases:
    def test_empty_export_round_trip(self) -> None:
        export = SimulationExport(
            control_trace=(),
            dwell_events=(),
            fail_safe_events=(),
            transition_events=(),
            metadata=_make_metadata(),
        )
        assert validate_export_replay(export) is True

    def test_empty_export_hash_stable(self) -> None:
        export = SimulationExport(
            control_trace=(),
            dwell_events=(),
            fail_safe_events=(),
            transition_events=(),
            metadata=_make_metadata(),
        )
        assert compute_trace_hash(export) == compute_trace_hash(export)
