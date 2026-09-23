# SPDX-License-Identifier: MPL-2.0
"""v173.0 stored-program control skeleton over an immutable fabric snapshot.

Separate programme, call and queue snapshots produce a bounded command stream.
The registered Crossbar adapter plans each call independently; the controller
validates native evidence before recording results. No call lifecycle, resource
mutation, user-code execution or wall-clock scheduler is introduced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from qec.sonify.canonical import canonical_json, canonical_sha256, validate_sha256
from qec.routing.crossbar.marker import CrossbarRequest
from qec.routing.crossbar.multistage import (
    MAX_SEARCH_EVALUATIONS, CrossbarFabric, MultiStageRequest,
    compile_path_program, demo_fabric, execute_path_program,
)
from qec.routing.crossbar.continuity import create_continuity_receipt

CONTRACT_VERSION = "173.0"
ADAPTER_ID = "crossbar-path-v172.2-continuity-v172.4"
POLICY = "logical-queue-independent-plan-once.v1"
MAX_CALLS = 16
MAX_PAYLOAD_BYTES = 4096
MAX_LOGICAL_TICK = 2**63 - 1
CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "fixed_builtin_programme_only": True,
    "programme_source_manifest_present": False,
    "translation_tables_present": False,
    "call_processing_state_machine_present": False,
    "priority_or_interrupt_scheduler_present": False,
    "feature_modules_present": False,
    "fabric_scope": "independent_plans_against_one_immutable_snapshot",
    "native_result_validation_required": True,
    "reservation_present": False,
    "connection_commit_present": False,
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "decoder_output_identity_is_caller_declared": True,
    "receipt_proves": "bounded_logical_dispatch_and_validated_native_planning_evidence",
    "receipt_does_not_prove": "simultaneous_capacity_or_call_lifecycle_or_authenticated_provenance",
})


def _seal(name: str, **fields: object) -> dict:
    unsigned = {"schema": "qec.ess-" + name + ".v1",
                "contract_version": CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _exact(actual: object, expected: dict) -> None:
    try:
        if type(actual) is not dict or canonical_json(actual) != canonical_json(expected):
            raise ValueError("INVALID_INPUT")
    except (TypeError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _text(value: object) -> None:
    if type(value) is not str or not 1 <= len(value) <= 128:
        raise ValueError("INVALID_INPUT")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("INVALID_INPUT") from exc


def _int(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def _sequence(value: object) -> tuple:
    if type(value) not in (tuple, list) or len(value) > MAX_CALLS:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


@dataclass(frozen=True)
class ProgramStore:
    """One fixed built-in dispatch programme; version and budget are explicit.

    This is a configuration snapshot, not the v173.1 full source manifest.
    Labels cannot select executable code or grant extra adapter authority.
    """

    program_id: str = "ess-plan"
    program_version: str = "1"
    marker_id: str = "ess-marker"
    max_search_evaluations: int = MAX_SEARCH_EVALUATIONS

    def __post_init__(self) -> None:
        for value in (self.program_id, self.program_version, self.marker_id):
            _text(value)
        _int(self.max_search_evaluations, MAX_SEARCH_EVALUATIONS, 1)

    def as_dict(self) -> dict:
        return _seal("program-store", program_id=self.program_id,
                     program_version=self.program_version, marker_id=self.marker_id,
                     max_search_evaluations=self.max_search_evaluations,
                     policy=POLICY, adapter_id=ADAPTER_ID, operation="plan_route")

    @classmethod
    def from_dict(cls, value: object) -> ProgramStore:
        try:
            result = cls(value["program_id"], value["program_version"],
                         value["marker_id"], value["max_search_evaluations"])
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class CallStore:
    """Sealed call data, ordered by unique native request ID; no lifecycle state."""

    calls: tuple[MultiStageRequest, ...]

    def __post_init__(self) -> None:
        calls = _sequence(self.calls)
        if any(type(call) is not MultiStageRequest for call in calls):
            raise ValueError("INVALID_INPUT")
        calls = tuple(MultiStageRequest.from_dict(call.as_dict()) for call in calls)
        for call in calls:
            _text(call.request.request_id)
            if len(call.request.payload) > MAX_PAYLOAD_BYTES:
                raise ValueError("INVALID_INPUT")
        ids = tuple(call.request.request_id for call in calls)
        if ids != tuple(sorted(set(ids))):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "calls", calls)

    def as_dict(self) -> dict:
        return _seal("call-store", calls=[call.as_dict() for call in self.calls])

    @classmethod
    def from_dict(cls, value: object) -> CallStore:
        try:
            calls = _sequence(value["calls"])
            result = cls(tuple(MultiStageRequest.from_dict(call) for call in calls))
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class InputEvent:
    """A plan request at a logical tick with an explicit FIFO sequence number."""

    logical_tick: int
    sequence: int
    call_id: str

    def __post_init__(self) -> None:
        _int(self.logical_tick, MAX_LOGICAL_TICK)
        _int(self.sequence, MAX_CALLS - 1)
        _text(self.call_id)

    def as_dict(self) -> dict:
        return _seal("input-event", logical_tick=self.logical_tick,
                     sequence=self.sequence, call_id=self.call_id, kind="plan_route")

    @classmethod
    def from_dict(cls, value: object) -> InputEvent:
        try:
            result = cls(value["logical_tick"], value["sequence"], value["call_id"])
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class EventQueue:
    """Closed input queue: ascending ticks, contiguous sequence, one event/call.

    Caller-declared order is validated, never repaired by sorting. Equal ticks
    retain explicit sequence order; no priority classes or interrupts exist yet.
    """

    events: tuple[InputEvent, ...]

    def __post_init__(self) -> None:
        events = _sequence(self.events)
        if any(type(event) is not InputEvent for event in events):
            raise ValueError("INVALID_INPUT")
        events = tuple(InputEvent.from_dict(event.as_dict()) for event in events)
        if (tuple(e.sequence for e in events) != tuple(range(len(events)))
                or tuple(e.logical_tick for e in events) != tuple(sorted(e.logical_tick for e in events))
                or len({e.call_id for e in events}) != len(events)):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "events", events)

    def as_dict(self) -> dict:
        return _seal("event-queue", ordering=["logical_tick", "sequence"],
                     events=[event.as_dict() for event in self.events])

    @classmethod
    def from_dict(cls, value: object) -> EventQueue:
        try:
            events = _sequence(value["events"])
            result = cls(tuple(InputEvent.from_dict(event) for event in events))
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class CrossbarFabricAdapter:
    """The sole v173.0 adapter: native planning without fabric mutation.

    Results remain untrusted until the ESS controller validates native replay
    against the exact fabric, request and compiled programme identities.
    """

    fabric: CrossbarFabric

    def __post_init__(self) -> None:
        if type(self.fabric) is not CrossbarFabric:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "fabric", CrossbarFabric.from_dict(self.fabric.as_dict()))

    def as_dict(self) -> dict:
        return _seal("fabric-adapter", adapter_id=ADAPTER_ID,
                     fabric_manifest=self.fabric.as_dict(), operations=["plan_route"],
                     state_semantics="immutable_snapshot", connection_commit_present=False)

    @classmethod
    def from_dict(cls, value: object) -> CrossbarFabricAdapter:
        try:
            result = cls(CrossbarFabric.from_dict(value["fabric_manifest"]))
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc

    def plan_route(self, request: MultiStageRequest, native_program: dict) -> dict:
        return execute_path_program(self.fabric.as_dict(), request, native_program)


@dataclass(frozen=True)
class SwitchInput:
    """Four detached stores/boundaries, with exactly one event for every call."""

    program_store: ProgramStore
    call_store: CallStore
    event_queue: EventQueue
    adapter: CrossbarFabricAdapter

    def __post_init__(self) -> None:
        for name, cls in (("program_store", ProgramStore), ("call_store", CallStore),
                          ("event_queue", EventQueue), ("adapter", CrossbarFabricAdapter)):
            value = getattr(self, name)
            if type(value) is not cls:
                raise ValueError("INVALID_INPUT")
            object.__setattr__(self, name, cls.from_dict(value.as_dict()))
        if ({c.request.request_id for c in self.call_store.calls}
                != {e.call_id for e in self.event_queue.events}):
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        return _seal("switch-input", program_store=self.program_store.as_dict(),
                     call_store=self.call_store.as_dict(), event_queue=self.event_queue.as_dict(),
                     adapter=self.adapter.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> SwitchInput:
        try:
            result = cls(ProgramStore.from_dict(value["program_store"]),
                         CallStore.from_dict(value["call_store"]),
                         EventQueue.from_dict(value["event_queue"]),
                         CrossbarFabricAdapter.from_dict(value["adapter"]))
            _exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def execute_switch(switch_input: SwitchInput) -> dict:
    """Drain a closed queue, validate native results and retain the exact evidence.

    All plans see the same fabric state. A native rejection is a valid result,
    not permission to retry, mutate the payload, or force a connection.
    """
    if type(switch_input) is not SwitchInput:
        raise ValueError("INVALID_INPUT")
    source = SwitchInput.from_dict(switch_input.as_dict())
    snapshot = source.as_dict()
    program = source.program_store
    fabric = source.adapter.fabric.as_dict()
    calls = {call.request.request_id: call for call in source.call_store.calls}
    commands, results = [], []
    for event in source.event_queue.events:
        request = calls[event.call_id]
        native_program = compile_path_program(
            fabric, request, marker_id=program.marker_id,
            max_search_evaluations=program.max_search_evaluations,
        )
        command = _seal("fabric-command", sequence=event.sequence,
                        logical_tick=event.logical_tick, call_id=event.call_id,
                        operation="plan_route", input_sha256=snapshot["sha256"],
                        program_store_sha256=snapshot["program_store"]["sha256"],
                        event_sha256=event.as_dict()["sha256"],
                        adapter_sha256=snapshot["adapter"]["sha256"],
                        request_sha256=request.as_dict()["sha256"],
                        native_program_sha256=native_program["sha256"])
        # The adapter cannot certify its own result. Replay with trusted pins in
        # the controller catches even a valid receipt substituted from another call.
        native = source.adapter.plan_route(request, native_program)
        evidence = create_continuity_receipt(
            native, expected_fabric_sha256=fabric["sha256"],
            expected_input_sha256=request.as_dict()["sha256"],
            expected_program_sha256=command["native_program_sha256"],
        )
        # Read only from the validated detached evidence, not the adapter object.
        verified = evidence["source_receipt"]
        commands.append(command)
        results.append(_seal("dispatch-result", command_sha256=command["sha256"],
                             call_id=event.call_id, outcome=verified["outcome"],
                             reason=verified["reason"], continuity_receipt=evidence))
    stream = _seal("command-stream", input_sha256=snapshot["sha256"], commands=commands)
    selected = sum(row["outcome"] == "plan_selected" for row in results)
    return _seal("switch-skeleton-receipt", input=snapshot, command_stream=stream,
                 results=results, dispatched_count=len(results), selected_count=selected,
                 rejected_count=len(results) - selected,
                 queue_drained=len(results) == len(source.event_queue.events),
                 all_adapter_results_verified=True, claim_boundary=dict(CLAIM_BOUNDARY))


def validate_switch_receipt(receipt: object, *, expected_input_sha256: str | None = None,
                            expected_program_store_sha256: str | None = None,
                            expected_adapter_sha256: str | None = None) -> dict:
    """Rebuild the whole stream and all evidence; pins bind trusted external inputs."""
    try:
        source = SwitchInput.from_dict(receipt["input"])
        snapshot = source.as_dict()
        pairs = {"input": (expected_input_sha256, snapshot["sha256"]),
                 "program_store": (expected_program_store_sha256, snapshot["program_store"]["sha256"]),
                 "adapter": (expected_adapter_sha256, snapshot["adapter"]["sha256"])}
        for expected, actual in pairs.values():
            if expected is not None and validate_sha256(expected) != actual:
                raise ValueError("INVALID_INPUT")
        replay = execute_switch(source)
        _exact(receipt, replay)
        return _seal("switch-skeleton-validation", ess_switch_skeleton_receipt_hash=replay["sha256"],
                     ess_command_stream_hash=replay["command_stream"]["sha256"],
                     dispatched_count=replay["dispatched_count"], selected_count=replay["selected_count"],
                     rejected_count=replay["rejected_count"], complete_replay_verified=True,
                     external_bindings_verified={key: pair[0] is not None for key, pair in pairs.items()},
                     all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def demo_switch_input() -> SwitchInput:
    """Two same-tick calls, a route and an unknown-endpoint rejection."""
    calls = tuple(MultiStageRequest("ingress", "egress", CrossbarRequest(
        name, "H000", destination, b"opaque-correction", "a" * 64,
    )) for name, destination in (("call-a", "V001"), ("call-b", "unknown")))
    return SwitchInput(ProgramStore(), CallStore(calls),
                       EventQueue((InputEvent(0, 0, "call-b"), InputEvent(0, 1, "call-a"))),
                       CrossbarFabricAdapter(demo_fabric()))
