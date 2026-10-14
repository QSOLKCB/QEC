# SPDX-License-Identifier: MPL-2.0
"""v173.2–.5 closed ESS lifecycle with owned native resources and full replay.

Native command ticks are dispatch ordinals. The separately sealed logical-tick
scheduler determines that order, including release interrupts and bounded retries.
No physical connection, user programme, live network or decoder is executed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq

from qec.routing.crossbar.core import demo_matrix
from qec.routing.crossbar.marker import CrossbarRequest
from qec.routing.crossbar.multistage import CrossbarFabric, InterstageLink, MultiStageRequest, demo_fabric
from qec.routing.crossbar.contention import (
    ContentionBatch, ContentionCommand, compile_contention_program, execute_contention_program,
)
from qec.routing.crossbar.continuity import create_continuity_receipt
from qec.sonify.canonical import canonical_sha256, validate_sha256
from .programme import (
    MAX_CALLS, MAX_NATIVE_COMMANDS, MAX_PAYLOAD_BYTES, MAX_TICK,
    FeatureModules, ProgramManifest, TranslationTable, bindings, bounded_json,
    exact, integer, routing_logic, seal, sequence, text,
)

ADAPTER_ID = "crossbar-owned-reservations-v172.3-continuity-v172.4"


def claim_boundary() -> dict:
    return {
        "classical_software_model_only": True, "native_reservations_and_release_present": True,
        "connect_is_software_owned_reservation": True, "physical_connection_commit_present": False,
        "logical_time_only": True, "fixed_builtin_programme_only": True,
        "native_replay_and_independent_continuity_required": True,
        "fanout_is_independent_branches": True, "force_accept_permitted": False,
        "payload_mutation_permitted": False, "decoder_output_mutation_permitted": False,
        "decoder_identity_is_caller_declared": True, "authenticated_provenance": False,
        "receipt_proves": "closed_bounded_call_lifecycle_and_owned_native_resource_transitions",
        "receipt_does_not_prove": "hardware_or_live_network_or_decoder_correctness_or_universal_fabric_equivalence",
    }


@dataclass(frozen=True)
class Call:
    call_id: str
    source: str
    destination: str
    payload: bytes
    decoder_output_sha256: str
    arrival_tick: int = 0
    hold_ticks: int = 1

    def __post_init__(self) -> None:
        for value in (self.call_id, self.source, self.destination):
            text(value)
        if type(self.payload) is not bytes or len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")
        try:
            validate_sha256(self.decoder_output_sha256)
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc
        integer(self.arrival_tick, MAX_TICK)
        integer(self.hold_ticks, MAX_TICK, 1)
        if self.arrival_tick + self.hold_ticks > MAX_TICK:
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        return seal("lifecycle-call", call_id=self.call_id, source=self.source, destination=self.destination,
                    payload_hex=self.payload.hex(), payload_sha256=hashlib.sha256(self.payload).hexdigest(),
                    decoder_output_sha256=self.decoder_output_sha256,
                    arrival_tick=self.arrival_tick, hold_ticks=self.hold_ticks)

    @classmethod
    def from_dict(cls, value: object) -> Call:
        try:
            bounded_json(value)
            raw = value["payload_hex"]
            if type(raw) is not str or len(raw) > MAX_PAYLOAD_BYTES * 2:
                raise ValueError("INVALID_INPUT")
            result = cls(value["call_id"], value["source"], value["destination"], bytes.fromhex(raw),
                         value["decoder_output_sha256"], value["arrival_tick"], value["hold_ticks"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _endpoints(value: object) -> tuple:
    rows = tuple(sequence(row, 3, 3) for row in sequence(value, 32))
    for row in rows:
        for item in row:
            text(item)
    names = tuple(row[0] for row in rows)
    if names != tuple(sorted(set(names))):
        raise ValueError("INVALID_INPUT")
    return rows


@dataclass(frozen=True)
class FabricAdapter:
    """Registered native boundary; mappings are (logical name, matrix ID, link ID).

    Mapping coordinates must exist at the first/last stage. Busy, unavailable and
    quarantined native links remain explicit admissible initial fault snapshots.
    """
    fabric: CrossbarFabric
    sources: tuple[tuple[str, str, str], ...]
    destinations: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        if type(self.fabric) is not CrossbarFabric:
            raise ValueError("INVALID_INPUT")
        if (sum(len(m.horizontal_links) * len(m.vertical_links)
                for stage in self.fabric.stages for m in stage) > 256
                or len(self.fabric.interstage_links) > 256):
            raise ValueError("INVALID_INPUT")
        for stage in self.fabric.stages:
            for matrix in stage:
                text(matrix.matrix_id)
                for link in (*matrix.horizontal_links, *matrix.vertical_links):
                    text(link.link_id)
        text(self.fabric.fabric_id)
        for wire in self.fabric.interstage_links:
            text(wire.link_id)
        fabric = CrossbarFabric.from_dict(self.fabric.as_dict())
        object.__setattr__(self, "fabric", fabric)
        for name, stage, axis in (("sources", fabric.stages[0], "horizontal_links"),
                                  ("destinations", fabric.stages[-1], "vertical_links")):
            rows = _endpoints(getattr(self, name))
            for _, matrix_id, link_id in rows:
                matrix = next((m for m in stage if m.matrix_id == matrix_id), None)
                if matrix is None or link_id not in {link.link_id for link in getattr(matrix, axis)}:
                    raise ValueError("INVALID_INPUT")
            object.__setattr__(self, name, rows)

    def as_dict(self) -> dict:
        return seal("lifecycle-fabric-adapter", adapter_id=ADAPTER_ID,
                    fabric_manifest=self.fabric.as_dict(),
                    sources=[{"logical": a, "matrix_id": b, "link_id": c} for a, b, c in self.sources],
                    destinations=[{"logical": a, "matrix_id": b, "link_id": c} for a, b, c in self.destinations],
                    operations=["reserve", "release"], initial_owned_reservations=[],
                    authority="untrusted-until-independent-native-replay-with-trusted-pins")

    @classmethod
    def from_dict(cls, value: object) -> FabricAdapter:
        try:
            bounded_json(value)
            result = cls(CrossbarFabric.from_dict(value["fabric_manifest"]),
                         tuple((r["logical"], r["matrix_id"], r["link_id"])
                               for r in sequence(value["sources"], 32)),
                         tuple((r["logical"], r["matrix_id"], r["link_id"])
                               for r in sequence(value["destinations"], 32)))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc

    def execute(self, batch: ContentionBatch, program: dict) -> dict:
        return execute_contention_program(batch, program)


def _targets(program: ProgramManifest, call: Call) -> tuple[str | None, tuple]:
    translated = dict(program.features.forwarding).get(call.destination, call.destination)
    return translated, dict(program.translations.rows).get(translated, (None,))


@dataclass(frozen=True)
class LifecycleInput:
    programme: ProgramManifest
    calls: tuple[Call, ...]
    adapter: FabricAdapter

    def __post_init__(self) -> None:
        for name, cls in (("programme", ProgramManifest), ("adapter", FabricAdapter)):
            item = getattr(self, name)
            if type(item) is not cls:
                raise ValueError("INVALID_INPUT")
            object.__setattr__(self, name, cls.from_dict(item.as_dict()))
        calls = sequence(self.calls, MAX_CALLS)
        if any(type(call) is not Call for call in calls):
            raise ValueError("INVALID_INPUT")
        calls = tuple(Call.from_dict(call.as_dict()) for call in calls)
        names = tuple(call.call_id for call in calls)
        if names != tuple(sorted(set(names))):
            raise ValueError("INVALID_INPUT")
        bound = 0
        for call in calls:
            if call.arrival_tick + sum(self.programme.features.retry_delays) + call.hold_ticks > MAX_TICK:
                raise ValueError("INVALID_INPUT")
            bound += len(_targets(self.programme, call)[1]) * (1 + len(self.programme.features.retry_delays)) * 2
        if bound > MAX_NATIVE_COMMANDS:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "calls", calls)

    def as_dict(self) -> dict:
        calls = seal("lifecycle-call-store", calls=[call.as_dict() for call in self.calls])
        return seal("lifecycle-input", programme=self.programme.as_dict(), call_store=calls,
                    adapter=self.adapter.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> LifecycleInput:
        try:
            bounded_json(value)
            result = cls(ProgramManifest.from_dict(value["programme"]),
                         tuple(Call.from_dict(call) for call in sequence(value["call_store"]["calls"], MAX_CALLS)),
                         FabricAdapter.from_dict(value["adapter"]))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def execute_call_processing(source: LifecycleInput) -> dict:
    """Drain the declared schedule; each connect owns capacity until native release.

    Prefixes are replayed from their immutable base to avoid undocumented mutable
    adapter state. This is the same cumulative ledger, never independent plans.
    """
    if type(source) is not LifecycleInput:
        raise ValueError("INVALID_INPUT")
    source = LifecycleInput.from_dict(source.as_dict())
    snapshot, program = source.as_dict(), source.programme
    call_map = {call.call_id: call for call in source.calls}
    source_map = {a: (b, c) for a, b, c in source.adapter.sources}
    destination_map = {a: (b, c) for a, b, c in source.adapter.destinations}
    queue, schedule, events, commands, semantic, attempts = [], [], [], [], [], []
    active = {}
    native_evidence = None
    root = seal("lifecycle-event-root", input_sha256=snapshot["sha256"])["sha256"]

    def event(state, tick, call, branch, attempt, **details):
        unsigned = {"sequence": len(events), "state": state, "logical_tick": tick,
                    "call_id": call.call_id, "branch": branch, "attempt": attempt,
                    "details": details, "previous_sha256": events[-1]["sha256"] if events else root}
        events.append({**unsigned, "sha256": canonical_sha256(unsigned)})

    def push(tick, kind, call_id, branch, attempt):
        priority = {"release": 0, "retry": 1, "arrival": 2}[kind]
        heapq.heappush(queue, (tick, priority, call_id, branch, attempt, kind))

    def native(command, tick, call, branch, attempt, destination):
        nonlocal native_evidence
        if len(commands) >= MAX_NATIVE_COMMANDS:
            raise ValueError("INVALID_INPUT")
        commands.append(command)
        batch = ContentionBatch(source.adapter.fabric, tuple(commands))
        native_program = compile_contention_program(batch, marker_id=program.marker_id,
                                                    max_search_evaluations=program.max_search_evaluations)
        # Capture pins before passing mutable dictionaries to an untrusted adapter.
        expected_input, expected_program = batch.as_dict()["sha256"], native_program["sha256"]
        raw = source.adapter.execute(batch, native_program)
        bounded_json(raw)
        # Even a hostile in-process adapter must not alter its sealed boundary.
        exact(source.adapter.as_dict(), snapshot["adapter"])
        native_evidence = create_continuity_receipt(
            raw, expected_input_sha256=expected_input, expected_program_sha256=expected_program,
            expected_fabric_sha256=source.adapter.fabric.as_dict()["sha256"])
        result = native_evidence["source_receipt"]["results"][-1]
        record = seal("semantic-fabric-command", sequence=len(semantic), logical_tick=tick,
                      native_tick=command.tick, operation=command.kind, call_id=call.call_id,
                      branch=branch, attempt=attempt, source=call.source, destination=destination,
                      payload_sha256=call.as_dict()["payload_sha256"],
                      decoder_output_sha256=call.decoder_output_sha256)
        semantic.append(record)
        return result, native_evidence

    for call in source.calls:
        for branch, _ in enumerate(_targets(program, call)[1]):
            push(call.arrival_tick, "arrival", call.call_id, branch, 0)
    while queue:
        tick, priority, call_id, branch, attempt, kind = heapq.heappop(queue)
        call = call_map[call_id]
        translated, targets = _targets(program, call)
        destination = targets[branch]
        schedule.append(seal("schedule-dispatch", sequence=len(schedule), logical_tick=tick,
                             fixed_priority=priority, call_id=call_id, branch=branch,
                             attempt=attempt, kind=kind))
        key = (call_id, branch, attempt)
        if kind == "release":
            held = active.pop(key)
            command = ContentionCommand(len(commands), f"release-{len(commands):02d}", "release",
                                        reservation_id=held["reservation_id"],
                                        expected_request_sha256=held["request_sha256"])
            result, evidence = native(command, tick, call, branch, attempt, destination)
            if result["outcome"] != "released":
                raise ValueError("INVALID_INPUT")
            event("release", tick, call, branch, attempt, native_reason=result["reason"],
                  native_command_sha256=command.as_dict()["sha256"],
                  native_receipt_sha256=evidence["source_receipt"]["sha256"],
                  state_after_sha256=result["state_after_sha256"])
            held["result"]["outcome"] = "released"
            held["result"]["release_tick"] = tick
            continue
        event("intake", tick, call, branch, attempt, call_sha256=call.as_dict()["sha256"])
        event("analyse", tick, call, branch, attempt, dialed_destination=call.destination,
              translated_destination=translated, selected_destination=destination,
              translation_table_sha256=program.translations.as_dict()["sha256"],
              feature_module_sha256=program.features.as_dict()["sha256"])
        reason = ("translation_not_found" if destination is None else
                  "adapter_source_not_found" if call.source not in source_map else
                  "adapter_destination_not_found" if destination not in destination_map else None)
        row = {"call_id": call_id, "branch": branch, "attempt": attempt,
               "logical_tick": tick, "destination": destination,
               "payload_sha256": call.as_dict()["payload_sha256"],
               "decoder_output_sha256": call.decoder_output_sha256,
               "outcome": "rejected", "native_reason": None, "reason": reason,
               "native_request_sha256": None, "reservation_sha256": None,
               "path_plan": None, "continuity_witness_sha256": None, "release_tick": None}
        attempts.append(row)
        if reason is not None:
            event("reject", tick, call, branch, attempt, reason=reason, native_result_present=False)
            continue
        source_matrix, horizontal = source_map[call.source]
        destination_matrix, vertical = destination_map[destination]
        request = MultiStageRequest(source_matrix, destination_matrix, CrossbarRequest(
            f"{call_id}/branch-{branch}/attempt-{attempt}", horizontal, vertical,
            call.payload, call.decoder_output_sha256))
        event("select", tick, call, branch, attempt, native_request_sha256=request.as_dict()["sha256"])
        command = ContentionCommand(len(commands), f"reserve-{len(commands):02d}", "reserve", request=request)
        result, evidence = native(command, tick, call, branch, attempt, destination)
        row["native_request_sha256"] = request.as_dict()["sha256"]
        row["native_reason"] = row["reason"] = result["reason"]
        if result["outcome"] == "rejected":
            event("reject", tick, call, branch, attempt, reason=result["reason"], native_result_present=True,
                  native_command_sha256=command.as_dict()["sha256"],
                  native_receipt_sha256=evidence["source_receipt"]["sha256"],
                  blocked_resources=result["blocked_resources"])
            if attempt < len(program.features.retry_delays):
                push(tick + program.features.retry_delays[attempt], "retry", call_id, branch, attempt + 1)
            continue
        if result["outcome"] != "reserved":
            raise ValueError("INVALID_INPUT")
        witness = evidence["routes"][-1]["witness"]
        if witness is None:
            raise ValueError("INVALID_INPUT")
        reservation = result["reservation"]
        event("connect", tick, call, branch, attempt, reservation_sha256=reservation["sha256"],
              native_command_sha256=command.as_dict()["sha256"],
              native_receipt_sha256=evidence["source_receipt"]["sha256"],
              state_after_sha256=result["state_after_sha256"])
        event("verify", tick, call, branch, attempt, continuity_witness_sha256=witness["sha256"],
              payload_sha256=witness["payload_sha256"], decoder_output_sha256=witness["decoder_output_sha256"])
        row.update(outcome="verified", reservation_sha256=reservation["sha256"],
                   path_plan=witness["path_plan"], continuity_witness_sha256=witness["sha256"])
        active[key] = {"reservation_id": command.command_id, "request_sha256": request.as_dict()["sha256"],
                       "result": row}
        push(tick + call.hold_ticks, "release", call_id, branch, attempt)
    if active or (native_evidence is not None and
                  native_evidence["source_receipt"]["final_state"]["active_reservations"]):
        raise ValueError("INVALID_INPUT")
    results = []
    for call in source.calls:
        rows = [row for row in attempts if row["call_id"] == call.call_id]
        branches = len(_targets(program, call)[1])
        released = sum(row["outcome"] == "released" for row in rows)
        results.append({"call_id": call.call_id, "outcome": "released" if released == branches else
                        "partially_released" if released else "rejected", "branch_count": branches,
                        "released_branch_count": released, "rejected_attempt_count": sum(
                            row["outcome"] == "rejected" for row in rows), "attempts": rows})
    schedule_receipt = seal("event-schedule-receipt", input_sha256=snapshot["sha256"],
                            routing_logic_sha256=routing_logic()["sha256"],
                            ordering=routing_logic()["scheduler_order"],
                            priorities=routing_logic()["fixed_priority"], dispatches=schedule,
                            logical_ticks_only=True, queue_drained=True)
    stream = seal("lifecycle-command-stream", input_sha256=snapshot["sha256"],
                  programme_sha256=program.as_dict()["sha256"], adapter_sha256=source.adapter.as_dict()["sha256"],
                  commands=semantic)
    receipt = seal("call-processing-receipt", input=snapshot, event_root_sha256=root, events=events,
                schedule=schedule_receipt, command_stream=stream, calls=results,
                native_continuity_receipt=native_evidence, reservations_active_at_end=0,
                connected_attempt_count=sum(row["outcome"] == "released" for row in attempts),
                rejected_attempt_count=sum(row["outcome"] == "rejected" for row in attempts),
                native_evidence_present=native_evidence is not None,
                claim_boundary=claim_boundary())
    bounded_json(receipt)
    return receipt


def validate_call_processing_receipt(receipt: object, *, expected_input_sha256=None,
                                     expected_programme_sha256=None, expected_adapter_sha256=None,
                                     expected_call_store_sha256=None) -> dict:
    try:
        bounded_json(receipt)
        source = LifecycleInput.from_dict(receipt["input"])
        snapshot = source.as_dict()
        pins = bindings({"input": (expected_input_sha256, snapshot["sha256"]),
                         "programme": (expected_programme_sha256, snapshot["programme"]["sha256"]),
                         "adapter": (expected_adapter_sha256, snapshot["adapter"]["sha256"]),
                         "call_store": (expected_call_store_sha256, snapshot["call_store"]["sha256"])})
        replay = execute_call_processing(source)
        exact(receipt, replay)
        return seal("call-processing-validation", ess_call_processing_receipt_hash=replay["sha256"],
                    ess_program_store_manifest_hash=snapshot["programme"]["sha256"],
                    complete_replay_verified=True, native_evidence_present=replay["native_evidence_present"],
                    nonempty_connection_evidence=replay["connected_attempt_count"] > 0,
                    all_reservations_released=True, external_bindings_verified=pins, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _semantics(receipt: dict) -> dict:
    return seal("fabric-semantic-observation", commands=receipt["command_stream"]["commands"],
                calls=[{"call_id": call["call_id"], "outcome": call["outcome"],
                        "attempts": [{key: row[key] for key in (
                            "branch", "attempt", "logical_tick", "destination", "payload_sha256",
                            "decoder_output_sha256", "outcome", "native_reason", "reason", "release_tick")}
                                     for row in call["attempts"]]} for call in receipt["calls"]])


def create_fabric_equivalence_receipt(left: LifecycleInput, right: LifecycleInput) -> dict:
    """Compare actual semantics for exactly two independently validated boundaries.

    Differences are valid evidence with equivalent=False. There is no assertion
    about other adapters, unseen calls, or equal internal resource capacities.
    """
    if type(left) is not LifecycleInput or type(right) is not LifecycleInput:
        raise ValueError("INVALID_INPUT")
    a, b = left.as_dict(), right.as_dict()
    if a["programme"] != b["programme"] or a["call_store"] != b["call_store"]:
        raise ValueError("INVALID_INPUT")
    if a["adapter"]["sha256"] == b["adapter"]["sha256"]:
        raise ValueError("INVALID_INPUT")
    receipts, observations = [], []
    for source in (left, right):
        receipt = execute_call_processing(source)
        validate_call_processing_receipt(receipt, expected_input_sha256=source.as_dict()["sha256"])
        receipts.append(receipt)
        observations.append(_semantics(receipt))
    nonempty = bool(left.calls)
    equal = observations[0] == observations[1]
    receipt = seal("fabric-equivalence-receipt", programme_sha256=a["programme"]["sha256"],
                call_store_sha256=a["call_store"]["sha256"], receipts=receipts, observations=observations,
                semantic_command_stream_equal=observations[0]["commands"] == observations[1]["commands"],
                equivalent=nonempty and equal, nonempty_case_coverage=nonempty,
                scope="exact-declared-calls-and-two-crossbar-native-reservation-adapters",
                supports_universal_equivalence=False, native_failures_preserved=True)
    bounded_json(receipt)
    return receipt


def validate_fabric_equivalence_receipt(receipt: object, *, expected_left_input_sha256=None,
                                        expected_right_input_sha256=None,
                                        expected_programme_sha256=None) -> dict:
    try:
        bounded_json(receipt)
        rows = sequence(receipt["receipts"], 2, 2)
        left, right = (LifecycleInput.from_dict(row["input"]) for row in rows)
        pins = bindings({"left_input": (expected_left_input_sha256, left.as_dict()["sha256"]),
                         "right_input": (expected_right_input_sha256, right.as_dict()["sha256"]),
                         "programme": (expected_programme_sha256, left.programme.as_dict()["sha256"])})
        replay = create_fabric_equivalence_receipt(left, right)
        exact(receipt, replay)
        return seal("fabric-equivalence-validation", ess_fabric_equivalence_receipt_hash=replay["sha256"],
                    equivalent=replay["equivalent"], complete_replay_verified=True,
                    external_bindings_verified=pins, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def demo_lifecycle_input(*, alternate_fabric: bool = False) -> LifecycleInput:
    if type(alternate_fabric) is not bool:
        raise ValueError("INVALID_INPUT")
    fabric = demo_fabric()
    if alternate_fabric:
        fabric = CrossbarFabric("ess-two-stage", ((demo_matrix("ingress", horizontal_count=2, vertical_count=2),),
                                                   (demo_matrix("egress", horizontal_count=2, vertical_count=2),)),
                                (InterstageLink("direct", "ingress", "V000", "egress", "H000"),))
    adapter = FabricAdapter(fabric, (("in", "ingress", "H000"),), (("out", "egress", "V001"),))
    program = ProgramManifest(TranslationTable((("dial", ("out",)),)), FeatureModules(retry_delays=(2,)))
    calls = (Call("call-a", "in", "dial", b"opaque-correction", "a" * 64, 0, 2),
             Call("call-b", "in", "dial", b"opaque-correction", "a" * 64, 0, 1))
    return LifecycleInput(program, calls, adapter)


def validate_event_schedule_receipt(value: object, source: LifecycleInput, *, expected_input_sha256=None) -> dict:
    """Schedule validation includes native outcomes that determine retry/release events."""
    try:
        bounded_json(value)
        if type(source) is not LifecycleInput:
            raise ValueError("INVALID_INPUT")
        pins = bindings({"input": (expected_input_sha256, source.as_dict()["sha256"])})
        replay = execute_call_processing(source)["schedule"]
        exact(value, replay)
        return seal("event-schedule-validation", ess_event_schedule_receipt_hash=replay["sha256"],
                    complete_replay_verified=True, external_bindings_verified=pins, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
