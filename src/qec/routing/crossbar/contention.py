# SPDX-License-Identifier: MPL-2.0
"""v172.3 bounded logical contention over immutable v172.2 fabrics.

Canonical commands drive an owned, in-memory reservation ledger. Each request
uses the published selector against the current effective snapshot. Whole paths
are acquired or revoked atomically; replay derives every state and decision.
This is a closed software-model batch, not a concurrent physical switch service.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType

from qec.sonify.canonical import canonical_sha256, validate_sha256
from .core import CrossbarMatrix
from .marker import CrossbarRequest, _exact, _text
from .multistage import (
    CrossbarFabric, MultiStageRequest, compile_path_program, demo_fabric,
    execute_path_program,
)

CONTENTION_CONTRACT_VERSION = "172.3"
MAX_COMMANDS = 32
MAX_CONTENTION_INTERSECTIONS = 4096
MAX_CONTENTION_RESOURCES = 4096
MAX_BATCH_PAYLOAD_BYTES = 1_048_576
MAX_BATCH_SEARCH_EVALUATIONS = 65536
COMMAND_KINDS = ("release", "quarantine", "reserve")
CONTENTION_POLICY = "logical-tick-release-quarantine-request-hash.v1"
CONTENTION_CLAIM_BOUNDARY = MappingProxyType({
    "classical_software_model_only": True,
    "reservation_present": True,
    "atomic_whole_path_reservation": True,
    "quarantine_revokes_whole_reservation": True,
    "connection_commit_present": False,
    "physical_actuation_present": False,
    "continuity_receipt_present": False,
    "payload_mutation_permitted": False,
    "decoder_output_mutation_permitted": False,
    "ownership_identity_is_not_authentication": True,
    "receipt_proves": "bounded_canonical_contention_and_owned_resource_transitions",
    "receipt_does_not_prove": "external_provenance_or_concurrent_physical_switch_safety",
})


def _seal(name: str, **fields: object) -> dict[str, object]:
    unsigned = {"schema": "qec.crossbar-" + name + ".v1",
                "contract_version": CONTENTION_CONTRACT_VERSION, **fields}
    return {**unsigned, "sha256": canonical_sha256(unsigned)}


def _integer(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


@dataclass(frozen=True)
class CrossbarResource:
    """An axis link qualified by matrix, or a fabric-wide interstage link."""
    kind: str
    matrix_id: str | None
    link_id: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in ("horizontal", "vertical", "interstage"):
            raise ValueError("INVALID_INPUT")
        _text(self.link_id)
        if self.kind == "interstage":
            if self.matrix_id is not None:
                raise ValueError("INVALID_INPUT")
        else:
            _text(self.matrix_id)

    def as_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "matrix_id": self.matrix_id, "link_id": self.link_id}

    @property
    def resource_id(self) -> str:
        return _seal("resource-id", **self.as_dict())["sha256"]

    @classmethod
    def from_dict(cls, value: object) -> CrossbarResource:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            resource = cls(**value)
            _exact(value, resource.as_dict())
            return resource
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class ContentionCommand:
    """Exactly one reserve, owner-bound release, or quarantine at a logical tick."""
    tick: int
    command_id: str
    kind: str
    request: MultiStageRequest | None = None
    reservation_id: str | None = None
    expected_request_sha256: str | None = None
    resource: CrossbarResource | None = None

    def __post_init__(self) -> None:
        _integer(self.tick, 2**31 - 1)
        _text(self.command_id)
        if type(self.kind) is not str or self.kind not in COMMAND_KINDS:
            raise ValueError("INVALID_INPUT")
        if self.kind == "reserve":
            if type(self.request) is not MultiStageRequest:
                raise ValueError("INVALID_INPUT")
            self.request.__post_init__()
            if any(x is not None for x in (self.reservation_id, self.expected_request_sha256, self.resource)):
                raise ValueError("INVALID_INPUT")
        elif self.kind == "release":
            _text(self.reservation_id)
            try:
                validate_sha256(self.expected_request_sha256)
            except (TypeError, ValueError) as exc:
                raise ValueError("INVALID_INPUT") from exc
            if self.request is not None or self.resource is not None:
                raise ValueError("INVALID_INPUT")
        else:
            if type(self.resource) is not CrossbarResource:
                raise ValueError("INVALID_INPUT")
            self.resource.__post_init__()
            if any(x is not None for x in (self.request, self.reservation_id, self.expected_request_sha256)):
                raise ValueError("INVALID_INPUT")

    @property
    def order_key(self) -> tuple:
        # This is a traversal key, never a hashed identity tuple.
        request_hash = self.request.as_dict()["sha256"] if self.request is not None else ""
        return (self.tick, COMMAND_KINDS.index(self.kind), request_hash, self.command_id)

    def as_dict(self) -> dict[str, object]:
        return _seal("contention-command", tick=self.tick, command_id=self.command_id,
                     kind=self.kind, request=None if self.request is None else self.request.as_dict(),
                     reservation_id=self.reservation_id, expected_request_sha256=self.expected_request_sha256,
                     resource=None if self.resource is None else self.resource.as_dict())

    @classmethod
    def from_dict(cls, value: object) -> ContentionCommand:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            command = cls(value["tick"], value["command_id"], value["kind"],
                          None if value["request"] is None else MultiStageRequest.from_dict(value["request"]),
                          value["reservation_id"], value["expected_request_sha256"],
                          None if value["resource"] is None else CrossbarResource.from_dict(value["resource"]))
            _exact(value, command.as_dict())
            return command
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _catalogue(fabric: CrossbarFabric) -> list[tuple[CrossbarResource, str]]:
    return [(CrossbarResource(link.axis, matrix.matrix_id, link.link_id), link.state)
            for stage in fabric.stages for matrix in stage
            for link in (*matrix.horizontal_links, *matrix.vertical_links)] + [
                (CrossbarResource("interstage", None, wire.link_id), wire.state)
                for wire in fabric.interstage_links]


@dataclass(frozen=True)
class ContentionBatch:
    fabric: CrossbarFabric
    commands: tuple[ContentionCommand, ...]

    def __post_init__(self) -> None:
        if type(self.fabric) is not CrossbarFabric or type(self.commands) not in (list, tuple):
            raise ValueError("INVALID_INPUT")
        if not 1 <= len(self.commands) <= MAX_COMMANDS:
            raise ValueError("INVALID_INPUT")
        if sum(len(m.horizontal_links) * len(m.vertical_links)
               for stage in self.fabric.stages for m in stage) > MAX_CONTENTION_INTERSECTIONS:
            raise ValueError("INVALID_INPUT")
        if len(_catalogue(self.fabric)) > MAX_CONTENTION_RESOURCES:
            raise ValueError("INVALID_INPUT")
        commands = tuple(self.commands)
        if any(type(c) is not ContentionCommand for c in commands):
            raise ValueError("INVALID_INPUT")
        for command in commands:
            command.__post_init__()
        if sum(len(c.request.request.payload) for c in commands if c.request is not None) > MAX_BATCH_PAYLOAD_BYTES:
            raise ValueError("INVALID_INPUT")
        if len({c.command_id for c in commands}) != len(commands):
            raise ValueError("INVALID_INPUT")
        if tuple(c.order_key for c in commands) != tuple(sorted(c.order_key for c in commands)):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "fabric", CrossbarFabric.from_dict(self.fabric.as_dict()))
        object.__setattr__(self, "commands", tuple(ContentionCommand.from_dict(c.as_dict()) for c in commands))

    def as_dict(self) -> dict[str, object]:
        return _seal("contention-input", fabric_manifest=self.fabric.as_dict(),
                     commands=[c.as_dict() for c in self.commands])

    @classmethod
    def from_dict(cls, value: object) -> ContentionBatch:
        if not isinstance(value, dict):
            raise ValueError("INVALID_INPUT")
        try:
            commands = value["commands"]
            if type(commands) is not list or not 1 <= len(commands) <= MAX_COMMANDS:
                raise ValueError("INVALID_INPUT")
            batch = cls(CrossbarFabric.from_dict(value["fabric_manifest"]),
                        tuple(ContentionCommand.from_dict(c) for c in commands))
            _exact(value, batch.as_dict())
            return batch
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def compile_contention_program(batch: ContentionBatch, *, marker_id: str = "marker-0",
                               max_search_evaluations: int = MAX_BATCH_SEARCH_EVALUATIONS) -> dict[str, object]:
    if type(batch) is not ContentionBatch:
        raise ValueError("INVALID_INPUT")
    batch.__post_init__()
    _text(marker_id)
    _integer(max_search_evaluations, MAX_BATCH_SEARCH_EVALUATIONS, 1)
    return _seal("contention-program", input_sha256=batch.as_dict()["sha256"],
                 marker_id=marker_id, policy=CONTENTION_POLICY,
                 max_search_evaluations=max_search_evaluations,
                 claim_boundary=dict(CONTENTION_CLAIM_BOUNDARY))


def execute_contention_program(batch: ContentionBatch, program: object) -> dict[str, object]:
    """Execute one closed batch from an empty ownership ledger; never import state claims."""
    if not isinstance(program, dict):
        raise ValueError("INVALID_INPUT")
    expected = compile_contention_program(batch, marker_id=program.get("marker_id"),
                                         max_search_evaluations=program.get("max_search_evaluations"))
    _exact(program, expected)
    input_record = batch.as_dict()
    catalogue = _catalogue(batch.fabric)
    resources = {r.resource_id: (r, state) for r, state in catalogue}
    resource_order = tuple(r.resource_id for r, _ in catalogue)
    owners = {}
    active = {}
    quarantined = set()
    events = []
    results = []
    evaluations = 0
    root = _seal("contention-event-root", input_sha256=input_record["sha256"],
                 program_sha256=expected["sha256"])["sha256"]

    def event(kind, **details):
        unsigned = {"sequence": len(events), "kind": kind, "details": details,
                    "previous_sha256": events[-1]["sha256"] if events else root}
        events.append({**unsigned, "sha256": canonical_sha256(unsigned)})

    def state_of(resource_id):
        if resource_id in quarantined:
            return "quarantined"
        return "busy" if resource_id in owners else resources[resource_id][1]

    def state():
        return _seal("reservation-state", base_fabric_sha256=input_record["fabric_manifest"]["sha256"],
                     resources=[{"resource": r.as_dict(), "resource_id": rid,
                                 "state": state_of(rid), "reservation_id": owners.get(rid)}
                                for rid in resource_order for r in (resources[rid][0],)],
                     active_reservations=[active[key] for key in sorted(active)])

    def effective_fabric():
        def axis(matrix, links):
            return tuple(replace(link, state=state_of(CrossbarResource(
                link.axis, matrix.matrix_id, link.link_id).resource_id)) for link in links)
        return CrossbarFabric(batch.fabric.fabric_id,
            tuple(tuple(CrossbarMatrix(m.matrix_id, axis(m, m.horizontal_links), axis(m, m.vertical_links))
                        for m in stage) for stage in batch.fabric.stages),
            tuple(replace(w, state=state_of(CrossbarResource("interstage", None, w.link_id).resource_id))
                  for w in batch.fabric.interstage_links)).as_dict()

    def drop(reservation_id):
        reservation = active.pop(reservation_id)
        for rid in reservation["resource_ids"]:
            del owners[rid]
        return reservation

    initial_state = state()
    event("input_sealed", input_sha256=input_record["sha256"])
    event("program_verified", program_sha256=expected["sha256"])
    for command in batch.commands:
        before = state()["sha256"]
        details = {"command_id": command.command_id, "command_sha256": command.as_dict()["sha256"],
                   "tick": command.tick, "kind": command.kind, "state_before_sha256": before,
                   "path_search_receipt": None, "blocked_resources": [],
                   "reservation": None, "revoked_reservation": None}
        event("command_started", command_id=command.command_id, tick=command.tick,
              state_before_sha256=before)
        if command.kind == "reserve":
            details["blocked_resources"] = [
                {"resource": r.as_dict(), "resource_id": rid, "state": state_of(rid),
                 "reservation_id": owners.get(rid),
                 "owner_request_sha256": active[owners[rid]]["request_sha256"] if rid in owners else None}
                for rid in resource_order for r in (resources[rid][0],) if state_of(rid) != "idle"]
            remaining = expected["max_search_evaluations"] - evaluations
            if remaining == 0:
                outcome, reason = "rejected", "batch_search_budget_exhausted"
            else:
                manifest = effective_fabric()
                search_program = compile_path_program(manifest, command.request,
                    marker_id=expected["marker_id"], max_search_evaluations=remaining)
                search = execute_path_program(manifest, command.request, search_program)
                details["path_search_receipt"] = search
                evaluations += search["search_evaluations"]
                if search["outcome"] == "rejected":
                    outcome, reason = "rejected", search["reason"]
                else:
                    chosen = set()
                    for coordinate in search["path_plan"]["coordinates"]:
                        for axis in ("horizontal", "vertical"):
                            chosen.add(CrossbarResource(axis, coordinate["matrix_id"],
                                coordinate["coordinate"][axis + "_link_id"]).resource_id)
                    chosen.update(CrossbarResource("interstage", None, link_id).resource_id
                                  for link_id in search["path_plan"]["interstage_link_ids"])
                    ordered = [rid for rid in resource_order if rid in chosen]
                    if len(ordered) != len(chosen) or any(state_of(rid) != "idle" for rid in ordered):
                        raise ValueError("INVALID_INPUT")
                    reservation = _seal("reservation", reservation_id=command.command_id,
                        request_sha256=command.request.as_dict()["sha256"],
                        command_sha256=command.as_dict()["sha256"],
                        path_search_receipt_sha256=search["sha256"], resource_ids=ordered,
                        path_plan=search["path_plan"])
                    active[command.command_id] = reservation
                    for rid in ordered:
                        owners[rid] = command.command_id
                    details["reservation"] = reservation
                    outcome, reason = "reserved", "complete_path_reserved"
                    event("path_reserved", reservation=reservation)
        elif command.kind == "release":
            reservation = active.get(command.reservation_id)
            if reservation is None:
                outcome, reason = "rejected", "reservation_not_active"
            elif reservation["request_sha256"] != command.expected_request_sha256:
                outcome, reason = "rejected", "reservation_owner_mismatch"
            else:
                details["reservation"] = drop(command.reservation_id)
                outcome, reason = "released", "owned_path_released"
                event("path_released", reservation=details["reservation"])
        else:
            rid = command.resource.resource_id
            if rid not in resources:
                outcome, reason = "rejected", "unknown_resource"
            elif state_of(rid) == "quarantined":
                outcome, reason = "rejected", "resource_already_quarantined"
            elif state_of(rid) == "unavailable":
                outcome, reason = "rejected", "resource_unavailable"
            else:
                if rid in owners:
                    details["revoked_reservation"] = drop(owners[rid])
                    event("reservation_revoked", reservation=details["revoked_reservation"], resource_id=rid)
                quarantined.add(rid)
                outcome, reason = "quarantined", "resource_quarantined"
                event("resource_quarantined", resource=command.resource.as_dict(), resource_id=rid)
        result = {**details, "outcome": outcome, "reason": reason,
                  "state_after_sha256": state()["sha256"], "marker_released": True}
        results.append(result)
        event("command_completed", command_id=command.command_id, outcome=outcome,
              reason=reason, result_sha256=canonical_sha256(result),
              state_after_sha256=result["state_after_sha256"])
        event("marker_released", marker_id=expected["marker_id"], command_id=command.command_id)
    return _seal("contention-receipt", input=input_record, program=expected,
                 event_root_sha256=root, initial_state=initial_state, results=results,
                 events=events, final_state=state(), search_evaluations=evaluations,
                 claim_boundary=dict(CONTENTION_CLAIM_BOUNDARY))


def validate_contention_receipt(receipt: object, *, expected_input_sha256: str | None = None,
                                 expected_program_sha256: str | None = None,
                                 expected_fabric_sha256: str | None = None) -> dict[str, object]:
    if not isinstance(receipt, dict):
        raise ValueError("INVALID_INPUT")
    try:
        batch = ContentionBatch.from_dict(receipt["input"])
        replay = execute_contention_program(batch, receipt["program"])
        _exact(receipt, replay)
        bindings = {"input": (expected_input_sha256, replay["input"]["sha256"]),
                    "program": (expected_program_sha256, replay["program"]["sha256"]),
                    "fabric": (expected_fabric_sha256, replay["input"]["fabric_manifest"]["sha256"])}
        for expected, actual in bindings.values():
            if expected is not None and validate_sha256(expected) != actual:
                raise ValueError("INVALID_INPUT")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    return _seal("contention-validation", crossbar_contention_receipt_hash=replay["sha256"],
                 input_sha256=replay["input"]["sha256"], program_sha256=replay["program"]["sha256"],
                 final_state_sha256=replay["final_state"]["sha256"],
                 replay_verified=True, atomic_reservations_verified=True,
                 release_ownership_verified=True, quarantine_verified=True,
                 marker_release_verified=True, all_passed=True,
                 external_bindings_verified={key: expected is not None for key, (expected, _) in bindings.items()})


def demo_contention_batch() -> ContentionBatch:
    """Two contenders, release/retry, quarantine revocation, then a blocked retry."""
    request = MultiStageRequest("ingress", "egress", CrossbarRequest(
        "contention-demo", "H000", "V001", b"opaque-correction", "a" * 64))
    commands = (
        ContentionCommand(0, "a", "reserve", request=request),
        ContentionCommand(0, "b", "reserve", request=request),
        ContentionCommand(1, "release-a", "release", reservation_id="a",
                          expected_request_sha256=request.as_dict()["sha256"]),
        ContentionCommand(1, "retry", "reserve", request=request),
        ContentionCommand(2, "quarantine", "quarantine", resource=CrossbarResource("horizontal", "ingress", "H000")),
        ContentionCommand(2, "retry-quarantined", "reserve", request=request),
    )
    return ContentionBatch(demo_fabric(), commands)
