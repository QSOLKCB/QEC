# SPDX-License-Identifier: MPL-2.0
"""v174.1 owned logical allocation, saturation, collision and release replay."""
from __future__ import annotations

from dataclasses import dataclass

from ._common import MAX_COMMANDS, MAX_SLOTS, MAX_TICK, digest, exact, integer, pins, seal, sequence, text

ALLOCATION_POLICY = "logical-tick-release-before-allocate-request-id-command-id-first-free.v1"


@dataclass(frozen=True)
class SlotState:
    slot: int
    state: str = "idle"
    owner_id: str | None = None
    owner_sha256: str | None = None

    def __post_init__(self) -> None:
        integer(self.slot, MAX_SLOTS - 1)
        if type(self.state) is not str or self.state not in ("idle", "busy", "quarantined"):
            raise ValueError("INVALID_INPUT")
        if self.state == "busy":
            text(self.owner_id)
            digest(self.owner_sha256)
        elif self.owner_id is not None or self.owner_sha256 is not None:
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        return {"slot": self.slot, "state": self.state, "owner_id": self.owner_id,
                "owner_sha256": self.owner_sha256}

    @classmethod
    def from_dict(cls, value: object) -> SlotState:
        try:
            result = cls(**value)
            exact(value, result.as_dict())
            return result
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class SlotCommand:
    tick: int
    command_id: str
    kind: str
    request_id: str
    request_sha256: str
    slot: int | None = None

    def __post_init__(self) -> None:
        integer(self.tick, MAX_TICK)
        text(self.command_id)
        text(self.request_id)
        digest(self.request_sha256)
        if type(self.kind) is not str or self.kind not in ("release", "allocate"):
            raise ValueError("INVALID_INPUT")
        if self.slot is not None:
            integer(self.slot, MAX_SLOTS - 1)
        elif self.kind == "release":
            raise ValueError("INVALID_INPUT")

    @property
    def order_key(self) -> tuple:
        return (self.tick, ("release", "allocate").index(self.kind), self.request_id, self.command_id)

    def as_dict(self) -> dict:
        return seal("slot-command", "174.1", tick=self.tick, command_id=self.command_id,
                    kind=self.kind, request_id=self.request_id, request_sha256=self.request_sha256,
                    slot=self.slot)

    @classmethod
    def from_dict(cls, value: object) -> SlotCommand:
        try:
            result = cls(*(value[key] for key in ("tick", "command_id", "kind", "request_id", "request_sha256", "slot")))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class AllocationInput:
    initial_state: tuple[SlotState, ...]
    commands: tuple[SlotCommand, ...]

    def __post_init__(self) -> None:
        states = sequence(self.initial_state, MAX_SLOTS)
        commands = sequence(self.commands, MAX_COMMANDS)
        if not states or any(type(x) is not SlotState for x in states) or any(type(x) is not SlotCommand for x in commands):
            raise ValueError("INVALID_INPUT")
        states = tuple(SlotState.from_dict(x.as_dict()) for x in states)
        commands = tuple(SlotCommand.from_dict(x.as_dict()) for x in commands)
        if tuple(x.slot for x in states) != tuple(range(len(states))):
            raise ValueError("INVALID_INPUT")
        if (tuple(x.order_key for x in commands) != tuple(sorted(x.order_key for x in commands))
                or len({x.command_id for x in commands}) != len(commands)
                or any(x.slot is not None and x.slot >= len(states) for x in commands)):
            raise ValueError("INVALID_INPUT")
        # One reservation per declared owner; arbitrary initial capacity is never inferred.
        owners = [(x.owner_id, x.owner_sha256) for x in states if x.state == "busy"]
        if len(owners) != len({owner[0] for owner in owners}):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "initial_state", states)
        object.__setattr__(self, "commands", commands)

    def as_dict(self) -> dict:
        return seal("allocation-input", "174.1", policy=ALLOCATION_POLICY,
                    initial_state=[x.as_dict() for x in self.initial_state],
                    commands=[x.as_dict() for x in self.commands])

    @classmethod
    def from_dict(cls, value: object) -> AllocationInput:
        try:
            result = cls(tuple(SlotState.from_dict(x) for x in sequence(value["initial_state"], MAX_SLOTS)),
                         tuple(SlotCommand.from_dict(x) for x in sequence(value["commands"], MAX_COMMANDS)))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def execute_slot_allocation(source: AllocationInput) -> dict:
    if type(source) is not AllocationInput:
        raise ValueError("INVALID_INPUT")
    source = AllocationInput.from_dict(source.as_dict())
    state = list(source.initial_state)
    events = []
    for command in source.commands:
        before = [x.as_dict() for x in state]
        selected = None
        accepted = False
        if command.kind == "release":
            current = state[command.slot]
            if (current.state, current.owner_id, current.owner_sha256) == ("busy", command.request_id, command.request_sha256):
                selected, accepted, reason = command.slot, True, "released"
                state[selected] = SlotState(selected)
            else:
                reason = "ownership_mismatch"
        else:
            existing = [x for x in state if x.state == "busy" and x.owner_id == command.request_id]
            if existing:
                reason = "already_allocated" if existing[0].owner_sha256 == command.request_sha256 else "owner_identity_conflict"
            elif command.slot is not None and state[command.slot].state != "idle":
                reason = "slot_collision" if state[command.slot].state == "busy" else "slot_quarantined"
            else:
                free = [x.slot for x in state if x.state == "idle"]
                if not free:
                    reason = "no_capacity"
                else:
                    selected = command.slot if command.slot is not None else free[0]
                    accepted, reason = True, "allocated"
                    state[selected] = SlotState(selected, "busy", command.request_id, command.request_sha256)
        events.append(seal("allocation-event", "174.1", sequence=len(events), command=command.as_dict(),
                           before=before, accepted=accepted, reason=reason, selected_slot=selected,
                           after=[x.as_dict() for x in state]))
    return seal("timeslot-allocation-receipt", "174.1", input=source.as_dict(), events=events,
                final_state=[x.as_dict() for x in state], policy=ALLOCATION_POLICY,
                successful_allocations=sum(e["reason"] == "allocated" for e in events),
                successful_releases=sum(e["reason"] == "released" for e in events),
                ownership_is_not_authentication=True)


def validate_allocation_receipt(receipt: object, *, expected_input_sha256: str | None = None) -> dict:
    try:
        source = AllocationInput.from_dict(receipt["input"])
        bindings = pins(input=(expected_input_sha256, source.as_dict()["sha256"]))
        replay = execute_slot_allocation(source)
        exact(receipt, replay)
        return seal("allocation-validation", "174.1", deterministic_timeslot_allocation_receipt_hash=replay["sha256"],
                    complete_replay_verified=True, external_bindings_verified=bindings, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
