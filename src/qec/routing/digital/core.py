# SPDX-License-Identifier: MPL-2.0
"""Closed, bounded digital transport with replayed allocation and reassembly.

These additive v174.0–v174.5 contracts are implemented but unreleased. They do
not change any released switch, decoder, source version or package version.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from ._common import (CLAIM_BOUNDARY, IDLE_SYMBOL, MAX_FRAMES, MAX_SLOTS, MAX_TICK,
                      digest, exact, hex_bytes, integer, payload_hash, pins, seal, sequence, text)
from .allocation import AllocationInput, SlotCommand, SlotState, execute_slot_allocation
from .frames import (HEADER_BYTES, DigitalFrame, FrameObservation, FrameSchema, encode_frames,
                     interchange_frames, synchronize_frames)

MAX_FAULTS = 16
FAULT_KINDS = ("bit_flip", "frame_loss", "duplicate_frame", "reorder_frames", "alignment_slip", "slot_collision")


@dataclass(frozen=True)
class DigitalRequest:
    request_id: str
    source: str
    destination: str
    payload: bytes
    decoder_output_sha256: str

    def __post_init__(self) -> None:
        for value in (self.request_id, self.source, self.destination):
            text(value)
        if type(self.payload) is not bytes or len(self.payload) > MAX_FRAMES:
            raise ValueError("INVALID_INPUT")
        digest(self.decoder_output_sha256)

    def as_dict(self) -> dict:
        return seal("request", "174.0", request_id=self.request_id, source=self.source,
                    destination=self.destination, payload_hex=self.payload.hex(),
                    payload_sha256=payload_hash(self.payload), payload_length=len(self.payload),
                    decoder_output_sha256=self.decoder_output_sha256)

    @classmethod
    def from_dict(cls, value: object) -> DigitalRequest:
        try:
            result = cls(value["request_id"], value["source"], value["destination"],
                         hex_bytes(value["payload_hex"], MAX_FRAMES), value["decoder_output_sha256"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class FrameFault:
    kind: str
    frame_index: int
    slot: int | None = None
    bit: int = 0

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in FAULT_KINDS:
            raise ValueError("INVALID_INPUT")
        integer(self.frame_index, MAX_FRAMES - 1)
        integer(self.bit, 15)
        if self.kind in ("bit_flip", "slot_collision"):
            integer(self.slot, MAX_SLOTS - 1)
        elif self.slot is not None:
            raise ValueError("INVALID_INPUT")
        if self.kind != "bit_flip" and self.bit != 0:
            raise ValueError("INVALID_INPUT")

    def as_dict(self) -> dict:
        return seal("frame-fault", "174.4", kind=self.kind, frame_index=self.frame_index,
                    slot=self.slot, bit=self.bit)

    @classmethod
    def from_dict(cls, value: object) -> FrameFault:
        try:
            result = cls(*(value[key] for key in ("kind", "frame_index", "slot", "bit")))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _slots(value: object, count: int) -> tuple[int, ...]:
    values = sequence(value, MAX_SLOTS)
    for slot in values:
        integer(slot, count - 1)
    if values != tuple(sorted(set(values))):
        raise ValueError("INVALID_INPUT")
    return values


def _endpoints(value: object) -> tuple[str, ...]:
    values = sequence(value, 64)
    for item in values:
        text(item)
    if not values or values != tuple(sorted(set(values))):
        raise ValueError("INVALID_INPUT")
    return values


@dataclass(frozen=True)
class DigitalSwitchInput:
    request: DigitalRequest
    frame_schema: FrameSchema = field(default_factory=FrameSchema)
    input_slot: int = 0
    output_slot: int | None = None
    start_tick: int = 0
    occupied_slots: tuple[int, ...] = ()
    quarantined_slots: tuple[int, ...] = ()
    destinations: tuple[str, ...] = ("destination",)
    sources: tuple[str, ...] = ("source",)
    faults: tuple[FrameFault, ...] = ()

    def __post_init__(self) -> None:
        if type(self.request) is not DigitalRequest or type(self.frame_schema) is not FrameSchema:
            raise ValueError("INVALID_INPUT")
        request = DigitalRequest.from_dict(self.request.as_dict())
        schema = FrameSchema.from_dict(self.frame_schema.as_dict())
        integer(self.input_slot, schema.slot_count - 1)
        if self.output_slot is not None:
            integer(self.output_slot, schema.slot_count - 1)
        integer(self.start_tick, MAX_TICK)
        count = max(1, len(request.payload))
        if count > schema.max_frames or self.start_tick + count + 1 > MAX_TICK:
            raise ValueError("INVALID_INPUT")
        occupied = _slots(self.occupied_slots, schema.slot_count)
        quarantined = _slots(self.quarantined_slots, schema.slot_count)
        if set(occupied) & set(quarantined):
            raise ValueError("INVALID_INPUT")
        faults = sequence(self.faults, MAX_FAULTS)
        if any(type(x) is not FrameFault for x in faults):
            raise ValueError("INVALID_INPUT")
        faults = tuple(FrameFault.from_dict(x.as_dict()) for x in faults)
        indices = tuple(x.frame_index for x in faults)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("INVALID_INPUT")
        affected = set()
        for fault in faults:
            if fault.frame_index >= count or (fault.slot is not None and fault.slot >= schema.slot_count):
                raise ValueError("INVALID_INPUT")
            footprint = {fault.frame_index}
            if fault.kind == "reorder_frames":
                if fault.frame_index + 1 >= count:
                    raise ValueError("INVALID_INPUT")
                footprint.add(fault.frame_index + 1)
            if affected & footprint:
                raise ValueError("INVALID_INPUT")
            affected.update(footprint)
        for name, value in (("request", request), ("frame_schema", schema),
                            ("occupied_slots", occupied), ("quarantined_slots", quarantined),
                            ("destinations", _endpoints(self.destinations)),
                            ("sources", _endpoints(self.sources)), ("faults", faults)):
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict:
        return seal("switch-input", "174.0", request=self.request.as_dict(), frame_schema=self.frame_schema.as_dict(),
                    input_slot=self.input_slot, output_slot=self.output_slot, start_tick=self.start_tick,
                    occupied_slots=list(self.occupied_slots), quarantined_slots=list(self.quarantined_slots),
                    destinations=list(self.destinations), sources=list(self.sources),
                    faults=[x.as_dict() for x in self.faults],
                    endpoint_policy="exact-catalogue-match-any-free-output-slot.v1")

    @classmethod
    def from_dict(cls, value: object) -> DigitalSwitchInput:
        try:
            result = cls(DigitalRequest.from_dict(value["request"]), FrameSchema.from_dict(value["frame_schema"]),
                         value["input_slot"], value["output_slot"], value["start_tick"],
                         value["occupied_slots"], value["quarantined_slots"], value["destinations"], value["sources"],
                         tuple(FrameFault.from_dict(x) for x in sequence(value["faults"], MAX_FAULTS)))
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def _initial_state(source: DigitalSwitchInput) -> tuple[SlotState, ...]:
    states = []
    for slot in range(source.frame_schema.slot_count):
        if slot in source.quarantined_slots:
            states.append(SlotState(slot, "quarantined"))
        elif slot in source.occupied_slots:
            # Synthetic snapshot owners must not alias the transaction's caller
            # ID. This does not reserve a caller namespace: all valid IDs remain
            # usable. Slot-qualified base labels are unique; only the sole base
            # matching this request can acquire the distinct suffix.
            owner_id = "initial-occupant-" + str(slot)
            if owner_id == source.request.request_id:
                owner_id += "-snapshot"
            states.append(SlotState(slot, "busy", owner_id,
                                    seal("initial-occupant", "174.1", slot=slot)["sha256"]))
        else:
            states.append(SlotState(slot))
    return tuple(states)


def _observe(frames: tuple[DigitalFrame, ...], faults: tuple[FrameFault, ...]) -> tuple[FrameObservation, ...]:
    observations = [(frame.index, FrameObservation(frame.as_dict()["wire_hex"])) for frame in frames]
    for fault in faults:
        position = next(i for i, (index, _) in enumerate(observations) if index == fault.frame_index)
        index, observation = observations[position]
        if fault.kind == "frame_loss":
            del observations[position]
        elif fault.kind == "duplicate_frame":
            observations.insert(position + 1, observations[position])
        elif fault.kind == "reorder_frames":
            observations[position], observations[position + 1] = observations[position + 1], observations[position]
        elif fault.kind == "alignment_slip":
            observations[position] = (index, replace(observation, boundary_offset=1))
        else:
            wire = bytearray.fromhex(observation.wire_hex)
            address = HEADER_BYTES + 2 * fault.slot
            symbol = int.from_bytes(wire[address:address + 2], "big")
            if fault.kind == "bit_flip":
                symbol ^= 1 << fault.bit
                collisions = ()
            else:
                symbol = 257
                collisions = (fault.slot,)
            wire[address:address + 2] = symbol.to_bytes(2, "big")
            observations[position] = (index, FrameObservation(wire.hex(), colliding_slots=collisions))
    return tuple(observation for _, observation in observations)


def execute_digital_switch(source: DigitalSwitchInput) -> dict:
    """Reserve output capacity, transmit bounded frames, replay TSI, then release.

    Byte identity is recomputed after transport, never inferred from a carried
    hash. Rejection suppresses *all* reassembled payload bytes, including prefixes.
    """
    if type(source) is not DigitalSwitchInput:
        raise ValueError("INVALID_INPUT")
    source = DigitalSwitchInput.from_dict(source.as_dict())
    request = source.request
    initial = _initial_state(source)
    commands = []
    reason = "unknown_source" if request.source not in source.sources else (
        "unknown_destination" if request.destination not in source.destinations else None)
    if reason is None:
        commands.append(SlotCommand(source.start_tick, "allocate", "allocate", request.request_id,
                                    request.as_dict()["sha256"], source.output_slot))
    allocation = execute_slot_allocation(AllocationInput(initial, tuple(commands)))
    selected = allocation["events"][0]["selected_slot"] if commands else None
    if reason is None and selected is None:
        reason = allocation["events"][0]["reason"]
    frames, observations, sync, interchange = (), (), None, None
    accepted, output, failure_classes = False, None, []
    if selected is not None:
        frames = encode_frames(source.frame_schema, request.payload, input_slot=source.input_slot, start_tick=source.start_tick + 1)
        observations = _observe(frames, source.faults)
        sync = synchronize_frames(source.frame_schema, observations, expected_count=len(frames), start_tick=source.start_tick + 1)
        mapping = list(range(source.frame_schema.slot_count))
        mapping[source.input_slot], mapping[selected] = mapping[selected], mapping[source.input_slot]
        interchange = interchange_frames(sync, tuple(mapping))
        failure_classes = list(sync["fault_classes"])
        if sync["accepted"]:
            symbols = [memory["after"][selected] for memory in interchange["memories"]]
            # Every unused slot must remain explicitly idle; corruption there is
            # observable too, even when the selected payload slot was unchanged.
            if any(symbol != IDLE_SYMBOL for memory in interchange["memories"]
                   for slot, symbol in enumerate(memory["after"]) if slot != selected):
                failure_classes.append("unexpected_nonidle_symbol")
            if not request.payload:
                candidate = b"" if symbols == [IDLE_SYMBOL] else None
            else:
                candidate = bytes(symbols) if all(0 <= symbol <= 255 for symbol in symbols) else None
            if candidate is None:
                failure_classes.append("invalid_payload_symbol")
            elif len(candidate) != len(request.payload) or payload_hash(candidate) != payload_hash(request.payload):
                failure_classes.append("payload_hash_mismatch")
            elif not failure_classes:
                accepted, output = True, candidate
        failure_classes = sorted(set(failure_classes))
        reason = "delivered" if accepted else failure_classes[0]
        commands.append(SlotCommand(source.start_tick + len(frames) + 1, "release", "release", request.request_id,
                                    request.as_dict()["sha256"], selected))
        allocation = execute_slot_allocation(AllocationInput(initial, tuple(commands)))
    else:
        failure_classes = [reason]
    return seal("frame-switch-receipt", "174.0", input=source.as_dict(),
                assembled_contract_versions=["174.0", "174.1", "174.2", "174.3", "174.4"],
                allocation_receipt=allocation, selected_slot=selected,
                frames=[frame.as_dict() for frame in frames], observations=[x.as_dict() for x in observations],
                sync_receipt=sync, interchange_receipt=interchange, accepted=accepted,
                outcome="delivered" if accepted else "rejected", reason=reason, failure_classes=failure_classes,
                destination=request.destination, original_payload_sha256=payload_hash(request.payload),
                reassembled_payload_hex=None if output is None else output.hex(),
                reassembled_payload_sha256=None if output is None else payload_hash(output),
                decoder_output_sha256=request.decoder_output_sha256,
                payload_identity_verified=accepted, transport_attempted=selected is not None,
                has_payload=bool(request.payload), output_reservation_released=selected is not None,
                initial_state_restored=allocation["final_state"] == [x.as_dict() for x in initial],
                claim_boundary=dict(CLAIM_BOUNDARY))


def validate_digital_receipt(receipt: object, *, expected_input_sha256: str | None = None,
                             expected_request_sha256: str | None = None,
                             expected_frame_schema_sha256: str | None = None) -> dict:
    try:
        source = DigitalSwitchInput.from_dict(receipt["input"])
        bindings = pins(input=(expected_input_sha256, source.as_dict()["sha256"]),
                        request=(expected_request_sha256, source.request.as_dict()["sha256"]),
                        frame_schema=(expected_frame_schema_sha256, source.frame_schema.as_dict()["sha256"]))
        replay = execute_digital_switch(source)
        exact(receipt, replay)
        return seal("frame-switch-validation", "174.0", digital_frame_switch_receipt_hash=replay["sha256"],
                    deterministic_timeslot_allocation_receipt_hash=replay["allocation_receipt"]["sha256"],
                    accepted=replay["accepted"], payload_identity_verified=replay["payload_identity_verified"],
                    complete_replay_verified=True, external_bindings_verified=bindings, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def demo_digital_input() -> DigitalSwitchInput:
    return DigitalSwitchInput(DigitalRequest("digital-demo", "source", "destination", b"QEC\x00\xff", "a" * 64),
                              FrameSchema(4, 256), input_slot=0, output_slot=2)
