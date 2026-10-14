# SPDX-License-Identifier: MPL-2.0
"""v174.0 fixed-width frames; v174.2 synchronization; v174.3 interchange."""
from __future__ import annotations

from dataclasses import dataclass

from ._common import IDLE_SYMBOL, MAX_FRAMES, MAX_SLOTS, MAX_TICK, exact, hex_bytes, integer, pins, seal, sequence

SYNC_MAGIC = b"\xd4\x7d"
HEADER_BYTES = 14  # sync magic (2), index (4), logical tick (8); all integers big-endian.


@dataclass(frozen=True)
class FrameSchema:
    slot_count: int = 4
    max_frames: int = 256

    def __post_init__(self) -> None:
        integer(self.slot_count, MAX_SLOTS, 1)
        integer(self.max_frames, MAX_FRAMES, 1)

    @property
    def frame_width_bytes(self) -> int:
        return HEADER_BYTES + 2 * self.slot_count

    def as_dict(self) -> dict:
        return seal("frame-schema", "174.0", slot_count=self.slot_count, max_frames=self.max_frames,
                    frame_width_bytes=self.frame_width_bytes, header_bytes=HEADER_BYTES,
                    byte_order="big", symbol_width_bytes=2, data_symbol_min=0, data_symbol_max=255,
                    idle_symbol=IDLE_SYMBOL, missing_symbol="absent_frame_or_slot",
                    invalid_symbol="257_through_65535", slot_numbering="zero_based",
                    sync_magic_hex=SYNC_MAGIC.hex(), sequence_width_bytes=4, logical_tick_width_bytes=8)

    @classmethod
    def from_dict(cls, value: object) -> FrameSchema:
        try:
            result = cls(value["slot_count"], value["max_frames"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class DigitalFrame:
    index: int
    logical_tick: int
    symbols: tuple[int, ...]

    def __post_init__(self) -> None:
        integer(self.index, MAX_FRAMES - 1)
        integer(self.logical_tick, MAX_TICK)
        symbols = sequence(self.symbols, MAX_SLOTS)
        if not symbols:
            raise ValueError("INVALID_INPUT")
        for symbol in symbols:
            integer(symbol, IDLE_SYMBOL)
        object.__setattr__(self, "symbols", symbols)

    def as_dict(self) -> dict:
        wire = SYNC_MAGIC + self.index.to_bytes(4, "big") + self.logical_tick.to_bytes(8, "big")
        wire += b"".join(symbol.to_bytes(2, "big") for symbol in self.symbols)
        return seal("frame", "174.0", index=self.index, logical_tick=self.logical_tick,
                    symbols=list(self.symbols), wire_hex=wire.hex())

    @classmethod
    def from_dict(cls, value: object) -> DigitalFrame:
        try:
            result = cls(value["index"], value["logical_tick"], value["symbols"])
            exact(value, result.as_dict())
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


@dataclass(frozen=True)
class FrameObservation:
    """Bounded captured wire bytes and explicitly declared receive-boundary faults.

    Bytes are deliberately allowed to be malformed: sync returns rejection
    evidence. Construction is strict about representation, not wire validity.
    """
    wire_hex: str
    boundary_offset: int = 0
    colliding_slots: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        hex_bytes(self.wire_hex, HEADER_BYTES + 2 * MAX_SLOTS + 2)
        integer(self.boundary_offset, HEADER_BYTES + 2 * MAX_SLOTS, -(HEADER_BYTES + 2 * MAX_SLOTS))
        slots = sequence(self.colliding_slots, MAX_SLOTS)
        for slot in slots:
            integer(slot, MAX_SLOTS - 1)
        if slots != tuple(sorted(set(slots))):
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "colliding_slots", slots)

    def as_dict(self) -> dict:
        return {"wire_hex": self.wire_hex, "boundary_offset": self.boundary_offset,
                "colliding_slots": list(self.colliding_slots)}

    @classmethod
    def from_dict(cls, value: object) -> FrameObservation:
        try:
            result = cls(**value)
            exact(value, result.as_dict())
            return result
        except (TypeError, ValueError) as exc:
            raise ValueError("INVALID_INPUT") from exc


def encode_frames(schema: FrameSchema, payload: bytes, *, input_slot: int = 0, start_tick: int = 0) -> tuple[DigitalFrame, ...]:
    if type(schema) is not FrameSchema or type(payload) is not bytes:
        raise ValueError("INVALID_INPUT")
    schema = FrameSchema.from_dict(schema.as_dict())
    integer(input_slot, schema.slot_count - 1)
    integer(start_tick, MAX_TICK)
    count = max(1, len(payload))
    if count > schema.max_frames or start_tick + count - 1 > MAX_TICK:
        raise ValueError("INVALID_INPUT")
    result = []
    for index in range(count):
        symbols = [IDLE_SYMBOL] * schema.slot_count
        if payload:
            symbols[input_slot] = payload[index]
        result.append(DigitalFrame(index, start_tick + index, tuple(symbols)))
    return tuple(result)


def synchronize_frames(schema: FrameSchema, observations: tuple[FrameObservation, ...], *, expected_count: int,
                       start_tick: int = 0) -> dict:
    if type(schema) is not FrameSchema:
        raise ValueError("INVALID_INPUT")
    schema = FrameSchema.from_dict(schema.as_dict())
    integer(expected_count, schema.max_frames, 1)
    integer(start_tick, MAX_TICK)
    if start_tick + expected_count - 1 > MAX_TICK:
        raise ValueError("INVALID_INPUT")
    observations = sequence(observations, 2 * MAX_FRAMES)
    if any(type(x) is not FrameObservation for x in observations):
        raise ValueError("INVALID_INPUT")
    observations = tuple(FrameObservation.from_dict(x.as_dict()) for x in observations)
    diagnostics, decoded, indices, issues = [], [], [], set()
    for position, observation in enumerate(observations):
        wire = bytes.fromhex(observation.wire_hex)
        faults = []
        if observation.boundary_offset:
            faults.append("alignment_slip")
        if observation.colliding_slots:
            if any(slot >= schema.slot_count for slot in observation.colliding_slots):
                raise ValueError("INVALID_INPUT")
            faults.append("slot_collision")
        record = None
        if len(wire) != schema.frame_width_bytes:
            faults.append("invalid_frame_width")
        elif wire[:2] != SYNC_MAGIC:
            faults.append("invalid_sync_magic")
        else:
            index = int.from_bytes(wire[2:6], "big")
            tick = int.from_bytes(wire[6:14], "big")
            symbols = [int.from_bytes(wire[i:i + 2], "big") for i in range(HEADER_BYTES, len(wire), 2)]
            indices.append(index)
            if not 0 <= index < expected_count:
                faults.append("sequence_out_of_range")
            if tick != start_tick + index or tick > MAX_TICK:
                faults.append("logical_tick_shift")
            if any(symbol > IDLE_SYMBOL for symbol in symbols):
                faults.append("invalid_symbol")
            record = {"index": index, "logical_tick": tick, "symbols": symbols}
            decoded.append(record)
        issues.update(faults)
        diagnostics.append({"position": position, "faults": sorted(faults), "decoded": record})
    missing = [index for index in range(expected_count) if index not in indices]
    duplicate = sorted(index for index in set(indices) if indices.count(index) > 1)
    if missing:
        issues.add("missing_frame")
    if duplicate:
        issues.add("duplicate_frame")
    if indices != sorted(indices):
        issues.add("reordered_frames")
    if len(observations) != expected_count:
        issues.add("frame_count_mismatch")
    source_identity = seal("sync-input", "174.2", frame_schema=schema.as_dict(),
                           observations=[x.as_dict() for x in observations],
                           expected_count=expected_count, start_tick=start_tick)["sha256"]
    return seal("frame-sync-receipt", "174.2", input_sha256=source_identity, frame_schema=schema.as_dict(),
                observations=[x.as_dict() for x in observations], expected_count=expected_count,
                start_tick=start_tick, diagnostics=diagnostics, missing_indices=missing,
                duplicated_indices=duplicate, fault_classes=sorted(issues), accepted=not issues,
                decoded_frames=decoded if not issues else [], fail_closed=True)


def validate_sync_receipt(receipt: object, *, expected_input_sha256: str | None = None,
                          expected_frame_schema_sha256: str | None = None,
                          expected_observations_sha256: str | None = None) -> dict:
    try:
        schema = FrameSchema.from_dict(receipt["frame_schema"])
        observations = tuple(FrameObservation.from_dict(x) for x in sequence(receipt["observations"], 2 * MAX_FRAMES))
        observation_identity = seal("observation-stream", "174.2", observations=[x.as_dict() for x in observations])["sha256"]
        bindings = pins(input=(expected_input_sha256, receipt["input_sha256"]),
                        frame_schema=(expected_frame_schema_sha256, schema.as_dict()["sha256"]),
                        observations=(expected_observations_sha256, observation_identity))
        replay = synchronize_frames(schema, observations, expected_count=receipt["expected_count"], start_tick=receipt["start_tick"])
        exact(receipt, replay)
        return seal("frame-sync-validation", "174.2", frame_sync_receipt_hash=replay["sha256"],
                    accepted=replay["accepted"], complete_replay_verified=True,
                    external_bindings_verified=bindings, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def interchange_frames(sync_receipt: dict, mapping: tuple[int, ...]) -> dict:
    """Write every input slot into a bijective output address; record each memory."""
    validate_sync_receipt(sync_receipt)
    schema = FrameSchema.from_dict(sync_receipt["frame_schema"])
    mapping = sequence(mapping, MAX_SLOTS)
    for slot in mapping:
        integer(slot, schema.slot_count - 1)
    if len(mapping) != schema.slot_count or tuple(sorted(mapping)) != tuple(range(schema.slot_count)):
        raise ValueError("INVALID_INPUT")
    memories = []
    if sync_receipt["accepted"]:
        for frame in sync_receipt["decoded_frames"]:
            output = [IDLE_SYMBOL] * schema.slot_count
            writes = []
            for source_slot, output_slot in enumerate(mapping):
                symbol = frame["symbols"][source_slot]
                output[output_slot] = symbol
                writes.append({"input_slot": source_slot, "output_slot": output_slot, "symbol": symbol})
            memories.append({"index": frame["index"], "logical_tick": frame["logical_tick"],
                             "before": [IDLE_SYMBOL] * schema.slot_count, "writes": writes,
                             "after": output})
    mapping_identity = seal("interchange-mapping", "174.3", mapping=list(mapping),
                            semantics="output_address_for_each_input_slot")["sha256"]
    return seal("timeslot-interchange-receipt", "174.3", sync_receipt=sync_receipt,
                mapping_sha256=mapping_identity,
                mapping=list(mapping), mapping_semantics="output_address_for_each_input_slot",
                memory_policy="reset_to_idle_then_complete_permutation_each_frame.v1",
                memories=memories, accepted=sync_receipt["accepted"], fail_closed=True)


def validate_interchange_receipt(receipt: object, *, expected_sync_receipt_sha256: str | None = None,
                                 expected_mapping_sha256: str | None = None) -> dict:
    try:
        bindings = pins(sync_receipt=(expected_sync_receipt_sha256, receipt["sync_receipt"]["sha256"]),
                        mapping=(expected_mapping_sha256, receipt["mapping_sha256"]))
        replay = interchange_frames(receipt["sync_receipt"], receipt["mapping"])
        exact(receipt, replay)
        return seal("interchange-validation", "174.3", timeslot_interchange_receipt_hash=replay["sha256"],
                    complete_replay_verified=True, accepted=replay["accepted"],
                    external_bindings_verified=bindings, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
