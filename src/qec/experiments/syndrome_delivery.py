# SPDX-License-Identifier: MPL-2.0
"""Opt-in, bounded synthetic syndrome transport replay with logical delays.

Split each ordered binary frame into contiguous transport fragments, apply an
explicit integer delay/jitter schedule, sort arrivals by (tick, frame, partition),
and reassemble before calling the same unchanged, caller-supplied pure decoder.
The baseline calls that decoder once per complete frame, followed by replay calls
in the identical frame order. Actual exact-binary correction vectors are compared.
No callback purity, correction validity, or logical error rate is inferred.

This is not Patchwork, a partitioned decoder/compiler, space-time decoding,
hardware latency measurement, QEC advantage evidence, or decoder promotion.
Partitioning changes only classical delivery. All ticks and the serial decode
cost are declared model inputs; no wall clock, randomness, network, or I/O is used.
Hashes bind canonical records, not provenance or scientific truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from qec.sonify.canonical import canonical_sha256

MAX_FRAMES = 4096
MAX_WIDTH = 4096
MAX_TOTAL_BITS = 1_048_576
MAX_PARTITIONS = 256
MAX_FRAGMENTS = 65_536
MAX_TICKS = 1_000_000_000

CLAIM_BOUNDARY = MappingProxyType({
    "synthetic_classical_transport_only": True,
    "same_unchanged_callback_for_both_paths": True,
    "callback_purity_proven": False,
    "corrections_proven_valid": False,
    "patchwork_decoder_or_compiler": False,
    "space_time_partition_decoding": False,
    "hardware_latency_measurement": False,
    "qec_advantage_evidence": False,
    "decoder_promotion": False,
    "hash_authenticates_origin": False,
})

Decoder = Callable[[tuple[int, ...]], tuple[int, ...] | list[int]]


def _integer(value: object, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("INVALID_INPUT")


def _sequence(value: object, maximum: int) -> tuple:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError("INVALID_INPUT")
    return tuple(value)


def _bits(value: object, width: int) -> tuple[int, ...]:
    bits = _sequence(value, MAX_WIDTH)
    if len(bits) != width or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("INVALID_INPUT")
    return bits


@dataclass(frozen=True, slots=True)
class SyndromeTrace:
    """Finite ordered frames; exact integer bits only, with an explicit width."""

    syndrome_width: int
    frames: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _integer(self.syndrome_width, MAX_WIDTH, 1)
        frames = _sequence(self.frames, MAX_FRAMES)
        if len(frames) * self.syndrome_width > MAX_TOTAL_BITS:
            raise ValueError("INVALID_INPUT")
        object.__setattr__(self, "frames", tuple(_bits(frame, self.syndrome_width) for frame in frames))


@dataclass(frozen=True, slots=True)
class DeliveryPlan:
    """Contiguous fragment widths and explicit integer classical delay model.

    ``jitter_ticks`` is a frame-by-partition matrix; () means all zeros.
    Jitter may be negative, but base delay plus jitter must be nonnegative.
    Frames are emitted at index * frame_period_ticks. A single serial decoder
    has declared decode_ticks cost in both paths. Completion exactly at the
    emission-relative deadline is on time; late frames are still decoded.
    """

    partition_widths: tuple[int, ...]
    base_delay_ticks: tuple[int, ...]
    jitter_ticks: tuple[tuple[int, ...], ...] = ()
    frame_period_ticks: int = 1
    deadline_ticks: int = 0
    decode_ticks: int = 0

    def __post_init__(self) -> None:
        widths = _sequence(self.partition_widths, MAX_PARTITIONS)
        delays = _sequence(self.base_delay_ticks, MAX_PARTITIONS)
        if not widths or len(widths) != len(delays):
            raise ValueError("INVALID_INPUT")
        for width in widths:
            _integer(width, MAX_WIDTH, 1)
        if sum(widths) > MAX_WIDTH:
            raise ValueError("INVALID_INPUT")
        for delay in delays:
            _integer(delay, MAX_TICKS)
        jitter = _sequence(self.jitter_ticks, MAX_FRAMES)
        if len(jitter) * len(widths) > MAX_FRAGMENTS:
            raise ValueError("INVALID_INPUT")
        rows = []
        for row in jitter:
            row = _sequence(row, MAX_PARTITIONS)
            if len(row) != len(widths):
                raise ValueError("INVALID_INPUT")
            for index, delta in enumerate(row):
                _integer(delta, MAX_TICKS, -MAX_TICKS)
                _integer(delays[index] + delta, MAX_TICKS)
            rows.append(row)
        _integer(self.frame_period_ticks, MAX_TICKS, 1)
        _integer(self.deadline_ticks, MAX_TICKS)
        _integer(self.decode_ticks, MAX_TICKS)
        object.__setattr__(self, "partition_widths", widths)
        object.__setattr__(self, "base_delay_ticks", delays)
        object.__setattr__(self, "jitter_ticks", tuple(rows))


def _inputs(trace: SyndromeTrace, plan: DeliveryPlan, decoder: Decoder,
            correction_width: int, decoder_label: str) -> dict:
    if type(trace) is not SyndromeTrace or type(plan) is not DeliveryPlan or not callable(decoder):
        raise ValueError("INVALID_INPUT")
    # Frozen dataclasses are convenience, not a trust boundary: callers can
    # bypass them with object.__setattr__. Revalidate and snapshot all fields.
    try:
        trace = SyndromeTrace(trace.syndrome_width, trace.frames)
        plan = DeliveryPlan(plan.partition_widths, plan.base_delay_ticks, plan.jitter_ticks,
                            plan.frame_period_ticks, plan.deadline_ticks, plan.decode_ticks)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    _integer(correction_width, MAX_WIDTH, 1)
    if len(trace.frames) * correction_width > MAX_TOTAL_BITS:
        raise ValueError("INVALID_INPUT")
    if (type(decoder_label) is not str or not 1 <= len(decoder_label) <= 128
            or decoder_label.strip() != decoder_label
            or any(not 32 <= ord(char) <= 126 for char in decoder_label)):
        raise ValueError("INVALID_INPUT")
    if (sum(plan.partition_widths) != trace.syndrome_width
            or len(trace.frames) * len(plan.partition_widths) > MAX_FRAGMENTS
            or (plan.jitter_ticks and len(plan.jitter_ticks) != len(trace.frames))):
        raise ValueError("INVALID_INPUT")
    jitter = plan.jitter_ticks or ((0,) * len(plan.partition_widths),) * len(trace.frames)
    return {
        "schema": "qec.syndrome-delivery-input.v1",
        "syndrome_width": trace.syndrome_width,
        "frames": [list(frame) for frame in trace.frames],
        "partition_widths": list(plan.partition_widths),
        "base_delay_ticks": list(plan.base_delay_ticks),
        "jitter_ticks": [list(row) for row in jitter],
        "frame_period_ticks": plan.frame_period_ticks,
        "deadline_ticks": plan.deadline_ticks,
        "decode_ticks": plan.decode_ticks,
        "correction_width": correction_width,
        "decoder_label": decoder_label,
        "decoder_label_authority": "caller_label_only",
    }


def _transport(source: dict) -> tuple[list[dict], list[tuple[int, ...]], list[dict]]:
    """Replay complete fragments and model chronological serial decode timing."""
    frames = source["frames"]
    widths = source["partition_widths"]
    arrivals = []
    for frame_index, frame in enumerate(frames):
        emitted = frame_index * source["frame_period_ticks"]
        _integer(emitted + source["deadline_ticks"], MAX_TICKS)
        offset = 0
        for partition_index, width in enumerate(widths):
            delay = source["base_delay_ticks"][partition_index]
            jitter = source["jitter_ticks"][frame_index][partition_index]
            arrival = emitted + delay + jitter
            _integer(arrival, MAX_TICKS)
            arrivals.append({
                "frame_index": frame_index,
                "partition_index": partition_index,
                "offset": offset,
                "bits": frame[offset:offset + width],
                "emitted_tick": emitted,
                "base_delay_ticks": delay,
                "jitter_ticks": jitter,
                "arrival_tick": arrival,
            })
            offset += width
    arrivals.sort(key=lambda row: (row["arrival_tick"], row["frame_index"], row["partition_index"]))
    pending: list[dict[int, tuple[int, ...]]] = [{} for _ in frames]
    ready_ticks = [0] * len(frames)
    ordered_frames, timings = [], []
    next_frame = 0
    baseline_finished = replay_finished = 0
    for arrival in arrivals:
        frame_index = arrival["frame_index"]
        pending[frame_index][arrival["partition_index"]] = tuple(arrival["bits"])
        ready_ticks[frame_index] = arrival["arrival_tick"]
        while next_frame < len(frames) and len(pending[next_frame]) == len(widths):
            reassembled = tuple(bit for partition in range(len(widths)) for bit in pending[next_frame][partition])
            ordered_frames.append(reassembled)
            emitted = next_frame * source["frame_period_ticks"]
            deadline = emitted + source["deadline_ticks"]
            baseline_start = max(emitted, baseline_finished)
            baseline_finished = baseline_start + source["decode_ticks"]
            ordered_ready = arrival["arrival_tick"]
            replay_start = max(ordered_ready, replay_finished)
            replay_finished = replay_start + source["decode_ticks"]
            _integer(baseline_finished, MAX_TICKS)
            _integer(replay_finished, MAX_TICKS)
            timings.append({
                "frame_index": next_frame,
                "emitted_tick": emitted,
                "deadline_tick": deadline,
                "all_fragments_arrived_tick": ready_ticks[next_frame],
                "chronological_reassembly_tick": ordered_ready,
                "baseline_start_tick": baseline_start,
                "baseline_finish_tick": baseline_finished,
                "replay_start_tick": replay_start,
                "replay_finish_tick": replay_finished,
                "baseline_deadline_missed": baseline_finished > deadline,
                "replay_deadline_missed": replay_finished > deadline,
            })
            next_frame += 1
    return arrivals, ordered_frames, timings


def run_syndrome_delivery(trace: SyndromeTrace, plan: DeliveryPlan, decoder: Decoder,
                          *, correction_width: int, decoder_label: str = "caller") -> dict:
    """Return a JSON-safe, canonical-hashed transport replay comparison.

    The callback must be pure/deterministic, accept an immutable binary tuple,
    and return a list/tuple of exactly correction_width exact integer bits.
    Both paths use the same callable; all baseline calls precede all replay
    calls. A stateful callback can produce mismatches, which are reported rather
    than trusted or suppressed. Callback exceptions propagate. No callback is
    invoked until every transport input and modeled time has been validated.

    Empty traces are valid and make zero calls; outputs_match is vacuously true
    and comparison_outcome is explicitly empty_trace. Hashes identify content,
    not the callback implementation or validity of its corrections.
    """
    source = _inputs(trace, plan, decoder, correction_width, decoder_label)
    arrivals, recovered, timings = _transport(source)
    baseline_frames = tuple(tuple(frame) for frame in source["frames"])
    baseline = [_bits(decoder(frame), correction_width) for frame in baseline_frames]
    replay = [_bits(decoder(frame), correction_width) for frame in recovered]
    mismatch_indices = [index for index in range(len(baseline)) if baseline[index] != replay[index]]
    input_indices = [index for index in range(len(baseline_frames)) if baseline_frames[index] != recovered[index]]
    result = {
        "schema": "qec.syndrome-delivery-receipt.v1",
        "input": source,
        "input_sha256": canonical_sha256(source),
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "arrival_tie_break": ["arrival_tick", "frame_index", "partition_index"],
        "arrivals": arrivals,
        "chronological_reassembled_frames": [list(frame) for frame in recovered],
        "timings": timings,
        "baseline_corrections": [list(bits) for bits in baseline],
        "replay_corrections": [list(bits) for bits in replay],
        "compared_frame_count": len(baseline_frames),
        "input_mismatch_indices": input_indices,
        "inputs_match": not input_indices,
        "output_mismatch_indices": mismatch_indices,
        "outputs_match": not mismatch_indices,
        "comparison_outcome": ("empty_trace" if not baseline_frames else
                               "mismatch" if mismatch_indices or input_indices else "match"),
        "baseline_deadline_miss_count": sum(row["baseline_deadline_missed"] for row in timings),
        "replay_deadline_miss_count": sum(row["replay_deadline_missed"] for row in timings),
    }
    return {**result, "sha256": canonical_sha256(result)}


def _exact(actual: object, expected: object) -> None:
    """Bound verification traversal by the known, freshly replayed schema."""
    if type(actual) is not type(expected):
        raise ValueError("INVALID_INPUT")
    if type(expected) is dict:
        if (len(actual) != len(expected) or any(type(key) is not str for key in actual)
                or actual.keys() != expected.keys()):
            raise ValueError("INVALID_INPUT")
        for key in sorted(expected):
            _exact(actual[key], expected[key])
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise ValueError("INVALID_INPUT")
        for value, wanted in zip(actual, expected):
            _exact(value, wanted)
    elif actual != expected:
        raise ValueError("INVALID_INPUT")


def validate_syndrome_delivery_receipt(receipt: object, trace: SyndromeTrace,
                                       plan: DeliveryPlan, decoder: Decoder, *,
                                       correction_width: int, decoder_label: str = "caller") -> None:
    """Replay against independently supplied inputs/callback; reject any change.

    This re-executes both decoder paths. Use the same pure decoder implementation
    and settings; passing a different/stateful callback may correctly fail replay.
    Rehashing modified claims is insufficient to pass. Verification proves only
    equality to this local replay, not authentic origin, decoder correctness,
    callback purity, performance improvement, or quantum protection.
    """
    expected = run_syndrome_delivery(trace, plan, decoder, correction_width=correction_width,
                                     decoder_label=decoder_label)
    _exact(receipt, expected)
