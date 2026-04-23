# SPDX-License-Identifier: MIT
"""Deterministic tests for v147.1 retro trace intake bridge."""

from __future__ import annotations

import json

import pytest

from qec.analysis.retro_target_registry import build_retro_target
from qec.analysis.retro_trace_intake_bridge import RetroTraceReceipt, build_retro_trace


def _target_receipt():
    return build_retro_target(
        target_id="z80-home-micro",
        isa_family="z80",
        word_size=8,
        address_width=16,
        ram_budget=48 * 1024,
        rom_budget=32 * 1024,
        cycle_budget=3_500_000,
        display_budget={"width": 256, "height": 192, "colors": 16},
        audio_budget={"channels": 3, "sample_rate": 44_100},
        input_budget={"buttons": 2, "axes": 0},
        fpu_policy="none",
        provenance="hardware",
    )


def _trace_payload():
    return {
        "cpu_trace": (
            {"pc": 0x1000, "a": 0x10, "flags": 0b00110000},
            {"pc": 0x1001, "a": 0x11, "flags": 0b00110001},
        ),
        "memory_trace": (
            {"address": 0x4000, "op": "read", "value": 0xAB},
            {"address": 0x4001, "op": "write", "value": 0xCD},
        ),
        "timing_trace": (
            {"cycle": 100.0, "frame": 1},
            {"cycle": 99.6, "frame": 1},
            {"cycle": 120, "frame": 2},
        ),
        "display_trace": ({"scanline": 0, "event": "start"},),
        "audio_trace": ({"channel": 1, "pattern": "pulse"},),
        "input_trace": ({"port": 1, "button": "A", "state": 1},),
        "metadata": {"emulator": "retroarch", "rom_hash": "abc123", "version": "1.0.0"},
    }


def _build(**overrides):
    payload = _trace_payload()
    payload.update(overrides)
    return build_retro_trace(target_receipt=_target_receipt(), **payload)


def test_deterministic_replay_same_input_same_hash() -> None:
    a = _build()
    b = _build()
    assert a.stable_hash == b.stable_hash
    assert a.to_canonical_json() == b.to_canonical_json()


def test_unordered_input_normalization() -> None:
    baseline = _build()
    shuffled = _build(
        cpu_trace=tuple(reversed(_trace_payload()["cpu_trace"])),
        memory_trace=tuple(reversed(_trace_payload()["memory_trace"])),
        timing_trace=tuple(reversed(_trace_payload()["timing_trace"])),
        metadata={"version": "1.0.0", "rom_hash": "abc123", "emulator": "retroarch"},
    )
    assert shuffled.stable_hash == baseline.stable_hash
    assert shuffled.to_canonical_json() == baseline.to_canonical_json()


def test_invalid_schema_rejection() -> None:
    with pytest.raises(ValueError, match="canonical primitive"):
        _build(cpu_trace=({"pc": {"nested": 1}},))


def test_timing_normalization_float_input() -> None:
    receipt = _build(timing_trace=({"cycle": 5.4}, {"cycle": 5.6}, {"cycle": 9.2}))
    assert receipt.normalized_timing == (5, 6, 9)
    assert all(isinstance(v, int) for v in receipt.normalized_timing)
    timing_payloads = [dict(payload) for _, event_type, payload in receipt.event_sequence if event_type == "timing"]
    assert all(payload["cycle"] in (5, 6, 9) for payload in timing_payloads)
    assert all(isinstance(payload["cycle"], int) for payload in timing_payloads)
    assert all("cycles" not in payload for payload in timing_payloads)


def test_hash_tamper_detection() -> None:
    receipt = _build()
    tampered = list(receipt.event_sequence)
    tampered[0] = (0, tampered[0][1], (("pc", 9999),))
    with pytest.raises(ValueError, match="stable_hash mismatch"):
        RetroTraceReceipt(
            target_id=receipt.target_id,
            trace_length=receipt.trace_length,
            event_sequence=tuple(tampered),
            normalized_timing=receipt.normalized_timing,
            metadata=receipt.metadata,
            trace_metrics=receipt.trace_metrics,
            stable_hash=receipt.stable_hash,
        )


def test_mutation_safety_event_sequence() -> None:
    receipt = _build()
    with pytest.raises(TypeError):
        receipt.event_sequence[0] = receipt.event_sequence[0]


def test_canonical_round_trip_rebuild_identical_hash() -> None:
    receipt = _build()
    parsed = json.loads(receipt.to_canonical_json())

    rebuilt = build_retro_trace(
        target_receipt=_target_receipt(),
        cpu_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "cpu"),
        memory_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "memory"),
        timing_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "timing"),
        display_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "display"),
        audio_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "audio"),
        input_trace=tuple(dict(item[2]) for item in parsed["event_sequence"] if item[1] == "input"),
        metadata=parsed["metadata"],
    )

    assert rebuilt.stable_hash == receipt.stable_hash
    assert rebuilt.to_canonical_json() == receipt.to_canonical_json()


def test_trace_length_mismatch_rejected() -> None:
    receipt = _build()
    with pytest.raises(ValueError, match="trace_length must equal len\\(event_sequence\\)"):
        RetroTraceReceipt(
            target_id=receipt.target_id,
            trace_length=receipt.trace_length + 1,
            event_sequence=receipt.event_sequence,
            normalized_timing=receipt.normalized_timing,
            metadata=receipt.metadata,
            trace_metrics=receipt.trace_metrics,
            stable_hash=receipt.stable_hash,
        )


def test_negative_timing_rejected() -> None:
    receipt = _build()
    with pytest.raises(ValueError, match="normalized_timing values must be non-negative"):
        RetroTraceReceipt(
            target_id=receipt.target_id,
            trace_length=receipt.trace_length,
            event_sequence=receipt.event_sequence,
            normalized_timing=(-1, 2),
            metadata=receipt.metadata,
            trace_metrics=receipt.trace_metrics,
            stable_hash=receipt.stable_hash,
        )


@pytest.mark.parametrize(
    ("field_name", "payload", "match_text"),
    (
        (
            "event_sequence",
            ((0, "cpu", (("pc", 1), ("pc", 2))),),
            "event\\.payload keys must be unique",
        ),
        (
            "metadata",
            (("emulator", "retroarch"), ("emulator", "retroarch-2")),
            "metadata keys must be unique",
        ),
        (
            "trace_metrics",
            (
                ("trace_completeness", 1.0),
                ("event_order_integrity", 1.0),
                ("timing_observability", 1.0),
                ("input_sparsity", 1.0),
                ("replay_consistency", 1.0),
                ("replay_consistency", 1.0),
            ),
            "trace_metrics keys must be unique",
        ),
    ),
)
def test_duplicate_keys_rejected(field_name: str, payload, match_text: str) -> None:
    receipt = _build()
    kwargs = {
        "target_id": receipt.target_id,
        "trace_length": receipt.trace_length,
        "event_sequence": receipt.event_sequence,
        "normalized_timing": receipt.normalized_timing,
        "metadata": receipt.metadata,
        "trace_metrics": receipt.trace_metrics,
        "stable_hash": receipt.stable_hash,
    }
    kwargs[field_name] = payload
    if field_name == "event_sequence":
        kwargs["trace_length"] = len(payload)
    with pytest.raises(ValueError, match=match_text):
        RetroTraceReceipt(**kwargs)


def test_missing_cycle_field_rejected() -> None:
    with pytest.raises(ValueError, match="must include 'cycle' or 'cycles'"):
        _build(timing_trace=({"frame": 1},))
