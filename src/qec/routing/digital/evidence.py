# SPDX-License-Identifier: MPL-2.0
"""v174.4 frozen fault battery and v174.5 evidence-backed bounded reports."""
from __future__ import annotations

from dataclasses import replace

from ._common import CLAIM_BOUNDARY, exact, pins, seal
from .allocation import AllocationInput, SlotCommand, SlotState, execute_slot_allocation
from .core import FrameFault, demo_digital_input, execute_digital_switch, validate_digital_receipt

BATTERY_CORPUS_ID = "digital-channel-fault-battery.v1"


def run_digital_fault_battery() -> dict:
    base = demo_digital_input()
    specs = [
        ("alignment-slip", replace(base, faults=(FrameFault("alignment_slip", 1),)), ["alignment_slip"]),
        ("baseline", base, []),
        ("bit-corruption", replace(base, faults=(FrameFault("bit_flip", 1, 0, 0),)), ["payload_hash_mismatch"]),
        ("duplicate-frame", replace(base, faults=(FrameFault("duplicate_frame", 1),)), ["duplicate_frame", "frame_count_mismatch"]),
        ("frame-loss", replace(base, faults=(FrameFault("frame_loss", 1),)), ["frame_count_mismatch", "missing_frame"]),
        ("idle-corruption", replace(base, faults=(FrameFault("bit_flip", 1, 1, 8),)), ["unexpected_nonidle_symbol"]),
        ("invalid-symbol", replace(base, faults=(FrameFault("bit_flip", 1, 0, 8),)), ["invalid_symbol"]),
        ("reordered-frames", replace(base, faults=(FrameFault("reorder_frames", 1),)), ["reordered_frames"]),
        ("saturated", replace(base, output_slot=None, occupied_slots=(0, 1), quarantined_slots=(2, 3)), ["no_capacity"]),
        ("slot-collision", replace(base, faults=(FrameFault("slot_collision", 1, 0),)), ["invalid_symbol", "slot_collision"]),
        ("slot-reservation-collision", replace(base, occupied_slots=(2,)), ["slot_collision"]),
        ("slot-reservation-quarantined", replace(base, quarantined_slots=(2,)), ["slot_quarantined"]),
        ("unknown-destination", replace(base, destinations=("other",)), ["unknown_destination"]),
    ]
    cases = []
    for name, source, expected in specs:
        receipt = execute_digital_switch(source)
        validation = validate_digital_receipt(receipt, expected_input_sha256=source.as_dict()["sha256"])
        cases.append({"case_id": name, "expected_failure_classes": expected,
                      "receipt": receipt, "validation": validation,
                      "passed": receipt["failure_classes"] == expected and receipt["accepted"] == (not expected)})
    allocation_input = AllocationInput((SlotState(0), SlotState(1)), (
        SlotCommand(0, "allocate-a", "allocate", "a", "a" * 64),
        SlotCommand(0, "allocate-b", "allocate", "b", "b" * 64),
        SlotCommand(0, "allocate-c", "allocate", "c", "c" * 64),
        SlotCommand(1, "release-wrong-owner", "release", "c", "c" * 64, 0),
        SlotCommand(2, "release-a", "release", "a", "a" * 64, 0),
        SlotCommand(2, "retry-c", "allocate", "c", "c" * 64),
    ))
    allocation = execute_slot_allocation(allocation_input)
    allocation_expected = ["allocated", "allocated", "no_capacity", "ownership_mismatch", "released", "allocated"]
    allocation_passed = [e["reason"] for e in allocation["events"]] == allocation_expected
    return seal("fault-battery", "174.4", corpus_id=BATTERY_CORPUS_ID, frozen_default_corpus=True,
                cases=cases, allocation_case={"receipt": allocation, "expected_reasons": allocation_expected,
                                             "passed": allocation_passed},
                case_count=len(cases) + 1, all_passed=all(x["passed"] for x in cases) and allocation_passed,
                claim_boundary=dict(CLAIM_BOUNDARY))


def validate_fault_battery(battery: object) -> dict:
    replay = run_digital_fault_battery()
    exact(battery, replay)
    return seal("fault-battery-validation", "174.4", digital_fault_battery_hash=replay["sha256"],
                frozen_corpus_replayed=True, complete_replay_verified=True, all_passed=replay["all_passed"])


def build_digital_report(receipt: dict) -> dict:
    validation = validate_digital_receipt(receipt)
    return seal("switch-report", "174.5", source_receipt=receipt, source_validation=validation,
                observations={"accepted": receipt["accepted"], "encoded_frame_count": len(receipt["frames"]),
                              "observed_frame_count": len(receipt["observations"]),
                              "selected_slot": receipt["selected_slot"],
                              "failure_classes": receipt["failure_classes"],
                              "reassembled_payload_bytes": None if not receipt["accepted"] else
                              len(bytes.fromhex(receipt["reassembled_payload_hex"])),
                              "logical_tick_units_are_not_physical_time": True},
                claim_boundary=dict(CLAIM_BOUNDARY),
                evidence_boundary="derived_summary_requires_full_native_source_replay")


def validate_digital_report(report: object, *, expected_receipt_sha256: str | None = None) -> dict:
    try:
        bindings = pins(receipt=(expected_receipt_sha256, report["source_receipt"]["sha256"]))
        replay = build_digital_report(report["source_receipt"])
        exact(report, replay)
        return seal("switch-report-validation", "174.5", digital_switch_report_hash=replay["sha256"],
                    complete_replay_verified=True, external_bindings_verified=bindings, all_passed=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
