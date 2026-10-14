# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from qec.routing.digital import (
    AllocationInput, DigitalFrame, DigitalRequest, DigitalSwitchInput, FrameFault, FrameObservation,
    FrameSchema, SlotCommand, SlotState, build_digital_report, demo_digital_input, encode_frames,
    execute_digital_switch, execute_slot_allocation, interchange_frames, run_digital_fault_battery,
    synchronize_frames, validate_allocation_receipt, validate_digital_receipt, validate_digital_report,
    validate_fault_battery, validate_interchange_receipt, validate_sync_receipt,
)
from qec.routing.digital._common import MAX_TICK
from qec.routing.digital.cli import main
from qec.routing.digital.lab import demo_svg, demo_wav, export_digital_demo
from qec.sonify.canonical import canonical_json, canonical_sha256

FIXTURES = Path(__file__).parent / "fixtures"


def rehash(value):
    if isinstance(value, list):
        return [rehash(x) for x in value]
    if isinstance(value, dict):
        result = {k: rehash(v) for k, v in value.items() if k != "sha256"}
        if "sha256" in value:
            result["sha256"] = canonical_sha256(result)
        return result
    return value


def receipt():
    return execute_digital_switch(demo_digital_input())


def test_frame_layout_and_exact_byte_roundtrip():
    source = demo_digital_input()
    result = execute_digital_switch(source)
    assert result["accepted"] and result["selected_slot"] == 2
    assert result["reassembled_payload_hex"] == source.request.payload.hex()
    assert result["reassembled_payload_sha256"] == hashlib.sha256(source.request.payload).hexdigest()
    assert result["initial_state_restored"] and result["output_reservation_released"]
    assert [e["reason"] for e in result["allocation_receipt"]["events"]] == ["allocated", "released"]
    frame = result["frames"][0]
    wire = bytes.fromhex(frame["wire_hex"])
    assert len(wire) == 22 and wire[:2] == b"\xd4\x7d"
    assert int.from_bytes(wire[2:6], "big") == 0
    assert int.from_bytes(wire[6:14], "big") == 1
    assert list(frame["symbols"]) == [81, 256, 256, 256]
    assert result["interchange_receipt"]["mapping"] == [2, 1, 0, 3]
    assert result["interchange_receipt"]["memories"][0]["after"] == [256, 256, 81, 256]
    validation = validate_digital_receipt(result, expected_input_sha256=source.as_dict()["sha256"],
                                          expected_request_sha256=source.request.as_dict()["sha256"],
                                          expected_frame_schema_sha256=source.frame_schema.as_dict()["sha256"])
    assert all(validation["external_bindings_verified"].values())


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\xff", bytes(range(256))])
def test_all_byte_symbols_and_empty_explicit_idle(payload):
    source = replace(demo_digital_input(), request=replace(demo_digital_input().request, payload=payload))
    result = execute_digital_switch(source)
    assert result["accepted"] and result["reassembled_payload_hex"] == payload.hex()
    assert len(result["frames"]) == max(1, len(payload))
    assert result["has_payload"] is bool(payload)
    if not payload:
        assert result["frames"][0]["symbols"] == [256] * 4


@pytest.mark.parametrize("occupied,quarantined,selected", [
    ((), (), 0), ((0,), (), 1), ((), (0,), 1), ((1,), (), 0), ((0,), (1,), None), ((), (0, 1), None), ((0, 1), (), None),
])
def test_first_free_native_capacity(occupied, quarantined, selected):
    source = replace(demo_digital_input(), frame_schema=FrameSchema(2), output_slot=None,
                     occupied_slots=occupied, quarantined_slots=quarantined)
    result = execute_digital_switch(source)
    assert result["selected_slot"] == selected
    assert result["accepted"] == (selected is not None)
    assert result["initial_state_restored"]
    if selected is None:
        assert result["reason"] == "no_capacity" and not result["transport_attempted"]
        assert result["frames"] == [] and result["sync_receipt"] is None
        assert result["reassembled_payload_hex"] is None
    else:
        assert selected not in occupied + quarantined


@pytest.mark.parametrize("changes,reason", [
    ({"occupied_slots": (2,)}, "slot_collision"),
    ({"quarantined_slots": (2,)}, "slot_quarantined"),
    ({"sources": ("other",)}, "unknown_source"),
    ({"destinations": ("other",)}, "unknown_destination"),
])
def test_native_rejections_no_fallback(changes, reason):
    result = execute_digital_switch(replace(demo_digital_input(), **changes))
    assert result["reason"] == reason and not result["accepted"]
    assert result["reassembled_payload_hex"] is None
    assert validate_digital_receipt(result)["all_passed"]


@pytest.mark.parametrize("kind,slot,expected", [
    ("bit_flip", 0, "payload_hash_mismatch"), ("frame_loss", None, "missing_frame"),
    ("duplicate_frame", None, "duplicate_frame"), ("reorder_frames", None, "reordered_frames"),
    ("alignment_slip", None, "alignment_slip"), ("slot_collision", 0, "slot_collision"),
])
def test_all_channel_faults_fail_closed(kind, slot, expected):
    result = execute_digital_switch(replace(demo_digital_input(), faults=(FrameFault(kind, 1, slot),)))
    assert not result["accepted"] and expected in result["failure_classes"]
    assert result["reassembled_payload_sha256"] is None and result["reassembled_payload_hex"] is None
    assert result["initial_state_restored"] and result["output_reservation_released"]
    assert validate_digital_receipt(result)["all_passed"]


def test_fault_battery_full_native_replay_and_negative_controls():
    battery = run_digital_fault_battery()
    assert battery["all_passed"] and battery["case_count"] == 14
    assert validate_fault_battery(battery)["frozen_corpus_replayed"]
    changed = deepcopy(battery)
    changed["cases"][0]["passed"] = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fault_battery(rehash(changed))
    changed = deepcopy(battery)
    changed["cases"].pop()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fault_battery(rehash(changed))


def test_saturation_release_ownership_and_same_tick_policy():
    state = (SlotState(0), SlotState(1))
    commands = (
        SlotCommand(0, "a", "allocate", "a", "a" * 64),
        SlotCommand(0, "b", "allocate", "b", "b" * 64),
        SlotCommand(0, "c", "allocate", "c", "c" * 64),
        SlotCommand(1, "wrong", "release", "a", "c" * 64, 0),
        SlotCommand(2, "release", "release", "a", "a" * 64, 0),
        SlotCommand(2, "retry", "allocate", "c", "c" * 64),
    )
    source = AllocationInput(state, commands)
    result = execute_slot_allocation(source)
    assert [x["reason"] for x in result["events"]] == ["allocated", "allocated", "no_capacity", "ownership_mismatch", "released", "allocated"]
    assert result["events"][-1]["selected_slot"] == 0
    assert validate_allocation_receipt(result, expected_input_sha256=source.as_dict()["sha256"])["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AllocationInput(state, commands[:4] + (commands[5], commands[4]))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_allocation_receipt(result, expected_input_sha256="f" * 64)


def test_allocator_identity_conflict_and_duplicate_owner_rejections():
    source = AllocationInput((SlotState(0), SlotState(1)), (
        SlotCommand(0, "a", "allocate", "a", "a" * 64),
        SlotCommand(1, "b", "allocate", "a", "a" * 64),
        SlotCommand(2, "c", "allocate", "a", "b" * 64),
    ))
    result = execute_slot_allocation(source)
    assert [e["reason"] for e in result["events"]] == ["allocated", "already_allocated", "owner_identity_conflict"]


@pytest.mark.parametrize("factory", [
    lambda: FrameSchema(True), lambda: FrameSchema(0), lambda: FrameSchema(17), lambda: FrameSchema(1, 1025),
    lambda: FrameSchema(1, 0), lambda: FrameSchema(1, 1.0),
    lambda: DigitalFrame(False, 0, (1,)), lambda: DigitalFrame(0, True, (1,)),
    lambda: DigitalFrame(0, 0, (257,)), lambda: DigitalFrame(0, 0, ()),
    lambda: DigitalFrame(0, 0, (False,)), lambda: DigitalFrame(0, 0, [1] * 17),
    lambda: DigitalRequest("a", "s", "d", bytearray(b"a"), "a" * 64),
    lambda: DigitalRequest("a", "s", "d", b"a", "A" * 64),
    lambda: DigitalRequest("a", "s", "d", b"a" * 1025, "a" * 64),
    lambda: DigitalRequest("\ud800", "s", "d", b"", "a" * 64),
    lambda: FrameObservation("AA"), lambda: FrameObservation("a"), lambda: FrameObservation("00 "),
    lambda: FrameObservation("", True), lambda: FrameObservation("", colliding_slots=(1, 0)),
    lambda: FrameFault("unknown", 0), lambda: FrameFault("bit_flip", 0, 0, 16),
    lambda: FrameFault("bit_flip", 0), lambda: FrameFault("frame_loss", 0, 0),
    lambda: SlotState(0, "busy"), lambda: SlotState(0, "idle", "owner", "a" * 64),
    lambda: SlotCommand(0, "c", "release", "a", "a" * 64),
    lambda: AllocationInput((), ()), lambda: AllocationInput((SlotState(1),), ()),
])
def test_strict_primitives_and_bounds(factory):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        factory()


@pytest.mark.parametrize("changes", [
    {"input_slot": True}, {"input_slot": 4}, {"output_slot": 4}, {"output_slot": False},
    {"start_tick": MAX_TICK}, {"start_tick": 1.0}, {"occupied_slots": (1, 0)},
    {"occupied_slots": (1, 1)}, {"occupied_slots": (1,), "quarantined_slots": (1,)},
    {"destinations": ()}, {"destinations": ("z", "a")}, {"sources": ("s", "s")},
    {"faults": (FrameFault("frame_loss", 99),)},
    {"faults": (FrameFault("reorder_frames", 4),)},
    {"faults": (FrameFault("reorder_frames", 0), FrameFault("frame_loss", 1))},
    {"faults": (FrameFault("frame_loss", 1), FrameFault("frame_loss", 0))},
    {"frame_schema": FrameSchema(4, 1)},
])
def test_invalid_switch_inputs_are_rejected(changes):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(demo_digital_input(), **changes)


def test_snapshots_frozen_detached_and_bounded():
    source = demo_digital_input()
    with pytest.raises(FrozenInstanceError):
        source.input_slot = 1
    exported = source.as_dict()
    exported["request"]["payload_hex"] = "ff"
    assert source.request.payload == b"QEC\x00\xff"
    assert DigitalSwitchInput.from_dict(source.as_dict()) == source
    assert DigitalFrame.from_dict(encode_frames(FrameSchema(), b"x")[0].as_dict()).symbols == (120, 256, 256, 256)


@pytest.mark.parametrize("path,value", [
    (("accepted",), False), (("payload_identity_verified",), False), (("selected_slot",), 1),
    (("destination",), "other"), (("reassembled_payload_hex",), "ff"),
    (("reassembled_payload_sha256",), "a" * 64), (("decoder_output_sha256",), "b" * 64),
    (("initial_state_restored",), False), (("failure_classes",), ["invented"]),
    (("frames", 0, "symbols", 0), 12), (("observations", 0, "boundary_offset"), 1),
    (("allocation_receipt", "events", 0, "selected_slot"), 1),
    (("allocation_receipt", "final_state", 2, "state"), "quarantined"),
    (("sync_receipt", "accepted"), False), (("sync_receipt", "decoded_frames", 0, "symbols", 0), 12),
    (("interchange_receipt", "mapping", 0), 1),
    (("interchange_receipt", "memories", 0, "after", 2), 12),
    (("claim_boundary", "physical_network_or_quantum_evidence"), True),
])
def test_rehashed_mutation_does_not_replace_replay(path, value):
    changed = receipt()
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_digital_receipt(rehash(changed))


@pytest.mark.parametrize("key", ["frames", "observations", "allocation_receipt", "sync_receipt", "interchange_receipt", "claim_boundary"])
def test_missing_evidence_is_not_accepted(key):
    changed = receipt()
    del changed[key]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_digital_receipt(rehash(changed))


def test_trust_pins_detect_valid_alternate_input_and_unpinned_is_not_provenance():
    source = demo_digital_input()
    other = replace(source, request=replace(source.request, payload=b"other"))
    result = execute_digital_switch(other)
    assert not any(validate_digital_receipt(result)["external_bindings_verified"].values())
    for kwargs in ({"expected_input_sha256": source.as_dict()["sha256"]},
                   {"expected_request_sha256": source.request.as_dict()["sha256"]},
                   {"expected_frame_schema_sha256": FrameSchema(2).as_dict()["sha256"]},
                   {"expected_input_sha256": "a"}, {"expected_input_sha256": True}):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_digital_receipt(result, **kwargs)


@pytest.mark.parametrize("mutator,expected", [
    (lambda wire: wire[:-1], "invalid_frame_width"),
    (lambda wire: b"xx" + wire[2:], "invalid_sync_magic"),
    (lambda wire: wire[:2] + (9).to_bytes(4, "big") + wire[6:], "sequence_out_of_range"),
    (lambda wire: wire[:6] + (9).to_bytes(8, "big") + wire[14:], "logical_tick_shift"),
    (lambda wire: wire[:14] + b"\xff\xff" + wire[16:], "invalid_symbol"),
])
def test_sync_raw_malformed_wire_evidence(mutator, expected):
    schema = FrameSchema(1)
    frame = encode_frames(schema, b"a")[0]
    observation = FrameObservation(mutator(bytes.fromhex(frame.as_dict()["wire_hex"])).hex())
    result = synchronize_frames(schema, (observation,), expected_count=1)
    assert not result["accepted"] and expected in result["fault_classes"]
    assert result["decoded_frames"] == []
    assert validate_sync_receipt(result)["all_passed"]
    exchange = interchange_frames(result, (0,))
    assert not exchange["accepted"] and exchange["memories"] == []
    assert validate_interchange_receipt(exchange)["all_passed"]


def test_missing_duplicate_and_idle_are_distinct():
    schema = FrameSchema(1)
    idle = FrameObservation(encode_frames(schema, b"")[0].as_dict()["wire_hex"])
    assert synchronize_frames(schema, (idle,), expected_count=1)["accepted"]
    missing = synchronize_frames(schema, (), expected_count=1)
    assert "missing_frame" in missing["fault_classes"]
    duplicate = synchronize_frames(schema, (idle, idle), expected_count=1)
    assert duplicate["duplicated_indices"] == [0]


@pytest.mark.parametrize("mapping", [(0, 0, 2, 3), (1, 0), (0, 1, 2, 4), (False, 1, 2, 3)])
def test_nonbijective_interchange_mapping_rejected(mapping):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        interchange_frames(receipt()["sync_receipt"], mapping)


def test_nested_independent_receipt_validation_and_pins():
    result = receipt()
    assert validate_sync_receipt(result["sync_receipt"], expected_frame_schema_sha256=result["input"]["frame_schema"]["sha256"])["all_passed"]
    assert validate_interchange_receipt(result["interchange_receipt"], expected_sync_receipt_sha256=result["sync_receipt"]["sha256"])["all_passed"]
    changed = deepcopy(result["interchange_receipt"])
    changed["memories"][0]["writes"][0]["symbol"] = 200
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_interchange_receipt(rehash(changed))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_interchange_receipt(result["interchange_receipt"], expected_sync_receipt_sha256="a" * 64)


def test_canonical_report_and_demo_exports_separated():
    result = receipt()
    report = build_digital_report(result)
    assert validate_digital_report(report, expected_receipt_sha256=result["sha256"])["all_passed"]
    changed = deepcopy(report)
    changed["observations"]["accepted"] = False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_digital_report(rehash(changed))
    demo = export_digital_demo(result)
    assert demo["canonical_evidence"] is False and demo["presentation_only"]
    assert demo["schema"] != result["schema"] and "sha256" not in demo
    assert demo_svg(result) == demo_svg(result) and "DEMO ONLY" in demo_svg(result)
    audio = demo_wav(result)
    assert audio == demo_wav(result)
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 8000, 5000)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_digital_receipt(demo)


def test_frozen_fixtures_and_every_sealed_hash():
    expected = json.loads((FIXTURES / "digital-receipt-v1.json").read_text())
    result = receipt()
    assert canonical_json(result) == canonical_json(expected)
    hashes = json.loads((FIXTURES / "digital-hashes-v1.json").read_text())
    assert result["sha256"] == hashes["receipt_sha256"]
    assert run_digital_fault_battery()["sha256"] == hashes["fault_battery_sha256"]
    assert hashlib.sha256(demo_wav(result)).hexdigest() == hashes["demo_wav_sha256"]
    def walk(value):
        if isinstance(value, dict):
            if "sha256" in value:
                assert value["sha256"] == canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)


def test_hashseed_and_repeated_process_determinism():
    script = "from qec.routing.digital import *; from qec.sonify.canonical import canonical_json; print(canonical_json(execute_digital_switch(demo_digital_input()))); print(run_digital_fault_battery()['sha256'])"
    outputs = [subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, text=True)
               for seed in ("0", "1", "987")]
    assert outputs[0] == outputs[1] == outputs[2]


def test_cli_end_to_end_and_native_exports(tmp_path, capsys):
    assert main(["demo", "--output-dir", str(tmp_path)]) == 0
    source = tmp_path / "digital_switch_input.json"
    assert main(["run", "--input", str(source), "--output-dir", str(tmp_path)]) == 0
    artifact = tmp_path / "digital_frame_switch_receipt.json"
    assert main(["validate", "--receipt", str(artifact)]) == 0
    assert main(["fault-battery", "--output-dir", str(tmp_path)]) == 0
    assert main(["lab", "--receipt", str(artifact), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "digital_lab_demo.svg").exists() and (tmp_path / "digital_lab_demo.wav").exists()
    assert main(["validate", "--receipt", str(artifact), "--expected-input-sha256", "a" * 64]) == 1
    capsys.readouterr()


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '{}', '[]', 'null'])
def test_cli_rejects_ambiguous_or_invalid_json(tmp_path, content, capsys):
    path = tmp_path / "bad.json"
    path.write_text(content)
    assert main(["run", "--input", str(path), "--output-dir", str(tmp_path)]) == 1
    assert "INVALID_INPUT" in capsys.readouterr().err


def test_cli_module_entrypoint(tmp_path):
    completed = subprocess.run([sys.executable, "-m", "qec.routing.digital", "demo", "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["schema"] == "qec.digital-switch-input.v1"


def test_unknown_keys_and_recomputed_payload_identity_rejected():
    source = demo_digital_input().as_dict()
    source["extra"] = True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DigitalSwitchInput.from_dict(rehash(source))
    request = demo_digital_input().request.as_dict()
    request["payload_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DigitalRequest.from_dict(rehash(request))
    request = demo_digital_input().request.as_dict()
    request["payload_hex"] = request["payload_hex"].upper()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        DigitalRequest.from_dict(rehash(request))


def test_sync_and_mapping_full_external_input_pins():
    result = receipt()
    sync = result["sync_receipt"]
    assert validate_sync_receipt(sync, expected_input_sha256=sync["input_sha256"])["external_bindings_verified"]["input"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_sync_receipt(sync, expected_input_sha256="0" * 64)
    interchange = result["interchange_receipt"]
    assert validate_interchange_receipt(interchange, expected_mapping_sha256=interchange["mapping_sha256"])["all_passed"]
    different = interchange_frames(sync, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_interchange_receipt(different, expected_mapping_sha256=interchange["mapping_sha256"])


def test_all_initial_owners_have_unique_ids():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AllocationInput((SlotState(0, "busy", "a", "a" * 64), SlotState(1, "busy", "a", "b" * 64)), ())


def test_cli_rejection_does_not_leave_previous_optional_evidence(tmp_path, capsys):
    source_path = tmp_path / "input.json"
    source_path.write_text(canonical_json(demo_digital_input().as_dict()))
    assert main(["run", "--input", str(source_path), "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / "frame_sync_receipt.json").exists()
    source_path.write_text(canonical_json(replace(demo_digital_input(), occupied_slots=(2,)).as_dict()))
    assert main(["run", "--input", str(source_path), "--output-dir", str(tmp_path)]) == 0
    assert not (tmp_path / "frame_sync_receipt.json").exists()
    assert not (tmp_path / "timeslot_interchange_receipt.json").exists()
    capsys.readouterr()


def test_cli_input_byte_limit_and_deep_json_fail_closed(tmp_path, capsys, monkeypatch):
    import qec.routing.digital.cli as cli
    path = tmp_path / "invalid.json"
    path.write_text("x" * 129)
    monkeypatch.setattr(cli, "MAX_INPUT_FILE_BYTES", 128)
    assert main(["run", "--input", str(path)]) == 1
    path.write_text('{')
    assert main(["run", "--input", str(path)]) == 1
    assert "INVALID_INPUT" in capsys.readouterr().err


def test_maximum_payload_frame_and_tick_bounds():
    source = DigitalSwitchInput(DigitalRequest("maximum", "source", "destination", bytes(range(256)) * 4, "a" * 64),
                                FrameSchema(16, 1024), output_slot=15, start_tick=MAX_TICK - 1025)
    result = execute_digital_switch(source)
    assert result["accepted"] and len(result["frames"]) == 1024
    assert result["allocation_receipt"]["events"][-1]["command"]["tick"] == MAX_TICK
    assert len(canonical_json(result).encode()) < 16 * 1024 * 1024
    assert validate_digital_receipt(result)["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source, start_tick=source.start_tick + 1)


def test_command_and_fault_budgets_are_finite():
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        AllocationInput((SlotState(0),), tuple(SlotCommand(i, "c" + str(i), "allocate", "a", "a" * 64) for i in range(65)))
    source = replace(demo_digital_input(), request=replace(demo_digital_input().request, payload=bytes(range(32))))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source, faults=tuple(FrameFault("frame_loss", i) for i in range(17)))


def test_multi_fault_plan_is_applied_by_original_frame_identity():
    source = replace(demo_digital_input(), faults=(FrameFault("frame_loss", 0), FrameFault("reorder_frames", 2),
                                                 FrameFault("duplicate_frame", 4)))
    result = execute_digital_switch(source)
    assert not result["accepted"]
    assert result["sync_receipt"]["missing_indices"] == [0]
    assert result["sync_receipt"]["duplicated_indices"] == [4]
    assert "reordered_frames" in result["failure_classes"]
    assert result["reassembled_payload_hex"] is None
    assert validate_digital_receipt(result)["all_passed"]


@pytest.mark.parametrize("request_id", ["ordinary", "initial-occupant-0", "initial-occupant-0-snapshot"])
def test_snapshot_owner_cannot_shadow_two_slot_caller(request_id):
    source = DigitalSwitchInput(DigitalRequest(request_id, "source", "destination", b"ab", "a" * 64),
                                FrameSchema(2), occupied_slots=(0,))
    result = execute_digital_switch(source)
    assert result["accepted"] and result["selected_slot"] == 1
    assert result["reassembled_payload_hex"] == b"ab".hex()
    assert result["initial_state_restored"]
    initial = result["allocation_receipt"]["input"]["initial_state"]
    assert initial[0]["owner_id"] != request_id
    assert validate_digital_receipt(result, expected_input_sha256=source.as_dict()["sha256"])["all_passed"]


@pytest.mark.parametrize("slot", range(16))
@pytest.mark.parametrize("state", ["occupied_slots", "quarantined_slots"])
@pytest.mark.parametrize("label", ["initial-occupant-{slot}", "initial-occupant-{slot}-snapshot", "quarantined-{slot}"])
def test_snapshot_slot_names_never_reserve_a_caller_namespace(slot, state, label):
    request_id = label.format(slot=slot)
    source = DigitalSwitchInput(DigitalRequest(request_id, "source", "destination", b"ab", "a" * 64),
                                FrameSchema(16), **{state: (slot,)})
    result = execute_digital_switch(source)
    assert result["accepted"] and result["selected_slot"] == (1 if slot == 0 else 0)
    assert result["initial_state_restored"]
    initial = result["allocation_receipt"]["input"]["initial_state"]
    assert all(item["owner_id"] != request_id for item in initial)
    assert validate_digital_receipt(result)["all_passed"]


@pytest.mark.parametrize("slot", range(16))
def test_synthetic_owners_remain_unique_when_all_slots_busy(slot):
    source = DigitalSwitchInput(DigitalRequest("initial-occupant-" + str(slot), "source", "destination", b"ab", "a" * 64),
                                FrameSchema(16), occupied_slots=tuple(range(16)))
    result = execute_digital_switch(source)
    assert not result["accepted"] and result["reason"] == "no_capacity"
    owners = [item["owner_id"] for item in result["allocation_receipt"]["input"]["initial_state"]]
    assert len(owners) == len(set(owners)) == 16
    assert source.request.request_id not in owners
