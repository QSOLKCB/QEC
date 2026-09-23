# SPDX-License-Identifier: MPL-2.0
"""Common-control contract: bounded authority, replay, and fail-closed evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.crossbar import (
    CrossbarRequest,
    compile_marker_program,
    demo_matrix,
    execute_marker_program,
    validate_common_control_receipt,
)
from qec.routing.crossbar.cli import main
from qec.routing.crossbar.marker import MAX_PAYLOAD_BYTES
from qec.sonify.canonical import canonical_json, canonical_sha256


@pytest.fixture
def inputs():
    matrix = demo_matrix("marker-test", horizontal_count=2, vertical_count=3).as_dict()
    request = CrossbarRequest("request-0", "H000", "V002", b"\x00correction\xff", "a" * 64)
    return matrix, request, compile_marker_program(matrix, request)


def resign(artifact):
    artifact["sha256"] = canonical_sha256({k: v for k, v in artifact.items() if k != "sha256"})
    return artifact


def test_selected_plan_is_exact_and_has_no_fabric_side_effects(inputs):
    matrix, request, program = inputs
    before = deepcopy((matrix, request.as_dict(), program))
    receipt = execute_marker_program(matrix, request, program)
    assert receipt["outcome"] == "plan_selected"
    assert receipt["closure_plan"] == [matrix["intersections"][2]]
    assert receipt["coordinate_evaluations"] == 1
    assert receipt["matrix_after_sha256"] == matrix["sha256"]
    assert receipt["payload_sha256"] == hashlib.sha256(request.payload).hexdigest()
    assert receipt["decoder_output_sha256"] == request.decoder_output_sha256
    assert receipt["marker_released"] is True
    assert (matrix, request.as_dict(), program) == before
    assert [event["kind"] for event in receipt["events"]] == [
        "register_sealed", "program_verified", "endpoints_checked",
        "coordinate_evaluated", "plan_selected", "marker_released",
    ]
    previous = receipt["event_root_sha256"]
    for index, event in enumerate(receipt["events"]):
        assert event["sequence"] == index
        assert event["previous_sha256"] == previous
        assert event["sha256"] == canonical_sha256({k: v for k, v in event.items() if k != "sha256"})
        previous = event["sha256"]
    assert canonical_json(receipt) == canonical_json(execute_marker_program(matrix, request, program))
    validation = validate_common_control_receipt(receipt)
    assert validation["all_passed"] is True
    assert validation["crossbar_common_control_receipt_hash"] == receipt["sha256"]
    assert not any(validation["external_bindings_verified"].values())


@pytest.mark.parametrize("axis,link", [("horizontal", "H000"), ("vertical", "V002")])
@pytest.mark.parametrize("state", ["busy", "quarantined", "unavailable"])
def test_each_nonidle_state_rejects_without_fallback(inputs, axis, link, state):
    _, request, _ = inputs
    matrix = demo_matrix("blocked", horizontal_count=2, vertical_count=3,
                         state_overrides={link: state}).as_dict()
    program = compile_marker_program(matrix, request)
    receipt = execute_marker_program(matrix, request, program)
    assert receipt["outcome"] == "rejected"
    assert receipt["reason"] == axis + "_" + state
    assert receipt["closure_plan"] == []
    assert receipt["coordinate_evaluations"] == 1
    assert receipt["marker_released"] is True
    assert validate_common_control_receipt(receipt)["all_passed"] is True


@pytest.mark.parametrize("horizontal,vertical,reason", [
    ("unknown", "V002", "unknown_horizontal_link"),
    ("H000", "unknown", "unknown_vertical_link"),
    ("unknown", "unknown", "unknown_horizontal_link"),
    ("V000", "H000", "unknown_horizontal_link"),
])
def test_unknown_or_wrong_axis_endpoints_have_explicit_receipts(inputs, horizontal, vertical, reason):
    matrix, request, _ = inputs
    request = replace(request, horizontal_link_id=horizontal, vertical_link_id=vertical)
    receipt = execute_marker_program(matrix, request, compile_marker_program(matrix, request))
    assert receipt["reason"] == reason
    assert receipt["coordinate_evaluations"] == 0
    assert receipt["closure_plan"] == []
    assert receipt["events"][-1]["kind"] == "marker_released"
    assert validate_common_control_receipt(receipt)["all_passed"] is True


def test_horizontal_block_has_declared_precedence(inputs):
    _, request, _ = inputs
    matrix = demo_matrix("both", horizontal_count=2, vertical_count=3,
                         state_overrides={"H000": "busy", "V002": "quarantined"}).as_dict()
    receipt = execute_marker_program(matrix, request, compile_marker_program(matrix, request))
    assert receipt["reason"] == "horizontal_busy"
    assert receipt["events"][2]["details"]["vertical_state"] == "quarantined"


@pytest.mark.parametrize("change", [
    {"payload": b"other"}, {"decoder_output_sha256": "b" * 64},
    {"horizontal_link_id": "H001"}, {"vertical_link_id": "V001"}, {"request_id": "other"},
])
def test_sealed_program_cannot_be_reused_for_changed_request(inputs, change):
    matrix, request, program = inputs
    other = replace(request, **change)
    assert other.as_dict()["sha256"] != request.as_dict()["sha256"]
    assert compile_marker_program(matrix, other)["sha256"] != program["sha256"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_marker_program(matrix, other, program)


def test_program_pins_entire_matrix_state_even_off_route(inputs):
    matrix, request, program = inputs
    changed = demo_matrix("marker-test", horizontal_count=2, vertical_count=3,
                         state_overrides={"H001": "busy"}).as_dict()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_marker_program(changed, request, program)
    changed["intersections"].pop()
    resign(changed)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        compile_marker_program(changed, request)


@pytest.mark.parametrize("field,value", [
    ("policy", "first-free"), ("fallback_permitted", True),
    ("max_coordinate_evaluations", 2), ("max_coordinate_evaluations", True),
    ("contract_version", "172.0"), ("extra_authority", "force-accept"),
])
def test_resigned_program_cannot_expand_authority(inputs, field, value):
    matrix, request, program = inputs
    program[field] = value
    resign(program)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_marker_program(matrix, request, program)


@pytest.mark.parametrize("field,value", [
    ("outcome", "connected"), ("reason", "forced"), ("closure_plan", []),
    ("coordinate_evaluations", True), ("marker_released", 1), ("marker_released", False),
    ("payload_sha256", "0" * 64), ("decoder_output_sha256", "0" * 64),
    ("matrix_after_sha256", "0" * 64), ("event_root_sha256", "0" * 64),
    ("extra", "unbound"), ("sha256", "0" * 64),
])
def test_modified_receipts_fail_even_with_recomputed_outer_hash(inputs, field, value):
    receipt = execute_marker_program(*inputs)
    receipt[field] = value
    if field != "sha256":
        resign(receipt)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_common_control_receipt(receipt)


@pytest.mark.parametrize("mutation", ["missing_release", "reordered", "resigned_event", "later_coordinate", "claim_bool"])
def test_replay_rejects_fabricated_trace_or_selection(inputs, mutation):
    matrix, request, program = inputs
    receipt = execute_marker_program(matrix, request, program)
    if mutation == "missing_release":
        receipt["events"].pop()
    elif mutation == "reordered":
        receipt["events"].reverse()
    elif mutation == "resigned_event":
        receipt["events"][0]["sequence"] = True
        resign(receipt["events"][0])
    elif mutation == "later_coordinate":
        receipt["closure_plan"] = [matrix["intersections"][0]]
    else:
        receipt["claim_boundary"]["reservation_present"] = 0
    resign(receipt)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_common_control_receipt(receipt)


def test_external_bindings_reject_a_coherently_different_run(inputs):
    matrix, request, program = inputs
    receipt = execute_marker_program(*inputs)
    pins = {"expected_matrix_sha256": matrix["sha256"],
            "expected_request_sha256": request.as_dict()["sha256"],
            "expected_program_sha256": program["sha256"]}
    assert all(validate_common_control_receipt(receipt, **pins)["external_bindings_verified"].values())
    for key in pins:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_common_control_receipt(receipt, **{**pins, key: "0" * 64})
    other = replace(request, payload=b"changed")
    other_receipt = execute_marker_program(matrix, other, compile_marker_program(matrix, other))
    assert validate_common_control_receipt(other_receipt)["all_passed"] is True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_common_control_receipt(other_receipt, **pins)


@pytest.mark.parametrize("field,value", [
    ("payload_hex", "00 636f7272656374696f6eff"),
    ("payload_hex", "00636F7272656374696F6EFF"),
    ("payload_length", True), ("sealed", 1), ("payload_sha256", "0" * 64),
    ("extra", False), ("sha256", "0" * 64),
])
def test_register_requires_exact_canonical_replay(inputs, field, value):
    register = inputs[1].as_dict()
    register[field] = value
    if field != "sha256":
        resign(register)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        CrossbarRequest.from_dict(register)


@pytest.mark.parametrize("change", [
    {"payload": bytearray(b"mutable")}, {"payload": "text"},
    {"payload": b"x" * (MAX_PAYLOAD_BYTES + 1)},
    {"request_id": ""}, {"request_id": "x" * 4097}, {"request_id": "\ud800"},
    {"horizontal_link_id": 0}, {"decoder_output_sha256": "A" * 64},
    {"decoder_output_sha256": False},
])
def test_invalid_requests_fail_before_computation(inputs, change):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(inputs[1], **change)


def test_request_immutability_empty_and_maximum_payload(inputs):
    request = inputs[1]
    with pytest.raises(FrozenInstanceError):
        request.payload = b"changed"
    for payload in (b"", b"x" * MAX_PAYLOAD_BYTES):
        value = replace(request, payload=payload)
        assert CrossbarRequest.from_dict(value.as_dict()) == value


def test_hash_seed_and_process_independence():
    code = '''from qec.routing.crossbar import *
from qec.sonify.canonical import canonical_json
m=demo_matrix("process",horizontal_count=2,vertical_count=2).as_dict()
r=CrossbarRequest("r","H000","V001",b"opaque","a"*64)
print(canonical_json(execute_marker_program(m,r,compile_marker_program(m,r))))'''
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={
        **os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(Path("src").resolve()),
    }) for seed in ("1", "927")]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("blocked", [False, True])
def test_cli_round_trip_and_rejection_status(tmp_path, capsys, blocked):
    manifest = tmp_path / "matrix.json"
    matrix = demo_matrix(state_overrides={"H000": "busy"} if blocked else {}).as_dict()
    manifest.write_text(canonical_json(matrix), encoding="utf-8")
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"\x00opaque\xff")
    output = tmp_path / "output"
    status = main([
        "marker", "--manifest", str(manifest), "--request-id", "cli",
        "--horizontal-link-id", "H000", "--vertical-link-id", "V001",
        "--payload-file", str(payload), "--decoder-output-sha256", "a" * 64,
        "--output-dir", str(output),
    ])
    assert status == (2 if blocked else 0)
    assert len(list(output.iterdir())) == 4
    receipt = output / "crossbar_common_control_receipt.json"
    assert main(["marker-validate", "--receipt", str(receipt),
                 "--expected-matrix-sha256", matrix["sha256"]]) == 0
    result = json.loads(receipt.read_text())
    assert result["outcome"] == ("rejected" if blocked else "plan_selected")
    assert bytes.fromhex(result["input_register"]["payload_hex"]) == payload.read_bytes()
    assert main(["marker-validate", "--receipt", str(receipt),
                 "--expected-matrix-sha256", "0" * 64]) == 1
    assert "INVALID_INPUT" in capsys.readouterr().err


def test_cli_rejects_duplicate_json_keys(tmp_path):
    receipt = tmp_path / "duplicate.json"
    receipt.write_text('{"schema": "first", "schema": "second"}')
    assert main(["marker-validate", "--receipt", str(receipt)]) == 1


def test_frozen_canonical_fixture_and_matrix_compatibility():
    fixture = Path(__file__).with_name("fixtures") / "common-control-v1.json"
    raw = fixture.read_text(encoding="utf-8")
    receipt = json.loads(raw)
    assert raw == canonical_json(receipt) + "\n"
    matrix = demo_matrix("v172-golden", horizontal_count=1, vertical_count=1).as_dict()
    assert receipt["matrix_manifest"] == matrix
    assert matrix["contract_version"] == "172.0"
    request = CrossbarRequest("golden", "H000", "V000", b"\x00golden\xff", "a" * 64)
    replay = execute_marker_program(matrix, request, compile_marker_program(matrix, request))
    assert receipt == replay
    assert validate_common_control_receipt(receipt)["all_passed"] is True
