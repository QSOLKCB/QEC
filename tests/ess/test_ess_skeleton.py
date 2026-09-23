# SPDX-License-Identifier: MPL-2.0
"""Boundary, replay and independent-observation tests for the v173.0 skeleton."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.crossbar.continuity import validate_continuity_receipt
from qec.routing.crossbar.multistage import compile_path_program, execute_path_program
from qec.routing.ess import (
    CallStore, CrossbarFabricAdapter, EventQueue, InputEvent, ProgramStore,
    SwitchInput, demo_switch_input, execute_switch, validate_switch_receipt,
)
from qec.routing.ess.cli import main
from qec.routing.ess.core import MAX_CALLS, MAX_LOGICAL_TICK, MAX_PAYLOAD_BYTES
from qec.sonify.canonical import canonical_json, canonical_sha256


def rehash(value):
    """Re-sign every nested artifact, so tests exercise semantics, not stale hashes."""
    if isinstance(value, list):
        return [rehash(item) for item in value]
    if isinstance(value, dict):
        result = {key: rehash(item) for key, item in value.items() if key != "sha256"}
        if "sha256" in value:
            result["sha256"] = canonical_sha256(result)
        return result
    return value


@pytest.fixture
def source():
    return demo_switch_input()


@pytest.fixture
def receipt(source):
    return execute_switch(source)


def test_queue_drives_dispatch_and_native_evidence(source, receipt):
    assert [call.request.request_id for call in source.call_store.calls] == ["call-a", "call-b"]
    assert [c["call_id"] for c in receipt["command_stream"]["commands"]] == ["call-b", "call-a"]
    assert [r["outcome"] for r in receipt["results"]] == ["rejected", "plan_selected"]
    assert receipt["results"][0]["reason"] == "unknown_destination_vertical_link"
    assert (receipt["dispatched_count"], receipt["selected_count"], receipt["rejected_count"]) == (2, 1, 1)
    for row in receipt["results"]:
        continuity = row["continuity_receipt"]
        assert validate_continuity_receipt(continuity)["all_passed"]
        native = continuity["source_receipt"]
        request = next(c for c in source.call_store.calls if c.request.request_id == row["call_id"])
        assert native["input_register"] == request.as_dict()
        assert bytes.fromhex(native["input_register"]["request"]["payload_hex"]) == request.request.payload
        assert native["input_register"]["request"]["decoder_output_sha256"] == request.request.decoder_output_sha256
    plan = receipt["results"][1]["continuity_receipt"]["source_receipt"]["path_plan"]
    assert plan["coordinates"][0]["matrix_id"] == "ingress"
    assert plan["coordinates"][-1]["coordinate"]["vertical_link_id"] == "V001"
    assert receipt["claim_boundary"]["connection_commit_present"] is False


def test_independent_plans_do_not_reserve_capacity(source):
    a = source.call_store.calls[0]
    b = replace(a, request=replace(a.request, request_id="call-b"))
    both = replace(source, call_store=CallStore((a, b)))
    before = canonical_json(both.as_dict())
    result = execute_switch(both)
    assert result["selected_count"] == 2
    plans = [r["continuity_receipt"]["source_receipt"]["path_plan"] for r in result["results"]]
    assert plans[0] == plans[1]
    assert canonical_json(both.as_dict()) == before
    assert canonical_json(execute_switch(both)) == canonical_json(result)


def test_blocked_and_budget_results_are_preserved(source):
    fabric = source.adapter.fabric
    blocked = replace(fabric, interstage_links=tuple(replace(w, state="busy") for w in fabric.interstage_links))
    result = execute_switch(replace(source, adapter=CrossbarFabricAdapter(blocked)))
    assert result["selected_count"] == 0
    assert result["results"][1]["reason"] == "no_admissible_complete_path"
    budget = execute_switch(replace(source, program_store=ProgramStore(max_search_evaluations=1)))
    assert budget["results"][1]["reason"] == "search_budget_exhausted"
    assert validate_switch_receipt(budget)["all_passed"] is True


def test_empty_and_maximum_batch(source):
    empty = replace(source, call_store=CallStore(()), event_queue=EventQueue(()))
    result = execute_switch(empty)
    assert result["dispatched_count"] == result["selected_count"] == 0
    assert result["queue_drained"] is True
    assert validate_switch_receipt(result)["all_passed"] is True
    call = source.call_store.calls[0]
    calls = tuple(replace(call, request=replace(call.request, request_id=f"call-{i:02d}",
                                               payload=b"x" * MAX_PAYLOAD_BYTES)) for i in range(MAX_CALLS))
    queue = EventQueue(tuple(InputEvent(MAX_LOGICAL_TICK, i, c.request.request_id) for i, c in enumerate(calls)))
    batch = replace(source, call_store=CallStore(calls), event_queue=queue)
    assert execute_switch(batch)["selected_count"] == MAX_CALLS


def test_immutability_detachment_and_roundtrip(source):
    calls = list(source.call_store.calls)
    events = list(source.event_queue.events)
    copied = replace(source, call_store=CallStore(calls), event_queue=EventQueue(events))
    calls.clear()
    events.clear()
    assert len(copied.call_store.calls) == len(copied.event_queue.events) == 2
    with pytest.raises(FrozenInstanceError):
        copied.program_store.program_version = "2"
    serialized = copied.as_dict()
    restored = SwitchInput.from_dict(serialized)
    serialized["program_store"]["program_id"] = "mutated"
    assert restored.as_dict() == source.as_dict()


@pytest.mark.parametrize("field,value", [
    ("program_id", ""), ("program_id", "x" * 129), ("program_id", "\ud800"),
    ("program_version", 1), ("marker_id", None), ("max_search_evaluations", True),
    ("max_search_evaluations", 0), ("max_search_evaluations", 131073),
    ("max_search_evaluations", 1.0),
])
def test_program_bounds(field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ProgramStore(**{field: value})


@pytest.mark.parametrize("tick,sequence,call_id", [
    (-1, 0, "a"), (MAX_LOGICAL_TICK + 1, 0, "a"), (True, 0, "a"),
    (1.0, 0, "a"), (0, True, "a"), (0, -1, "a"), (0, MAX_CALLS, "a"),
    (0, 0, ""), (0, 0, "x" * 129),
])
def test_event_bounds(tick, sequence, call_id):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        InputEvent(tick, sequence, call_id)


@pytest.mark.parametrize("events", [
    (InputEvent(0, 1, "a"),),
    (InputEvent(1, 0, "a"), InputEvent(0, 1, "b")),
    (InputEvent(0, 0, "a"), InputEvent(0, 0, "b")),
    (InputEvent(0, 0, "a"), InputEvent(0, 1, "a")),
    (None,), "a", [InputEvent(0, 0, "a")] * (MAX_CALLS + 1),
])
def test_queue_rejects_invalid_or_ambiguous_order(events):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EventQueue(events)


def test_call_store_and_cross_references(source):
    a, b = source.call_store.calls
    for calls in ((b, a), (a, a), (None,), (a,) * (MAX_CALLS + 1),
                  (replace(a, request=replace(a.request, payload=b"x" * (MAX_PAYLOAD_BYTES + 1))),)):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            CallStore(calls)
    for queue in (EventQueue(()), EventQueue((InputEvent(0, 0, "unknown"),))):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            replace(source, event_queue=queue)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source, program_store={})


@pytest.mark.parametrize("cls", [ProgramStore, CallStore, InputEvent, EventQueue, CrossbarFabricAdapter, SwitchInput])
@pytest.mark.parametrize("value", [None, [], {}, {"schema": "unknown"}])
def test_malformed_serialization(cls, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        cls.from_dict(value)


@pytest.mark.parametrize("path,value", [
    (("program_store", "policy"), "execute-arbitrary-code"),
    (("program_store", "adapter_id"), "unregistered"),
    (("program_store", "operation"), "force_connect"),
    (("program_store", "max_search_evaluations"), True),
    (("event_queue", "events", 0, "kind"), "release"),
    (("event_queue", "events", 0, "logical_tick"), 0.0),
    (("adapter", "connection_commit_present"), True),
    (("adapter", "state_semantics"), "mutable"),
])
def test_rehashed_input_cannot_expand_authority(source, path, value):
    raw = source.as_dict()
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    # Floats cannot even be signed using the canonical hash helper.
    if type(value) is not float:
        raw = rehash(raw)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        SwitchInput.from_dict(raw)


@pytest.mark.parametrize("mutation", ["omit", "swap", "outcome", "command", "boundary", "count", "native", "extra"])
def test_rehashed_evidence_tampering(receipt, mutation):
    raw = copy.deepcopy(receipt)
    if mutation == "omit":
        raw["results"].pop()
    elif mutation == "swap":
        raw["results"].reverse()
    elif mutation == "outcome":
        raw["results"][0]["outcome"] = "plan_selected"
    elif mutation == "command":
        raw["command_stream"]["commands"][0]["logical_tick"] = 1
    elif mutation == "boundary":
        raw["claim_boundary"]["connection_commit_present"] = True
    elif mutation == "count":
        raw["selected_count"] = True
    elif mutation == "native":
        raw["results"][0]["continuity_receipt"] = raw["results"][1]["continuity_receipt"]
    else:
        raw["allow_override"] = True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_switch_receipt(rehash(raw))


@pytest.mark.parametrize("change", ["request", "fabric", "program", "invalid"])
def test_adapter_cannot_substitute_evidence_or_rewrite_program(source, monkeypatch, change):
    original = CrossbarFabricAdapter.plan_route

    def wrong(self, request, program):
        if change == "invalid":
            return {"all_passed": True, "outcome": "plan_selected"}
        if change == "request":
            request = replace(request, request=replace(request.request, payload=b"substituted"))
        fabric = self.fabric
        if change == "fabric":
            fabric = replace(fabric, fabric_id="different")
        altered = compile_path_program(fabric.as_dict(), request,
                                       marker_id="other" if change == "program" else program["marker_id"])
        if change == "program":
            program.clear()
            program.update(altered)
        return execute_path_program(fabric.as_dict(), request, altered)

    monkeypatch.setattr(CrossbarFabricAdapter, "plan_route", wrong)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_switch(source)
    monkeypatch.setattr(CrossbarFabricAdapter, "plan_route", original)


def test_identity_separation_and_trusted_pins(source, receipt):
    changed = replace(source, program_store=replace(source.program_store, program_version="2"))
    other = execute_switch(changed)
    assert changed.call_store.as_dict() == source.call_store.as_dict()
    assert changed.event_queue.as_dict() == source.event_queue.as_dict()
    assert other["sha256"] != receipt["sha256"]
    assert other["command_stream"]["sha256"] != receipt["command_stream"]["sha256"]
    assert validate_switch_receipt(other)["all_passed"]
    pins = {"expected_input_sha256": source.as_dict()["sha256"],
            "expected_program_store_sha256": source.program_store.as_dict()["sha256"],
            "expected_adapter_sha256": source.adapter.as_dict()["sha256"]}
    assert all(validate_switch_receipt(receipt, **pins)["external_bindings_verified"].values())
    for name in pins:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_switch_receipt(receipt, **{name: "0" * 64})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_switch_receipt(other, expected_input_sha256=pins["expected_input_sha256"])
    a, b = source.call_store.calls
    payload = replace(source, call_store=CallStore((replace(a, request=replace(a.request, payload=b"changed")), b)))
    assert payload.program_store.as_dict() == source.program_store.as_dict()
    assert execute_switch(payload)["command_stream"]["sha256"] != receipt["command_stream"]["sha256"]


def test_frozen_fixture(source, receipt):
    fixture = json.loads((Path(__file__).parent / "fixtures/skeleton-v1.json").read_text())
    assert source.as_dict() == fixture["input"]
    assert receipt["sha256"] == fixture["receipt_sha256"]
    assert receipt["command_stream"]["sha256"] == fixture["command_stream_sha256"]
    assert [r["sha256"] for r in receipt["results"]] == fixture["result_hashes"]


@pytest.mark.parametrize("seed", ["0", "42", "random"])
def test_hash_seed_independence(seed, receipt):
    script = "from qec.routing.ess import *; print(execute_switch(demo_switch_input())['sha256'])"
    run = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed},
                         text=True, capture_output=True, check=True)
    assert run.stdout.strip() == receipt["sha256"]


def test_cli_roundtrip_and_fail_closed(tmp_path, capsys):
    assert main(["demo", "--output-dir", str(tmp_path)]) == 0
    input_path = tmp_path / "ess_switch_input.json"
    assert main(["run", "--input", str(input_path), "--output-dir", str(tmp_path)]) == 0
    receipt_path = tmp_path / "ess_switch_skeleton_receipt.json"
    assert main(["validate", "--receipt", str(receipt_path)]) == 0
    assert main(["validate", "--receipt", str(receipt_path), "--expected-input-sha256", "0" * 64]) == 1
    raw = json.loads(receipt_path.read_text())
    assert json.loads((tmp_path / "ess_command_stream.json").read_text()) == raw["command_stream"]
    original = receipt_path.read_bytes()
    assert main(["run", "--input", str(input_path), "--output-dir", str(tmp_path)]) == 0
    assert receipt_path.read_bytes() == original
    input_path.write_text('{"schema":1,"schema":2}')
    assert main(["run", "--input", str(input_path)]) == 1
    assert main(["validate", "--receipt", str(tmp_path / "missing")]) == 1
    assert capsys.readouterr().err
