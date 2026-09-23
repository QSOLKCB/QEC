# SPDX-License-Identifier: MPL-2.0
"""Owned contention state transitions and independently checked allocation invariants."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.crossbar import (
    ContentionBatch, ContentionCommand, CrossbarResource, CrossbarRequest,
    MultiStageRequest, CrossbarFabric, demo_matrix, demo_fabric,
    compile_contention_program, execute_contention_program,
    validate_contention_receipt, demo_contention_batch,
)
from qec.routing.crossbar.cli import main
from qec.routing.crossbar.contention import (
    MAX_COMMANDS, MAX_BATCH_SEARCH_EVALUATIONS,
)
from qec.sonify.canonical import canonical_json, canonical_sha256


def intent(name="r", horizontal="H000", vertical="V001"):
    return MultiStageRequest("ingress", "egress", CrossbarRequest(
        name, horizontal, vertical, b"\x00opaque\xff", "a" * 64))


def reserve(tick=0, name="a", request=None):
    return ContentionCommand(tick, name, "reserve", request=intent() if request is None else request)


def release(tick=1, name="release-a", target="a", request=None):
    return ContentionCommand(tick, name, "release", reservation_id=target,
        expected_request_sha256=(intent() if request is None else request).as_dict()["sha256"])


def quarantine(tick=1, name="quarantine", resource=None):
    return ContentionCommand(tick, name, "quarantine", resource=resource or CrossbarResource(
        "horizontal", "ingress", "H000"))


def batch(*commands, fabric=None):
    return ContentionBatch(demo_fabric() if fabric is None else fabric,
                           tuple(sorted(commands, key=lambda c: c.order_key)))


def run(b=None, budget=MAX_BATCH_SEARCH_EVALUATIONS):
    b = demo_contention_batch() if b is None else b
    return execute_contention_program(b, compile_contention_program(b, max_search_evaluations=budget))


def resign(value):
    value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
    return value


def outcomes(receipt):
    return [x["outcome"] for x in receipt["results"]]


def assert_ledger(state):
    """Independent ownership invariant: exact disjoint whole reservation coverage."""
    owners = {}
    for reservation in state["active_reservations"]:
        assert len(reservation["resource_ids"]) == 3 * len(reservation["path_plan"]["coordinates"]) - 1
        for rid in reservation["resource_ids"]:
            assert rid not in owners
            owners[rid] = reservation["reservation_id"]
    for row in state["resources"]:
        assert row["reservation_id"] == owners.get(row["resource_id"])
        if row["reservation_id"] is not None:
            assert row["state"] == "busy"


def test_demo_contention_release_retry_and_quarantine_revocation():
    b = demo_contention_batch()
    before = b.as_dict()
    receipt = run(b)
    assert outcomes(receipt) == ["reserved", "rejected", "released", "reserved", "quarantined", "rejected"]
    first, blocked, released, retry, fault, last = receipt["results"]
    assert len(first["reservation"]["resource_ids"]) == 8
    assert len(blocked["blocked_resources"]) == 9  # eight held links plus unavailable exit-a
    assert {r["reservation_id"] for r in blocked["blocked_resources"] if r["state"] == "busy"} == {"a"}
    assert {r["owner_request_sha256"] for r in blocked["blocked_resources"] if r["state"] == "busy"} == {first["reservation"]["request_sha256"]}
    assert blocked["state_before_sha256"] == blocked["state_after_sha256"]
    assert released["state_after_sha256"] == receipt["initial_state"]["sha256"]
    assert retry["reservation"]["path_plan"] == first["reservation"]["path_plan"]
    assert fault["revoked_reservation"] == retry["reservation"]
    assert last["reservation"] is None
    assert receipt["final_state"]["active_reservations"] == []
    assert all(r["reservation_id"] is None for r in receipt["final_state"]["resources"])
    assert b.as_dict() == before
    assert validate_contention_receipt(receipt)["all_passed"] is True
    assert_ledger(receipt["final_state"])


def test_request_hash_precedes_command_id_for_same_tick_tie():
    candidates = [reserve(name="z", request=intent("alpha")), reserve(name="a", request=intent("beta"))]
    candidates.sort(key=lambda c: c.request.as_dict()["sha256"])
    candidates[0] = replace(candidates[0], command_id="z-winner")
    candidates[1] = replace(candidates[1], command_id="a-loser")
    receipt = run(batch(*reversed(candidates)))
    assert [r["command_id"] for r in receipt["results"]] == ["z-winner", "a-loser"]
    assert outcomes(receipt) == ["reserved", "rejected"]


def test_same_request_tie_uses_command_id():
    receipt = run(batch(reserve(name="z"), reserve(name="a")))
    assert [r["command_id"] for r in receipt["results"]] == ["a", "z"]
    assert outcomes(receipt) == ["reserved", "rejected"]


def test_same_tick_release_then_quarantine_then_reserve():
    receipt = run(batch(reserve(), reserve(1, "retry"), quarantine(), release()))
    assert [r["kind"] for r in receipt["results"]] == ["reserve", "release", "quarantine", "reserve"]
    assert outcomes(receipt) == ["reserved", "released", "quarantined", "rejected"]
    assert receipt["results"][2]["revoked_reservation"] is None


def test_release_before_same_tick_creation_rejects_forward_reference():
    receipt = run(batch(reserve(1), release(1)))
    assert outcomes(receipt) == ["rejected", "reserved"]
    assert receipt["results"][0]["reason"] == "reservation_not_active"


@pytest.mark.parametrize("command,reason", [
    (release(target="absent"), "reservation_not_active"),
    (replace(release(), expected_request_sha256="b" * 64), "reservation_owner_mismatch"),
])
def test_invalid_release_cannot_clear_owned_resources(command, reason):
    receipt = run(batch(reserve(), command))
    assert receipt["results"][-1]["reason"] == reason
    assert receipt["results"][-1]["state_before_sha256"] == receipt["results"][-1]["state_after_sha256"]
    assert len(receipt["final_state"]["active_reservations"]) == 1
    assert_ledger(receipt["final_state"])


def test_duplicate_release_does_not_release_another_reservation():
    receipt = run(batch(reserve(), release(), reserve(1, "retry"), release(2, "stale")))
    assert outcomes(receipt) == ["reserved", "released", "reserved", "rejected"]
    assert receipt["final_state"]["active_reservations"][0]["reservation_id"] == "retry"
    assert_ledger(receipt["final_state"])


@pytest.mark.parametrize("resource", [
    CrossbarResource("horizontal", "ingress", "H000"),
    CrossbarResource("vertical", "middle-b", "V000"),
    CrossbarResource("interstage", None, "entry-b"),
])
def test_quarantine_any_held_resource_revokes_whole_path(resource):
    receipt = run(batch(reserve(), quarantine(resource=resource), release(2)))
    assert outcomes(receipt) == ["reserved", "quarantined", "rejected"]
    assert len(receipt["results"][1]["revoked_reservation"]["resource_ids"]) == 8
    row = next(r for r in receipt["final_state"]["resources"] if r["resource_id"] == resource.resource_id)
    assert row["state"] == "quarantined" and row["reservation_id"] is None
    assert not receipt["final_state"]["active_reservations"]
    assert_ledger(receipt["final_state"])


@pytest.mark.parametrize("resource,reason", [
    (CrossbarResource("interstage", None, "unknown"), "unknown_resource"),
    (CrossbarResource("horizontal", "unknown", "H000"), "unknown_resource"),
    (CrossbarResource("vertical", "ingress", "H000"), "unknown_resource"),
    (CrossbarResource("interstage", None, "exit-a"), "resource_unavailable"),
])
def test_invalid_quarantine_does_not_change_state(resource, reason):
    receipt = run(batch(quarantine(resource=resource)))
    assert receipt["results"][0]["reason"] == reason
    assert receipt["final_state"] == receipt["initial_state"]


def test_quarantine_is_sticky_and_duplicate_is_explicit():
    receipt = run(batch(quarantine(0, "q0"), quarantine(1, "q1"), reserve(2)))
    assert outcomes(receipt) == ["quarantined", "rejected", "rejected"]
    assert receipt["results"][1]["reason"] == "resource_already_quarantined"
    assert receipt["results"][2]["blocked_resources"][0]["state"] == "quarantined"


def test_alternate_route_allows_disjoint_reservation():
    fabric = demo_fabric(dead_end_first=False)
    receipt = run(batch(reserve(), reserve(1, "other", intent("other", "H001", "V000")), fabric=fabric))
    assert outcomes(receipt) == ["reserved", "reserved"]
    a, b = [r["reservation"] for r in receipt["results"]]
    assert a["path_plan"]["interstage_link_ids"] == ["entry-a", "exit-a"]
    assert b["path_plan"]["interstage_link_ids"] == ["entry-b", "exit-b"]
    assert set(a["resource_ids"]).isdisjoint(b["resource_ids"])
    assert_ledger(receipt["final_state"])


def test_quarantine_unowned_link_does_not_revoke_another_path():
    receipt = run(batch(reserve(), quarantine(resource=CrossbarResource("interstage", None, "entry-a"))))
    assert receipt["results"][1]["revoked_reservation"] is None
    assert len(receipt["final_state"]["active_reservations"]) == 1
    assert_ledger(receipt["final_state"])


def test_releasing_one_disjoint_path_preserves_the_other():
    receipt = run(batch(reserve(), reserve(1, "other", intent("other", "H001", "V000")),
                        release(2), fabric=demo_fabric(dead_end_first=False)))
    assert outcomes(receipt) == ["reserved", "reserved", "released"]
    assert [r["reservation_id"] for r in receipt["final_state"]["active_reservations"]] == ["other"]
    assert_ledger(receipt["final_state"])


@pytest.mark.parametrize("state", ["busy", "quarantined", "unavailable"])
def test_initial_blocked_endpoint_has_no_fabricated_owner(state):
    fabric = demo_fabric()
    ingress = fabric.stages[0][0]
    ingress = replace(ingress, horizontal_links=(replace(ingress.horizontal_links[0], state=state), ingress.horizontal_links[1]))
    fabric = replace(fabric, stages=((ingress,), *fabric.stages[1:]))
    receipt = run(batch(reserve(), release(), fabric=fabric))
    assert outcomes(receipt) == ["rejected", "rejected"]
    resource = receipt["results"][0]["blocked_resources"][0]
    assert resource["state"] == state
    assert resource["reservation_id"] is None and resource["owner_request_sha256"] is None
    assert receipt["final_state"] == receipt["initial_state"]


@pytest.mark.parametrize("budget,expected", [(19, ["rejected", "rejected", "rejected"]),
    (20, ["reserved", "released", "rejected"]), (39, ["reserved", "released", "rejected"]),
    (40, ["reserved", "released", "reserved"])])
def test_batch_budget_cannot_be_reset_per_request_and_releases_still_execute(budget, expected):
    receipt = run(batch(reserve(), release(), reserve(2, "retry")), budget)
    assert outcomes(receipt) == expected
    assert receipt["search_evaluations"] == budget
    assert receipt["results"][-1]["reason"] == ({19: "batch_search_budget_exhausted", 20: "batch_search_budget_exhausted", 39: "search_budget_exhausted", 40: "complete_path_reserved"}[budget])
    assert validate_contention_receipt(receipt)["all_passed"]
    assert_ledger(receipt["final_state"])


def test_quarantine_executes_after_search_budget_exhaustion():
    receipt = run(batch(reserve(), quarantine()), 20)
    assert outcomes(receipt) == ["reserved", "quarantined"]
    assert not receipt["final_state"]["active_reservations"]


def test_unknown_endpoint_consumes_no_search_budget():
    receipt = run(batch(reserve(request=replace(intent(), source_matrix_id="unknown"))))
    assert receipt["results"][0]["reason"] == "unknown_source_matrix"
    assert receipt["search_evaluations"] == 0
    assert receipt["initial_state"] == receipt["final_state"]


def reference_demo(commands, dead_end_first):
    """Small independent forward-route ledger oracle for the two-branch demo."""
    owners, reservations, faults, expected = {}, {}, set(), []
    blocked = {("interstage", None, "exit-a")} if dead_end_first else set()
    for command in commands:
        if command.kind == "reserve":
            path = None
            for suffix, middle, source_v, destination_h in (
                ("a", "middle-a", "V000", "H000"), ("b", "middle-b", "V001", "H001")):
                candidate = {
                    ("horizontal", "ingress", command.request.request.horizontal_link_id),
                    ("vertical", "ingress", source_v),
                    ("horizontal", middle, "H000"), ("vertical", middle, "V000"),
                    ("horizontal", "egress", destination_h),
                    ("vertical", "egress", command.request.request.vertical_link_id),
                    ("interstage", None, "entry-" + suffix), ("interstage", None, "exit-" + suffix),
                }
                if not (candidate & (blocked | faults | set(owners))):
                    path = candidate
                    break
            if path is None:
                expected.append("rejected")
            else:
                expected.append("reserved")
                reservations[command.command_id] = path
                owners.update({r: command.command_id for r in path})
        elif command.kind == "release":
            path = reservations.pop(command.reservation_id, None)
            expected.append("released" if path else "rejected")
            if path:
                for r in path: del owners[r]
        else:
            r = (command.resource.kind, command.resource.matrix_id, command.resource.link_id)
            if r in owners:
                for held in reservations.pop(owners[r]): del owners[held]
            faults.add(r)
            expected.append("quarantined")
    return expected, owners, faults


def test_every_prefix_has_atomic_owned_state():
    """Exercise every prefix of 64 deterministic release/fault/endpoint scenarios."""
    for h, v, do_release, do_quarantine, alternate, at_same_tick in product((False, True), repeat=6):
        commands = [reserve()]
        if do_release:
            commands.append(release())
        if do_quarantine:
            commands.append(quarantine(1 if at_same_tick else 2))
        commands.append(reserve(1 if at_same_tick else 3, "other",
            intent("other", "H001" if h else "H000", "V000" if v else "V001")))
        commands.sort(key=lambda c: c.order_key)
        for end in range(1, len(commands) + 1):
            receipt = run(batch(*commands[:end], fabric=demo_fabric(dead_end_first=not alternate)))
            assert_ledger(receipt["final_state"])
            expected, owners, faults = reference_demo(commands[:end], dead_end_first=not alternate)
            assert outcomes(receipt) == expected
            for row in receipt["final_state"]["resources"]:
                resource = row["resource"]
                key = (resource["kind"], resource["matrix_id"], resource["link_id"])
                assert row["reservation_id"] == owners.get(key)
                if key in faults: assert row["state"] == "quarantined"
            for result in receipt["results"]:
                if result["outcome"] == "rejected":
                    assert result["state_before_sha256"] == result["state_after_sha256"]
                    assert result["reservation"] is None


@pytest.mark.parametrize("value", [0, -1, True, 1.0, None, MAX_BATCH_SEARCH_EVALUATIONS + 1])
def test_invalid_budget(value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        compile_contention_program(demo_contention_batch(), max_search_evaluations=value)


@pytest.mark.parametrize("fields", [
    {"tick": True}, {"tick": -1}, {"tick": 2**31}, {"tick": 0.0}, {"command_id": ""},
    {"command_id": "\ud800"}, {"command_id": []}, {"kind": "commit"},
    {"request": None}, {"reservation_id": "x"}, {"expected_request_sha256": "a" * 64},
    {"resource": CrossbarResource("interstage", None, "entry-a")},
])
def test_invalid_reserve_fields(fields):
    with pytest.raises(ValueError):
        replace(reserve(), **fields)


@pytest.mark.parametrize("fields", [
    {"reservation_id": None}, {"expected_request_sha256": None}, {"expected_request_sha256": "A" * 64},
    {"request": intent()}, {"resource": CrossbarResource("interstage", None, "entry-a")},
])
def test_invalid_release_fields(fields):
    with pytest.raises(ValueError):
        replace(release(), **fields)


@pytest.mark.parametrize("fields", [
    {"resource": None}, {"reservation_id": "a"}, {"expected_request_sha256": "a" * 64}, {"request": intent()},
])
def test_invalid_quarantine_fields(fields):
    with pytest.raises(ValueError):
        replace(quarantine(), **fields)


@pytest.mark.parametrize("args", [("interstage", "ingress", "entry-a"), ("horizontal", None, "H000"),
    ("horizontal", "ingress", ""), ("coordinate", "ingress", "H000"), (True, None, "x")])
def test_invalid_resource(args):
    with pytest.raises(ValueError):
        CrossbarResource(*args)


def test_batch_rejects_duplicate_ids_noncanonical_order_and_extra_fields():
    with pytest.raises(ValueError):
        ContentionBatch(demo_fabric(), (reserve(), reserve()))
    with pytest.raises(ValueError):
        ContentionBatch(demo_fabric(), (reserve(1), reserve(0, "b")))
    for key in ("commands", "fabric_manifest", "extra"):
        obj = demo_contention_batch().as_dict()
        if key == "commands": obj[key].reverse()
        elif key == "fabric_manifest": obj[key]["fabric_id"] = "changed"
        else: obj[key] = True
        with pytest.raises(ValueError):
            ContentionBatch.from_dict(resign(obj))


def test_batch_bounds_and_immutable_copy():
    commands = [reserve()]
    b = ContentionBatch(demo_fabric(), commands)
    commands.clear()
    assert len(b.commands) == 1
    with pytest.raises(FrozenInstanceError): b.commands = ()
    for count in (0, MAX_COMMANDS + 1):
        with pytest.raises(ValueError):
            batch(*(reserve(i, str(i)) for i in range(count)))
    assert len(batch(*(reserve(i, str(i)) for i in range(MAX_COMMANDS))).commands) == MAX_COMMANDS
    big = CrossbarFabric("big", ((demo_matrix("a", horizontal_count=64, vertical_count=64),),
                                  (demo_matrix("b", horizontal_count=1, vertical_count=1),)), ())
    with pytest.raises(ValueError): batch(reserve(), fabric=big)
    wide = CrossbarFabric("wide", ((demo_matrix("a", horizontal_count=1, vertical_count=4095),),
                                   (demo_matrix("b", horizontal_count=1, vertical_count=1),)), ())
    with pytest.raises(ValueError): batch(reserve(), fabric=wide)
    payload = replace(intent().request, payload=b"x" * (1_048_576 // 2 + 1))
    heavy = replace(intent(), request=payload)
    with pytest.raises(ValueError): batch(reserve(request=heavy), reserve(1, "b", heavy))


@pytest.mark.parametrize("field,value", [("policy", "last-wins"), ("max_search_evaluations", True),
    ("input_sha256", "b" * 64), ("contract_version", "172.2"), ("extra", True)])
def test_resigned_program_cannot_change_contract(field, value):
    b = demo_contention_batch()
    p = compile_contention_program(b)
    p[field] = value
    with pytest.raises(ValueError): execute_contention_program(b, resign(p))


@pytest.mark.parametrize("case", ["partial", "owner", "missing_busy", "fake_release", "resurrect", "event",
    "budget", "reorder", "numeric_bool", "marker", "claim", "search_snapshot", "release_identity"])
def test_rehashed_receipt_tampering_is_rejected(case):
    r = deepcopy(run())
    if case == "partial": r["results"][0]["reservation"]["resource_ids"].pop()
    elif case == "owner": r["results"][1]["blocked_resources"][0]["reservation_id"] = "b"
    elif case == "missing_busy": r["results"][1]["blocked_resources"].pop()
    elif case == "fake_release": r["results"][1]["outcome"] = "released"
    elif case == "resurrect": r["final_state"]["active_reservations"] = [r["results"][0]["reservation"]]
    elif case == "event": r["events"][0]["details"] = {}; resign(r["events"][0])
    elif case == "budget": r["search_evaluations"] -= 1
    elif case == "reorder": r["results"].reverse()
    elif case == "numeric_bool": r["results"][0]["tick"] = False
    elif case == "marker": r["results"][0]["marker_released"] = False
    elif case == "claim": r["claim_boundary"]["connection_commit_present"] = True
    elif case == "search_snapshot": r["results"][1]["path_search_receipt"] = r["results"][0]["path_search_receipt"]
    elif case == "release_identity": r["input"]["commands"][2]["expected_request_sha256"] = "b" * 64
    with pytest.raises(ValueError): validate_contention_receipt(resign(r))


def test_event_chain_state_chain_and_trusted_input_bindings():
    receipt = run()
    previous = receipt["event_root_sha256"]
    for i, event in enumerate(receipt["events"]):
        assert event["sequence"] == i and event["previous_sha256"] == previous
        assert event["sha256"] == canonical_sha256({k: v for k, v in event.items() if k != "sha256"})
        previous = event["sha256"]
    previous = receipt["initial_state"]["sha256"]
    for result in receipt["results"]:
        assert result["state_before_sha256"] == previous
        previous = result["state_after_sha256"]
    assert previous == receipt["final_state"]["sha256"]
    assert len([e for e in receipt["events"] if e["kind"] == "marker_released"]) == len(receipt["results"])
    pins = dict(expected_input_sha256=receipt["input"]["sha256"],
                expected_program_sha256=receipt["program"]["sha256"],
                expected_fabric_sha256=receipt["input"]["fabric_manifest"]["sha256"])
    assert all(validate_contention_receipt(receipt, **pins)["external_bindings_verified"].values())
    for key in pins:
        with pytest.raises(ValueError): validate_contention_receipt(receipt, **{**pins, key: "b" * 64})
    changed = run(batch(reserve(request=intent("different"))))
    assert validate_contention_receipt(changed)["all_passed"]
    with pytest.raises(ValueError): validate_contention_receipt(changed, expected_input_sha256=pins["expected_input_sha256"])


def test_subprocess_hash_seed_determinism():
    script = '''from qec.routing.crossbar import *
from qec.sonify.canonical import canonical_json
b=demo_contention_batch()
print(canonical_json(execute_contention_program(b,compile_contention_program(b))))'''
    outputs = [subprocess.check_output([sys.executable, "-c", script],
               env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ("0", "913")]
    assert outputs[0] == outputs[1] == (canonical_json(run()) + "\n").encode()


def test_cli_roundtrip_and_invalid_inputs(tmp_path, capsys):
    assert main(["contention-demo", "--output-dir", str(tmp_path)]) == 0
    input_path = tmp_path / "crossbar_contention_input.json"
    assert main(["contend", "--input", str(input_path), "--output-dir", str(tmp_path)]) == 0
    receipt_path = tmp_path / "crossbar_contention_receipt.json"
    assert main(["contention-validate", "--receipt", str(receipt_path)]) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt == run()
    assert main(["contention-validate", "--receipt", str(receipt_path), "--expected-input-sha256", "b" * 64]) == 1
    assert main(["contend", "--input", str(input_path), "--max-search-evaluations", "0"]) == 1
    input_path.write_text('{"commands": [], "commands": []}')
    assert main(["contend", "--input", str(input_path)]) == 1
    receipt_path.write_text(canonical_json(resign({**receipt, "search_evaluations": 0})))
    assert main(["contention-validate", "--receipt", str(receipt_path)]) == 1
    capsys.readouterr()


def test_resource_id_is_qualified_and_contract_bound():
    resources = [CrossbarResource("horizontal", "a", "same"), CrossbarResource("vertical", "a", "same"),
                 CrossbarResource("horizontal", "b", "same"), CrossbarResource("interstage", None, "same")]
    assert len({r.resource_id for r in resources}) == 4
    for resource in resources:
        unsigned = {"schema": "qec.crossbar-resource-id.v1", "contract_version": "172.3", **resource.as_dict()}
        assert resource.resource_id == canonical_sha256(unsigned)


@pytest.mark.parametrize("kind", ["reserve", "release", "quarantine"])
def test_command_parser_rejects_extra_authority_and_changed_hash(kind):
    command = {"reserve": reserve(), "release": release(), "quarantine": quarantine()}[kind]
    for change in ("extra", "hash", "schema"):
        value = command.as_dict()
        if change == "extra": value["force"] = True; resign(value)
        elif change == "hash": value["sha256"] = "b" * 64
        else: value["schema"] = "qec.other.v1"; resign(value)
        with pytest.raises(ValueError): ContentionCommand.from_dict(value)


def test_maximum_batch_executes_with_one_owner_and_complete_rejections():
    receipt = run(batch(*(reserve(i, str(i)) for i in range(MAX_COMMANDS))))
    assert outcomes(receipt) == ["reserved"] + ["rejected"] * (MAX_COMMANDS - 1)
    assert receipt["search_evaluations"] == MAX_COMMANDS * 20
    assert len(receipt["final_state"]["active_reservations"]) == 1
    assert_ledger(receipt["final_state"])


def test_initial_busy_quarantine_does_not_create_or_release_external_owner():
    fabric = demo_fabric()
    wires = tuple(replace(w, state="busy") if w.link_id == "entry-b" else w for w in fabric.interstage_links)
    fabric = replace(fabric, interstage_links=wires)
    receipt = run(batch(quarantine(resource=CrossbarResource("interstage", None, "entry-b")), fabric=fabric))
    assert outcomes(receipt) == ["quarantined"]
    assert receipt["results"][0]["revoked_reservation"] is None
    assert not receipt["final_state"]["active_reservations"]
    row = next(r for r in receipt["final_state"]["resources"] if r["resource"]["link_id"] == "entry-b")
    assert row["state"] == "quarantined" and row["reservation_id"] is None


def test_frozen_contention_fixture():
    receipt = json.loads((Path(__file__).parent / "fixtures" / "contention-v1.json").read_text())
    b = ContentionBatch.from_dict(receipt["input"])
    assert receipt == execute_contention_program(b, compile_contention_program(b))
    assert validate_contention_receipt(receipt)["all_passed"] is True
    assert outcomes(receipt) == ["reserved", "rejected", "released", "reserved", "quarantined", "rejected"]
