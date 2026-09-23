# SPDX-License-Identifier: MPL-2.0
"""v172.2 selection compared with exhaustive paths, plus contract regressions."""
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
    CrossbarFabric, CrossbarLink, CrossbarMatrix, CrossbarRequest, InterstageLink,
    MultiStageRequest, compile_path_program, demo_fabric, demo_matrix,
    execute_path_program, validate_path_search_receipt,
)
from qec.routing.crossbar.cli import main
from qec.routing.crossbar.multistage import MAX_SEARCH_EVALUATIONS
from qec.sonify.canonical import canonical_json, canonical_sha256


def request():
    return MultiStageRequest("ingress", "egress",
                             CrossbarRequest("r", "H000", "V001", b"\x00opaque\xff", "a" * 64))


def run(fabric=None, intent=None, budget=MAX_SEARCH_EVALUATIONS):
    fabric = demo_fabric() if fabric is None else fabric
    intent = request() if intent is None else intent
    manifest = fabric.as_dict()
    program = compile_path_program(manifest, intent, max_search_evaluations=budget)
    return execute_path_program(manifest, intent, program)


def resign(value):
    value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
    return value


def exhaustive_paths(fabric, intent):
    """Independent forward DFS oracle, intentionally unsuitable for large fabrics."""
    matrices = {m.matrix_id: m for stage in fabric.stages for m in stage}

    def visit(stage, matrix_id, horizontal_id, coordinates, wires):
        matrix = matrices[matrix_id]
        horizontal = next(h for h in matrix.horizontal_links if h.link_id == horizontal_id)
        if horizontal.state != "idle":
            return
        for vertical in matrix.vertical_links:
            if vertical.state != "idle":
                continue
            path = coordinates + [{"stage": stage, "matrix_id": matrix_id,
                                   "coordinate": matrix.coordinate(horizontal_id, vertical.link_id).as_dict()}]
            if stage == len(fabric.stages) - 1:
                if (matrix_id == intent.destination_matrix_id
                        and vertical.link_id == intent.request.vertical_link_id):
                    yield {"coordinates": path, "interstage_link_ids": wires}
            else:
                for wire in fabric.interstage_links:
                    if (wire.source_matrix_id == matrix_id
                            and wire.source_vertical_link_id == vertical.link_id
                            and wire.state == "idle"):
                        yield from visit(stage + 1, wire.target_matrix_id,
                                         wire.target_horizontal_link_id, path, wires + [wire.link_id])
    return list(visit(0, intent.source_matrix_id, intent.request.horizontal_link_id, [], []))


def test_lookahead_avoids_a_locally_free_dead_end():
    fabric = demo_fabric()
    before = fabric.as_dict()
    receipt = run(fabric)
    assert receipt["path_plan"]["interstage_link_ids"] == ["entry-b", "exit-b"]
    assert [c["matrix_id"] for c in receipt["path_plan"]["coordinates"]] == ["ingress", "middle-b", "egress"]
    assert receipt["search_complete"] is True
    assert receipt["search_evaluations"] == 20
    assert any(e["details"].get("link_id") == "entry-a"
               and e["details"].get("reason") == "suffix_unreachable" for e in receipt["events"])
    assert fabric.as_dict() == before
    assert receipt["fabric_after_sha256"] == before["sha256"]
    assert receipt["input_register"]["request"]["contract_version"] == "172.1"
    assert receipt["fabric_manifest"]["stages"][0][0]["contract_version"] == "172.0"
    assert receipt["events"][-1]["kind"] == "marker_released"
    assert validate_path_search_receipt(receipt)["first_complete_path_verified"] is True
    assert canonical_json(receipt) == canonical_json(run(fabric))


def test_first_complete_path_when_both_branches_are_available():
    fabric = demo_fabric(dead_end_first=False)
    paths = exhaustive_paths(fabric, request())
    assert len(paths) == 2
    assert run(fabric)["path_plan"] == paths[0]
    assert paths[0]["interstage_link_ids"] == ["entry-a", "exit-a"]


def test_selection_matches_exhaustive_oracle_for_256_availability_assignments():
    base = demo_fabric(dead_end_first=False)
    for mask in product((False, True), repeat=8):
        links = tuple(replace(w, state="busy" if mask[i] else "idle")
                      for i, w in enumerate(base.interstage_links))
        stages = []
        for si, stage in enumerate(base.stages):
            changed = []
            for mi, matrix in enumerate(stage):
                horizontal = list(matrix.horizontal_links)
                vertical = list(matrix.vertical_links)
                # Exercise source, both branch inputs, and the final destination.
                if si == 0 and mask[4]:
                    horizontal[0] = replace(horizontal[0], state="unavailable")
                if si == 1 and mask[5 + mi]:
                    horizontal[0] = replace(horizontal[0], state="quarantined")
                if si == 2 and mask[7]:
                    vertical[1] = replace(vertical[1], state="busy")
                changed.append(CrossbarMatrix(matrix.matrix_id, horizontal, vertical))
            stages.append(changed)
        fabric = CrossbarFabric("oracle", stages, links)
        paths = exhaustive_paths(fabric, request())
        receipt = run(fabric)
        assert receipt["path_plan"] == (paths[0] if paths else None), mask
        assert receipt["outcome"] == ("plan_selected" if paths else "rejected"), mask
        assert receipt["search_complete"] is True
        assert receipt["search_evaluations"] == 20


def test_wire_priority_uses_matrix_order_before_link_name():
    base = demo_fabric(dead_end_first=False)
    links = (
        replace(base.interstage_links[0], link_id="zzz"),
        replace(base.interstage_links[1], link_id="aaa", source_vertical_link_id="V000"),
        *base.interstage_links[2:],
    )
    fabric = CrossbarFabric("tie", base.stages, links)
    assert run(fabric)["path_plan"]["interstage_link_ids"][0] == "zzz"
    blocked = replace(fabric, interstage_links=(*links[:2], replace(links[2], state="busy"), links[3]))
    assert run(blocked)["path_plan"]["interstage_link_ids"][0] == "aaa"


@pytest.mark.parametrize("state", ["busy", "quarantined", "unavailable"])
def test_interstage_link_states_can_block_every_complete_path(state):
    base = demo_fabric()
    fabric = replace(base, interstage_links=tuple(replace(w, state=state) for w in base.interstage_links))
    receipt = run(fabric)
    assert receipt["reason"] == "no_admissible_complete_path"
    assert receipt["path_plan"] is None
    validation = validate_path_search_receipt(receipt)
    assert validation["all_passed"] is True
    assert validation["first_complete_path_verified"] is False


def test_disconnected_fabric_is_valid_but_has_no_path():
    receipt = run(replace(demo_fabric(), interstage_links=()))
    assert receipt["reason"] == "no_admissible_complete_path"
    assert receipt["search_evaluations"] == 16


@pytest.mark.parametrize("budget", [1, 4, 10, 19])
def test_budget_exhaustion_is_distinct_from_no_path(budget):
    receipt = run(budget=budget)
    assert receipt["outcome"] == "rejected"
    assert receipt["reason"] == "search_budget_exhausted"
    assert receipt["search_evaluations"] == budget
    assert receipt["search_complete"] is False
    assert receipt["path_plan"] is None
    assert receipt["marker_released"] is True
    assert len([e for e in receipt["events"] if e["kind"].endswith("_evaluated")]) == budget
    assert validate_path_search_receipt(receipt)["all_passed"] is True


def test_exact_budget_succeeds_without_a_spurious_exhaustion():
    assert run(budget=20)["outcome"] == "plan_selected"


@pytest.mark.parametrize("budget", [0, -1, True, 1.0, None, MAX_SEARCH_EVALUATIONS + 1])
def test_invalid_budget_rejected(budget):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        run(budget=budget)


@pytest.mark.parametrize("change,reason", [
    ({"source_matrix_id": "middle-a"}, "unknown_source_matrix"),
    ({"destination_matrix_id": "ingress"}, "unknown_destination_matrix"),
    ({"request": replace(request().request, horizontal_link_id="V000")}, "unknown_source_horizontal_link"),
    ({"request": replace(request().request, vertical_link_id="H000")}, "unknown_destination_vertical_link"),
])
def test_unknown_endpoints_reject_without_search(change, reason):
    receipt = run(intent=replace(request(), **change))
    assert receipt["reason"] == reason
    assert receipt["search_evaluations"] == 0
    assert receipt["search_complete"] is False
    assert receipt["path_plan"] is None
    assert validate_path_search_receipt(receipt)["all_passed"] is True


@pytest.mark.parametrize("change", [
    {"source_matrix_id": "missing"}, {"target_matrix_id": "ingress"},
    {"target_matrix_id": "egress"}, {"source_vertical_link_id": "H000"},
    {"target_horizontal_link_id": "V000"},
])
def test_malformed_wiring_is_rejected(change):
    base = demo_fabric()
    wires = (replace(base.interstage_links[0], **change), *base.interstage_links[1:])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(base, interstage_links=wires)


@pytest.mark.parametrize("mutation", ["order", "duplicate_id", "duplicate_edge", "extra_field", "state"])
def test_resigned_malformed_fabric_is_rejected(mutation):
    manifest = demo_fabric().as_dict()
    links = manifest["interstage_links"]
    if mutation == "order":
        links.reverse()
    elif mutation == "duplicate_id":
        links[1]["link_id"] = links[0]["link_id"]
    elif mutation == "duplicate_edge":
        copy = deepcopy(links[0]); copy["link_id"] = "other"
        links.insert(1, copy)
    elif mutation == "extra_field":
        manifest["force_accept"] = True
    else:
        links[0]["state"] = "reserved"
    resign(manifest)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        CrossbarFabric.from_dict(manifest)


def test_stage_matrix_order_uniqueness_and_global_bounds():
    base = demo_fabric()
    a, middle, z = base.stages
    bad_stages = [(a,), ((), z), (a, tuple(reversed(middle)), z),
                  (a, a), (a,) * 17,
                  (tuple(demo_matrix(f"m{i:03}", horizontal_count=1, vertical_count=1) for i in range(64)), z),
                  ((demo_matrix("large-a", horizontal_count=256, vertical_count=256),),
                   (demo_matrix("large-b", horizontal_count=256, vertical_count=256),))]
    for stages in bad_stages:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            CrossbarFabric("bad", stages, ())
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(base, interstage_links=(base.interstage_links[0],) * 4097)


def test_maximum_stage_chain_and_minimum_two_stage_fabric():
    for count in (2, 16):
        stages = tuple((demo_matrix(f"stage-{i}", horizontal_count=1, vertical_count=1),) for i in range(count))
        links = tuple(InterstageLink(f"w{i}", f"stage-{i}", "V000", f"stage-{i+1}", "H000")
                      for i in range(count - 1))
        fabric = CrossbarFabric("chain", stages, links)
        intent = MultiStageRequest("stage-0", f"stage-{count-1}",
                                   CrossbarRequest("chain", "H000", "V000", b"", "0" * 64))
        receipt = run(fabric, intent)
        assert len(receipt["path_plan"]["coordinates"]) == count
        assert len(receipt["path_plan"]["interstage_link_ids"]) == count - 1
        assert receipt["search_evaluations"] == 3 * count - 1


def test_fabric_copies_input_sequences_and_request_is_frozen():
    base = demo_fabric()
    stages = [list(stage) for stage in base.stages]
    wires = list(base.interstage_links)
    fabric = CrossbarFabric("copy", stages, wires)
    expected = fabric.as_dict()
    stages[1].clear(); wires.clear()
    assert fabric.as_dict() == expected
    with pytest.raises(FrozenInstanceError):
        request().source_matrix_id = "other"
    with pytest.raises(FrozenInstanceError):
        base.interstage_links[0].state = "busy"


@pytest.mark.parametrize("field,value", [
    ("policy", "greedy"), ("contract_version", "172.1"),
    ("max_search_evaluations", True), ("extra_authority", "commit"),
])
def test_resigned_program_cannot_change_policy_or_contract(field, value):
    manifest = demo_fabric().as_dict()
    program = compile_path_program(manifest, request())
    program[field] = value
    resign(program)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_path_program(manifest, request(), program)


def test_program_binds_request_fabric_and_budget():
    manifest = demo_fabric().as_dict()
    program = compile_path_program(manifest, request())
    other = replace(request(), request=replace(request().request, payload=b"changed"))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_path_program(manifest, other, program)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_path_program(demo_fabric(dead_end_first=False).as_dict(), request(), program)
    assert compile_path_program(manifest, request(), max_search_evaluations=1)["sha256"] != program["sha256"]


@pytest.mark.parametrize("mutation", ["last_path", "partial_path", "skipped_wire", "false_claim", "bool_count", "release", "event", "register", "rejection"])
def test_full_replay_rejects_resigned_fabrications(mutation):
    fabric = demo_fabric(dead_end_first=False)
    receipt = run(fabric)
    if mutation == "last_path":
        receipt["path_plan"] = exhaustive_paths(fabric, request())[-1]
    elif mutation == "partial_path":
        receipt["path_plan"]["coordinates"].pop()
    elif mutation == "skipped_wire":
        receipt["path_plan"]["interstage_link_ids"].pop()
    elif mutation == "false_claim":
        receipt["claim_boundary"]["reservation_present"] = 0
    elif mutation == "bool_count":
        receipt["search_evaluations"] = True
    elif mutation == "release":
        receipt["events"].pop()
    elif mutation == "event":
        receipt["events"][3]["details"]["state"] = "busy"
        resign(receipt["events"][3])
    elif mutation == "register":
        receipt["input_register"]["destination_matrix_id"] = "middle-a"
        resign(receipt["input_register"])
    else:
        receipt["outcome"] = "rejected"
        receipt["reason"] = "no_admissible_complete_path"
        receipt["path_plan"] = None
    resign(receipt)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_path_search_receipt(receipt)


def test_event_chain_and_trusted_external_bindings():
    receipt = run()
    previous = receipt["event_root_sha256"]
    for index, event in enumerate(receipt["events"]):
        assert event["sequence"] == index
        assert event["previous_sha256"] == previous
        assert event["sha256"] == canonical_sha256({k: v for k, v in event.items() if k != "sha256"})
        previous = event["sha256"]
    pins = {"expected_fabric_sha256": receipt["fabric_sha256"],
            "expected_request_sha256": receipt["input_register"]["sha256"],
            "expected_program_sha256": receipt["path_program"]["sha256"]}
    assert all(validate_path_search_receipt(receipt, **pins)["external_bindings_verified"].values())
    for key in pins:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_path_search_receipt(receipt, **{**pins, key: "0" * 64})
    other = run(demo_fabric(dead_end_first=False))
    assert validate_path_search_receipt(other)["all_passed"] is True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_path_search_receipt(other, **pins)


def test_process_and_hash_seed_determinism():
    code = '''from qec.routing.crossbar import *
from qec.sonify.canonical import canonical_json
m=demo_fabric().as_dict()
r=MultiStageRequest("ingress","egress",CrossbarRequest("r","H000","V001",b"opaque","a"*64))
print(canonical_json(execute_path_program(m,r,compile_path_program(m,r))))'''
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={
        **os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(Path("src").resolve())})
        for seed in ("0", "839")]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("budget,exit_code", [(20, 0), (19, 2)])
def test_cli_roundtrip_and_budget_rejection(tmp_path, budget, exit_code):
    assert main(["fabric-demo", "--output-dir", str(tmp_path)]) == 0
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"opaque")
    args = ["path-search", "--fabric", str(tmp_path / "crossbar_fabric_manifest.json"),
            "--source-matrix-id", "ingress", "--destination-matrix-id", "egress",
            "--request-id", "cli", "--horizontal-link-id", "H000", "--vertical-link-id", "V001",
            "--payload-file", str(payload), "--decoder-output-sha256", "a" * 64,
            "--output-dir", str(tmp_path), "--max-search-evaluations", str(budget)]
    assert main(args) == exit_code
    receipt = tmp_path / "crossbar_path_search_receipt.json"
    assert main(["path-validate", "--receipt", str(receipt)]) == 0
    assert main(["path-validate", "--receipt", str(receipt), "--expected-fabric-sha256", "0" * 64]) == 1
    data = json.loads(receipt.read_text())
    data["marker_released"] = False
    receipt.write_text(canonical_json(resign(data)))
    assert main(["path-validate", "--receipt", str(receipt)]) == 1


def test_frozen_multistage_fixture():
    raw = (Path(__file__).with_name("fixtures") / "path-search-v1.json").read_text(encoding="utf-8")
    assert raw == canonical_json(run()) + "\n"
    assert validate_path_search_receipt(json.loads(raw))["all_passed"] is True
