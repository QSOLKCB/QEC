# SPDX-License-Identifier: MPL-2.0
"""Independent forward walks, exact source coverage and lifecycle boundaries."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.crossbar import (
    CrossbarFabric, CrossbarRequest, InterstageLink, MultiStageRequest,
    ContentionBatch, ContentionCommand, demo_fabric, demo_matrix, demo_contention_batch,
    compile_path_program, execute_path_program, compile_contention_program,
    execute_contention_program, verify_path_continuity, create_continuity_receipt,
    validate_continuity_receipt,
)
from qec.routing.crossbar.cli import main
from qec.sonify.canonical import canonical_json, canonical_sha256


def request():
    return MultiStageRequest("ingress", "egress", CrossbarRequest(
        "continuity-demo", "H000", "V001", b"\x00opaque\xff", "a" * 64))


def search(fabric=None, intent=None, budget=131072):
    fabric = demo_fabric().as_dict() if fabric is None else fabric.as_dict()
    intent = request() if intent is None else intent
    return execute_path_program(fabric, intent,
        compile_path_program(fabric, intent, max_search_evaluations=budget))


def contend(batch=None, budget=65536):
    batch = demo_contention_batch() if batch is None else batch
    return execute_contention_program(batch,
        compile_contention_program(batch, max_search_evaluations=budget))


def resign(value):
    value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
    return value


def test_forward_route_and_exact_payload_identity():
    source = search()
    before = deepcopy(source)
    receipt = create_continuity_receipt(source, expected_source_sha256=source["sha256"])
    row = receipt["routes"][0]
    witness = row["witness"]
    assert receipt["selected_route_count"] == 1
    assert witness["path_plan"] == source["path_plan"]
    assert [s["kind"] for s in witness["steps"]] == [
        "coordinate", "interstage", "coordinate", "interstage", "coordinate"]
    assert [s["wire"]["link_id"] for s in witness["steps"] if s["kind"] == "interstage"] == ["entry-b", "exit-b"]
    assert witness["payload_sha256"] == source["payload_sha256"]
    assert witness["decoder_output_sha256"] == "a" * 64
    assert row["reservation_active_at_batch_end"] is None
    assert validate_continuity_receipt(receipt)["continuity_verified"] is True
    assert source == before
    source["path_plan"]["coordinates"].clear()
    assert receipt["source_receipt"] == before
    witness["steps"].clear()
    assert before["path_plan"]["coordinates"]


@pytest.mark.parametrize("mutation", [
    lambda p: p["coordinates"].pop(),
    lambda p: p["coordinates"].append(deepcopy(p["coordinates"][-1])),
    lambda p: p["coordinates"].reverse(),
    lambda p: p["coordinates"].__setitem__(1, deepcopy(p["coordinates"][0])),
    lambda p: p["coordinates"][0].__setitem__("stage", False),
    lambda p: p["coordinates"][1].__setitem__("stage", 1.0),
    lambda p: p["coordinates"][1].__setitem__("matrix_id", "middle-a"),
    lambda p: p["coordinates"][1]["coordinate"].__setitem__("horizontal_link_id", "H001"),
    lambda p: p["coordinates"][0]["coordinate"].__setitem__("intersection_id", "0" * 64),
    lambda p: p["coordinates"][0]["coordinate"].__setitem__("vertical_ordinal", True),
    lambda p: p["coordinates"][0]["coordinate"].__setitem__("vertical_link_id", []),
    lambda p: p["interstage_link_ids"].pop(),
    lambda p: p["interstage_link_ids"].append("exit-b"),
    lambda p: p["interstage_link_ids"].reverse(),
    lambda p: p["interstage_link_ids"].__setitem__(0, "entry-a"),
    lambda p: p["interstage_link_ids"].__setitem__(1, "entry-b"),
    lambda p: p["interstage_link_ids"].__setitem__(1, []),
    lambda p: p.__setitem__("extra_branch", []),
    lambda p: p.__setitem__("coordinates", tuple(p["coordinates"])),
])
def test_walk_rejects_invalid_plans_without_selector_replay(mutation):
    source = search()
    plan = deepcopy(source["path_plan"])
    mutation(plan)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_path_continuity(source["fabric_manifest"], request(), plan)


@pytest.mark.parametrize("change", [
    {"source_matrix_id": "missing"}, {"destination_matrix_id": "missing"},
    {"request": replace(request().request, horizontal_link_id="H001")},
    {"request": replace(request().request, vertical_link_id="V000")},
])
def test_endpoint_intent_cannot_be_substituted(change):
    source = search()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_path_continuity(source["fabric_manifest"], replace(request(), **change), source["path_plan"])


@pytest.mark.parametrize("kind", ["horizontal", "vertical", "interstage"])
@pytest.mark.parametrize("state", ["busy", "quarantined", "unavailable"])
def test_selected_resources_must_be_idle(kind, state):
    fabric = demo_fabric()
    source = search(fabric)
    if kind == "interstage":
        fabric = replace(fabric, interstage_links=tuple(
            replace(w, state=state) if w.link_id == "entry-b" else w for w in fabric.interstage_links))
    else:
        stages = list(fabric.stages)
        matrix = stages[0][0]
        attr = kind + "_links"
        link_id = "H000" if kind == "horizontal" else "V001"
        matrix = replace(matrix, **{attr: tuple(replace(l, state=state) if l.link_id == link_id else l
                                               for l in getattr(matrix, attr))})
        stages[0] = (matrix,)
        fabric = replace(fabric, stages=stages)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        verify_path_continuity(fabric.as_dict(), request(), source["path_plan"])


def test_exhaustive_route_membership_against_explicit_two_branch_oracle():
    fabric = demo_fabric(dead_end_first=False)
    # Enumerate the selected matrix choice and both proposed wires, independently
    # of the selector. Exactly the two matching branch triples form routes.
    for middle, entry, exit_wire in product(("a", "b"), repeat=3):
        matrices = [fabric.stages[0][0], fabric.stages[1][0 if middle == "a" else 1], fabric.stages[2][0]]
        plan = {"coordinates": [
            {"stage": i, "matrix_id": m.matrix_id, "coordinate": m.coordinate(h, v).as_dict()}
            for i, (m, h, v) in enumerate(zip(matrices,
                ("H000", "H000", "H000" if exit_wire == "a" else "H001"),
                ("V000" if entry == "a" else "V001", "V000", "V001")))],
            "interstage_link_ids": ["entry-" + entry, "exit-" + exit_wire]}
        if middle == entry == exit_wire:
            witness = verify_path_continuity(fabric.as_dict(), request(), plan)
            assert witness["coordinate_count"] == 3
        else:
            with pytest.raises(ValueError, match="INVALID_INPUT"):
                verify_path_continuity(fabric.as_dict(), request(), plan)
    # The walk accepts either continuous branch; source replay still enforces
    # the selector's first-path rule before issuing a continuity receipt.
    alternate = search()
    forged = search(fabric)
    forged["path_plan"] = alternate["path_plan"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        create_continuity_receipt(resign(forged))


@pytest.mark.parametrize("stages", [2, 16])
def test_stage_bounds_and_repeated_local_link_ids(stages):
    matrices = [demo_matrix(f"m{i:02d}", horizontal_count=1, vertical_count=1) for i in range(stages)]
    fabric = CrossbarFabric("chain", tuple((m,) for m in matrices), tuple(
        InterstageLink(f"w{i:02d}", matrices[i].matrix_id, "V000", matrices[i+1].matrix_id, "H000")
        for i in range(stages-1)))
    intent = MultiStageRequest(matrices[0].matrix_id, matrices[-1].matrix_id,
                              replace(request().request, vertical_link_id="V000"))
    proof = create_continuity_receipt(search(fabric, intent))["routes"][0]["witness"]
    assert len(proof["steps"]) == 2 * stages - 1
    assert len({row["coordinate"]["intersection_id"] for row in proof["path_plan"]["coordinates"]}) == stages


def test_contention_complete_coverage_and_historical_lifecycle_scope():
    source = contend()
    receipt = create_continuity_receipt(source)
    assert [r["command_id"] for r in receipt["routes"]] == ["a", "b", "retry", "retry-quarantined"]
    assert [r["outcome"] for r in receipt["routes"]] == ["continuous", "not_selected", "continuous", "not_selected"]
    assert [r["reservation_active_at_batch_end"] for r in receipt["routes"]] == [False, None, False, None]
    assert receipt["selected_route_count"] == 2
    assert validate_continuity_receipt(receipt)["complete_selection_coverage_verified"]
    for row in receipt["routes"]:
        if row["witness"]:
            original = next(r for r in source["results"] if r["command_id"] == row["command_id"])
            assert row["reservation_sha256"] == original["reservation"]["sha256"]
            assert row["witness"]["fabric_sha256"] == original["path_search_receipt"]["fabric_sha256"]
            assert row["state_before_sha256"] == original["state_before_sha256"]


def test_active_reservations_and_alternate_route_snapshots():
    first = request()
    second = replace(first, request=replace(first.request, horizontal_link_id="H001", vertical_link_id="V000"))
    batch = ContentionBatch(demo_fabric(dead_end_first=False), (
        ContentionCommand(0, "a", "reserve", request=first),
        ContentionCommand(1, "b", "reserve", request=second)))
    receipt = create_continuity_receipt(contend(batch))
    assert [r["reservation_active_at_batch_end"] for r in receipt["routes"]] == [True, True]
    assert [r["witness"]["path_plan"]["interstage_link_ids"] for r in receipt["routes"]] == [
        ["entry-a", "exit-a"], ["entry-b", "exit-b"]]
    assert receipt["routes"][0]["witness"]["fabric_sha256"] != receipt["routes"][1]["witness"]["fabric_sha256"]


@pytest.mark.parametrize("source", [
    lambda: search(budget=1),
    lambda: search(intent=replace(request(), source_matrix_id="missing")),
    lambda: search(replace(demo_fabric(), interstage_links=())),
    lambda: contend(ContentionBatch(demo_fabric(), (ContentionCommand(
        0, "release", "release", reservation_id="absent", expected_request_sha256="a"*64),))),
])
def test_valid_no_selection_is_not_a_continuity_proof(source):
    receipt = create_continuity_receipt(source())
    validation = validate_continuity_receipt(receipt)
    assert receipt["outcome"] == "no_selected_route"
    assert validation["all_passed"] is True
    assert validation["continuity_verified"] is False
    assert all(r["witness"] is None for r in receipt["routes"])


def test_batch_budget_exhaustion_preserves_attempt_without_search():
    batch = ContentionBatch(demo_fabric(), tuple(ContentionCommand(i, str(i), "reserve", request=request()) for i in range(32)))
    source = contend(batch, budget=1)
    receipt = create_continuity_receipt(source)
    assert len(receipt["routes"]) == 32
    assert receipt["routes"][0]["source_reason"] == "search_budget_exhausted"
    assert all(r["path_search_receipt_sha256"] is None and r["source_reason"] == "batch_search_budget_exhausted"
               for r in receipt["routes"][1:])


@pytest.mark.parametrize("mutation", [
    lambda r: r["routes"].pop(),
    lambda r: r["routes"].reverse(),
    lambda r: r["routes"][0].__setitem__("reservation_active_at_batch_end", True),
    lambda r: r["routes"][0].__setitem__("command_id", "retry"),
    lambda r: r["routes"][0].__setitem__("state_before_sha256", "0"*64),
    lambda r: r["routes"][0].__setitem__("reservation_sha256", "0"*64),
    lambda r: r["routes"][0]["witness"]["steps"].pop(),
    lambda r: r["routes"][0]["witness"].__setitem__("payload_sha256", "0"*64),
    lambda r: r["routes"][0]["witness"].__setitem__("decoder_output_sha256", "0"*64),
    lambda r: r.__setitem__("selected_route_count", True),
    lambda r: r.__setitem__("source_receipt_sha256", "0"*64),
    lambda r: r["events"][0].__setitem__("previous_sha256", "0"*64),
    lambda r: r["claim_boundary"].__setitem__("connection_commit_present", True),
    lambda r: r["source_receipt"]["results"][0]["reservation"]["resource_ids"].pop(),
])
def test_rehashed_tampering_rejected(mutation):
    receipt = create_continuity_receipt(contend())
    mutation(receipt)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_continuity_receipt(resign(receipt))


@pytest.mark.parametrize("source_fn", [search, contend])
def test_trusted_bindings(source_fn):
    source = source_fn()
    receipt = create_continuity_receipt(source)
    path = "input_register" in source
    pins = {"expected_source_sha256": source["sha256"],
            "expected_fabric_sha256": source["fabric_sha256"] if path else source["input"]["fabric_manifest"]["sha256"],
            "expected_input_sha256": source["input_register" if path else "input"]["sha256"],
            "expected_program_sha256": source["path_program" if path else "program"]["sha256"]}
    assert all(validate_continuity_receipt(receipt, **pins)["external_bindings_verified"].values())
    assert not any(validate_continuity_receipt(receipt)["external_bindings_verified"].values())
    assert create_continuity_receipt(source, **pins) == receipt
    for key in pins:
        for bad in ("0"*64, True, "short"):
            wrong = {**pins, key: bad}
            with pytest.raises(ValueError, match="INVALID_INPUT"):
                validate_continuity_receipt(receipt, **wrong)
            with pytest.raises(ValueError, match="INVALID_INPUT"):
                create_continuity_receipt(source, **wrong)


def test_event_chain_fixture_and_hashseed_determinism():
    receipt = create_continuity_receipt(search())
    previous = receipt["event_root_sha256"]
    for sequence, event in enumerate(receipt["events"]):
        assert event["sequence"] == sequence and event["previous_sha256"] == previous
        assert resign(deepcopy(event)) == event
        previous = event["sha256"]
    fixture = Path(__file__).with_name("fixtures") / "continuity-v1.json"
    assert fixture.read_text() == canonical_json(receipt) + "\n"
    code = ('from qec.routing.crossbar import *; from qec.sonify.canonical import canonical_json; '
            'b=demo_contention_batch(); print(canonical_json(create_continuity_receipt('
            'execute_contention_program(b, compile_contention_program(b)))))')
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
               for seed in ("0", "41", "random")]
    assert len(set(outputs)) == 1


@pytest.mark.parametrize("source_fn", [search, contend, lambda: search(budget=1)])
def test_cli_roundtrip_and_no_selection_exit(tmp_path, capsys, source_fn):
    source = source_fn()
    path = tmp_path / "source.json"
    path.write_text(canonical_json(source))
    receipt = create_continuity_receipt(source)
    expected = 0 if receipt["selected_route_count"] else 2
    assert main(["continuity", "--receipt", str(path), "--output-dir", str(tmp_path),
                 "--expected-source-sha256", source["sha256"]]) == expected
    capsys.readouterr()
    target = tmp_path / "crossbar_continuity_receipt.json"
    assert json.loads(target.read_text()) == receipt
    assert main(["continuity-validate", "--receipt", str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["all_passed"]
    assert main(["continuity-validate", "--receipt", str(target), "--expected-source-sha256", "0"*64]) == 1
    path.write_text('{"schema": "x", "schema": "y"}')
    assert main(["continuity", "--receipt", str(path)]) == 1


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "unknown"}, {"schema": "qec.crossbar-continuity-validation.v1"}])
def test_malformed_sources(bad):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        create_continuity_receipt(bad)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_continuity_receipt(bad)
