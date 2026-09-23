# SPDX-License-Identifier: MPL-2.0
"""Shared-case admission oracle, explicit capability limits and source binding."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing import panel, strowger
from qec.routing.crossbar import validate_continuity_receipt
from qec.routing.crossbar.cli import main
from qec.routing.equivalence import (
    ARCHITECTURES, FAULTS, EquivalenceCase, EquivalenceCorpus,
    demo_equivalence_corpus, equivalence_adapter_manifest,
    run_equivalence_battery, validate_equivalence_matrix,
)
from qec.sonify.canonical import canonical_json, canonical_sha256

FIXTURES = Path(__file__).with_name("fixtures")


def case(name="a", destination=0, states=("idle", "idle"), **kw):
    return EquivalenceCase(name, destination, states, **kw)


def small_corpus():
    return EquivalenceCorpus((case(payload=b"\x00\xff"), case("b", 3, ("busy", "idle")),
                              case("c", 2, ("quarantined", "busy"))))


@pytest.fixture(scope="module")
def matrix():
    return run_equivalence_battery(small_corpus())


@pytest.fixture(scope="module")
def battery():
    return run_equivalence_battery(demo_equivalence_corpus())


def resign_all(value):
    """Rehash every nested artifact, proving rejection is more than checksum checking."""
    if isinstance(value, list):
        for child in value:
            resign_all(child)
    elif isinstance(value, dict):
        for child in value.values():
            resign_all(child)
        for field in ("sha256", "event_sha256"):
            if field in value:
                value[field] = canonical_sha256({k: v for k, v in value.items() if k != field})
    return value


def test_complete_availability_grid_and_negative_controls(battery):
    corpus = demo_equivalence_corpus()
    assert battery["default_corpus_used"] is True
    assert battery["case_count"] == 41
    assert battery["equivalent_case_count"] == 36
    assert battery["expected_difference_count"] == 5
    assert battery["all_cases_equivalent"] is False
    assert battery["battery_passed"] is True
    assert len({(c.destination, c.lane_states) for c in corpus.cases if c.fault == "none"}) == 36
    for c, row in zip(corpus.cases, battery["results"]):
        assert row["case_sha256"] == c.as_dict()["sha256"]
        assert [p["architecture"] for p in row["projections"]] == list(ARCHITECTURES)
        assert all(row["request_checks"].values()) and all(row["payload_checks"].values())
        if c.fault == "none":
            # Independent explicit truth table for all 3x3 capacity combinations.
            expected_lane = {( "idle", "idle"): 0, ("idle", "busy"): 0,
                ("idle", "quarantined"): 0, ("busy", "idle"): 1, ("quarantined", "idle"): 1,
                ("busy", "busy"): None, ("busy", "quarantined"): None,
                ("quarantined", "busy"): None, ("quarantined", "quarantined"): None}[c.lane_states]
            for projection in row["projections"]:
                assert projection["selected_lane"] == expected_lane
                assert projection["routing_outcome"] == ("capacity_blocked" if expected_lane is None else "route_available")
                assert projection["reached_destination"] == (None if expected_lane is None else f"correction/destination-{c.destination}")
            assert row["equivalent"] and all(p["equivalent"] for p in row["pairwise"])
        else:
            assert not row["equivalent"]
            assert sum(p["equivalent"] for p in row["pairwise"]) == 1
            affected = c.fault.split("_", 1)[0]
            projection = next(p for p in row["projections"] if p["architecture"] == affected)
            assert projection["routing_outcome"].startswith(affected + ".")
            assert projection["selected_lane"] is None
            assert row["passed"]
    validation = validate_equivalence_matrix(battery)
    assert validation["all_passed"] and validation["replay_verified"]
    assert not validation["all_cases_equivalent"]


def test_all_native_sources_remain_independently_replayable(battery):
    for row in battery["results"]:
        evidence = row["evidence"]
        assert strowger.validate_receipt(evidence["strowger"])["replayed"]
        assert panel.validate_route_receipt(evidence["panel"])["replayed"]
        assert validate_continuity_receipt(evidence["crossbar"])["all_passed"]
        assert evidence["strowger"]["qec_version"] == "170.3.0"
        assert evidence["panel"]["contract_version"] == "171.2"
        assert evidence["crossbar"]["contract_version"] == "172.4"
        assert evidence["crossbar"]["source_receipt"]["contract_version"] == "172.2"


def test_commit_and_payload_capability_differences_are_explicit(matrix):
    row = matrix["results"][0]
    s, p, c = row["projections"]
    assert [s["native_commit_present"], p["native_commit_present"], c["native_commit_present"]] == [True, True, False]
    assert s["native_payload_sha256"] is None
    assert p["native_payload_sha256"] == c["native_payload_sha256"]
    assert s["native_decoder_output_sha256"] is p["native_decoder_output_sha256"] is None
    assert c["native_decoder_output_sha256"] == "a" * 64
    assert len({canonical_sha256(v["events"]) for v in row["evidence"].values() if "events" in v}) == 3
    assert matrix["claim_boundary"]["three_way_native_payload_equivalence"] is False
    assert matrix["claim_boundary"]["commit_semantics_equivalent"] is False


def test_payload_and_decoder_changes_bind_envelope_without_inventing_native_support():
    a = run_equivalence_battery(EquivalenceCorpus((case(payload=b"one"),)))
    b = run_equivalence_battery(EquivalenceCorpus((case(payload=b"two", decoder_output_sha256="b"*64),)))
    assert a["sha256"] != b["sha256"]
    ar, br = a["results"][0], b["results"][0]
    assert ar["evidence"]["strowger"] == br["evidence"]["strowger"]
    assert ar["evidence"]["panel"]["sha256"] != br["evidence"]["panel"]["sha256"]
    assert ar["evidence"]["crossbar"]["sha256"] != br["evidence"]["crossbar"]["sha256"]
    assert all(r["equivalent"] for r in (ar, br))


@pytest.mark.parametrize("states,expected_panel", [
    (("busy", "idle"), {"busy_banks": ["bank-0"], "unavailable_paths": []}),
    (("quarantined", "idle"), {"busy_banks": [], "unavailable_paths": [f"lane-0-destination-{i}" for i in range(4)]}),
])
def test_state_adapters_preserve_declared_blocking_causes(states, expected_panel):
    row = run_equivalence_battery(EquivalenceCorpus((case(states=states),)))["results"][0]
    assert row["evidence"]["strowger"]["initial_state"]["trunks"]["lane"] == [states[0], "free"]
    for key, value in expected_panel.items():
        assert row["evidence"]["panel"]["fault_plan"][key] == value
    wires = row["evidence"]["crossbar"]["source_receipt"]["fabric_manifest"]["interstage_links"]
    assert [w["state"] for w in wires] == list(states)


@pytest.mark.parametrize("mutation", [
    lambda m: m["results"].pop(),
    lambda m: m["results"].reverse(),
    lambda m: m["results"].append(deepcopy(m["results"][0])),
    lambda m: m["results"][0].__setitem__("equivalent", False),
    lambda m: m["results"][0].__setitem__("expected_equivalent", False),
    lambda m: m["results"][0].__setitem__("case_sha256", "0"*64),
    lambda m: m["results"][0]["projections"][0].__setitem__("selected_lane", 1),
    lambda m: m["results"][0]["projections"][0].__setitem__("requested_destination", "forged"),
    lambda m: m["results"][0]["projections"][0].__setitem__("reached_destination", "forged"),
    lambda m: m["results"][0]["projections"][0].__setitem__("native_payload_sha256", "a"*64),
    lambda m: m["results"][0]["projections"][2].__setitem__("native_commit_present", True),
    lambda m: m["results"][0]["pairwise"][0]["checks"].__setitem__("selected_lane", 1),
    lambda m: m["adapter_manifest"]["state_mapping"]["panel"].__setitem__("quarantined", "available"),
    lambda m: m["adapter_manifest"]["destinations"].reverse(),
    lambda m: m["adapter_manifest"]["comparison_fields"].pop(),
    lambda m: m["claim_boundary"].__setitem__("commit_semantics_equivalent", True),
    lambda m: m.__setitem__("case_count", True),
    lambda m: m.__setitem__("battery_passed", 1),
    lambda m: m["results"][0]["evidence"]["strowger"]["route"].__setitem__("connector", [1, 1]),
    lambda m: m["results"][0]["evidence"]["panel"]["payload_identity"].__setitem__("after_sha256", "0"*64),
    lambda m: m["results"][0]["evidence"]["crossbar"]["routes"][0]["witness"]["steps"].pop(),
])
def test_fully_rehashed_tampering_rejected(matrix, mutation):
    altered = deepcopy(matrix)
    mutation(altered)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(resign_all(altered))


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_valid_but_unrelated_native_receipt_cannot_be_substituted(matrix, architecture):
    altered = deepcopy(matrix)
    altered["results"][0]["evidence"][architecture] = deepcopy(matrix["results"][1]["evidence"][architecture])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(resign_all(altered))


def test_valid_alternate_initial_state_cannot_replace_declared_case(matrix):
    alternate = run_equivalence_battery(EquivalenceCorpus((case(states=("busy", "idle")),)))
    modified = deepcopy(matrix)
    modified["results"][0] = alternate["results"][0]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(resign_all(modified))


def test_trusted_corpus_and_adapter_bindings(matrix):
    pins = {"expected_corpus_sha256": matrix["corpus"]["sha256"],
            "expected_adapter_sha256": matrix["adapter_manifest"]["sha256"]}
    assert all(validate_equivalence_matrix(matrix, **pins)["external_bindings_verified"].values())
    for key in pins:
        for bad in ("0"*64, True, "short"):
            with pytest.raises(ValueError, match="INVALID_INPUT"):
                validate_equivalence_matrix(matrix, **{**pins, key: bad})


@pytest.mark.parametrize("kw", [
    {"case_id": ""}, {"case_id": "a"*129}, {"case_id": "\ud800"},
    {"destination": True}, {"destination": 1.0}, {"destination": -1}, {"destination": 4},
    {"lane_states": []}, {"lane_states": ["idle"]}, {"lane_states": ["idle"]*3},
    {"lane_states": {"idle", "busy"}}, {"lane_states": ("unavailable", "idle")},
    {"lane_states": (True, "idle")}, {"payload": bytearray(b"x")}, {"payload": "x"},
    {"payload": b"x"*4097}, {"decoder_output_sha256": "A"*64}, {"decoder_output_sha256": None},
    {"fault": "unknown"}, {"fault": []},
    {"fault": "panel_motor_stall", "lane_states": ("busy", "idle")},
])
def test_invalid_case_types_and_bounds(kw):
    fields = {"case_id": "a", "destination": 0, "lane_states": ("idle", "idle")}
    with pytest.raises(ValueError):
        EquivalenceCase(**{**fields, **kw})


def test_immutable_copy_canonical_order_and_corpus_bounds():
    states = ["idle", "busy"]
    c = case(states=states, payload=b"x"*4096)
    states.clear()
    assert c.lane_states == ("idle", "busy")
    cases = [c]
    corpus = EquivalenceCorpus(cases)
    before = corpus.as_dict()
    cases.clear()
    assert corpus.as_dict() == before
    with pytest.raises(FrozenInstanceError):
        c.destination = 1
    for items in ((), (case("b"), case("a")), (case(), case()), tuple(case(f"c{i:02d}") for i in range(65))):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            EquivalenceCorpus(items)
    assert len(EquivalenceCorpus(tuple(case(f"c{i:02d}") for i in range(64))).cases) == 64
    assert len(EquivalenceCorpus(tuple(case(f"c{i:02d}", payload=b"x"*4096) for i in range(16))).cases) == 16
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EquivalenceCorpus(tuple(case(f"c{i:02d}", payload=b"x"*4096) for i in range(17)))


@pytest.mark.parametrize("mutation", [
    lambda c: c["cases"][0].__setitem__("payload_hex", "FF"),
    lambda c: c["cases"][0].__setitem__("payload_hex", "00 " * 3000),
    lambda c: c["cases"][0].__setitem__("payload_sha256", "0"*64),
    lambda c: c["cases"][0].__setitem__("extra", 1),
    lambda c: c.__setitem__("cases", []),
    lambda c: c.__setitem__("cases", c["cases"]*65),
    lambda c: c.__setitem__("extra", 1),
])
def test_malformed_serialized_corpus(mutation):
    c = EquivalenceCorpus((case(),)).as_dict()
    mutation(c)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        EquivalenceCorpus.from_dict(resign_all(c))


def test_regression_produces_failed_battery_not_false_equivalence(monkeypatch, tmp_path, capsys):
    original = panel.PanelExchange.route

    def unexpected_stall(self, request, *, faults=None):
        faults = panel.PanelFaultPlan() if faults is None else faults
        return original(self, request, faults=replace(faults, stalled_motor_groups=("motor-0", "motor-1")))

    monkeypatch.setattr(panel.PanelExchange, "route", unexpected_stall)
    corpus = EquivalenceCorpus((case(),))
    matrix = run_equivalence_battery(corpus)
    assert not matrix["battery_passed"]
    assert not matrix["results"][0]["equivalent"]
    assert matrix["results"][0]["reference_checks"]["panel"] is False
    assert validate_equivalence_matrix(matrix)["all_passed"] is False
    path = tmp_path / "corpus.json"
    path.write_text(canonical_json(corpus.as_dict()))
    assert main(["equivalence", "--corpus", str(path), "--output-dir", str(tmp_path)]) == 2
    capsys.readouterr()
    assert main(["equivalence-validate", "--matrix", str(tmp_path / "switch_equivalence_matrix.json")]) == 2


def test_frozen_corpus_matrix_identities_and_hashseed_determinism(battery):
    corpus_text = (FIXTURES / "equivalence-corpus-v1.json").read_text()
    assert corpus_text == canonical_json(demo_equivalence_corpus().as_dict()) + "\n"
    golden = json.loads((FIXTURES / "equivalence-v1.json").read_text())
    assert golden == {
        "matrix_sha256": battery["sha256"], "corpus_sha256": battery["corpus"]["sha256"],
        "adapter_sha256": battery["adapter_manifest"]["sha256"],
        "result_hashes": [r["sha256"] for r in battery["results"]],
    }
    code = ('from qec.routing.equivalence import *; '
            'print(run_equivalence_battery(demo_equivalence_corpus())["sha256"])')
    outputs = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": seed})
               for seed in ("0", "42", "random")]
    assert len(set(outputs)) == 1
    assert outputs[0].decode().strip() == battery["sha256"]


def test_cli_default_battery_and_failure_modes(tmp_path, capsys):
    assert main(["equivalence-demo", "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    corpus = tmp_path / "switch_equivalence_corpus.json"
    matrix = tmp_path / "switch_equivalence_matrix.json"
    assert main(["equivalence", "--corpus", str(corpus), "--output-dir", str(tmp_path)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["case_count"] == 41 and validation["all_passed"]
    assert main(["equivalence-validate", "--matrix", str(matrix),
        "--expected-corpus-sha256", validation["corpus_sha256"],
        "--expected-adapter-sha256", validation["adapter_manifest_sha256"]]) == 0
    assert main(["equivalence-validate", "--matrix", str(matrix), "--expected-corpus-sha256", "0"*64]) == 1
    corpus.write_text('{"cases": [], "cases": []}')
    assert main(["equivalence", "--corpus", str(corpus)]) == 1
    assert main(["equivalence-validate", "--matrix", str(tmp_path / "absent")]) == 1


@pytest.mark.parametrize("bad", [None, [], {}, {"schema": "unknown"}])
def test_malformed_matrices(bad):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(bad)


@pytest.mark.parametrize("value", [3.0, float("inf"), float("nan")])
def test_noncanonical_numbers_rejected_before_hashing(matrix, value):
    altered = deepcopy(matrix)
    altered["case_count"] = value
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(altered)


def test_valid_subset_is_not_claimed_as_default_battery(matrix, battery):
    assert validate_equivalence_matrix(matrix)["all_passed"]
    assert validate_equivalence_matrix(matrix)["default_corpus_used"] is False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_equivalence_matrix(matrix, expected_corpus_sha256=battery["corpus"]["sha256"])
