# SPDX-License-Identifier: MPL-2.0
"""Packet contracts, end-to-end replay, faults, and hostile evidence mutations."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.packet import (
    FaultPlan, PacketEndpoint, PacketEnvelope, PacketFault, PacketInput, PacketRequest, PacketTopology,
    QueuePolicy, Route, SoftswitchObservation, canonical_packet_schema, create_packet_demo_export,
    create_packet_report, create_softswitch_adapter_receipt, demo_packet_input, execute_packet_fabric,
    lookup_route, packetize, validate_packet_receipt, validate_packet_report,
    validate_softswitch_adapter_receipt,
)
from qec.routing.packet.cli import main
from qec.routing.packet.core import (
    MAX_FRAGMENT_BYTES, MAX_FRAGMENTS, MAX_HOPS, MAX_LOGICAL_TICK, MAX_NODES, MAX_PACKETS,
    MAX_PAYLOAD_BYTES, MAX_QUEUE_CAPACITY, MAX_REQUESTS, MAX_ROUTES,
)
from qec.sonify.canonical import canonical_json, canonical_sha256


def rehash(value):
    if type(value) is list:
        return [rehash(v) for v in value]
    if type(value) is dict:
        result = {k: rehash(v) for k, v in value.items() if k != "sha256"}
        if "sha256" in value:
            result["sha256"] = canonical_sha256(result)
        return result
    return value


@pytest.fixture
def source():
    return demo_packet_input()


@pytest.fixture
def receipt(source):
    return execute_packet_fabric(source)


def test_native_success_and_every_chain_anchor(source, receipt):
    assert receipt["accepted_count"] == 1
    assert receipt["rejected_count"] == 0
    assert receipt["failover_receipt"]["failover_count"] == 2
    result = receipt["results"][0]
    assert result["outcome"] == "accepted"
    assert result["accepted_payload_sha256"] == hashlib.sha256(source.requests[0].payload).hexdigest()
    assert bytes.fromhex(result["reassembled_payload_hex"]) == source.requests[0].payload
    assert [path["nodes"] for path in result["packet_paths"]] == [["a", "c", "d"], ["a", "c", "d"], ["a", "b", "d"]]
    previous = receipt["input"]["sha256"]
    packet_previous = {p["sha256"]: p["sha256"] for p in receipt["envelopes"]}
    for e in receipt["hop_receipt_chain"]["events"]:
        unsigned = {k: v for k, v in e.items() if k != "sha256"}
        assert canonical_sha256(unsigned) == e["sha256"]
        assert e["previous_event_sha256"] == previous
        previous = e["sha256"]
        packet = e["packet_sha256"]
        if packet is not None:
            assert e["previous_packet_event_sha256"] == packet_previous[packet]
            packet_previous[packet] = e["sha256"]
    assert previous == receipt["hop_receipt_chain"]["chain_head_sha256"]
    assert validate_packet_receipt(receipt)["all_passed"]


@pytest.mark.parametrize("scenario,accepted,reason", [
    ("success", 1, "payload_verified"), ("failover", 1, "payload_verified"),
    ("overflow", 0, "missing_fragments"), ("drop", 0, "missing_fragments"),
    ("corrupt", 0, "payload_mismatch"), ("recovery", 2, "payload_verified"),
])
def test_declared_fault_battery(scenario, accepted, reason):
    source = demo_packet_input(scenario)
    original = canonical_json(source.as_dict())
    receipt = execute_packet_fabric(source)
    assert receipt["accepted_count"] == accepted
    assert receipt["results"][0]["reason"] == reason
    assert canonical_json(execute_packet_fabric(source)) == canonical_json(receipt)
    assert canonical_json(source.as_dict()) == original
    assert validate_packet_receipt(receipt)["all_passed"]
    assert receipt["claim_boundary"]["hardware_or_quantum_evidence"] is False
    if scenario == "corrupt":
        assert receipt["results"][0]["accepted_payload_sha256"] is None
        assert receipt["envelopes"][0]["payload_hex"] == source.requests[0].payload[:8].hex()
    if scenario == "recovery":
        assert all(path["nodes"] == ["a", "b", "d"] for path in receipt["results"][1]["packet_paths"])


def test_longest_exact_prefix_then_metric_then_id():
    routes = [Route("a", "", ("d",), 0, "default"), Route("a", "1", ("c",), 0, "short"),
              Route("a", "10", ("b",), 9, "higher-cost"), Route("a", "10", ("c",), 1, "z"),
              Route("a", "10", ("d",), 1, "a"), Route("a", "11", ("b",), 0, "wrong-prefix")]
    topology = PacketTopology(["d", "b", "c", "a"], (), routes)
    assert lookup_route(topology, "a", "10101").route_id == "a"
    assert lookup_route(topology, "a", "11").route_id == "wrong-prefix"
    assert lookup_route(topology, "a", "1").route_id == "short"
    assert lookup_route(topology, "a", "0").route_id == "default"
    assert lookup_route(topology, "b", "0") is None
    assert PacketTopology(tuple(reversed(topology.nodes)), (), tuple(reversed(routes))).as_dict() == topology.as_dict()


def test_no_less_specific_fallback_or_undeclared_alternate():
    topology = PacketTopology(("a", "b", "c"), (PacketEndpoint("101", "c"),),
                              (Route("a", "10", ("b",), route_id="specific"),
                               Route("a", "", ("c",), route_id="default")))
    source = PacketInput((PacketRequest("one", "a", "101", b"x", "1" * 64),), topology,
                         fault_plan=FaultPlan((PacketFault(0, "a", "b", source_state="busy"),)))
    receipt = execute_packet_fabric(source)
    assert receipt["accepted_count"] == 0
    assert receipt["results"][0]["drops"][0]["reason"] == "all_next_hops_unavailable"
    attempts = [e for e in receipt["hop_receipt_chain"]["events"] if e["kind"] == "route_attempt"]
    assert len(attempts) == 1 and attempts[0]["details"]["next_hop"] == "b"
    assert receipt["input"]["fault_plan"]["faults"][0]["source_state"] == "busy"


def test_one_overflow_rule_priority_and_service(source):
    r = source.requests[0]
    requests = (replace(r, request_id="a-low", payload=b"a", priority=7),
                replace(r, request_id="b-high", payload=b"b", priority=0),
                replace(r, request_id="c-high", payload=b"c", priority=0))
    batch = replace(source, requests=requests, queue_policy=QueuePolicy(2), fault_plan=FaultPlan())
    receipt = execute_packet_fabric(batch)
    assert [r["outcome"] for r in receipt["results"]] == ["rejected", "accepted", "accepted"]
    dequeues = [e for e in receipt["hop_receipt_chain"]["events"] if e["kind"] == "dequeue" and e["node"] == "a"]
    by_hash = {p["sha256"]: p["request_id"] for p in receipt["envelopes"]}
    assert [by_hash[e["packet_sha256"]] for e in dequeues] == ["b-high", "c-high"]
    drops = [e for e in receipt["hop_receipt_chain"]["events"] if e["kind"] == "queue_overflow"]
    assert len(drops) == 1
    assert drops[0]["details"]["decision"] == "drop_arriving_packet_no_eviction"
    assert all(len(e["details"].get("queue_after", [])) <= 2 for e in receipt["hop_receipt_chain"]["events"])


def test_overflow_never_evicts_waiting_lower_priority(source):
    r = source.requests[0]
    requests = (replace(r, request_id="a", payload=b"a", priority=7),
                replace(r, request_id="b", payload=b"b", priority=7),
                replace(r, request_id="c", payload=b"c", priority=7),
                replace(r, request_id="d", payload=b"d", priority=0, logical_tick=1),
                replace(r, request_id="e", payload=b"e", priority=0, logical_tick=1))
    receipt = execute_packet_fabric(replace(source, requests=requests, queue_policy=QueuePolicy(3), fault_plan=FaultPlan()))
    assert receipt["results"][-1]["drops"][0]["reason"] == "queue_overflow"
    assert receipt["results"][1]["outcome"] == "accepted"
    assert receipt["results"][2]["outcome"] == "accepted"


def test_unknown_destination_no_route_hop_limit_and_all_down(source):
    r = source.requests[0]
    unknown = execute_packet_fabric(replace(source, requests=(replace(r, destination="000"),)))
    assert unknown["results"][0]["drops"][0]["reason"] == "unknown_destination"
    no_route = replace(source.topology, routes=())
    result = execute_packet_fabric(replace(source, topology=no_route, fault_plan=FaultPlan()))
    assert result["results"][0]["drops"][0]["reason"] == "no_route"
    loop = replace(source.topology, routes=(Route("a", "", ("b",), route_id="ab"), Route("b", "", ("a",), route_id="ba")))
    result = execute_packet_fabric(replace(source, topology=loop, queue_policy=QueuePolicy(16, 2), fault_plan=FaultPlan()))
    assert all(d["reason"] == "hop_limit" for d in result["results"][0]["drops"])
    assert result["results"][0]["packet_paths"][0]["nodes"] == ["a", "b", "a"]
    all_down = FaultPlan((PacketFault(0, "a", "b", duration=10), PacketFault(0, "a", "c", duration=10, source_state="quarantined")))
    result = execute_packet_fabric(replace(source, fault_plan=all_down))
    assert all(d["reason"] == "all_next_hops_unavailable" for d in result["results"][0]["drops"])
    assert validate_packet_receipt(result)["all_passed"]


def test_packet_and_provenance_identities_are_separate(source):
    request = source.requests[0]
    original = packetize(request, 8)
    new_origin = packetize(replace(request, provenance_sha256="f" * 64), 8)
    assert original[0].as_dict()["fragment_sha256"] == new_origin[0].as_dict()["fragment_sha256"]
    assert original[0].as_dict()["sha256"] != new_origin[0].as_dict()["sha256"]
    assert original[0].as_dict()["sequence_identity"]["sha256"] != new_origin[0].as_dict()["sequence_identity"]["sha256"]
    for index, packet in enumerate(original):
        assert packet.sequence == index
        assert PacketEnvelope.from_dict(packet.as_dict()) == packet
    success = execute_packet_fabric(replace(source, fault_plan=FaultPlan()))
    failure = execute_packet_fabric(source)
    assert success["envelopes"] == failure["envelopes"]
    assert success["hop_receipt_chain"]["sha256"] != failure["hop_receipt_chain"]["sha256"]


def test_empty_payload_and_empty_batch(source):
    r = replace(source.requests[0], payload=b"")
    assert len(packetize(r)) == 1
    receipt = execute_packet_fabric(replace(source, requests=(r,)))
    assert receipt["accepted_count"] == 1
    assert receipt["results"][0]["reassembled_payload_hex"] == ""
    corrupted = execute_packet_fabric(replace(source, requests=(r,), fault_plan=FaultPlan((PacketFault(0, "a", "b", "corrupt"),))))
    assert corrupted["results"][0]["reason"] == "payload_mismatch"
    empty = execute_packet_fabric(replace(source, requests=()))
    assert empty["packet_count"] == empty["accepted_count"] == empty["rejected_count"] == 0
    assert validate_packet_receipt(empty)["nonempty_evidence"] is False


def test_destination_source_is_zero_hop_delivery(source):
    r = replace(source.requests[0], source="d", payload=b"x")
    result = execute_packet_fabric(replace(source, requests=(r,)))
    assert result["accepted_count"] == 1
    assert result["results"][0]["packet_paths"][0]["nodes"] == ["d"]
    assert not any(e["kind"] == "forward" for e in result["hop_receipt_chain"]["events"])


def test_frozen_objects_detach_containers(source):
    nodes, routes = list(source.topology.nodes), list(source.topology.routes)
    topology = PacketTopology(nodes, source.topology.destinations, routes)
    nodes.clear()
    routes.clear()
    assert topology.as_dict() == source.topology.as_dict()
    with pytest.raises(FrozenInstanceError):
        topology.nodes = ()
    requests = list(source.requests)
    input_copy = replace(source, requests=requests)
    requests.clear()
    assert input_copy.as_dict() == source.as_dict()
    assert PacketInput.from_dict(source.as_dict()) == source


@pytest.mark.parametrize("cls", [PacketRequest, PacketEnvelope, PacketEndpoint, Route, PacketTopology, QueuePolicy, PacketFault, FaultPlan, PacketInput, SoftswitchObservation])
@pytest.mark.parametrize("value", [None, [], {}, {"schema": "wrong"}])
def test_malformed_serializations(cls, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        cls.from_dict(value)


@pytest.mark.parametrize("field,value", [
    ("request_id", ""), ("request_id", "x" * 65), ("request_id", "a/b"), ("request_id", "\ud800"),
    ("source", None), ("source", "é"), ("destination", ""), ("destination", "10x"), ("destination", "1" * 33),
    ("payload", bytearray(b"x")), ("payload", "x"), ("payload", b"x" * (MAX_PAYLOAD_BYTES + 1)),
    ("provenance_sha256", "F" * 64), ("provenance_sha256", "0" * 63), ("priority", True),
    ("priority", 1.0), ("priority", -1), ("priority", 8), ("logical_tick", True),
    ("logical_tick", 1.0), ("logical_tick", -1), ("logical_tick", MAX_LOGICAL_TICK + 1),
])
def test_request_bounds(source, field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source.requests[0], **{field: value})


@pytest.mark.parametrize("field,value", [("capacity", True), ("capacity", 1.0), ("capacity", 0), ("capacity", MAX_QUEUE_CAPACITY + 1),
                                         ("max_hops", False), ("max_hops", 1.0), ("max_hops", 0), ("max_hops", MAX_HOPS + 1)])
def test_queue_bounds(field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        QueuePolicy(**{field: value})


@pytest.mark.parametrize("field,value", [("fragment_bytes", True), ("fragment_bytes", 1.0), ("fragment_bytes", 0),
                                         ("fragment_bytes", MAX_FRAGMENT_BYTES + 1), ("requests", "x"),
                                         ("requests", (None,)), ("queue_policy", {}), ("topology", {}), ("fault_plan", {})])
def test_input_bounds(source, field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source, **{field: value})


@pytest.mark.parametrize("field,value", [("sequence", True), ("sequence", 1.0), ("sequence", -1), ("sequence", MAX_FRAGMENTS),
                                         ("offset", True), ("offset", MAX_PAYLOAD_BYTES), ("fragment_count", 0),
                                         ("fragment_count", MAX_FRAGMENTS + 1), ("payload", b"x" * (MAX_FRAGMENT_BYTES + 1))])
def test_envelope_bounds(source, field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(packetize(source.requests[0])[0], **{field: value})


@pytest.mark.parametrize("field,value", [("metric", True), ("metric", 1.0), ("metric", -1), ("metric", 65536),
                                         ("next_hops", ()), ("next_hops", ("b", "b")), ("next_hops", ("a",)),
                                         ("prefix", "2"), ("next_hops", "b")])
def test_route_bounds(field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(Route("a", "", ("b",)), **{field: value})


@pytest.mark.parametrize("field,value", [("logical_tick", True), ("logical_tick", 1.0), ("logical_tick", -1),
                                         ("duration", True), ("duration", 0), ("duration", MAX_LOGICAL_TICK + 2),
                                         ("kind", "random"), ("kind", None), ("source_state", "idle"), ("next_hop", "a")])
def test_fault_bounds(field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(PacketFault(0, "a", "b"), **{field: value})


def test_topology_crossrefs_and_global_bounds(source):
    for topology in [lambda: PacketTopology((), (), ()), lambda: PacketTopology(("a", "a"), (), ()),
                     lambda: PacketTopology(("a",), (PacketEndpoint("1", "b"),), ()),
                     lambda: PacketTopology(("a",), (), (Route("a", "", ("b",)),)),
                     lambda: replace(source.topology, routes=(source.topology.routes[0],) * 2),
                     lambda: replace(source.topology, destinations=source.topology.destinations * 2),
                     lambda: PacketTopology(tuple(f"n{i}" for i in range(MAX_NODES + 1)), (), ()),
                     lambda: replace(source.topology, routes=source.topology.routes * (MAX_ROUTES + 1))]:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            topology()
    for changed in [lambda: replace(source, requests=source.requests * 2),
                    lambda: replace(source, requests=(replace(source.requests[0], source="unknown"),)),
                    lambda: replace(source, fault_plan=FaultPlan((PacketFault(0, "b", "c"),))),
                    lambda: replace(source, requests=tuple(replace(source.requests[0], request_id=f"r{i}") for i in range(MAX_REQUESTS + 1))),
                    lambda: FaultPlan((PacketFault(0, "a", "b", duration=3), PacketFault(2, "a", "b"))),
                    lambda: PacketFault(MAX_LOGICAL_TICK, "a", "b", duration=2),
                    lambda: packetize(replace(source.requests[0], payload=b"x" * MAX_PAYLOAD_BYTES), 1)]:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            changed()


def test_maximum_fragment_and_packet_boundaries(source):
    r = replace(source.requests[0], payload=b"x" * MAX_PAYLOAD_BYTES)
    assert len(packetize(r, 64)) == MAX_FRAGMENTS
    requests = tuple(replace(r, request_id=f"r{i}") for i in range(4))
    batch = replace(source, requests=requests, fragment_bytes=64)
    assert sum(len(packetize(r, batch.fragment_bytes)) for r in batch.requests) == MAX_PACKETS
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(batch, requests=requests + (replace(r, request_id="extra"),))
    # Max payload survives reassembly when the declared queue can hold its batch.
    receipt = execute_packet_fabric(replace(source, requests=(r,), queue_policy=QueuePolicy(64), fragment_bytes=64, fault_plan=FaultPlan()))
    assert receipt["accepted_count"] == 1
    assert bytes.fromhex(receipt["results"][0]["reassembled_payload_hex"]) == r.payload


def test_all_trust_pins_and_valid_substitution(source, receipt):
    pins = {"expected_input_sha256": source.as_dict()["sha256"], "expected_topology_sha256": source.topology.as_dict()["sha256"],
            "expected_queue_policy_sha256": source.queue_policy.as_dict()["sha256"],
            "expected_fault_plan_sha256": source.fault_plan.as_dict()["sha256"], "expected_receipt_sha256": receipt["sha256"]}
    result = validate_packet_receipt(receipt, **pins)
    assert all(result["external_bindings_verified"].values())
    for key in pins:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_packet_receipt(receipt, **{**pins, key: "0" * 64})
    different = execute_packet_fabric(replace(source, requests=(replace(source.requests[0], provenance_sha256="c" * 64),)))
    assert validate_packet_receipt(different)["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_packet_receipt(different, expected_input_sha256=source.as_dict()["sha256"])


@pytest.mark.parametrize("path,value", [
    (("accepted_count",), 99), (("accepted_count",), True), (("accepted_count",), 1.0),
    (("envelopes", 0, "payload_hex"), "00"), (("envelopes", 0, "sequence"), 2),
    (("envelopes", 0, "provenance_sha256"), "f" * 64),
    (("results", 0, "reassembled_payload_hex"), "00"), (("results", 0, "destination"), "1111"),
    (("results", 0, "outcome"), "rejected"), (("results", 0, "packet_paths", 0, "nodes"), ["a", "d"]),
    (("hop_receipt_chain", "events", 0, "details", "offset"), 1),
    (("hop_receipt_chain", "events", 1, "details", "queue_after"), []),
    (("hop_receipt_chain", "events", 0, "previous_packet_event_sha256"), "b" * 64),
    (("failover_receipt", "failover_count"), 0),
    (("claim_boundary", "external_observations_are_authoritative"), True),
    (("claim_boundary", "logical_time_only"), 1), (("final_queues", 0, "packets"), ["invented"]),
])
def test_rehashed_mutations_rejected(receipt, path, value):
    mutant = deepcopy(receipt)
    cursor = mutant
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_packet_receipt(mutant if type(value) is float else rehash(mutant))


@pytest.mark.parametrize("target", ["envelopes", "results", "events", "attempts"])
def test_omission_reordering_and_duplicates_fail(receipt, target):
    for change in (lambda rows: rows[:-1], lambda rows: rows + rows[:1], lambda rows: list(reversed(rows))):
        mutant = deepcopy(receipt)
        container, field = (mutant, target)
        if target == "events":
            container, field = mutant["hop_receipt_chain"], "events"
        if target == "attempts":
            container, field = mutant["failover_receipt"], "attempt_event_sha256"
        old = container[field]
        container[field] = change(old)
        if container[field] == old:
            continue
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_packet_receipt(rehash(mutant))


def test_noncanonical_field_addition_input_and_schema_mutation(source, receipt):
    for field in ("schema", "contract_version", "canonical_packet_schema_sha256"):
        mutant = source.as_dict()
        mutant[field] = "unexpected"
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            PacketInput.from_dict(rehash(mutant))
    mutant = deepcopy(receipt)
    mutant["ambient_network_state"] = "live"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_packet_receipt(rehash(mutant))
    mutant = source.as_dict()
    mutant["requests"][0]["payload_hex"] = "AA"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        PacketInput.from_dict(rehash(mutant))


def test_report_is_replayed_and_detached(receipt):
    report = create_packet_report(receipt)
    assert validate_packet_report(report)["all_passed"]
    receipt["accepted_count"] = 123
    assert validate_packet_report(report)["all_passed"]
    report["accepted_count"] = 123
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_packet_report(rehash(report))


def test_external_observation_cannot_override_packet_proof(receipt):
    observation = SoftswitchObservation(0, "sip", "offline-capture", "c" * 64, "INVITE observed")
    adapter = create_softswitch_adapter_receipt((observation,), packet_receipt=receipt)
    assert validate_softswitch_adapter_receipt(adapter, packet_receipt=receipt)["all_passed"]
    assert adapter["external"] is True and adapter["authoritative"] is False
    assert adapter["network_activation_present"] is False
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_softswitch_adapter_receipt(adapter)
    mutant = deepcopy(adapter)
    mutant["authoritative"] = True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_softswitch_adapter_receipt(rehash(mutant), packet_receipt=receipt)
    assert "softswitch" not in canonical_json(receipt["input"])
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        create_softswitch_adapter_receipt((replace(observation, logical_index=1),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(observation, logical_index=True)


def test_external_unbound_capture_and_pins():
    receipt = create_softswitch_adapter_receipt(())
    assert validate_softswitch_adapter_receipt(receipt, expected_receipt_sha256=receipt["sha256"])["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_softswitch_adapter_receipt(receipt, expected_receipt_sha256="0" * 64)


def test_demo_namespaces_and_offline_lab(receipt):
    demo = create_packet_demo_export(receipt)
    assert demo["schema"] == "qec.packet-demo.v1"
    assert demo["demo_only"] and not demo["cryptographic_proof"] and not demo["hardware_evidence"]
    assert "sha256" not in canonical_json(demo)
    root = Path(__file__).resolve().parents[2]
    html = (root / "docs/labs/packet/index.html").read_text()
    assert "qec.packet-demo.v1" in html
    assert "AudioContext" in html
    for forbidden in ("fetch(", "XMLHttpRequest", "WebSocket", 'src="http', 'href="http'):
        assert forbidden not in html


def test_cli_roundtrip_and_duplicate_key_rejection(tmp_path, capsys):
    assert main(["demo", "--output-dir", str(tmp_path)]) == 0
    source = tmp_path / "packet_fabric_input.json"
    assert main(["run", "--input", str(source), "--output-dir", str(tmp_path)]) == 0
    receipt = tmp_path / "packet_fabric_receipt.json"
    snapshot = json.loads(source.read_text())
    assert main(["validate", "--receipt", str(receipt), "--expected-input-sha256", snapshot["sha256"]]) == 0
    assert len(list(tmp_path.glob("*.json"))) == 11
    duplicate = tmp_path / "bad.json"
    duplicate.write_text('{"schema":"one","schema":"two"}')
    assert main(["run", "--input", str(duplicate), "--output-dir", str(tmp_path)]) == 1
    assert "INVALID_INPUT" in capsys.readouterr().err


def test_frozen_fixture_and_hashseed_replay(receipt):
    fixture = Path(__file__).parent / "fixtures/packet_replay_hashes.json"
    expected = json.loads(fixture.read_text())
    assert receipt["sha256"] == expected["failover_receipt_sha256"]
    assert receipt["input"]["sha256"] == expected["failover_input_sha256"]
    assert canonical_packet_schema()["sha256"] == expected["canonical_schema_sha256"]
    command = "from qec.routing.packet import *; from qec.sonify.canonical import canonical_json; print(canonical_json(execute_packet_fabric(demo_packet_input())))"
    outputs = []
    for seed in ("0", "42", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}
        outputs.append(subprocess.check_output([sys.executable, "-c", command], env=env, text=True))
    assert outputs[0] == outputs[1] == outputs[2] == canonical_json(receipt) + "\n"


def test_registered_views_require_complete_replay(receipt):
    from qec.routing.packet import validate_packet_artifact
    artifacts = [receipt["canonical_packet_schema"], receipt["input"]["topology"], receipt["input"]["queue_policy"],
                 receipt["hop_receipt_chain"], receipt["failover_receipt"], receipt["results"][0]]
    for artifact in artifacts:
        assert validate_packet_artifact(artifact, fabric_receipt=receipt,
                                        expected_input_sha256=receipt["input"]["sha256"])["all_passed"]
        mutant = deepcopy(artifact)
        mutant["extra_authority"] = True
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_packet_artifact(rehash(mutant), fabric_receipt=receipt)
    broken_source = deepcopy(receipt)
    broken_source["hop_receipt_chain"]["events"].pop()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_packet_artifact(artifacts[-1], fabric_receipt=rehash(broken_source))


def test_demo_export_is_detached_from_native_input(receipt):
    demo = create_packet_demo_export(receipt)
    demo["nodes"].clear()
    demo["events"].clear()
    assert validate_packet_receipt(receipt)["all_passed"]


@pytest.mark.parametrize("destination", ["00", "01", "10", "11"])
@pytest.mark.parametrize("state0", ["idle", "busy", "quarantined"])
@pytest.mark.parametrize("state1", ["idle", "busy", "quarantined"])
def test_two_lane_migration_contract(destination, state0, state1):
    destinations = tuple(PacketEndpoint(f"{n:02b}", f"d{n}") for n in range(4))
    routes = [Route("ingress", "", ("lane0", "lane1"), route_id="ingress")]
    for lane in ("lane0", "lane1"):
        routes.extend(Route(lane, e.address, (e.node,), route_id=f"{lane}-{e.node}") for e in destinations)
    topology = PacketTopology(("ingress", "lane0", "lane1", "d0", "d1", "d2", "d3"), destinations, routes)
    faults = tuple(PacketFault(0, "ingress", f"lane{index}", duration=1024, source_state=state)
                   for index, state in enumerate((state0, state1)) if state != "idle")
    source = PacketInput((PacketRequest("migration", "ingress", destination, b"correction", "d" * 64),),
                         topology, fault_plan=FaultPlan(faults))
    receipt = execute_packet_fabric(source)
    result = receipt["results"][0]
    assert validate_packet_receipt(receipt, expected_input_sha256=source.as_dict()["sha256"])["all_passed"]
    if state0 == state1 == "idle" or state0 == "idle" or state1 == "idle":
        lane = "lane0" if state0 == "idle" else "lane1"
        assert result["outcome"] == "accepted"
        assert result["destination"] == destination
        assert result["packet_paths"][0]["nodes"] == ["ingress", lane, f"d{int(destination, 2)}"]
    else:
        assert result["outcome"] == "rejected"
        assert result["drops"][0]["reason"] == "all_next_hops_unavailable"


def test_fault_precedence_and_nonoverlapping_recovery(source):
    faults = FaultPlan((PacketFault(0, "a", "b", "link_down"), PacketFault(0, "a", "b", "corrupt"),
                       PacketFault(0, "a", "b", "drop"), PacketFault(1, "a", "b", "link_down")))
    result = execute_packet_fabric(replace(source, fault_plan=faults))
    assert result["accepted_count"] == 1
    assert not any(e["kind"] in ("fault_drop", "fault_corrupt") for e in result["hop_receipt_chain"]["events"])
    for tick in (True, 1.0, -1):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            faults.active(tick, "a", "b")


def test_serialized_order_must_be_canonical(source):
    table = source.topology.as_dict()
    table["routes"].reverse()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        PacketTopology.from_dict(rehash(table))
