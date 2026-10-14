# SPDX-License-Identifier: MPL-2.0
"""Source-pinned stored programme, lifecycle, schedule, feature and adapter negatives."""
import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from qec.routing.crossbar.contention import ContentionBatch, compile_contention_program, execute_contention_program
from qec.routing.crossbar.continuity import validate_continuity_receipt
from qec.routing.ess.programme import (
    MAX_CALLS, MAX_PAYLOAD_BYTES, MAX_TICK, FeatureModules, ProgramManifest,
    TranslationTable, validate_program_manifest,
)
from qec.routing.ess.lifecycle import (
    Call, FabricAdapter, LifecycleInput, create_fabric_equivalence_receipt,
    demo_lifecycle_input, execute_call_processing, validate_call_processing_receipt,
    validate_event_schedule_receipt, validate_fabric_equivalence_receipt,
)
from qec.routing.ess.lifecycle_cli import main
from qec.sonify.canonical import canonical_json, canonical_sha256


def rehash(value):
    if type(value) is list:
        return [rehash(item) for item in value]
    if type(value) is dict:
        # A source file's sha256 hashes actual bytes, not its metadata row.
        if set(value) == {"path", "byte_count", "sha256"}:
            return copy.deepcopy(value)
        result = {key: rehash(item) for key, item in value.items() if key != "sha256"}
        if "sha256" in value:
            result["sha256"] = canonical_sha256(result)
        return result
    return value


@pytest.fixture
def source():
    return demo_lifecycle_input()


@pytest.fixture
def receipt(source):
    return execute_call_processing(source)


def test_real_reservations_release_and_exact_state_machine(receipt):
    a, b = receipt["calls"]
    assert (a["outcome"], b["outcome"]) == ("released", "released")
    assert b["attempts"][0]["native_reason"] == "no_admissible_complete_path"
    assert b["attempts"][0]["path_plan"] is None
    assert b["attempts"][1]["logical_tick"] == a["attempts"][0]["release_tick"] == 2
    assert receipt["connected_attempt_count"] == 2
    assert receipt["rejected_attempt_count"] == 1
    states = lambda call, attempt: [e["state"] for e in receipt["events"]
                                    if e["call_id"] == call and e["attempt"] == attempt]
    assert states("call-a", 0) == ["intake", "analyse", "select", "connect", "verify", "release"]
    assert states("call-b", 0) == ["intake", "analyse", "select", "reject"]
    assert states("call-b", 1) == states("call-a", 0)
    evidence = receipt["native_continuity_receipt"]
    assert validate_continuity_receipt(evidence)["all_passed"]
    native = evidence["source_receipt"]
    assert [r["outcome"] for r in native["results"]] == ["reserved", "rejected", "released", "reserved", "released"]
    assert native["results"][1]["blocked_resources"]
    assert native["final_state"]["active_reservations"] == []
    assert receipt["claim_boundary"]["physical_connection_commit_present"] is False
    assert len(receipt["sha256"]) == 64


def test_schedule_priority_release_retry_arrival(source):
    c = replace(source.calls[1], call_id="call-c", arrival_tick=2)
    source = replace(source, calls=(*source.calls, c))
    receipt = execute_call_processing(source)
    at_two = [row for row in receipt["schedule"]["dispatches"] if row["logical_tick"] == 2]
    assert [row["kind"] for row in at_two] == ["release", "retry", "arrival"]
    assert [row["fixed_priority"] for row in at_two] == [0, 1, 2]
    assert validate_event_schedule_receipt(receipt["schedule"], source,
                                          expected_input_sha256=source.as_dict()["sha256"])["all_passed"]
    wrong = copy.deepcopy(receipt["schedule"])
    wrong["dispatches"].reverse()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_event_schedule_receipt(rehash(wrong), source)


def test_source_manifest_hashes_whole_files_and_separate_data(source):
    manifest = source.programme.as_dict()
    root = Path(__file__).parents[2] / "src"
    for row in manifest["source_manifest"]["files"]:
        data = (root / row["path"]).read_bytes()
        assert row["byte_count"] == len(data)
        assert row["sha256"] == hashlib.sha256(data).hexdigest()
    assert validate_program_manifest(manifest, expected_source_sha256=manifest["source_sha256"])["all_passed"]
    changed = replace(source.programme, program_version="2")
    assert changed.as_dict()["sha256"] != manifest["sha256"]
    assert changed.translations.as_dict() == source.programme.translations.as_dict()
    translated = replace(source.programme, translations=TranslationTable((("dial", ("elsewhere",)),)))
    assert translated.as_dict()["translation_table_sha256"] != manifest["translation_table_sha256"]
    assert translated.as_dict()["source_sha256"] == manifest["source_sha256"]
    assert translated.as_dict()["routing_logic_sha256"] == manifest["routing_logic_sha256"]
    other = execute_call_processing(replace(source, programme=changed))
    assert other["input"]["call_store"] == source.as_dict()["call_store"]
    assert other["command_stream"]["sha256"] != execute_call_processing(source)["command_stream"]["sha256"]


@pytest.mark.parametrize("mutation", ["source", "logic", "table", "features", "extra", "version"])
def test_rehashed_programme_forgery(source, mutation):
    value = source.programme.as_dict()
    if mutation == "source":
        value["source_manifest"]["files"][0]["sha256"] = "0" * 64
        value["source_manifest"] = rehash(value["source_manifest"])
        value["source_sha256"] = value["source_manifest"]["sha256"]
    elif mutation == "logic":
        value["routing_logic"]["transitions"].append(["reject", "connect"])
        value["routing_logic"] = rehash(value["routing_logic"])
        value["routing_logic_sha256"] = value["routing_logic"]["sha256"]
    elif mutation == "table":
        value["translations"]["rows"][0]["destinations"] = ["other"]
    elif mutation == "features":
        value["features"]["force_accept_permitted"] = True
    elif mutation == "version":
        value["contract_version"] = "173.0"
    else:
        value["load_code"] = "outside.py"
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ProgramManifest.from_dict(rehash(value))


def test_forwarding_fanout_and_retry_are_explicit(source):
    table = TranslationTable((("group", ("out", "out-2")),))
    features = FeatureModules((("dial", "group"),), (2,), 2)
    adapter = replace(source.adapter, destinations=(("out", "egress", "V001"), ("out-2", "egress", "V000")))
    transformed = LifecycleInput(ProgramManifest(table, features), (source.calls[0],), adapter)
    receipt = execute_call_processing(transformed)
    assert receipt["calls"][0]["released_branch_count"] == 2
    assert receipt["calls"][0]["rejected_attempt_count"] == 1
    analyses = [event for event in receipt["events"] if event["state"] == "analyse"]
    assert all(event["details"]["dialed_destination"] == "dial" for event in analyses)
    assert all(event["details"]["translated_destination"] == "group" for event in analyses)
    for row in receipt["calls"][0]["attempts"]:
        assert row["payload_sha256"] == source.calls[0].as_dict()["payload_sha256"]
        assert row["decoder_output_sha256"] == source.calls[0].decoder_output_sha256
    without_retry = replace(transformed, programme=replace(transformed.programme,
                                                          features=replace(features, retry_delays=())))
    partial = execute_call_processing(without_retry)
    assert partial["calls"][0]["outcome"] == "partially_released"
    assert partial["calls"][0]["released_branch_count"] == 1
    assert validate_call_processing_receipt(partial)["all_passed"]


def test_busy_quarantine_budget_and_missing_translation_stay_rejected(source):
    for state in ("busy", "quarantined", "unavailable"):
        fabric = replace(source.adapter.fabric, interstage_links=tuple(
            replace(wire, state=state) for wire in source.adapter.fabric.interstage_links))
        receipt = execute_call_processing(replace(source, adapter=replace(source.adapter, fabric=fabric)))
        assert receipt["connected_attempt_count"] == 0
        assert receipt["rejected_attempt_count"] == 4
        assert all(row["outcome"] == "rejected" for row in receipt["calls"])
        assert not validate_call_processing_receipt(receipt)["nonempty_connection_evidence"]
    budget = execute_call_processing(replace(source, programme=replace(source.programme, max_search_evaluations=1)))
    assert budget["connected_attempt_count"] == 0
    assert budget["calls"][0]["attempts"][0]["native_reason"] == "search_budget_exhausted"
    assert budget["calls"][1]["attempts"][0]["native_reason"] == "batch_search_budget_exhausted"
    missing = execute_call_processing(replace(source, calls=(replace(source.calls[0], destination="missing"),)))
    assert missing["calls"][0]["attempts"][0]["reason"] == "translation_not_found"
    assert missing["native_evidence_present"] is False
    assert [row["state"] for row in missing["events"]] == ["intake", "analyse", "reject"]


def test_empty_is_not_connection_or_equivalence_evidence(source):
    empty = replace(source, calls=())
    receipt = execute_call_processing(empty)
    assert receipt["events"] == []
    assert receipt["native_continuity_receipt"] is None
    result = validate_call_processing_receipt(receipt)
    assert result["all_passed"] and not result["nonempty_connection_evidence"]
    other = replace(demo_lifecycle_input(alternate_fabric=True), calls=())
    eq = create_fabric_equivalence_receipt(empty, other)
    assert not eq["equivalent"] and not eq["nonempty_case_coverage"]


def test_detachment_frozen_serialization(source):
    calls = list(source.calls)
    endpoints = [list(row) for row in source.adapter.sources]
    copied = replace(source, calls=calls, adapter=replace(source.adapter, sources=endpoints))
    calls.clear()
    endpoints[0][0] = "changed"
    assert copied.as_dict() == source.as_dict()
    raw = copied.as_dict()
    restored = LifecycleInput.from_dict(raw)
    raw["programme"]["features"]["retry_delays"].append(3)
    assert restored.as_dict() == source.as_dict()
    for instance, name, value in ((source, "calls", ()), (source.programme, "program_version", "2"),
                                  (source.adapter, "sources", ()), (source.calls[0], "payload", b"x")):
        with pytest.raises(FrozenInstanceError):
            setattr(instance, name, value)


@pytest.mark.parametrize("field,value", [("arrival_tick", True), ("arrival_tick", -1),
    ("arrival_tick", MAX_TICK), ("hold_ticks", 0), ("hold_ticks", 1.0), ("source", ""),
    ("destination", "x" * 129), ("call_id", "\ud800"), ("payload", bytearray(b"x")),
    ("payload", b"x" * (MAX_PAYLOAD_BYTES + 1)), ("decoder_output_sha256", "short")])
def test_call_bounds(source, field, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source.calls[0], **{field: value})


@pytest.mark.parametrize("features", [dict(retry_delays=(0,)), dict(retry_delays=(True,)),
    dict(retry_delays=(1, 2, 3)), dict(max_fanout=0), dict(max_fanout=5), dict(max_fanout=True),
    dict(forwarding=(("a", "b"), ("b", "a"))), dict(forwarding=(("a", "b"), ("b", "c"))),
    dict(forwarding=(("a", "b"), ("a", "c")))])
def test_feature_bounds(features):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        FeatureModules(**features)


def test_batch_order_expansion_and_endpoint_bounds(source):
    for calls in ((source.calls[1], source.calls[0]), (source.calls[0], source.calls[0]),
                  source.calls * (MAX_CALLS + 1), [None]):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            replace(source, calls=calls)
    calls = tuple(replace(source.calls[0], call_id=f"call-{i}") for i in range(MAX_CALLS))
    # 8 calls * 3 attempts * 2 native commands exceeds 32, rejected before execution.
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source, calls=calls, programme=replace(source.programme, features=FeatureModules(retry_delays=(1, 2))))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source.adapter, sources=(("in", "not-there", "H000"),))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source.programme, translations=TranslationTable((("dial", ("one", "two")),)))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        replace(source.programme, features=FeatureModules(forwarding=(("dial", "absent"),)))


@pytest.mark.parametrize("cls", [TranslationTable, FeatureModules, ProgramManifest, Call, FabricAdapter, LifecycleInput])
@pytest.mark.parametrize("value", [None, [], {}, {"schema": "unknown"}])
def test_malformed_serialization(cls, value):
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        cls.from_dict(value)


@pytest.mark.parametrize("mutation", ["omit", "swap", "accept", "reserve", "release", "schedule", "stream",
                                      "native", "count", "boundary", "extra"])
def test_rehashed_receipt_semantic_tampering(receipt, mutation):
    raw = copy.deepcopy(receipt)
    if mutation == "omit":
        raw["events"].pop()
    elif mutation == "swap":
        raw["calls"].reverse()
    elif mutation == "accept":
        raw["calls"][1]["attempts"][0]["outcome"] = "released"
    elif mutation == "reserve":
        raw["calls"][0]["attempts"][0]["reservation_sha256"] = "0" * 64
    elif mutation == "release":
        raw["calls"][0]["attempts"][0]["release_tick"] = 0
    elif mutation == "schedule":
        raw["schedule"]["dispatches"][0]["fixed_priority"] = 0
    elif mutation == "stream":
        raw["command_stream"]["commands"][0]["operation"] = "force_accept"
    elif mutation == "native":
        raw["native_continuity_receipt"]["source_receipt"]["results"][1]["outcome"] = "reserved"
    elif mutation == "count":
        raw["connected_attempt_count"] = True
    elif mutation == "boundary":
        raw["claim_boundary"]["decoder_output_mutation_permitted"] = True
    else:
        raw["proof_override"] = True
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_call_processing_receipt(rehash(raw))


@pytest.mark.parametrize("change", ["request", "fabric", "program", "fake"])
def test_adapter_cannot_substitute_valid_evidence(source, monkeypatch, change):
    def wrong(self, batch, program):
        if change == "fake":
            return {"outcome": "reserved", "all_passed": True}
        commands = batch.commands
        fabric = batch.fabric
        if change == "request":
            request = commands[0].request
            altered = replace(request, request=replace(request.request, payload=b"substitution"))
            commands = (replace(commands[0], request=altered), *commands[1:])
        elif change == "fabric":
            fabric = replace(fabric, fabric_id="substituted")
        modified = ContentionBatch(fabric, commands)
        altered_program = compile_contention_program(modified, marker_id="substituted" if change == "program" else program["marker_id"],
                                                     max_search_evaluations=program["max_search_evaluations"])
        if change == "program":
            program.clear()
            program.update(altered_program)
        return execute_contention_program(modified, altered_program)
    monkeypatch.setattr(FabricAdapter, "execute", wrong)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_call_processing(source)


def test_pins_bind_exact_input_and_reject_valid_substitution(source, receipt):
    snapshot = source.as_dict()
    pins = {"expected_input_sha256": snapshot["sha256"], "expected_programme_sha256": snapshot["programme"]["sha256"],
            "expected_adapter_sha256": snapshot["adapter"]["sha256"], "expected_call_store_sha256": snapshot["call_store"]["sha256"]}
    assert all(validate_call_processing_receipt(receipt, **pins)["external_bindings_verified"].values())
    for key in pins:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_call_processing_receipt(receipt, **{key: "0" * 64})
    other = execute_call_processing(replace(source, calls=(replace(source.calls[0], payload=b"different"),)))
    assert validate_call_processing_receipt(other)["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_call_processing_receipt(other, **pins)


def test_two_supported_fabrics_and_negative_controls(source):
    other = demo_lifecycle_input(alternate_fabric=True)
    equivalence = create_fabric_equivalence_receipt(source, other)
    assert equivalence["equivalent"] and equivalence["semantic_command_stream_equal"]
    left_path = equivalence["receipts"][0]["calls"][0]["attempts"][0]["path_plan"]
    right_path = equivalence["receipts"][1]["calls"][0]["attempts"][0]["path_plan"]
    assert len(left_path["coordinates"]) == 3 and len(right_path["coordinates"]) == 2
    pins = {"expected_left_input_sha256": source.as_dict()["sha256"], "expected_right_input_sha256": other.as_dict()["sha256"],
            "expected_programme_sha256": source.programme.as_dict()["sha256"]}
    assert all(validate_fabric_equivalence_receipt(equivalence, **pins)["external_bindings_verified"].values())
    blocked = replace(other.adapter.fabric, interstage_links=tuple(replace(w, state="quarantined") for w in other.adapter.fabric.interstage_links))
    negative = create_fabric_equivalence_receipt(source, replace(other, adapter=replace(other.adapter, fabric=blocked)))
    assert not negative["equivalent"] and not negative["semantic_command_stream_equal"]
    assert validate_fabric_equivalence_receipt(negative)["all_passed"]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        create_fabric_equivalence_receipt(source, source)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        create_fabric_equivalence_receipt(source, replace(other, calls=()))
    for field in ("equivalent", "native_failures_preserved", "supports_universal_equivalence"):
        tampered = copy.deepcopy(equivalence)
        tampered[field] = not tampered[field]
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_fabric_equivalence_receipt(rehash(tampered))
    substituted = copy.deepcopy(equivalence)
    substituted["receipts"][1] = equivalence["receipts"][0]
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_fabric_equivalence_receipt(rehash(substituted))


def test_maximum_native_command_bound_and_detached_inputs(source):
    # Eight independent calls with one allowed retry: declared worst case = 32.
    calls = tuple(replace(source.calls[0], call_id=f"call-{i}", arrival_tick=i * 4,
                          payload=b"x" * MAX_PAYLOAD_BYTES) for i in range(MAX_CALLS))
    source = replace(source, calls=calls)
    before = canonical_json(source.as_dict())
    result = execute_call_processing(source)
    assert result["connected_attempt_count"] == MAX_CALLS
    assert len(result["command_stream"]["commands"]) == 16
    assert canonical_json(source.as_dict()) == before


def test_frozen_lifecycle_fixture(source, receipt):
    fixture = json.loads((Path(__file__).parent / "fixtures/lifecycle-v2.json").read_text())
    assert source.as_dict() == fixture["input"]
    assert receipt["sha256"] == fixture["receipt_sha256"]
    assert receipt["command_stream"]["sha256"] == fixture["command_stream_sha256"]
    assert receipt["schedule"]["sha256"] == fixture["schedule_sha256"]


@pytest.mark.parametrize("seed", ["0", "42", "random"])
def test_hash_seed_independent(seed, receipt):
    script = "from qec.routing.ess.lifecycle import *; print(execute_call_processing(demo_lifecycle_input())['sha256'])"
    result = subprocess.run([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed},
                            capture_output=True, text=True, check=True)
    assert result.stdout.strip() == receipt["sha256"]


def test_cli_roundtrip_failclosed_and_old_fixture_unmodified(tmp_path, capsys):
    old = (Path(__file__).parent / "fixtures/skeleton-v1.json").read_bytes()
    assert main(["demo", "--output-dir", str(tmp_path)]) == 0
    input_path = tmp_path / "ess_lifecycle_input.json"
    assert main(["run", "--input", str(input_path), "--output-dir", str(tmp_path)]) == 0
    receipt = tmp_path / "ess_call_processing_receipt.json"
    data = receipt.read_bytes()
    assert main(["validate", "--receipt", str(receipt)]) == 0
    assert main(["validate", "--receipt", str(receipt), "--expected-input-sha256", "0" * 64]) == 1
    assert main(["run", "--input", str(input_path), "--output-dir", str(tmp_path)]) == 0
    assert data == receipt.read_bytes()
    assert main(["equivalence-demo", "--output-dir", str(tmp_path)]) == 0
    assert main(["validate-equivalence", "--receipt", str(tmp_path / "ess_fabric_equivalence_receipt.json")]) == 0
    input_path.write_text('{"schema":1,"schema":2}')
    assert main(["run", "--input", str(input_path)]) == 1
    assert main(["validate", "--receipt", str(tmp_path / "absent")]) == 1
    assert (Path(__file__).parent / "fixtures/skeleton-v1.json").read_bytes() == old
    assert capsys.readouterr().err


def test_installed_source_drift_changes_identity_and_rejects_old_programme(source, monkeypatch):
    old = source.programme.as_dict()
    original = Path.read_bytes
    def changed_bytes(path):
        data = original(path)
        if path.name == "programme.py" and path.parent.name == "ess":
            return data + b"\n# declared source revision\n"
        return data
    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    current = replace(source.programme)
    assert current.as_dict()["source_sha256"] != old["source_sha256"]
    assert current.as_dict()["sha256"] != old["sha256"]
    assert current.translations.as_dict() == source.programme.translations.as_dict()
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        ProgramManifest.from_dict(old)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_call_processing(source)


def test_json_depth_size_types_and_programme_pin_bounds(source, receipt):
    nested = []
    for _ in range(70):
        nested = [nested]
    for invalid in (nested, [None] * 8193, {"x": "x" * 16385}, {"x": 0.0}):
        for parse in (LifecycleInput.from_dict, ProgramManifest.from_dict, validate_call_processing_receipt):
            with pytest.raises(ValueError, match="INVALID_INPUT"):
                parse(invalid)
    for pin in ("not-a-hash", "0" * 64, True):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            validate_program_manifest(source.programme.as_dict(), expected_manifest_sha256=pin)
    for value in (0, True, 65537, 1.0):
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            replace(source.programme, max_search_evaluations=value)


def test_preflight_capacity_covers_two_large_native_receipts():
    from qec.routing.ess.programme import bounded_json
    # A measured maximum-intersection 16-attempt native receipt has ~294k nodes.
    # Comparing two sides must not hit the older 500k-node single-receipt limit.
    row = {"native_fields": ["bounded"] * 1000}
    bounded_json({"left": [row] * 300, "right": [row] * 300})
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        bounded_json([row] * 2000)


def test_adapter_cannot_mutate_its_sealed_boundary(source, monkeypatch):
    original = FabricAdapter.execute
    def mutating(self, batch, program):
        result = original(self, batch, program)
        object.__setattr__(self, "sources", (("altered", "ingress", "H000"),))
        return result
    before = canonical_json(source.as_dict())
    monkeypatch.setattr(FabricAdapter, "execute", mutating)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        execute_call_processing(source)
    assert canonical_json(source.as_dict()) == before


@pytest.mark.parametrize("mutation", ["cyclic", "overdeep", "oversized-list", "oversized-string", "non-json"])
def test_native_adapter_output_preflight_before_replay(source, monkeypatch, mutation):
    original = FabricAdapter.execute
    def malformed(self, batch, program):
        raw = original(self, batch, program)
        if mutation == "cyclic":
            raw["bad"] = raw
        elif mutation == "overdeep":
            nested = []
            for _ in range(70):
                nested = [nested]
            raw["bad"] = nested
        elif mutation == "oversized-list":
            raw["bad"] = [None] * 8193
        elif mutation == "oversized-string":
            raw["bad"] = "x" * 16385
        else:
            raw["bad"] = object()
        return raw
    monkeypatch.setattr(FabricAdapter, "execute", malformed)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        execute_call_processing(source)


@pytest.mark.parametrize("value", [None, True, False, 0, -12345, [], {},
    {"emoji-😀": "😀" * 30, "escapes": "\x00\n\t\\\"" * 20},
    {"a": [True, False, None, -7, {"quoted\"key": "\b\f\r"}], "empty": []},
])
def test_exact_canonical_utf8_byte_bound_includes_keys_and_json_overhead(value, monkeypatch):
    from qec.routing.ess import programme
    size = len(canonical_json(value).encode("utf-8"))
    monkeypatch.setattr(programme, "MAX_JSON_BYTES", size)
    programme.bounded_json(value)
    monkeypatch.setattr(programme, "MAX_JSON_BYTES", size - 1)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        programme.bounded_json(value)


def test_cli_reads_actual_capped_bytes_without_stat_race(monkeypatch):
    import io
    from qec.routing.ess import lifecycle_cli, programme
    read_sizes = []
    class TrackedReader(io.BytesIO):
        def read(self, size=-1):
            read_sizes.append(size)
            return super().read(size)
    # A producer can grow a file after stat. The only authority is bounded actual bytes.
    def no_stat(self, *args, **kwargs):
        raise AssertionError("pre-read stat is not a byte limit")
    monkeypatch.setattr(programme, "MAX_JSON_BYTES", 32)
    monkeypatch.setattr(Path, "stat", no_stat)
    monkeypatch.setattr(Path, "open", lambda self, mode: TrackedReader(b'{"x":"' + b'x' * 10000 + b'"}'))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        lifecycle_cli._read(Path("racing-input.json"))
    assert read_sizes == [33]


def test_cli_exact_byte_boundary_and_no_unbudgeted_newline(tmp_path, monkeypatch):
    from qec.routing.ess import lifecycle_cli, programme
    value = {"😀\"key": "\x00😀" * 10}
    encoded = canonical_json(value).encode("utf-8")
    path = tmp_path / "bounded.json"
    monkeypatch.setattr(programme, "MAX_JSON_BYTES", len(encoded))
    lifecycle_cli._write(path, value)
    assert path.read_bytes() == encoded
    assert lifecycle_cli._read(path) == value
    path.write_bytes(encoded + b" ")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        lifecycle_cli._read(path)
    monkeypatch.setattr(programme, "MAX_JSON_BYTES", len(encoded) - 1)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        lifecycle_cli._write(tmp_path / "not-emitted.json", value)
    assert not (tmp_path / "not-emitted.json").exists()


@pytest.mark.parametrize("target", ["call-processing-receipt", "fabric-equivalence-receipt"])
def test_emitted_receipts_fail_closed_on_output_byte_limit(source, monkeypatch, target):
    from qec.routing.ess import lifecycle, programme
    original = lifecycle.bounded_json
    seen = []
    def reject_oversized_output(value):
        if type(value) is dict and value.get("schema") == f"qec.ess-{target}.v2":
            seen.append(value["schema"])
            with monkeypatch.context() as patch:
                patch.setattr(programme, "MAX_JSON_BYTES", len(canonical_json(value).encode("utf-8")) - 1)
                original(value)
        else:
            original(value)
    monkeypatch.setattr(lifecycle, "bounded_json", reject_oversized_output)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        if target == "call-processing-receipt":
            lifecycle.execute_call_processing(source)
        else:
            lifecycle.create_fabric_equivalence_receipt(source, demo_lifecycle_input(alternate_fabric=True))
    assert seen == [f"qec.ess-{target}.v2"]
