# SPDX-License-Identifier: MPL-2.0
"""Bounded paired comparison, adverse controls, and fully rehashed attacks."""
from copy import deepcopy
import importlib.util
import inspect
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys

import pytest

from qec.experiments.acoustic.channel import PCM, read_wav
from qec.experiments.acoustic.evidence import (
    CASE_NAMES, CHANNEL_METHODS, METHOD_NAMES, build_battery_source, build_demonstration,
    build_trial_source, run_comparative_battery, run_comparative_trial,
    validate_comparative_battery, validate_comparative_trial, validate_demonstration,
)
from qec.sonify.canonical import canonical_bytes, canonical_json, canonical_sha256


def rehash(value):
    value = deepcopy(value)
    value["sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "sha256"})
    return value


def pins(source):
    return {"expected_source_sha256": source["sha256"],
            "expected_reference_sha256": source["measured_reference"]["sha256"]}


def rows(trial):
    return {m["method"]: m for m in trial["methods"]}


@pytest.fixture(scope="module")
def battery():
    return run_comparative_battery(build_battery_source())


@pytest.fixture(scope="module")
def export_module():
    script = Path(__file__).resolve().parents[2] / "scripts/generate_acoustic_trial_evidence.py"
    spec = importlib.util.spec_from_file_location("acoustic_export", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_battery_covers_all_faults_and_both_alphabets(battery):
    assert battery["bounded_trial_count"] == 38
    assert {t["source"]["case_name"] for t in battery["trials"]} == set(CASE_NAMES)
    assert {t["source"]["alphabet_size"] for t in battery["trials"]} == {3, 4}
    families = {f["kind"] for t in battery["trials"] for f in t["source"]["noise_plan"]["faults"]}
    assert families == {"broadband", "colored", "correlated", "speech_like", "impulses", "clipping",
                        "timing_slip", "missing_samples", "phase_drift", "nonlinear", "echo", "aliasing"}
    assert all(tuple(m["method"] for m in t["methods"]) == METHOD_NAMES for t in battery["trials"])
    assert validate_comparative_battery(battery, expected_source_sha256=battery["source"]["sha256"])["all_passed"]
    assert len(canonical_bytes(battery)) < 4_000_000
    assert "native_receipt" not in canonical_json(battery)


@pytest.mark.parametrize("alphabet", [3, 4])
def test_benefit_no_benefit_adverse_and_no_tone_controls(alphabet):
    def trial(case):
        return rows(run_comparative_trial(build_trial_source(case, alphabet_size=alphabet)))
    clean = trial("clean")
    assert all(m["metrics"]["exact_model_success_frames"] == 1 for m in clean.values())
    assert clean["combined"]["waveform_squared_error"] == clean["unfiltered"]["waveform_squared_error"] == 0
    benefit = trial("broadband")
    assert benefit["unfiltered"]["metrics"]["raw_receiver_accepted_frames"] == 0
    assert benefit["combined"]["metrics"]["independently_verified_frames"] == 1
    assert benefit["cancellation_only"]["waveform_squared_error"] == 0
    fec = trial("first_repeat_substitution")
    assert fec["unfiltered"]["metrics"]["raw_receiver_false_accept_frames"] == 1
    assert fec["fec_only"]["metrics"]["exact_model_success_frames"] == 1
    adverse = trial("wrong_phase_reference")
    assert adverse["combined"]["waveform_squared_error"] > adverse["unfiltered"]["waveform_squared_error"]
    for method in CHANNEL_METHODS:
        no_tone = trial("no_tone")[method]["metrics"]
        assert no_tone["symbol_erasures"] == 4
        assert no_tone["ser_decoded_symbols"] == {"numerator": 0, "denominator": 0, "defined": False}
        assert no_tone["ber_decoded_labels"] == {"numerator": 0, "denominator": 0, "defined": False}


@pytest.mark.parametrize("alphabet", [3, 4])
def test_valid_coherent_substitution_false_accepts_are_counted_before_veto(alphabet):
    trial = run_comparative_trial(build_trial_source("coherent_substitution", alphabet_size=alphabet))
    for method in CHANNEL_METHODS:
        row = rows(trial)[method]
        assert row["metrics"]["raw_receiver_accepted_frames"] == 1
        assert row["metrics"]["raw_receiver_false_accept_frames"] == 1
        assert row["metrics"]["symbol_errors"] == 4
        assert row["metrics"]["symbol_erasures"] == 0
        assert row["metrics"]["independently_verified_frames"] == 0
        assert row["decoder_decision"]["attempted"] is False
        assert row["logical_outcome"]["status"] == "no_correction"


def test_channel_resources_and_extra_privileged_references_are_explicit():
    trial = run_comparative_trial(build_trial_source("broadband"))
    methods = rows(trial)
    for key in ("transmitted_pcm_sha256", "transmitted_sample_count", "nominal_carrier_bandwidth_hz",
                "transmitted_energy_sum_squares", "mean_transmitted_sample_power", "whole_frame_latency_budget_seconds"):
        assert all(methods[n]["resources"][key] == methods["unfiltered"]["resources"][key] for n in CHANNEL_METHODS)
    assert methods["unfiltered"]["resources"]["unused_repetitions_remain_in_budget"]
    assert methods["combined"]["resources"]["extra_noise_reference_pcm_bytes"] > 0
    assert methods["unfiltered"]["resources"]["extra_noise_reference_pcm_bytes"] == 0
    assert methods["exact_clean_reference"]["resources"]["extra_clean_reference_pcm_bytes"] > 0
    for n in ("direct_syndrome_reference", "exact_clean_reference"):
        assert methods[n]["resources"]["equal_channel_resource_comparison"] is False
        assert methods[n]["resources"]["equal_total_resources_claimed"] is False
    assert methods["robust_median_period"]["policy"]["estimator"] == "median_period"
    assert "two_bit_natural_binary" in methods["unfiltered"]["metrics"]["ber_scope"]


def test_direct_reference_has_no_apparent_receiver_rate(battery):
    for row in [rows(t)["direct_syndrome_reference"]["metrics"] for t in battery["trials"]] + [
            next(s for s in battery["summary"] if s["method"] == "direct_syndrome_reference")]:
        assert row["raw_receiver_metrics_applicable"] is False
        assert row["recovery_verification_metrics_applicable"] is False
        assert row["independently_verified_frames"] == 0
        assert row["raw_receiver_frame_denominator"] == 0
        assert row["raw_false_accept_rate_all_frames"] == {"numerator": 0, "denominator": 0, "defined": False}
        assert row["raw_false_accept_rate_accepted_frames"]["defined"] is False


def test_no_runtime_or_memory_in_canonical_receipts(battery):
    text = canonical_json(battery)
    assert '"runtime_ns"' not in text and '"peak_python_tracemalloc_bytes"' not in text
    assert all(t["source"]["claim_boundary"]["quantum_state_evolution"] is False for t in battery["trials"])


@pytest.mark.parametrize("field", ["metric", "pcm", "source", "method", "budget", "candidate", "decoder", "logical", "claim", "extra"])
def test_rehashed_trial_tampering_is_rejected(field):
    source = build_trial_source("broadband")
    trial = run_comparative_trial(source)
    forged = deepcopy(trial)
    if field == "metric":
        forged["methods"][0]["metrics"]["symbol_errors"] += 1
    elif field == "pcm":
        forged["observed"] = forged["original"]
    elif field == "source":
        forged["source"]["seed"] += 1
        forged["source"] = rehash(forged["source"])
    elif field == "method":
        forged["methods"].reverse()
    elif field == "budget":
        forged["methods"][0]["resources"]["extra_noise_reference_pcm_bytes"] = 9
    elif field == "candidate":
        forged["methods"][0]["predicted_symbols"] = [0] * 4
    elif field == "decoder":
        forged["methods"][0]["decoder_decision"]["attempted"] = True
    elif field == "logical":
        forged["methods"][0]["logical_outcome"]["success"] = True
    elif field == "claim":
        forged["claim_boundary"]["physical_quantum_protection"] = True
    else:
        forged["extra"] = "unrecognized"
    forged["methods"] = [rehash(m) for m in forged["methods"]]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_trial(rehash(forged), **pins(source))


def test_fully_consistent_replacement_source_needs_new_external_pin():
    source = build_trial_source("broadband")
    replacement = run_comparative_trial(build_trial_source("broadband", seed=99))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_trial(replacement, **pins(source))
    receipt = run_comparative_trial(source)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_trial(receipt, expected_source_sha256=source["sha256"], expected_reference_sha256="a" * 64)


@pytest.mark.parametrize("bad", [None, [], 1, True, "bad", {}, {"source": None}, {"source": {"seed": True}}])
def test_malformed_receipts_fail_closed(bad):
    source = build_trial_source("clean")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_trial(bad, **pins(source))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_battery(bad, expected_source_sha256="a" * 64)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_demonstration(bad, **pins(source))


@pytest.mark.parametrize("kwargs", [dict(cases=()), dict(cases=("clean",) * 2), dict(cases=("bad",)),
                                     dict(alphabets=()), dict(alphabets=(True,)), dict(alphabets=(3, 3)),
                                     dict(seed=True), dict(seed=-1), dict(seed=2**32)])
def test_bounded_battery_configuration(kwargs):
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        build_battery_source(**kwargs)


def test_required_external_pins_and_cross_process_determinism():
    for validate in (validate_comparative_trial, validate_comparative_battery, validate_demonstration):
        for name, parameter in inspect.signature(validate).parameters.items():
            if name.startswith("expected_"):
                assert parameter.default is inspect.Parameter.empty
    script = "from qec.experiments.acoustic.evidence import *; print(run_comparative_trial(build_trial_source('broadband'))['sha256'])"
    results = [subprocess.check_output([sys.executable, "-c", script], text=True,
               env={**os.environ, "PYTHONHASHSEED": seed}).strip() for seed in ("0", "953")]
    assert results[0] == results[1]


def test_demonstration_contains_actual_six_roles_and_replays():
    source = build_trial_source("broadband")
    demo = build_demonstration(source)
    assert set(demo["roles"]) == {"original", "observed", "filtered", "candidate", "rejected", "verified"}
    assert demo["roles"]["candidate"]["accepted"]
    assert not demo["roles"]["rejected"]["accepted"]
    assert demo["roles"]["verified"]["verified_recovery"]
    assert demo["roles"]["original"] == demo["roles"]["filtered"]
    assert PCM.from_dict(demo["roles"]["observed"]).samples != PCM.from_dict(demo["roles"]["original"]).samples
    assert validate_demonstration(demo, **pins(source))["all_passed"]
    forged = deepcopy(demo)
    forged["roles"]["rejected"] = forged["roles"]["candidate"]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_demonstration(rehash(forged), **pins(source))


def test_rehashed_battery_aggregate_tampering():
    source = build_battery_source(cases=("clean",), alphabets=(3,))
    battery = run_comparative_battery(source)
    battery["summary"][0]["exact_model_success_frames"] = 100
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        validate_comparative_battery(rehash(battery), expected_source_sha256=source["sha256"])


def test_export_wav_json_sources_native_bytes_manifest_and_tamper(tmp_path, export_module):
    output = tmp_path / "export"
    result = export_module.generate(output, cases=("broadband",), alphabets=(3,))
    kwargs = {"expected_manifest_sha256": result["manifest_sha256"],
              "expected_battery_source_sha256": result["battery_source_sha256"],
              "expected_demonstration_source_sha256": result["demonstration_source_sha256"],
              "expected_reference_sha256": result["measured_reference_sha256"]}
    assert export_module.validate_export(output, **kwargs)["all_passed"]
    manifest = export_module._read_json(output / "manifest.json")
    environment = export_module._read_json(output / "environment.json")
    assert type(environment["runtime_ns"]) is int and environment["runtime_ns"] > 0
    assert environment["peak_python_tracemalloc_bytes"] > 0
    assert environment["performance_is_canonical_proof"] is False
    assert type(environment["clean_working_tree_at_start"]) is bool
    paths = {r["path"] for r in environment["source_files"]}
    assert {"src/qec/sonify/canonical.py", "src/qec/decoder/qutrit/exact.py",
            "src/qec/decoder/ququart/exact.py", "src/qec/experiments/acoustic/evidence.py"} <= paths
    assert len(manifest["transport_demonstration"]) == 6
    assert manifest["digital_binding_specimen"]["payload_bytes"] < 512
    assert manifest["artifact_total_bytes_excluding_manifest"] < 6_000_000
    for role in ("original", "observed", "filtered"):
        pcm = read_wav((output / "lab" / (role + ".wav")).read_bytes())
        assert pcm.as_dict() == export_module._read_json(output / "lab" / (role + ".json"))
    for row in manifest["transport_demonstration"]:
        receipt = export_module._read_json(output / row["path"])
        role = export_module._read_json(output / "lab" / (row["role"] + ".json"))
        assert bytes.fromhex(receipt["reassembled_payload_hex"]) == canonical_bytes(role)
    with pytest.raises(ValueError, match="empty directory"):
        export_module.generate(output, cases=("clean",), alphabets=(3,))
    # A fresh manifest hash cannot make unsupported scientific claims valid.
    for field in ("spectral_repository_or_ui_integration", "environment_metadata_is_canonical_scientific_proof"):
        forged_manifest = rehash({**manifest, field: True})
        export_module._write_json(output / "manifest.json", forged_manifest)
        forged_kwargs = {**kwargs, "expected_manifest_sha256": forged_manifest["sha256"]}
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            export_module.validate_export(output, **forged_kwargs)
    export_module._write_json(output / "manifest.json", manifest)
    # Valid JSON with a forged role, plus a fully rehashed file manifest, still
    # cannot pass independent source replay even if the new manifest is pinned.
    target = output / "lab/rejected.json"
    export_module._write_json(target, export_module._read_json(output / "lab/candidate.json"))
    for row in manifest["files"]:
        if row["path"] == "lab/rejected.json":
            row["sha256"] = export_module._sha(target.read_bytes())
            row["bytes"] = target.stat().st_size
    manifest["artifact_total_bytes_excluding_manifest"] = sum(f["bytes"] for f in manifest["files"])
    manifest = rehash(manifest)
    export_module._write_json(output / "manifest.json", manifest)
    kwargs["expected_manifest_sha256"] = manifest["sha256"]
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.validate_export(output, **kwargs)


@pytest.mark.parametrize("field,value", [("spectral_repository_or_ui_integration", True),
    ("environment_metadata_is_canonical_scientific_proof", True), ("schema", "unknown"),
    ("contract_version", "999"), ("transport_scope", "physical_network"), ("unknown", 1)])
def test_manifest_claim_schema_shape_cannot_be_reauthorized_by_fresh_pin(export_module, field, value):
    valid = {**export_module._MANIFEST_FIXED, **{key: None for key in export_module._MANIFEST_DYNAMIC}}
    forged = {**valid, field: value}
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._strict_metadata(forged, export_module._MANIFEST_FIXED, export_module._MANIFEST_DYNAMIC)


@pytest.mark.parametrize("field,value", [("performance_is_canonical_proof", True),
    ("physical_latency_or_hardware_QA", True), ("performance_is_environment_dependent", False),
    ("measurement_scope", "hardware_RSS"), ("schema", "unknown"), ("unknown", 1)])
def test_environment_claim_schema_shape_is_strict(export_module, field, value):
    valid = {**export_module._ENVIRONMENT_FIXED, **{key: None for key in export_module._ENVIRONMENT_DYNAMIC}}
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._strict_metadata({**valid, field: value}, export_module._ENVIRONMENT_FIXED,
                                       export_module._ENVIRONMENT_DYNAMIC)


def test_export_bound_fails_before_writing_oversize_files(tmp_path, export_module, monkeypatch):
    monkeypatch.setattr(export_module, "MAX_EXPORT_TOTAL_BYTES", 1024)
    output = tmp_path / "bounded"
    with pytest.raises(ValueError, match="32 MiB bound"):
        export_module.generate(output, cases=("clean",), alphabets=(3,))
    assert not any(output.rglob("*.json"))


def test_manifest_paths_are_portable_forward_slashes(export_module):
    root = PureWindowsPath("C:/work/QEC")
    file = root / "src/qec/decoder/qutrit/exact.py"
    assert export_module._portable_relative(file, root) == "src/qec/decoder/qutrit/exact.py"
    assert all("\\" not in row["path"] for row in export_module._code_manifest())


def test_export_inventory_preflight_rejects_cumulative_size_before_payload_read(tmp_path, export_module, monkeypatch):
    manifest = {**export_module._MANIFEST_FIXED,
                **{key: None for key in export_module._MANIFEST_DYNAMIC},
                "battery_source_sha256": "a" * 64, "demonstration_source_sha256": "b" * 64,
                "measured_reference_sha256": "c" * 64, "files": []}
    for name in ("first.bin", "second.bin"):
        path = tmp_path / name
        with path.open("wb") as stream:
            stream.truncate(17 * 1024 * 1024)
        manifest["files"].append({"path": name, "sha256": "0" * 64, "bytes": path.stat().st_size})
    manifest = rehash(manifest)
    export_module._write_json(tmp_path / "manifest.json", manifest)
    original = Path.read_bytes
    payload_reads = []
    def guarded_read(path):
        if path.name != "manifest.json":
            payload_reads.append(path.name)
            pytest.fail("oversized inventory read artifact bytes before total-size preflight")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.validate_export(tmp_path, expected_manifest_sha256=manifest["sha256"],
            expected_battery_source_sha256="a" * 64, expected_demonstration_source_sha256="b" * 64,
            expected_reference_sha256="c" * 64)
    assert payload_reads == []
