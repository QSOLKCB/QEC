# SPDX-License-Identifier: MPL-2.0
"""Bounded paired comparison, adverse controls, and fully rehashed attacks."""
from copy import deepcopy
import importlib.util
import inspect
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
from types import ModuleType, SimpleNamespace

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
    # The manifest exemption applies only to the root file, never a basename
    # elsewhere in the exported directory. The original external pins remain.
    extra = output / "extra"
    extra.mkdir()
    (extra / "manifest.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.validate_export(output, **kwargs)
    (extra / "manifest.json").unlink()
    extra.rmdir()
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
    original = export_module._InventoryAccess.read
    payload_reads = []
    def guarded_read(self, name, **kwargs):
        if name != "manifest.json":
            payload_reads.append(name)
            pytest.fail("oversized inventory read artifact bytes before total-size preflight")
        return original(self, name, **kwargs)
    monkeypatch.setattr(export_module._InventoryAccess, "read", guarded_read)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.validate_export(tmp_path, expected_manifest_sha256=manifest["sha256"],
            expected_battery_source_sha256="a" * 64, expected_demonstration_source_sha256="b" * 64,
            expected_reference_sha256="c" * 64)
    assert payload_reads == []


def _minimal_export_manifest(output, export_module, files=()):
    manifest = rehash({**export_module._MANIFEST_FIXED,
        **{key: None for key in export_module._MANIFEST_DYNAMIC},
        "battery_source_sha256": "a" * 64, "demonstration_source_sha256": "b" * 64,
        "measured_reference_sha256": "c" * 64, "files": list(files)})
    export_module._write_json(output / "manifest.json", manifest)
    return {"expected_manifest_sha256": manifest["sha256"], "expected_battery_source_sha256": "a" * 64,
            "expected_demonstration_source_sha256": "b" * 64, "expected_reference_sha256": "c" * 64}


def _bounded_subprocess(script, export_module):
    header = ("import importlib.util, pathlib, os\n"
              f"spec=importlib.util.spec_from_file_location('export', {export_module.__file__!r})\n"
              "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n")
    result = subprocess.run([sys.executable, "-c", header + script], text=True, capture_output=True,
                            timeout=5, env={**os.environ, "PYTHONPATH": str(export_module.ROOT / "src")})
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "INVALID_INPUT"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO support required")
@pytest.mark.parametrize("name", ["manifest.json", "battery.json", "lab/original.wav", "unlisted.fifo"])
def test_export_fifo_entries_fail_closed_without_blocking(tmp_path, export_module, name):
    files = [] if name in ("manifest.json", "unlisted.fifo") else [{"path": name, "sha256": "0" * 64, "bytes": 0}]
    kwargs = _minimal_export_manifest(tmp_path, export_module, files)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    os.mkfifo(path)
    # A broken implementation is killed by the parent test after five seconds;
    # the FIFO has no writer and is never left in an unbounded pytest read.
    _bounded_subprocess(f"""
try:
    module.validate_export(pathlib.Path({str(tmp_path)!r}), **{kwargs!r})
except ValueError as error:
    assert str(error) == 'INVALID_INPUT'
    print(str(error))
else:
    raise AssertionError('FIFO evidence was accepted')
""", export_module)


@pytest.mark.skipif(not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"), reason="POSIX nonblocking FIFO support required")
def test_regular_file_swapped_to_fifo_during_open_cannot_block(tmp_path, export_module):
    target = tmp_path / "payload.json"
    target.write_bytes(b"{}\n")
    _bounded_subprocess(f"""
target = pathlib.Path({str(target)!r})
original_open = os.open
original_read = os.read
reads = []
def raced_open(path, flags, *args, **kwargs):
    assert flags & os.O_NONBLOCK
    target.unlink()
    os.mkfifo(target)
    return original_open(path, flags, *args, **kwargs)
def guarded_read(*args):
    reads.append(args)
    raise AssertionError('nonregular descriptor reached read')
os.open = raced_open
os.read = guarded_read
try:
    module._read_regular_bytes(target)
except ValueError as error:
    assert str(error) == 'INVALID_INPUT'
    assert not reads
    print(str(error))
else:
    raise AssertionError('FIFO replacement was accepted')
""", export_module)


@pytest.mark.skipif(os.name != "posix", reason="portable symlink creation is not available to every Windows user")
@pytest.mark.parametrize("kind", ["symlink_file", "symlink_directory", "socket"])
def test_unlisted_nonregular_inventory_entries_are_rejected(tmp_path, export_module, monkeypatch, kind):
    import socket
    kwargs = _minimal_export_manifest(tmp_path, export_module)
    node = tmp_path / "unlisted"
    connection = None
    try:
        if kind == "socket":
            try:
                connection = socket.socket(socket.AF_UNIX)
                # AF_UNIX pathname limits are shorter than some hosted pytest
                # temporary paths; bind the same real socket by its short name.
                monkeypatch.chdir(tmp_path)
                connection.bind(node.name)
            except PermissionError:
                pytest.skip("sandbox denies creation of a Unix-domain socket")
        elif kind == "symlink_file":
            node.symlink_to(tmp_path / "manifest.json")
        else:
            node.symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="^INVALID_INPUT$"):
            export_module.validate_export(tmp_path, **kwargs)
    finally:
        if connection is not None:
            connection.close()


@pytest.mark.parametrize("change", ["grow", "truncate", "replace"])
def test_descriptor_identity_rejects_preopen_mutation_before_read(tmp_path, export_module, monkeypatch, change):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    original_open = os.open
    reads = []
    def raced_open(target, flags, *args, **kwargs):
        if change == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(b"different")
            replacement.replace(path)
        else:
            path.write_bytes(b"changed-and-longer" if change == "grow" else b"")
        return original_open(target, flags, *args, **kwargs)
    def forbidden_read(*args):
        reads.append(args)
        pytest.fail("changed descriptor should be rejected before any read")
    monkeypatch.setattr(os, "open", raced_open)
    monkeypatch.setattr(os, "read", forbidden_read)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._read_regular_bytes(path)
    assert not reads


@pytest.mark.parametrize("change", ["grow", "truncate"])
def test_bounded_descriptor_read_rejects_growth_or_short_read(tmp_path, export_module, monkeypatch, change):
    path = tmp_path / "payload"
    path.write_bytes(b"original")
    initial_size = path.stat().st_size
    original_read = os.read
    requested = []
    def raced_read(descriptor, count):
        requested.append(count)
        if len(requested) == 1:
            path.write_bytes(b"much-longer-than-original" if change == "grow" else b"")
        return original_read(descriptor, count)
    monkeypatch.setattr(os, "read", raced_read)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._read_regular_bytes(path)
    assert requested == [initial_size + 1]


def test_file_reader_accepts_only_regular_bounded_bytes(tmp_path, export_module, monkeypatch):
    path = tmp_path / "regular"
    observed = []
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    original_lstat, original_fstat = Path.lstat, os.fstat
    def record_lstat(target, *args, **kwargs):
        info = original_lstat(target, *args, **kwargs)
        if target == path:
            observed.append({"api": "lstat", **{field: getattr(info, field) for field in fields}})
        return info
    def record_fstat(descriptor):
        info = original_fstat(descriptor)
        observed.append({"api": "fstat", **{field: getattr(info, field) for field in fields}})
        return info
    monkeypatch.setattr(Path, "lstat", record_lstat)
    monkeypatch.setattr(os, "fstat", record_fstat)
    for content in (b"", b"{}\n", bytes(range(256)) * 300):
        path.write_bytes(content)
        observed.clear()
        try:
            actual = export_module._read_regular_bytes(path)
        except ValueError:
            # Record only the actual guard's metadata, never payload bytes or
            # paths. Keep the failure and all production checks unchanged.
            pytest.fail(f"valid regular file rejected; platform={sys.platform}; "
                        f"python={sys.version_info[:3]}; metadata={observed!r}", pytrace=False)
        assert actual == content
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._read_regular_bytes(tmp_path)


def test_windows_path_descriptor_identity_ignores_ctime_difference(export_module, monkeypatch):
    monkeypatch.setattr(export_module.os, "name", "nt")
    path_stat = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=5)
    descriptor_stat = SimpleNamespace(st_dev=1, st_ino=2, st_size=3, st_mtime_ns=4, st_ctime_ns=6)

    assert not export_module._same_file_identity(path_stat, descriptor_stat)
    assert export_module._same_file_identity(path_stat, descriptor_stat, path_descriptor=True)
    descriptor_stat.st_ino = 7
    assert not export_module._same_file_identity(path_stat, descriptor_stat, path_descriptor=True)


def test_inventory_excludes_exact_root_manifest_only(tmp_path, export_module):
    (tmp_path / "manifest.json").write_bytes(b"{}\n")
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra/manifest.json").write_bytes(b"{}\n")
    assert export_module._inventory_files(tmp_path) == ["extra/manifest.json"]


@pytest.mark.parametrize("kind", ["files", "directories", "depth"])
def test_inventory_discovery_has_finite_file_directory_and_depth_bounds(tmp_path, export_module, kind):
    if kind == "files":
        for index in range(129):
            (tmp_path / f"{index}.json").write_bytes(b"{}\n")
    elif kind == "directories":
        for index in range(17):
            (tmp_path / str(index)).mkdir()
    else:
        (tmp_path / "a/b/c/d/e").mkdir(parents=True)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._inventory_files(tmp_path)


@pytest.mark.parametrize("builder_name", ["build_battery_source", "build_trial_source"])
def test_generation_pins_source_before_any_fixture_builder(tmp_path, export_module, monkeypatch, builder_name):
    source_file = tmp_path / "source.py"
    source_file.write_bytes(b"before")
    snapshots = []
    def source_manifest():
        value = [{"path": "source.py", "sha256": export_module._sha(source_file.read_bytes()),
                  "bytes": source_file.stat().st_size}]
        snapshots.append(value)
        return value
    original = getattr(export_module, builder_name)
    invoked = []
    def mutate_during_builder(*args, **kwargs):
        assert snapshots, "source snapshot must precede the first fixture computation"
        result = original(*args, **kwargs)
        source_file.write_bytes(b"after")
        invoked.append(True)
        return result
    monkeypatch.setattr(export_module, "_code_manifest", source_manifest)
    monkeypatch.setattr(export_module, builder_name, mutate_during_builder)
    # This fixture deliberately replaces executable functions. Isolate the
    # existing mid-generation source-pin check from the separate attestation
    # guard, which correctly rejects monkeypatched loaded code up front.
    monkeypatch.setattr(export_module, "_loaded_code_matches_disk", lambda: None)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="source changed during export"):
        export_module.generate(output, cases=("clean",), alphabets=(3,))
    assert invoked and not output.exists()
    assert snapshots[0] != snapshots[-1]


def test_generation_rejects_oversized_environment_before_computation(tmp_path, export_module, monkeypatch):
    original_git = export_module._git
    calls = []
    def oversized_status(*args):
        return "x" * 1_000_001 if args == ("status", "--porcelain") else original_git(*args)
    def forbidden_generation(*args, **kwargs):
        calls.append(True)
        pytest.fail("invalid environment must fail before source generation")
    monkeypatch.setattr(export_module, "_git", oversized_status)
    monkeypatch.setattr(export_module, "build_battery_source", forbidden_generation)
    output = tmp_path / "oversized"
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.generate(output, cases=("clean",), alphabets=(3,))
    assert not calls and not output.exists()


def test_generation_environment_schema_accepts_exact_status_boundary(export_module):
    value = export_module._environment_snapshot("a" * 40, "x" * 1_000_000, export_module._code_manifest())
    export_module._validate_environment(value)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._validate_environment({**value, "git_status_porcelain_at_start": "x" * 1_000_001})


def _different_stat(info, **changes):
    from types import SimpleNamespace
    keys = ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return SimpleNamespace(**{**{key: getattr(info, key) for key in keys}, **changes})


def test_windows_cross_api_ctime_semantics_keep_valid_reader_bytes(tmp_path, export_module, monkeypatch):
    path = tmp_path / "valid"
    path.write_bytes(b"known bytes")
    original_identity, original_fstat = export_module._same_file_identity, os.fstat
    def windows_identity(left, right, *, path_descriptor=False):
        with monkeypatch.context() as context:
            context.setattr(export_module, "os", SimpleNamespace(name="nt"))
            return original_identity(left, right, path_descriptor=path_descriptor)
    monkeypatch.setattr(export_module, "_same_file_identity", windows_identity)
    monkeypatch.setattr(os, "fstat", lambda fd: _different_stat(original_fstat(fd), st_ctime_ns=original_fstat(fd).st_ctime_ns + 1000))
    assert export_module._read_regular_bytes(path) == b"known bytes"


@pytest.mark.parametrize("api", ["lstat", "fstat"])
def test_windows_compatibility_still_checks_full_same_api_ctime(tmp_path, export_module, monkeypatch, api):
    path = tmp_path / "payload"
    path.write_bytes(b"fixed")
    identity = export_module._same_file_identity
    def windows_identity(left, right, *, path_descriptor=False):
        with monkeypatch.context() as context:
            context.setattr(export_module, "os", SimpleNamespace(name="nt"))
            return identity(left, right, path_descriptor=path_descriptor)
    monkeypatch.setattr(export_module, "_same_file_identity", windows_identity)
    calls = []
    if api == "lstat":
        original = Path.lstat
        def changed(target, *args, **kwargs):
            info = original(target, *args, **kwargs)
            if target == path:
                calls.append(True)
                return _different_stat(info, st_ctime_ns=info.st_ctime_ns + int(len(calls) > 1))
            return info
        monkeypatch.setattr(Path, "lstat", changed)
    else:
        original = os.fstat
        def changed(descriptor):
            info = original(descriptor)
            calls.append(True)
            return _different_stat(info, st_ctime_ns=info.st_ctime_ns + 1000 + int(len(calls) > 1))
        monkeypatch.setattr(os, "fstat", changed)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._read_regular_bytes(path)
    assert len(calls) >= 2


def test_regular_mode_reparse_file_is_rejected_before_open(tmp_path, export_module, monkeypatch):
    path = tmp_path / "reparse"
    path.write_bytes(b"data")
    original = path.lstat()
    info = _different_stat(original, st_file_attributes=0x400)
    monkeypatch.setattr(Path, "lstat", lambda *args, **kwargs: info)
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("reparse file opened"))
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._read_regular_bytes(path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory descriptor regression")
@pytest.mark.parametrize("stage", ["before_open", "before_scan", "during_snapshot"])
def test_directory_symlink_replacement_never_traverses_outside(tmp_path, export_module, stage):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "child").mkdir()
    (outside / "must_not_visit").write_bytes(b"outside")
    _bounded_subprocess(f"""
root, outside = pathlib.Path({str(root)!r}), pathlib.Path({str(outside)!r})
child = root / 'child'
initial_inode = child.stat().st_ino
outside_inode = outside.stat().st_ino
original_open, original_scan = os.open, os.scandir
changed = []
def replace_child():
    if not changed:
        child.rename(root / 'old-child')
        child.symlink_to(outside, target_is_directory=True)
        changed.append(True)
def checked_scan(fd):
    assert isinstance(fd, int), 'directory traversal must use a retained fd'
    assert os.fstat(fd).st_ino != outside_inode, 'outside directory was visited'
    if {stage!r} == 'before_scan' and os.fstat(fd).st_ino == initial_inode:
        replace_child()
    return original_scan(fd)
def raced_open(path, flags, *args, **kwargs):
    if {stage!r} == 'before_open' and path == 'child':
        replace_child()
    return original_open(path, flags, *args, **kwargs)
os.open, os.scandir = raced_open, checked_scan
os.supports_dir_fd = os.supports_dir_fd | {{raced_open}}
os.supports_fd = os.supports_fd | {{checked_scan}}
try:
    with module._open_inventory(root) as names:
        assert 'child/must_not_visit' not in names
        if {stage!r} == 'during_snapshot':
            replace_child()
except ValueError as error:
    assert changed, 'race injection was not reached'
    assert str(error) == 'INVALID_INPUT'
    print(str(error))
else:
    raise AssertionError('directory replacement should fail closed')
""", export_module)


def _windows_directory_record(name, *, attributes=0x80, file_id=b"f" * 16, tag=0, following=0):
    import struct
    encoded = name.encode("utf-16-le")
    return struct.pack("<II6q4I16s", following, 0, 10, 20, 30, 40, 0, 0,
                       attributes, len(encoded), 0, tag, file_id) + encoded


def test_windows_directory_parser_preserves_full_identity_and_undefined_tag(export_module):
    value = _windows_directory_record("valid.json", file_id=bytes(range(16)), tag=0x12345678)
    assert export_module._parse_windows_directory_buffer(value) == [("valid.json", 0x80, 0x12345678, bytes(range(16)))]


@pytest.mark.parametrize("kind", ["odd_name", "truncated", "backward", "outside", "misaligned", "separator", "nul", "utf16", "too_many"])
def test_windows_directory_parser_rejects_malformed_native_records(export_module, kind):
    import struct
    value = bytearray(_windows_directory_record("name"))
    if kind == "odd_name":
        struct.pack_into("<I", value, 60, 3)
    elif kind == "truncated":
        value = value[:-1]
    elif kind == "backward":
        struct.pack_into("<I", value, 0, 8)
    elif kind == "outside":
        struct.pack_into("<I", value, 0, 65536)
    elif kind == "misaligned":
        struct.pack_into("<I", value, 0, len(value) + 1)
    elif kind == "separator":
        value = _windows_directory_record("a/b")
    elif kind == "nul":
        value = _windows_directory_record("a\0b")
    elif kind == "utf16":
        value[88:90] = b"\x00\xdc"
    else:
        record = _windows_directory_record("abcd", following=96)
        value = record * 160 + _windows_directory_record("last")
    with pytest.raises(ValueError):
        export_module._parse_windows_directory_buffer(bytes(value))


def _mock_windows_kernel(monkeypatch, mode):
    import ctypes
    import struct
    from types import SimpleNamespace
    state = SimpleNamespace(opened=[], closed=[], queries=[], enumerated={}, error=0, mutation=False,
                            open_paths=[], id_opens=[], id_kinds=[], reads=[], offsets={}, file_handles=set(), data=b"retained payload")
    class NativeFunction:
        def __init__(self, callback):
            self.callback = callback
        def __call__(self, *args):
            return self.callback(*args)
    def identifier(value):
        return value[:8] + b"\0" * 8 if mode in ("ntfs_low64", "low64_denied", "low64_sharing", "low64_mismatch", "low64_volume") else value
    def create(path, access, sharing, security, disposition, flags, template):
        assert access == 0x81 and sharing == 1 and disposition == 3 and flags == 0x02200000
        assert state.opened == [], "only the root may be opened by pathname"
        state.open_paths.append(path)
        handle = len(state.opened) + 1
        state.opened.append(handle)
        return handle
    def open_id(hint, descriptor, access, sharing, security, flags):
        assert hint == 1 and access in (0x81, 0x80000000) and sharing == 1 and flags == 0x02200000
        size, kind, native_id = struct.unpack("<II16s", ctypes.string_at(descriptor, 24))
        assert size == 24 and kind in (0, 2)
        assert kind != 0 or native_id[8:] == b"\0" * 8
        state.id_opens.append((native_id, access))
        state.id_kinds.append(kind)
        if mode == "id_unsupported" or mode in ("ntfs_low64", "low64_denied", "low64_sharing", "low64_mismatch", "low64_volume") and kind == 2:
            state.error = 5 if mode == "low64_denied" else 32 if mode == "low64_sharing" else 87
            return ctypes.c_void_p(-1).value
        handle = len(state.opened) + 1
        state.opened.append(handle)
        if access == 0x80000000:
            state.file_handles.add(handle)
        return handle
    def query(handle, kind, buffer, size):
        state.queries.append((handle, kind))
        attributes = 0x80 if handle in state.file_handles else 0x10
        if handle == 2 and (mode == "reparse_child" or state.mutation and mode == "junction"):
            attributes |= 0x400
        if handle >= 3 and mode == "reparse_file":
            attributes |= 0x400
        if kind == 9:
            # Tag is undefined for a non-reparse directory; arbitrary values
            # must not make otherwise valid directory metadata fail.
            value = struct.pack("<II", attributes, 0x12345678)
        elif kind == 18:
            native_id = identifier(b"f" * 16 if handle in state.file_handles else bytes([handle]) * 16)
            if handle >= 3 and mode == "wrong_file_id":
                native_id = native_id[:15] + b"x"
            if handle == 2 and mode in ("wrong_high_id", "low64_mismatch"):
                native_id = native_id[:15] + b"z"
            if mode == "zero_id":
                native_id = b"\0" * 16
            volume = 99 + int(handle == 2 and mode in ("wrong_volume", "low64_volume"))
            value = struct.pack("<Q16s", volume, native_id)
        elif kind == 0:
            change = 40 + int(state.mutation and (handle == 2 or mode == "file_mutation"))
            value = struct.pack("<4qI4x", 10, 20, 30, change, attributes)
        elif kind == 1:
            file_size = 32 * 1024 * 1024 + 1 if mode == "oversize_file" else len(state.data)
            value = struct.pack("<qqIBB2x", file_size, file_size, 1, 0, 0)
        elif kind == 0x13:
            count = state.enumerated.get(handle, 0)
            state.enumerated[handle] = count + 1
            if mode == "endless_dots":
                value = _windows_directory_record(".", attributes=0x10)
            elif mode == "unsupported" and handle == 2:
                state.error = 87
                return 0
            elif count:
                state.error = 18
                return 0
            elif handle == 1 and mode == "root_manifest":
                value = _windows_directory_record("manifest.json")
            elif handle == 1:
                value = _windows_directory_record("child", attributes=0x10, file_id=identifier(b"\x02" * 16), tag=0xFFFFFFFF)
            else:
                value = _windows_directory_record("payload.json", file_id=identifier(b"f" * 16), tag=0xFFFFFFFF)
        else:
            pytest.fail(f"unexpected native information class {kind}")
        assert len(value) <= size
        ctypes.memmove(buffer, value, len(value))
        return 1
    def read(handle, buffer, requested, returned, overlapped):
        assert handle in state.file_handles and requested <= 65536 and overlapped is None
        state.reads.append((handle, requested))
        if mode == "read_error":
            return 0
        data = state.data[:-1] if mode == "short_read" else state.data + b"x" if mode == "growing_read" else state.data
        offset = state.offsets.get(handle, 0)
        chunk = data[offset:offset + requested]
        state.offsets[handle] = offset + len(chunk)
        ctypes.memmove(buffer, chunk, len(chunk))
        from ctypes import wintypes
        ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD)).contents.value = len(chunk)
        if mode == "file_mutation":
            state.mutation = True
        return 1
    def seek(handle, distance, new_position, method):
        assert distance == 0 and method == 0 and new_position is None
        state.offsets[handle] = 0
        return int(mode != "seek_error")
    def close(handle):
        state.closed.append(handle)
        return 1
    kernel = SimpleNamespace(CreateFileW=NativeFunction(create), GetFileInformationByHandleEx=NativeFunction(query),
        OpenFileById=NativeFunction(open_id), GetFileType=NativeFunction(lambda handle: 3 if mode == "pipe_file" else 1),
        ReadFile=NativeFunction(read), SetFilePointerEx=NativeFunction(seek), CloseHandle=NativeFunction(close))
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: kernel, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: state.error, raising=False)
    return state


def test_windows_native_inventory_retains_handles_and_ignores_undefined_tags(tmp_path, export_module, monkeypatch):
    state = _mock_windows_kernel(monkeypatch, "valid")
    with export_module._inventory_windows(tmp_path) as names:
        assert names == ["child/payload.json"]
        assert state.closed == []
    assert state.closed == [2, 1]


@pytest.mark.parametrize("mode", ["wrong_high_id", "wrong_volume", "reparse_child", "zero_id", "unsupported", "endless_dots", "mutation", "junction"])
def test_windows_native_inventory_fails_closed_and_closes_handles(tmp_path, export_module, monkeypatch, mode):
    state = _mock_windows_kernel(monkeypatch, mode)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        with export_module._inventory_windows(tmp_path) as names:
            assert mode in ("mutation", "junction")
            assert names == ["child/payload.json"]
            state.mutation = True
    assert sorted(state.closed) == state.opened
    if mode in ("wrong_high_id", "wrong_volume", "reparse_child"):
        assert (2, 0x13) not in state.queries, "rejected child must never be enumerated"
    if mode == "endless_dots":
        assert state.queries.count((1, 0x13)) == 256


@pytest.mark.skipif(os.name != "nt", reason="actual Windows junction mutation regression")
def test_windows_held_directory_reparse_mutation_is_prevented_or_rejected(tmp_path, export_module):
    import ctypes
    from ctypes import wintypes
    import struct
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    child = root / "child"
    child.mkdir()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    control = kernel.DeviceIoControl
    control.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                        wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID)
    control.restype = wintypes.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = (wintypes.HANDLE,), wintypes.BOOL
    changed = False
    try:
        try:
            with export_module._open_inventory(root):
                handle = create(str(child), 0x100, 7, None, 3, 0x02200000, None)  # FILE_WRITE_ATTRIBUTES
                if handle in (None, ctypes.c_void_p(-1).value):
                    assert ctypes.get_last_error() in (5, 32), "unexpected directory-write failure"
                    return  # Sharing/permission protection prevented the mutation.
                try:
                    substitute = ("\\??\\" + str(outside)).encode("utf-16-le")
                    printable = str(outside).encode("utf-16-le")
                    names = substitute + b"\0\0" + printable + b"\0\0"
                    value = struct.pack("<I6H", 0xA0000003, 8 + len(names), 0,
                                        0, len(substitute), len(substitute) + 2, len(printable)) + names
                    data = ctypes.create_string_buffer(value)
                    returned = wintypes.DWORD()
                    if not control(handle, 0x000900A4, data, len(value), None, 0, ctypes.byref(returned), None):
                        assert ctypes.get_last_error() in (5, 32), "unexpected reparse mutation failure"
                        return
                    changed = True
                finally:
                    close(handle)
        except ValueError as error:
            assert changed and str(error) == "INVALID_INPUT"
        else:
            pytest.fail("successful junction mutation was accepted after snapshot")
    finally:
        if changed:
            child.rmdir()  # Remove the junction itself, never its target contents.


@pytest.mark.skipif(os.name != "posix", reason="POSIX descriptor-relative payload regression")
@pytest.mark.parametrize("stage", ["manifest", "payload_stat", "payload_open", "payload_read"])
def test_parent_replacement_never_stats_opens_or_reads_outside(tmp_path, export_module, stage):
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "lab").mkdir(parents=True)
    (outside / "lab").mkdir(parents=True)
    payload = b"original payload"
    (root / "lab/payload.json").write_bytes(payload)
    (outside / "lab/payload.json").write_bytes(b"outside payload")
    (outside / "manifest.json").write_bytes(b"outside manifest")
    kwargs = _minimal_export_manifest(root, export_module, [{"path": "lab/payload.json",
        "sha256": export_module._sha(payload), "bytes": len(payload)}])
    _bounded_subprocess(f"""
root, outside = pathlib.Path({str(root)!r}), pathlib.Path({str(outside)!r})
outside_inodes = {{p.stat().st_ino for p in outside.rglob('*')}} | {{outside.stat().st_ino}}
payload_inode = (root / 'lab/payload.json').stat().st_ino
original_stat, original_lstat, original_open, original_read = os.stat, os.lstat, os.open, os.read
original_access_read, original_access_stat = module._InventoryAccess.read, module._InventoryAccess.stat
changed, payload_reads = [], []
def replace_parent():
    if not changed:
        if {stage!r} == 'manifest':
            root.rename(root.with_name('retained-root'))
            root.symlink_to(outside, target_is_directory=True)
        else:
            (root / 'lab').rename(root / 'retained-lab')
            (root / 'lab').symlink_to(outside / 'lab', target_is_directory=True)
        changed.append(True)
def checked_stat(function):
    def check(path, *args, **kwargs):
        info = function(path, *args, **kwargs)
        assert info.st_ino not in outside_inodes, 'outside inode metadata was observed'
        return info
    return check
def checked_open(path, flags, *args, **kwargs):
    if {stage!r} == 'payload_open' and pathlib.Path(path).name == 'payload.json':
        replace_parent()
    if changed:
        assert kwargs.get('dir_fd') is not None, 'file opened through an ancestor pathname'
        assert len(pathlib.Path(path).parts) == 1, 'file open must use a discovered basename'
    descriptor = original_open(path, flags, *args, **kwargs)
    assert os.fstat(descriptor).st_ino not in outside_inodes, 'outside descriptor opened'
    return descriptor
def checked_read(descriptor, count):
    inode = os.fstat(descriptor).st_ino
    assert inode not in outside_inodes, 'outside descriptor reached read'
    if inode == payload_inode:
        payload_reads.append(True)
        if {stage!r} == 'payload_read':
            replace_parent()
    return original_read(descriptor, count)
def access_read(self, name, **kwargs):
    if {stage!r} == 'manifest' and name == 'manifest.json':
        replace_parent()
    return original_access_read(self, name, **kwargs)
def access_stat(self, name):
    if {stage!r} == 'payload_stat':
        replace_parent()
    return original_access_stat(self, name)
os.stat, os.lstat = checked_stat(original_stat), checked_stat(original_lstat)
os.open, os.read = checked_open, checked_read
os.supports_dir_fd = os.supports_dir_fd | {{os.open, os.stat}}
os.supports_follow_symlinks = os.supports_follow_symlinks | {{os.stat}}
module._InventoryAccess.read, module._InventoryAccess.stat = access_read, access_stat
try:
    module.validate_export(root, **{kwargs!r})
except ValueError as error:
    assert changed, 'race injection was not reached'
    assert payload_reads, 'descriptor-relative original payload was not reached'
    assert str(error) == 'INVALID_INPUT'
    print(str(error))
else:
    raise AssertionError('parent replacement was accepted')
""", export_module)


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "lab//data.json", "./data.json",
                                  "lab/../data.json", "C:/outside.json", "lab\\data.json", "unknown.json"])
def test_manifest_unknown_or_aliased_names_fail_before_payload_access(tmp_path, export_module, monkeypatch, name):
    kwargs = _minimal_export_manifest(tmp_path, export_module, [{"path": name, "sha256": "0" * 64, "bytes": 0}])
    monkeypatch.setattr(export_module._InventoryAccess, "stat", lambda *args: pytest.fail("unknown payload stat"))
    original = export_module._InventoryAccess.read
    def read(self, name, **kwargs):
        assert name == "manifest.json", "unknown payload read"
        return original(self, name, **kwargs)
    monkeypatch.setattr(export_module._InventoryAccess, "read", read)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module.validate_export(tmp_path, **kwargs)


def test_windows_payload_snapshot_uses_only_native_file_ids(tmp_path, export_module, monkeypatch):
    state = _mock_windows_kernel(monkeypatch, "valid")
    monkeypatch.setattr(Path, "lstat", lambda *args: pytest.fail("pathname payload metadata"))
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("pathname payload open"))
    with export_module._inventory_windows(tmp_path, file_access=True) as access:
        info = access.stat("child/payload.json")
        assert info.st_size == len(state.data)
        assert state.reads == []
        assert access.read("child/payload.json", expected_stat=info) == state.data
        assert access.read("child/payload.json", expected_stat=info) == state.data
        assert state.closed == []
    assert len(state.open_paths) == 1
    assert state.id_opens == [(b"\x02" * 16, 0x81), (b"f" * 16, 0x80000000)]
    assert state.closed == [3, 2, 1]


@pytest.mark.parametrize("mode", ["id_unsupported", "wrong_file_id", "pipe_file", "reparse_file", "oversize_file",
                                  "short_read", "growing_read", "read_error", "seek_error", "file_mutation"])
def test_windows_payload_native_guards_fail_closed(tmp_path, export_module, monkeypatch, mode):
    state = _mock_windows_kernel(monkeypatch, mode)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        with export_module._inventory_windows(tmp_path, file_access=True) as access:
            info = access.stat("child/payload.json")
            access.read("child/payload.json", expected_stat=info)
    assert sorted(state.closed) == state.opened
    if mode in ("id_unsupported", "wrong_file_id", "pipe_file", "reparse_file", "oversize_file", "seek_error"):
        assert state.reads == []
    if mode == "id_unsupported":
        assert state.id_kinds == [2], "nonzero high ID bits must never be truncated for a retry"


def test_windows_manifest_read_uses_inventoried_file_id(tmp_path, export_module, monkeypatch):
    state = _mock_windows_kernel(monkeypatch, "root_manifest")
    monkeypatch.setattr(Path, "lstat", lambda *args: pytest.fail("manifest pathname metadata"))
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("manifest pathname open"))
    with export_module._inventory_windows(tmp_path, file_access=True) as access:
        assert access.names == []
        assert access.read("manifest.json") == state.data
    assert len(state.open_paths) == 1
    assert state.id_opens == [(b"f" * 16, 0x80000000)]
    assert state.closed == [2, 1]


def test_windows_low64_id_capability_retry_keeps_full_identity_checks(tmp_path, export_module, monkeypatch):
    state = _mock_windows_kernel(monkeypatch, "ntfs_low64")
    with export_module._inventory_windows(tmp_path, file_access=True) as access:
        assert access.read("child/payload.json") == state.data
    assert state.id_kinds == [2, 0, 2, 0]
    assert state.closed == [3, 2, 1]


@pytest.mark.parametrize("mode", ["low64_denied", "low64_sharing", "low64_mismatch", "low64_volume"])
def test_windows_id_capability_retry_never_bypasses_denials_or_full_id_checks(tmp_path, export_module, monkeypatch, mode):
    state = _mock_windows_kernel(monkeypatch, mode)
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        with export_module._inventory_windows(tmp_path, file_access=True):
            pytest.fail("invalid child reached snapshot")
    assert state.id_kinds == ([2] if mode in ("low64_denied", "low64_sharing") else [2, 0])
    assert (2, 0x13) not in state.queries
    assert state.reads == []
    assert sorted(state.closed) == state.opened


@pytest.mark.skipif(os.name != "posix", reason="POSIX retained descriptor tests")
def test_writer_survives_child_symlink_substitution_without_external_writes(tmp_path, export_module):
    # Reproduces the published generation-time substitution; writing to a
    # held directory descriptor must not follow its replaced pathname.
    from qec.experiments.acoustic.export_fs import SecureExportWriter
    output, outside = tmp_path / "export", tmp_path / "outside"
    outside.mkdir()
    with SecureExportWriter(output, 4096, 4096) as writer:
        writer.mkdir("lab")
        (output / "lab").rename(output / "held-lab")
        (output / "lab").symlink_to(outside, target_is_directory=True)
        writer.save("lab/payload.json", b"safe artifact")
        assert list(outside.iterdir()) == []
        assert (output / "held-lab/payload.json").read_bytes() == b"safe artifact"
        writer.check_root()
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        export_module._inventory_files(output)


def test_simultaneous_writers_cannot_claim_the_same_empty_export(tmp_path):
    from qec.experiments.acoustic.export_fs import SecureExportWriter
    output = tmp_path / "export"
    with SecureExportWriter(output, 4096, 4096) as first:
        with pytest.raises((FileExistsError, ValueError, OSError)):
            with SecureExportWriter(output, 4096, 4096) as second:
                second.save("payload.json", b"competing")
        first.save("payload.json", b"first")
    assert (output / "payload.json").read_bytes() == b"first"
    assert not (output / ".qec-acoustic-export-claim").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX identity observation test")
def test_posix_inventory_retains_enumerated_file_identity(tmp_path, export_module):
    observed = tmp_path / "artifact.json"
    observed.write_bytes(b"same bytes")
    replacement = tmp_path / "replacement"
    with pytest.raises(ValueError, match="^INVALID_INPUT$"):
        with export_module._inventory_posix(tmp_path, file_access=True) as access:
            replacement.write_bytes(b"same bytes")
            replacement.replace(observed)
            with pytest.raises(ValueError, match="^INVALID_INPUT$"):
                access.stat("artifact.json")
            with pytest.raises(ValueError, match="^INVALID_INPUT$"):
                access.read("artifact.json")


def test_imported_code_source_mismatch_is_rejected(tmp_path, export_module, monkeypatch):
    # A module was imported from old source, then the tracked bytes changed
    # *before* generation's manifest snapshot.
    path = tmp_path / "loaded.py"
    module = ModuleType("qec.provenance_fixture")
    module.__file__ = str(path)
    before = "def build():\n    return 1\n"
    after = "def build():\n    return 2\n"
    path.write_text(before)
    exec(compile(before, str(path), "exec"), module.__dict__)
    path.write_text(after)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(export_module, "ROOT", tmp_path)
    monkeypatch.setattr(export_module, "_code_manifest",
                        lambda: [{"path": "loaded.py", "sha256": export_module._sha(path.read_bytes()),
                                  "bytes": path.stat().st_size}])
    with pytest.raises(ValueError, match="imported source changed"):
        export_module._loaded_code_matches_disk()


def test_windows_native_directory_record_layout_has_88_byte_header(export_module):
    import struct
    # The documented layout has two DWORDs, six 64-bit quantities, four
    # DWORDs, one 16-byte FILE_ID_128, then the filename (offset 88).
    assert struct.calcsize("<II6q4I16s") == 88
    record = struct.pack("<II6q4I16s", 0, 0, *([0] * 6), 0x20, 6, 0, 0, bytes(range(16)))
    parsed = export_module._parse_windows_directory_buffer(record + "abc".encode("utf-16-le"))
    assert parsed == [("abc", 0x20, 0, bytes(range(16)))]

