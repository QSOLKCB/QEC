# SPDX-License-Identifier: MPL-2.0
"""Export bounded replayable comparisons, actual WAV/JSON, and a native byte demo.

The canonical battery excludes runtime, memory, git state and dependency versions.
Those are separately measured/recorded in environment.json and a file manifest.
No installation, hardware testing, SPECTRAL UI integration or publishing occurs.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import time
import tracemalloc

from qec.experiments.acoustic.channel import PCM, read_wav, wav_bytes
from qec.experiments.acoustic.evidence import (
    CASE_NAMES, build_battery_source, build_demonstration, build_trial_source,
    run_comparative_battery, validate_comparative_battery, validate_demonstration,
)
from qec.experiments.acoustic.transport import (
    ARTIFACT_ROLES, TRANSPORTS, AcousticTransportArtifact,
    execute_acoustic_transport, validate_acoustic_transport_receipt,
)
from qec.sonify.canonical import canonical_bytes, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
MAX_EXPORT_FILE_BYTES = 32 * 1024 * 1024
MAX_EXPORT_TOTAL_BYTES = 32 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def _portable_relative(path, root):
    """Canonical manifest paths always use forward slashes, including Windows."""
    return path.relative_to(root).as_posix()


def _code_manifest():
    # Hash actual source bytes, including imported decoder and native transport
    # dependency trees. This is deliberately broader than just entrypoints.
    files = [ROOT / "scripts/generate_acoustic_trial_evidence.py"]
    files.extend(ROOT / name for name in ("src/qec/__init__.py", "src/qec/experiments/__init__.py",
                                          "src/qec/routing/__init__.py"))
    for relative in ("src/qec/experiments/acoustic", "src/qec/decoder", "src/qec/sonify",
                     "src/qec/routing/digital", "src/qec/routing/packet"):
        files.extend(sorted((ROOT / relative).rglob("*.py")))
    for name in ("pyproject.toml", "requirements.txt", "requirements-test.txt", "dev-requirements.txt"):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    return [{"path": _portable_relative(path, ROOT), "sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}
            for path in sorted(set(files))]


def _write_json(path, value):
    path.write_bytes(canonical_bytes(value) + b"\n")


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _regular_stat(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 <= info.st_size <= MAX_EXPORT_FILE_BYTES:
        raise ValueError("INVALID_INPUT")
    return info


def _read_regular_bytes(path, *, expected_stat=None):
    """Read one bounded regular-file snapshot without blocking on a FIFO race.

    POSIX nonblocking/no-follow flags are used when available. Every platform
    checks both lstat and descriptor fstat before reading; identity, size and
    timestamps must remain stable through the bounded read. No special file or
    symlink is accepted. The extra byte detects growth beyond the declared size.
    """
    descriptor = None
    try:
        before = _regular_stat(path)
        if expected_stat is not None and _file_identity(before) != _file_identity(expected_stat):
            raise ValueError("INVALID_INPUT")
        flags = os.O_RDONLY
        for name in ("O_NONBLOCK", "O_NOFOLLOW", "O_CLOEXEC", "O_BINARY", "O_NOINHERIT"):
            flags |= getattr(os, name, 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
            raise ValueError("INVALID_INPUT")
        chunks, count = [], 0
        while count <= opened.st_size:
            chunk = os.read(descriptor, min(65536, opened.st_size + 1 - count))
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
        if count != opened.st_size or _file_identity(os.fstat(descriptor)) != _file_identity(opened):
            raise ValueError("INVALID_INPUT")
        return b"".join(chunks)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_INPUT") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _parse_json(data):
    try:
        value = json.loads(data)
        if type(value) is not dict or canonical_bytes(value) + b"\n" != data:
            raise ValueError("INVALID_INPUT")
        return value
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def _read_json(path):
    return _parse_json(_read_regular_bytes(path))


def _inventory_files(output):
    """Bound discovery and reject every special entry; exempt only root manifest."""
    names, pending, directories = [], [(output, 0)], 0
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    directories += 1
                    if directories > 16 or depth >= 4:
                        raise ValueError("INVALID_INPUT")
                    pending.append((path, depth + 1))
                elif not stat.S_ISREG(info.st_mode):
                    raise ValueError("INVALID_INPUT")
                elif path != output / "manifest.json":
                    names.append(_portable_relative(path, output))
                    if len(names) > 128:
                        raise ValueError("INVALID_INPUT")
    return sorted(names)


_MANIFEST_FIXED = {
    "schema": "qec.acoustic-export-manifest.v1", "contract_version": "177.5",
    "transport_scope": "six_actual_role_payloads_in_separate_demo; no_per_battery_transport_claim",
    "spectral_compatibility": "ordinary_mono_PCM16_WAV_and_canonical_JSON_export_only",
    "spectral_repository_or_ui_integration": False,
    "environment_metadata_is_canonical_scientific_proof": False,
}
_MANIFEST_DYNAMIC = ("sha256", "canonical_battery_sha256", "battery_source_sha256",
    "demonstration_source_sha256", "measured_reference_sha256", "source_commit_sha", "source_files",
    "clean_working_tree_at_start", "dependencies", "files", "transport_demonstration",
    "artifact_total_bytes_excluding_manifest", "digital_binding_specimen")
_ENVIRONMENT_FIXED = {
    "schema": "qec.acoustic-environment.v1",
    "measurement_scope": "battery_build_and_replay_plus_export_and_native_demo; not_process_RSS",
    "performance_is_environment_dependent": True, "performance_is_canonical_proof": False,
    "physical_latency_or_hardware_QA": False, "full_suite_gate": "separate_exact_commit_test_run",
}
_ENVIRONMENT_DYNAMIC = ("source_commit_sha", "clean_working_tree_at_start", "git_status_porcelain_at_start",
    "source_files", "python", "platform", "dependencies", "runtime_ns", "peak_python_tracemalloc_bytes")


def _strict_metadata(value, fixed, dynamic):
    from qec.experiments.acoustic._common import exact
    # Reconstruct the exact permitted schema; external hash pins never authorize
    # altered scientific claim flags or unknown fields.
    exact(value, {**fixed, **{key: value[key] for key in dynamic}})


def _validate_environment(value):
    _strict_metadata(value, _ENVIRONMENT_FIXED, _ENVIRONMENT_DYNAMIC)
    if (type(value["source_commit_sha"]) is not str or
            re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value["source_commit_sha"]) is None):
        raise ValueError("INVALID_INPUT")
    for key, limit in (("python", 128), ("platform", 2048), ("git_status_porcelain_at_start", 1_000_000)):
        if type(value[key]) is not str or len(value[key]) > limit:
            raise ValueError("INVALID_INPUT")
    if (type(value["clean_working_tree_at_start"]) is not bool or
            value["clean_working_tree_at_start"] != (not bool(value["git_status_porcelain_at_start"]))):
        raise ValueError("INVALID_INPUT")
    for key in ("runtime_ns", "peak_python_tracemalloc_bytes"):
        if type(value[key]) is not int or not 0 <= value[key] < 2**63:
            raise ValueError("INVALID_INPUT")
    dependencies = value["dependencies"]
    if (type(dependencies) is not dict or set(dependencies) != {"qec", "numpy", "scipy", "pytest"}
            or any(type(v) is not str or not 1 <= len(v) <= 128 for v in dependencies.values())):
        raise ValueError("INVALID_INPUT")
    if value["source_files"] != _code_manifest():
        raise ValueError("INVALID_INPUT")


def generate(output: Path, *, cases=CASE_NAMES, alphabets=(3, 4), seed=1775, transports=("packet",)):
    """Create an empty-directory export; return canonical identity and source pins."""
    if type(output) is not Path:
        output = Path(output)
    if (not transports or type(transports) not in (tuple, list) or len(transports) > 2
            or any(type(t) is not str or t not in TRANSPORTS for t in transports)
            or len(set(transports)) != len(transports)):
        raise ValueError("INVALID_INPUT")
    source = build_battery_source(cases=cases, alphabets=alphabets, seed=seed)
    demo_source = build_trial_source("broadband", alphabet_size=3, seed=seed)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be an empty directory")
    commit, dirty, code_files = _git("rev-parse", "HEAD"), _git("status", "--porcelain"), _code_manifest()
    output.mkdir(parents=True, exist_ok=True)
    def save(path, value):
        data = canonical_bytes(value) + b"\n"
        written = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if len(data) > MAX_EXPORT_FILE_BYTES or written + len(data) > MAX_EXPORT_TOTAL_BYTES:
            raise ValueError("export exceeds 32 MiB bound; choose fewer full-stage transports")
        path.write_bytes(data)
    tracemalloc.start()
    start = time.perf_counter_ns()
    try:
        battery = run_comparative_battery(source)
        validation = validate_comparative_battery(battery, expected_source_sha256=source["sha256"])
        demo = build_demonstration(demo_source)
        demo_validation = validate_demonstration(demo, expected_source_sha256=demo_source["sha256"],
                                                expected_reference_sha256=demo_source["measured_reference"]["sha256"])
        save(output / "battery.json", battery)
        save(output / "battery_source.json", source)
        save(output / "battery_validation.json", validation)
        save(output / "demonstration.json", demo)
        save(output / "demonstration_source.json", demo_source)
        save(output / "demonstration_validation.json", demo_validation)
        (output / "trials").mkdir()
        for index, trial in enumerate(battery["trials"]):
            stem = f"{index:02d}_q{trial['source']['alphabet_size']}_{trial['source']['case_name']}"
            save(output / "trials" / (stem + ".json"), trial)
            save(output / "trials" / (stem + "_source.json"), trial["source"])
        (output / "lab").mkdir()
        (output / "transport").mkdir()
        transport_index = []
        for role in ARTIFACT_ROLES:
            value = demo["roles"][role]
            save(output / "lab" / (role + ".json"), value)
            if role in ("original", "observed", "filtered"):
                (output / "lab" / (role + ".wav")).write_bytes(wav_bytes(PCM.from_dict(value)))
            artifact = AcousticTransportArtifact.from_artifact(value, role=role, source_sha256=demo_source["sha256"])
            for transport in transports:
                receipt = execute_acoustic_transport(artifact, transport=transport)
                result = validate_acoustic_transport_receipt(
                    receipt, expected_artifact_sha256=artifact.as_dict()["sha256"],
                    expected_source_sha256=demo_source["sha256"], expected_input_sha256=receipt["input"]["sha256"])
                name = f"transport/{role}_{transport}.json"
                save(output / name, receipt)
                transport_index.append({"role": role, "transport": transport, "path": name,
                                        "artifact_sha256": artifact.as_dict()["sha256"],
                                        "input_sha256": receipt["input"]["sha256"],
                                        "receipt_sha256": receipt["sha256"], "native_replay": result["native_replay"],
                                        "payload_bytes": len(artifact.payload)})
        # A short Digital binding record is intentionally distinct from the six
        # full Packet stage payloads. It does not claim to transport PCM.
        digital_specimen = {"schema": "qec.acoustic-digital-binding-specimen.v1",
                            "source_sha256": demo_source["sha256"],
                            "candidate_observation_sha256": demo["roles"]["candidate"]["sha256"],
                            "scope": "binding_record_only; not_full_observation_or_PCM"}
        digital_artifact = AcousticTransportArtifact.from_artifact(
            digital_specimen, role="candidate", source_sha256=demo_source["sha256"])
        digital_receipt = execute_acoustic_transport(digital_artifact, transport="digital")
        validate_acoustic_transport_receipt(digital_receipt,
            expected_artifact_sha256=digital_artifact.as_dict()["sha256"], expected_source_sha256=demo_source["sha256"],
            expected_input_sha256=digital_receipt["input"]["sha256"])
        save(output / "lab/digital_binding_specimen.json", digital_specimen)
        save(output / "transport/digital_binding_specimen.json", digital_receipt)
        elapsed_ns = time.perf_counter_ns() - start
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if _git("rev-parse", "HEAD") != commit or _code_manifest() != code_files:
        raise ValueError("source changed during export; rerun against a stable checkout")
    environment = {"schema": "qec.acoustic-environment.v1", "source_commit_sha": commit,
                   "clean_working_tree_at_start": not bool(dirty), "git_status_porcelain_at_start": dirty,
                   "source_files": code_files, "python": platform.python_version(), "platform": platform.platform(),
                   "dependencies": {name: _version(name) for name in ("qec", "numpy", "scipy", "pytest")},
                   "runtime_ns": elapsed_ns, "peak_python_tracemalloc_bytes": peak_bytes,
                   "measurement_scope": "battery_build_and_replay_plus_export_and_native_demo; not_process_RSS",
                   "performance_is_environment_dependent": True, "performance_is_canonical_proof": False,
                   "physical_latency_or_hardware_QA": False, "full_suite_gate": "separate_exact_commit_test_run"}
    save(output / "environment.json", environment)
    files = [{"path": _portable_relative(p, output), "sha256": _sha(p.read_bytes()), "bytes": p.stat().st_size}
             for p in sorted(output.rglob("*")) if p.is_file()]
    manifest = {"schema": "qec.acoustic-export-manifest.v1", "contract_version": "177.5",
                "canonical_battery_sha256": battery["sha256"], "battery_source_sha256": source["sha256"],
                "demonstration_source_sha256": demo_source["sha256"],
                "measured_reference_sha256": demo_source["measured_reference"]["sha256"],
                "source_commit_sha": commit, "source_files": code_files,
                "clean_working_tree_at_start": not bool(dirty), "dependencies": environment["dependencies"],
                "files": files, "transport_demonstration": transport_index,
                "artifact_total_bytes_excluding_manifest": sum(f["bytes"] for f in files),
                "digital_binding_specimen": {"path": "transport/digital_binding_specimen.json",
                    "input_sha256": digital_receipt["input"]["sha256"],
                    "artifact_sha256": digital_artifact.as_dict()["sha256"], "payload_bytes": len(digital_artifact.payload),
                    "scope": "binding_record_only; not_one_of_six_full_stage_payloads"},
                "transport_scope": "six_actual_role_payloads_in_separate_demo; no_per_battery_transport_claim",
                "spectral_compatibility": "ordinary_mono_PCM16_WAV_and_canonical_JSON_export_only",
                "spectral_repository_or_ui_integration": False, "environment_metadata_is_canonical_scientific_proof": False}
    manifest["sha256"] = canonical_sha256(manifest)
    save(output / "manifest.json", manifest)
    return {"battery_sha256": battery["sha256"], "battery_source_sha256": source["sha256"],
            "demonstration_source_sha256": demo_source["sha256"],
            "measured_reference_sha256": demo_source["measured_reference"]["sha256"],
            "manifest_sha256": manifest["sha256"], "trial_count": len(battery["trials"])}


def validate_export(output: Path, *, expected_manifest_sha256: str, expected_battery_source_sha256: str,
                    expected_demonstration_source_sha256: str, expected_reference_sha256: str):
    """Replay disk evidence against externally retained source AND manifest pins."""
    from qec.experiments.acoustic._common import digest, exact
    try:
        output = Path(output)
        for pin in (expected_manifest_sha256, expected_battery_source_sha256,
                    expected_demonstration_source_sha256, expected_reference_sha256):
            digest(pin)
        manifest = _read_json(output / "manifest.json")
        _strict_metadata(manifest, _MANIFEST_FIXED, _MANIFEST_DYNAMIC)
        unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
        if (manifest["sha256"] != expected_manifest_sha256 or canonical_sha256(unsigned) != expected_manifest_sha256
                or manifest["battery_source_sha256"] != expected_battery_source_sha256
                or manifest["demonstration_source_sha256"] != expected_demonstration_source_sha256
                or manifest["measured_reference_sha256"] != expected_reference_sha256):
            raise ValueError("INVALID_INPUT")
        if type(manifest["files"]) is not list or len(manifest["files"]) > 128:
            raise ValueError("INVALID_INPUT")
        inventory, total_bytes = [], 0
        # Stat the complete bounded inventory before reading any artifact bytes.
        # A hostile manifest cannot turn the 32 MiB total into 128 x 32 MiB reads.
        for row in manifest["files"]:
            if type(row["path"]) is not str or "\\" in row["path"]:
                raise ValueError("INVALID_INPUT")
            relative = Path(row["path"])
            if (relative.is_absolute() or ".." in relative.parts or (output / relative).is_symlink()
                    or not (output / relative).resolve().is_relative_to(output.resolve())):
                raise ValueError("INVALID_INPUT")
            path = output / relative
            info = _regular_stat(path)
            size = info.st_size
            total_bytes += size
            if size > MAX_EXPORT_FILE_BYTES or total_bytes > MAX_EXPORT_TOTAL_BYTES:
                raise ValueError("INVALID_INPUT")
            inventory.append((row["path"], path, info))
        if _inventory_files(output) != sorted(name for name, _, _ in inventory):
            raise ValueError("INVALID_INPUT")
        contents = {name: _read_regular_bytes(path, expected_stat=info) for name, path, info in inventory}
        expected_files = [{"path": name, "sha256": _sha(contents[name]), "bytes": len(contents[name])}
                          for name, _, _ in inventory]
        exact({"files": manifest["files"]}, {"files": expected_files})
        def read_json(path):
            return _parse_json(contents[_portable_relative(path, output)])
        battery = read_json(output / "battery.json")
        exact(read_json(output / "battery_source.json"), battery["source"])
        validation = validate_comparative_battery(battery, expected_source_sha256=expected_battery_source_sha256)
        exact(read_json(output / "battery_validation.json"), validation)
        if manifest["canonical_battery_sha256"] != battery["sha256"]:
            raise ValueError("INVALID_INPUT")
        for index, trial in enumerate(battery["trials"]):
            stem = f"{index:02d}_q{trial['source']['alphabet_size']}_{trial['source']['case_name']}"
            exact(read_json(output / "trials" / (stem + ".json")), trial)
            exact(read_json(output / "trials" / (stem + "_source.json")), trial["source"])
        demo = read_json(output / "demonstration.json")
        exact(read_json(output / "demonstration_source.json"), demo["source"])
        demo_validation = validate_demonstration(demo, expected_source_sha256=expected_demonstration_source_sha256,
                                                expected_reference_sha256=expected_reference_sha256)
        exact(read_json(output / "demonstration_validation.json"), demo_validation)
        for role in ARTIFACT_ROLES:
            value = read_json(output / "lab" / (role + ".json"))
            exact(value, demo["roles"][role])
            if role in ("original", "observed", "filtered"):
                exact(read_wav(contents["lab/" + role + ".wav"]).as_dict(), value)
        if type(manifest["transport_demonstration"]) is not list or not 6 <= len(manifest["transport_demonstration"]) <= 12:
            raise ValueError("INVALID_INPUT")
        seen = set()
        for row in manifest["transport_demonstration"]:
            role, transport = row["role"], row["transport"]
            if role not in ARTIFACT_ROLES or transport not in TRANSPORTS or (role, transport) in seen:
                raise ValueError("INVALID_INPUT")
            seen.add((role, transport))
            if row["path"] != f"transport/{role}_{transport}.json":
                raise ValueError("INVALID_INPUT")
            artifact = AcousticTransportArtifact.from_artifact(demo["roles"][role], role=role,
                                                               source_sha256=expected_demonstration_source_sha256)
            receipt = read_json(output / row["path"])
            result = validate_acoustic_transport_receipt(receipt,
                expected_artifact_sha256=artifact.as_dict()["sha256"], expected_source_sha256=expected_demonstration_source_sha256,
                expected_input_sha256=row["input_sha256"])
            exact(row, {"role": role, "transport": transport, "path": f"transport/{role}_{transport}.json",
                        "artifact_sha256": artifact.as_dict()["sha256"], "input_sha256": receipt["input"]["sha256"],
                        "receipt_sha256": receipt["sha256"], "native_replay": "FULL", "payload_bytes": len(artifact.payload)})
            if (not result["accepted"] or receipt["input"]["transport"] != transport or
                    receipt["sha256"] != row["receipt_sha256"] or artifact.as_dict()["sha256"] != row["artifact_sha256"]
                    or len(artifact.payload) != row["payload_bytes"] or row["native_replay"] != "FULL"):
                raise ValueError("INVALID_INPUT")
        if {r for r, _ in seen} != set(ARTIFACT_ROLES):
            raise ValueError("INVALID_INPUT")
        expected_specimen = {"schema": "qec.acoustic-digital-binding-specimen.v1",
                             "source_sha256": expected_demonstration_source_sha256,
                             "candidate_observation_sha256": demo["roles"]["candidate"]["sha256"],
                             "scope": "binding_record_only; not_full_observation_or_PCM"}
        exact(read_json(output / "lab/digital_binding_specimen.json"), expected_specimen)
        artifact = AcousticTransportArtifact.from_artifact(expected_specimen, role="candidate",
                                                           source_sha256=expected_demonstration_source_sha256)
        entry = manifest["digital_binding_specimen"]
        if (entry["path"] != "transport/digital_binding_specimen.json" or
                entry["artifact_sha256"] != artifact.as_dict()["sha256"] or entry["payload_bytes"] != len(artifact.payload)):
            raise ValueError("INVALID_INPUT")
        digital_receipt = read_json(output / entry["path"])
        result = validate_acoustic_transport_receipt(digital_receipt,
            expected_artifact_sha256=artifact.as_dict()["sha256"], expected_source_sha256=expected_demonstration_source_sha256,
            expected_input_sha256=entry["input_sha256"])
        exact(entry, {"path": "transport/digital_binding_specimen.json",
                      "input_sha256": digital_receipt["input"]["sha256"],
                      "artifact_sha256": artifact.as_dict()["sha256"], "payload_bytes": len(artifact.payload),
                      "scope": "binding_record_only; not_one_of_six_full_stage_payloads"})
        if not result["accepted"] or digital_receipt["input"]["transport"] != "digital":
            raise ValueError("INVALID_INPUT")
        if manifest["artifact_total_bytes_excluding_manifest"] != sum(f["bytes"] for f in expected_files):
            raise ValueError("INVALID_INPUT")
        environment = read_json(output / "environment.json")
        _validate_environment(environment)
        for key in ("source_commit_sha", "source_files", "clean_working_tree_at_start", "dependencies"):
            exact({key: manifest[key]}, {key: environment[key]})
        if environment["source_files"] != _code_manifest():
            raise ValueError("INVALID_INPUT")
        return {"all_passed": True, "full_comparative_replay": True, "full_native_replay": True,
                "manifest_sha256": expected_manifest_sha256, "physical_provenance_authenticated": False}
    except (OSError, KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
        raise ValueError("INVALID_INPUT") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1775)
    parser.add_argument("--case", action="append", choices=CASE_NAMES)
    parser.add_argument("--alphabet", action="append", type=int, choices=(3, 4))
    parser.add_argument("--transport", action="append", choices=TRANSPORTS)
    args = parser.parse_args(argv)
    result = generate(args.output, cases=tuple(args.case or CASE_NAMES), alphabets=tuple(args.alphabet or (3, 4)),
                      seed=args.seed, transports=tuple(args.transport or ("packet",)))
    print(canonical_bytes(result).decode("utf-8"))


if __name__ == "__main__":
    main()
