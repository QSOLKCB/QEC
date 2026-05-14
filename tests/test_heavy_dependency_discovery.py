from dataclasses import FrozenInstanceError, replace
import json

import pytest

from qec.analysis.heavy_dependency_discovery import (
    HeavyDependencyTarget,
    build_default_unprobed_manifest,
    build_heavy_dependency_discovery_manifest,
    build_probe_result,
    get_heavy_dependency_targets,
    probe_current_environment,
    validate_heavy_dependency_discovery_manifest,
    validate_heavy_dependency_probe_result,
    validate_heavy_dependency_target,
)


def _expected_names():
    return [
        "numpy", "scipy", "pandas", "matplotlib", "qutip", "qiskit",
        "qiskit_aer", "stim", "pymatching", "mido", "qldpc_internal", "qldpc_external",
    ]


def test_targets_fixed_registry_and_order_and_split_qldpc():
    names = [t.dependency_name for t in get_heavy_dependency_targets()]
    assert names == _expected_names()


def test_target_hash_and_json_stability_and_frozen():
    t1 = get_heavy_dependency_targets()[0]
    t2 = get_heavy_dependency_targets()[0]
    assert t1.target_hash == t2.target_hash
    json.dumps(t1.to_dict(), sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        t1.dependency_name = "x"


def test_target_builder_rejects_non_registry_field_combinations():
    from qec.analysis.heavy_dependency_discovery import build_heavy_dependency_target
    with pytest.raises(ValueError, match="REGISTRY_FIELD_MISMATCH"):
        build_heavy_dependency_target("numpy", "wrong_import", "NUMERIC_CORE", "AUTHORITATIVE_UPSTREAM_REQUIRED", False, ())
    with pytest.raises(ValueError, match="REGISTRY_FIELD_MISMATCH"):
        build_heavy_dependency_target("numpy", "numpy", "WRONG_CATEGORY", "AUTHORITATIVE_UPSTREAM_REQUIRED", False, ())
    with pytest.raises(ValueError, match="REGISTRY_FIELD_MISMATCH"):
        build_heavy_dependency_target("numpy", "numpy", "NUMERIC_CORE", "WRONG_POLICY", False, ())
    with pytest.raises(ValueError, match="REGISTRY_FIELD_MISMATCH"):
        build_heavy_dependency_target("numpy", "numpy", "NUMERIC_CORE", "AUTHORITATIVE_UPSTREAM_REQUIRED", True, ())



def test_probe_hash_deterministic_and_validation_rejections():
    p1 = build_probe_result("numpy", "AVAILABLE", version="1.0")
    p2 = build_probe_result("numpy", "AVAILABLE", version="1.0")
    assert p1.probe_hash == p2.probe_hash
    with pytest.raises(ValueError, match="INVALID_DEPENDENCY_NAME"):
        build_probe_result("unknown", "AVAILABLE")
    with pytest.raises(ValueError, match="INVALID_AVAILABILITY_STATUS"):
        build_probe_result("numpy", "BROKEN")
    with pytest.raises(ValueError, match="INVALID_PROBE_MODE"):
        build_probe_result("numpy", "AVAILABLE", probe_mode="BAD")
    with pytest.raises(ValueError, match="INVALID_SOURCE_POLICY"):
        build_probe_result("numpy", "AVAILABLE", policy_status="BAD")
    with pytest.raises(ValueError, match="POLICY_STATUS_CONFLICT"):
        build_probe_result("qldpc_external", "AVAILABLE")
    with pytest.raises(ValueError, match="POLICY_STATUS_CONFLICT"):
        build_probe_result("qldpc_internal", "AVAILABLE")


def test_default_manifest_deterministic_and_counts_and_hashes():
    m1 = build_default_unprobed_manifest()
    m2 = build_default_unprobed_manifest()
    assert m1.to_dict() == m2.to_dict()
    assert m1.target_count == 12
    assert m1.not_probed_count == 12
    assert m1.available_count == 0
    assert m1.unavailable_count == 0
    assert m1.blocked_by_policy_count == 0
    assert m1.internal_available_count == 0
    assert m1.first_target_hash == m1.targets[0].target_hash
    assert m1.final_target_hash == m1.targets[-1].target_hash
    assert m1.first_probe_hash == m1.probe_results[0].probe_hash
    assert m1.final_probe_hash == m1.probe_results[-1].probe_hash
    assert m1.to_canonical_json() == m2.to_canonical_json()
    assert m1.to_canonical_bytes() == m2.to_canonical_bytes()


def test_manifest_build_validation_and_ordering_rules():
    probes = [build_probe_result(name, "NOT_PROBED") for name in _expected_names()]
    probes[0] = build_probe_result("numpy", "AVAILABLE", version="1")
    probes[1] = build_probe_result("scipy", "UNAVAILABLE")
    probes[10] = build_probe_result("qldpc_internal", "INTERNAL_AVAILABLE", probe_mode="INTERNAL_MODULE")
    probes[11] = build_probe_result("qldpc_external", "BLOCKED_BY_POLICY")
    manifest = build_heavy_dependency_discovery_manifest(probes)
    assert manifest.available_count == 1
    assert manifest.unavailable_count == 1
    assert manifest.internal_available_count == 1
    assert manifest.blocked_by_policy_count == 1
    assert manifest.not_probed_count == 8
    assert validate_heavy_dependency_discovery_manifest(manifest)

    with pytest.raises(ValueError, match="DUPLICATE_DEPENDENCY"):
        build_heavy_dependency_discovery_manifest(list(probes[:-1]) + [probes[-2]])
    with pytest.raises(ValueError, match="DISCOVERY_COUNT_MISMATCH"):
        build_heavy_dependency_discovery_manifest(probes[:-1])
    with pytest.raises(ValueError, match="DEPENDENCY_ORDER_MISMATCH"):
        bad = list(probes)
        bad[0], bad[1] = bad[1], bad[0]
        build_heavy_dependency_discovery_manifest(bad)


def test_hash_validator_errors_and_bool_rejection_in_manifest_counts():
    t = get_heavy_dependency_targets()[0]
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_heavy_dependency_target(replace(t, target_hash="abc"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_heavy_dependency_target(replace(t, target_hash="0" * 64))

    p = build_probe_result("numpy", "AVAILABLE")
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_heavy_dependency_probe_result(replace(p, probe_hash="abc"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_heavy_dependency_probe_result(replace(p, probe_hash="0" * 64))

    m = build_default_unprobed_manifest()
    with pytest.raises(ValueError, match="INVALID_HASH_FORMAT"):
        validate_heavy_dependency_discovery_manifest(replace(m, heavy_dependency_discovery_manifest_hash="abc"))
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_heavy_dependency_discovery_manifest(replace(m, heavy_dependency_discovery_manifest_hash="0" * 64))
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        validate_heavy_dependency_discovery_manifest(replace(m, target_count=True))


def test_probe_current_environment_monkeypatched_and_no_direct_imports(monkeypatch):
    calls = []

    def fake_find_spec(name):
        calls.append(("find_spec", name))
        return object() if name in {"numpy"} else None

    def fake_version(name):
        calls.append(("version", name))
        if name == "numpy":
            return "1.2.3"
        raise Exception("notfound")

    from qec.analysis import heavy_dependency_discovery as mod

    monkeypatch.setattr(mod.importlib_util, "find_spec", fake_find_spec)

    class _Meta:
        class PackageNotFoundError(Exception):
            pass

        @staticmethod
        def version(name):
            calls.append(("version", name))
            if name == "numpy":
                return "1.2.3"
            raise _Meta.PackageNotFoundError()

    monkeypatch.setattr(mod, "importlib_metadata", _Meta)
    manifest = probe_current_environment()
    by_name = {p.dependency_name: p for p in manifest.probe_results}
    assert by_name["qldpc_external"].availability_status == "BLOCKED_BY_POLICY"
    assert by_name["qldpc_internal"].availability_status == "INTERNAL_AVAILABLE"
    assert by_name["numpy"].availability_status == "AVAILABLE"
    assert by_name["numpy"].version == "1.2.3"
    assert by_name["numpy"].version_source == "importlib.metadata.version"
    assert ("find_spec", "numpy") in calls
    assert ("version", "numpy") in calls


def test_source_scan_forbidden_tokens_absent():
    txt = open("src/qec/analysis/heavy_dependency_discovery.py", "r", encoding="utf-8").read()
    forbidden = [
        "import qutip", "import qiskit", "import matplotlib", "import pandas", "import stim",
        "import pymatching", "import mido", "import requests", "urllib.request", "subprocess",
        "os.system", "shell=True", "eval(", "exec(", "__import__(", "importlib.import_module",
        "pip", "time.time", "datetime.now", "random.",
    ]
    for token in forbidden:
        assert token not in txt
