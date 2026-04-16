# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.1.1 experiment packaging format."""

from __future__ import annotations

import json

import pytest

from qec.simulation.experiment_packaging_format import (
    EXPERIMENT_PACKAGING_FORMAT_VERSION,
    ExperimentPackageManifest,
    ExperimentPackageValidationError,
    build_experiment_package,
    package_replay_identity,
)


def _h(ch: str) -> str:
    return ch * 64


def _manifest_payload() -> dict:
    return {
        "format_version": EXPERIMENT_PACKAGING_FORMAT_VERSION,
        "package_kind": "correlated_noise_simulation",
        "experiment_id": "exp-v138-001",
        "simulator_release": "v138.1.0",
        "simulator_module": "qec.simulation.correlated_noise_simulator",
        "scenario_hash": _h("a"),
        "realization_hashes": (_h("b"), _h("c")),
        "topology_family": "grid",
        "code_family": "surface_code",
        "seed": 42,
        "parameter_hash": _h("d"),
        "policy_flags": ("deterministic", "replay_safe"),
        "benchmark_id": "bench-001",
        "manifest_lineage_hash": _h("e"),
        "notes": ("release-v138.1.x", "packaging"),
    }


def _artifact_payloads() -> tuple[dict, ...]:
    return (
        {
            "artifact_role": "simulation_manifest",
            "artifact_hash": _h("1"),
            "artifact_kind": "json_document",
            "serialization_format": "json",
            "content_bytes": 120,
            "lineage_hash": _h("f"),
            "metadata": {"order": 1, "nested": {"k2": 2, "k1": 1}},
        },
        {
            "artifact_role": "simulation_receipt",
            "artifact_hash": _h("2"),
            "artifact_kind": "json_document",
            "serialization_format": "json",
            "content_bytes": 80,
            "lineage_hash": _h("e"),
            "metadata": {"order": 2, "notes": ["ok", True]},
        },
    )


def test_deterministic_package_equality_for_dict_ordering() -> None:
    manifest_a = _manifest_payload()
    manifest_b = {
        "experiment_id": manifest_a["experiment_id"],
        "package_kind": manifest_a["package_kind"],
        "simulator_release": manifest_a["simulator_release"],
        "simulator_module": manifest_a["simulator_module"],
        "scenario_hash": manifest_a["scenario_hash"],
        "realization_hashes": tuple(reversed(manifest_a["realization_hashes"])),
        "topology_family": manifest_a["topology_family"],
        "code_family": manifest_a["code_family"],
        "seed": manifest_a["seed"],
        "parameter_hash": manifest_a["parameter_hash"],
        "policy_flags": tuple(reversed(manifest_a["policy_flags"])),
        "notes": tuple(reversed(manifest_a["notes"])),
        "benchmark_id": manifest_a["benchmark_id"],
        "manifest_lineage_hash": manifest_a["manifest_lineage_hash"],
        "format_version": manifest_a["format_version"],
    }
    artifacts = _artifact_payloads()

    package_a = build_experiment_package(manifest=manifest_a, artifacts=artifacts, upstream_receipt_hashes=[_h("9"), _h("8")])
    package_b = build_experiment_package(manifest=manifest_b, artifacts=artifacts, upstream_receipt_hashes=[_h("8"), _h("9")])

    assert package_a.to_canonical_json() == package_b.to_canonical_json()
    assert package_a.stable_hash() == package_b.stable_hash()


def test_artifact_ordering_stability() -> None:
    artifacts = _artifact_payloads()
    package_a = build_experiment_package(manifest=_manifest_payload(), artifacts=artifacts)
    package_b = build_experiment_package(manifest=_manifest_payload(), artifacts=tuple(reversed(artifacts)))

    assert package_a.artifacts == package_b.artifacts
    assert package_a.receipt.package_hash == package_b.receipt.package_hash


def test_manifest_dataclass_input_is_canonicalized_before_hashing() -> None:
    manifest_dict = _manifest_payload()
    manifest_dataclass = ExperimentPackageManifest(
        format_version=manifest_dict["format_version"],
        package_kind=manifest_dict["package_kind"],
        experiment_id=manifest_dict["experiment_id"],
        simulator_release=manifest_dict["simulator_release"],
        simulator_module=manifest_dict["simulator_module"],
        scenario_hash=manifest_dict["scenario_hash"],
        realization_hashes=tuple(reversed(manifest_dict["realization_hashes"])),
        topology_family=manifest_dict["topology_family"],
        code_family=manifest_dict["code_family"],
        seed=manifest_dict["seed"],
        parameter_hash=manifest_dict["parameter_hash"],
        policy_flags=tuple(reversed(manifest_dict["policy_flags"])),
        benchmark_id=manifest_dict["benchmark_id"],
        manifest_lineage_hash=manifest_dict["manifest_lineage_hash"],
        notes=tuple(reversed(manifest_dict["notes"])),
    )

    package_from_dict = build_experiment_package(manifest=manifest_dict, artifacts=_artifact_payloads())
    package_from_dataclass = build_experiment_package(manifest=manifest_dataclass, artifacts=_artifact_payloads())

    assert package_from_dataclass.manifest.realization_hashes == tuple(sorted(manifest_dict["realization_hashes"]))
    assert package_from_dataclass.manifest.policy_flags == tuple(sorted(manifest_dict["policy_flags"]))
    assert package_from_dataclass.manifest.notes == tuple(sorted(manifest_dict["notes"]))
    assert package_from_dict.receipt.manifest_hash == package_from_dataclass.receipt.manifest_hash
    assert package_from_dict.receipt.package_hash == package_from_dataclass.receipt.package_hash


def test_validation_failure_on_malformed_manifest() -> None:
    bad_manifest = _manifest_payload()
    bad_manifest["experiment_id"] = " "
    with pytest.raises(ExperimentPackageValidationError, match="manifest.experiment_id must be non-empty"):
        build_experiment_package(manifest=bad_manifest, artifacts=_artifact_payloads())


def test_validation_failure_on_malformed_artifact_hash() -> None:
    bad_artifact = dict(_artifact_payloads()[0])
    bad_artifact["artifact_hash"] = ""
    with pytest.raises(ExperimentPackageValidationError, match="artifact.artifact_hash"):
        build_experiment_package(manifest=_manifest_payload(), artifacts=(bad_artifact,))


def test_nan_and_inf_rejection() -> None:
    nan_artifact = dict(_artifact_payloads()[0])
    nan_artifact["metadata"] = {"bad": float("nan")}
    with pytest.raises(ExperimentPackageValidationError, match="non-canonical numeric value"):
        build_experiment_package(manifest=_manifest_payload(), artifacts=(nan_artifact,))

    inf_artifact = dict(_artifact_payloads()[0])
    inf_artifact["metadata"] = {"bad": float("inf")}
    with pytest.raises(ExperimentPackageValidationError, match="non-canonical numeric value"):
        build_experiment_package(manifest=_manifest_payload(), artifacts=(inf_artifact,))


def test_package_hash_sensitivity() -> None:
    package_a = build_experiment_package(manifest=_manifest_payload(), artifacts=_artifact_payloads())

    changed_manifest = _manifest_payload()
    changed_manifest["experiment_id"] = "exp-v138-002"
    package_b = build_experiment_package(manifest=changed_manifest, artifacts=_artifact_payloads())

    changed_artifact = dict(_artifact_payloads()[0])
    changed_artifact["artifact_hash"] = _h("3")
    package_c = build_experiment_package(manifest=_manifest_payload(), artifacts=(changed_artifact, _artifact_payloads()[1]))

    assert package_a.receipt.package_hash != package_b.receipt.package_hash
    assert package_a.receipt.package_hash != package_c.receipt.package_hash


def test_upstream_lineage_preservation() -> None:
    upstream = (_h("4"), _h("5"), _h("6"))
    package = build_experiment_package(manifest=_manifest_payload(), artifacts=_artifact_payloads(), upstream_receipt_hashes=upstream)

    replay = package_replay_identity(package)
    assert package.manifest.realization_hashes == tuple(sorted(_manifest_payload()["realization_hashes"]))
    assert package.receipt.upstream_receipt_hashes == tuple(sorted(upstream))
    assert replay["upstream_receipt_hashes"] == tuple(sorted(upstream))


def test_canonical_json_round_trip_stability() -> None:
    package = build_experiment_package(manifest=_manifest_payload(), artifacts=_artifact_payloads())
    json_a = package.to_canonical_json()
    parsed = json.loads(json_a)
    json_b = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert json_a == json_b
