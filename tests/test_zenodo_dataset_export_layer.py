# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.16 Zenodo dataset export layer."""

from __future__ import annotations

import json

import pytest

from qec.runtime.multi_model_invocation_matrix import build_multi_model_invocation_matrix
from qec.runtime.persona_drift_tensor import build_persona_drift_tensor
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact
from qec.runtime.stable_evaluation_receipt_pack import build_stable_evaluation_receipt_pack
from qec.runtime.technical_rigor_metric_pack import build_technical_rigor_metric_pack
from qec.runtime.wrapper_divergence_study import build_wrapper_divergence_study
from qec.runtime.zenodo_dataset_export_layer import (
    ZenodoDatasetExportValidationError,
    build_reproducibility_metadata,
    build_zenodo_dataset_export_bundle,
    validate_zenodo_dataset_export_bundle,
    zenodo_export_projection,
)


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "zenodo-001",
            "prompt_text": "Provide a deterministic comparative answer.",
            "system_prompt": "Remain deterministic and strict.",
            "wrapper_metadata": {"wrapper": "none"},
            "model_name": "chatgpt_native",
            "invocation_route": "comparative",
            "repetition_count": 1,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "zenodo_dataset_export_layer"},
        }
    )


def _invocation_matrix(artifact):
    prompt_hash = artifact.receipt.prompt_hash
    return build_multi_model_invocation_matrix(
        artifact,
        [
            {
                "invocation_id": "inv-direct",
                "model_name": "chatgpt_native",
                "provider_name": "openai",
                "route_name": "chatgpt_direct",
                "prompt_hash": prompt_hash,
                "repetition_index": 0,
                "execution_mode": "planned",
                "metadata": {},
            },
            {
                "invocation_id": "inv-wrapped",
                "model_name": "chatgpt_5_4_sider",
                "provider_name": "sider",
                "route_name": "chatgpt_via_sider",
                "prompt_hash": prompt_hash,
                "repetition_index": 0,
                "execution_mode": "planned",
                "metadata": {},
            },
        ],
    )


def _rigor_pack(artifact, matrix):
    return build_technical_rigor_metric_pack(
        artifact,
        matrix,
        [
            {
                "metric_id": "r-1",
                "invocation_id": "inv-direct",
                "metric_name": "constraint_coverage",
                "score": 0.9,
                "evidence_count": 2,
                "metadata": {},
            },
            {
                "metric_id": "r-2",
                "invocation_id": "inv-wrapped",
                "metric_name": "scope_adherence",
                "score": 0.7,
                "evidence_count": 1,
                "metadata": {},
            },
        ],
    )


def _drift_tensor(artifact, matrix):
    return build_persona_drift_tensor(
        artifact,
        matrix,
        [
            {
                "metric_id": "d-1",
                "invocation_id": "inv-direct",
                "axis_name": "lexical_stability",
                "score": 0.8,
                "metadata": {},
            },
            {
                "metric_id": "d-2",
                "invocation_id": "inv-wrapped",
                "axis_name": "thematic_persistence",
                "score": 0.6,
                "metadata": {},
            },
        ],
    )


def _wrapper_study(artifact, matrix):
    return build_wrapper_divergence_study(
        artifact,
        matrix,
        [
            {
                "metric_id": "w-1",
                "primary_invocation_id": "inv-direct",
                "comparison_invocation_id": "inv-wrapped",
                "axis_name": "route_output_divergence",
                "score": 0.2,
                "metadata": {},
            }
        ],
    )


def _all_artifacts():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    rigor = _rigor_pack(artifact, matrix)
    drift = _drift_tensor(artifact, matrix)
    wrapper = _wrapper_study(artifact, matrix)
    stable_pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    return artifact, matrix, rigor, drift, wrapper, stable_pack


def _build_bundle():
    artifact, matrix, rigor, drift, wrapper, stable_pack = _all_artifacts()
    bundle = build_zenodo_dataset_export_bundle(
        artifact,
        matrix,
        rigor,
        drift,
        wrapper,
        stable_pack,
        dataset_id="qec-frontier-v1382",
        title="Frontier Comparative Determinism Harness Dataset",
        version_tag="v138.2.16",
        author="QSOLKCB",
        affiliation="Quantum Systems Open Lab",
        keywords=("Determinism", "Wrapper Divergence", "determinism", "quantum error correction"),
        methodology_notes="Deterministic comparative runtime archival bundle.",
        reproducibility_metadata={"archive_class": "publication-grade"},
    )
    return bundle, artifact, matrix, rigor, drift, wrapper, stable_pack


def test_same_input_same_bytes():
    a, *_ = _build_bundle()
    b, *_ = _build_bundle()
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_manifest_field_validation():
    bundle, artifact, matrix, rigor, drift, wrapper, stable_pack = _build_bundle()
    tampered = bundle.to_dict()
    tampered["manifest"]["title"] = ""
    report = validate_zenodo_dataset_export_bundle(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
        stable_evaluation_receipt_pack=stable_pack,
    )
    assert report.valid is False
    assert "manifest.title must be non-empty" in report.errors


def test_hash_mismatch_rejection():
    bundle, artifact, matrix, rigor, drift, wrapper, stable_pack = _build_bundle()
    tampered = bundle.to_dict()
    tampered["rigor_metric_pack_hash"] = "x" * 64
    report = validate_zenodo_dataset_export_bundle(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
        stable_evaluation_receipt_pack=stable_pack,
    )
    assert report.valid is False
    assert "bundle.rigor_metric_pack_hash mismatch" in report.errors


def test_duplicate_keyword_normalization():
    bundle, *_ = _build_bundle()
    assert bundle.manifest.keywords == (
        "determinism",
        "quantum error correction",
        "wrapper divergence",
    )


def test_canonical_json_round_trip():
    bundle, *_ = _build_bundle()
    payload = json.loads(bundle.to_canonical_json())
    projection = zenodo_export_projection(payload)
    assert projection["bundle_hash"] == bundle.manifest.export_hash


def test_receipt_tamper_detection():
    bundle, artifact, matrix, rigor, drift, wrapper, stable_pack = _build_bundle()
    tampered = bundle.to_dict()
    tampered["receipt"]["receipt_hash"] = "0" * 64
    report = validate_zenodo_dataset_export_bundle(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
        stable_evaluation_receipt_pack=stable_pack,
    )
    assert report.valid is False
    assert "receipt.receipt_hash mismatch" in report.errors


def test_projection_stability():
    bundle, *_ = _build_bundle()
    a = zenodo_export_projection(bundle)
    b = zenodo_export_projection(bundle)
    assert a == b


def test_malformed_metadata_rejection():
    artifact, matrix, rigor, drift, wrapper, stable_pack = _all_artifacts()
    with pytest.raises(ZenodoDatasetExportValidationError):
        build_zenodo_dataset_export_bundle(
            artifact,
            matrix,
            rigor,
            drift,
            wrapper,
            stable_pack,
            dataset_id="qec-frontier-v1382",
            title="t",
            version_tag="v",
            author="a",
            affiliation="aff",
            reproducibility_metadata={"bad": float("nan")},
        )


def test_reproducibility_metadata_determinism():
    bundle, *_ = _build_bundle()
    lineage = bundle.manifest.reproducibility_metadata["hash_lineage"]
    a = build_reproducibility_metadata(
        prompt_hash=lineage["prompt_hash"],
        matrix_hash=lineage["matrix_hash"],
        rigor_metric_pack_hash=lineage["rigor_metric_pack_hash"],
        drift_tensor_hash=lineage["drift_tensor_hash"],
        wrapper_study_hash=lineage["wrapper_study_hash"],
        stable_receipt_pack_hash=lineage["stable_receipt_pack_hash"],
    )
    b = build_reproducibility_metadata(
        prompt_hash=lineage["prompt_hash"],
        matrix_hash=lineage["matrix_hash"],
        rigor_metric_pack_hash=lineage["rigor_metric_pack_hash"],
        drift_tensor_hash=lineage["drift_tensor_hash"],
        wrapper_study_hash=lineage["wrapper_study_hash"],
        stable_receipt_pack_hash=lineage["stable_receipt_pack_hash"],
    )
    assert a == b


def test_lineage_consistency_regression():
    bundle, *_ = _build_bundle()
    lineage = bundle.manifest.reproducibility_metadata["hash_lineage"]
    assert lineage["prompt_hash"] == bundle.prompt_artifact_hash
    assert lineage["matrix_hash"] == bundle.invocation_matrix_hash
    assert lineage["stable_receipt_pack_hash"] == bundle.stable_receipt_pack_hash


def test_export_bundle_hash_determinism():
    a, *_ = _build_bundle()
    b, *_ = _build_bundle()
    assert a.manifest.export_hash == b.manifest.export_hash


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("title", ""),
        ("author", ""),
        ("version_tag", ""),
    ),
)
def test_empty_title_author_version_rejection(field, value):
    artifact, matrix, rigor, drift, wrapper, stable_pack = _all_artifacts()
    kwargs = {
        "dataset_id": "qec-frontier-v1382",
        "title": "Frontier Comparative Determinism Harness Dataset",
        "version_tag": "v138.2.16",
        "author": "QSOLKCB",
        "affiliation": "Quantum Systems Open Lab",
    }
    kwargs[field] = value
    with pytest.raises(ZenodoDatasetExportValidationError):
        build_zenodo_dataset_export_bundle(
            artifact,
            matrix,
            rigor,
            drift,
            wrapper,
            stable_pack,
            **kwargs,
        )
