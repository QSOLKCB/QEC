# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.11 multi-model invocation matrix."""

from __future__ import annotations

import json

from qec.runtime.multi_model_invocation_matrix import (
    InvocationRecord,
    ModelInvocationSpec,
    build_multi_model_invocation_matrix,
    invocation_matrix_projection,
    simulate_invocation_records,
    validate_invocation_matrix,
)
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "pmx-001",
            "prompt_text": "Compare deterministic responses.",
            "system_prompt": "Be explicit.",
            "wrapper_metadata": {"wrapper": "none"},
            "model_name": "chatgpt_native",
            "invocation_route": "direct",
            "repetition_count": 2,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "matrix"},
        }
    )


def _spec_mappings(prompt_hash: str):
    return [
        {
            "invocation_id": "inv-b",
            "model_name": "grok",
            "provider_name": "xai",
            "route_name": "xai_direct",
            "prompt_hash": prompt_hash,
            "repetition_index": 1,
            "execution_mode": "planned",
            "metadata": {"lane": 2},
        },
        {
            "invocation_id": "inv-a",
            "model_name": "claude",
            "provider_name": "anthropic",
            "route_name": "anthropic_direct",
            "prompt_hash": prompt_hash,
            "repetition_index": 0,
            "execution_mode": "planned",
            "metadata": {"lane": 1},
        },
    ]


def test_same_input_same_bytes():
    artifact = _prompt_artifact()
    specs = _spec_mappings(artifact.receipt.prompt_hash)
    a = build_multi_model_invocation_matrix(artifact, specs)
    b = build_multi_model_invocation_matrix(artifact, specs)
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_ordering():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    assert [r.invocation_id for r in matrix.records] == ["inv-a", "inv-b"]


def test_duplicate_invocation_id_rejection():
    artifact = _prompt_artifact()
    specs = _spec_mappings(artifact.receipt.prompt_hash)
    specs[1]["invocation_id"] = specs[0]["invocation_id"]
    matrix = build_multi_model_invocation_matrix(artifact, specs)
    assert matrix.validation.valid is False
    assert "invocation_id must be unique" in matrix.validation.errors


def test_invalid_execution_mode_rejection():
    artifact = _prompt_artifact()
    specs = _spec_mappings(artifact.receipt.prompt_hash)
    specs[0]["execution_mode"] = "live"
    matrix = build_multi_model_invocation_matrix(artifact, specs)
    assert matrix.validation.valid is False
    assert "spec.execution_mode must be one of: planned, simulated, observed" in matrix.validation.errors


def test_invalid_status_rejection():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    tampered = {
        **matrix.to_dict(),
        "records": [
            {
                **matrix.records[0].to_dict(),
                "status": "unknown",
            },
            matrix.records[1].to_dict(),
        ],
    }
    report = validate_invocation_matrix(tampered)
    assert report.valid is False
    assert "record.status must be one of: pending, completed, failed, invalid" in report.errors


def test_prompt_hash_mismatch_rejection():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    tampered = {
        **matrix.to_dict(),
        "records": [
            {**matrix.records[0].to_dict(), "prompt_hash": "bad"},
            matrix.records[1].to_dict(),
        ],
    }
    report = validate_invocation_matrix(tampered)
    assert report.valid is False
    assert "record.prompt_hash mismatch" in report.errors


def test_record_invocation_id_mapping_mismatch_rejection():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    tampered = {
        **matrix.to_dict(),
        "records": [
            {**matrix.records[0].to_dict(), "invocation_id": "inv-z"},
            matrix.records[1].to_dict(),
        ],
    }
    report = validate_invocation_matrix(tampered)
    assert report.valid is False
    assert "record.invocation_id set must match spec.invocation_id set" in report.errors


def test_mapping_dataclass_parity():
    artifact = _prompt_artifact()
    mappings = _spec_mappings(artifact.receipt.prompt_hash)
    dataclasses = tuple(ModelInvocationSpec(**m) for m in mappings)
    a = build_multi_model_invocation_matrix(artifact, mappings)
    b = build_multi_model_invocation_matrix(artifact, dataclasses)
    assert a.receipt.matrix_hash == b.receipt.matrix_hash


def test_canonical_json_round_trip():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    payload = json.loads(matrix.to_canonical_json())
    rebuilt = build_multi_model_invocation_matrix(artifact, payload["invocation_specs"])
    assert rebuilt.receipt.matrix_hash == matrix.receipt.matrix_hash


def test_receipt_tamper_detection():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    tampered = {**matrix.to_dict(), "receipt": {**matrix.receipt.to_dict(), "receipt_hash": "0" * 64}}
    report = validate_invocation_matrix(tampered)
    assert report.valid is False
    assert "receipt.receipt_hash mismatch" in report.errors


def test_projection_stability():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    assert invocation_matrix_projection(matrix) == invocation_matrix_projection(matrix)


def test_simulated_record_generation():
    artifact = _prompt_artifact()
    specs = _spec_mappings(artifact.receipt.prompt_hash)
    records = simulate_invocation_records(specs, response_hashes={"inv-a": "abc"})
    assert isinstance(records[0], InvocationRecord)
    assert [r.execution_mode for r in records] == ["simulated", "simulated"]
    assert records[0].response_hash == "abc"
    assert records[1].response_hash is None


def test_deterministic_model_ordering():
    artifact = _prompt_artifact()
    matrix = build_multi_model_invocation_matrix(artifact, _spec_mappings(artifact.receipt.prompt_hash))
    projection = invocation_matrix_projection(matrix)
    assert projection["ordered_models"] == ["claude", "grok"]
