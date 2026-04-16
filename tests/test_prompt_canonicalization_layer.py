# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.10 prompt canonicalization layer."""

from __future__ import annotations

import json

from qec.runtime.prompt_canonicalization_layer import (
    PromptSpec,
    build_canonical_prompt_artifact,
    canonical_prompt_projection,
    validate_prompt_spec,
)


def _valid_mapping() -> dict:
    return {
        "prompt_id": "prompt-001",
        "prompt_text": "Solve deterministically.",
        "system_prompt": "You are precise.",
        "wrapper_metadata": {"wrapper": "none", "version": "1"},
        "model_name": "chatgpt",
        "invocation_route": "chatgpt_direct",
        "repetition_count": 2,
        "temperature_setting": "0",
        "policy_flags": ("SAFE_MODE", " deterministic "),
        "metadata": {"suite": "frontier"},
    }


def test_same_input_same_bytes():
    payload = _valid_mapping()
    a = build_canonical_prompt_artifact(payload)
    b = build_canonical_prompt_artifact(payload)
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_surrounding_whitespace_normalization():
    payload = _valid_mapping()
    payload["prompt_id"] = "  prompt-001  "
    payload["prompt_text"] = "  keep inner   text  "
    payload["model_name"] = "  chatgpt  "
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.spec.prompt_id == "prompt-001"
    assert artifact.spec.prompt_text == "keep inner   text"
    assert artifact.spec.model_name == "chatgpt"


def test_policy_flag_normalization_and_sorting():
    payload = _valid_mapping()
    payload["policy_flags"] = [" ZETA", "alpha", "Beta "]
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.spec.policy_flags == ("alpha", "beta", "zeta")


def test_duplicate_policy_flag_rejection():
    payload = _valid_mapping()
    payload["policy_flags"] = ["safe", " SAFE "]
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.validation.valid is False
    assert "spec.policy_flags must be unique after normalization" in artifact.validation.errors


def test_missing_required_prompt_text_rejection():
    payload = _valid_mapping()
    payload["prompt_text"] = " "
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.validation.valid is False
    assert "spec.prompt_text must be non-empty" in artifact.validation.errors


def test_missing_required_model_name_rejection():
    payload = _valid_mapping()
    payload["model_name"] = ""
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.validation.valid is False
    assert "spec.model_name must be non-empty" in artifact.validation.errors


def test_system_prompt_none_allowed():
    payload = _valid_mapping()
    payload["system_prompt"] = None
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.validation.valid is True
    assert artifact.spec.system_prompt is None


def test_system_prompt_blank_string_rejected():
    payload = _valid_mapping()
    payload["system_prompt"] = "  "
    artifact = build_canonical_prompt_artifact(payload)
    assert artifact.validation.valid is False
    assert "spec.system_prompt must be non-empty when provided" in artifact.validation.errors


def test_wrapper_metadata_canonicalization_stability():
    a = _valid_mapping()
    b = _valid_mapping()
    a["wrapper_metadata"] = {"z": 1, "a": {"k2": "v2", "k1": "v1"}}
    b["wrapper_metadata"] = {"a": {"k1": "v1", "k2": "v2"}, "z": 1}
    artifact_a = build_canonical_prompt_artifact(a)
    artifact_b = build_canonical_prompt_artifact(b)
    assert artifact_a.receipt.wrapper_hash == artifact_b.receipt.wrapper_hash


def test_mapping_and_dataclass_inputs_produce_same_hash():
    payload = _valid_mapping()
    mapping_artifact = build_canonical_prompt_artifact(payload)
    dataclass_artifact = build_canonical_prompt_artifact(PromptSpec(**payload))
    assert mapping_artifact.receipt.prompt_hash == dataclass_artifact.receipt.prompt_hash


def test_receipt_tamper_detection():
    artifact = build_canonical_prompt_artifact(_valid_mapping())
    tampered = {
        **artifact.to_dict(),
        "receipt": {**artifact.receipt.to_dict(), "receipt_hash": "0" * 64},
    }
    report = validate_prompt_spec(tampered["spec"], tampered)
    assert report.valid is False
    assert "receipt.receipt_hash mismatch" in report.errors


def test_validate_rejects_raw_spec_artifact_pair_mismatch():
    base = _valid_mapping()
    artifact = build_canonical_prompt_artifact(base)
    raw_spec = {**base, "prompt_text": "Different caller text."}
    report = validate_prompt_spec(raw_spec, artifact.to_dict())
    assert report.valid is False
    assert "artifact.spec mismatch" in report.errors
    assert "receipt.prompt_hash mismatch" in report.errors
    assert "receipt.spec_hash mismatch" in report.errors
    assert "receipt.receipt_hash mismatch" in report.errors


def test_canonical_json_round_trip():
    artifact = build_canonical_prompt_artifact(_valid_mapping())
    payload = json.loads(artifact.to_canonical_json())
    rebuilt = build_canonical_prompt_artifact(payload["spec"])
    assert rebuilt.receipt.prompt_hash == artifact.receipt.prompt_hash
    assert payload["receipt"]["receipt_hash"] == artifact.receipt.receipt_hash


def test_projection_stability():
    artifact = build_canonical_prompt_artifact(_valid_mapping())
    a = canonical_prompt_projection(artifact)
    b = canonical_prompt_projection(artifact)
    assert a == b
