# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.13 persona drift tensor."""

from __future__ import annotations

import json
import math

from qec.runtime.multi_model_invocation_matrix import build_multi_model_invocation_matrix
from qec.runtime.persona_drift_tensor import (
    PersonaDriftReceipt,
    build_persona_drift_tensor,
    compute_aggregate_stability,
    compute_drift_magnitude,
    persona_drift_projection,
    validate_persona_drift_tensor,
)
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "drift-001",
            "prompt_text": "Respond in a stable persona across repeated runs.",
            "system_prompt": "Preserve tone and style deterministically.",
            "wrapper_metadata": {"wrapper": "none"},
            "model_name": "chatgpt_native",
            "invocation_route": "direct",
            "repetition_count": 2,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "drift"},
        }
    )


def _invocation_matrix(artifact):
    prompt_hash = artifact.receipt.prompt_hash
    return build_multi_model_invocation_matrix(
        artifact,
        [
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
        ],
    )


def _metric_mappings():
    return [
        {
            "metric_id": "m-3",
            "invocation_id": "inv-b",
            "axis_name": "tone_drift",
            "score": 0.6,
            "metadata": {"source": "eval"},
        },
        {
            "metric_id": "m-1",
            "invocation_id": "inv-a",
            "axis_name": "lexical_stability",
            "score": 0.9,
            "metadata": {"source": "eval"},
        },
        {
            "metric_id": "m-2",
            "invocation_id": "inv-a",
            "axis_name": "thematic_persistence",
            "score": 0.8,
            "metadata": {"source": "eval"},
        },
    ]


def test_same_input_same_bytes():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    a = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    b = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_tensor_ordering():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    assert [(m.invocation_id, m.axis_name, m.metric_id) for m in tensor.metrics] == [
        ("inv-a", "lexical_stability", "m-1"),
        ("inv-a", "thematic_persistence", "m-2"),
        ("inv-b", "tone_drift", "m-3"),
    ]


def test_invalid_axis_name_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["axis_name"] = "unsupported"
    tensor = build_persona_drift_tensor(artifact, matrix, metrics)
    assert tensor.validation.valid is False
    assert any("metric.axis_name must be one of" in e for e in tensor.validation.errors)


def test_duplicate_metric_id_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[2]["metric_id"] = "m-1"
    tensor = build_persona_drift_tensor(artifact, matrix, metrics)
    assert tensor.validation.valid is False
    assert "metric_id must be unique" in tensor.validation.errors


def test_invocation_id_mismatch_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["invocation_id"] = "inv-missing"
    tensor = build_persona_drift_tensor(artifact, matrix, metrics)
    assert tensor.validation.valid is False
    assert "metric.invocation_id must exist in matrix" in tensor.validation.errors


def test_score_bounds_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["score"] = 1.2
    tensor = build_persona_drift_tensor(artifact, matrix, metrics)
    assert tensor.validation.valid is False
    assert "metric.score must be within [0.0, 1.0]" in tensor.validation.errors


def test_nan_inf_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)

    nan_tensor = build_persona_drift_tensor(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.nan}, _metric_mappings()[1]],
    )
    assert nan_tensor.validation.valid is False
    assert any("finite" in e for e in nan_tensor.validation.errors)

    inf_tensor = build_persona_drift_tensor(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.inf}, _metric_mappings()[1]],
    )
    assert inf_tensor.validation.valid is False
    assert any("finite" in e for e in inf_tensor.validation.errors)


def test_missing_matrix_invocation_coverage_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, [_metric_mappings()[0]])
    assert tensor.validation.valid is False
    assert "every matrix invocation must have at least one drift metric" in tensor.validation.errors


def test_canonical_json_round_trip():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    payload = json.loads(tensor.to_canonical_json())
    rebuilt = build_persona_drift_tensor(artifact, matrix, payload["metrics"])
    assert rebuilt.receipt.tensor_hash == tensor.receipt.tensor_hash


def test_receipt_tamper_detection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())

    forged_provisional = PersonaDriftReceipt(
        prompt_hash=tensor.receipt.prompt_hash,
        matrix_hash=tensor.receipt.matrix_hash,
        tensor_hash=tensor.receipt.tensor_hash,
        receipt_hash="",
        validation_passed=False,
    )
    tampered = {
        **tensor.to_dict(),
        "receipt": {
            **tensor.receipt.to_dict(),
            "validation_passed": False,
            "receipt_hash": forged_provisional.stable_hash(),
        },
    }
    report = validate_persona_drift_tensor(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
    )
    assert report.valid is False
    assert "receipt.validation_passed mismatch" in report.errors


def test_validation_handles_non_iterable_metrics_payload():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    payload = tensor.to_dict()
    payload["metrics"] = None
    report = validate_persona_drift_tensor(
        payload,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
    )
    assert report.valid is False
    assert "tensor.metrics must be an iterable sequence" in report.errors


def test_validation_handles_integer_metrics_payload():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    payload = tensor.to_dict()
    payload["metrics"] = 5
    report = validate_persona_drift_tensor(
        payload,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
    )
    assert report.valid is False
    assert "tensor.metrics must be an iterable sequence" in report.errors


def test_projection_stability():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    assert persona_drift_projection(tensor) == persona_drift_projection(tensor)


def test_drift_magnitude_determinism():
    assert compute_drift_magnitude(0.75) == 0.25


def test_aggregate_stability_determinism():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    tensor = build_persona_drift_tensor(artifact, matrix, _metric_mappings())
    helper_score = compute_aggregate_stability(tensor.metrics)
    assert helper_score == tensor.aggregate_stability_score
