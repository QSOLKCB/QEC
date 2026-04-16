# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.14 wrapper divergence study."""

from __future__ import annotations

import json
import math

from qec.runtime.multi_model_invocation_matrix import build_multi_model_invocation_matrix
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact
from qec.runtime.wrapper_divergence_study import (
    WrapperDivergenceReceipt,
    build_wrapper_divergence_study,
    compute_aggregate_divergence,
    validate_wrapper_divergence_study,
    wrapper_divergence_projection,
)


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "wrapper-001",
            "prompt_text": "Produce the same deterministic answer via direct and wrapped routes.",
            "system_prompt": "Remain deterministic.",
            "wrapper_metadata": {"wrapper": "study"},
            "model_name": "chatgpt_native",
            "invocation_route": "comparative",
            "repetition_count": 1,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "wrapper_divergence"},
        }
    )


def _invocation_matrix(artifact):
    prompt_hash = artifact.receipt.prompt_hash
    return build_multi_model_invocation_matrix(
        artifact,
        [
            {
                "invocation_id": "inv-sider",
                "model_name": "chatgpt_5_4_sider",
                "provider_name": "sider",
                "route_name": "chatgpt_via_sider",
                "prompt_hash": prompt_hash,
                "repetition_index": 0,
                "execution_mode": "planned",
                "metadata": {"lane": 2},
            },
            {
                "invocation_id": "inv-direct",
                "model_name": "chatgpt_native",
                "provider_name": "openai",
                "route_name": "chatgpt_direct",
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
            "primary_invocation_id": "inv-direct",
            "comparison_invocation_id": "inv-sider",
            "axis_name": "tone_shift",
            "score": 0.3,
            "metadata": {"source": "eval"},
        },
        {
            "metric_id": "m-1",
            "primary_invocation_id": "inv-direct",
            "comparison_invocation_id": "inv-sider",
            "axis_name": "route_output_divergence",
            "score": 0.5,
            "metadata": {"source": "eval"},
        },
        {
            "metric_id": "m-2",
            "primary_invocation_id": "inv-direct",
            "comparison_invocation_id": "inv-sider",
            "axis_name": "wrapper_prompt_shift",
            "score": 0.7,
            "metadata": {"source": "eval"},
        },
    ]


def test_same_input_same_bytes():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    a = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    b = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_ordering():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    study = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    assert [
        (m.primary_invocation_id, m.comparison_invocation_id, m.axis_name, m.metric_id) for m in study.metrics
    ] == [
        ("inv-direct", "inv-sider", "route_output_divergence", "m-1"),
        ("inv-direct", "inv-sider", "tone_shift", "m-3"),
        ("inv-direct", "inv-sider", "wrapper_prompt_shift", "m-2"),
    ]


def test_invalid_axis_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["axis_name"] = "unsupported"
    study = build_wrapper_divergence_study(artifact, matrix, metrics)
    assert study.validation.valid is False
    assert any("metric.axis_name must be one of" in e for e in study.validation.errors)


def test_duplicate_metric_id_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[2]["metric_id"] = "m-1"
    study = build_wrapper_divergence_study(artifact, matrix, metrics)
    assert study.validation.valid is False
    assert "metric_id must be unique" in study.validation.errors


def test_invocation_id_mismatch_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["comparison_invocation_id"] = "inv-missing"
    study = build_wrapper_divergence_study(artifact, matrix, metrics)
    assert study.validation.valid is False
    assert "metric.comparison_invocation_id must exist in matrix" in study.validation.errors


def test_same_invocation_compared_to_itself_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["comparison_invocation_id"] = "inv-direct"
    study = build_wrapper_divergence_study(artifact, matrix, metrics)
    assert study.validation.valid is False
    assert "metric.primary_invocation_id must differ from metric.comparison_invocation_id" in study.validation.errors


def test_score_bounds_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["score"] = 1.1
    study = build_wrapper_divergence_study(artifact, matrix, metrics)
    assert study.validation.valid is False
    assert "metric.score must be within [0.0, 1.0]" in study.validation.errors


def test_nan_inf_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)

    nan_study = build_wrapper_divergence_study(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.nan}, _metric_mappings()[1]],
    )
    assert nan_study.validation.valid is False
    assert any("finite" in e for e in nan_study.validation.errors)

    inf_study = build_wrapper_divergence_study(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.inf}, _metric_mappings()[1]],
    )
    assert inf_study.validation.valid is False
    assert any("finite" in e for e in inf_study.validation.errors)


def test_canonical_json_round_trip():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    study = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    payload = json.loads(study.to_canonical_json())
    rebuilt = build_wrapper_divergence_study(artifact, matrix, payload["metrics"])
    assert rebuilt.receipt.study_hash == study.receipt.study_hash


def test_receipt_tamper_detection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    study = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())

    forged_provisional = WrapperDivergenceReceipt(
        prompt_hash=study.receipt.prompt_hash,
        matrix_hash=study.receipt.matrix_hash,
        study_hash=study.receipt.study_hash,
        receipt_hash="",
        validation_passed=False,
    )
    tampered = {
        **study.to_dict(),
        "receipt": {
            **study.receipt.to_dict(),
            "validation_passed": False,
            "receipt_hash": forged_provisional.stable_hash(),
        },
    }
    report = validate_wrapper_divergence_study(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
    )
    assert report.valid is False
    assert "receipt.validation_passed mismatch" in report.errors


def test_projection_stability():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    study = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    assert wrapper_divergence_projection(study) == wrapper_divergence_projection(study)


def test_aggregate_divergence_determinism():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    study = build_wrapper_divergence_study(artifact, matrix, _metric_mappings())
    helper_score = compute_aggregate_divergence(study.metrics)
    assert helper_score == study.aggregate_divergence_score == 0.5
