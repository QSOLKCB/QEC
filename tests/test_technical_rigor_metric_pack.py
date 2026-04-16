# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.12 technical rigor metric pack."""

from __future__ import annotations

import json
import math

from qec.runtime.multi_model_invocation_matrix import build_multi_model_invocation_matrix
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact
from qec.runtime.technical_rigor_metric_pack import (
    RigorEvaluationReceipt,
    build_technical_rigor_metric_pack,
    compute_aggregate_rigor_score,
    rigor_metric_projection,
    validate_rigor_metric_pack,
)


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "rigor-001",
            "prompt_text": "Produce deterministic benchmark response.",
            "system_prompt": "Be exact.",
            "wrapper_metadata": {"wrapper": "none"},
            "model_name": "chatgpt_native",
            "invocation_route": "direct",
            "repetition_count": 2,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "rigor"},
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
            "metric_id": "m-2",
            "invocation_id": "inv-b",
            "metric_name": "scope_adherence",
            "score": 0.6,
            "evidence_count": 1,
            "metadata": {"source": "eval"},
        },
        {
            "metric_id": "m-1",
            "invocation_id": "inv-a",
            "metric_name": "constraint_coverage",
            "score": 0.8,
            "evidence_count": 3,
            "metadata": {"source": "eval"},
        },
    ]


def test_same_input_same_bytes():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    a = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    b = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_deterministic_metric_ordering():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    assert [(m.invocation_id, m.metric_name, m.metric_id) for m in pack.metrics] == [
        ("inv-a", "constraint_coverage", "m-1"),
        ("inv-b", "scope_adherence", "m-2"),
    ]


def test_invalid_metric_name_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["metric_name"] = "unsupported"
    pack = build_technical_rigor_metric_pack(artifact, matrix, metrics)
    assert pack.validation.valid is False
    assert any("metric.metric_name must be one of" in e for e in pack.validation.errors)


def test_duplicate_metric_id_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(
        artifact,
        matrix,
        [
            {
                "metric_id": "dup",
                "invocation_id": "inv-a",
                "metric_name": "constraint_coverage",
                "score": 0.8,
                "evidence_count": 1,
                "metadata": {},
            },
            {
                "metric_id": "dup",
                "invocation_id": "inv-b",
                "metric_name": "scope_adherence",
                "score": 0.2,
                "evidence_count": 1,
                "metadata": {},
            },
        ],
    )
    assert pack.validation.valid is False
    assert "metric_id must be unique" in pack.validation.errors


def test_invocation_id_mismatch_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["invocation_id"] = "inv-missing"
    pack = build_technical_rigor_metric_pack(artifact, matrix, metrics)
    assert pack.validation.valid is False
    assert "metric.invocation_id must exist in matrix" in pack.validation.errors


def test_missing_matrix_invocation_metric_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, [_metric_mappings()[0]])
    assert pack.validation.valid is False
    assert "every matrix invocation must have at least one metric" in pack.validation.errors


def test_score_bounds_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["score"] = 1.5
    pack = build_technical_rigor_metric_pack(artifact, matrix, metrics)
    assert pack.validation.valid is False
    assert "metric.score must be within [0.0, 1.0]" in pack.validation.errors


def test_nan_inf_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)

    nan_pack = build_technical_rigor_metric_pack(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.nan}, _metric_mappings()[1]],
    )
    assert nan_pack.validation.valid is False
    assert any("finite" in e for e in nan_pack.validation.errors)

    inf_pack = build_technical_rigor_metric_pack(
        artifact,
        matrix,
        [{**_metric_mappings()[0], "score": math.inf}, _metric_mappings()[1]],
    )
    assert inf_pack.validation.valid is False
    assert any("finite" in e for e in inf_pack.validation.errors)


def test_negative_evidence_count_rejection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    metrics = _metric_mappings()
    metrics[0]["evidence_count"] = -1
    pack = build_technical_rigor_metric_pack(artifact, matrix, metrics)
    assert pack.validation.valid is False
    assert "metric.evidence_count must be >= 0" in pack.validation.errors


def test_canonical_json_round_trip():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    payload = json.loads(pack.to_canonical_json())
    rebuilt = build_technical_rigor_metric_pack(artifact, matrix, payload["metrics"])
    assert rebuilt.receipt.metric_pack_hash == pack.receipt.metric_pack_hash


def test_receipt_tamper_detection():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())

    forged_provisional = RigorEvaluationReceipt(
        prompt_hash=pack.receipt.prompt_hash,
        matrix_hash=pack.receipt.matrix_hash,
        metric_pack_hash=pack.receipt.metric_pack_hash,
        receipt_hash="",
        validation_passed=False,
    )
    tampered = {
        **pack.to_dict(),
        "receipt": {
            **pack.receipt.to_dict(),
            "validation_passed": False,
            "receipt_hash": forged_provisional.stable_hash(),
        },
    }
    report = validate_rigor_metric_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
    )
    assert report.valid is False
    assert "receipt.validation_passed mismatch" in report.errors


def test_projection_stability():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    assert rigor_metric_projection(pack) == rigor_metric_projection(pack)


def test_aggregate_score_determinism():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    pack = build_technical_rigor_metric_pack(artifact, matrix, _metric_mappings())
    helper_score = compute_aggregate_rigor_score(pack.metrics)
    assert helper_score == pack.aggregate_score == 0.7
