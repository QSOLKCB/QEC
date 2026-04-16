# SPDX-License-Identifier: MIT
"""Deterministic tests for v138.2.15 stable evaluation receipt pack."""

from __future__ import annotations

import json
import math

import pytest

from qec.runtime.multi_model_invocation_matrix import build_multi_model_invocation_matrix
from qec.runtime.persona_drift_tensor import build_persona_drift_tensor
from qec.runtime.prompt_canonicalization_layer import build_canonical_prompt_artifact
from qec.runtime.stable_evaluation_receipt_pack import (
    StableEvaluationReceipt,
    StableEvaluationReceiptPackValidationError,
    build_stable_evaluation_receipt_pack,
    compute_composite_evaluation_score,
    stable_evaluation_projection,
    validate_stable_evaluation_receipt_pack,
)
from qec.runtime.technical_rigor_metric_pack import build_technical_rigor_metric_pack
from qec.runtime.wrapper_divergence_study import build_wrapper_divergence_study


def _prompt_artifact():
    return build_canonical_prompt_artifact(
        {
            "prompt_id": "stable-pack-001",
            "prompt_text": "Provide a deterministic comparative answer.",
            "system_prompt": "Remain deterministic and strict.",
            "wrapper_metadata": {"wrapper": "none"},
            "model_name": "chatgpt_native",
            "invocation_route": "comparative",
            "repetition_count": 1,
            "temperature_setting": "0",
            "policy_flags": ("deterministic",),
            "metadata": {"suite": "stable_evaluation_receipt_pack"},
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
                "metadata": {"lane": 1},
            },
            {
                "invocation_id": "inv-wrapped",
                "model_name": "chatgpt_5_4_sider",
                "provider_name": "sider",
                "route_name": "chatgpt_via_sider",
                "prompt_hash": prompt_hash,
                "repetition_index": 0,
                "execution_mode": "planned",
                "metadata": {"lane": 2},
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
        ]
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
            },
            {
                "metric_id": "w-2",
                "primary_invocation_id": "inv-direct",
                "comparison_invocation_id": "inv-wrapped",
                "axis_name": "tone_shift",
                "score": 0.4,
                "metadata": {},
            },
        ],
    )


def _all_artifacts():
    artifact = _prompt_artifact()
    matrix = _invocation_matrix(artifact)
    rigor = _rigor_pack(artifact, matrix)
    drift = _drift_tensor(artifact, matrix)
    wrapper = _wrapper_study(artifact, matrix)
    return artifact, matrix, rigor, drift, wrapper


def test_same_input_same_bytes():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    a = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    b = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    assert a.to_canonical_json().encode("utf-8") == b.to_canonical_json().encode("utf-8")


def test_cross_hash_mismatch_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    tampered = {**pack.to_dict(), "rigor_pack_hash": "x" * 64}
    report = validate_stable_evaluation_receipt_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert report.valid is False
    assert "receipt_pack.rigor_pack_hash mismatch" in report.errors


def test_prompt_hash_mismatch_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    tampered = {**pack.to_dict(), "prompt_hash": "y" * 64}
    report = validate_stable_evaluation_receipt_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert report.valid is False
    assert "receipt_pack.prompt_hash mismatch" in report.errors


def test_matrix_hash_mismatch_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    tampered = {**pack.to_dict(), "matrix_hash": "z" * 64}
    report = validate_stable_evaluation_receipt_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert report.valid is False
    assert "receipt_pack.matrix_hash mismatch" in report.errors


def test_score_bounds_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    tampered = {**pack.to_dict(), "aggregate_divergence_score": 1.1}
    report = validate_stable_evaluation_receipt_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert report.valid is False
    assert "receipt_pack.aggregate_divergence_score must be within [0.0, 1.0]" in report.errors


def test_nan_inf_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)

    nan_tampered = {**pack.to_dict(), "aggregate_rigor_score": math.nan}
    nan_report = validate_stable_evaluation_receipt_pack(
        nan_tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert nan_report.valid is False
    assert any("finite" in e for e in nan_report.errors)

    inf_tampered = {**pack.to_dict(), "aggregate_stability_score": math.inf}
    inf_report = validate_stable_evaluation_receipt_pack(
        inf_tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert inf_report.valid is False
    assert any("finite" in e for e in inf_report.errors)


def test_canonical_json_round_trip():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    payload = json.loads(pack.to_canonical_json())
    rebuilt = stable_evaluation_projection(payload)
    assert rebuilt["pack_hash"] == pack.receipt.pack_hash
    assert rebuilt["receipt_hash"] != ""


def test_receipt_tamper_detection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)

    forged_provisional = StableEvaluationReceipt(
        prompt_hash=pack.receipt.prompt_hash,
        matrix_hash=pack.receipt.matrix_hash,
        rigor_pack_hash=pack.receipt.rigor_pack_hash,
        drift_tensor_hash=pack.receipt.drift_tensor_hash,
        wrapper_study_hash=pack.receipt.wrapper_study_hash,
        pack_hash=pack.receipt.pack_hash,
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
    report = validate_stable_evaluation_receipt_pack(
        tampered,
        canonical_prompt_artifact=artifact,
        invocation_matrix=matrix,
        rigor_metric_pack=rigor,
        drift_tensor=drift,
        wrapper_divergence_study=wrapper,
    )
    assert report.valid is False
    assert "receipt.validation_passed mismatch" in report.errors


def test_projection_stability():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)
    assert stable_evaluation_projection(pack) == stable_evaluation_projection(pack)


def test_composite_score_determinism():
    score = compute_composite_evaluation_score(0.8, 0.6, 0.1)
    assert score == (0.8 + 0.6 + 0.9) / 3.0


def test_malformed_artifact_rejection():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    with pytest.raises(StableEvaluationReceiptPackValidationError):
        build_stable_evaluation_receipt_pack(
            artifact,
            matrix,
            {"aggregate_score": 0.5},
            drift,
            wrapper,
        )


def test_lineage_consistency_regression():
    artifact, matrix, rigor, drift, wrapper = _all_artifacts()
    pack = build_stable_evaluation_receipt_pack(artifact, matrix, rigor, drift, wrapper)

    assert pack.prompt_hash == artifact.receipt.prompt_hash
    assert pack.matrix_hash == matrix.receipt.matrix_hash
    assert pack.rigor_pack_hash == rigor.receipt.metric_pack_hash
    assert pack.drift_tensor_hash == drift.receipt.tensor_hash
    assert pack.wrapper_study_hash == wrapper.receipt.study_hash
    assert pack.validation.valid is True
