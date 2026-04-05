"""Tests for v137.3.0 deterministic NLP governance layer."""

from __future__ import annotations

import hashlib
import json

import pytest

from qec.analysis.deterministic_nlp_governance_layer import (
    DETERMINISTIC_NLP_GOVERNANCE_VERSION,
    export_governed_output_bundle,
    export_governed_output_bytes,
    synthesize_deterministic_language_policy,
)


def test_repeated_run_determinism() -> None:
    text = "Critical outage: restore safety and report status"
    run_a = synthesize_deterministic_language_policy(text, policy_mode="strict")
    run_b = synthesize_deterministic_language_policy(text, policy_mode="strict")
    assert run_a == run_b


def test_canonical_json_bytes_stability() -> None:
    artifact = synthesize_deterministic_language_policy(
        "Analyze status and debug failure", policy_mode="balanced"
    )
    bytes_a = export_governed_output_bytes(artifact)
    bytes_b = export_governed_output_bytes(artifact)
    assert bytes_a == bytes_b


def test_stable_output_identity_chain() -> None:
    artifact = synthesize_deterministic_language_policy(
        "Urgent failure incident", policy_mode="strict"
    )
    assert len(artifact.identity_chain) == 5
    assert artifact.identity_chain[-1] == artifact.stable_hash
    for digest in artifact.identity_chain:
        assert len(digest) == 64
        int(digest, 16)


def test_deterministic_intent_classification_tie_break() -> None:
    artifact = synthesize_deterministic_language_policy(
        "recover inspect", policy_mode="balanced"
    )
    # recover -> stabilize, inspect -> diagnose, equal score => lexical tie-break
    assert artifact.plan.selected_intent == "diagnose"
    assert artifact.plan.selected_action == "emit_diagnostic_snapshot"


def test_deterministic_action_graph_generation() -> None:
    artifact = synthesize_deterministic_language_policy(
        "critical error status", policy_mode="strict"
    )
    graph = artifact.graph
    assert graph == tuple(sorted(graph, key=lambda edge: (edge.source, edge.target, edge.edge_type)))
    assert any(edge.edge_type == "selected_path" for edge in graph)


def test_governed_response_stability() -> None:
    text = "stabilize and restore safe operation"
    r1 = synthesize_deterministic_language_policy(text, policy_mode="strict")
    r2 = synthesize_deterministic_language_policy(text, policy_mode="strict")
    assert r1.plan == r2.plan
    assert export_governed_output_bundle(r1) == export_governed_output_bundle(r2)


def test_fail_fast_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        synthesize_deterministic_language_policy("   ", policy_mode="strict")

    with pytest.raises(ValueError, match="policy_mode"):
        synthesize_deterministic_language_policy("status", policy_mode="adaptive")


def test_stable_hash_matches_export_payload() -> None:
    artifact = synthesize_deterministic_language_policy(
        "critical restore report", policy_mode="strict"
    )
    bundle = export_governed_output_bundle(artifact)
    canonical_json = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    assert export_governed_output_bytes(artifact) == canonical_json.encode("utf-8")
    assert hashlib.sha256(export_governed_output_bytes(artifact)).hexdigest() == expected
    assert artifact.version == DETERMINISTIC_NLP_GOVERNANCE_VERSION
