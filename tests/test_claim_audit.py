from __future__ import annotations

from copy import deepcopy

import pytest

from qec.analysis.claim_audit import (
    CLAIM_AUDIT_SCHEMA_VERSION,
    ClaimAuditDecision,
    ClaimAuditFinding,
    build_claim_audit_receipt,
    compile_claim_audit,
    normalize_claim_audit_input,
    run_claim_audit,
    stable_claim_audit_hash,
)


def _raw_input() -> dict[str, object]:
    return {
        "claim_id": "claim-1",
        "claim_text": "Logical error rate is below threshold.",
        "experiment_hash": "exp-hash-001",
        "evidence_graph_hash": "graph-hash-001",
        "measurement_ids": ["m-2", "m-1"],
        "criterion_ids": ["c-2", "c-1"],
        "expected_relations": {
            "c-1": {"measurement_id": "m-1", "node_ids": ["n-2", "n-1"]},
            "c-2": {"measurement_id": "m-2", "node_ids": ["n-3"]},
        },
        "provenance": {"source": "unit-test", "layer": 4},
        "schema_version": CLAIM_AUDIT_SCHEMA_VERSION,
    }


def test_supported_verdict_when_criteria_satisfied_and_evidence_present() -> None:
    _, decision, findings, _ = compile_claim_audit(
        _raw_input(),
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    assert decision.verdict == "supported"
    assert len(decision.supporting_finding_ids) == 2
    assert all(f.finding_type == "criterion_satisfied" for f in findings)


def test_contradicted_verdict_when_required_criterion_failed() -> None:
    _, decision, findings, _ = compile_claim_audit(
        _raw_input(),
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "failed", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    assert decision.verdict == "contradicted"
    assert any(f.finding_type == "criterion_failed" for f in findings)


def test_inconclusive_verdict_when_evidence_missing() -> None:
    _, decision, findings, _ = compile_claim_audit(
        _raw_input(),
        available_measurements={"m-1": "present"},
        available_criteria={"c-1": "satisfied"},
        evidence_graph=None,
    )
    assert decision.verdict == "inconclusive"
    assert any(f.finding_type == "measurement_missing" for f in findings)
    assert any(f.finding_type == "evidence_missing" for f in findings)


def test_hash_stability_across_repeated_runs() -> None:
    raw = _raw_input()
    out1 = compile_claim_audit(
        raw,
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    out2 = compile_claim_audit(
        raw,
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    assert out1[1].to_canonical_bytes() == out2[1].to_canonical_bytes()
    assert stable_claim_audit_hash(out1[1], out1[2]) == stable_claim_audit_hash(out2[1], out2[2])


def test_input_ordering_independence_for_related_ids_and_findings() -> None:
    raw_a = _raw_input()
    raw_b = _raw_input()
    raw_b["measurement_ids"] = ["m-1", "m-2"]
    raw_b["criterion_ids"] = ["c-1", "c-2"]
    raw_b["expected_relations"] = {
        "c-2": {"measurement_id": "m-2", "node_ids": ["n-3"]},
        "c-1": {"measurement_id": "m-1", "node_ids": ["n-1", "n-2"]},
    }

    out_a = compile_claim_audit(
        raw_a,
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    out_b = compile_claim_audit(
        raw_b,
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    assert out_a[0].to_canonical_bytes() == out_b[0].to_canonical_bytes()
    assert out_a[1].to_canonical_bytes() == out_b[1].to_canonical_bytes()


def test_unknown_referenced_measurement_emits_missing_measurement_finding() -> None:
    raw = _raw_input()
    _, _, findings, _ = compile_claim_audit(
        raw,
        available_measurements={"m-1": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    assert any(f.finding_type == "measurement_missing" and f.related_measurement_id == "m-2" for f in findings)


def test_lineage_gap_and_conflict_findings_emitted() -> None:
    _, _, findings, _ = compile_claim_audit(
        _raw_input(),
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-3"], "conflicts": [("n-1", "n-2")]},
    )
    assert any(f.finding_type == "lineage_gap" for f in findings)
    assert any(f.finding_type == "evidence_conflict" for f in findings)


def test_receipt_stability() -> None:
    _, decision, findings, receipt_a = compile_claim_audit(
        _raw_input(),
        available_measurements={"m-1": "present", "m-2": "present"},
        available_criteria={"c-1": "satisfied", "c-2": "satisfied"},
        evidence_graph={"nodes": ["n-1", "n-2", "n-3"], "conflicts": []},
    )
    receipt_b = build_claim_audit_receipt(decision, findings)
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_defensive_copy_behavior() -> None:
    source = _raw_input()
    copy = deepcopy(source)
    audit_input, _, _, _ = compile_claim_audit(copy)
    copy["claim_text"] = "mutated"
    copy["measurement_ids"] = ["x"]
    assert audit_input.claim_text == "Logical error rate is below threshold."
    assert audit_input.measurement_ids == ("m-1", "m-2")


def test_unsupported_schema_version_rejected() -> None:
    raw = _raw_input()
    raw["schema_version"] = "v0.0.1"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        normalize_claim_audit_input(raw)


def test_unsupported_severity_and_finding_type_rejected() -> None:
    decision = ClaimAuditDecision(
        claim_id="c",
        verdict="inconclusive",
        supporting_finding_ids=(),
        contradicting_finding_ids=(),
        inconclusive_finding_ids=("f-1",),
        rationale_summary="inconclusive: 1 evidence gaps, 0 missing measurements",
        experiment_hash="e",
        evidence_graph_hash="g",
        audit_hash="",
        schema_version=CLAIM_AUDIT_SCHEMA_VERSION,
    )

    bad_type = ClaimAuditFinding(
        finding_id="f-1",
        finding_type="unknown",
        related_measurement_id="",
        related_criterion_id="",
        related_node_ids=(),
        message="x",
        severity="info",
    )
    with pytest.raises(ValueError, match="unsupported finding type"):
        stable_claim_audit_hash(decision, (bad_type,))

    bad_severity = ClaimAuditFinding(
        finding_id="f-1",
        finding_type="evidence_missing",
        related_measurement_id="",
        related_criterion_id="",
        related_node_ids=(),
        message="x",
        severity="fatal",
    )
    with pytest.raises(ValueError, match="unsupported severity"):
        build_claim_audit_receipt(decision, (bad_severity,))


def test_duplicate_finding_ids_rejected() -> None:
    audit_input = normalize_claim_audit_input(_raw_input())
    decision, _ = run_claim_audit(audit_input)
    repeated = (
        ClaimAuditFinding(
            finding_id="f-1",
            finding_type="evidence_missing",
            related_measurement_id="",
            related_criterion_id="",
            related_node_ids=(),
            message="a",
            severity="warning",
        ),
        ClaimAuditFinding(
            finding_id="f-1",
            finding_type="lineage_gap",
            related_measurement_id="",
            related_criterion_id="",
            related_node_ids=(),
            message="b",
            severity="error",
        ),
    )
    with pytest.raises(ValueError, match="duplicate finding IDs"):
        stable_claim_audit_hash(decision, repeated)


def test_empty_schema_version_rejected() -> None:
    raw = _raw_input()
    raw["schema_version"] = ""
    with pytest.raises(ValueError, match="schema_version must be a non-empty string"):
        normalize_claim_audit_input(raw)


def test_unavailable_measurement_status_emits_missing() -> None:
    """When a measurement status is 'missing' or similar, it should emit measurement_missing."""
    raw = _raw_input()
    raw["measurement_ids"] = ["m-1"]
    raw["criterion_ids"] = ["c-1"]
    raw["expected_relations"] = {"c-1": {"measurement_id": "m-1", "node_ids": ["n-1"]}}

    _, decision, findings, _ = compile_claim_audit(
        raw,
        available_measurements={"m-1": "missing"},
        available_criteria={"c-1": "satisfied"},
        evidence_graph={"nodes": ["n-1"], "conflicts": []},
    )
    assert decision.verdict == "inconclusive"
    assert any(f.finding_type == "measurement_missing" and f.related_measurement_id == "m-1" for f in findings)


def test_unavailable_measurement_with_status_object_emits_missing() -> None:
    """When a measurement status is {'status': 'missing'}, it should emit measurement_missing."""
    raw = _raw_input()
    raw["measurement_ids"] = ["m-1"]
    raw["criterion_ids"] = ["c-1"]
    raw["expected_relations"] = {"c-1": {"measurement_id": "m-1", "node_ids": ["n-1"]}}

    _, decision, findings, _ = compile_claim_audit(
        raw,
        available_measurements={"m-1": {"status": "missing"}},
        available_criteria={"c-1": "satisfied"},
        evidence_graph={"nodes": ["n-1"], "conflicts": []},
    )
    assert decision.verdict == "inconclusive"
    assert any(f.finding_type == "measurement_missing" and f.related_measurement_id == "m-1" for f in findings)


def test_duplicate_finding_ids_rejected_in_build_receipt() -> None:
    """build_claim_audit_receipt should reject duplicate finding IDs."""
    audit_input = normalize_claim_audit_input(_raw_input())
    decision, _ = run_claim_audit(audit_input)
    repeated = (
        ClaimAuditFinding(
            finding_id="f-1",
            finding_type="evidence_missing",
            related_measurement_id="",
            related_criterion_id="",
            related_node_ids=(),
            message="a",
            severity="warning",
        ),
        ClaimAuditFinding(
            finding_id="f-1",
            finding_type="lineage_gap",
            related_measurement_id="",
            related_criterion_id="",
            related_node_ids=(),
            message="b",
            severity="error",
        ),
    )
    with pytest.raises(ValueError, match="duplicate finding IDs"):
        build_claim_audit_receipt(decision, repeated)


def test_empty_object_round_trips_correctly() -> None:
    """Empty objects {} should serialize as {} not []."""
    raw = _raw_input()
    raw["provenance"] = {}
    audit_input = normalize_claim_audit_input(raw)
    d = audit_input.to_dict()
    assert d["provenance"] == {}
    assert isinstance(d["provenance"], dict)
