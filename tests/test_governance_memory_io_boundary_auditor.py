from __future__ import annotations

import json

from qec.orchestration.governance_memory_io_boundary_auditor import (
    BoundaryAuditFinding,
    BoundaryAuditRule,
    GovernanceMemoryIOBoundaryAuditReport,
    audit_governance_memory_io_boundaries,
    compare_boundary_audit_replay,
    summarize_boundary_findings,
    validate_boundary_audit_report,
)


def _base_kwargs() -> dict:
    return {
        "covenant_metadata": {"covenant_id": "cov-1", "capsule_id": "cap-1"},
        "payload": {"op": "observe", "value": 1},
        "state": {"register": 1},
        "proof_chain": ["p0", "p1"],
        "action_scope": "observe",
        "io_surface": "none",
        "replay_identity": "rid-1",
        "declared_replay_identity": "rid-1",
        "transition_receipt": {"receipt_hash": "curr", "prior_receipt_hash": "prev"},
        "prior_transition_receipt": {"receipt_hash": "prev"},
    }


def test_deterministic_repeated_audits():
    kwargs = _base_kwargs()
    report_a = audit_governance_memory_io_boundaries(**kwargs)
    report_b = audit_governance_memory_io_boundaries(**kwargs)
    assert report_a.to_canonical_json() == report_b.to_canonical_json()
    assert report_a.stable_hash() == report_b.stable_hash()


def test_stable_hash_reproducibility_for_rules_and_findings():
    rule = BoundaryAuditRule(
        rule_id="rule-x",
        dimension="payload_size",
        max_value=10,
        allowed_values=(),
        required=True,
        severity="high",
    )
    finding = BoundaryAuditFinding(
        finding_code="payload_too_large",
        dimension="payload_size",
        severity="high",
        rule_id="payload_size_max",
        message="payload size exceeds boundary",
        expected="<= 10",
        observed="11",
    )
    assert rule.stable_hash() == rule.stable_hash()
    assert finding.stable_hash() == finding.stable_hash()


def test_malformed_input_returns_findings_not_exceptions():
    report = audit_governance_memory_io_boundaries(
        covenant_metadata=3,
        payload={"op": "x"},
        state={"s": 1},
        proof_chain=object(),
        action_scope="forbidden",
        io_surface="external",
        replay_identity="rid-x",
        declared_replay_identity="rid-y",
        transition_receipt={},
        prior_transition_receipt={},
    )
    codes = {item.finding_code for item in report.findings}
    assert "malformed_covenant_metadata" in codes
    assert "proof_depth_exceeded" in codes
    assert "invalid_replay_identity" in codes


def test_oversized_payload_finding():
    rules = (
        BoundaryAuditRule("payload_size_max", "payload_size", 8, (), True, "high"),
        BoundaryAuditRule("state_size_max", "state_size", 99999, (), True, "high"),
    )
    kwargs = _base_kwargs()
    kwargs["payload"] = {"big": "x" * 100}
    report = audit_governance_memory_io_boundaries(**kwargs, boundary_rules=rules)
    assert any(f.finding_code == "payload_too_large" for f in report.findings)


def test_oversized_state_finding():
    rules = (
        BoundaryAuditRule("payload_size_max", "payload_size", 99999, (), True, "high"),
        BoundaryAuditRule("state_size_max", "state_size", 8, (), True, "high"),
    )
    kwargs = _base_kwargs()
    kwargs["state"] = {"big": "y" * 100}
    report = audit_governance_memory_io_boundaries(**kwargs, boundary_rules=rules)
    assert any(f.finding_code == "state_too_large" for f in report.findings)


def test_proof_depth_finding():
    rules = (BoundaryAuditRule("proof_chain_depth_max", "proof_chain_depth", 1, (), True, "medium"),)
    kwargs = _base_kwargs()
    kwargs["proof_chain"] = ["p0", "p1", "p2"]
    report = audit_governance_memory_io_boundaries(**kwargs, boundary_rules=rules)
    assert any(f.finding_code == "proof_depth_exceeded" for f in report.findings)


def test_replay_identity_finding():
    kwargs = _base_kwargs()
    kwargs["replay_identity"] = "rid-a"
    kwargs["declared_replay_identity"] = "rid-b"
    report = audit_governance_memory_io_boundaries(**kwargs)
    assert any(f.finding_code == "invalid_replay_identity" for f in report.findings)


def test_missing_receipt_finding():
    kwargs = _base_kwargs()
    kwargs["transition_receipt"] = {"receipt_hash": "curr", "prior_receipt_hash": "missing"}
    kwargs["prior_transition_receipt"] = {"receipt_hash": "prev"}
    report = audit_governance_memory_io_boundaries(**kwargs)
    assert any(f.finding_code == "missing_receipt_continuity" for f in report.findings)


def test_validator_never_raises_and_reports_violations():
    valid, violations = validate_boundary_audit_report(object())
    assert valid is False
    assert "invalid_report_type" in violations


def test_validator_passes_for_valid_report():
    report = audit_governance_memory_io_boundaries(**_base_kwargs())
    valid, violations = validate_boundary_audit_report(report)
    assert valid is True
    assert violations == ()


def test_malformed_payload_emits_finding_not_exception():
    kwargs = _base_kwargs()
    kwargs["payload"] = object()
    report = audit_governance_memory_io_boundaries(**kwargs)
    assert any(f.finding_code == "malformed_payload" for f in report.findings)


def test_malformed_state_emits_finding_not_exception():
    kwargs = _base_kwargs()
    kwargs["state"] = object()
    report = audit_governance_memory_io_boundaries(**kwargs)
    assert any(f.finding_code == "malformed_state" for f in report.findings)


def test_replay_comparison_stability():
    kwargs = _base_kwargs()
    report_a = audit_governance_memory_io_boundaries(**kwargs)
    report_b = audit_governance_memory_io_boundaries(**kwargs)
    cmp_data = compare_boundary_audit_replay(report_a, report_b)
    assert cmp_data["match"] is True
    assert cmp_data["mismatch_fields"] == ()


def test_canonical_json_round_trip():
    report = audit_governance_memory_io_boundaries(**_base_kwargs())
    loaded = json.loads(report.to_canonical_json())
    assert loaded["report_hash"] == report.report_hash
    assert loaded["audit_receipt"]["receipt_hash"] == report.audit_receipt.receipt_hash


def test_deterministic_findings_ordering():
    kwargs = _base_kwargs()
    kwargs["covenant_metadata"] = {}
    kwargs["action_scope"] = "bad"
    kwargs["io_surface"] = "external"
    kwargs["replay_identity"] = "x"
    kwargs["declared_replay_identity"] = "y"
    kwargs["transition_receipt"] = {}
    kwargs["prior_transition_receipt"] = {}
    report = audit_governance_memory_io_boundaries(**kwargs)
    key_view = [
        (
            f.severity,
            f.rule_id,
            f.finding_code,
            f.dimension,
            f.message,
            f.expected,
            f.observed,
        )
        for f in report.findings
    ]
    assert key_view == sorted(key_view)


def test_summary_contains_all_severity_levels():
    findings = (
        BoundaryAuditFinding("a", "d", "critical", "r1", "m", "e", "o"),
        BoundaryAuditFinding("b", "d", "low", "r2", "m", "e", "o"),
    )
    summary = summarize_boundary_findings(findings)
    assert summary == {"critical": 1, "high": 0, "medium": 0, "low": 1}


def test_report_type_is_frozen_dataclass_contract():
    report = audit_governance_memory_io_boundaries(**_base_kwargs())
    assert isinstance(report, GovernanceMemoryIOBoundaryAuditReport)
