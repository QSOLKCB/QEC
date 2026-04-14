"""v137.18.2 — Governance Memory / I-O Boundary Auditor.

Deterministic, non-mutating boundary audit primitives for covenant execution.
This module performs audit/report only and never blocks execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple


_AUDITOR_VERSION = "v137.18.2"
_ALLOWED_ACTION_SCOPES: Tuple[str, ...] = ("certify", "observe", "summarize", "traverse", "validate")
_ALLOWED_IO_SURFACES: Tuple[str, ...] = ("none", "internal")
_SUPPORTED_SEVERITIES: Tuple[str, ...] = ("critical", "high", "medium", "low")


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_len(value: Any) -> int:
    return len(_canonical_json(value).encode("utf-8"))


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


@dataclass(frozen=True)
class BoundaryAuditRule:
    rule_id: str
    dimension: str
    max_value: int | None
    allowed_values: Tuple[str, ...]
    required: bool
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "dimension": self.dimension,
            "max_value": self.max_value,
            "allowed_values": list(self.allowed_values),
            "required": self.required,
            "severity": self.severity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class BoundaryAuditFinding:
    finding_code: str
    dimension: str
    severity: str
    rule_id: str
    message: str
    expected: str
    observed: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_code": self.finding_code,
            "dimension": self.dimension,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
            "expected": self.expected,
            "observed": self.observed,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class BoundaryAuditReceipt:
    auditor_version: str
    replay_identity: str
    replay_binding: str
    transition_receipt_hash: str
    prior_transition_receipt_hash: str
    findings_hash: str
    continuity_ok: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auditor_version": self.auditor_version,
            "replay_identity": self.replay_identity,
            "replay_binding": self.replay_binding,
            "transition_receipt_hash": self.transition_receipt_hash,
            "prior_transition_receipt_hash": self.prior_transition_receipt_hash,
            "findings_hash": self.findings_hash,
            "continuity_ok": self.continuity_ok,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceMemoryIOBoundaryAuditReport:
    within_boundary: bool
    findings: Tuple[BoundaryAuditFinding, ...]
    violated_rules: Tuple[str, ...]
    severity_summary: Dict[str, int]
    audit_receipt: BoundaryAuditReceipt
    report_hash: str
    replay_binding: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "within_boundary": self.within_boundary,
            "findings": [finding.to_dict() for finding in self.findings],
            "violated_rules": list(self.violated_rules),
            "severity_summary": dict(self.severity_summary),
            "audit_receipt": self.audit_receipt.to_dict(),
            "report_hash": self.report_hash,
            "replay_binding": self.replay_binding,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.report_hash


def _default_boundary_rules() -> Tuple[BoundaryAuditRule, ...]:
    return (
        BoundaryAuditRule("action_scope_allowed", "action_scope", None, _ALLOWED_ACTION_SCOPES, True, "medium"),
        BoundaryAuditRule("io_surface_allowed", "io_surface", None, _ALLOWED_IO_SURFACES, True, "high"),
        BoundaryAuditRule("payload_size_max", "payload_size", 4096, (), True, "high"),
        BoundaryAuditRule("proof_chain_depth_max", "proof_chain_depth", 8, (), True, "medium"),
        BoundaryAuditRule("replay_identity_valid", "replay_identity", None, (), True, "critical"),
        BoundaryAuditRule("state_size_max", "state_size", 8192, (), True, "high"),
        BoundaryAuditRule("transition_receipt_continuity", "transition_receipt_continuity", None, (), True, "critical"),
    )


def _build_finding(
    *,
    finding_code: str,
    dimension: str,
    severity: str,
    rule_id: str,
    message: str,
    expected: str,
    observed: str,
) -> BoundaryAuditFinding:
    return BoundaryAuditFinding(
        finding_code=finding_code,
        dimension=dimension,
        severity=severity,
        rule_id=rule_id,
        message=message,
        expected=expected,
        observed=observed,
    )


def summarize_boundary_findings(findings: Sequence[BoundaryAuditFinding]) -> Dict[str, int]:
    summary: Dict[str, int] = {level: 0 for level in _SUPPORTED_SEVERITIES}
    for finding in findings:
        level = finding.severity if finding.severity in summary else "low"
        summary[level] += 1
    return summary


def build_boundary_audit_receipt(
    *,
    replay_identity: str,
    replay_binding: str,
    transition_receipt_hash: str,
    prior_transition_receipt_hash: str,
    findings: Sequence[BoundaryAuditFinding],
    continuity_ok: bool,
) -> BoundaryAuditReceipt:
    findings_hash_payload = [finding.stable_hash() for finding in findings]
    findings_hash = _sha256_hex(_canonical_json(findings_hash_payload).encode("utf-8"))
    preimage = {
        "auditor_version": _AUDITOR_VERSION,
        "continuity_ok": continuity_ok,
        "findings_hash": findings_hash,
        "prior_transition_receipt_hash": prior_transition_receipt_hash,
        "replay_binding": replay_binding,
        "replay_identity": replay_identity,
        "transition_receipt_hash": transition_receipt_hash,
    }
    receipt_hash = _sha256_hex(_canonical_json(preimage).encode("utf-8"))
    return BoundaryAuditReceipt(
        auditor_version=_AUDITOR_VERSION,
        replay_identity=replay_identity,
        replay_binding=replay_binding,
        transition_receipt_hash=transition_receipt_hash,
        prior_transition_receipt_hash=prior_transition_receipt_hash,
        findings_hash=findings_hash,
        continuity_ok=continuity_ok,
        receipt_hash=receipt_hash,
    )


def compare_boundary_audit_replay(
    report_a: GovernanceMemoryIOBoundaryAuditReport,
    report_b: GovernanceMemoryIOBoundaryAuditReport,
) -> Dict[str, Any]:
    fields = (
        "report_hash",
        "replay_binding",
        "within_boundary",
        "violated_rules",
        "severity_summary",
    )
    mismatches = []
    for field in fields:
        if getattr(report_a, field) != getattr(report_b, field):
            mismatches.append(field)
    return {
        "match": len(mismatches) == 0,
        "mismatch_fields": tuple(mismatches),
        "report_a_hash": report_a.report_hash,
        "report_b_hash": report_b.report_hash,
    }


def validate_boundary_audit_report(report: Any) -> Tuple[bool, Tuple[str, ...]]:
    violations: list[str] = []
    try:
        if not isinstance(report, GovernanceMemoryIOBoundaryAuditReport):
            violations.append("invalid_report_type")
        else:
            for idx, finding in enumerate(report.findings):
                if not isinstance(finding, BoundaryAuditFinding):
                    violations.append(f"invalid_finding_type:{idx}")
            if not isinstance(report.audit_receipt, BoundaryAuditReceipt):
                violations.append("invalid_receipt_type")
            recomputed_summary = summarize_boundary_findings(report.findings)
            if recomputed_summary != report.severity_summary:
                violations.append("severity_summary_mismatch")
            recomputed_hash = _sha256_hex(report.to_canonical_json().encode("utf-8"))
            if report.report_hash != recomputed_hash:
                violations.append("report_hash_mismatch")
    except Exception as exc:  # pragma: no cover - defensive non-raising guarantee
        violations.append(f"validator_internal_error:{type(exc).__name__}")
    return (len(violations) == 0, tuple(sorted(violations)))


def audit_governance_memory_io_boundaries(
    *,
    covenant_metadata: Any,
    payload: Any,
    state: Any,
    proof_chain: Any,
    action_scope: Any,
    io_surface: Any,
    replay_identity: Any,
    declared_replay_identity: Any,
    transition_receipt: Any,
    prior_transition_receipt: Any,
    boundary_rules: Sequence[BoundaryAuditRule] | None = None,
) -> GovernanceMemoryIOBoundaryAuditReport:
    rules = tuple(boundary_rules) if boundary_rules is not None else _default_boundary_rules()
    rules_by_id = {rule.rule_id: rule for rule in sorted(rules, key=lambda item: item.rule_id)}
    findings: list[BoundaryAuditFinding] = []

    metadata = _as_mapping(covenant_metadata)
    if metadata is None:
        findings.append(
            _build_finding(
                finding_code="malformed_covenant_metadata",
                dimension="metadata",
                severity="high",
                rule_id="metadata_shape",
                message="covenant metadata must be a mapping",
                expected="mapping",
                observed=type(covenant_metadata).__name__,
            )
        )
    else:
        required_keys = ("covenant_id", "capsule_id")
        for required in required_keys:
            value = metadata.get(required)
            if not isinstance(value, str) or not value.strip():
                findings.append(
                    _build_finding(
                        finding_code="malformed_covenant_metadata",
                        dimension="metadata",
                        severity="high",
                        rule_id="metadata_shape",
                        message=f"missing or invalid metadata key: {required}",
                        expected="non-empty string",
                        observed=str(value),
                    )
                )

    payload_size = _canonical_len(payload)
    payload_rule = rules_by_id.get("payload_size_max")
    if payload_rule is not None and payload_rule.max_value is not None and payload_size > payload_rule.max_value:
        findings.append(
            _build_finding(
                finding_code="payload_too_large",
                dimension="payload_size",
                severity=payload_rule.severity,
                rule_id=payload_rule.rule_id,
                message="payload size exceeds boundary",
                expected=f"<= {payload_rule.max_value}",
                observed=str(payload_size),
            )
        )

    state_size = _canonical_len(state)
    state_rule = rules_by_id.get("state_size_max")
    if state_rule is not None and state_rule.max_value is not None and state_size > state_rule.max_value:
        findings.append(
            _build_finding(
                finding_code="state_too_large",
                dimension="state_size",
                severity=state_rule.severity,
                rule_id=state_rule.rule_id,
                message="state size exceeds boundary",
                expected=f"<= {state_rule.max_value}",
                observed=str(state_size),
            )
        )

    proof_depth_rule = rules_by_id.get("proof_chain_depth_max")
    proof_depth = len(proof_chain) if isinstance(proof_chain, Sequence) and not isinstance(proof_chain, (str, bytes, bytearray)) else -1
    if proof_depth < 0:
        findings.append(
            _build_finding(
                finding_code="proof_depth_exceeded",
                dimension="proof_chain_depth",
                severity=proof_depth_rule.severity if proof_depth_rule is not None else "medium",
                rule_id="proof_chain_depth_max",
                message="proof chain is malformed",
                expected="sequence",
                observed=type(proof_chain).__name__,
            )
        )
    elif proof_depth_rule is not None and proof_depth_rule.max_value is not None and proof_depth > proof_depth_rule.max_value:
        findings.append(
            _build_finding(
                finding_code="proof_depth_exceeded",
                dimension="proof_chain_depth",
                severity=proof_depth_rule.severity,
                rule_id=proof_depth_rule.rule_id,
                message="proof chain depth exceeds boundary",
                expected=f"<= {proof_depth_rule.max_value}",
                observed=str(proof_depth),
            )
        )

    action_scope_rule = rules_by_id.get("action_scope_allowed")
    action_scope_text = str(action_scope)
    if action_scope_rule is not None and action_scope_text not in action_scope_rule.allowed_values:
        findings.append(
            _build_finding(
                finding_code="action_scope_disallowed",
                dimension="action_scope",
                severity=action_scope_rule.severity,
                rule_id=action_scope_rule.rule_id,
                message="action scope is outside allowed covenant surface",
                expected=",".join(action_scope_rule.allowed_values),
                observed=action_scope_text,
            )
        )

    io_surface_rule = rules_by_id.get("io_surface_allowed")
    io_surface_text = str(io_surface)
    if io_surface_rule is not None and io_surface_text not in io_surface_rule.allowed_values:
        findings.append(
            _build_finding(
                finding_code="disallowed_io_surface",
                dimension="io_surface",
                severity=io_surface_rule.severity,
                rule_id=io_surface_rule.rule_id,
                message="io surface is disallowed",
                expected=",".join(io_surface_rule.allowed_values),
                observed=io_surface_text,
            )
        )

    replay_rule = rules_by_id.get("replay_identity_valid")
    replay_text = str(replay_identity)
    declared_text = str(declared_replay_identity)
    if not replay_text or replay_text != declared_text:
        findings.append(
            _build_finding(
                finding_code="invalid_replay_identity",
                dimension="replay_identity",
                severity=replay_rule.severity if replay_rule is not None else "critical",
                rule_id="replay_identity_valid",
                message="replay identity does not match declared identity",
                expected=declared_text,
                observed=replay_text,
            )
        )

    continuity_rule = rules_by_id.get("transition_receipt_continuity")
    transition_map = _as_mapping(transition_receipt)
    prior_map = _as_mapping(prior_transition_receipt)
    continuity_ok = False
    transition_receipt_hash = ""
    prior_receipt_hash = ""
    if transition_map is not None:
        transition_receipt_hash = str(transition_map.get("receipt_hash", ""))
        prior_receipt_hash = str(transition_map.get("prior_receipt_hash", ""))
    prior_receipt_actual_hash = ""
    if prior_map is not None:
        prior_receipt_actual_hash = str(prior_map.get("receipt_hash", ""))
    if transition_receipt_hash and prior_receipt_hash and prior_receipt_actual_hash:
        continuity_ok = prior_receipt_hash == prior_receipt_actual_hash
    if not continuity_ok:
        findings.append(
            _build_finding(
                finding_code="missing_receipt_continuity",
                dimension="transition_receipt_continuity",
                severity=continuity_rule.severity if continuity_rule is not None else "critical",
                rule_id="transition_receipt_continuity",
                message="transition receipt continuity missing or mismatched",
                expected="prior_receipt_hash == prior_transition_receipt.receipt_hash",
                observed=f"{prior_receipt_hash} != {prior_receipt_actual_hash}",
            )
        )

    findings_sorted = tuple(
        sorted(
            findings,
            key=lambda item: (
                item.severity,
                item.rule_id,
                item.finding_code,
                item.dimension,
                item.message,
                item.expected,
                item.observed,
            ),
        )
    )

    violated_rules = tuple(sorted({finding.rule_id for finding in findings_sorted}))
    severity_summary = summarize_boundary_findings(findings_sorted)

    replay_binding_payload = {
        "declared_replay_identity": declared_text,
        "replay_identity": replay_text,
        "transition_receipt_hash": transition_receipt_hash,
    }
    replay_binding = _sha256_hex(_canonical_json(replay_binding_payload).encode("utf-8"))
    receipt = build_boundary_audit_receipt(
        replay_identity=replay_text,
        replay_binding=replay_binding,
        transition_receipt_hash=transition_receipt_hash,
        prior_transition_receipt_hash=prior_receipt_actual_hash,
        findings=findings_sorted,
        continuity_ok=continuity_ok,
    )

    report_preimage = {
        "audit_receipt": receipt.to_dict(),
        "findings": [finding.to_dict() for finding in findings_sorted],
        "replay_binding": replay_binding,
        "severity_summary": severity_summary,
        "violated_rules": violated_rules,
        "within_boundary": len(findings_sorted) == 0,
    }
    report_hash = _sha256_hex(_canonical_json(report_preimage).encode("utf-8"))
    return GovernanceMemoryIOBoundaryAuditReport(
        within_boundary=len(findings_sorted) == 0,
        findings=findings_sorted,
        violated_rules=violated_rules,
        severity_summary=severity_summary,
        audit_receipt=receipt,
        report_hash=report_hash,
        replay_binding=replay_binding,
    )
