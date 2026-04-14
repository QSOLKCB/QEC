"""v137.18.3 — Policy Execution Firewall Kernel.

Deterministic firewall decision layer for proof-carrying action capsules,
covenant execution results, and governance boundary audit reports.

This module is decision/receipt only:
- no side effects
- no mutation of upstream artifacts
- no external I/O
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from qec.orchestration.deterministic_covenant_engine import DeterministicCovenantExecution
from qec.orchestration.governance_memory_io_boundary_auditor import (
    GovernanceMemoryIOBoundaryAuditReport,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    ProofCarryingAgentActionCapsule,
)


_ALLOWED_DECISIONS: Tuple[str, ...] = ("allow", "deny")
_SEVERITY_LEVELS: Tuple[str, ...] = ("low", "medium", "high", "critical")
_SEVERITY_RANK: Dict[str, int] = {name: idx for idx, name in enumerate(_SEVERITY_LEVELS)}


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


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _stable_sorted_unique_text(values: Sequence[Any]) -> Tuple[str, ...]:
    normalized = {_safe_str(value).strip() for value in values}
    normalized.discard("")
    return tuple(sorted(normalized))


def _contains_severity_at_or_above(summary: Mapping[str, Any], threshold: str) -> bool:
    threshold_rank = _SEVERITY_RANK[threshold]
    for key, count in summary.items():
        level = _safe_str(key).strip().lower()
        if level not in _SEVERITY_RANK:
            continue
        if _SEVERITY_RANK[level] > threshold_rank:
            try:
                if int(count) > 0:
                    return True
            except Exception:
                return True
    return False


@dataclass(frozen=True)
class FirewallPolicyRule:
    rule_id: str
    allowed_action_types: Tuple[str, ...]
    allowed_action_scopes: Tuple[str, ...]
    require_within_boundary: bool
    max_allowed_severity: str
    disallowed_violated_rule_ids: Tuple[str, ...]
    require_receipt_continuity: bool
    required_replay_identity_prefix: str
    required_covenant_rule_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "allowed_action_types": list(self.allowed_action_types),
            "allowed_action_scopes": list(self.allowed_action_scopes),
            "require_within_boundary": self.require_within_boundary,
            "max_allowed_severity": self.max_allowed_severity,
            "disallowed_violated_rule_ids": list(self.disallowed_violated_rule_ids),
            "require_receipt_continuity": self.require_receipt_continuity,
            "required_replay_identity_prefix": self.required_replay_identity_prefix,
            "required_covenant_rule_id": self.required_covenant_rule_id,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class FirewallDecisionReason:
    reason_code: str
    dimension: str
    rule_id: str
    outcome: str
    expected: str
    observed: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "dimension": self.dimension,
            "rule_id": self.rule_id,
            "outcome": self.outcome,
            "expected": self.expected,
            "observed": self.observed,
            "message": self.message,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class FirewallExecutionReceipt:
    decision_hash: str
    decision_value: str
    action_capsule_hash: str
    covenant_execution_hash: str
    boundary_audit_report_hash: str
    policy_rules_hash: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_hash": self.decision_hash,
            "decision_value": self.decision_value,
            "action_capsule_hash": self.action_capsule_hash,
            "covenant_execution_hash": self.covenant_execution_hash,
            "boundary_audit_report_hash": self.boundary_audit_report_hash,
            "policy_rules_hash": self.policy_rules_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class PolicyExecutionFirewallDecision:
    decision_value: str
    reasons: Tuple[FirewallDecisionReason, ...]
    violations: Tuple[str, ...]
    decision_hash: str
    execution_receipt: FirewallExecutionReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_value": self.decision_value,
            "reasons": [reason.to_dict() for reason in self.reasons],
            "violations": list(self.violations),
            "decision_hash": self.decision_hash,
            "execution_receipt": self.execution_receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.decision_hash


def build_firewall_policy_rule(
    *,
    rule_id: str,
    allowed_action_types: Sequence[str] = (),
    allowed_action_scopes: Sequence[str] = (),
    require_within_boundary: bool = True,
    max_allowed_severity: str = "critical",
    disallowed_violated_rule_ids: Sequence[str] = (),
    require_receipt_continuity: bool = True,
    required_replay_identity_prefix: str = "",
    required_covenant_rule_id: str = "",
) -> FirewallPolicyRule:
    normalized_rule_id = _safe_str(rule_id).strip()
    if not normalized_rule_id:
        raise ValueError("rule_id must be non-empty")

    severity = _safe_str(max_allowed_severity).strip().lower()
    if severity not in _SEVERITY_RANK:
        raise ValueError("max_allowed_severity must be one of: low, medium, high, critical")

    return FirewallPolicyRule(
        rule_id=normalized_rule_id,
        allowed_action_types=_stable_sorted_unique_text(tuple(allowed_action_types)),
        allowed_action_scopes=_stable_sorted_unique_text(tuple(allowed_action_scopes)),
        require_within_boundary=bool(require_within_boundary),
        max_allowed_severity=severity,
        disallowed_violated_rule_ids=_stable_sorted_unique_text(tuple(disallowed_violated_rule_ids)),
        require_receipt_continuity=bool(require_receipt_continuity),
        required_replay_identity_prefix=_safe_str(required_replay_identity_prefix).strip(),
        required_covenant_rule_id=_safe_str(required_covenant_rule_id).strip(),
    )


def _reason(
    *,
    reason_code: str,
    dimension: str,
    rule_id: str,
    outcome: str,
    expected: Any,
    observed: Any,
    message: str,
) -> FirewallDecisionReason:
    return FirewallDecisionReason(
        reason_code=reason_code,
        dimension=dimension,
        rule_id=rule_id,
        outcome=outcome,
        expected=_safe_str(expected),
        observed=_safe_str(observed),
        message=message,
    )


def _sorted_reasons(reasons: Sequence[FirewallDecisionReason]) -> Tuple[FirewallDecisionReason, ...]:
    return tuple(
        sorted(
            reasons,
            key=lambda item: (
                item.dimension,
                item.reason_code,
                item.rule_id,
                item.outcome,
                item.expected,
                item.observed,
                item.message,
            ),
        )
    )


def _as_sequence(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return ()


def _hash_inputs(
    action_capsule: Any,
    covenant_execution: Any,
    boundary_audit_report: Any,
    policy_rules: Sequence[FirewallPolicyRule],
) -> Tuple[str, str, str, str]:
    action_hash = _safe_str(getattr(action_capsule, "stable_hash", lambda: "")())
    covenant_hash = _safe_str(getattr(covenant_execution, "stable_hash", lambda: "")())
    report_hash = _safe_str(getattr(boundary_audit_report, "stable_hash", lambda: "")())
    rules_hash = _sha256_hex(
        _canonical_json([rule.stable_hash() for rule in policy_rules]).encode("utf-8")
    )
    return action_hash, covenant_hash, report_hash, rules_hash


def build_firewall_execution_receipt(
    *,
    decision_value: str,
    decision_hash: str,
    action_capsule_hash: str,
    covenant_execution_hash: str,
    boundary_audit_report_hash: str,
    policy_rules_hash: str,
) -> FirewallExecutionReceipt:
    preimage = {
        "action_capsule_hash": action_capsule_hash,
        "boundary_audit_report_hash": boundary_audit_report_hash,
        "covenant_execution_hash": covenant_execution_hash,
        "decision_hash": decision_hash,
        "decision_value": decision_value,
        "policy_rules_hash": policy_rules_hash,
    }
    receipt_hash = _sha256_hex(_canonical_json(preimage).encode("utf-8"))
    return FirewallExecutionReceipt(
        decision_hash=decision_hash,
        decision_value=decision_value,
        action_capsule_hash=action_capsule_hash,
        covenant_execution_hash=covenant_execution_hash,
        boundary_audit_report_hash=boundary_audit_report_hash,
        policy_rules_hash=policy_rules_hash,
        receipt_hash=receipt_hash,
    )


def evaluate_policy_execution_firewall(
    action_capsule: Any,
    covenant_execution: Any,
    boundary_audit_report: Any,
    policy_rules: Sequence[FirewallPolicyRule],
) -> PolicyExecutionFirewallDecision:
    reasons: list[FirewallDecisionReason] = []
    violations: list[str] = []

    action_type = _safe_str(getattr(action_capsule, "action_type", "")).strip()
    action_scope = _safe_str(getattr(action_capsule, "action_scope", "")).strip()
    capsule_replay_identity = _safe_str(getattr(action_capsule, "replay_identity", "")).strip()

    covenant_rule_id = _safe_str(getattr(getattr(covenant_execution, "rule", None), "rule_id", "")).strip()
    covenant_replay_identity = _safe_str(
        getattr(getattr(covenant_execution, "receipt", None), "replay_identity", "")
    ).strip()

    within_boundary_value = getattr(boundary_audit_report, "within_boundary", None)
    within_boundary = bool(within_boundary_value) if isinstance(within_boundary_value, bool) else None
    violated_rules = _stable_sorted_unique_text(
        _as_sequence(getattr(boundary_audit_report, "violated_rules", ()))
    )
    severity_summary = getattr(boundary_audit_report, "severity_summary", {})
    if not isinstance(severity_summary, Mapping):
        severity_summary = {}
    continuity_ok_value = getattr(getattr(boundary_audit_report, "audit_receipt", None), "continuity_ok", None)
    continuity_ok = bool(continuity_ok_value) if isinstance(continuity_ok_value, bool) else None

    malformed = False
    if not isinstance(action_capsule, ProofCarryingAgentActionCapsule):
        malformed = True
        violations.append("malformed_capsule_metadata")
        reasons.append(
            _reason(
                reason_code="malformed_input",
                dimension="action_capsule",
                rule_id="system",
                outcome="deny",
                expected="ProofCarryingAgentActionCapsule",
                observed=type(action_capsule).__name__,
                message="malformed capsule metadata",
            )
        )
    if not isinstance(covenant_execution, DeterministicCovenantExecution):
        malformed = True
        violations.append("malformed_covenant_execution")
        reasons.append(
            _reason(
                reason_code="malformed_input",
                dimension="covenant_execution",
                rule_id="system",
                outcome="deny",
                expected="DeterministicCovenantExecution",
                observed=type(covenant_execution).__name__,
                message="malformed covenant execution",
            )
        )
    if not isinstance(boundary_audit_report, GovernanceMemoryIOBoundaryAuditReport):
        malformed = True
        violations.append("malformed_audit_report")
        reasons.append(
            _reason(
                reason_code="malformed_input",
                dimension="boundary_audit_report",
                rule_id="system",
                outcome="deny",
                expected="GovernanceMemoryIOBoundaryAuditReport",
                observed=type(boundary_audit_report).__name__,
                message="malformed audit report",
            )
        )

    for rule in sorted(policy_rules, key=lambda item: (item.rule_id, item.stable_hash())):
        if rule.allowed_action_types:
            if action_type not in rule.allowed_action_types:
                violations.append("disallowed_action_type")
                reasons.append(
                    _reason(
                        reason_code="disallowed_action_type",
                        dimension="action_type",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=rule.allowed_action_types,
                        observed=action_type,
                        message="action type denied by firewall rule",
                    )
                )
            else:
                reasons.append(
                    _reason(
                        reason_code="allowed_action_type",
                        dimension="action_type",
                        rule_id=rule.rule_id,
                        outcome="allow",
                        expected=rule.allowed_action_types,
                        observed=action_type,
                        message="action type allowed by firewall rule",
                    )
                )

        if rule.allowed_action_scopes:
            if action_scope not in rule.allowed_action_scopes:
                violations.append("disallowed_action_scope")
                reasons.append(
                    _reason(
                        reason_code="disallowed_action_scope",
                        dimension="action_scope",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=rule.allowed_action_scopes,
                        observed=action_scope,
                        message="action scope denied by firewall rule",
                    )
                )
            else:
                reasons.append(
                    _reason(
                        reason_code="allowed_action_scope",
                        dimension="action_scope",
                        rule_id=rule.rule_id,
                        outcome="allow",
                        expected=rule.allowed_action_scopes,
                        observed=action_scope,
                        message="action scope allowed by firewall rule",
                    )
                )

        if rule.require_within_boundary:
            if within_boundary is not True:
                violations.append("boundary_failure")
                reasons.append(
                    _reason(
                        reason_code="boundary_failure",
                        dimension="within_boundary",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=True,
                        observed=within_boundary_value,
                        message="audit report must be within boundary",
                    )
                )
            else:
                reasons.append(
                    _reason(
                        reason_code="boundary_pass",
                        dimension="within_boundary",
                        rule_id=rule.rule_id,
                        outcome="allow",
                        expected=True,
                        observed=within_boundary,
                        message="audit report is within boundary",
                    )
                )

        if _contains_severity_at_or_above(severity_summary, rule.max_allowed_severity):
            violations.append("blocked_severity_level")
            reasons.append(
                _reason(
                    reason_code="blocked_severity_level",
                    dimension="severity_summary",
                    rule_id=rule.rule_id,
                    outcome="deny",
                    expected=rule.max_allowed_severity,
                    observed=severity_summary,
                    message="severity exceeds maximum allowed threshold",
                )
            )
        else:
            reasons.append(
                _reason(
                    reason_code="severity_allowed",
                    dimension="severity_summary",
                    rule_id=rule.rule_id,
                    outcome="allow",
                    expected=rule.max_allowed_severity,
                    observed=severity_summary,
                    message="severity is within allowed threshold",
                )
            )

        blocked_rules = tuple(
            sorted(set(violated_rules).intersection(set(rule.disallowed_violated_rule_ids)))
        )
        if blocked_rules:
            violations.append("blocked_violated_rule_id")
            reasons.append(
                _reason(
                    reason_code="blocked_violated_rule_id",
                    dimension="violated_rules",
                    rule_id=rule.rule_id,
                    outcome="deny",
                    expected=rule.disallowed_violated_rule_ids,
                    observed=blocked_rules,
                    message="violated rule id is blocked by firewall policy",
                )
            )

        if rule.require_receipt_continuity:
            if continuity_ok is not True:
                violations.append("receipt_continuity_failure")
                reasons.append(
                    _reason(
                        reason_code="receipt_continuity_failure",
                        dimension="transition_receipt_continuity",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=True,
                        observed=continuity_ok_value,
                        message="transition receipt continuity check failed",
                    )
                )

        prefix = rule.required_replay_identity_prefix
        if prefix:
            replay_candidates = (capsule_replay_identity, covenant_replay_identity)
            if not all(identity.startswith(prefix) for identity in replay_candidates):
                violations.append("replay_identity_mismatch")
                reasons.append(
                    _reason(
                        reason_code="replay_identity_mismatch",
                        dimension="replay_identity",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=prefix,
                        observed={
                            "capsule": capsule_replay_identity,
                            "covenant": covenant_replay_identity,
                        },
                        message="replay identity prefix mismatch",
                    )
                )

        if rule.required_covenant_rule_id:
            if covenant_rule_id != rule.required_covenant_rule_id:
                violations.append("covenant_rule_id_mismatch")
                reasons.append(
                    _reason(
                        reason_code="covenant_rule_id_mismatch",
                        dimension="covenant_rule_id",
                        rule_id=rule.rule_id,
                        outcome="deny",
                        expected=rule.required_covenant_rule_id,
                        observed=covenant_rule_id,
                        message="covenant rule id does not match firewall policy",
                    )
                )
            else:
                reasons.append(
                    _reason(
                        reason_code="covenant_rule_id_match",
                        dimension="covenant_rule_id",
                        rule_id=rule.rule_id,
                        outcome="allow",
                        expected=rule.required_covenant_rule_id,
                        observed=covenant_rule_id,
                        message="covenant rule id matches firewall policy",
                    )
                )

    if malformed and "malformed_audit_report" not in violations:
        violations.append("malformed_audit_report")

    decision_value = "deny" if violations else "allow"
    unique_violations = tuple(sorted(set(violations)))
    ordered_reasons = _sorted_reasons(reasons)

    decision_body = {
        "decision_value": decision_value,
        "reasons": [reason.to_dict() for reason in ordered_reasons],
        "violations": list(unique_violations),
    }
    decision_hash = _sha256_hex(_canonical_json(decision_body).encode("utf-8"))

    action_hash, covenant_hash, report_hash, rules_hash = _hash_inputs(
        action_capsule,
        covenant_execution,
        boundary_audit_report,
        tuple(policy_rules),
    )
    receipt = build_firewall_execution_receipt(
        decision_value=decision_value,
        decision_hash=decision_hash,
        action_capsule_hash=action_hash,
        covenant_execution_hash=covenant_hash,
        boundary_audit_report_hash=report_hash,
        policy_rules_hash=rules_hash,
    )

    return PolicyExecutionFirewallDecision(
        decision_value=decision_value,
        reasons=ordered_reasons,
        violations=unique_violations,
        decision_hash=decision_hash,
        execution_receipt=receipt,
    )


def validate_firewall_decision(decision: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        if not isinstance(decision, PolicyExecutionFirewallDecision):
            return ("malformed_firewall_decision",)

        if decision.decision_value not in _ALLOWED_DECISIONS:
            violations.append("invalid_decision_value")

        expected_body = {
            "decision_value": decision.decision_value,
            "reasons": [reason.to_dict() for reason in decision.reasons],
            "violations": list(decision.violations),
        }
        expected_decision_hash = _sha256_hex(_canonical_json(expected_body).encode("utf-8"))
        if decision.decision_hash != expected_decision_hash:
            violations.append("decision_hash_mismatch")

        receipt = decision.execution_receipt
        expected_receipt = build_firewall_execution_receipt(
            decision_value=receipt.decision_value,
            decision_hash=receipt.decision_hash,
            action_capsule_hash=receipt.action_capsule_hash,
            covenant_execution_hash=receipt.covenant_execution_hash,
            boundary_audit_report_hash=receipt.boundary_audit_report_hash,
            policy_rules_hash=receipt.policy_rules_hash,
        )
        if receipt.receipt_hash != expected_receipt.receipt_hash:
            violations.append("receipt_hash_mismatch")
        if receipt.decision_hash != decision.decision_hash:
            violations.append("receipt_decision_hash_mismatch")
        if receipt.decision_value != decision.decision_value:
            violations.append("receipt_decision_value_mismatch")
        if decision.decision_value == "allow" and decision.violations:
            violations.append("allow_with_violations")
        if decision.decision_value == "deny" and not decision.violations:
            violations.append("deny_without_violations")
    except Exception as exc:  # pragma: no cover
        return (f"validator_error:{type(exc).__name__}",)

    return tuple(sorted(set(violations)))


def compare_firewall_replay(
    decision_a: PolicyExecutionFirewallDecision,
    decision_b: PolicyExecutionFirewallDecision,
) -> Dict[str, Any]:
    fields = (
        "decision_value",
        "decision_hash",
        "violations",
    )
    mismatches: list[str] = []
    for field in fields:
        if getattr(decision_a, field) != getattr(decision_b, field):
            mismatches.append(field)

    if decision_a.execution_receipt.receipt_hash != decision_b.execution_receipt.receipt_hash:
        mismatches.append("execution_receipt.receipt_hash")

    reasons_a = tuple(reason.stable_hash() for reason in decision_a.reasons)
    reasons_b = tuple(reason.stable_hash() for reason in decision_b.reasons)
    if reasons_a != reasons_b:
        mismatches.append("reasons")

    return {
        "match": len(mismatches) == 0,
        "mismatch_fields": tuple(mismatches),
        "decision_a_hash": decision_a.decision_hash,
        "decision_b_hash": decision_b.decision_hash,
    }
