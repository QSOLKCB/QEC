"""Deterministic Layer-4 stability policy lattice.

Theory invariants preserved by this module:

- POLICY_STATE_LATTICE_LAW:
  Runtime policy transitions follow an explicit ranked lattice ordering.
- DETERMINISTIC_VIOLATION_DETECTION:
  Violation classification is rule-based and order-stable.
- BOUNDED_ACTION_RISK_INVARIANT:
  Action risk is finite-validated, clamped to [0, 1], and precision-stable.
- REPLAY_SAFE_POLICY_AUDIT_CHAIN:
  Policy decisions are recorded in an append-only parent-linked SHA-256 chain.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

_POLICY_ORDER: tuple[str, ...] = ("allow", "observe", "defer", "throttle", "deny")
_POLICY_RANK: dict[str, int] = {state: idx for idx, state in enumerate(_POLICY_ORDER)}
_ALLOWED_VIOLATIONS: tuple[str, ...] = (
    "none",
    "quota_violation",
    "dependency_violation",
    "stability_violation",
    "repeated_failure",
    "unknown_action",
)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_state(state: Any, field: str = "state") -> str:
    if not isinstance(state, str):
        raise ValueError(f"{field} must be str")
    normalized = " ".join(state.strip().split())
    if normalized not in _POLICY_RANK:
        raise ValueError(f"unknown {field}: {state!r}")
    return normalized


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return _clamp01(out)


@dataclass(frozen=True)
class PolicyNode:
    state_id: str
    state_label: str
    rank: int
    allowed_transitions: tuple[str, ...]
    bounded: bool
    node_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "state_label": self.state_label,
            "rank": self.rank,
            "allowed_transitions": list(self.allowed_transitions),
            "bounded": self.bounded,
            "node_hash": self.node_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class StabilityPolicyGraph:
    nodes: tuple[PolicyNode, ...]
    edges: tuple[tuple[str, str], ...]
    graph_hash: str
    graph_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [list(e) for e in self.edges],
            "graph_hash": self.graph_hash,
            "graph_valid": self.graph_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyViolationRecord:
    action_id: str
    violation_kind: str
    severity: float
    deterministic: bool
    violation_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "violation_kind": self.violation_kind,
            "severity": self.severity,
            "deterministic": self.deterministic,
            "violation_hash": self.violation_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyRiskAssessment:
    action_risk_score: float
    policy_state: str
    violation_detected: bool
    bounded_score: bool
    assessment_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_risk_score": self.action_risk_score,
            "policy_state": self.policy_state,
            "violation_detected": self.violation_detected,
            "bounded_score": self.bounded_score,
            "assessment_hash": self.assessment_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyAuditEntry:
    sequence_id: int
    action_id: str
    prior_state: str
    next_state: str
    parent_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "action_id": self.action_id,
            "prior_state": self.prior_state,
            "next_state": self.next_state,
            "parent_hash": self.parent_hash,
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyAuditTrail:
    entries: tuple[PolicyAuditEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyTransitionReport:
    total_actions: int
    violations: int
    highest_risk: float
    stable: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_actions": self.total_actions,
            "violations": self.violations,
            "highest_risk": self.highest_risk,
            "stable": self.stable,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def build_stability_policy_graph(
    transitions: Mapping[str, Iterable[str]] | Iterable[tuple[str, Iterable[str]]] | None = None,
) -> StabilityPolicyGraph:
    raw = list((transitions.items() if isinstance(transitions, Mapping) else transitions) or ())
    if not raw:
        raw = [
            ("allow", ("observe", "defer", "throttle", "deny")),
            ("observe", ("defer", "throttle", "deny")),
            ("defer", ("throttle", "deny")),
            ("throttle", ("deny",)),
            ("deny", ()),
        ]

    seen: set[str] = set()
    cleaned: dict[str, tuple[str, ...]] = {}
    for state_raw, allowed_raw in raw:
        state = _canonical_state(state_raw, "state_id")
        if state in seen:
            raise ValueError(f"duplicate state_id: {state}")
        seen.add(state)
        allowed = tuple(sorted({_canonical_state(s, "transition") for s in allowed_raw}, key=_POLICY_RANK.get))
        cleaned[state] = allowed

    if set(cleaned.keys()) != set(_POLICY_ORDER):
        raise ValueError("policy lattice must define canonical states")

    nodes: list[PolicyNode] = []
    edges: list[tuple[str, str]] = []
    for state in _POLICY_ORDER:
        rank = _POLICY_RANK[state]
        allowed = cleaned[state]
        for nxt in allowed:
            if _POLICY_RANK[nxt] <= rank:
                raise ValueError("invalid transition violates monotonic escalation")
            edges.append((state, nxt))
        node_payload = {
            "state_id": state,
            "state_label": state,
            "rank": rank,
            "allowed_transitions": list(allowed),
            "bounded": True,
        }
        nodes.append(
            PolicyNode(
                state_id=state,
                state_label=state,
                rank=rank,
                allowed_transitions=allowed,
                bounded=True,
                node_hash=_hash_sha256(node_payload),
            )
        )

    edges_sorted = tuple(sorted(edges, key=lambda e: (_POLICY_RANK[e[0]], _POLICY_RANK[e[1]], e[0], e[1])))
    graph_payload = {
        "nodes": [n.to_dict() for n in nodes],
        "edges": [list(e) for e in edges_sorted],
        "graph_valid": True,
    }
    graph = StabilityPolicyGraph(
        nodes=tuple(nodes),
        edges=edges_sorted,
        graph_hash=_hash_sha256(graph_payload),
        graph_valid=True,
    )
    validate_stability_policy_graph(graph)
    return graph


def validate_stability_policy_graph(graph: StabilityPolicyGraph) -> bool:
    state_ids = tuple(node.state_id for node in graph.nodes)
    if len(state_ids) != len(set(state_ids)):
        raise ValueError("duplicate state_id")
    if state_ids != _POLICY_ORDER:
        raise ValueError("nodes must follow canonical ordering")

    node_lookup = {node.state_id: node for node in graph.nodes}
    recomputed_edges: list[tuple[str, str]] = []
    for node in graph.nodes:
        if node.rank != _POLICY_RANK[node.state_id]:
            raise ValueError("monotonic rank ordering violated")
        payload = {
            "state_id": node.state_id,
            "state_label": node.state_label,
            "rank": node.rank,
            "allowed_transitions": list(node.allowed_transitions),
            "bounded": node.bounded,
        }
        if node.node_hash != _hash_sha256(payload):
            raise ValueError("node hash instability")
        for nxt in node.allowed_transitions:
            if nxt not in node_lookup:
                raise ValueError("invalid transitions")
            if _POLICY_RANK[nxt] <= node.rank:
                raise ValueError("invalid transitions")
            recomputed_edges.append((node.state_id, nxt))

    expected_edges = tuple(sorted(recomputed_edges, key=lambda e: (_POLICY_RANK[e[0]], _POLICY_RANK[e[1]], e[0], e[1])))
    if graph.edges != expected_edges:
        raise ValueError("edge validity failed")

    payload = {
        "nodes": [n.to_dict() for n in graph.nodes],
        "edges": [list(e) for e in graph.edges],
        "graph_valid": True,
    }
    if graph.graph_hash != _hash_sha256(payload):
        raise ValueError("graph hash stability failed")
    if graph.graph_valid is not True:
        raise ValueError("contradictory graph_valid flag")
    return True


def detect_policy_violation(
    action_metadata: Mapping[str, Any],
    workflow_state: Mapping[str, Any],
    prior_risk_signals: Mapping[str, Any] | None = None,
) -> PolicyViolationRecord:
    signals = dict(prior_risk_signals or {})
    action_id = str(action_metadata.get("action_id", "")).strip() or "unknown_action"

    known_action = bool(action_metadata.get("known_action", action_id != "unknown_action"))
    quota_remaining = float(action_metadata.get("quota_remaining", signals.get("quota_remaining", 1.0)))
    unmet_dependencies = int(action_metadata.get("unmet_dependencies", 0))
    dependencies_satisfied = bool(action_metadata.get("dependencies_satisfied", unmet_dependencies == 0))
    workflow_instability = _bounded_float(workflow_state.get("workflow_instability", 0.0), "workflow_instability")
    repeated_failures = int(signals.get("repeated_failures", 0))
    failure_pressure = _bounded_float(signals.get("failure_pressure", 0.0), "failure_pressure")

    if not known_action:
        kind = "unknown_action"
    elif bool(action_metadata.get("quota_exceeded", False)) or quota_remaining <= 0.0:
        kind = "quota_violation"
    elif (not dependencies_satisfied) or unmet_dependencies > 0:
        kind = "dependency_violation"
    elif workflow_instability >= 0.8:
        kind = "stability_violation"
    elif repeated_failures >= 3 or failure_pressure >= 0.7:
        kind = "repeated_failure"
    else:
        kind = "none"

    severity_map = {
        "none": 0.0,
        "quota_violation": 0.55,
        "dependency_violation": 0.65,
        "stability_violation": 0.75,
        "repeated_failure": 0.85,
        "unknown_action": 1.0,
    }
    severity = round(severity_map[kind], 12)
    payload = {
        "action_id": action_id,
        "violation_kind": kind,
        "severity": severity,
        "deterministic": True,
    }
    return PolicyViolationRecord(
        action_id=action_id,
        violation_kind=kind,
        severity=severity,
        deterministic=True,
        violation_hash=_hash_sha256(payload),
    )


def compute_bounded_action_risk_score(
    *,
    workflow_instability: float,
    blocked_ratio: float,
    failure_pressure: float,
    violation_severity: float,
    dependency_pressure: float = 0.0,
) -> float:
    w_instability = _bounded_float(workflow_instability, "workflow_instability")
    w_blocked = _bounded_float(blocked_ratio, "blocked_ratio")
    w_failure = _bounded_float(failure_pressure, "failure_pressure")
    w_violation = _bounded_float(violation_severity, "violation_severity")
    w_dependency = _bounded_float(dependency_pressure, "dependency_pressure")

    score = (
        0.30 * w_instability
        + 0.25 * w_blocked
        + 0.20 * w_failure
        + 0.15 * w_violation
        + 0.10 * w_dependency
    )
    return round(_clamp01(score), 12)


def evaluate_policy_state_transition(
    prior_state: str,
    action_risk_score: float,
    violation_record: PolicyViolationRecord,
    graph: StabilityPolicyGraph | None = None,
) -> str:
    policy_graph = graph or build_stability_policy_graph()
    validate_stability_policy_graph(policy_graph)

    prior = _canonical_state(prior_state, "prior_state")
    risk = _bounded_float(action_risk_score, "action_risk_score")
    if violation_record.violation_kind not in _ALLOWED_VIOLATIONS:
        raise ValueError("unknown violation_kind")

    if violation_record.violation_kind == "unknown_action":
        return "deny"

    if risk < 0.2:
        base = "allow"
    elif risk < 0.4:
        base = "observe"
    elif risk < 0.6:
        base = "defer"
    elif risk < 0.8:
        base = "throttle"
    else:
        base = "deny"

    bump = 0
    if violation_record.severity >= 0.9:
        bump = 2
    elif violation_record.severity >= 0.6:
        bump = 1

    final_rank = min(4, max(_POLICY_RANK[prior], _POLICY_RANK[base]) + bump)
    next_state = _POLICY_ORDER[final_rank]

    transitions = {node.state_id: node.allowed_transitions for node in policy_graph.nodes}
    if next_state != prior and next_state not in transitions[prior]:
        return "deny"
    return next_state


def empty_policy_audit_trail() -> PolicyAuditTrail:
    return PolicyAuditTrail(entries=(), head_hash="", chain_valid=True)


def validate_policy_audit_trail(trail: PolicyAuditTrail) -> bool:
    expected_parent = ""
    expected_head = ""
    for idx, entry in enumerate(trail.entries):
        if entry.sequence_id != idx:
            raise ValueError("sequence continuity violation")
        if entry.parent_hash != expected_parent:
            raise ValueError("parent linkage violation")

        payload = {
            "sequence_id": entry.sequence_id,
            "action_id": entry.action_id,
            "prior_state": entry.prior_state,
            "next_state": entry.next_state,
            "parent_hash": entry.parent_hash,
        }
        expected_hash = _hash_sha256(payload)
        if entry.entry_hash != expected_hash:
            raise ValueError("corrupted audit entry hash")

        expected_parent = entry.entry_hash
        expected_head = entry.entry_hash

    if trail.head_hash != expected_head:
        raise ValueError("head hash mismatch")
    if trail.chain_valid is not True:
        raise ValueError("contradictory chain_valid flag")
    return True


def append_policy_audit_entry(
    trail: PolicyAuditTrail,
    *,
    action_id: str,
    prior_state: str,
    next_state: str,
) -> PolicyAuditTrail:
    validate_policy_audit_trail(trail)

    action = str(action_id).strip()
    if not action:
        raise ValueError("action_id must be non-empty")
    prior = _canonical_state(prior_state, "prior_state")
    nxt = _canonical_state(next_state, "next_state")

    sequence_id = len(trail.entries)
    parent_hash = trail.head_hash
    payload = {
        "sequence_id": sequence_id,
        "action_id": action,
        "prior_state": prior,
        "next_state": nxt,
        "parent_hash": parent_hash,
    }
    entry = PolicyAuditEntry(
        sequence_id=sequence_id,
        action_id=action,
        prior_state=prior,
        next_state=nxt,
        parent_hash=parent_hash,
        entry_hash=_hash_sha256(payload),
    )
    return PolicyAuditTrail(
        entries=trail.entries + (entry,),
        head_hash=entry.entry_hash,
        chain_valid=True,
    )


def run_stability_policy_lattice(
    action_candidate: Mapping[str, Any],
    workflow_state: Mapping[str, Any],
    prior_audit_trail: PolicyAuditTrail | None = None,
    *,
    prior_state: str = "allow",
    prior_risk_signals: Mapping[str, Any] | None = None,
) -> tuple[PolicyRiskAssessment, PolicyViolationRecord, PolicyTransitionReport, PolicyAuditTrail]:
    graph = build_stability_policy_graph()
    validate_stability_policy_graph(graph)

    violation = detect_policy_violation(action_candidate, workflow_state, prior_risk_signals)
    risk = compute_bounded_action_risk_score(
        workflow_instability=workflow_state.get("workflow_instability", 0.0),
        blocked_ratio=workflow_state.get("blocked_ratio", 0.0),
        failure_pressure=(prior_risk_signals or {}).get("failure_pressure", 0.0),
        violation_severity=violation.severity,
        dependency_pressure=workflow_state.get("dependency_pressure", 0.0),
    )
    next_state = evaluate_policy_state_transition(prior_state, risk, violation, graph=graph)

    assessment_payload = {
        "action_risk_score": risk,
        "policy_state": next_state,
        "violation_detected": violation.violation_kind != "none",
        "bounded_score": 0.0 <= risk <= 1.0,
    }
    assessment = PolicyRiskAssessment(
        action_risk_score=risk,
        policy_state=next_state,
        violation_detected=violation.violation_kind != "none",
        bounded_score=0.0 <= risk <= 1.0,
        assessment_hash=_hash_sha256(assessment_payload),
    )

    trail = prior_audit_trail if prior_audit_trail is not None else empty_policy_audit_trail()
    updated_trail = append_policy_audit_entry(
        trail,
        action_id=violation.action_id,
        prior_state=prior_state,
        next_state=next_state,
    )

    prior_violations = int(workflow_state.get("prior_violations", 0))
    violations = prior_violations + (1 if violation.violation_kind != "none" else 0)
    highest_risk = round(
        max(risk, _bounded_float(workflow_state.get("highest_risk_so_far", 0.0), "highest_risk_so_far")),
        12,
    )
    stable = next_state in ("allow", "observe") and violation.violation_kind == "none"
    report_payload = {
        "total_actions": len(updated_trail.entries),
        "violations": violations,
        "highest_risk": highest_risk,
        "stable": stable,
    }
    report = PolicyTransitionReport(
        total_actions=len(updated_trail.entries),
        violations=violations,
        highest_risk=highest_risk,
        stable=stable,
        report_hash=_hash_sha256(report_payload),
    )
    return assessment, violation, report, updated_trail


__all__ = [
    "PolicyNode",
    "StabilityPolicyGraph",
    "PolicyViolationRecord",
    "PolicyRiskAssessment",
    "PolicyAuditEntry",
    "PolicyAuditTrail",
    "PolicyTransitionReport",
    "build_stability_policy_graph",
    "validate_stability_policy_graph",
    "detect_policy_violation",
    "compute_bounded_action_risk_score",
    "evaluate_policy_state_transition",
    "empty_policy_audit_trail",
    "append_policy_audit_entry",
    "validate_policy_audit_trail",
    "run_stability_policy_lattice",
]
