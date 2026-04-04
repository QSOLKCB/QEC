from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


_GOVERNANCE_PRECISION = 12


def _round(value: float) -> float:
    return float(round(float(value), _GOVERNANCE_PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _norm_token(value: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("empty token is not allowed")
    return token


def _norm_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(_norm_token(v) for v in values))


def _norm_metadata(values: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized = tuple(sorted((_norm_token(k), _norm_token(v)) for k, v in values))
    return normalized


class PolicyState(str, Enum):
    """Explicit governance lattice order for GOVERNANCE_STATE_LATTICE."""

    DENY = "DENY"
    ESCALATE = "ESCALATE"
    SANDBOX = "SANDBOX"
    OBSERVE = "OBSERVE"
    ALLOW = "ALLOW"


@dataclass(frozen=True)
class GovernanceActionRequest:
    action_id: str
    action_kind: str
    tool_name: str
    target_scope: str
    capability_tags: tuple[str, ...]
    risk_signals: tuple[str, ...]
    replay_context_hash: str
    provenance_hash: str
    metadata: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_kind": self.action_kind,
            "tool_name": self.tool_name,
            "target_scope": self.target_scope,
            "capability_tags": list(self.capability_tags),
            "risk_signals": list(self.risk_signals),
            "replay_context_hash": self.replay_context_hash,
            "provenance_hash": self.provenance_hash,
            "metadata": [[k, v] for k, v in self.metadata],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    action_kind: str
    tool_name: str
    scope_prefix: str
    required_capabilities: tuple[str, ...]
    forbidden_capabilities: tuple[str, ...]
    max_risk_score: float
    decision: PolicyState
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "action_kind": self.action_kind,
            "tool_name": self.tool_name,
            "scope_prefix": self.scope_prefix,
            "required_capabilities": list(self.required_capabilities),
            "forbidden_capabilities": list(self.forbidden_capabilities),
            "max_risk_score": _round(self.max_risk_score),
            "decision": self.decision.value,
            "priority": int(self.priority),
        }


@dataclass(frozen=True)
class PolicyLattice:
    """Deterministic order and operations for GOVERNANCE_STATE_LATTICE."""

    order: tuple[PolicyState, ...] = (
        PolicyState.DENY,
        PolicyState.ESCALATE,
        PolicyState.SANDBOX,
        PolicyState.OBSERVE,
        PolicyState.ALLOW,
    )

    def rank(self, state: PolicyState) -> int:
        return self.order.index(state)

    def join(self, left: PolicyState, right: PolicyState) -> PolicyState:
        return self.order[max(self.rank(left), self.rank(right))]

    def meet(self, left: PolicyState, right: PolicyState) -> PolicyState:
        return self.order[min(self.rank(left), self.rank(right))]


@dataclass(frozen=True)
class ActionPermissionGraph:
    action_to_tools: tuple[tuple[str, tuple[str, ...]], ...]
    tool_to_scopes: tuple[tuple[str, tuple[str, ...]], ...]
    action_required_capabilities: tuple[tuple[str, tuple[str, ...]], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_to_tools": [[action, list(tools)] for action, tools in self.action_to_tools],
            "tool_to_scopes": [[tool, list(scopes)] for tool, scopes in self.tool_to_scopes],
            "action_required_capabilities": [
                [action, list(caps)] for action, caps in self.action_required_capabilities
            ],
        }


def _pair_lookup(table: tuple[tuple[str, tuple[str, ...]], ...], key: str) -> tuple[str, ...] | None:
    for item_key, item_values in table:
        if item_key == key:
            return item_values
    return None


@dataclass(frozen=True)
class GovernanceDecision:
    decision: PolicyState
    matched_rule_ids: tuple[str, ...]
    effective_policy_state: PolicyState
    risk_score: float
    allowed: bool
    replay_safe: bool
    denial_reason: str | None
    parent_ledger_hash: str | None
    decision_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "matched_rule_ids": list(self.matched_rule_ids),
            "effective_policy_state": self.effective_policy_state.value,
            "risk_score": _round(self.risk_score),
            "allowed": bool(self.allowed),
            "replay_safe": bool(self.replay_safe),
            "denial_reason": self.denial_reason,
            "parent_ledger_hash": self.parent_ledger_hash,
            "decision_hash": self.decision_hash,
        }


@dataclass(frozen=True)
class GovernanceLedgerEntry:
    sequence_id: int
    action_id: str
    replay_context_hash: str
    decision_hash: str
    parent_hash: str
    rule_ids: tuple[str, ...]
    risk_score: float
    decision: PolicyState

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": int(self.sequence_id),
            "action_id": self.action_id,
            "replay_context_hash": self.replay_context_hash,
            "decision_hash": self.decision_hash,
            "parent_hash": self.parent_hash,
            "rule_ids": list(self.rule_ids),
            "risk_score": _round(self.risk_score),
            "decision": self.decision.value,
        }

    def entry_hash(self) -> str:
        return _stable_sha256(self.to_dict())


@dataclass(frozen=True)
class GovernanceLedger:
    entries: tuple[GovernanceLedgerEntry, ...]
    head_hash: str
    chain_valid: bool
    governance_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": bool(self.chain_valid),
            "governance_only": bool(self.governance_only),
        }


def build_policy_lattice(order: Sequence[PolicyState] | None = None) -> PolicyLattice:
    if order is None:
        return PolicyLattice()
    ordered = tuple(order)
    if tuple(sorted(ordered, key=lambda x: x.value)) != tuple(sorted(PolicyState, key=lambda x: x.value)):
        raise ValueError("policy lattice must contain each policy state exactly once")
    return PolicyLattice(order=ordered)


def _normalize_graph_pairs(mapping: Mapping[str, Sequence[str]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    items = []
    for key, values in mapping.items():
        items.append((_norm_token(key), _norm_tuple(values)))
    return tuple(sorted(items, key=lambda pair: pair[0]))


def build_action_permission_graph(
    action_to_tools: Mapping[str, Sequence[str]],
    tool_to_scopes: Mapping[str, Sequence[str]],
    action_required_capabilities: Mapping[str, Sequence[str]] | None = None,
) -> ActionPermissionGraph:
    return ActionPermissionGraph(
        action_to_tools=_normalize_graph_pairs(action_to_tools),
        tool_to_scopes=_normalize_graph_pairs(tool_to_scopes),
        action_required_capabilities=_normalize_graph_pairs(action_required_capabilities or {}),
    )


def normalize_governance_action_request(request: GovernanceActionRequest) -> GovernanceActionRequest:
    normalized = GovernanceActionRequest(
        action_id=_norm_token(request.action_id),
        action_kind=_norm_token(request.action_kind),
        tool_name=_norm_token(request.tool_name),
        target_scope=_norm_token(request.target_scope),
        capability_tags=_norm_tuple(request.capability_tags),
        risk_signals=_norm_tuple(request.risk_signals),
        replay_context_hash=str(request.replay_context_hash).strip(),
        provenance_hash=str(request.provenance_hash).strip(),
        metadata=_norm_metadata(request.metadata),
    )
    if not normalized.replay_context_hash:
        raise ValueError("replay_context_hash is required")
    if not normalized.provenance_hash:
        raise ValueError("provenance_hash is required")
    return normalized


_RISK_WEIGHTS: dict[str, float] = {
    "unknown_tool": 0.25,
    "unknown_action_kind": 0.25,
    "privileged_scope": 0.2,
    "forbidden_capability": 0.25,
    "missing_replay_context": 0.2,
    "missing_provenance": 0.15,
    "high_impact_action": 0.2,
    "cross_scope_request": 0.1,
}


def compute_bounded_risk_score(signals: Sequence[str]) -> float:
    risk = 0.0
    for signal in tuple(sorted(_norm_token(signal) for signal in signals)):
        risk += _RISK_WEIGHTS.get(signal, 0.0)
    return _round(max(0.0, min(1.0, risk)))


def _scope_matches(prefix: str, target_scope: str) -> bool:
    return prefix == "*" or target_scope.startswith(prefix)


def _match_rules(
    request: GovernanceActionRequest,
    rules: Sequence[PolicyRule],
) -> tuple[PolicyRule, ...]:
    matched = []
    for rule in rules:
        if rule.action_kind != "*" and rule.action_kind != request.action_kind:
            continue
        if rule.tool_name != "*" and rule.tool_name != request.tool_name:
            continue
        if not _scope_matches(rule.scope_prefix, request.target_scope):
            continue
        if any(cap in request.capability_tags for cap in rule.forbidden_capabilities):
            matched.append(rule)
            continue
        if any(cap not in request.capability_tags for cap in rule.required_capabilities):
            continue
        matched.append(rule)
    return tuple(
        sorted(
            matched,
            key=lambda r: (-int(r.priority), r.rule_id, PolicyLattice().rank(r.decision)),
        )
    )


def evaluate_governance_decision(
    request: GovernanceActionRequest,
    policy_lattice: PolicyLattice,
    permission_graph: ActionPermissionGraph,
    policy_rules: Sequence[PolicyRule],
    parent_ledger_hash: str | None,
) -> GovernanceDecision:
    """POLICY_EXECUTION_FENCE deterministic boundary for pre-execution governance."""

    normalized = normalize_governance_action_request(request)

    risk_signals = list(normalized.risk_signals)
    tools = _pair_lookup(permission_graph.action_to_tools, normalized.action_kind)
    if tools is None:
        risk_signals.append("unknown_action_kind")
    elif normalized.tool_name not in tools:
        risk_signals.append("unknown_tool")

    scopes = _pair_lookup(permission_graph.tool_to_scopes, normalized.tool_name)
    if scopes is None or not any(_scope_matches(scope, normalized.target_scope) for scope in scopes):
        risk_signals.append("cross_scope_request")

    required_caps = _pair_lookup(permission_graph.action_required_capabilities, normalized.action_kind) or ()
    if any(cap not in normalized.capability_tags for cap in required_caps):
        risk_signals.append("forbidden_capability")

    risk_score = compute_bounded_risk_score(risk_signals)

    matched_rules = _match_rules(normalized, policy_rules)
    matched_ids = tuple(rule.rule_id for rule in matched_rules)

    states: list[PolicyState] = []
    denial_reason: str | None = None

    if tools is None or normalized.tool_name not in (tools or ()):
        states.append(PolicyState.DENY)
        denial_reason = "unknown action-to-tool edge"
    elif scopes is None or not any(_scope_matches(scope, normalized.target_scope) for scope in scopes):
        states.append(PolicyState.DENY)
        denial_reason = "scope not permitted"

    if any(cap not in normalized.capability_tags for cap in required_caps):
        states.append(PolicyState.ESCALATE)
        denial_reason = denial_reason or "missing required capabilities"

    for rule in matched_rules:
        if risk_score > _round(rule.max_risk_score):
            states.append(PolicyState.ESCALATE)
            denial_reason = denial_reason or "risk exceeds policy"
        states.append(rule.decision)

    if not states:
        states.append(PolicyState.DENY)
        denial_reason = "deny-by-default"

    effective_state = states[0]
    for state in states[1:]:
        effective_state = policy_lattice.meet(effective_state, state)

    allowed = effective_state == PolicyState.ALLOW
    replay_safe = bool(normalized.replay_context_hash and normalized.provenance_hash)

    decision_payload = {
        "decision": effective_state.value,
        "matched_rule_ids": list(matched_ids),
        "effective_policy_state": effective_state.value,
        "risk_score": _round(risk_score),
        "allowed": bool(allowed),
        "replay_safe": bool(replay_safe),
        "denial_reason": denial_reason,
        "parent_ledger_hash": parent_ledger_hash,
    }
    decision_hash = _stable_sha256(decision_payload)

    return GovernanceDecision(
        decision=effective_state,
        matched_rule_ids=matched_ids,
        effective_policy_state=effective_state,
        risk_score=risk_score,
        allowed=allowed,
        replay_safe=replay_safe,
        denial_reason=denial_reason,
        parent_ledger_hash=parent_ledger_hash,
        decision_hash=decision_hash,
    )


def _compute_ledger_head(entries: Sequence[GovernanceLedgerEntry]) -> str:
    if not entries:
        return "0" * 64
    return entries[-1].entry_hash()


def append_governance_ledger_entry(
    ledger: GovernanceLedger,
    request: GovernanceActionRequest,
    decision: GovernanceDecision,
    parent_hash: str,
) -> GovernanceLedger:
    if parent_hash != ledger.head_hash:
        raise ValueError("parent hash mismatch")

    entry = GovernanceLedgerEntry(
        sequence_id=len(ledger.entries),
        action_id=request.action_id,
        replay_context_hash=request.replay_context_hash,
        decision_hash=decision.decision_hash,
        parent_hash=parent_hash,
        rule_ids=decision.matched_rule_ids,
        risk_score=decision.risk_score,
        decision=decision.decision,
    )
    entries = ledger.entries + (entry,)
    updated = GovernanceLedger(entries=entries, head_hash=_compute_ledger_head(entries), chain_valid=True, governance_only=True)
    validate_governance_ledger(updated)
    return updated


def validate_governance_ledger(ledger: GovernanceLedger) -> bool:
    expected_parent = "0" * 64
    for index, entry in enumerate(ledger.entries):
        if entry.sequence_id != index:
            raise ValueError("ledger sequence mismatch")
        if entry.parent_hash != expected_parent:
            raise ValueError("ledger parent hash mismatch")
        expected_parent = entry.entry_hash()
    if ledger.head_hash != expected_parent:
        raise ValueError("ledger head hash mismatch")
    return True


def run_agent_governance_fence(
    request: GovernanceActionRequest,
    policy_lattice: PolicyLattice,
    permission_graph: ActionPermissionGraph,
    policy_rules: Sequence[PolicyRule],
    prior_ledger: GovernanceLedger,
) -> tuple[GovernanceDecision, GovernanceLedger]:
    """DETERMINISTIC_ACTION_BOUNDARY + REPLAY_SAFE_PERMISSION_CHAIN orchestration."""

    validate_governance_ledger(prior_ledger)
    decision = evaluate_governance_decision(
        request=request,
        policy_lattice=policy_lattice,
        permission_graph=permission_graph,
        policy_rules=policy_rules,
        parent_ledger_hash=prior_ledger.head_hash,
    )
    updated_ledger = append_governance_ledger_entry(
        ledger=prior_ledger,
        request=normalize_governance_action_request(request),
        decision=decision,
        parent_hash=prior_ledger.head_hash,
    )
    return decision, updated_ledger


def empty_governance_ledger() -> GovernanceLedger:
    return GovernanceLedger(entries=(), head_hash="0" * 64, chain_valid=True, governance_only=True)
