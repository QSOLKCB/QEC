"""v149.0 — Deterministic multi-agent governance arbitration."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any, Final, Sequence

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

MULTI_AGENT_GOVERNANCE_SCHEMA_VERSION: Final[str] = "1.0"
MULTI_AGENT_GOVERNANCE_MODULE_VERSION: Final[str] = "v149.0"

_ALLOWED_RECOMMENDATIONS: Final[tuple[str, ...]] = (
    "MAINTAIN_POLICY",
    "TIGHTEN_POLICY",
    "RELAX_POLICY",
    "ESCALATE_GOVERNANCE",
    "REVIEW_MEMORY",
)
_ALLOWED_STATUSES: Final[tuple[str, ...]] = (
    "CONSENSUS",
    "PRIORITY_SELECTED",
    "CONFIDENCE_SELECTED",
    "ESCALATED_CONFLICT",
    "EMPTY",
)

_RECOMMENDATION_SEVERITY: Final[dict[str, int]] = {
    "RELAX_POLICY": 0,
    "MAINTAIN_POLICY": 1,
    "TIGHTEN_POLICY": 2,
    "REVIEW_MEMORY": 3,
    "ESCALATE_GOVERNANCE": 4,
}


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round_public_metric(value: float) -> float:
    return float(round(float(value), 12))


def _require_canonical_token(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty canonical string")
    token = value.strip()
    if not token or token != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return token


def _require_sha256_hex(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid SHA-256 hex") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    return value


@dataclass(frozen=True)
class GovernanceAgentState:
    agent_id: str
    control_loop_id: str
    memory_hash: str
    governance_hash: str
    recommendation: str
    confidence: float
    priority: int
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.agent_id, name="agent_id")
        _require_canonical_token(self.control_loop_id, name="control_loop_id")
        _require_sha256_hex(self.memory_hash, name="memory_hash")
        _require_sha256_hex(self.governance_hash, name="governance_hash")
        if self.recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("recommendation must be a supported governance label")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be bounded [0.0, 1.0]")
        confidence = float(self.confidence)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("confidence must be bounded [0.0, 1.0]")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or self.priority < 0:
            raise ValueError("priority must be a non-negative integer")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "agent_id": self.agent_id,
            "control_loop_id": self.control_loop_id,
            "memory_hash": self.memory_hash,
            "governance_hash": self.governance_hash,
            "recommendation": self.recommendation,
            "confidence": float(self.confidence),
            "priority": int(self.priority),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class GovernanceArbitrationDecision:
    decision_id: str
    selected_agent_id: str
    selected_recommendation: str
    arbitration_status: str
    conflict_detected: bool
    participating_agent_ids: tuple[str, ...]
    rejected_agent_ids: tuple[str, ...]
    reason: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_sha256_hex(self.decision_id, name="decision_id")
        if self.selected_agent_id != "NONE":
            _require_canonical_token(self.selected_agent_id, name="selected_agent_id")
        if self.selected_recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("selected_recommendation must be a supported governance label")
        if self.arbitration_status not in _ALLOWED_STATUSES:
            raise ValueError("arbitration_status is not supported")
        if not isinstance(self.conflict_detected, bool):
            raise ValueError("conflict_detected must be bool")
        if isinstance(self.reason, bool) or not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be non-empty string")
        normalized_participants = tuple(sorted(_require_canonical_token(v, name="participating_agent_ids") for v in self.participating_agent_ids))
        normalized_rejected = tuple(sorted(_require_canonical_token(v, name="rejected_agent_ids") for v in self.rejected_agent_ids))
        if len(set(normalized_participants)) != len(normalized_participants):
            raise ValueError("participating_agent_ids must be unique")
        if len(set(normalized_rejected)) != len(normalized_rejected):
            raise ValueError("rejected_agent_ids must be unique")
        if any(value not in set(normalized_participants) for value in normalized_rejected):
            raise ValueError("rejected_agent_ids must be subset of participating_agent_ids")
        if self.selected_agent_id != "NONE" and self.selected_agent_id not in set(normalized_participants):
            raise ValueError("selected_agent_id must participate")
        if self.selected_agent_id != "NONE" and self.selected_agent_id in set(normalized_rejected):
            raise ValueError("selected_agent_id cannot be rejected")
        object.__setattr__(self, "participating_agent_ids", normalized_participants)
        object.__setattr__(self, "rejected_agent_ids", normalized_rejected)

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "decision_id": self.decision_id,
            "selected_agent_id": self.selected_agent_id,
            "selected_recommendation": self.selected_recommendation,
            "arbitration_status": self.arbitration_status,
            "conflict_detected": self.conflict_detected,
            "participating_agent_ids": self.participating_agent_ids,
            "rejected_agent_ids": self.rejected_agent_ids,
            "reason": self.reason,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class MultiAgentGovernanceReceipt:
    schema_version: str
    module_version: str
    agent_count: int
    decision: GovernanceArbitrationDecision
    consensus_score: float
    conflict_count: int
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.schema_version != MULTI_AGENT_GOVERNANCE_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if self.module_version != MULTI_AGENT_GOVERNANCE_MODULE_VERSION:
            raise ValueError("unsupported module_version")
        if isinstance(self.agent_count, bool) or not isinstance(self.agent_count, int) or self.agent_count < 0:
            raise ValueError("agent_count must be non-negative integer")
        if self.agent_count != len(self.decision.participating_agent_ids):
            raise ValueError("agent_count must match decision participating_agent_ids length")
        if isinstance(self.consensus_score, bool) or not isinstance(self.consensus_score, (int, float)):
            raise ValueError("consensus_score must be finite float")
        consensus_score = _round_public_metric(float(self.consensus_score))
        if consensus_score < 0.0 or consensus_score > 1.0:
            raise ValueError("consensus_score must be in [0.0, 1.0]")
        if isinstance(self.conflict_count, bool) or not isinstance(self.conflict_count, int) or self.conflict_count < 0:
            raise ValueError("conflict_count must be non-negative integer")
        object.__setattr__(self, "consensus_score", consensus_score)

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "agent_count": int(self.agent_count),
            "decision": self.decision.to_dict(),
            "consensus_score": _round_public_metric(float(self.consensus_score)),
            "conflict_count": int(self.conflict_count),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _agent_conflict_sort_key(agent: GovernanceAgentState) -> tuple[Any, ...]:
    return (
        -agent.priority,
        -float(agent.confidence),
        -_RECOMMENDATION_SEVERITY[agent.recommendation],
        agent.agent_id,
        agent.stable_hash(),
    )


def _select_consensus_agent(agents: Sequence[GovernanceAgentState]) -> GovernanceAgentState:
    highest_priority = max(agent.priority for agent in agents)
    priority_candidates = tuple(agent for agent in agents if agent.priority == highest_priority)
    highest_confidence = max(float(agent.confidence) for agent in priority_candidates)
    confidence_candidates = tuple(agent for agent in priority_candidates if float(agent.confidence) == highest_confidence)
    return sorted(confidence_candidates, key=lambda a: (a.agent_id, a.stable_hash()))[0]


def arbitrate_multi_agent_governance(agents: Sequence[GovernanceAgentState]) -> MultiAgentGovernanceReceipt:
    canonical_agents = tuple(agents)
    seen: set[str] = set()
    for agent in canonical_agents:
        if agent.agent_id in seen:
            raise ValueError("duplicate agent_id is not allowed")
        seen.add(agent.agent_id)

    if not canonical_agents:
        participating = tuple()
        rejected = tuple()
        decision_payload = {
            "status": "EMPTY",
            "selected_agent_id": "NONE",
            "selected_recommendation": "REVIEW_MEMORY",
            "participating_agent_ids": participating,
            "rejected_agent_ids": rejected,
            "reason": "No governance agents provided",
        }
        decision = GovernanceArbitrationDecision(
            decision_id=sha256_hex(decision_payload),
            selected_agent_id="NONE",
            selected_recommendation="REVIEW_MEMORY",
            arbitration_status="EMPTY",
            conflict_detected=False,
            participating_agent_ids=participating,
            rejected_agent_ids=rejected,
            reason="No governance agents provided",
        )
        return MultiAgentGovernanceReceipt(
            schema_version=MULTI_AGENT_GOVERNANCE_SCHEMA_VERSION,
            module_version=MULTI_AGENT_GOVERNANCE_MODULE_VERSION,
            agent_count=0,
            decision=decision,
            consensus_score=1.0,
            conflict_count=0,
        )

    participating_agent_ids = tuple(sorted(agent.agent_id for agent in canonical_agents))
    recommendation_set = {agent.recommendation for agent in canonical_agents}

    if len(recommendation_set) == 1:
        selected = _select_consensus_agent(canonical_agents)
        rejected_agent_ids = tuple(agent_id for agent_id in participating_agent_ids if agent_id != selected.agent_id)
        decision_payload = {
            "status": "CONSENSUS",
            "selected_agent_id": selected.agent_id,
            "selected_recommendation": selected.recommendation,
            "participating_agent_ids": participating_agent_ids,
            "rejected_agent_ids": rejected_agent_ids,
            "reason": "All participating agents produced the same recommendation",
        }
        decision = GovernanceArbitrationDecision(
            decision_id=sha256_hex(decision_payload),
            selected_agent_id=selected.agent_id,
            selected_recommendation=selected.recommendation,
            arbitration_status="CONSENSUS",
            conflict_detected=False,
            participating_agent_ids=participating_agent_ids,
            rejected_agent_ids=rejected_agent_ids,
            reason="All participating agents produced the same recommendation",
        )
    else:
        sorted_agents = tuple(sorted(canonical_agents, key=_agent_conflict_sort_key))
        top_priority = max(agent.priority for agent in sorted_agents)
        priority_candidates = tuple(agent for agent in sorted_agents if agent.priority == top_priority)
        top_confidence = max(float(agent.confidence) for agent in priority_candidates)
        confidence_candidates = tuple(agent for agent in priority_candidates if float(agent.confidence) == top_confidence)

        if len(priority_candidates) == 1:
            selected = priority_candidates[0]
            status = "PRIORITY_SELECTED"
            selected_agent_id = selected.agent_id
            selected_recommendation = selected.recommendation
            reason = "Conflict resolved by strict priority winner"
            rejected_agent_ids = tuple(agent_id for agent_id in participating_agent_ids if agent_id != selected_agent_id)
        elif len(confidence_candidates) == 1:
            selected = confidence_candidates[0]
            status = "CONFIDENCE_SELECTED"
            selected_agent_id = selected.agent_id
            selected_recommendation = selected.recommendation
            reason = "Conflict resolved by strict confidence winner within highest priority"
            rejected_agent_ids = tuple(agent_id for agent_id in participating_agent_ids if agent_id != selected_agent_id)
        else:
            competing_recommendations = {agent.recommendation for agent in confidence_candidates}
            if len(competing_recommendations) > 1:
                status = "ESCALATED_CONFLICT"
                selected_agent_id = "NONE"
                selected_recommendation = "ESCALATE_GOVERNANCE"
                reason = "Competing top-priority and top-confidence recommendations require escalation"
                rejected_agent_ids = participating_agent_ids
            else:
                selected = sorted(confidence_candidates, key=_agent_conflict_sort_key)[0]
                status = "CONFIDENCE_SELECTED"
                selected_agent_id = selected.agent_id
                selected_recommendation = selected.recommendation
                reason = "Conflict resolved deterministically after confidence tie"
                rejected_agent_ids = tuple(agent_id for agent_id in participating_agent_ids if agent_id != selected_agent_id)

        decision_payload = {
            "status": status,
            "selected_agent_id": selected_agent_id,
            "selected_recommendation": selected_recommendation,
            "participating_agent_ids": participating_agent_ids,
            "rejected_agent_ids": rejected_agent_ids,
            "reason": reason,
        }
        decision = GovernanceArbitrationDecision(
            decision_id=sha256_hex(decision_payload),
            selected_agent_id=selected_agent_id,
            selected_recommendation=selected_recommendation,
            arbitration_status=status,
            conflict_detected=True,
            participating_agent_ids=participating_agent_ids,
            rejected_agent_ids=rejected_agent_ids,
            reason=reason,
        )

    selected_recommendation = decision.selected_recommendation
    match_count = sum(1 for agent in canonical_agents if agent.recommendation == selected_recommendation)
    consensus_score = 1.0 if not canonical_agents else _round_public_metric(match_count / len(canonical_agents))
    conflict_count = sum(1 for agent in canonical_agents if agent.recommendation != selected_recommendation)

    return MultiAgentGovernanceReceipt(
        schema_version=MULTI_AGENT_GOVERNANCE_SCHEMA_VERSION,
        module_version=MULTI_AGENT_GOVERNANCE_MODULE_VERSION,
        agent_count=len(canonical_agents),
        decision=decision,
        consensus_score=consensus_score,
        conflict_count=conflict_count,
    )


__all__ = [
    "MULTI_AGENT_GOVERNANCE_MODULE_VERSION",
    "MULTI_AGENT_GOVERNANCE_SCHEMA_VERSION",
    "GovernanceAgentState",
    "GovernanceArbitrationDecision",
    "MultiAgentGovernanceReceipt",
    "arbitrate_multi_agent_governance",
]
