"""v149.1 — Deterministic hierarchical memory/shared-memory arbitration."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
from typing import Any, Final, Sequence

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION: Final[str] = "v149.1"

_ALLOWED_RECOMMENDATIONS: Final[tuple[str, ...]] = (
    "MAINTAIN_POLICY",
    "TIGHTEN_POLICY",
    "RELAX_POLICY",
    "ESCALATE_GOVERNANCE",
    "REVIEW_MEMORY",
)
_ALLOWED_PROMOTION_STATUSES: Final[tuple[str, ...]] = (
    "EMPTY",
    "CONSENSUS_PROMOTED",
    "PRIORITY_PROMOTED",
    "CONFIDENCE_PROMOTED",
    "RECURSIVE_ESCALATION",
)
_ALLOWED_DECISION_STATUSES: Final[tuple[str, ...]] = (
    "EMPTY",
    "GLOBAL_MEMORY_READY",
    "RECURSIVE_GOVERNANCE_REQUIRED",
)

_RECOMMENDATION_SEVERITY: Final[dict[str, int]] = {
    "RELAX_POLICY": 0,
    "MAINTAIN_POLICY": 1,
    "TIGHTEN_POLICY": 2,
    "REVIEW_MEMORY": 3,
    "ESCALATE_GOVERNANCE": 4,
}
_RESERVED_NONE: Final[str] = "NONE"

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


def _require_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    return number


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _normalize_unique_sorted_tokens(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be tuple")
    normalized = tuple(sorted(_require_canonical_token(v, name=name) for v in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _recommendation_sort_value(recommendation: str) -> int:
    return _RECOMMENDATION_SEVERITY[recommendation]


@dataclass(frozen=True)
class LocalMemoryState:
    agent_id: str
    memory_key: str
    recommendation: str
    confidence: float
    priority: int
    local_epoch: int
    source_receipt_hash: str
    evidence_hash: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.agent_id, name="agent_id")
        if self.agent_id == _RESERVED_NONE:
            raise ValueError('agent_id "NONE" is reserved')
        _require_canonical_token(self.memory_key, name="memory_key")
        if self.recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("recommendation must be a supported governance label")
        object.__setattr__(self, "confidence", _round_public_metric(_require_probability(self.confidence, name="confidence")))
        _require_non_negative_int(self.priority, name="priority")
        _require_non_negative_int(self.local_epoch, name="local_epoch")
        _require_sha256_hex(self.source_receipt_hash, name="source_receipt_hash")
        _require_sha256_hex(self.evidence_hash, name="evidence_hash")

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
            "memory_key": self.memory_key,
            "recommendation": self.recommendation,
            "confidence": _round_public_metric(float(self.confidence)),
            "priority": int(self.priority),
            "local_epoch": int(self.local_epoch),
            "source_receipt_hash": self.source_receipt_hash,
            "evidence_hash": self.evidence_hash,
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
class GlobalMemoryProjection:
    memory_key: str
    promotion_status: str
    selected_agent_id: str
    selected_recommendation: str
    participating_agent_ids: tuple[str, ...]
    rejected_agent_ids: tuple[str, ...]
    contributing_local_hashes: tuple[str, ...]
    aggregate_priority: int
    aggregate_confidence: float
    consensus_score: float
    conflict_count: int
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.memory_key, name="memory_key")
        if self.promotion_status not in _ALLOWED_PROMOTION_STATUSES:
            raise ValueError("promotion_status is not supported")
        if self.selected_agent_id != _RESERVED_NONE:
            _require_canonical_token(self.selected_agent_id, name="selected_agent_id")
        if self.selected_recommendation != _RESERVED_NONE and self.selected_recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("selected_recommendation must be a supported governance label")

        participating = _normalize_unique_sorted_tokens(self.participating_agent_ids, name="participating_agent_ids")
        rejected = _normalize_unique_sorted_tokens(self.rejected_agent_ids, name="rejected_agent_ids")

        if not isinstance(self.contributing_local_hashes, tuple):
            raise ValueError("contributing_local_hashes must be tuple")
        local_hashes = tuple(sorted(_require_sha256_hex(v, name="contributing_local_hashes") for v in self.contributing_local_hashes))
        if len(set(local_hashes)) != len(local_hashes):
            raise ValueError("contributing_local_hashes must be unique")

        if any(agent_id not in set(participating) for agent_id in rejected):
            raise ValueError("rejected_agent_ids must be subset of participating_agent_ids")

        if self.selected_agent_id == _RESERVED_NONE:
            if self.promotion_status not in {"EMPTY", "RECURSIVE_ESCALATION"}:
                raise ValueError('selected_agent_id "NONE" only allowed for EMPTY or RECURSIVE_ESCALATION')
        elif self.selected_agent_id not in set(participating):
            raise ValueError("selected_agent_id must participate")

        if self.selected_recommendation == _RESERVED_NONE:
            if self.promotion_status not in {"EMPTY", "RECURSIVE_ESCALATION"}:
                raise ValueError('selected_recommendation "NONE" only allowed for EMPTY or RECURSIVE_ESCALATION')
        if self.selected_agent_id != _RESERVED_NONE and self.selected_agent_id in set(rejected):
            raise ValueError("selected_agent_id cannot be rejected")

        _require_non_negative_int(self.aggregate_priority, name="aggregate_priority")
        object.__setattr__(self, "aggregate_confidence", _round_public_metric(_require_probability(self.aggregate_confidence, name="aggregate_confidence")))
        object.__setattr__(self, "consensus_score", _round_public_metric(_require_probability(self.consensus_score, name="consensus_score")))
        _require_non_negative_int(self.conflict_count, name="conflict_count")

        if len(participating) != len(local_hashes):
            raise ValueError("participating_agent_ids and contributing_local_hashes length mismatch")
        if self.conflict_count > len(participating):
            raise ValueError("conflict_count must not exceed participant count")

        object.__setattr__(self, "participating_agent_ids", participating)
        object.__setattr__(self, "rejected_agent_ids", rejected)
        object.__setattr__(self, "contributing_local_hashes", local_hashes)

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
            "memory_key": self.memory_key,
            "promotion_status": self.promotion_status,
            "selected_agent_id": self.selected_agent_id,
            "selected_recommendation": self.selected_recommendation,
            "participating_agent_ids": self.participating_agent_ids,
            "rejected_agent_ids": self.rejected_agent_ids,
            "contributing_local_hashes": self.contributing_local_hashes,
            "aggregate_priority": int(self.aggregate_priority),
            "aggregate_confidence": _round_public_metric(float(self.aggregate_confidence)),
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


@dataclass(frozen=True)
class RecursiveGovernanceMemoryDecision:
    decision_status: str
    selected_memory_key: str
    selected_agent_id: str
    selected_recommendation: str
    participating_memory_keys: tuple[str, ...]
    escalation_memory_keys: tuple[str, ...]
    global_projection_hashes: tuple[str, ...]
    recursive_governance_required: bool
    recursion_depth: int
    decision_reason: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.decision_status not in _ALLOWED_DECISION_STATUSES:
            raise ValueError("decision_status is not supported")

        if self.selected_memory_key != _RESERVED_NONE:
            _require_canonical_token(self.selected_memory_key, name="selected_memory_key")
        elif self.decision_status == "GLOBAL_MEMORY_READY":
            raise ValueError('selected_memory_key "NONE" only allowed for EMPTY or RECURSIVE_GOVERNANCE_REQUIRED')

        if self.selected_agent_id != _RESERVED_NONE:
            _require_canonical_token(self.selected_agent_id, name="selected_agent_id")
        elif self.decision_status == "GLOBAL_MEMORY_READY":
            raise ValueError('selected_agent_id "NONE" only allowed for EMPTY or RECURSIVE_GOVERNANCE_REQUIRED')

        if self.selected_recommendation != _RESERVED_NONE and self.selected_recommendation not in _ALLOWED_RECOMMENDATIONS:
            raise ValueError("selected_recommendation must be a supported governance label")
        if self.selected_recommendation == _RESERVED_NONE and self.decision_status == "GLOBAL_MEMORY_READY":
            raise ValueError('selected_recommendation "NONE" only allowed for EMPTY or RECURSIVE_GOVERNANCE_REQUIRED')

        participating_keys = _normalize_unique_sorted_tokens(self.participating_memory_keys, name="participating_memory_keys")
        escalation_keys = _normalize_unique_sorted_tokens(self.escalation_memory_keys, name="escalation_memory_keys")
        if any(key not in set(participating_keys) for key in escalation_keys):
            raise ValueError("escalation_memory_keys must be subset of participating_memory_keys")

        if not isinstance(self.global_projection_hashes, tuple):
            raise ValueError("global_projection_hashes must be tuple")
        projection_hashes = tuple(sorted(_require_sha256_hex(v, name="global_projection_hashes") for v in self.global_projection_hashes))
        if len(set(projection_hashes)) != len(projection_hashes):
            raise ValueError("global_projection_hashes must be unique")
        if len(projection_hashes) != len(participating_keys):
            raise ValueError("global_projection_hashes length must match participating_memory_keys")

        if not isinstance(self.recursive_governance_required, bool):
            raise ValueError("recursive_governance_required must be bool")
        _require_non_negative_int(self.recursion_depth, name="recursion_depth")
        _require_canonical_token(self.decision_reason, name="decision_reason")

        object.__setattr__(self, "participating_memory_keys", participating_keys)
        object.__setattr__(self, "escalation_memory_keys", escalation_keys)
        object.__setattr__(self, "global_projection_hashes", projection_hashes)

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
            "decision_status": self.decision_status,
            "selected_memory_key": self.selected_memory_key,
            "selected_agent_id": self.selected_agent_id,
            "selected_recommendation": self.selected_recommendation,
            "participating_memory_keys": self.participating_memory_keys,
            "escalation_memory_keys": self.escalation_memory_keys,
            "global_projection_hashes": self.global_projection_hashes,
            "recursive_governance_required": self.recursive_governance_required,
            "recursion_depth": int(self.recursion_depth),
            "decision_reason": self.decision_reason,
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
class HierarchicalMemoryArbitrationReceipt:
    module_version: str
    local_memory_count: int
    global_memory_count: int
    recursion_depth: int
    global_projections: tuple[GlobalMemoryProjection, ...]
    decision: RecursiveGovernanceMemoryDecision
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.module_version != HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION:
            raise ValueError("unsupported module_version")
        _require_non_negative_int(self.local_memory_count, name="local_memory_count")
        _require_non_negative_int(self.global_memory_count, name="global_memory_count")
        _require_non_negative_int(self.recursion_depth, name="recursion_depth")

        if not isinstance(self.global_projections, tuple):
            raise ValueError("global_projections must be tuple")
        if any(not isinstance(p, GlobalMemoryProjection) for p in self.global_projections):
            raise ValueError("global_projections must contain GlobalMemoryProjection")
        sorted_projections = tuple(sorted(self.global_projections, key=lambda p: p.memory_key))
        if len(set(p.memory_key for p in sorted_projections)) != len(sorted_projections):
            raise ValueError("global_projections must have unique memory_key")

        if self.global_memory_count != len(sorted_projections):
            raise ValueError("global_memory_count must match global_projections length")
        if self.recursion_depth != self.decision.recursion_depth:
            raise ValueError("recursion_depth must match decision recursion_depth")

        object.__setattr__(self, "global_projections", sorted_projections)

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
            "module_version": self.module_version,
            "local_memory_count": int(self.local_memory_count),
            "global_memory_count": int(self.global_memory_count),
            "recursion_depth": int(self.recursion_depth),
            "global_projections": tuple(projection.to_dict() for projection in self.global_projections),
            "decision": self.decision.to_dict(),
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _local_sort_key(local_state: LocalMemoryState) -> tuple[Any, ...]:
    return (
        local_state.memory_key,
        -local_state.priority,
        -float(local_state.confidence),
        -_recommendation_sort_value(local_state.recommendation),
        local_state.agent_id,
        local_state.stable_hash(),
    )


def _projection_tie_key(local_state: LocalMemoryState) -> tuple[Any, ...]:
    return (
        -local_state.priority,
        -float(local_state.confidence),
        -_recommendation_sort_value(local_state.recommendation),
        local_state.agent_id,
        local_state.stable_hash(),
    )


def _build_projection(memory_key: str, local_states: tuple[LocalMemoryState, ...]) -> GlobalMemoryProjection:
    if not local_states:
        return GlobalMemoryProjection(
            memory_key=memory_key,
            promotion_status="EMPTY",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_agent_ids=tuple(),
            rejected_agent_ids=tuple(),
            contributing_local_hashes=tuple(),
            aggregate_priority=0,
            aggregate_confidence=0.0,
            consensus_score=1.0,
            conflict_count=0,
        )

    sorted_locals = tuple(sorted(local_states, key=_projection_tie_key))
    participating = tuple(sorted(local.agent_id for local in sorted_locals))
    local_hashes = tuple(sorted(local.stable_hash() for local in sorted_locals))
    aggregate_priority = sum(local.priority for local in sorted_locals)
    aggregate_confidence = _round_public_metric(sum(float(local.confidence) for local in sorted_locals) / len(sorted_locals))

    recommendations = tuple(local.recommendation for local in sorted_locals)
    recommendation_set = set(recommendations)

    if len(recommendation_set) == 1:
        selected = sorted_locals[0]
        selected_recommendation = selected.recommendation
        rejected = tuple(agent_id for agent_id in participating if agent_id != selected.agent_id)
        consensus_score = aggregate_confidence
        return GlobalMemoryProjection(
            memory_key=memory_key,
            promotion_status="CONSENSUS_PROMOTED",
            selected_agent_id=selected.agent_id,
            selected_recommendation=selected_recommendation,
            participating_agent_ids=participating,
            rejected_agent_ids=rejected,
            contributing_local_hashes=local_hashes,
            aggregate_priority=aggregate_priority,
            aggregate_confidence=aggregate_confidence,
            consensus_score=consensus_score,
            conflict_count=0,
        )

    top_priority = sorted_locals[0].priority
    priority_candidates = tuple(local for local in sorted_locals if local.priority == top_priority)
    if len(priority_candidates) == 1:
        selected = priority_candidates[0]
        rejected = tuple(agent_id for agent_id in participating if agent_id != selected.agent_id)
        return GlobalMemoryProjection(
            memory_key=memory_key,
            promotion_status="PRIORITY_PROMOTED",
            selected_agent_id=selected.agent_id,
            selected_recommendation=selected.recommendation,
            participating_agent_ids=participating,
            rejected_agent_ids=rejected,
            contributing_local_hashes=local_hashes,
            aggregate_priority=aggregate_priority,
            aggregate_confidence=aggregate_confidence,
            consensus_score=_round_public_metric(sum(1 for r in recommendations if r == selected.recommendation) / len(recommendations)),
            conflict_count=sum(1 for r in recommendations if r != selected.recommendation),
        )

    top_confidence = max(float(local.confidence) for local in priority_candidates)
    confidence_candidates = tuple(local for local in priority_candidates if float(local.confidence) == top_confidence)
    if len(confidence_candidates) == 1:
        selected = sorted(confidence_candidates, key=_projection_tie_key)[0]
        rejected = tuple(agent_id for agent_id in participating if agent_id != selected.agent_id)
        return GlobalMemoryProjection(
            memory_key=memory_key,
            promotion_status="CONFIDENCE_PROMOTED",
            selected_agent_id=selected.agent_id,
            selected_recommendation=selected.recommendation,
            participating_agent_ids=participating,
            rejected_agent_ids=rejected,
            contributing_local_hashes=local_hashes,
            aggregate_priority=aggregate_priority,
            aggregate_confidence=aggregate_confidence,
            consensus_score=_round_public_metric(sum(1 for r in recommendations if r == selected.recommendation) / len(recommendations)),
            conflict_count=sum(1 for r in recommendations if r != selected.recommendation),
        )

    if len({local.recommendation for local in confidence_candidates}) > 1:
        return GlobalMemoryProjection(
            memory_key=memory_key,
            promotion_status="RECURSIVE_ESCALATION",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_agent_ids=participating,
            rejected_agent_ids=participating,
            contributing_local_hashes=local_hashes,
            aggregate_priority=aggregate_priority,
            aggregate_confidence=aggregate_confidence,
            consensus_score=_round_public_metric(max(sum(1 for r in recommendations if r == candidate) for candidate in recommendation_set) / len(recommendations)),
            conflict_count=len(recommendation_set),
        )

    structurally_tied = confidence_candidates
    selected = sorted(structurally_tied, key=_projection_tie_key)[0]
    rejected = tuple(agent_id for agent_id in participating if agent_id != selected.agent_id)
    return GlobalMemoryProjection(
        memory_key=memory_key,
        promotion_status="CONFIDENCE_PROMOTED",
        selected_agent_id=selected.agent_id,
        selected_recommendation=selected.recommendation,
        participating_agent_ids=participating,
        rejected_agent_ids=rejected,
        contributing_local_hashes=local_hashes,
        aggregate_priority=aggregate_priority,
        aggregate_confidence=aggregate_confidence,
        consensus_score=_round_public_metric(sum(1 for r in recommendations if r == selected.recommendation) / len(recommendations)),
        conflict_count=sum(1 for r in recommendations if r != selected.recommendation),
    )


def _global_projection_sort_key(projection: GlobalMemoryProjection) -> tuple[Any, ...]:
    return (
        -projection.aggregate_priority,
        -float(projection.aggregate_confidence),
        -_recommendation_sort_value(projection.selected_recommendation),
        projection.memory_key,
        projection.selected_agent_id,
        projection.stable_hash(),
    )


def arbitrate_hierarchical_memory(
    local_memories: Sequence[LocalMemoryState],
    *,
    recursion_depth: int = 1,
) -> HierarchicalMemoryArbitrationReceipt:
    if isinstance(recursion_depth, bool) or not isinstance(recursion_depth, int) or recursion_depth < 0:
        raise ValueError("recursion_depth must be a non-negative integer")

    canonical_memories = tuple(local_memories)
    seen_pairs: set[tuple[str, str]] = set()
    for item in canonical_memories:
        if not isinstance(item, LocalMemoryState):
            raise ValueError("local_memories must contain LocalMemoryState")
        pair = (item.agent_id, item.memory_key)
        if pair in seen_pairs:
            raise ValueError("duplicate (agent_id, memory_key) is not allowed")
        seen_pairs.add(pair)

    sorted_memories = tuple(sorted(canonical_memories, key=_local_sort_key))

    grouped_memories: dict[str, list[LocalMemoryState]] = {}
    for local in sorted_memories:
        grouped_memories.setdefault(local.memory_key, []).append(local)

    projections = tuple(
        _build_projection(memory_key, tuple(local_states))
        for memory_key, local_states in grouped_memories.items()
    )

    projection_hashes = tuple(sorted(projection.stable_hash() for projection in projections))

    if not projections:
        decision = RecursiveGovernanceMemoryDecision(
            decision_status="EMPTY",
            selected_memory_key="NONE",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_memory_keys=tuple(),
            escalation_memory_keys=tuple(),
            global_projection_hashes=tuple(),
            recursive_governance_required=False,
            recursion_depth=recursion_depth,
            decision_reason="empty",
        )
    elif any(projection.promotion_status == "RECURSIVE_ESCALATION" for projection in projections):
        decision = RecursiveGovernanceMemoryDecision(
            decision_status="RECURSIVE_GOVERNANCE_REQUIRED",
            selected_memory_key="NONE",
            selected_agent_id="NONE",
            selected_recommendation="NONE",
            participating_memory_keys=tuple(sorted(projection.memory_key for projection in projections)),
            escalation_memory_keys=tuple(sorted(projection.memory_key for projection in projections if projection.promotion_status == "RECURSIVE_ESCALATION")),
            global_projection_hashes=projection_hashes,
            recursive_governance_required=True,
            recursion_depth=recursion_depth,
            decision_reason="local_memory_conflict",
        )
    else:
        selected_recommendations = {projection.selected_recommendation for projection in projections}
        if len(selected_recommendations) == 1:
            selected = sorted(projections, key=lambda p: (p.memory_key, p.selected_agent_id, p.stable_hash()))[0]
            decision = RecursiveGovernanceMemoryDecision(
                decision_status="GLOBAL_MEMORY_READY",
                selected_memory_key=selected.memory_key,
                selected_agent_id=selected.selected_agent_id,
                selected_recommendation=selected.selected_recommendation,
                participating_memory_keys=tuple(sorted(projection.memory_key for projection in projections)),
                escalation_memory_keys=tuple(),
                global_projection_hashes=projection_hashes,
                recursive_governance_required=False,
                recursion_depth=recursion_depth,
                decision_reason="global_consensus",
            )
        else:
            sorted_projections = tuple(sorted(projections, key=_global_projection_sort_key))
            top = sorted_projections[0]
            tied = tuple(
                p
                for p in sorted_projections
                if p.aggregate_priority == top.aggregate_priority
                and float(p.aggregate_confidence) == float(top.aggregate_confidence)
            )
            if len({p.selected_recommendation for p in tied}) > 1:
                decision = RecursiveGovernanceMemoryDecision(
                    decision_status="RECURSIVE_GOVERNANCE_REQUIRED",
                    selected_memory_key="NONE",
                    selected_agent_id="NONE",
                    selected_recommendation="NONE",
                    participating_memory_keys=tuple(sorted(projection.memory_key for projection in projections)),
                    escalation_memory_keys=tuple(sorted(projection.memory_key for projection in tied)),
                    global_projection_hashes=projection_hashes,
                    recursive_governance_required=True,
                    recursion_depth=recursion_depth,
                    decision_reason="global_memory_conflict",
                )
            else:
                decision = RecursiveGovernanceMemoryDecision(
                    decision_status="GLOBAL_MEMORY_READY",
                    selected_memory_key=top.memory_key,
                    selected_agent_id=top.selected_agent_id,
                    selected_recommendation=top.selected_recommendation,
                    participating_memory_keys=tuple(sorted(projection.memory_key for projection in projections)),
                    escalation_memory_keys=tuple(),
                    global_projection_hashes=projection_hashes,
                    recursive_governance_required=False,
                    recursion_depth=recursion_depth,
                    decision_reason="global_priority_selection",
                )

    return HierarchicalMemoryArbitrationReceipt(
        module_version=HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION,
        local_memory_count=len(sorted_memories),
        global_memory_count=len(projections),
        recursion_depth=recursion_depth,
        global_projections=projections,
        decision=decision,
    )


__all__ = [
    "HIERARCHICAL_MEMORY_ARBITRATION_MODULE_VERSION",
    "LocalMemoryState",
    "GlobalMemoryProjection",
    "RecursiveGovernanceMemoryDecision",
    "HierarchicalMemoryArbitrationReceipt",
    "arbitrate_hierarchical_memory",
]
