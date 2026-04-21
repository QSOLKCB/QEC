"""v139.4 — Distributed Proof Aggregation.

Deterministic analysis-only aggregation of node-local proof consensus outputs
into a canonical global proof artifact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

ALLOWED_AGGREGATION_ACTION_TYPES: tuple[str, ...] = (
    "compare_proof_bundle",
    "filter_node",
    "include_node",
    "exclude_node",
    "aggregate_proof",
    "emit_global_proof",
)
DISTRIBUTED_PROOF_AGGREGATION_SCHEMA_VERSION = "v139.4"


def _is_sha256_hex(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _validate_non_empty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty str")


def _validate_bool(value: bool, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool")


def _validate_fraction(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric in [0,1]")
    out = float(value)
    if not math.isfinite(out) or not 0.0 <= out <= 1.0:
        raise ValueError(f"{field_name} must be finite numeric in [0,1]")
    return out


@dataclass(frozen=True)
class AggregationInput:
    node_id: str
    epoch_index: int
    proof_bundle_hash: str
    replay_identity: str
    consensus_ready: bool
    consensus_confidence: float
    consensus_risk: float
    claim_count: int

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        if isinstance(self.epoch_index, bool) or not isinstance(self.epoch_index, int) or self.epoch_index < 0:
            raise ValueError("epoch_index must be int >= 0")
        if not _is_sha256_hex(self.proof_bundle_hash):
            raise ValueError("proof_bundle_hash must be 64-char lowercase sha256 hex")
        if not _is_sha256_hex(self.replay_identity):
            raise ValueError("replay_identity must be 64-char lowercase sha256 hex")
        _validate_bool(self.consensus_ready, "consensus_ready")
        object.__setattr__(
            self,
            "consensus_confidence",
            _validate_fraction(self.consensus_confidence, "consensus_confidence"),
        )
        object.__setattr__(
            self,
            "consensus_risk",
            _validate_fraction(self.consensus_risk, "consensus_risk"),
        )
        if isinstance(self.claim_count, bool) or not isinstance(self.claim_count, int) or self.claim_count < 0:
            raise ValueError("claim_count must be int >= 0")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "epoch_index": self.epoch_index,
            "proof_bundle_hash": self.proof_bundle_hash,
            "replay_identity": self.replay_identity,
            "consensus_ready": self.consensus_ready,
            "consensus_confidence": self.consensus_confidence,
            "consensus_risk": self.consensus_risk,
            "claim_count": self.claim_count,
        }


@dataclass(frozen=True)
class AggregationPolicy:
    require_consensus_ready: bool
    minimum_confidence_threshold: float
    maximum_risk_threshold: float
    require_epoch_alignment: bool
    allow_partial_aggregation: bool
    minimum_participation_fraction: float

    def __post_init__(self) -> None:
        _validate_bool(self.require_consensus_ready, "require_consensus_ready")
        _validate_bool(self.require_epoch_alignment, "require_epoch_alignment")
        _validate_bool(self.allow_partial_aggregation, "allow_partial_aggregation")
        object.__setattr__(
            self,
            "minimum_confidence_threshold",
            _validate_fraction(self.minimum_confidence_threshold, "minimum_confidence_threshold"),
        )
        object.__setattr__(
            self,
            "maximum_risk_threshold",
            _validate_fraction(self.maximum_risk_threshold, "maximum_risk_threshold"),
        )
        object.__setattr__(
            self,
            "minimum_participation_fraction",
            _validate_fraction(self.minimum_participation_fraction, "minimum_participation_fraction"),
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "require_consensus_ready": self.require_consensus_ready,
            "minimum_confidence_threshold": self.minimum_confidence_threshold,
            "maximum_risk_threshold": self.maximum_risk_threshold,
            "require_epoch_alignment": self.require_epoch_alignment,
            "allow_partial_aggregation": self.allow_partial_aggregation,
            "minimum_participation_fraction": self.minimum_participation_fraction,
        }


@dataclass(frozen=True)
class AggregationNodeStatus:
    node_id: str
    admissible: bool
    epoch_aligned: bool
    confidence_ok: bool
    risk_ok: bool
    contributes_to_aggregation: bool
    aggregation_weight: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        for name in (
            "admissible",
            "epoch_aligned",
            "confidence_ok",
            "risk_ok",
            "contributes_to_aggregation",
        ):
            _validate_bool(getattr(self, name), name)
        object.__setattr__(self, "aggregation_weight", _validate_fraction(self.aggregation_weight, "aggregation_weight"))
        if not isinstance(self.reasons, tuple) or any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("reasons must be tuple[str, ...] of non-empty strings")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "admissible": self.admissible,
            "epoch_aligned": self.epoch_aligned,
            "confidence_ok": self.confidence_ok,
            "risk_ok": self.risk_ok,
            "contributes_to_aggregation": self.contributes_to_aggregation,
            "aggregation_weight": self.aggregation_weight,
            "reasons": self.reasons,
        }


@dataclass(frozen=True)
class AggregationAction:
    action_index: int
    action_type: str
    source_node_id: str
    target_node_id: str
    blocking: bool
    ready: bool
    detail: str

    def __post_init__(self) -> None:
        if isinstance(self.action_index, bool) or not isinstance(self.action_index, int) or self.action_index < 0:
            raise ValueError("action_index must be int >= 0")
        if self.action_type not in ALLOWED_AGGREGATION_ACTION_TYPES:
            raise ValueError("invalid action_type")
        _validate_non_empty_str(self.source_node_id, "source_node_id")
        _validate_non_empty_str(self.target_node_id, "target_node_id")
        _validate_bool(self.blocking, "blocking")
        _validate_bool(self.ready, "ready")
        _validate_non_empty_str(self.detail, "detail")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "action_index": self.action_index,
            "action_type": self.action_type,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "blocking": self.blocking,
            "ready": self.ready,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DistributedProofAggregationReceipt:
    aggregation_inputs: tuple[AggregationInput, ...]
    policy_snapshot: AggregationPolicy
    node_statuses: tuple[AggregationNodeStatus, ...]
    aggregation_actions: tuple[AggregationAction, ...]
    cluster_epoch: int
    reference_node_id: str
    aggregated_proof_hash: str
    participation_fraction: float
    structurally_consistent: bool
    aggregation_ready: bool
    aggregation_confidence: float
    aggregation_risk: float
    rationale: tuple[str, ...]
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.aggregation_inputs, tuple) or not self.aggregation_inputs:
            raise ValueError("aggregation_inputs must be non-empty tuple")
        if any(not isinstance(item, AggregationInput) for item in self.aggregation_inputs):
            raise ValueError("aggregation_inputs must contain AggregationInput")
        if not isinstance(self.policy_snapshot, AggregationPolicy):
            raise ValueError("policy_snapshot must be AggregationPolicy")
        if not isinstance(self.node_statuses, tuple) or any(not isinstance(item, AggregationNodeStatus) for item in self.node_statuses):
            raise ValueError("node_statuses must be tuple[AggregationNodeStatus, ...]")
        if not isinstance(self.aggregation_actions, tuple) or any(not isinstance(item, AggregationAction) for item in self.aggregation_actions):
            raise ValueError("aggregation_actions must be tuple[AggregationAction, ...]")
        if isinstance(self.cluster_epoch, bool) or not isinstance(self.cluster_epoch, int) or self.cluster_epoch < 0:
            raise ValueError("cluster_epoch must be int >= 0")
        _validate_non_empty_str(self.reference_node_id, "reference_node_id")
        if not _is_sha256_hex(self.aggregated_proof_hash):
            raise ValueError("aggregated_proof_hash must be 64-char lowercase sha256 hex")
        object.__setattr__(self, "participation_fraction", _validate_fraction(self.participation_fraction, "participation_fraction"))
        _validate_bool(self.structurally_consistent, "structurally_consistent")
        _validate_bool(self.aggregation_ready, "aggregation_ready")
        object.__setattr__(self, "aggregation_confidence", _validate_fraction(self.aggregation_confidence, "aggregation_confidence"))
        object.__setattr__(self, "aggregation_risk", _validate_fraction(self.aggregation_risk, "aggregation_risk"))
        if not isinstance(self.rationale, tuple) or any(not isinstance(item, str) or not item for item in self.rationale):
            raise ValueError("rationale must be tuple[str, ...] of non-empty strings")
        _validate_non_empty_str(self.schema_version, "schema_version")

        expected_replay_identity = _compute_replay_identity(
            self.aggregation_inputs,
            self.policy_snapshot,
            self.cluster_epoch,
            self.reference_node_id,
            self.aggregated_proof_hash,
        )
        if not _is_sha256_hex(self.replay_identity):
            raise ValueError("replay_identity must be 64-char lowercase sha256 hex")
        if self.replay_identity != expected_replay_identity:
            raise ValueError("replay_identity mismatch with receipt contents")
        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be 64-char lowercase sha256 hex")
        if self.stable_hash_value() != self.stable_hash:
            raise ValueError("stable_hash must match stable_hash_value")

    def _hash_payload(self) -> dict[str, _JSONValue]:
        return _build_receipt_hash_payload(
            aggregation_inputs=self.aggregation_inputs,
            policy_snapshot=self.policy_snapshot,
            node_statuses=self.node_statuses,
            aggregation_actions=self.aggregation_actions,
            cluster_epoch=self.cluster_epoch,
            reference_node_id=self.reference_node_id,
            aggregated_proof_hash=self.aggregated_proof_hash,
            participation_fraction=self.participation_fraction,
            structurally_consistent=self.structurally_consistent,
            aggregation_ready=self.aggregation_ready,
            aggregation_confidence=self.aggregation_confidence,
            aggregation_risk=self.aggregation_risk,
            rationale=self.rationale,
            schema_version=self.schema_version,
            replay_identity=self.replay_identity,
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self._hash_payload())


def _build_receipt_hash_payload(
    *,
    aggregation_inputs: tuple[AggregationInput, ...],
    policy_snapshot: AggregationPolicy,
    node_statuses: tuple[AggregationNodeStatus, ...],
    aggregation_actions: tuple[AggregationAction, ...],
    cluster_epoch: int,
    reference_node_id: str,
    aggregated_proof_hash: str,
    participation_fraction: float,
    structurally_consistent: bool,
    aggregation_ready: bool,
    aggregation_confidence: float,
    aggregation_risk: float,
    rationale: tuple[str, ...],
    schema_version: str,
    replay_identity: str,
) -> dict[str, _JSONValue]:
    return {
        "aggregation_inputs": tuple(item.to_dict() for item in aggregation_inputs),
        "policy_snapshot": policy_snapshot.to_dict(),
        "node_statuses": tuple(item.to_dict() for item in node_statuses),
        "aggregation_actions": tuple(item.to_dict() for item in aggregation_actions),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "aggregated_proof_hash": aggregated_proof_hash,
        "participation_fraction": participation_fraction,
        "structurally_consistent": structurally_consistent,
        "aggregation_ready": aggregation_ready,
        "aggregation_confidence": aggregation_confidence,
        "aggregation_risk": aggregation_risk,
        "rationale": rationale,
        "schema_version": schema_version,
        "replay_identity": replay_identity,
    }


def _validate_inputs(aggregation_inputs: tuple[AggregationInput, ...]) -> tuple[AggregationInput, ...]:
    if not isinstance(aggregation_inputs, tuple) or not aggregation_inputs:
        raise ValueError("aggregation_inputs must be a non-empty tuple")
    if any(not isinstance(item, AggregationInput) for item in aggregation_inputs):
        raise ValueError("aggregation_inputs must contain AggregationInput values")
    ordered = tuple(sorted(aggregation_inputs, key=lambda item: item.node_id))
    node_ids = tuple(item.node_id for item in ordered)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node_id")
    return ordered


def _compute_cluster_epoch(aggregation_inputs: tuple[AggregationInput, ...]) -> int:
    counts = Counter(item.epoch_index for item in aggregation_inputs)
    return min((-count, epoch) for epoch, count in counts.items())[1]


def _reference_priority(node: AggregationInput, policy: AggregationPolicy, cluster_epoch: int) -> tuple[int, int, float, float, str]:
    policy_satisfied = (not policy.require_consensus_ready) or node.consensus_ready
    epoch_satisfied = (not policy.require_epoch_alignment) or (node.epoch_index == cluster_epoch)
    return (
        0 if policy_satisfied else 1,
        0 if epoch_satisfied else 1,
        -node.consensus_confidence,
        node.consensus_risk,
        node.node_id,
    )


def _select_reference_node(
    aggregation_inputs: tuple[AggregationInput, ...],
    policy: AggregationPolicy,
    cluster_epoch: int,
) -> AggregationInput:
    return min(aggregation_inputs, key=lambda node: _reference_priority(node, policy, cluster_epoch))


def _admissibility(
    item: AggregationInput,
    policy: AggregationPolicy,
    cluster_epoch: int,
) -> tuple[bool, bool, bool, bool, tuple[str, ...]]:
    epoch_aligned = item.epoch_index == cluster_epoch
    confidence_ok = item.consensus_confidence >= policy.minimum_confidence_threshold
    risk_ok = item.consensus_risk <= policy.maximum_risk_threshold
    ready_ok = (not policy.require_consensus_ready) or item.consensus_ready
    admissible = ready_ok and confidence_ok and risk_ok and ((not policy.require_epoch_alignment) or epoch_aligned)

    reasons: list[str] = []
    reasons.append("consensus ready" if item.consensus_ready else "consensus not ready")
    reasons.append("confidence threshold met" if confidence_ok else "confidence below threshold")
    reasons.append("risk threshold met" if risk_ok else "risk above threshold")
    reasons.append("epoch aligned" if epoch_aligned else "epoch mismatch")
    return admissible, epoch_aligned, confidence_ok, risk_ok, tuple(reasons)


def _compute_aggregated_proof_hash(included_nodes: tuple[AggregationInput, ...]) -> str:
    hashes = tuple(sorted(node.proof_bundle_hash for node in included_nodes))
    return sha256_hex(hashes)


def _compute_replay_identity(
    aggregation_inputs: tuple[AggregationInput, ...],
    policy: AggregationPolicy,
    cluster_epoch: int,
    reference_node_id: str,
    aggregated_proof_hash: str,
) -> str:
    payload: dict[str, _JSONValue] = {
        "aggregation_inputs": tuple(item.to_dict() for item in aggregation_inputs),
        "policy_snapshot": policy.to_dict(),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "aggregated_proof_hash": aggregated_proof_hash,
        "schema_version": DISTRIBUTED_PROOF_AGGREGATION_SCHEMA_VERSION,
    }
    return sha256_hex(payload)


def _build_actions(
    ordered_inputs: tuple[AggregationInput, ...],
    statuses: tuple[AggregationNodeStatus, ...],
    *,
    reference_node_id: str,
    aggregation_ready: bool,
) -> tuple[AggregationAction, ...]:
    actions: list[AggregationAction] = []

    for item in ordered_inputs:
        actions.append(
            AggregationAction(
                action_index=len(actions),
                action_type="compare_proof_bundle",
                source_node_id=reference_node_id,
                target_node_id=item.node_id,
                blocking=False,
                ready=True,
                detail=f"compare reference bundle with {item.node_id}",
            )
        )

    for status in statuses:
        actions.append(
            AggregationAction(
                action_index=len(actions),
                action_type="filter_node",
                source_node_id=reference_node_id,
                target_node_id=status.node_id,
                blocking=not status.admissible,
                ready=True,
                detail=("node passes admissibility" if status.admissible else "node fails admissibility"),
            )
        )

    for status in statuses:
        if status.contributes_to_aggregation:
            actions.append(
                AggregationAction(
                    action_index=len(actions),
                    action_type="include_node",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=False,
                    ready=True,
                    detail="node included in aggregation set",
                )
            )

    for status in statuses:
        if not status.contributes_to_aggregation:
            actions.append(
                AggregationAction(
                    action_index=len(actions),
                    action_type="exclude_node",
                    source_node_id=reference_node_id,
                    target_node_id=status.node_id,
                    blocking=True,
                    ready=True,
                    detail="node excluded from aggregation set",
                )
            )

    actions.append(
        AggregationAction(
            action_index=len(actions),
            action_type="aggregate_proof",
            source_node_id=reference_node_id,
            target_node_id=reference_node_id,
            blocking=not aggregation_ready,
            ready=aggregation_ready,
            detail=(
                "global proof identity aggregated" if aggregation_ready else "global proof identity withheld pending policy"
            ),
        )
    )
    actions.append(
        AggregationAction(
            action_index=len(actions),
            action_type="emit_global_proof",
            source_node_id=reference_node_id,
            target_node_id=reference_node_id,
            blocking=not aggregation_ready,
            ready=aggregation_ready,
            detail=("aggregation receipt emitted" if aggregation_ready else "aggregation receipt emitted not-ready"),
        )
    )

    return tuple(actions)


def run_distributed_proof_aggregation(
    aggregation_inputs: tuple[AggregationInput, ...],
    policy: AggregationPolicy,
) -> DistributedProofAggregationReceipt:
    ordered_inputs = _validate_inputs(aggregation_inputs)
    if not isinstance(policy, AggregationPolicy):
        raise ValueError("policy must be AggregationPolicy")

    cluster_epoch = _compute_cluster_epoch(ordered_inputs)
    reference_node = _select_reference_node(ordered_inputs, policy, cluster_epoch)

    statuses: list[AggregationNodeStatus] = []
    included_nodes: list[AggregationInput] = []

    for item in ordered_inputs:
        admissible, epoch_aligned, confidence_ok, risk_ok, reasons = _admissibility(item, policy, cluster_epoch)
        contributes = admissible
        if contributes:
            included_nodes.append(item)
        statuses.append(
            AggregationNodeStatus(
                node_id=item.node_id,
                admissible=admissible,
                epoch_aligned=epoch_aligned,
                confidence_ok=confidence_ok,
                risk_ok=risk_ok,
                contributes_to_aggregation=contributes,
                aggregation_weight=(1.0 if contributes else 0.0),
                reasons=reasons,
            )
        )

    status_tuple = tuple(statuses)
    included_tuple = tuple(included_nodes)

    aggregated_proof_hash = _compute_aggregated_proof_hash(included_tuple)
    participation_fraction = float(len(included_tuple) / len(ordered_inputs))

    partial_policy_ok = policy.allow_partial_aggregation or all(
        status.admissible and status.contributes_to_aggregation for status in status_tuple
    )
    structurally_consistent = partial_policy_ok

    if included_tuple:
        aggregation_confidence = float(sum(node.consensus_confidence for node in included_tuple) / len(included_tuple))
        aggregation_risk = float(max(node.consensus_risk for node in included_tuple))
    else:
        aggregation_confidence = 0.0
        aggregation_risk = 1.0

    has_admissible_contributor = any(
        status.admissible and status.contributes_to_aggregation for status in status_tuple
    )
    aggregation_ready = (
        structurally_consistent
        and len(included_tuple) > 0
        and has_admissible_contributor
        and participation_fraction >= policy.minimum_participation_fraction
    )

    rationale: list[str] = ["reference proof node selected deterministically"]
    if any((not status.confidence_ok) for status in status_tuple):
        rationale.append("node excluded due to low confidence")
    if any((not status.risk_ok) for status in status_tuple):
        rationale.append("node excluded due to high risk")
    if policy.require_epoch_alignment and any((not status.epoch_aligned) for status in status_tuple):
        rationale.append("epoch mismatch disallowed by policy")
    if policy.allow_partial_aggregation:
        rationale.append("partial aggregation allowed by policy")
    if aggregation_ready:
        rationale.append("global proof aggregation complete")
    else:
        rationale.append("global proof aggregation incomplete")

    actions = _build_actions(
        ordered_inputs,
        status_tuple,
        reference_node_id=reference_node.node_id,
        aggregation_ready=aggregation_ready,
    )

    replay_identity = _compute_replay_identity(
        ordered_inputs,
        policy,
        cluster_epoch,
        reference_node.node_id,
        aggregated_proof_hash,
    )

    stable_hash = sha256_hex(
        _build_receipt_hash_payload(
            aggregation_inputs=ordered_inputs,
            policy_snapshot=policy,
            node_statuses=status_tuple,
            aggregation_actions=actions,
            cluster_epoch=cluster_epoch,
            reference_node_id=reference_node.node_id,
            aggregated_proof_hash=aggregated_proof_hash,
            participation_fraction=participation_fraction,
            structurally_consistent=structurally_consistent,
            aggregation_ready=aggregation_ready,
            aggregation_confidence=aggregation_confidence,
            aggregation_risk=aggregation_risk,
            rationale=tuple(rationale),
            schema_version=DISTRIBUTED_PROOF_AGGREGATION_SCHEMA_VERSION,
            replay_identity=replay_identity,
        )
    )
    return DistributedProofAggregationReceipt(
        aggregation_inputs=ordered_inputs,
        policy_snapshot=policy,
        node_statuses=status_tuple,
        aggregation_actions=actions,
        cluster_epoch=cluster_epoch,
        reference_node_id=reference_node.node_id,
        aggregated_proof_hash=aggregated_proof_hash,
        participation_fraction=participation_fraction,
        structurally_consistent=structurally_consistent,
        aggregation_ready=aggregation_ready,
        aggregation_confidence=aggregation_confidence,
        aggregation_risk=aggregation_risk,
        rationale=tuple(rationale),
        schema_version=DISTRIBUTED_PROOF_AGGREGATION_SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


def export_distributed_proof_aggregation_bytes(receipt: DistributedProofAggregationReceipt) -> bytes:
    if not isinstance(receipt, DistributedProofAggregationReceipt):
        raise ValueError("receipt must be DistributedProofAggregationReceipt")
    return canonical_bytes(receipt.to_dict())


__all__ = [
    "ALLOWED_AGGREGATION_ACTION_TYPES",
    "DISTRIBUTED_PROOF_AGGREGATION_SCHEMA_VERSION",
    "AggregationInput",
    "AggregationPolicy",
    "AggregationNodeStatus",
    "AggregationAction",
    "DistributedProofAggregationReceipt",
    "run_distributed_proof_aggregation",
    "export_distributed_proof_aggregation_bytes",
]
