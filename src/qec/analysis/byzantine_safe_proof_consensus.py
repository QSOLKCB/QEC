"""v139.2 — Byzantine-Safe Proof Consensus.

Deterministic analysis-only proof consensus receipt generation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping
from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

ALLOWED_PROOF_VERDICTS: tuple[str, ...] = ("accept", "reject", "uncertain")
ALLOWED_CONSENSUS_ACTION_TYPES: tuple[str, ...] = (
    "compare_bundle",
    "compare_verdicts",
    "flag_contradiction",
    "align_epoch",
    "hold_node",
    "admit_proof",
    "emit_proof_view",
)
BYZANTINE_SAFE_PROOF_CONSENSUS_SCHEMA_VERSION = "v139.2"


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


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


@dataclass(frozen=True)
class ProofClaim:
    claim_id: str
    claim_type: str
    proof_hash: str
    proof_subject: str
    proof_verdict: str
    confidence_score: float
    replay_identity: str
    payload_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.claim_id, "claim_id")
        _validate_non_empty_str(self.claim_type, "claim_type")
        _validate_non_empty_str(self.proof_subject, "proof_subject")
        if self.proof_verdict not in ALLOWED_PROOF_VERDICTS:
            raise ValueError("proof_verdict must be one of accept/reject/uncertain")
        object.__setattr__(self, "confidence_score", _validate_fraction(self.confidence_score, "confidence_score"))
        if not _is_sha256_hex(self.proof_hash):
            raise ValueError("proof_hash must be 64-char lowercase sha256 hex")
        if not _is_sha256_hex(self.replay_identity):
            raise ValueError("replay_identity must be 64-char lowercase sha256 hex")
        if not _is_sha256_hex(self.payload_hash):
            raise ValueError("payload_hash must be 64-char lowercase sha256 hex")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type,
            "proof_hash": self.proof_hash,
            "proof_subject": self.proof_subject,
            "proof_verdict": self.proof_verdict,
            "confidence_score": self.confidence_score,
            "replay_identity": self.replay_identity,
            "payload_hash": self.payload_hash,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeProofBundle:
    node_id: str
    node_role: str
    epoch_index: int
    proof_claims: tuple[ProofClaim, ...]
    bundle_hash: str
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        _validate_non_empty_str(self.node_role, "node_role")
        if isinstance(self.epoch_index, bool) or not isinstance(self.epoch_index, int) or self.epoch_index < 0:
            raise ValueError("epoch_index must be int >= 0")
        if not isinstance(self.proof_claims, tuple) or not self.proof_claims:
            raise ValueError("proof_claims must be a non-empty tuple")
        if any(not isinstance(claim, ProofClaim) for claim in self.proof_claims):
            raise ValueError("proof_claims must contain ProofClaim values")
        claim_ids = tuple(c.claim_id for c in self.proof_claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("duplicate claim_id within bundle")
        if claim_ids != tuple(sorted(claim_ids)):
            raise ValueError("proof_claims must be ordered by claim_id ascending")
        if not _is_sha256_hex(self.bundle_hash):
            raise ValueError("bundle_hash must be 64-char lowercase sha256 hex")

        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise ValueError("metadata must be Mapping[str, str] | None")
            normalized = {k: v for k, v in self.metadata.items()}
            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in normalized.items()):
                raise ValueError("metadata must be string->string only")
            object.__setattr__(self, "metadata", MappingProxyType(dict(sorted(normalized.items()))))

    def to_dict(self) -> dict[str, _JSONValue]:
        out: dict[str, _JSONValue] = {
            "node_id": self.node_id,
            "node_role": self.node_role,
            "epoch_index": self.epoch_index,
            "proof_claims": tuple(claim.to_dict() for claim in self.proof_claims),
            "bundle_hash": self.bundle_hash,
            "metadata": None,
        }
        if self.metadata is not None:
            out["metadata"] = {k: self.metadata[k] for k in sorted(self.metadata.keys())}
        return out

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ProofConsensusPolicy:
    require_matching_epoch: bool
    allow_role_mixing: bool
    minimum_quorum_fraction: float
    minimum_confidence_score: float
    require_unanimous_verdict: bool
    allow_uncertain_claims: bool
    maximum_claim_divergence_fraction: float

    def __post_init__(self) -> None:
        _validate_bool(self.require_matching_epoch, "require_matching_epoch")
        _validate_bool(self.allow_role_mixing, "allow_role_mixing")
        _validate_bool(self.require_unanimous_verdict, "require_unanimous_verdict")
        _validate_bool(self.allow_uncertain_claims, "allow_uncertain_claims")
        object.__setattr__(
            self,
            "minimum_quorum_fraction",
            _validate_fraction(self.minimum_quorum_fraction, "minimum_quorum_fraction"),
        )
        object.__setattr__(
            self,
            "minimum_confidence_score",
            _validate_fraction(self.minimum_confidence_score, "minimum_confidence_score"),
        )
        object.__setattr__(
            self,
            "maximum_claim_divergence_fraction",
            _validate_fraction(self.maximum_claim_divergence_fraction, "maximum_claim_divergence_fraction"),
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "require_matching_epoch": self.require_matching_epoch,
            "allow_role_mixing": self.allow_role_mixing,
            "minimum_quorum_fraction": self.minimum_quorum_fraction,
            "minimum_confidence_score": self.minimum_confidence_score,
            "require_unanimous_verdict": self.require_unanimous_verdict,
            "allow_uncertain_claims": self.allow_uncertain_claims,
            "maximum_claim_divergence_fraction": self.maximum_claim_divergence_fraction,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class NodeProofConsensusStatus:
    node_id: str
    admissible: bool
    epoch_aligned: bool
    role_aligned: bool
    bundle_hash_aligned: bool
    verdict_aligned: bool
    confidence_ok: bool
    divergence_ok: bool
    matched_claim_fraction: float
    contradiction_fraction: float
    consensus_confidence: float
    consensus_risk: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.node_id, "node_id")
        for name in (
            "admissible",
            "epoch_aligned",
            "role_aligned",
            "bundle_hash_aligned",
            "verdict_aligned",
            "confidence_ok",
            "divergence_ok",
        ):
            _validate_bool(getattr(self, name), name)
        for name in (
            "matched_claim_fraction",
            "contradiction_fraction",
            "consensus_confidence",
            "consensus_risk",
        ):
            object.__setattr__(self, name, _validate_fraction(getattr(self, name), name))
        if not isinstance(self.reasons, tuple) or any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("reasons must be tuple[str, ...] of non-empty strings")

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "node_id": self.node_id,
            "admissible": self.admissible,
            "epoch_aligned": self.epoch_aligned,
            "role_aligned": self.role_aligned,
            "bundle_hash_aligned": self.bundle_hash_aligned,
            "verdict_aligned": self.verdict_aligned,
            "confidence_ok": self.confidence_ok,
            "divergence_ok": self.divergence_ok,
            "matched_claim_fraction": self.matched_claim_fraction,
            "contradiction_fraction": self.contradiction_fraction,
            "consensus_confidence": self.consensus_confidence,
            "consensus_risk": self.consensus_risk,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ProofConsensusAction:
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
        if self.action_type not in ALLOWED_CONSENSUS_ACTION_TYPES:
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

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self.to_dict())


@dataclass(frozen=True)
class ByzantineProofConsensusReceipt:
    node_proof_bundles: tuple[NodeProofBundle, ...]
    policy_snapshot: ProofConsensusPolicy
    node_statuses: tuple[NodeProofConsensusStatus, ...]
    consensus_actions: tuple[ProofConsensusAction, ...]
    cluster_epoch: int
    reference_node_id: str
    reference_bundle_hash: str
    structurally_consistent: bool
    consensus_ready: bool
    consensus_confidence: float
    consensus_risk: float
    rationale: tuple[str, ...]
    schema_version: str
    replay_identity: str
    stable_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.node_proof_bundles, tuple) or not self.node_proof_bundles:
            raise ValueError("node_proof_bundles must be non-empty tuple")
        if any(not isinstance(item, NodeProofBundle) for item in self.node_proof_bundles):
            raise ValueError("node_proof_bundles must contain NodeProofBundle")
        if not isinstance(self.policy_snapshot, ProofConsensusPolicy):
            raise ValueError("policy_snapshot must be ProofConsensusPolicy")
        if not isinstance(self.node_statuses, tuple) or any(not isinstance(item, NodeProofConsensusStatus) for item in self.node_statuses):
            raise ValueError("node_statuses must be tuple[NodeProofConsensusStatus, ...]")
        if not isinstance(self.consensus_actions, tuple) or any(not isinstance(item, ProofConsensusAction) for item in self.consensus_actions):
            raise ValueError("consensus_actions must be tuple[ProofConsensusAction, ...]")
        if isinstance(self.cluster_epoch, bool) or not isinstance(self.cluster_epoch, int) or self.cluster_epoch < 0:
            raise ValueError("cluster_epoch must be int >= 0")
        _validate_non_empty_str(self.reference_node_id, "reference_node_id")
        if not _is_sha256_hex(self.reference_bundle_hash):
            raise ValueError("reference_bundle_hash must be 64-char lowercase sha256 hex")
        _validate_bool(self.structurally_consistent, "structurally_consistent")
        _validate_bool(self.consensus_ready, "consensus_ready")
        object.__setattr__(self, "consensus_confidence", _validate_fraction(self.consensus_confidence, "consensus_confidence"))
        object.__setattr__(self, "consensus_risk", _validate_fraction(self.consensus_risk, "consensus_risk"))
        if not isinstance(self.rationale, tuple) or any(not isinstance(item, str) or not item for item in self.rationale):
            raise ValueError("rationale must be tuple[str, ...] of non-empty strings")
        _validate_non_empty_str(self.schema_version, "schema_version")
        if not _is_sha256_hex(self.replay_identity):
            raise ValueError("replay_identity must be 64-char lowercase sha256 hex")
        expected_replay_identity = _compute_replay_identity(
            self.node_proof_bundles,
            self.policy_snapshot,
            self.cluster_epoch,
            self.reference_node_id,
            self.reference_bundle_hash,
        )
        if self.replay_identity != expected_replay_identity:
            raise ValueError("replay_identity mismatch with receipt contents")
        if not _is_sha256_hex(self.stable_hash):
            raise ValueError("stable_hash must be 64-char lowercase sha256 hex")
        if self.stable_hash_value() != self.stable_hash:
            raise ValueError("stable_hash must match stable_hash_value")

    def _hash_payload(self) -> dict[str, _JSONValue]:
        return _build_receipt_hash_payload(
            node_proof_bundles=self.node_proof_bundles,
            policy_snapshot=self.policy_snapshot,
            node_statuses=self.node_statuses,
            consensus_actions=self.consensus_actions,
            cluster_epoch=self.cluster_epoch,
            reference_node_id=self.reference_node_id,
            reference_bundle_hash=self.reference_bundle_hash,
            structurally_consistent=self.structurally_consistent,
            consensus_ready=self.consensus_ready,
            consensus_confidence=self.consensus_confidence,
            consensus_risk=self.consensus_risk,
            rationale=self.rationale,
            schema_version=self.schema_version,
            replay_identity=self.replay_identity,
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        out = self._hash_payload()
        out["stable_hash"] = self.stable_hash
        return out

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return sha256_hex(self._hash_payload())


def _build_receipt_hash_payload(
    *,
    node_proof_bundles: tuple[NodeProofBundle, ...],
    policy_snapshot: ProofConsensusPolicy,
    node_statuses: tuple[NodeProofConsensusStatus, ...],
    consensus_actions: tuple[ProofConsensusAction, ...],
    cluster_epoch: int,
    reference_node_id: str,
    reference_bundle_hash: str,
    structurally_consistent: bool,
    consensus_ready: bool,
    consensus_confidence: float,
    consensus_risk: float,
    rationale: tuple[str, ...],
    schema_version: str,
    replay_identity: str,
) -> dict[str, _JSONValue]:
    return {
        "node_proof_bundles": tuple(item.to_dict() for item in node_proof_bundles),
        "policy_snapshot": policy_snapshot.to_dict(),
        "node_statuses": tuple(item.to_dict() for item in node_statuses),
        "consensus_actions": tuple(item.to_dict() for item in consensus_actions),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "reference_bundle_hash": reference_bundle_hash,
        "structurally_consistent": structurally_consistent,
        "consensus_ready": consensus_ready,
        "consensus_confidence": consensus_confidence,
        "consensus_risk": consensus_risk,
        "rationale": rationale,
        "schema_version": schema_version,
        "replay_identity": replay_identity,
    }


def _cluster_epoch(bundles: tuple[NodeProofBundle, ...]) -> int:
    counts = Counter(bundle.epoch_index for bundle in bundles)
    return min((-count, epoch) for epoch, count in counts.items())[1]


def _average_confidence(bundle: NodeProofBundle) -> float:
    return float(sum(claim.confidence_score for claim in bundle.proof_claims) / len(bundle.proof_claims))


def _select_reference_bundle(
    bundles: tuple[NodeProofBundle, ...],
    *,
    cluster_epoch: int,
    policy: ProofConsensusPolicy,
) -> NodeProofBundle:
    eligible = tuple(bundle for bundle in bundles if (not policy.require_matching_epoch or bundle.epoch_index == cluster_epoch))
    pool = eligible if eligible else bundles
    ranked = sorted(
        pool,
        key=lambda b: (-_average_confidence(b), -len(b.proof_claims), b.bundle_hash, b.node_id),
    )
    return ranked[0]


def _compare_bundle(
    node_bundle: NodeProofBundle,
    reference_bundle: NodeProofBundle,
) -> tuple[float, float, bool, bool]:
    reference = {claim.claim_id: claim for claim in reference_bundle.proof_claims}
    comparable = 0
    matched = 0
    contradiction = 0
    uncertain_shared = False
    for claim in node_bundle.proof_claims:
        ref_claim = reference.get(claim.claim_id)
        if ref_claim is None or claim.proof_subject != ref_claim.proof_subject:
            continue
        comparable += 1
        if claim.proof_verdict == ref_claim.proof_verdict:
            matched += 1
        pair = {claim.proof_verdict, ref_claim.proof_verdict}
        if pair == {"accept", "reject"}:
            contradiction += 1
        if "uncertain" in pair:
            uncertain_shared = True
    if comparable == 0:
        return 0.0, 0.0, uncertain_shared, False
    return matched / comparable, contradiction / comparable, uncertain_shared, True


def _consensus_verdict_for_counts(counts: Mapping[str, int], *, require_unanimous: bool) -> tuple[str, bool]:
    present = tuple(sorted((v, c) for v, c in counts.items() if c > 0))
    if not present:
        return "uncertain", True
    if require_unanimous:
        unique = tuple(v for v, c in present if c > 0)
        if len(unique) == 1:
            return unique[0], True
        return "uncertain", False

    max_count = max(c for _, c in present)
    leaders = tuple(sorted(v for v, c in present if c == max_count))
    if len(leaders) == 1:
        return leaders[0], True
    if "uncertain" in leaders:
        return "uncertain", True
    return leaders[0], True


def _build_claim_consensus(
    bundles: tuple[NodeProofBundle, ...],
    *,
    require_unanimous: bool,
) -> tuple[dict[tuple[str, str], str], bool]:
    tallies: dict[tuple[str, str], dict[str, int]] = {}
    for bundle in bundles:
        for claim in bundle.proof_claims:
            key = (claim.claim_id, claim.proof_subject)
            if key not in tallies:
                tallies[key] = {"accept": 0, "reject": 0, "uncertain": 0}
            tallies[key][claim.proof_verdict] += 1

    verdict_map: dict[tuple[str, str], str] = {}
    verdict_rule_satisfied = True
    for key in sorted(tallies.keys()):
        verdict, satisfied = _consensus_verdict_for_counts(tallies[key], require_unanimous=require_unanimous)
        verdict_map[key] = verdict
        verdict_rule_satisfied = verdict_rule_satisfied and satisfied
    return verdict_map, verdict_rule_satisfied


def _validate_input_bundles(node_proof_bundles: tuple[NodeProofBundle, ...]) -> tuple[NodeProofBundle, ...]:
    if not isinstance(node_proof_bundles, tuple) or not node_proof_bundles:
        raise ValueError("node_proof_bundles must be a non-empty tuple")
    if any(not isinstance(bundle, NodeProofBundle) for bundle in node_proof_bundles):
        raise ValueError("node_proof_bundles must contain NodeProofBundle values")
    sorted_bundles = tuple(sorted(node_proof_bundles, key=lambda b: b.node_id))
    ids = tuple(bundle.node_id for bundle in sorted_bundles)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate node_id")
    return sorted_bundles


def _build_replay_identity_payload(
    bundles: tuple[NodeProofBundle, ...],
    policy: ProofConsensusPolicy,
    cluster_epoch: int,
    reference_node_id: str,
    reference_bundle_hash: str,
) -> dict[str, _JSONValue]:
    return {
        "node_proof_bundles": tuple(bundle.to_dict() for bundle in bundles),
        "policy_snapshot": policy.to_dict(),
        "cluster_epoch": cluster_epoch,
        "reference_node_id": reference_node_id,
        "reference_bundle_hash": reference_bundle_hash,
        "schema_version": BYZANTINE_SAFE_PROOF_CONSENSUS_SCHEMA_VERSION,
    }


def _compute_replay_identity(
    bundles: tuple[NodeProofBundle, ...],
    policy: ProofConsensusPolicy,
    cluster_epoch: int,
    reference_node_id: str,
    reference_bundle_hash: str,
) -> str:
    return sha256_hex(
        _build_replay_identity_payload(
            bundles,
            policy,
            cluster_epoch,
            reference_node_id,
            reference_bundle_hash,
        )
    )


def _compute_node_proof_status(
    bundle: NodeProofBundle,
    reference_bundle: NodeProofBundle,
    policy: ProofConsensusPolicy,
    cluster_epoch: int,
) -> dict[str, Any]:
    matched_fraction, contradiction_fraction, uncertain_shared, has_comparable = _compare_bundle(bundle, reference_bundle)
    epoch_aligned = bundle.epoch_index == cluster_epoch
    role_aligned = policy.allow_role_mixing or bundle.node_role == reference_bundle.node_role
    bundle_hash_aligned = bundle.bundle_hash == reference_bundle.bundle_hash
    avg_confidence = _average_confidence(bundle)
    confidence_ok = avg_confidence >= policy.minimum_confidence_score
    divergence_ok = contradiction_fraction <= policy.maximum_claim_divergence_fraction

    reasons: list[str] = []
    reasons.append("epoch aligned" if epoch_aligned else "epoch mismatch")
    reasons.append("role aligned" if role_aligned else "role mismatch")
    reasons.append("confidence threshold met" if confidence_ok else "confidence below threshold")
    reasons.append("contradiction within bounds" if divergence_ok else "contradiction exceeds bounds")
    if uncertain_shared:
        reasons.append("uncertain claims present in shared comparison")
    if not has_comparable:
        reasons.append("no comparable claims with reference")

    admissible = (
        has_comparable
        and (epoch_aligned or not policy.require_matching_epoch)
        and role_aligned
        and confidence_ok
        and divergence_ok
        and (policy.allow_uncertain_claims or not uncertain_shared)
    )

    confidence = _clamp01((matched_fraction + (1.0 - contradiction_fraction) + avg_confidence) / 3.0)
    if uncertain_shared and not policy.allow_uncertain_claims:
        confidence = _clamp01(confidence * 0.5)

    return {
        "bundle": bundle,
        "epoch_aligned": epoch_aligned,
        "role_aligned": role_aligned,
        "bundle_hash_aligned": bundle_hash_aligned,
        "matched_claim_fraction": matched_fraction,
        "contradiction_fraction": contradiction_fraction,
        "confidence_ok": confidence_ok,
        "divergence_ok": divergence_ok,
        "uncertain_shared": uncertain_shared,
        "has_comparable": has_comparable,
        "admissible": admissible,
        "consensus_confidence": confidence,
        "consensus_risk": _clamp01(1.0 - confidence),
        "reasons": tuple(reasons),
    }


def _evaluate_verdict_consensus(
    provisional: tuple[dict[str, Any], ...],
    policy: ProofConsensusPolicy,
) -> tuple[tuple[NodeProofConsensusStatus, ...], tuple[NodeProofBundle, ...], bool]:
    admissible_bundles = tuple(item["bundle"] for item in provisional if item["admissible"])
    verdict_map, verdict_rule_satisfied = _build_claim_consensus(
        admissible_bundles,
        require_unanimous=policy.require_unanimous_verdict,
    )

    statuses: list[NodeProofConsensusStatus] = []
    for item in provisional:
        bundle = item["bundle"]
        node_verdict_aligned = True
        for claim in bundle.proof_claims:
            key = (claim.claim_id, claim.proof_subject)
            expected = verdict_map.get(key)
            if expected is not None and claim.proof_verdict != expected:
                node_verdict_aligned = False
                break

        statuses.append(
            NodeProofConsensusStatus(
                node_id=bundle.node_id,
                admissible=item["admissible"],
                epoch_aligned=item["epoch_aligned"],
                role_aligned=item["role_aligned"],
                bundle_hash_aligned=item["bundle_hash_aligned"],
                verdict_aligned=node_verdict_aligned,
                confidence_ok=item["confidence_ok"],
                divergence_ok=item["divergence_ok"],
                matched_claim_fraction=item["matched_claim_fraction"],
                contradiction_fraction=item["contradiction_fraction"],
                consensus_confidence=item["consensus_confidence"],
                consensus_risk=item["consensus_risk"],
                reasons=item["reasons"],
            )
        )
    return tuple(statuses), admissible_bundles, verdict_rule_satisfied


def _compute_structural_consistency(
    statuses: tuple[NodeProofConsensusStatus, ...],
    provisional: tuple[dict[str, Any], ...],
    admissible_bundles: tuple[NodeProofBundle, ...],
    verdict_rule_satisfied: bool,
    policy: ProofConsensusPolicy,
    bundle_count: int,
) -> tuple[bool, bool, bool, bool, bool, float]:
    quorum_fraction = len(admissible_bundles) / bundle_count
    quorum_ok = quorum_fraction >= policy.minimum_quorum_fraction

    all_epoch_ok = all((s.epoch_aligned or not policy.require_matching_epoch) for s in statuses if s.admissible)
    all_role_ok = all(s.role_aligned for s in statuses if s.admissible)
    all_divergence_ok = all(s.divergence_ok for s in statuses if s.admissible)
    all_comparable_ok = all(item["has_comparable"] for item in provisional)
    structurally_consistent = (
        bool(admissible_bundles)
        and all_divergence_ok
        and all_comparable_ok
        and (verdict_rule_satisfied if policy.require_unanimous_verdict else True)
    )
    consensus_ready = (
        structurally_consistent
        and quorum_ok
        and all_epoch_ok
        and all_role_ok
        and all_divergence_ok
        and verdict_rule_satisfied
    )
    return structurally_consistent, consensus_ready, all_epoch_ok, all_role_ok, all_divergence_ok, quorum_fraction


def _build_consensus_rationale(
    *,
    policy: ProofConsensusPolicy,
    all_epoch_ok: bool,
    all_role_ok: bool,
    all_divergence_ok: bool,
    quorum_ok: bool,
    verdict_rule_satisfied: bool,
    consensus_ready: bool,
) -> tuple[str, ...]:
    rationale: list[str] = ["reference proof bundle selected deterministically"]
    rationale.append("epoch alignment satisfied" if all_epoch_ok else "epoch alignment unsatisfied")
    rationale.append("role mixing allowed by policy" if policy.allow_role_mixing else ("role alignment satisfied" if all_role_ok else "role mismatch disallowed by policy"))
    rationale.append("proof verdict agreement satisfies policy" if verdict_rule_satisfied else "proof verdict agreement violates policy")
    rationale.append("contradiction fraction within policy maximum" if all_divergence_ok else "contradiction fraction exceeds policy maximum")
    rationale.append("quorum fraction satisfies policy" if quorum_ok else "quorum fraction below policy minimum")
    rationale.append("proof consensus ready" if consensus_ready else "proof consensus not ready")
    return tuple(rationale)


def _build_consensus_actions(
    statuses: tuple[NodeProofConsensusStatus, ...],
    reference_bundle: NodeProofBundle,
    policy: ProofConsensusPolicy,
    consensus_ready: bool,
) -> tuple[ProofConsensusAction, ...]:
    actions: list[ProofConsensusAction] = []
    next_index = 0
    for status in statuses:
        actions.append(
            ProofConsensusAction(
                action_index=next_index,
                action_type="compare_bundle",
                source_node_id=reference_bundle.node_id,
                target_node_id=status.node_id,
                blocking=False,
                ready=True,
                detail="compare node bundle against deterministic reference bundle",
            )
        )
        next_index += 1
        actions.append(
            ProofConsensusAction(
                action_index=next_index,
                action_type="compare_verdicts",
                source_node_id=reference_bundle.node_id,
                target_node_id=status.node_id,
                blocking=False,
                ready=True,
                detail="compare shared claim verdicts against policy consensus semantics",
            )
        )
        next_index += 1
        if not status.epoch_aligned:
            actions.append(
                ProofConsensusAction(
                    action_index=next_index,
                    action_type="align_epoch",
                    source_node_id=status.node_id,
                    target_node_id=reference_bundle.node_id,
                    blocking=policy.require_matching_epoch,
                    ready=not policy.require_matching_epoch,
                    detail="epoch mismatch requires policy-governed alignment",
                )
            )
            next_index += 1
        if status.contradiction_fraction > 0.0:
            actions.append(
                ProofConsensusAction(
                    action_index=next_index,
                    action_type="flag_contradiction",
                    source_node_id=status.node_id,
                    target_node_id=reference_bundle.node_id,
                    blocking=status.contradiction_fraction > policy.maximum_claim_divergence_fraction,
                    ready=status.contradiction_fraction <= policy.maximum_claim_divergence_fraction,
                    detail="shared claims include contradiction against reference verdicts",
                )
            )
            next_index += 1
        actions.append(
            ProofConsensusAction(
                action_index=next_index,
                action_type="admit_proof" if status.admissible else "hold_node",
                source_node_id=status.node_id,
                target_node_id=reference_bundle.node_id,
                blocking=not status.admissible,
                ready=status.admissible,
                detail="bundle admissibility derived from deterministic policy checks",
            )
        )
        next_index += 1
    actions.append(
        ProofConsensusAction(
            action_index=next_index,
            action_type="emit_proof_view",
            source_node_id=reference_bundle.node_id,
            target_node_id=reference_bundle.node_id,
            blocking=not consensus_ready,
            ready=consensus_ready,
            detail="emit advisory proof consensus view",
        )
    )
    return tuple(actions)


def run_byzantine_safe_proof_consensus(
    node_proof_bundles: tuple[NodeProofBundle, ...],
    policy: ProofConsensusPolicy,
) -> ByzantineProofConsensusReceipt:
    bundles = _validate_input_bundles(node_proof_bundles)
    if not isinstance(policy, ProofConsensusPolicy):
        raise ValueError("policy must be ProofConsensusPolicy")

    cluster_epoch = _cluster_epoch(bundles)
    reference_bundle = _select_reference_bundle(bundles, cluster_epoch=cluster_epoch, policy=policy)

    provisional = tuple(
        _compute_node_proof_status(bundle, reference_bundle, policy, cluster_epoch)
        for bundle in bundles
    )
    statuses_t, admissible_bundles, verdict_rule_satisfied = _evaluate_verdict_consensus(provisional, policy)
    (
        structural_consistent,
        consensus_ready,
        all_epoch_ok,
        all_role_ok,
        all_divergence_ok,
        quorum_fraction,
    ) = _compute_structural_consistency(
        statuses_t,
        provisional,
        admissible_bundles,
        verdict_rule_satisfied,
        policy,
        len(bundles),
    )
    quorum_ok = quorum_fraction >= policy.minimum_quorum_fraction
    rationale_tuple = _build_consensus_rationale(
        policy=policy,
        all_epoch_ok=all_epoch_ok,
        all_role_ok=all_role_ok,
        all_divergence_ok=all_divergence_ok,
        quorum_ok=quorum_ok,
        verdict_rule_satisfied=verdict_rule_satisfied,
        consensus_ready=consensus_ready,
    )
    actions_tuple = _build_consensus_actions(statuses_t, reference_bundle, policy, consensus_ready)

    consensus_confidence = _clamp01(
        sum(status.consensus_confidence for status in statuses_t if status.admissible) / max(1, len(admissible_bundles))
    )
    consensus_confidence = _clamp01(consensus_confidence * quorum_fraction)
    consensus_risk = _clamp01(1.0 - consensus_confidence)

    replay_identity = _compute_replay_identity(
        bundles,
        policy,
        cluster_epoch,
        reference_bundle.node_id,
        reference_bundle.bundle_hash,
    )

    stable_hash = sha256_hex(
        _build_receipt_hash_payload(
            node_proof_bundles=bundles,
            policy_snapshot=policy,
            node_statuses=statuses_t,
            consensus_actions=actions_tuple,
            cluster_epoch=cluster_epoch,
            reference_node_id=reference_bundle.node_id,
            reference_bundle_hash=reference_bundle.bundle_hash,
            structurally_consistent=structural_consistent,
            consensus_ready=consensus_ready,
            consensus_confidence=consensus_confidence,
            consensus_risk=consensus_risk,
            rationale=rationale_tuple,
            schema_version=BYZANTINE_SAFE_PROOF_CONSENSUS_SCHEMA_VERSION,
            replay_identity=replay_identity,
        )
    )
    return ByzantineProofConsensusReceipt(
        node_proof_bundles=bundles,
        policy_snapshot=policy,
        node_statuses=statuses_t,
        consensus_actions=actions_tuple,
        cluster_epoch=cluster_epoch,
        reference_node_id=reference_bundle.node_id,
        reference_bundle_hash=reference_bundle.bundle_hash,
        structurally_consistent=structural_consistent,
        consensus_ready=consensus_ready,
        consensus_confidence=consensus_confidence,
        consensus_risk=consensus_risk,
        rationale=rationale_tuple,
        schema_version=BYZANTINE_SAFE_PROOF_CONSENSUS_SCHEMA_VERSION,
        replay_identity=replay_identity,
        stable_hash=stable_hash,
    )


def export_byzantine_safe_proof_consensus_bytes(receipt: ByzantineProofConsensusReceipt) -> bytes:
    if not isinstance(receipt, ByzantineProofConsensusReceipt):
        raise ValueError("receipt must be ByzantineProofConsensusReceipt")
    return receipt.to_canonical_bytes()
