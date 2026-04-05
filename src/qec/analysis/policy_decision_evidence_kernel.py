"""v137.4.3 — Policy Decision Artifact + Evidence Receipts.

Deterministic Layer 4 policy decision artifact with immutable evidence receipts,
append-only lineage, replay-safe verification, and canonical export.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_POLICY_SCHEMA_VERSION = 1
_RECEIPT_SCHEMA_VERSION = 1
_GENESIS_RECEIPT_PARENT_HASH = "0" * 64


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if normalized == "":
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _validate_hash_hex(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc
    return value


def _canonical_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_canonical_json(item) for item in value)
    if isinstance(value, list):
        return tuple(_canonical_json(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("payload keys must be strings")
        return {k: _canonical_json(value[k]) for k in sorted(value.keys())}
    raise ValueError(f"unsupported payload value type: {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class PolicyDecisionArtifact:
    """Immutable deterministic policy decision artifact."""

    policy_id: str
    decision_verdict: str
    supporting_evidence_hashes: tuple[str, ...]
    parent_provenance_root: str
    originating_sovereignty_event_hash: str
    originating_privilege_decision_hash: str
    stable_decision_hash: str
    schema_version: int = _POLICY_SCHEMA_VERSION


@dataclass(frozen=True)
class EvidenceReceipt:
    """Immutable append-only evidence receipt bound to a policy artifact."""

    index: int
    schema_version: int
    parent_receipt_hash: str
    policy_decision_hash: str
    evidence_hash: str
    evidence_position: int
    receipt_hash: str


def _compute_decision_hash(
    *,
    policy_id: str,
    decision_verdict: str,
    supporting_evidence_hashes: tuple[str, ...],
    parent_provenance_root: str,
    originating_sovereignty_event_hash: str,
    originating_privilege_decision_hash: str,
    schema_version: int,
) -> str:
    payload = {
        "policy_id": policy_id,
        "decision_verdict": decision_verdict,
        "supporting_evidence_hashes": supporting_evidence_hashes,
        "parent_provenance_root": parent_provenance_root,
        "originating_sovereignty_event_hash": originating_sovereignty_event_hash,
        "originating_privilege_decision_hash": originating_privilege_decision_hash,
        "schema_version": schema_version,
    }
    return _sha256_hex(_canonical_json_bytes(payload))


def _compute_receipt_hash(
    *,
    index: int,
    schema_version: int,
    parent_receipt_hash: str,
    policy_decision_hash: str,
    evidence_hash: str,
    evidence_position: int,
) -> str:
    payload = {
        "index": index,
        "schema_version": schema_version,
        "parent_receipt_hash": parent_receipt_hash,
        "policy_decision_hash": policy_decision_hash,
        "evidence_hash": evidence_hash,
        "evidence_position": evidence_position,
    }
    return _sha256_hex(_canonical_json_bytes(payload))


def _normalize_evidence_hashes(evidence_hashes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(evidence_hashes, (str, bytes, bytearray)) or not isinstance(
        evidence_hashes, Sequence
    ):
        raise ValueError(
            "supporting_evidence_hashes must be a sequence of hash strings"
        )
    normalized: list[tuple[str, int]] = []
    for idx, evidence_hash in enumerate(evidence_hashes):
        field_name = f"supporting_evidence_hashes[{idx}]"
        normalized.append((_validate_hash_hex(evidence_hash, field_name=field_name), idx))
    ordered = tuple(item[0] for item in sorted(normalized, key=lambda item: (item[0], item[1])))
    if len(set(ordered)) != len(ordered):
        raise ValueError("supporting_evidence_hashes must be unique")
    return ordered


def create_policy_decision_artifact(
    *,
    policy_id: str,
    decision_verdict: str,
    supporting_evidence_hashes: Sequence[str],
    parent_provenance_root: str,
    originating_sovereignty_event_hash: str,
    originating_privilege_decision_hash: str,
    schema_version: int = _POLICY_SCHEMA_VERSION,
) -> PolicyDecisionArtifact:
    """Create deterministic policy decision artifact bound to evidence hashes."""
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    canonical_policy_id = _validate_non_empty_str(policy_id, field_name="policy_id")
    canonical_verdict = _validate_non_empty_str(decision_verdict, field_name="decision_verdict")
    canonical_evidence = _normalize_evidence_hashes(supporting_evidence_hashes)
    if len(canonical_evidence) == 0:
        raise ValueError("supporting_evidence_hashes must be non-empty")

    parent_root = _validate_hash_hex(parent_provenance_root, field_name="parent_provenance_root")
    sovereignty_hash = _validate_hash_hex(
        originating_sovereignty_event_hash,
        field_name="originating_sovereignty_event_hash",
    )
    privilege_hash = _validate_hash_hex(
        originating_privilege_decision_hash,
        field_name="originating_privilege_decision_hash",
    )

    decision_hash = _compute_decision_hash(
        policy_id=canonical_policy_id,
        decision_verdict=canonical_verdict,
        supporting_evidence_hashes=canonical_evidence,
        parent_provenance_root=parent_root,
        originating_sovereignty_event_hash=sovereignty_hash,
        originating_privilege_decision_hash=privilege_hash,
        schema_version=schema_version,
    )

    return PolicyDecisionArtifact(
        policy_id=canonical_policy_id,
        decision_verdict=canonical_verdict,
        supporting_evidence_hashes=canonical_evidence,
        parent_provenance_root=parent_root,
        originating_sovereignty_event_hash=sovereignty_hash,
        originating_privilege_decision_hash=privilege_hash,
        stable_decision_hash=decision_hash,
        schema_version=schema_version,
    )


def append_evidence_receipt(
    artifact: PolicyDecisionArtifact,
    receipts: Sequence[EvidenceReceipt],
    *,
    evidence_hash: str,
    schema_version: int = _RECEIPT_SCHEMA_VERSION,
) -> tuple[EvidenceReceipt, ...]:
    """Append immutable evidence receipt entry to the receipt lineage."""
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")
    verify_policy_evidence_chain(artifact, receipts)

    validated_evidence_hash = _validate_hash_hex(evidence_hash, field_name="evidence_hash")
    if validated_evidence_hash not in artifact.supporting_evidence_hashes:
        raise ValueError("evidence_hash is not declared by artifact")

    expected_position = len(receipts)
    if expected_position >= len(artifact.supporting_evidence_hashes):
        raise ValueError("all declared evidence receipts are already appended")

    required_evidence_hash = artifact.supporting_evidence_hashes[expected_position]
    if validated_evidence_hash != required_evidence_hash:
        raise ValueError("evidence_hash append order must match canonical artifact ordering")

    parent_hash = receipts[-1].receipt_hash if receipts else _GENESIS_RECEIPT_PARENT_HASH
    receipt_hash = _compute_receipt_hash(
        index=expected_position,
        schema_version=schema_version,
        parent_receipt_hash=parent_hash,
        policy_decision_hash=artifact.stable_decision_hash,
        evidence_hash=validated_evidence_hash,
        evidence_position=expected_position,
    )

    new_receipt = EvidenceReceipt(
        index=expected_position,
        schema_version=schema_version,
        parent_receipt_hash=parent_hash,
        policy_decision_hash=artifact.stable_decision_hash,
        evidence_hash=validated_evidence_hash,
        evidence_position=expected_position,
        receipt_hash=receipt_hash,
    )
    return tuple(receipts) + (new_receipt,)


def verify_policy_evidence_chain(
    artifact: PolicyDecisionArtifact,
    receipts: Sequence[EvidenceReceipt],
) -> bool:
    """Replay-safe verification for policy artifact and evidence receipt lineage."""
    if not isinstance(artifact, PolicyDecisionArtifact):
        raise ValueError("artifact must be a PolicyDecisionArtifact")
    if artifact.schema_version <= 0:
        raise ValueError("artifact schema_version must be positive")

    _validate_non_empty_str(artifact.policy_id, field_name="policy_id")
    _validate_non_empty_str(artifact.decision_verdict, field_name="decision_verdict")
    _validate_hash_hex(artifact.parent_provenance_root, field_name="parent_provenance_root")
    _validate_hash_hex(
        artifact.originating_sovereignty_event_hash,
        field_name="originating_sovereignty_event_hash",
    )
    _validate_hash_hex(
        artifact.originating_privilege_decision_hash,
        field_name="originating_privilege_decision_hash",
    )
    _validate_hash_hex(artifact.stable_decision_hash, field_name="stable_decision_hash")

    canonical_evidence = _normalize_evidence_hashes(artifact.supporting_evidence_hashes)
    if canonical_evidence != artifact.supporting_evidence_hashes:
        raise ValueError("artifact supporting_evidence_hashes must already be canonical")

    expected_decision_hash = _compute_decision_hash(
        policy_id=artifact.policy_id,
        decision_verdict=artifact.decision_verdict,
        supporting_evidence_hashes=artifact.supporting_evidence_hashes,
        parent_provenance_root=artifact.parent_provenance_root,
        originating_sovereignty_event_hash=artifact.originating_sovereignty_event_hash,
        originating_privilege_decision_hash=artifact.originating_privilege_decision_hash,
        schema_version=artifact.schema_version,
    )
    if artifact.stable_decision_hash != expected_decision_hash:
        raise ValueError("stable_decision_hash mismatch")

    if not isinstance(receipts, Sequence):
        raise ValueError("receipts must be a sequence")
    if len(receipts) > len(artifact.supporting_evidence_hashes):
        raise ValueError("receipt count exceeds declared supporting evidence")

    expected_parent = _GENESIS_RECEIPT_PARENT_HASH
    seen_receipt_hashes: set[str] = set()
    for i, receipt in enumerate(receipts):
        if not isinstance(receipt, EvidenceReceipt):
            raise ValueError("receipts must contain EvidenceReceipt entries")
        if receipt.index != i:
            raise ValueError("receipt index sequence is not append-only")
        if receipt.schema_version <= 0:
            raise ValueError("receipt schema_version must be positive")

        _validate_hash_hex(receipt.parent_receipt_hash, field_name="parent_receipt_hash")
        _validate_hash_hex(receipt.policy_decision_hash, field_name="policy_decision_hash")
        _validate_hash_hex(receipt.evidence_hash, field_name="evidence_hash")
        _validate_hash_hex(receipt.receipt_hash, field_name="receipt_hash")

        if receipt.parent_receipt_hash != expected_parent:
            raise ValueError("receipt parent hash mismatch")
        if receipt.policy_decision_hash != artifact.stable_decision_hash:
            raise ValueError("receipt policy_decision_hash mismatch")
        if receipt.evidence_position != i:
            raise ValueError("receipt evidence_position mismatch")
        if receipt.evidence_hash != artifact.supporting_evidence_hashes[i]:
            raise ValueError("receipt evidence hash mismatch against artifact")

        expected_receipt_hash = _compute_receipt_hash(
            index=receipt.index,
            schema_version=receipt.schema_version,
            parent_receipt_hash=receipt.parent_receipt_hash,
            policy_decision_hash=receipt.policy_decision_hash,
            evidence_hash=receipt.evidence_hash,
            evidence_position=receipt.evidence_position,
        )
        if receipt.receipt_hash != expected_receipt_hash:
            raise ValueError("receipt hash mismatch")
        if receipt.receipt_hash in seen_receipt_hashes:
            raise ValueError("replay detected: duplicate receipt hash")

        seen_receipt_hashes.add(receipt.receipt_hash)
        expected_parent = receipt.receipt_hash

    return True


def compute_evidence_root(receipts: Sequence[EvidenceReceipt]) -> str:
    """Compute stable evidence root from append-only receipt lineage."""
    if not isinstance(receipts, Sequence):
        raise ValueError("receipts must be a sequence")
    if len(receipts) == 0:
        return _sha256_hex(b"policy-evidence-empty-root")

    digest = hashlib.sha256()
    for receipt in receipts:
        if not isinstance(receipt, EvidenceReceipt):
            raise ValueError("receipts must contain EvidenceReceipt entries")
        digest.update(receipt.receipt_hash.encode("ascii"))
    return digest.hexdigest()


def export_policy_evidence_bytes(
    artifact: PolicyDecisionArtifact,
    receipts: Sequence[EvidenceReceipt],
) -> bytes:
    """Export canonical policy artifact + evidence receipts bytes."""
    verify_policy_evidence_chain(artifact, receipts)

    payload = {
        "policy_artifact": {
            "policy_id": artifact.policy_id,
            "decision_verdict": artifact.decision_verdict,
            "supporting_evidence_hashes": artifact.supporting_evidence_hashes,
            "parent_provenance_root": artifact.parent_provenance_root,
            "originating_sovereignty_event_hash": artifact.originating_sovereignty_event_hash,
            "originating_privilege_decision_hash": artifact.originating_privilege_decision_hash,
            "stable_decision_hash": artifact.stable_decision_hash,
            "schema_version": artifact.schema_version,
        },
        "evidence_root": compute_evidence_root(receipts),
        "evidence_receipts": [
            {
                "index": receipt.index,
                "schema_version": receipt.schema_version,
                "parent_receipt_hash": receipt.parent_receipt_hash,
                "policy_decision_hash": receipt.policy_decision_hash,
                "evidence_hash": receipt.evidence_hash,
                "evidence_position": receipt.evidence_position,
                "receipt_hash": receipt.receipt_hash,
            }
            for receipt in receipts
        ],
    }
    return _canonical_json_bytes(payload)


def generate_decision_lineage_receipt(
    artifact: PolicyDecisionArtifact,
    receipts: Sequence[EvidenceReceipt],
) -> dict[str, Any]:
    """Generate deterministic lineage receipt for decision + evidence chain."""
    canonical_bytes = export_policy_evidence_bytes(artifact, receipts)
    tip_receipt_hash = receipts[-1].receipt_hash if receipts else _GENESIS_RECEIPT_PARENT_HASH
    return {
        "policy_id": artifact.policy_id,
        "stable_decision_hash": artifact.stable_decision_hash,
        "receipt_count": len(receipts),
        "tip_receipt_hash": tip_receipt_hash,
        "evidence_root": compute_evidence_root(receipts),
        "lineage_digest_sha256": _sha256_hex(canonical_bytes),
    }


__all__ = [
    "PolicyDecisionArtifact",
    "EvidenceReceipt",
    "create_policy_decision_artifact",
    "append_evidence_receipt",
    "verify_policy_evidence_chain",
    "compute_evidence_root",
    "export_policy_evidence_bytes",
    "generate_decision_lineage_receipt",
]
