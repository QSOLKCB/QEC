"""Replay certification battery for v137.4.x sovereignty artifacts.

Produces canonical bytes and stable SHA-256 root hashes for verifying
replayed supervisory artifacts without depending on decoder internals.
All operations are deterministic and replay-safe (same inputs → same bytes).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_VERSION = "v137.4.4"


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_from_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("float values must be finite")
        return float(round(numeric, 12))
    raise ValueError(f"unsupported scalar type: {type(value)!r}")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        original_keys: dict[str, Any] = {}
        for original_key, item in sorted(value.items(), key=lambda entry: str(entry[0])):
            normalized_key = str(original_key)
            if normalized_key in normalized_mapping:
                raise ValueError(
                    f"mapping contains multiple keys that normalize to {normalized_key!r} "
                    f"(original keys: {original_keys[normalized_key]!r} and {original_key!r})"
                )
            original_keys[normalized_key] = original_key
            normalized_mapping[normalized_key] = _normalize_value(item)
        return normalized_mapping
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return _normalize_scalar(value)


def _normalize_records(records: Sequence[Mapping[str, Any]], *, name: str) -> tuple[dict[str, Any], ...]:
    if not records:
        raise ValueError(f"{name} must not be empty")
    normalized: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{name}[{index}] must be a mapping")
        canonical_record = _normalize_value(record)
        if not isinstance(canonical_record, dict):
            raise ValueError(f"{name}[{index}] must normalize to a mapping")
        digest = _sha256_hex_from_mapping(canonical_record)
        if digest in seen_hashes:
            raise ValueError(f"duplicate record detected in {name}")
        seen_hashes.add(digest)
        normalized.append(canonical_record)

    decorated = [
        (_sha256_hex_from_mapping(item), index, item)
        for index, item in enumerate(normalized)
    ]
    decorated.sort(key=lambda entry: (entry[0], entry[1]))
    return tuple(entry[2] for entry in decorated)


def _record_set_hash(records: tuple[dict[str, Any], ...], *, namespace: str) -> str:
    payload = {
        "namespace": namespace,
        "version": _VERSION,
        "records": records,
    }
    return _sha256_hex_from_mapping(payload)


@dataclass(frozen=True)
class ReplayCertificationReport:
    sovereignty_event_hash: str
    capability_decision_hash: str
    provenance_chain_hash: str
    policy_evidence_hash: str
    pre_report_hash: str
    certification_root_hash: str
    sovereignty_event_count: int
    capability_decision_count: int
    provenance_chain_count: int
    policy_evidence_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "sovereignty_event_hash": self.sovereignty_event_hash,
            "capability_decision_hash": self.capability_decision_hash,
            "provenance_chain_hash": self.provenance_chain_hash,
            "policy_evidence_hash": self.policy_evidence_hash,
            "pre_report_hash": self.pre_report_hash,
            "certification_root_hash": self.certification_root_hash,
            "sovereignty_event_count": int(self.sovereignty_event_count),
            "capability_decision_count": int(self.capability_decision_count),
            "provenance_chain_count": int(self.provenance_chain_count),
            "policy_evidence_count": int(self.policy_evidence_count),
            "replay_safe": True,
            "deterministic": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CertificationReceipt:
    certification_root_hash: str
    report_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "certification_root_hash": self.certification_root_hash,
            "report_hash": self.report_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def verify_byte_identity(reference_bytes: bytes, replay_bytes: bytes) -> str:
    if not reference_bytes:
        raise ValueError("reference_bytes must not be empty")
    if not replay_bytes:
        raise ValueError("replay_bytes must not be empty")
    if reference_bytes != replay_bytes:
        raise ValueError("byte identity validation failed")
    return hashlib.sha256(reference_bytes).hexdigest()


def certify_chain_integrity(provenance_chain: Sequence[Mapping[str, Any]]) -> str:
    if not provenance_chain:
        raise ValueError("provenance_chain must not be empty")

    chain_root = _sha256_hex_from_mapping({"namespace": "provenance-chain", "version": _VERSION})
    seen_link_ids: set[str] = set()

    for index, link in enumerate(provenance_chain):
        if not isinstance(link, Mapping):
            raise ValueError(f"provenance_chain[{index}] must be a mapping")
        if "link_id" not in link:
            raise ValueError(f"provenance_chain[{index}] missing link_id")

        link_id = str(link["link_id"]).strip()
        if not link_id:
            raise ValueError(f"provenance_chain[{index}] link_id must be non-empty")
        if link_id in seen_link_ids:
            raise ValueError(f"duplicate link_id in provenance_chain: {link_id}")
        seen_link_ids.add(link_id)

        declared_parent = link.get("parent_hash")
        if declared_parent is not None and str(declared_parent) != chain_root:
            raise ValueError(
                f"provenance chain parent hash mismatch at provenance_chain[{index}] "
                f"(link_id={link_id!r}): expected parent_hash={chain_root!r}, "
                f"received parent_hash={str(declared_parent)!r}"
            )

        payload = _normalize_value(link.get("payload", {}))
        link_payload = {
            "index": int(index),
            "link_id": link_id,
            "parent_hash": chain_root,
            "payload": payload,
        }
        chain_root = _sha256_hex_from_mapping(link_payload)

    return chain_root


def _build_report(
    sovereignty_events: Sequence[Mapping[str, Any]],
    capability_decisions: Sequence[Mapping[str, Any]],
    provenance_chain: Sequence[Mapping[str, Any]],
    policy_evidence: Sequence[Mapping[str, Any]],
) -> ReplayCertificationReport:
    normalized_sovereignty = _normalize_records(sovereignty_events, name="sovereignty_events")
    normalized_capability = _normalize_records(capability_decisions, name="capability_decisions")
    normalized_policy = _normalize_records(policy_evidence, name="policy_evidence")

    sovereignty_event_hash = _record_set_hash(normalized_sovereignty, namespace="sovereignty-events")
    capability_decision_hash = _record_set_hash(normalized_capability, namespace="capability-decisions")
    policy_evidence_hash = _record_set_hash(normalized_policy, namespace="policy-evidence")
    provenance_chain_hash = certify_chain_integrity(provenance_chain)

    root_payload = {
        "version": _VERSION,
        "hashes": {
            "sovereignty_event_hash": sovereignty_event_hash,
            "capability_decision_hash": capability_decision_hash,
            "provenance_chain_hash": provenance_chain_hash,
            "policy_evidence_hash": policy_evidence_hash,
        },
    }
    certification_root_hash = _sha256_hex_from_mapping(root_payload)

    pre_report = {
        "version": _VERSION,
        "sovereignty_event_hash": sovereignty_event_hash,
        "capability_decision_hash": capability_decision_hash,
        "provenance_chain_hash": provenance_chain_hash,
        "policy_evidence_hash": policy_evidence_hash,
        "certification_root_hash": certification_root_hash,
        "sovereignty_event_count": len(normalized_sovereignty),
        "capability_decision_count": len(normalized_capability),
        "provenance_chain_count": len(provenance_chain),
        "policy_evidence_count": len(normalized_policy),
    }
    byte_identity_hash = hashlib.sha256(_canonical_json(pre_report).encode("utf-8")).hexdigest()

    return ReplayCertificationReport(
        sovereignty_event_hash=sovereignty_event_hash,
        capability_decision_hash=capability_decision_hash,
        provenance_chain_hash=provenance_chain_hash,
        policy_evidence_hash=policy_evidence_hash,
        pre_report_hash=byte_identity_hash,
        certification_root_hash=certification_root_hash,
        sovereignty_event_count=len(normalized_sovereignty),
        capability_decision_count=len(normalized_capability),
        provenance_chain_count=len(provenance_chain),
        policy_evidence_count=len(normalized_policy),
    )


def export_certification_bytes(report: ReplayCertificationReport) -> bytes:
    return report.to_canonical_bytes()


def run_replay_certification_battery(
    sovereignty_events: Sequence[Mapping[str, Any]],
    capability_decisions: Sequence[Mapping[str, Any]],
    provenance_chain: Sequence[Mapping[str, Any]],
    policy_evidence: Sequence[Mapping[str, Any]],
) -> ReplayCertificationReport:
    report_a = _build_report(
        sovereignty_events=sovereignty_events,
        capability_decisions=capability_decisions,
        provenance_chain=provenance_chain,
        policy_evidence=policy_evidence,
    )
    report_b = _build_report(
        sovereignty_events=sovereignty_events,
        capability_decisions=capability_decisions,
        provenance_chain=provenance_chain,
        policy_evidence=policy_evidence,
    )
    verify_byte_identity(export_certification_bytes(report_a), export_certification_bytes(report_b))
    if report_a.certification_root_hash != report_b.certification_root_hash:
        raise ValueError("stable certification root mismatch")

    return report_a


def generate_certification_receipt(report: ReplayCertificationReport) -> CertificationReceipt:
    report_hash = _sha256_hex_from_mapping(report.to_dict())
    receipt_hash = _sha256_hex_from_mapping(
        {
            "version": _VERSION,
            "certification_root_hash": report.certification_root_hash,
            "report_hash": report_hash,
        }
    )
    return CertificationReceipt(
        certification_root_hash=report.certification_root_hash,
        report_hash=report_hash,
        receipt_hash=receipt_hash,
    )
