from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION = "v137.10.7"

_ALLOWED_FINDING_TYPES = frozenset(
    {
        "audit_pass",
        "replay_pass",
        "proof_pass",
        "rejection_pass",
        "missing_prerequisite",
        "blocking_failure",
        "certification_constraint",
    }
)
_ALLOWED_SEVERITIES = frozenset({"info", "warning", "error"})
_ALLOWED_VERDICTS = frozenset({"certified", "conditionally_certified", "rejected"})
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _normalize_token(value: Any, *, name: str) -> str:
    if value is None or callable(value):
        raise ValueError(f"{name} must be non-empty")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _normalize_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    normalized = tuple(_normalize_token(item, name=name) for item in list(values))
    return tuple(item for item, _ in sorted(((item, idx) for idx, item in enumerate(normalized)), key=lambda x: (x[0], x[1])))


def _normalize_provenance(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping")
    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key, raw in value.items():
        k = _normalize_token(key, name="provenance key")
        v = _normalize_token(raw, name="provenance value")
        if k in seen:
            raise ValueError(f"duplicate provenance key after normalization: {k}")
        seen.add(k)
        items.append((k, v))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def _normalize_prior_verdicts(value: Any) -> tuple[tuple[str, str, str], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("prior_verdicts must be a sequence")
    entries: list[tuple[str, str, str]] = []
    for idx, item in enumerate(list(value)):
        if not isinstance(item, Mapping):
            raise ValueError(f"prior_verdicts[{idx}] must be a mapping")
        artifact_id = _normalize_token(item.get("artifact_id"), name="prior_verdict artifact_id")
        verdict = _normalize_token(item.get("certification_verdict"), name="prior_verdict certification_verdict")
        cert_hash = _normalize_token(item.get("certification_hash"), name="prior_verdict certification_hash")
        entries.append((artifact_id, verdict, cert_hash))
    entries.sort(key=lambda x: (x[0], x[1], x[2]))
    return tuple(entries)


@dataclass(frozen=True)
class CertificationInput:
    artifact_id: str
    claim_id: str
    audit_hash: str
    replay_hash: str
    proof_report_hash: str
    rejection_battery_hash: str
    prior_verdicts: tuple[tuple[str, str, str], ...]
    blocking_findings: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "claim_id": self.claim_id,
            "audit_hash": self.audit_hash,
            "replay_hash": self.replay_hash,
            "proof_report_hash": self.proof_report_hash,
            "rejection_battery_hash": self.rejection_battery_hash,
            "prior_verdicts": [
                {
                    "artifact_id": artifact_id,
                    "certification_verdict": verdict,
                    "certification_hash": certification_hash,
                }
                for artifact_id, verdict, certification_hash in self.prior_verdicts
            ],
            "blocking_findings": list(self.blocking_findings),
            "provenance": {k: v for k, v in self.provenance},
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CertificationFinding:
    finding_id: str
    finding_type: str
    message: str
    blocking: bool
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "message": self.message,
            "blocking": self.blocking,
            "severity": self.severity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CertificationDecision:
    artifact_id: str
    certification_verdict: str
    finding_ids: tuple[str, ...]
    blocking_findings: int
    rationale_summary: str
    certification_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "certification_verdict": self.certification_verdict,
            "finding_ids": list(self.finding_ids),
            "blocking_findings": self.blocking_findings,
            "rationale_summary": self.rationale_summary,
            "certification_hash": self.certification_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class CertificationReceipt:
    certification_hash: str
    artifact_id: str
    certification_verdict: str
    finding_count: int
    blocking_findings: int
    byte_length: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "certification_hash": self.certification_hash,
            "artifact_id": self.artifact_id,
            "certification_verdict": self.certification_verdict,
            "finding_count": self.finding_count,
            "blocking_findings": self.blocking_findings,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_certification_input(raw_input: Mapping[str, Any]) -> CertificationInput:
    if isinstance(raw_input, CertificationInput):
        validate_certification_input(raw_input)
        return CertificationInput(**raw_input.__dict__)
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")

    schema_version = _normalize_token(
        raw_input.get("schema_version", SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION),
        name="schema_version",
    )

    cert_input = CertificationInput(
        artifact_id=_normalize_token(raw_input.get("artifact_id"), name="artifact_id"),
        claim_id=_normalize_token(raw_input.get("claim_id"), name="claim_id"),
        audit_hash=_normalize_token(raw_input.get("audit_hash"), name="audit_hash"),
        replay_hash=_normalize_token(raw_input.get("replay_hash"), name="replay_hash"),
        proof_report_hash=_normalize_token(raw_input.get("proof_report_hash"), name="proof_report_hash"),
        rejection_battery_hash=_normalize_token(raw_input.get("rejection_battery_hash"), name="rejection_battery_hash"),
        prior_verdicts=_normalize_prior_verdicts(raw_input.get("prior_verdicts")),
        blocking_findings=_normalize_string_tuple(raw_input.get("blocking_findings"), name="blocking_findings"),
        provenance=_normalize_provenance(raw_input.get("provenance")),
        schema_version=schema_version,
    )
    validate_certification_input(cert_input)
    return cert_input


def validate_certification_input(cert_input: CertificationInput) -> None:
    if cert_input.schema_version != SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if not cert_input.artifact_id:
        raise ValueError("artifact_id must be non-empty")
    if not cert_input.claim_id:
        raise ValueError("claim_id must be non-empty")

    for name, digest in (
        ("audit_hash", cert_input.audit_hash),
        ("replay_hash", cert_input.replay_hash),
        ("proof_report_hash", cert_input.proof_report_hash),
        ("rejection_battery_hash", cert_input.rejection_battery_hash),
    ):
        if not _HEX64_RE.match(digest):
            raise ValueError(f"{name} must be a 64-character lowercase hex string")

    for artifact_id, verdict, cert_hash in cert_input.prior_verdicts:
        if verdict not in _ALLOWED_VERDICTS:
            raise ValueError("unsupported verdict")
        if not artifact_id:
            raise ValueError("prior_verdict artifact_id must be non-empty")
        if not _HEX64_RE.match(cert_hash):
            raise ValueError("prior_verdict certification_hash must be a 64-character lowercase hex string")


def _make_finding(*, finding_type: str, message: str, blocking: bool, severity: str, index: int) -> CertificationFinding:
    payload = f"{finding_type}|{message}|{int(blocking)}|{severity}|{index}".encode("utf-8")
    return CertificationFinding(
        finding_id=hashlib.sha256(payload).hexdigest(),
        finding_type=finding_type,
        message=message,
        blocking=blocking,
        severity=severity,
    )


def run_certification_kernel(cert_input: CertificationInput) -> tuple[CertificationDecision, tuple[CertificationFinding, ...]]:
    validate_certification_input(cert_input)

    findings: list[CertificationFinding] = []
    findings.append(_make_finding(finding_type="audit_pass", message="audit hash prerequisite satisfied", blocking=False, severity="info", index=0))
    findings.append(_make_finding(finding_type="replay_pass", message="replay hash prerequisite satisfied", blocking=False, severity="info", index=1))
    findings.append(_make_finding(finding_type="proof_pass", message="proof report prerequisite satisfied", blocking=False, severity="info", index=2))
    findings.append(_make_finding(finding_type="rejection_pass", message="rejection battery prerequisite satisfied", blocking=False, severity="info", index=3))

    if cert_input.blocking_findings:
        findings.append(
            _make_finding(
                finding_type="blocking_failure",
                message=f"blocking findings present: {len(cert_input.blocking_findings)}",
                blocking=True,
                severity="error",
                index=4,
            )
        )

    if not cert_input.audit_hash or not cert_input.replay_hash or not cert_input.proof_report_hash or not cert_input.rejection_battery_hash:
        findings.append(
            _make_finding(
                finding_type="missing_prerequisite",
                message="one or more prerequisite hashes are missing",
                blocking=True,
                severity="error",
                index=5,
            )
        )

    prior_by_artifact: dict[str, tuple[str, str]] = {}
    inconsistent = False
    for artifact_id, verdict, cert_hash in cert_input.prior_verdicts:
        prev = prior_by_artifact.get(artifact_id)
        if prev is None:
            prior_by_artifact[artifact_id] = (verdict, cert_hash)
        elif prev != (verdict, cert_hash):
            inconsistent = True
            break
    if inconsistent:
        findings.append(
            _make_finding(
                finding_type="certification_constraint",
                message="prior verdict chain is inconsistent",
                blocking=True,
                severity="error",
                index=6,
            )
        )

    decorated = [
        (item.finding_id, idx, item)
        for idx, item in enumerate(findings)
    ]
    decorated.sort(key=lambda x: (x[0], x[1]))
    ordered_findings = tuple(item for _, _, item in decorated)

    blocking_count = sum(1 for item in ordered_findings if item.blocking)
    warning_count = sum(1 for item in ordered_findings if item.severity == "warning")

    if blocking_count > 0:
        verdict = "rejected"
        blocker_types = sorted({item.finding_type.replace("_failure", "") for item in ordered_findings if item.blocking})
        rationale = f"rejected: {' + '.join(blocker_types)} blockers present"
    elif warning_count > 0:
        verdict = "conditionally_certified"
        rationale = f"conditionally_certified: {warning_count} non-blocking findings"
    else:
        verdict = "certified"
        rationale = "certified: all prerequisite kernels passed"

    decision = CertificationDecision(
        artifact_id=cert_input.artifact_id,
        certification_verdict=verdict,
        finding_ids=tuple(item.finding_id for item in ordered_findings),
        blocking_findings=blocking_count,
        rationale_summary=rationale,
        certification_hash="",
        schema_version=cert_input.schema_version,
    )
    certification_hash = stable_certification_hash(decision, ordered_findings)
    decision = CertificationDecision(
        artifact_id=decision.artifact_id,
        certification_verdict=decision.certification_verdict,
        finding_ids=decision.finding_ids,
        blocking_findings=decision.blocking_findings,
        rationale_summary=decision.rationale_summary,
        certification_hash=certification_hash,
        schema_version=decision.schema_version,
    )
    return decision, ordered_findings


def stable_certification_hash(decision: CertificationDecision, findings: Sequence[CertificationFinding]) -> str:
    if decision.certification_verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    if decision.schema_version != SCIENTIFIC_CERTIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")

    seen: set[str] = set()
    ordered_findings: list[CertificationFinding] = []
    for item in findings:
        if item.finding_id in seen:
            raise ValueError("duplicate finding IDs")
        seen.add(item.finding_id)
        if item.finding_type not in _ALLOWED_FINDING_TYPES:
            raise ValueError("unsupported finding types")
        if item.severity not in _ALLOWED_SEVERITIES:
            raise ValueError("unsupported severity")
        ordered_findings.append(item)
    ordered_findings.sort(key=lambda x: x.finding_id)

    if tuple(item.finding_id for item in ordered_findings) != tuple(sorted(decision.finding_ids)):
        raise ValueError("decision finding_ids must match findings")

    payload = {
        "artifact_id": decision.artifact_id,
        "certification_verdict": decision.certification_verdict,
        "finding_ids": sorted(decision.finding_ids),
        "blocking_findings": int(decision.blocking_findings),
        "rationale_summary": decision.rationale_summary,
        "findings": [item.to_dict() for item in ordered_findings],
        "schema_version": decision.schema_version,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_certification_receipt(decision: CertificationDecision, findings: Sequence[CertificationFinding]) -> CertificationReceipt:
    expected_hash = stable_certification_hash(decision, findings)
    if decision.certification_hash != expected_hash:
        raise ValueError("certification hash mismatch")
    validation_passed = decision.certification_verdict in _ALLOWED_VERDICTS
    decision_bytes = decision.to_canonical_bytes()
    return CertificationReceipt(
        certification_hash=decision.certification_hash,
        artifact_id=decision.artifact_id,
        certification_verdict=decision.certification_verdict,
        finding_count=len(tuple(findings)),
        blocking_findings=decision.blocking_findings,
        byte_length=len(decision_bytes),
        validation_passed=validation_passed,
        schema_version=decision.schema_version,
    )


def compile_certification_decision(raw_input: CertificationInput | Mapping[str, Any]) -> tuple[CertificationDecision, tuple[CertificationFinding, ...], CertificationReceipt]:
    cert_input = normalize_certification_input(raw_input)
    decision, findings = run_certification_kernel(cert_input)
    receipt = build_certification_receipt(decision, findings)
    return decision, findings, receipt
