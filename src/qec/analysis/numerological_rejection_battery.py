from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION = "v137.10.6"

_ALLOWED_REJECTION_TYPES = frozenset(
    {
        "unsupported_symbolic_repetition",
        "unsupported_numeric_motif",
        "ratio_without_measurement_grounding",
        "symmetry_without_evidence",
        "constant_reification",
        "phenomenology_without_artifact",
        "evidence_gap_amplification",
    }
)
_ALLOWED_SEVERITIES = frozenset({"info", "warning", "error"})
_ALLOWED_VERDICTS = frozenset({"accepted", "flagged", "rejected"})

_NUMERIC_TOKEN_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_RATIO_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?/[+-]?\d+(?:\.\d+)?$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

_SYMMETRY_MARKERS = frozenset({"symmetry", "symmetric", "mirror", "balanced", "palindrome", "triality"})
_CONSTANT_MARKERS = frozenset({"constant", "universal", "fundamental", "sacred"})
_PHENOMENOLOGY_MARKERS = frozenset({"phenomenology", "phenomenological", "narrative", "intuition", "experience"})


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
    return tuple(sorted(normalized))


def _normalize_numeric_constant(value: Any) -> str:
    if callable(value):
        raise ValueError("malformed numeric constants")
    if isinstance(value, bool):
        raise ValueError("malformed numeric constants")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("malformed numeric constants")
        return format(value, ".17g")
    if isinstance(value, str):
        token = value.strip()
        if not token:
            raise ValueError("malformed numeric constants")
        if _NUMERIC_TOKEN_RE.match(token) or _RATIO_TOKEN_RE.match(token):
            return token
    raise ValueError("malformed numeric constants")


def _normalize_numeric_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("numeric_constants must be a sequence")
    normalized = tuple(_normalize_numeric_constant(item) for item in list(values))
    return tuple(sorted(normalized))


def _normalize_provenance(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("provenance must be a mapping")
    items: list[tuple[str, str]] = []
    seen_keys: set[str] = set()
    for key, val in value.items():
        if callable(key) or callable(val):
            raise ValueError("callable leakage")
        normalized_key = _normalize_token(key, name="provenance key")
        normalized_value = _normalize_token(val, name="provenance value")
        if normalized_key in seen_keys:
            raise ValueError(f"duplicate provenance key after normalization: {normalized_key!r}")
        seen_keys.add(normalized_key)
        items.append((normalized_key, normalized_value))
    items.sort(key=lambda x: x[0])
    return tuple(items)


@dataclass(frozen=True)
class RejectionBatteryInput:
    artifact_id: str
    claim_id: str
    audit_hash: str
    proof_report_hash: str
    symbolic_tokens: tuple[str, ...]
    numeric_constants: tuple[str, ...]
    cited_measurement_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    cited_criterion_ids: tuple[str, ...]
    provenance: tuple[tuple[str, str], ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "claim_id": self.claim_id,
            "audit_hash": self.audit_hash,
            "proof_report_hash": self.proof_report_hash,
            "symbolic_tokens": list(self.symbolic_tokens),
            "numeric_constants": list(self.numeric_constants),
            "cited_measurement_ids": list(self.cited_measurement_ids),
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "cited_criterion_ids": list(self.cited_criterion_ids),
            "provenance": {k: v for k, v in self.provenance},
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class RejectionFinding:
    finding_id: str
    rejection_type: str
    message: str
    related_token: str
    related_numeric_value: str
    severity: str
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "rejection_type": self.rejection_type,
            "message": self.message,
            "related_token": self.related_token,
            "related_numeric_value": self.related_numeric_value,
            "severity": self.severity,
            "blocking": self.blocking,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class RejectionDecision:
    artifact_id: str
    rejection_verdict: str
    finding_ids: tuple[str, ...]
    blocking_findings: int
    rationale_summary: str
    battery_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "rejection_verdict": self.rejection_verdict,
            "finding_ids": list(self.finding_ids),
            "blocking_findings": self.blocking_findings,
            "rationale_summary": self.rationale_summary,
            "battery_hash": self.battery_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class RejectionBatteryReceipt:
    battery_hash: str
    artifact_id: str
    rejection_verdict: str
    finding_count: int
    blocking_findings: int
    byte_length: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "battery_hash": self.battery_hash,
            "artifact_id": self.artifact_id,
            "rejection_verdict": self.rejection_verdict,
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


def normalize_rejection_battery_input(raw_input: Mapping[str, Any]) -> RejectionBatteryInput:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")
    raw_schema = raw_input.get("schema_version", NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION)
    schema_version = _normalize_token(raw_schema, name="schema_version")
    return RejectionBatteryInput(
        artifact_id=_normalize_token(raw_input.get("artifact_id"), name="artifact_id"),
        claim_id=_normalize_token(raw_input.get("claim_id"), name="claim_id"),
        audit_hash=_normalize_token(raw_input.get("audit_hash"), name="audit_hash"),
        proof_report_hash=_normalize_token(raw_input.get("proof_report_hash"), name="proof_report_hash"),
        symbolic_tokens=_normalize_string_tuple(raw_input.get("symbolic_tokens"), name="symbolic_tokens"),
        numeric_constants=_normalize_numeric_tuple(raw_input.get("numeric_constants")),
        cited_measurement_ids=_normalize_string_tuple(raw_input.get("cited_measurement_ids"), name="cited_measurement_ids"),
        cited_evidence_ids=_normalize_string_tuple(raw_input.get("cited_evidence_ids"), name="cited_evidence_ids"),
        cited_criterion_ids=_normalize_string_tuple(raw_input.get("cited_criterion_ids"), name="cited_criterion_ids"),
        provenance=_normalize_provenance(raw_input.get("provenance")),
        schema_version=schema_version,
    )


def validate_rejection_battery_input(battery_input: RejectionBatteryInput) -> None:
    if battery_input.schema_version != NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if not battery_input.artifact_id:
        raise ValueError("artifact_id is required")
    if not battery_input.claim_id:
        raise ValueError("claim_id is required")
    if not _HEX64_RE.match(battery_input.audit_hash):
        raise ValueError("audit_hash must be a 64-character lowercase hex string")
    if not _HEX64_RE.match(battery_input.proof_report_hash):
        raise ValueError("proof_report_hash must be a 64-character lowercase hex string")


def _validate_findings(findings: Sequence[RejectionFinding]) -> None:
    seen: set[str] = set()
    for finding in findings:
        if finding.finding_id in seen:
            raise ValueError("duplicate finding IDs")
        seen.add(finding.finding_id)
        if finding.rejection_type not in _ALLOWED_REJECTION_TYPES:
            raise ValueError("unsupported rejection types")
        if finding.severity not in _ALLOWED_SEVERITIES:
            raise ValueError("unsupported severity")


def _has_repetition(values: tuple[str, ...]) -> bool:
    if not values:
        return False
    return len(set(values)) < len(values)


def _select_repeated_token(tokens: tuple[str, ...]) -> str | None:
    """Return the most-frequent repeated token, tie-breaking lexicographically."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    repeated = [t for t, c in counts.items() if c > 1]
    if not repeated:
        return None
    max_count = max(counts[t] for t in repeated)
    return min(t for t in repeated if counts[t] == max_count)


def _select_repeated_numeric(constants: tuple[str, ...]) -> str | None:
    """Return the most-frequent repeated numeric constant, tie-breaking lexicographically."""
    counts: dict[str, int] = {}
    for c in constants:
        counts[c] = counts.get(c, 0) + 1
    repeated = [c for c, n in counts.items() if n > 1]
    if not repeated:
        return None
    max_count = max(counts[c] for c in repeated)
    return min(c for c in repeated if counts[c] == max_count)


def _new_finding(*, rejection_type: str, message: str, related_token: str = "", related_numeric_value: str = "", severity: str = "warning", blocking: bool = False) -> RejectionFinding:
    fid_src = f"{rejection_type}|{message}|{related_token}|{related_numeric_value}|{severity}|{int(blocking)}"
    finding_id = hashlib.sha256(fid_src.encode("utf-8")).hexdigest()[:16]
    return RejectionFinding(
        finding_id=finding_id,
        rejection_type=rejection_type,
        message=message,
        related_token=related_token,
        related_numeric_value=related_numeric_value,
        severity=severity,
        blocking=blocking,
    )


def run_rejection_battery(
    battery_input: RejectionBatteryInput,
    *,
    available_measurements: Sequence[str] | None = None,
    available_evidence: Sequence[str] | None = None,
    available_criteria: Sequence[str] | None = None,
) -> tuple[RejectionDecision, tuple[RejectionFinding, ...]]:
    validate_rejection_battery_input(battery_input)

    available_measurement_set = set(available_measurements or ())
    available_evidence_set = set(available_evidence or ())
    available_criteria_set = set(available_criteria or ())

    cited_measurements = set(battery_input.cited_measurement_ids)
    cited_evidence = set(battery_input.cited_evidence_ids)
    cited_criteria = set(battery_input.cited_criterion_ids)

    # None means availability catalog is unknown (skip intersection check).
    # An explicit empty list means no items are available (full rejection applies).
    has_measurement_grounding = bool(cited_measurements) and (
        available_measurements is None or bool(cited_measurements & available_measurement_set)
    )
    has_evidence_linkage = bool(cited_evidence) and (
        available_evidence is None or bool(cited_evidence & available_evidence_set)
    )
    has_criterion_linkage = bool(cited_criteria) and (
        available_criteria is None or bool(cited_criteria & available_criteria_set)
    )

    findings: list[RejectionFinding] = []

    if _has_repetition(battery_input.symbolic_tokens) and not (has_evidence_linkage or has_criterion_linkage):
        findings.append(
            _new_finding(
                rejection_type="unsupported_symbolic_repetition",
                message="repeated symbolic tokens without linked evidence or criterion references",
                related_token=_select_repeated_token(battery_input.symbolic_tokens) or "",
            )
        )

    if _has_repetition(battery_input.numeric_constants) and not has_measurement_grounding:
        findings.append(
            _new_finding(
                rejection_type="unsupported_numeric_motif",
                message="repeated numeric constants not tied to cited measurements",
                related_numeric_value=_select_repeated_numeric(battery_input.numeric_constants) or "",
            )
        )

    has_ratio_or_constant = bool(battery_input.numeric_constants)
    if has_ratio_or_constant and not has_measurement_grounding:
        findings.append(
            _new_finding(
                rejection_type="ratio_without_measurement_grounding",
                message="ratios/constants referenced without measurement grounding",
                related_numeric_value=battery_input.numeric_constants[0],
                severity="error",
                blocking=True,
            )
        )

    if any(token.lower() in _SYMMETRY_MARKERS for token in battery_input.symbolic_tokens) and not has_evidence_linkage:
        findings.append(
            _new_finding(
                rejection_type="symmetry_without_evidence",
                message="symmetry motifs present without evidence linkage",
                related_token=next(token for token in battery_input.symbolic_tokens if token.lower() in _SYMMETRY_MARKERS),
            )
        )

    if _has_repetition(battery_input.numeric_constants) and any(
        token.lower() in _CONSTANT_MARKERS for token in battery_input.symbolic_tokens
    ):
        findings.append(
            _new_finding(
                rejection_type="constant_reification",
                message="constants repeatedly elevated to explanatory status",
                related_numeric_value=_select_repeated_numeric(battery_input.numeric_constants) or "",
                related_token=next(token for token in battery_input.symbolic_tokens if token.lower() in _CONSTANT_MARKERS),
            )
        )

    if any(token.lower() in _PHENOMENOLOGY_MARKERS for token in battery_input.symbolic_tokens) and not has_evidence_linkage:
        findings.append(
            _new_finding(
                rejection_type="phenomenology_without_artifact",
                message="phenomenological narrative detected without evidence links",
                related_token=next(token for token in battery_input.symbolic_tokens if token.lower() in _PHENOMENOLOGY_MARKERS),
                severity="error",
                blocking=True,
            )
        )

    missing_evidence = bool(cited_evidence) and bool(available_evidence_set) and not bool(cited_evidence & available_evidence_set)
    if missing_evidence and _has_repetition(battery_input.symbolic_tokens):
        findings.append(
            _new_finding(
                rejection_type="evidence_gap_amplification",
                message="missing evidence cited while symbolic inflation persists",
                related_token=_select_repeated_token(battery_input.symbolic_tokens) or "",
                severity="error",
                blocking=True,
            )
        )

    ordered_findings = tuple(sorted(findings, key=lambda f: (f.rejection_type, f.finding_id)))
    _validate_findings(ordered_findings)

    blocking_count = sum(1 for finding in ordered_findings if finding.blocking and finding.severity == "error")
    if not ordered_findings:
        verdict = "accepted"
        rationale = "accepted: 0 rejection findings"
    elif blocking_count > 0:
        verdict = "rejected"
        first_blocking = next(f for f in ordered_findings if f.blocking and f.severity == "error")
        rationale = f"rejected: {blocking_count} blocking {first_blocking.rejection_type.replace('_', ' ')}"
    else:
        verdict = "flagged"
        first = ordered_findings[0]
        rationale = f"flagged: {len(ordered_findings)} {first.rejection_type.replace('_', ' ')}"

    decision = RejectionDecision(
        artifact_id=battery_input.artifact_id,
        rejection_verdict=verdict,
        finding_ids=tuple(sorted(f.finding_id for f in ordered_findings)),
        blocking_findings=blocking_count,
        rationale_summary=rationale,
        battery_hash="",
        schema_version=battery_input.schema_version,
    )
    battery_hash = stable_rejection_battery_hash(decision, ordered_findings)
    decision = RejectionDecision(
        artifact_id=decision.artifact_id,
        rejection_verdict=decision.rejection_verdict,
        finding_ids=decision.finding_ids,
        blocking_findings=decision.blocking_findings,
        rationale_summary=decision.rationale_summary,
        battery_hash=battery_hash,
        schema_version=decision.schema_version,
    )

    if decision.rejection_verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    return decision, ordered_findings


def stable_rejection_battery_hash(decision: RejectionDecision, findings: Sequence[RejectionFinding]) -> str:
    if decision.rejection_verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    _validate_findings(findings)
    expected_ids = sorted(f.finding_id for f in findings)
    actual_ids = sorted(decision.finding_ids)
    if expected_ids != actual_ids:
        raise ValueError("decision finding_ids do not match provided findings")
    payload = {
        "decision": {
            "artifact_id": decision.artifact_id,
            "rejection_verdict": decision.rejection_verdict,
            "finding_ids": list(decision.finding_ids),
            "blocking_findings": decision.blocking_findings,
            "rationale_summary": decision.rationale_summary,
            "schema_version": decision.schema_version,
        },
        "findings": [
            finding.to_dict()
            for finding in sorted(findings, key=lambda f: (f.rejection_type, f.finding_id))
        ],
    }
    canonical = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_rejection_battery_receipt(
    decision: RejectionDecision,
    findings: Sequence[RejectionFinding],
) -> RejectionBatteryReceipt:
    if decision.rejection_verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    _validate_findings(findings)
    if decision.schema_version != NUMEROLOGICAL_REJECTION_BATTERY_SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    # Recompute and verify the battery hash for integrity; do not trust decision.battery_hash blindly.
    computed_hash = stable_rejection_battery_hash(decision, findings)
    if decision.battery_hash and decision.battery_hash != computed_hash:
        raise ValueError("battery_hash mismatch: decision hash does not match computed hash")
    battery_hash = computed_hash
    ordered_findings = tuple(sorted(findings, key=lambda f: (f.rejection_type, f.finding_id)))
    serialized = _canonical_json(
        {
            "decision": decision.to_dict(),
            "findings": [finding.to_dict() for finding in ordered_findings],
        }
    ).encode("utf-8")
    return RejectionBatteryReceipt(
        battery_hash=battery_hash,
        artifact_id=decision.artifact_id,
        rejection_verdict=decision.rejection_verdict,
        finding_count=len(ordered_findings),
        blocking_findings=sum(1 for finding in ordered_findings if finding.blocking and finding.severity == "error"),
        byte_length=len(serialized),
        validation_passed=True,
        schema_version=decision.schema_version,
    )


def compile_rejection_battery(
    raw_input: Mapping[str, Any],
    *,
    available_measurements: Sequence[str] | None = None,
    available_evidence: Sequence[str] | None = None,
    available_criteria: Sequence[str] | None = None,
) -> tuple[RejectionDecision, tuple[RejectionFinding, ...], RejectionBatteryReceipt]:
    battery_input = normalize_rejection_battery_input(raw_input)
    validate_rejection_battery_input(battery_input)
    decision, findings = run_rejection_battery(
        battery_input,
        available_measurements=available_measurements,
        available_evidence=available_evidence,
        available_criteria=available_criteria,
    )
    receipt = build_rejection_battery_receipt(decision, findings)
    return decision, findings, receipt
