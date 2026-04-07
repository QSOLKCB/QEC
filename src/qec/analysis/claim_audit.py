from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

CLAIM_AUDIT_SCHEMA_VERSION = "v137.10.3"
_ALLOWED_VERDICTS = frozenset({"supported", "contradicted", "inconclusive"})
_ALLOWED_FINDING_TYPES = frozenset(
    {
        "criterion_satisfied",
        "criterion_failed",
        "evidence_missing",
        "evidence_conflict",
        "measurement_missing",
        "lineage_gap",
    }
)
_ALLOWED_SEVERITIES = frozenset({"info", "warning", "error"})

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | tuple["JsonValue", ...] | tuple[tuple[str, "JsonValue"], ...]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _normalize_token(value: Any, *, name: str) -> str:
    if value is None or callable(value):
        raise ValueError(f"{name} is required")
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be non-empty")
    return token


def _normalize_string_tuple(values: Any, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized = tuple(_normalize_token(item, name=name) for item in list(values))
    ordered = tuple(item for item, _ in sorted(((v, idx) for idx, v in enumerate(normalized)), key=lambda x: (x[0], x[1])))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{name} contains duplicates")
    return ordered


def _normalize_json_value(value: Any, *, path: str) -> JsonValue:
    if callable(value):
        raise ValueError(f"{path} must not be callable")
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        items: list[tuple[str, JsonValue]] = []
        for key, sub_value in value.items():
            norm_key = _normalize_token(key, name=f"{path} key")
            items.append((norm_key, _normalize_json_value(sub_value, path=f"{path}.{norm_key}")))
        items.sort(key=lambda item: item[0])
        deduped: list[tuple[str, JsonValue]] = []
        seen: set[str] = set()
        for key, sub_value in items:
            if key in seen:
                raise ValueError(f"duplicate key after normalization at {path}: {key}")
            seen.add(key)
            deduped.append((key, sub_value))
        return tuple(deduped)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_normalize_json_value(item, path=path) for item in list(value))
    raise ValueError(f"{path} contains unsupported type: {type(value).__name__}")


def _json_value_to_python(value: JsonValue) -> Any:
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple) and len(value[0]) == 2 and isinstance(value[0][0], str):
            return {k: _json_value_to_python(v) for k, v in value}  # type: ignore[misc]
        return [_json_value_to_python(item) for item in value]
    return value


def _normalize_expected_relations(value: Any) -> tuple[tuple[str, tuple[tuple[str, JsonValue], ...]], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("expected_relations must be a mapping")
    items: list[tuple[str, tuple[tuple[str, JsonValue], ...]]] = []
    for criterion_id, relation_payload in value.items():
        key = _normalize_token(criterion_id, name="expected_relations criterion_id")
        if not isinstance(relation_payload, Mapping):
            raise ValueError("expected_relations entries must be mappings")
        relation_copy = dict(relation_payload)
        if "node_ids" in relation_copy:
            relation_copy["node_ids"] = list(_normalize_string_tuple(relation_copy["node_ids"], name=f"expected_relations.{key}.node_ids"))
        normalized_relation = _normalize_json_value(relation_copy, path=f"expected_relations.{key}")
        assert isinstance(normalized_relation, tuple)
        items.append((key, normalized_relation))
    items.sort(key=lambda item: item[0])
    if len({k for k, _ in items}) != len(items):
        raise ValueError("expected_relations contains duplicate criterion IDs")
    return tuple(items)


@dataclass(frozen=True)
class ClaimAuditInput:
    claim_id: str
    claim_text: str
    experiment_hash: str
    evidence_graph_hash: str
    measurement_ids: tuple[str, ...]
    criterion_ids: tuple[str, ...]
    expected_relations: tuple[tuple[str, tuple[tuple[str, JsonValue], ...]], ...]
    provenance: tuple[tuple[str, JsonValue], ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "experiment_hash": self.experiment_hash,
            "evidence_graph_hash": self.evidence_graph_hash,
            "measurement_ids": list(self.measurement_ids),
            "criterion_ids": list(self.criterion_ids),
            "expected_relations": {key: _json_value_to_python(value) for key, value in self.expected_relations},
            "provenance": _json_value_to_python(self.provenance),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ClaimAuditFinding:
    finding_id: str
    finding_type: str
    related_measurement_id: str
    related_criterion_id: str
    related_node_ids: tuple[str, ...]
    message: str
    severity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "related_measurement_id": self.related_measurement_id,
            "related_criterion_id": self.related_criterion_id,
            "related_node_ids": list(self.related_node_ids),
            "message": self.message,
            "severity": self.severity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ClaimAuditDecision:
    claim_id: str
    verdict: str
    supporting_finding_ids: tuple[str, ...]
    contradicting_finding_ids: tuple[str, ...]
    inconclusive_finding_ids: tuple[str, ...]
    rationale_summary: str
    experiment_hash: str
    evidence_graph_hash: str
    audit_hash: str
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "supporting_finding_ids": list(self.supporting_finding_ids),
            "contradicting_finding_ids": list(self.contradicting_finding_ids),
            "inconclusive_finding_ids": list(self.inconclusive_finding_ids),
            "rationale_summary": self.rationale_summary,
            "experiment_hash": self.experiment_hash,
            "evidence_graph_hash": self.evidence_graph_hash,
            "audit_hash": self.audit_hash,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ClaimAuditReceipt:
    audit_hash: str
    claim_id: str
    verdict: str
    finding_count: int
    supporting_count: int
    contradicting_count: int
    inconclusive_count: int
    byte_length: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_hash": self.audit_hash,
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "finding_count": self.finding_count,
            "supporting_count": self.supporting_count,
            "contradicting_count": self.contradicting_count,
            "inconclusive_count": self.inconclusive_count,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_claim_audit_input(raw_input: Mapping[str, Any]) -> ClaimAuditInput:
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be a mapping")
    schema_version = _normalize_token(raw_input.get("schema_version") or CLAIM_AUDIT_SCHEMA_VERSION, name="schema_version")
    claim_input = ClaimAuditInput(
        claim_id=_normalize_token(raw_input.get("claim_id"), name="claim_id"),
        claim_text=_normalize_token(raw_input.get("claim_text"), name="claim_text"),
        experiment_hash=_normalize_token(raw_input.get("experiment_hash"), name="experiment_hash"),
        evidence_graph_hash=_normalize_token(raw_input.get("evidence_graph_hash"), name="evidence_graph_hash"),
        measurement_ids=_normalize_string_tuple(raw_input.get("measurement_ids"), name="measurement_ids"),
        criterion_ids=_normalize_string_tuple(raw_input.get("criterion_ids"), name="criterion_ids"),
        expected_relations=_normalize_expected_relations(raw_input.get("expected_relations")),
        provenance=_normalize_json_value(raw_input.get("provenance", {}), path="provenance"),  # type: ignore[arg-type]
        schema_version=schema_version,
    )
    validate_claim_audit_input(claim_input)
    return claim_input


def validate_claim_audit_input(audit_input: ClaimAuditInput) -> None:
    if not isinstance(audit_input, ClaimAuditInput):
        raise ValueError("audit_input must be ClaimAuditInput")
    if audit_input.schema_version != CLAIM_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if not audit_input.claim_id:
        raise ValueError("claim_id must be non-empty")
    if not audit_input.claim_text:
        raise ValueError("claim_text must be non-empty")
    if not audit_input.experiment_hash:
        raise ValueError("experiment_hash must be non-empty")
    if not audit_input.evidence_graph_hash:
        raise ValueError("evidence_graph_hash must be non-empty")


def _normalize_available_statuses(payload: Any, *, name: str) -> dict[str, str]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    out: dict[str, str] = {}
    for key, value in payload.items():
        item_id = _normalize_token(key, name=f"{name} key")
        status: str
        if isinstance(value, Mapping):
            status = _normalize_token(value.get("status"), name=f"{name}[{item_id}].status")
        else:
            status = _normalize_token(value, name=f"{name}[{item_id}]")
        out[item_id] = status
    return out


def _extract_graph_nodes(evidence_graph: Mapping[str, Any] | None) -> set[str]:
    if evidence_graph is None:
        return set()
    nodes_value = evidence_graph.get("nodes", ())
    if isinstance(nodes_value, Mapping):
        return { _normalize_token(k, name="evidence_graph.nodes key") for k in nodes_value.keys() }
    if isinstance(nodes_value, Sequence) and not isinstance(nodes_value, (str, bytes)):
        return { _normalize_token(v, name="evidence_graph.nodes item") for v in list(nodes_value) }
    raise ValueError("evidence_graph.nodes must be a mapping or sequence")


def _extract_conflicts(evidence_graph: Mapping[str, Any] | None) -> set[tuple[str, str]]:
    if evidence_graph is None:
        return set()
    conflicts = evidence_graph.get("conflicts", ())
    if not isinstance(conflicts, Sequence) or isinstance(conflicts, (str, bytes)):
        raise ValueError("evidence_graph.conflicts must be a sequence")
    pairs: set[tuple[str, str]] = set()
    for item in list(conflicts):
        left: str
        right: str
        if isinstance(item, Mapping):
            left = _normalize_token(item.get("left"), name="conflicts.left")
            right = _normalize_token(item.get("right"), name="conflicts.right")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            left = _normalize_token(item[0], name="conflicts[0]")
            right = _normalize_token(item[1], name="conflicts[1]")
        else:
            raise ValueError("conflict entries must be 2-item sequences or mappings")
        pair = tuple(sorted((left, right)))
        pairs.add((pair[0], pair[1]))
    return pairs


def _relation_mapping(audit_input: ClaimAuditInput) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for criterion_id, relation_tuple in audit_input.expected_relations:
        relation = _json_value_to_python(relation_tuple)
        assert isinstance(relation, dict)
        out[criterion_id] = relation
    return out


def _sorted_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(v for v, _ in sorted(((item, idx) for idx, item in enumerate(values)), key=lambda x: (x[0], x[1])))


def _validate_finding(finding: ClaimAuditFinding) -> None:
    if finding.finding_type not in _ALLOWED_FINDING_TYPES:
        raise ValueError("unsupported finding type")
    if finding.severity not in _ALLOWED_SEVERITIES:
        raise ValueError("unsupported severity")
    if not isinstance(finding.related_node_ids, tuple):
        raise ValueError("related_node_ids must be tuple")


def _build_rationale_summary(verdict: str, findings: Sequence[ClaimAuditFinding]) -> str:
    failed = sum(1 for f in findings if f.finding_type == "criterion_failed")
    satisfied = sum(1 for f in findings if f.finding_type == "criterion_satisfied")
    gaps = sum(1 for f in findings if f.finding_type == "lineage_gap")
    missing_measurements = sum(1 for f in findings if f.finding_type == "measurement_missing")
    evidence_gaps = sum(1 for f in findings if f.finding_type in {"evidence_missing", "evidence_conflict"})
    if verdict == "supported":
        return f"supported: {satisfied} satisfied criteria, {failed} contradictions, {gaps} lineage gaps"
    if verdict == "contradicted":
        return f"contradicted: {failed} failed criteria, {missing_measurements} missing measurements"
    return f"inconclusive: {evidence_gaps} evidence gaps, {missing_measurements} missing measurements"


def run_claim_audit(
    audit_input: ClaimAuditInput,
    *,
    available_measurements: Mapping[str, Any] | None = None,
    available_criteria: Mapping[str, Any] | None = None,
    evidence_graph: Mapping[str, Any] | None = None,
) -> tuple[ClaimAuditDecision, tuple[ClaimAuditFinding, ...]]:
    validate_claim_audit_input(audit_input)

    measurement_status = _normalize_available_statuses(available_measurements, name="available_measurements")
    criterion_status = _normalize_available_statuses(available_criteria, name="available_criteria")
    graph_nodes = _extract_graph_nodes(evidence_graph)
    graph_conflicts = _extract_conflicts(evidence_graph)
    relation_map = _relation_mapping(audit_input)

    findings: list[ClaimAuditFinding] = []

    def emit(
        finding_type: str,
        *,
        related_measurement_id: str = "",
        related_criterion_id: str = "",
        related_node_ids: Sequence[str] = (),
        message: str,
        severity: str,
    ) -> None:
        finding = ClaimAuditFinding(
            finding_id=f"f-{len(findings) + 1:04d}",
            finding_type=finding_type,
            related_measurement_id=related_measurement_id,
            related_criterion_id=related_criterion_id,
            related_node_ids=_sorted_tuple(tuple(related_node_ids)),
            message=message,
            severity=severity,
        )
        _validate_finding(finding)
        findings.append(finding)

    for measurement_id in audit_input.measurement_ids:
        if measurement_id not in measurement_status:
            emit(
                "measurement_missing",
                related_measurement_id=measurement_id,
                message=f"measurement_missing:{measurement_id}",
                severity="warning",
            )

    for criterion_id in audit_input.criterion_ids:
        status = criterion_status.get(criterion_id, "")
        if status == "failed":
            emit(
                "criterion_failed",
                related_criterion_id=criterion_id,
                message=f"criterion_failed:{criterion_id}",
                severity="error",
            )
        elif status == "satisfied":
            emit(
                "criterion_satisfied",
                related_criterion_id=criterion_id,
                message=f"criterion_satisfied:{criterion_id}",
                severity="info",
            )
        else:
            emit(
                "evidence_missing",
                related_criterion_id=criterion_id,
                message=f"criterion_status_missing:{criterion_id}",
                severity="warning",
            )

        relation = relation_map.get(criterion_id, {})
        relation_measurement = relation.get("measurement_id")
        if relation_measurement is not None:
            expected_mid = _normalize_token(relation_measurement, name="expected measurement_id")
            if expected_mid not in measurement_status:
                emit(
                    "measurement_missing",
                    related_measurement_id=expected_mid,
                    related_criterion_id=criterion_id,
                    message=f"relation_measurement_missing:{criterion_id}:{expected_mid}",
                    severity="warning",
                )

        node_ids = relation.get("node_ids", ())
        if isinstance(node_ids, Sequence) and not isinstance(node_ids, (str, bytes)):
            required_nodes = _sorted_tuple(tuple(_normalize_token(item, name="node_id") for item in list(node_ids)))
        elif node_ids in (None, ()):
            required_nodes = ()
        else:
            raise ValueError("expected_relations node_ids must be a sequence")

        if required_nodes:
            if evidence_graph is None:
                emit(
                    "evidence_missing",
                    related_criterion_id=criterion_id,
                    related_node_ids=required_nodes,
                    message=f"evidence_graph_missing:{criterion_id}",
                    severity="warning",
                )
            else:
                missing_nodes = tuple(node for node in required_nodes if node not in graph_nodes)
                if missing_nodes:
                    emit(
                        "lineage_gap",
                        related_criterion_id=criterion_id,
                        related_node_ids=missing_nodes,
                        message=f"lineage_gap:{criterion_id}",
                        severity="error",
                    )
                for left_index in range(len(required_nodes)):
                    for right_index in range(left_index + 1, len(required_nodes)):
                        pair = tuple(sorted((required_nodes[left_index], required_nodes[right_index])))
                        if (pair[0], pair[1]) in graph_conflicts:
                            emit(
                                "evidence_conflict",
                                related_criterion_id=criterion_id,
                                related_node_ids=(pair[0], pair[1]),
                                message=f"evidence_conflict:{criterion_id}:{pair[0]}:{pair[1]}",
                                severity="warning",
                            )

    findings_sorted = tuple(sorted(findings, key=lambda f: f.finding_id))
    finding_ids = {f.finding_id for f in findings_sorted}
    if len(finding_ids) != len(findings_sorted):
        raise ValueError("duplicate finding IDs")

    supporting = _sorted_tuple([f.finding_id for f in findings_sorted if f.finding_type == "criterion_satisfied"])
    contradicting = _sorted_tuple([f.finding_id for f in findings_sorted if f.finding_type == "criterion_failed"])
    inconclusive = _sorted_tuple([f.finding_id for f in findings_sorted if f.finding_id not in set(supporting + contradicting)])

    if contradicting:
        verdict = "contradicted"
    elif inconclusive or not supporting:
        verdict = "inconclusive"
    else:
        verdict = "supported"

    rationale = _build_rationale_summary(verdict, findings_sorted)

    provisional = ClaimAuditDecision(
        claim_id=audit_input.claim_id,
        verdict=verdict,
        supporting_finding_ids=supporting,
        contradicting_finding_ids=contradicting,
        inconclusive_finding_ids=inconclusive,
        rationale_summary=rationale,
        experiment_hash=audit_input.experiment_hash,
        evidence_graph_hash=audit_input.evidence_graph_hash,
        audit_hash="",
        schema_version=audit_input.schema_version,
    )
    audit_hash = stable_claim_audit_hash(provisional, findings_sorted)
    decision = ClaimAuditDecision(**{**provisional.to_dict(), "audit_hash": audit_hash})
    return decision, findings_sorted


def stable_claim_audit_hash(decision: ClaimAuditDecision, findings: Sequence[ClaimAuditFinding]) -> str:
    if decision.verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    seen_ids: set[str] = set()
    finding_payload = []
    for finding in sorted(findings, key=lambda item: item.finding_id):
        _validate_finding(finding)
        if finding.finding_id in seen_ids:
            raise ValueError("duplicate finding IDs")
        seen_ids.add(finding.finding_id)
        finding_payload.append(finding.to_dict())
    payload = {
        "decision": {**decision.to_dict(), "audit_hash": ""},
        "findings": finding_payload,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_claim_audit_receipt(decision: ClaimAuditDecision, findings: Sequence[ClaimAuditFinding]) -> ClaimAuditReceipt:
    if decision.verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    findings_sorted = tuple(sorted(findings, key=lambda item: item.finding_id))
    for finding in findings_sorted:
        _validate_finding(finding)
    return ClaimAuditReceipt(
        audit_hash=decision.audit_hash,
        claim_id=decision.claim_id,
        verdict=decision.verdict,
        finding_count=len(findings_sorted),
        supporting_count=len(decision.supporting_finding_ids),
        contradicting_count=len(decision.contradicting_finding_ids),
        inconclusive_count=len(decision.inconclusive_finding_ids),
        byte_length=len(decision.to_canonical_bytes()) + sum(len(item.to_canonical_bytes()) for item in findings_sorted),
        validation_passed=True,
        schema_version=decision.schema_version,
    )


def compile_claim_audit(
    raw_input: Mapping[str, Any],
    *,
    available_measurements: Mapping[str, Any] | None = None,
    available_criteria: Mapping[str, Any] | None = None,
    evidence_graph: Mapping[str, Any] | None = None,
) -> tuple[ClaimAuditInput, ClaimAuditDecision, tuple[ClaimAuditFinding, ...], ClaimAuditReceipt]:
    audit_input = normalize_claim_audit_input(dict(raw_input))
    decision, findings = run_claim_audit(
        audit_input,
        available_measurements=available_measurements,
        available_criteria=available_criteria,
        evidence_graph=evidence_graph,
    )
    receipt = build_claim_audit_receipt(decision, findings)
    return audit_input, decision, findings, receipt
