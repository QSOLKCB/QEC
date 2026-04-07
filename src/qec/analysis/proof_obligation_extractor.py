from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

PROOF_OBLIGATION_SCHEMA_VERSION = "v137.10.5"
_ALLOWED_VERDICTS = frozenset({"supported", "contradicted", "inconclusive"})
_ALLOWED_OBLIGATION_TYPES = frozenset(
    {
        "evidence_presence",
        "criterion_satisfaction",
        "lineage_integrity",
        "measurement_availability",
        "conflict_absence",
        "replay_consistency",
    }
)
_ALLOWED_OBLIGATION_STATUS = frozenset({"required", "satisfied", "unsatisfied", "inapplicable"})

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | tuple["JsonValue", ...] | tuple[tuple[str, "JsonValue"], ...]


@dataclass(frozen=True)
class _EmptyObjectMarker:
    ...


_EMPTY_OBJECT = _EmptyObjectMarker()


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
        if not deduped:
            return _EMPTY_OBJECT
        return tuple(deduped)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_normalize_json_value(item, path=path) for item in list(value))
    raise ValueError(f"{path} contains unsupported type: {type(value).__name__}")


def _json_value_to_python(value: JsonValue) -> Any:
    if isinstance(value, _EmptyObjectMarker):
        return {}
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
    for relation_id, relation_payload in value.items():
        key = _normalize_token(relation_id, name="expected_relations relation")
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
        raise ValueError("expected_relations contains duplicate relation IDs")
    return tuple(items)


def _extract_related_nodes(relation: tuple[tuple[str, JsonValue], ...]) -> tuple[str, ...]:
    for key, value in relation:
        if key != "node_ids":
            continue
        if not isinstance(value, tuple):
            raise ValueError("expected_relations.node_ids must be a sequence")
        nodes: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("expected_relations.node_ids must contain strings")
            nodes.append(_normalize_token(item, name="expected_relations node_id"))
        return _normalize_string_tuple(nodes, name="expected_relations node_ids")
    return ()


def _normalize_available_ids(values: Any, *, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    return _normalize_string_tuple(values, name=name)


def _extract_replay_pass(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key in ("deterministic_pass", "pass", "passed", "validation_passed"):
            if key in value:
                return bool(value[key])
    return bool(value)


def _obligation_id(*, claim_id: str, obligation_type: str, scope: str) -> str:
    payload = f"{claim_id}|{obligation_type}|{scope}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProofObligationInput:
    claim_id: str
    claim_text: str
    experiment_hash: str
    evidence_graph_hash: str
    audit_hash: str
    verdict: str
    measurement_ids: tuple[str, ...]
    criterion_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]
    expected_relations: tuple[tuple[str, tuple[tuple[str, JsonValue], ...]], ...]
    provenance: tuple[tuple[str, JsonValue], ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "experiment_hash": self.experiment_hash,
            "evidence_graph_hash": self.evidence_graph_hash,
            "audit_hash": self.audit_hash,
            "verdict": self.verdict,
            "measurement_ids": list(self.measurement_ids),
            "criterion_ids": list(self.criterion_ids),
            "finding_ids": list(self.finding_ids),
            "expected_relations": {key: _json_value_to_python(value) for key, value in self.expected_relations},
            "provenance": _json_value_to_python(self.provenance),
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ProofObligation:
    obligation_id: str
    obligation_type: str
    claim_id: str
    related_measurement_id: str
    related_criterion_id: str
    related_finding_id: str
    related_node_ids: tuple[str, ...]
    requirement_text: str
    blocking: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "obligation_type": self.obligation_type,
            "claim_id": self.claim_id,
            "related_measurement_id": self.related_measurement_id,
            "related_criterion_id": self.related_criterion_id,
            "related_finding_id": self.related_finding_id,
            "related_node_ids": list(self.related_node_ids),
            "requirement_text": self.requirement_text,
            "blocking": self.blocking,
            "status": self.status,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ProofObligationReport:
    report_id: str
    claim_id: str
    verdict: str
    total_obligations: int
    blocking_obligations: int
    satisfied_obligations: int
    unsatisfied_obligations: int
    inapplicable_obligations: int
    obligations: tuple[ProofObligation, ...]
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "claim_id": self.claim_id,
            "verdict": self.verdict,
            "total_obligations": self.total_obligations,
            "blocking_obligations": self.blocking_obligations,
            "satisfied_obligations": self.satisfied_obligations,
            "unsatisfied_obligations": self.unsatisfied_obligations,
            "inapplicable_obligations": self.inapplicable_obligations,
            "obligations": [item.to_dict() for item in self.obligations],
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ProofObligationReceipt:
    report_hash: str
    report_id: str
    claim_id: str
    total_obligations: int
    blocking_obligations: int
    satisfied_obligations: int
    unsatisfied_obligations: int
    byte_length: int
    validation_passed: bool
    schema_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_hash": self.report_hash,
            "report_id": self.report_id,
            "claim_id": self.claim_id,
            "total_obligations": self.total_obligations,
            "blocking_obligations": self.blocking_obligations,
            "satisfied_obligations": self.satisfied_obligations,
            "unsatisfied_obligations": self.unsatisfied_obligations,
            "byte_length": self.byte_length,
            "validation_passed": self.validation_passed,
            "schema_version": self.schema_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def normalize_proof_obligation_input(raw_input: ProofObligationInput | Mapping[str, Any]) -> ProofObligationInput:
    if isinstance(raw_input, ProofObligationInput):
        validate_proof_obligation_input(raw_input)
        return ProofObligationInput(**raw_input.__dict__)
    if not isinstance(raw_input, Mapping):
        raise ValueError("raw_input must be mapping or ProofObligationInput")

    raw_schema = raw_input.get("schema_version")
    if raw_schema is not None and (not isinstance(raw_schema, str) or raw_schema.strip() == ""):
        raise ValueError("schema_version must be a non-empty string if provided")
    schema_version = _normalize_token(raw_schema if raw_schema is not None else PROOF_OBLIGATION_SCHEMA_VERSION, name="schema_version")

    obligation_input = ProofObligationInput(
        claim_id=_normalize_token(raw_input.get("claim_id"), name="claim_id"),
        claim_text=_normalize_token(raw_input.get("claim_text"), name="claim_text"),
        experiment_hash=_normalize_token(raw_input.get("experiment_hash"), name="experiment_hash"),
        evidence_graph_hash=_normalize_token(raw_input.get("evidence_graph_hash"), name="evidence_graph_hash"),
        audit_hash=_normalize_token(raw_input.get("audit_hash"), name="audit_hash"),
        verdict=_normalize_token(raw_input.get("verdict"), name="verdict").lower(),
        measurement_ids=_normalize_string_tuple(raw_input.get("measurement_ids"), name="measurement_ids"),
        criterion_ids=_normalize_string_tuple(raw_input.get("criterion_ids"), name="criterion_ids"),
        finding_ids=_normalize_string_tuple(raw_input.get("finding_ids"), name="finding_ids"),
        expected_relations=_normalize_expected_relations(raw_input.get("expected_relations")),
        provenance=_normalize_json_value(raw_input.get("provenance", {}), path="provenance"),  # type: ignore[arg-type]
        schema_version=schema_version,
    )
    validate_proof_obligation_input(obligation_input)
    return obligation_input


def validate_proof_obligation_input(obligation_input: ProofObligationInput) -> None:
    if obligation_input.schema_version != PROOF_OBLIGATION_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if obligation_input.verdict not in _ALLOWED_VERDICTS:
        raise ValueError("unsupported verdict")
    _normalize_token(obligation_input.claim_id, name="claim_id")
    _normalize_token(obligation_input.claim_text, name="claim_text")
    _normalize_token(obligation_input.experiment_hash, name="experiment_hash")
    _normalize_token(obligation_input.evidence_graph_hash, name="evidence_graph_hash")
    _normalize_token(obligation_input.audit_hash, name="audit_hash")
    _normalize_string_tuple(obligation_input.measurement_ids, name="measurement_ids")
    _normalize_string_tuple(obligation_input.criterion_ids, name="criterion_ids")
    _normalize_string_tuple(obligation_input.finding_ids, name="finding_ids")
    for relation_id, relation_payload in obligation_input.expected_relations:
        _normalize_token(relation_id, name="expected_relations relation")
        _extract_related_nodes(relation_payload)
    _normalize_json_value(obligation_input.provenance, path="provenance")


def _validate_obligation(obligation: ProofObligation) -> None:
    if obligation.obligation_type not in _ALLOWED_OBLIGATION_TYPES:
        raise ValueError("unsupported obligation_type")
    if obligation.status not in _ALLOWED_OBLIGATION_STATUS:
        raise ValueError("unsupported obligation status")
    _normalize_token(obligation.obligation_id, name="obligation_id")
    _normalize_token(obligation.claim_id, name="claim_id")
    _normalize_token(obligation.requirement_text, name="requirement_text")
    _normalize_string_tuple(obligation.related_node_ids, name="related_node_ids")


def _build_report(claim_id: str, verdict: str, obligations: Sequence[ProofObligation], schema_version: str) -> ProofObligationReport:
    sorted_obligations = tuple(sorted(obligations, key=lambda item: (item.obligation_type, item.obligation_id)))
    ids = [item.obligation_id for item in sorted_obligations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate obligation IDs in report construction")
    for item in sorted_obligations:
        _validate_obligation(item)

    total = len(sorted_obligations)
    blocking = sum(1 for item in sorted_obligations if item.blocking and item.status == "unsatisfied")
    satisfied = sum(1 for item in sorted_obligations if item.status == "satisfied")
    unsatisfied = sum(1 for item in sorted_obligations if item.status == "unsatisfied")
    inapplicable = sum(1 for item in sorted_obligations if item.status == "inapplicable")
    report_material = {
        "claim_id": claim_id,
        "verdict": verdict,
        "obligations": [item.to_dict() for item in sorted_obligations],
        "schema_version": schema_version,
    }
    report_id = hashlib.sha256(_canonical_json(report_material).encode("utf-8")).hexdigest()
    report = ProofObligationReport(
        report_id=report_id,
        claim_id=claim_id,
        verdict=verdict,
        total_obligations=total,
        blocking_obligations=blocking,
        satisfied_obligations=satisfied,
        unsatisfied_obligations=unsatisfied,
        inapplicable_obligations=inapplicable,
        obligations=sorted_obligations,
        schema_version=schema_version,
    )
    return report


def extract_proof_obligations(
    obligation_input: ProofObligationInput,
    *,
    available_findings: Sequence[str] | None = None,
    available_measurements: Sequence[str] | None = None,
    available_criteria: Sequence[str] | None = None,
    evidence_graph: Mapping[str, Any] | None = None,
    replay_report: Mapping[str, Any] | None = None,
) -> tuple[ProofObligation, ...]:
    validate_proof_obligation_input(obligation_input)

    finding_ids = _normalize_available_ids(available_findings, name="available_findings")
    measurement_ids = _normalize_available_ids(available_measurements, name="available_measurements")
    criterion_ids = _normalize_available_ids(available_criteria, name="available_criteria")

    finding_set = set(finding_ids)
    measurement_set = set(measurement_ids)
    criterion_set = set(criterion_ids)

    evidence_nodes: set[str] = set()
    if evidence_graph is not None:
        if not isinstance(evidence_graph, Mapping):
            raise ValueError("evidence_graph must be mapping")
        # Accept "nodes" (canonical) or "node_ids" (legacy) key.
        if "nodes" in evidence_graph:
            raw_nodes = evidence_graph["nodes"]
            evidence_nodes.update(_normalize_string_tuple(raw_nodes, name="evidence_graph.nodes"))
        elif "node_ids" in evidence_graph:
            raw_nodes = evidence_graph["node_ids"]
            evidence_nodes.update(_normalize_string_tuple(raw_nodes, name="evidence_graph.node_ids"))

    obligations: list[ProofObligation] = []
    for measurement_id in obligation_input.measurement_ids:
        status = "satisfied" if measurement_id in measurement_set else "unsatisfied"
        obligations.append(
            ProofObligation(
                obligation_id=_obligation_id(
                    claim_id=obligation_input.claim_id,
                    obligation_type="measurement_availability",
                    scope=measurement_id,
                ),
                obligation_type="measurement_availability",
                claim_id=obligation_input.claim_id,
                related_measurement_id=measurement_id,
                related_criterion_id="",
                related_finding_id="",
                related_node_ids=(),
                requirement_text=f"measurement {measurement_id} must be available",
                blocking=status == "unsatisfied",
                status=status,
            )
        )

    for criterion_id in obligation_input.criterion_ids:
        status = "satisfied" if criterion_id in criterion_set else "unsatisfied"
        obligations.append(
            ProofObligation(
                obligation_id=_obligation_id(
                    claim_id=obligation_input.claim_id,
                    obligation_type="criterion_satisfaction",
                    scope=criterion_id,
                ),
                obligation_type="criterion_satisfaction",
                claim_id=obligation_input.claim_id,
                related_measurement_id="",
                related_criterion_id=criterion_id,
                related_finding_id="",
                related_node_ids=(),
                requirement_text=f"criterion {criterion_id} must be satisfied",
                blocking=status == "unsatisfied",
                status=status,
            )
        )

    for relation_id, relation_payload in obligation_input.expected_relations:
        related_nodes = _extract_related_nodes(relation_payload)
        if related_nodes:
            status = "satisfied" if set(related_nodes).issubset(evidence_nodes) else "unsatisfied"
            obligations.append(
                ProofObligation(
                    obligation_id=_obligation_id(
                        claim_id=obligation_input.claim_id,
                        obligation_type="lineage_integrity",
                        scope=f"{relation_id}:{','.join(related_nodes)}",
                    ),
                    obligation_type="lineage_integrity",
                    claim_id=obligation_input.claim_id,
                    related_measurement_id="",
                    related_criterion_id="",
                    related_finding_id="",
                    related_node_ids=related_nodes,
                    requirement_text=f"lineage nodes {','.join(related_nodes)} must be linked",
                    blocking=status == "unsatisfied",
                    status=status,
                )
            )
        else:
            status = "required" if evidence_graph is not None else "unsatisfied"
            obligations.append(
                ProofObligation(
                    obligation_id=_obligation_id(
                        claim_id=obligation_input.claim_id,
                        obligation_type="evidence_presence",
                        scope=relation_id,
                    ),
                    obligation_type="evidence_presence",
                    claim_id=obligation_input.claim_id,
                    related_measurement_id="",
                    related_criterion_id="",
                    related_finding_id="",
                    related_node_ids=(),
                    requirement_text=f"evidence relation {relation_id} must be present",
                    blocking=evidence_graph is None,
                    status=status,
                )
            )

    conflict_ids = tuple(fid for fid in obligation_input.finding_ids if fid in finding_set)
    conflict_from_verdict = obligation_input.verdict == "contradicted"
    conflict_present = conflict_from_verdict or bool(conflict_ids)
    status = "unsatisfied" if conflict_present else "satisfied"
    obligations.append(
        ProofObligation(
            obligation_id=_obligation_id(
                claim_id=obligation_input.claim_id,
                obligation_type="conflict_absence",
                scope=",".join(conflict_ids) if conflict_ids else obligation_input.verdict,
            ),
            obligation_type="conflict_absence",
            claim_id=obligation_input.claim_id,
            related_measurement_id="",
            related_criterion_id="",
            related_finding_id=conflict_ids[0] if conflict_ids else "",
            related_node_ids=(),
            requirement_text="no conflicting evidence findings may remain unresolved",
            blocking=status == "unsatisfied",
            status=status,
        )
    )

    if replay_report is not None:
        if not isinstance(replay_report, Mapping):
            raise ValueError("replay_report must be mapping")
        replay_pass = _extract_replay_pass(replay_report)
        replay_status = "satisfied" if replay_pass else "unsatisfied"
        obligations.append(
            ProofObligation(
                obligation_id=_obligation_id(
                    claim_id=obligation_input.claim_id,
                    obligation_type="replay_consistency",
                    scope=str(replay_pass).lower(),
                ),
                obligation_type="replay_consistency",
                claim_id=obligation_input.claim_id,
                related_measurement_id="",
                related_criterion_id="",
                related_finding_id="",
                related_node_ids=(),
                requirement_text="replay report must pass determinism checks",
                blocking=replay_status == "unsatisfied",
                status=replay_status,
            )
        )

    return tuple(sorted(obligations, key=lambda item: (item.obligation_type, item.obligation_id)))


def stable_proof_obligation_hash(report: ProofObligationReport) -> str:
    if report.schema_version != PROOF_OBLIGATION_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    return hashlib.sha256(report.to_canonical_bytes()).hexdigest()


def validate_proof_obligation_report(report: ProofObligationReport) -> None:
    if report.schema_version != PROOF_OBLIGATION_SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    ids = [item.obligation_id for item in report.obligations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate obligation IDs in report")
    for item in report.obligations:
        _validate_obligation(item)
    recomputed_total = len(report.obligations)
    recomputed_blocking = sum(1 for item in report.obligations if item.blocking and item.status == "unsatisfied")
    recomputed_satisfied = sum(1 for item in report.obligations if item.status == "satisfied")
    recomputed_unsatisfied = sum(1 for item in report.obligations if item.status == "unsatisfied")
    if report.total_obligations != recomputed_total:
        raise ValueError("total_obligations count mismatch")
    if report.blocking_obligations != recomputed_blocking:
        raise ValueError("blocking_obligations count mismatch")
    if report.satisfied_obligations != recomputed_satisfied:
        raise ValueError("satisfied_obligations count mismatch")
    if report.unsatisfied_obligations != recomputed_unsatisfied:
        raise ValueError("unsatisfied_obligations count mismatch")


def build_proof_obligation_receipt(report: ProofObligationReport) -> ProofObligationReceipt:
    validate_proof_obligation_report(report)
    report_hash = stable_proof_obligation_hash(report)
    return ProofObligationReceipt(
        report_hash=report_hash,
        report_id=report.report_id,
        claim_id=report.claim_id,
        total_obligations=report.total_obligations,
        blocking_obligations=report.blocking_obligations,
        satisfied_obligations=report.satisfied_obligations,
        unsatisfied_obligations=report.unsatisfied_obligations,
        byte_length=len(report.to_canonical_bytes()),
        validation_passed=True,
        schema_version=report.schema_version,
    )


def compile_proof_obligation_report(
    raw_input: ProofObligationInput | Mapping[str, Any],
    *,
    available_findings: Sequence[str] | None = None,
    available_measurements: Sequence[str] | None = None,
    available_criteria: Sequence[str] | None = None,
    evidence_graph: Mapping[str, Any] | None = None,
    replay_report: Mapping[str, Any] | None = None,
) -> tuple[ProofObligationInput, ProofObligationReport, ProofObligationReceipt]:
    obligation_input = normalize_proof_obligation_input(raw_input)
    obligations = extract_proof_obligations(
        obligation_input,
        available_findings=available_findings,
        available_measurements=available_measurements,
        available_criteria=available_criteria,
        evidence_graph=evidence_graph,
        replay_report=replay_report,
    )
    report = _build_report(
        claim_id=obligation_input.claim_id,
        verdict=obligation_input.verdict,
        obligations=obligations,
        schema_version=obligation_input.schema_version,
    )
    receipt = build_proof_obligation_receipt(report)
    return obligation_input, report, receipt


__all__ = [
    "PROOF_OBLIGATION_SCHEMA_VERSION",
    "ProofObligation",
    "ProofObligationInput",
    "ProofObligationReceipt",
    "ProofObligationReport",
    "build_proof_obligation_receipt",
    "compile_proof_obligation_report",
    "extract_proof_obligations",
    "normalize_proof_obligation_input",
    "stable_proof_obligation_hash",
    "validate_proof_obligation_input",
    "validate_proof_obligation_report",
]
