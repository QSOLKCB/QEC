"""v137.21.5 — Intake Firewall Kernel.

Deterministic intake firewall boundary for external artifacts entering interface,
benchmark, orchestration, and verification pathways.

This module is additive and decoder-untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

FIREWALL_VERSION = "v137.21.5"

DECISION_ALLOW = "allow"
DECISION_WARN = "warn"
DECISION_QUARANTINE = "quarantine"
DECISION_REJECT = "reject"

CHECK_CATEGORY_ORDER: Tuple[str, ...] = (
    "schema_integrity",
    "payload_safety",
    "metadata_policy",
    "provenance_integrity",
    "contract_compliance",
    "source_admissibility",
    "benchmark_protection",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _canonicalize(value: Any, *, field: str, strict_string_keys: bool) -> Any:
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for key in value.keys():
            if strict_string_keys and not isinstance(key, str):
                raise TypeError(f"{field} keys must be str")
            string_key = key if isinstance(key, str) else str(key)
            if string_key in normalized:
                raise ValueError(f"{field} contains duplicate canonical key: {string_key!r}")
            normalized[string_key] = _canonicalize(value[key], field=f"{field}.{string_key}", strict_string_keys=strict_string_keys)
        return {key: normalized[key] for key in sorted(normalized.keys())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item, field=field, strict_string_keys=strict_string_keys) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains non-finite float")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"{field} contains non-canonical type: {type(value).__name__}")


def _iter_mapping_paths(value: Any, base_path: str = "") -> Tuple[Tuple[str, str], ...]:
    rows: list[Tuple[str, str]] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node.keys()):
                if not isinstance(key, str):
                    continue
                child = f"{path}.{key}" if path else key
                rows.append((child, key))
                _walk(node[key], child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for idx, item in enumerate(node):
                _walk(item, f"{path}[{idx}]")

    _walk(value, base_path)
    return tuple(rows)


def _structure_metrics(value: Any) -> Dict[str, int]:
    metrics = {"max_nesting_depth": 0, "max_mapping_width": 0, "max_sequence_length": 0}

    def _walk(node: Any, depth: int) -> None:
        if depth > metrics["max_nesting_depth"]:
            metrics["max_nesting_depth"] = depth
        if isinstance(node, Mapping):
            width = len(node)
            if width > metrics["max_mapping_width"]:
                metrics["max_mapping_width"] = width
            for key in sorted(node.keys(), key=lambda k: str(k)):
                _walk(node[key], depth + 1)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            if len(node) > metrics["max_sequence_length"]:
                metrics["max_sequence_length"] = len(node)
            for item in node:
                _walk(item, depth + 1)

    _walk(value, 0)
    return metrics


def _freeze_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_snapshot(item) for item in value)
    return value


@dataclass(frozen=True)
class IntakeArtifact:
    artifact_id: str
    artifact_type: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    declared_contract: str
    source_channel: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "payload": _canonicalize(dict(self.payload), field="payload", strict_string_keys=True),
            "metadata": _canonicalize(dict(self.metadata), field="metadata", strict_string_keys=True),
            "provenance": _canonicalize(dict(self.provenance), field="provenance", strict_string_keys=True),
            "declared_contract": self.declared_contract,
            "source_channel": self.source_channel,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntakeFirewallPolicy:
    policy_version: str
    allowed_artifact_types: Tuple[str, ...]
    required_provenance_fields: Tuple[str, ...]
    forbidden_metadata_keys: Tuple[str, ...]
    max_nesting_depth: int
    max_mapping_width: int
    max_sequence_length: int
    required_declared_contracts: Tuple[str, ...]
    forbidden_payload_field_names: Tuple[str, ...]
    allowed_source_channels: Tuple[str, ...]
    strict_string_only_keys: bool
    quarantine_incomplete_provenance: bool
    advisory_payload_field_names: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "allowed_artifact_types": list(self.allowed_artifact_types),
            "required_provenance_fields": list(self.required_provenance_fields),
            "forbidden_metadata_keys": list(self.forbidden_metadata_keys),
            "max_nesting_depth": self.max_nesting_depth,
            "max_mapping_width": self.max_mapping_width,
            "max_sequence_length": self.max_sequence_length,
            "required_declared_contracts": list(self.required_declared_contracts),
            "forbidden_payload_field_names": list(self.forbidden_payload_field_names),
            "allowed_source_channels": list(self.allowed_source_channels),
            "strict_string_only_keys": self.strict_string_only_keys,
            "quarantine_incomplete_provenance": self.quarantine_incomplete_provenance,
            "advisory_payload_field_names": list(self.advisory_payload_field_names),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntakeFirewallCheck:
    name: str
    category: str
    required: bool
    passed: bool
    decision_effect: str
    observed_value: Any
    policy_value: Any
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "required": self.required,
            "passed": self.passed,
            "decision_effect": self.decision_effect,
            "observed_value": _canonicalize(self.observed_value, field="observed_value", strict_string_keys=True),
            "policy_value": _canonicalize(self.policy_value, field="policy_value", strict_string_keys=True),
            "explanation": self.explanation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntakeFirewallReport:
    artifact_id: str
    checks: Tuple[IntakeFirewallCheck, ...]
    counts_by_status: Mapping[str, int]
    failing_required_checks: Tuple[str, ...]
    warnings: Tuple[str, ...]
    quarantine_reasons: Tuple[str, ...]
    rejection_reasons: Tuple[str, ...]
    decision: str
    contract_valid: bool
    provenance_complete: bool
    policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts_by_status", MappingProxyType(dict(self.counts_by_status)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "checks": [check.to_dict() for check in self.checks],
            "counts_by_status": dict(self.counts_by_status),
            "failing_required_checks": list(self.failing_required_checks),
            "warnings": list(self.warnings),
            "quarantine_reasons": list(self.quarantine_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "decision": self.decision,
            "contract_valid": self.contract_valid,
            "provenance_complete": self.provenance_complete,
            "policy_version": self.policy_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntakeFirewallReceipt:
    version: str
    artifact_hash: str
    policy_hash: str
    report_hash: str
    decision: str
    admitted: bool
    quarantined: bool
    rejected: bool
    rationale: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "artifact_hash": self.artifact_hash,
            "policy_hash": self.policy_hash,
            "report_hash": self.report_hash,
            "decision": self.decision,
            "admitted": self.admitted,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "rationale": list(self.rationale),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class IntakeFirewallKernel:
    policy: IntakeFirewallPolicy

    @staticmethod
    def default_policy() -> IntakeFirewallPolicy:
        return IntakeFirewallPolicy(
            policy_version=FIREWALL_VERSION,
            allowed_artifact_types=("interface_capture", "benchmark_corpus", "orchestration_input", "verification_input"),
            required_provenance_fields=("origin", "source_id", "chain_of_custody"),
            forbidden_metadata_keys=(
                "benchmark_override",
                "sealed_corpus_override",
                "public_corpus_relabel",
                "decoder_patch",
            ),
            max_nesting_depth=6,
            max_mapping_width=64,
            max_sequence_length=512,
            required_declared_contracts=("interface.normalization.v1", "benchmark.formal.v1", "orchestration.input.v1"),
            forbidden_payload_field_names=("decoder_override", "benchmark_override", "sealed_corpus_override"),
            allowed_source_channels=("ingest_api", "benchmark_harness", "orchestration_bus", "verification_suite"),
            strict_string_only_keys=True,
            quarantine_incomplete_provenance=True,
            advisory_payload_field_names=("deprecated_field",),
        )

    @staticmethod
    def build_artifact(raw: Any, *, strict_string_keys: bool = True) -> IntakeArtifact:
        if not isinstance(raw, Mapping):
            raise TypeError("artifact must be mapping")
        required = (
            "artifact_id",
            "artifact_type",
            "payload",
            "metadata",
            "provenance",
            "declared_contract",
            "source_channel",
        )
        for field in required:
            if field not in raw:
                raise ValueError(f"artifact missing required field: {field}")
        payload = _canonicalize(raw["payload"], field="payload", strict_string_keys=strict_string_keys)
        metadata = _canonicalize(raw["metadata"], field="metadata", strict_string_keys=strict_string_keys)
        provenance = _canonicalize(raw["provenance"], field="provenance", strict_string_keys=strict_string_keys)
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be mapping")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be mapping")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be mapping")
        return IntakeArtifact(
            artifact_id=_normalize_text(raw["artifact_id"], field="artifact_id"),
            artifact_type=_normalize_text(raw["artifact_type"], field="artifact_type"),
            payload=payload,
            metadata=metadata,
            provenance=provenance,
            declared_contract=_normalize_text(raw["declared_contract"], field="declared_contract"),
            source_channel=_normalize_text(raw["source_channel"], field="source_channel"),
        )

    @staticmethod
    def build_policy(raw: Any) -> IntakeFirewallPolicy:
        if isinstance(raw, IntakeFirewallPolicy):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("policy must be mapping")

        def _tuple_str(field: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
            value = raw.get(field, default)
            if isinstance(value, (str, bytes, bytearray)):
                seq = (value,)
            elif isinstance(value, Sequence):
                seq = tuple(value)
            else:
                raise TypeError(f"{field} must be sequence of str")
            normalized = []
            for item in seq:
                text = _normalize_text(item, field=field)
                normalized.append(text)
            return tuple(sorted(set(normalized)))

        return IntakeFirewallPolicy(
            policy_version=_normalize_text(raw.get("policy_version", FIREWALL_VERSION), field="policy_version"),
            allowed_artifact_types=_tuple_str("allowed_artifact_types", IntakeFirewallKernel.default_policy().allowed_artifact_types),
            required_provenance_fields=_tuple_str("required_provenance_fields", IntakeFirewallKernel.default_policy().required_provenance_fields),
            forbidden_metadata_keys=_tuple_str("forbidden_metadata_keys", IntakeFirewallKernel.default_policy().forbidden_metadata_keys),
            max_nesting_depth=int(raw.get("max_nesting_depth", 6)),
            max_mapping_width=int(raw.get("max_mapping_width", 64)),
            max_sequence_length=int(raw.get("max_sequence_length", 512)),
            required_declared_contracts=_tuple_str("required_declared_contracts", IntakeFirewallKernel.default_policy().required_declared_contracts),
            forbidden_payload_field_names=_tuple_str("forbidden_payload_field_names", IntakeFirewallKernel.default_policy().forbidden_payload_field_names),
            allowed_source_channels=_tuple_str("allowed_source_channels", IntakeFirewallKernel.default_policy().allowed_source_channels),
            strict_string_only_keys=bool(raw.get("strict_string_only_keys", True)),
            quarantine_incomplete_provenance=bool(raw.get("quarantine_incomplete_provenance", True)),
            advisory_payload_field_names=_tuple_str("advisory_payload_field_names", ()),
        )

    def _check(self, *, name: str, category: str, required: bool, passed: bool, decision_effect: str, observed_value: Any, policy_value: Any, explanation: str) -> IntakeFirewallCheck:
        observed_snapshot = _freeze_snapshot(_canonicalize(observed_value, field="observed_value", strict_string_keys=True))
        policy_snapshot = _freeze_snapshot(_canonicalize(policy_value, field="policy_value", strict_string_keys=True))
        return IntakeFirewallCheck(
            name=name,
            category=category,
            required=required,
            passed=passed,
            decision_effect=decision_effect,
            observed_value=observed_snapshot,
            policy_value=policy_snapshot,
            explanation=explanation,
        )

    def evaluate(self, artifact: Any) -> Tuple[IntakeFirewallReport, IntakeFirewallReceipt]:
        checks: list[IntakeFirewallCheck] = []
        try:
            normalized_artifact = self.build_artifact(artifact, strict_string_keys=self.policy.strict_string_only_keys)
            envelope_ok = True
            envelope_error = ""
        except Exception as exc:
            normalized_artifact = IntakeArtifact(
                artifact_id="unknown",
                artifact_type="unknown",
                payload={},
                metadata={},
                provenance={},
                declared_contract="unknown",
                source_channel="unknown",
            )
            envelope_ok = False
            envelope_error = str(exc)

        checks.append(
            self._check(
                name="artifact_envelope_structure",
                category="schema_integrity",
                required=True,
                passed=envelope_ok,
                decision_effect=DECISION_REJECT if not envelope_ok else DECISION_ALLOW,
                observed_value="valid" if envelope_ok else envelope_error,
                policy_value="required_fields_and_types",
                explanation="Artifact envelope must satisfy required fields and canonical types.",
            )
        )

        metrics = _structure_metrics(normalized_artifact.payload)
        checks.append(
            self._check(
                name="payload_max_nesting_depth",
                category="payload_safety",
                required=True,
                passed=metrics["max_nesting_depth"] <= self.policy.max_nesting_depth,
                decision_effect=DECISION_REJECT,
                observed_value=metrics["max_nesting_depth"],
                policy_value=self.policy.max_nesting_depth,
                explanation="Payload nesting depth must be bounded.",
            )
        )
        checks.append(
            self._check(
                name="payload_max_mapping_width",
                category="payload_safety",
                required=True,
                passed=metrics["max_mapping_width"] <= self.policy.max_mapping_width,
                decision_effect=DECISION_REJECT,
                observed_value=metrics["max_mapping_width"],
                policy_value=self.policy.max_mapping_width,
                explanation="Payload mapping width must be bounded.",
            )
        )
        checks.append(
            self._check(
                name="payload_max_sequence_length",
                category="payload_safety",
                required=True,
                passed=metrics["max_sequence_length"] <= self.policy.max_sequence_length,
                decision_effect=DECISION_REJECT,
                observed_value=metrics["max_sequence_length"],
                policy_value=self.policy.max_sequence_length,
                explanation="Payload sequence length must be bounded.",
            )
        )

        payload_key_paths = _iter_mapping_paths(normalized_artifact.payload)
        forbidden_payload_hits = tuple(sorted(path for path, key in payload_key_paths if key in self.policy.forbidden_payload_field_names))
        checks.append(
            self._check(
                name="forbidden_payload_field_names",
                category="payload_safety",
                required=True,
                passed=len(forbidden_payload_hits) == 0,
                decision_effect=DECISION_REJECT,
                observed_value=list(forbidden_payload_hits),
                policy_value=list(self.policy.forbidden_payload_field_names),
                explanation="Forbidden payload fields are rejected.",
            )
        )
        advisory_hits = tuple(sorted(path for path, key in payload_key_paths if key in self.policy.advisory_payload_field_names))
        checks.append(
            self._check(
                name="advisory_payload_fields",
                category="payload_safety",
                required=False,
                passed=len(advisory_hits) == 0,
                decision_effect=DECISION_WARN,
                observed_value=list(advisory_hits),
                policy_value=list(self.policy.advisory_payload_field_names),
                explanation="Advisory payload fields are tracked as warnings.",
            )
        )

        metadata_paths = _iter_mapping_paths(normalized_artifact.metadata)
        forbidden_metadata_hits = tuple(sorted(path for path, key in metadata_paths if key in self.policy.forbidden_metadata_keys))
        checks.append(
            self._check(
                name="forbidden_metadata_keys",
                category="metadata_policy",
                required=True,
                passed=len(forbidden_metadata_hits) == 0,
                decision_effect=DECISION_REJECT,
                observed_value=list(forbidden_metadata_hits),
                policy_value=list(self.policy.forbidden_metadata_keys),
                explanation="Forbidden metadata keys are rejected.",
            )
        )

        missing_provenance = tuple(sorted(field for field in self.policy.required_provenance_fields if field not in normalized_artifact.provenance))
        provenance_complete = len(missing_provenance) == 0
        provenance_effect = DECISION_ALLOW
        if not provenance_complete:
            provenance_effect = DECISION_QUARANTINE if self.policy.quarantine_incomplete_provenance else DECISION_REJECT
        checks.append(
            self._check(
                name="required_provenance_fields",
                category="provenance_integrity",
                required=True,
                passed=provenance_complete,
                decision_effect=provenance_effect,
                observed_value=list(missing_provenance),
                policy_value=list(self.policy.required_provenance_fields),
                explanation="Required provenance must be complete.",
            )
        )

        contract_valid = normalized_artifact.declared_contract in self.policy.required_declared_contracts
        checks.append(
            self._check(
                name="declared_contract_allowed",
                category="contract_compliance",
                required=True,
                passed=contract_valid,
                decision_effect=DECISION_REJECT,
                observed_value=normalized_artifact.declared_contract,
                policy_value=list(self.policy.required_declared_contracts),
                explanation="Declared contract must be in the allowed contract set.",
            )
        )

        source_valid = normalized_artifact.source_channel in self.policy.allowed_source_channels
        checks.append(
            self._check(
                name="source_channel_allowed",
                category="source_admissibility",
                required=True,
                passed=source_valid,
                decision_effect=DECISION_REJECT,
                observed_value=normalized_artifact.source_channel,
                policy_value=list(self.policy.allowed_source_channels),
                explanation="Source channel must be in the allowed source set.",
            )
        )

        benchmark_type = normalized_artifact.artifact_type == "benchmark_corpus"
        artifact_type_valid = normalized_artifact.artifact_type in self.policy.allowed_artifact_types
        checks.append(
            self._check(
                name="artifact_type_allowed",
                category="schema_integrity",
                required=True,
                passed=artifact_type_valid,
                decision_effect=DECISION_REJECT,
                observed_value=normalized_artifact.artifact_type,
                policy_value=list(self.policy.allowed_artifact_types),
                explanation="Artifact type must be explicitly allowed.",
            )
        )

        benchmark_override_hits = tuple(
            sorted(
                path
                for path, key in metadata_paths
                if key in ("benchmark_override", "sealed_corpus_override", "public_corpus_relabel")
            )
        )
        benchmark_passed = (not benchmark_type) or (provenance_complete and len(benchmark_override_hits) == 0)
        benchmark_effect = DECISION_ALLOW
        if benchmark_type and not provenance_complete:
            benchmark_effect = DECISION_QUARANTINE if self.policy.quarantine_incomplete_provenance else DECISION_REJECT
        if benchmark_override_hits:
            benchmark_effect = DECISION_REJECT
        checks.append(
            self._check(
                name="benchmark_intake_guard",
                category="benchmark_protection",
                required=True,
                passed=benchmark_passed,
                decision_effect=benchmark_effect,
                observed_value={"is_benchmark": benchmark_type, "metadata_hits": list(benchmark_override_hits), "provenance_complete": provenance_complete},
                policy_value={"quarantine_incomplete_provenance": self.policy.quarantine_incomplete_provenance},
                explanation="Benchmark artifacts require complete provenance and no override metadata.",
            )
        )

        checks_tuple = tuple(sorted(checks, key=lambda c: (CHECK_CATEGORY_ORDER.index(c.category), c.name)))

        counts_by_status = {"passed": 0, "failed": 0, "required_failed": 0, "advisory_failed": 0}
        failing_required: list[str] = []
        warnings: list[str] = []
        quarantine_reasons: list[str] = []
        rejection_reasons: list[str] = []

        for check in checks_tuple:
            if check.passed:
                counts_by_status["passed"] += 1
                continue
            counts_by_status["failed"] += 1
            if check.required:
                counts_by_status["required_failed"] += 1
                failing_required.append(check.name)
            else:
                counts_by_status["advisory_failed"] += 1
                warnings.append(check.name)
            if check.decision_effect == DECISION_REJECT:
                rejection_reasons.append(check.name)
            elif check.decision_effect == DECISION_QUARANTINE:
                quarantine_reasons.append(check.name)
            elif check.decision_effect == DECISION_WARN:
                warnings.append(check.name)

        if rejection_reasons:
            decision = DECISION_REJECT
        elif quarantine_reasons:
            decision = DECISION_QUARANTINE
        elif warnings:
            decision = DECISION_WARN
        else:
            decision = DECISION_ALLOW

        report = IntakeFirewallReport(
            artifact_id=normalized_artifact.artifact_id,
            checks=checks_tuple,
            counts_by_status=counts_by_status,
            failing_required_checks=tuple(sorted(set(failing_required))),
            warnings=tuple(sorted(set(warnings))),
            quarantine_reasons=tuple(sorted(set(quarantine_reasons))),
            rejection_reasons=tuple(sorted(set(rejection_reasons))),
            decision=decision,
            contract_valid=contract_valid,
            provenance_complete=provenance_complete,
            policy_version=self.policy.policy_version,
        )

        rationale = tuple(
            sorted(
                [f"decision:{decision}"]
                + [f"reject:{name}" for name in report.rejection_reasons]
                + [f"quarantine:{name}" for name in report.quarantine_reasons]
                + [f"warn:{name}" for name in report.warnings]
            )
        )
        receipt = IntakeFirewallReceipt(
            version=FIREWALL_VERSION,
            artifact_hash=normalized_artifact.stable_hash(),
            policy_hash=self.policy.stable_hash(),
            report_hash=report.stable_hash(),
            decision=decision,
            admitted=decision in (DECISION_ALLOW, DECISION_WARN),
            quarantined=decision == DECISION_QUARANTINE,
            rejected=decision == DECISION_REJECT,
            rationale=rationale,
        )
        return report, receipt


def run_intake_firewall(*, artifact: Any, policy: Any | None = None) -> Tuple[IntakeFirewallReport, IntakeFirewallReceipt]:
    normalized_policy = IntakeFirewallKernel.default_policy() if policy is None else IntakeFirewallKernel.build_policy(policy)
    kernel = IntakeFirewallKernel(policy=normalized_policy)
    return kernel.evaluate(artifact)


def validate_intake_artifact(artifact: Any, *, policy: Any | None = None) -> Dict[str, Any]:
    report, _ = run_intake_firewall(artifact=artifact, policy=policy)
    return {
        "valid": report.decision in (DECISION_ALLOW, DECISION_WARN),
        "decision": report.decision,
        "rejection_reasons": list(report.rejection_reasons),
        "quarantine_reasons": list(report.quarantine_reasons),
        "warnings": list(report.warnings),
    }


def summarize_intake_firewall_report(report: IntakeFirewallReport) -> Dict[str, Any]:
    return {
        "artifact_id": report.artifact_id,
        "decision": report.decision,
        "contract_valid": report.contract_valid,
        "provenance_complete": report.provenance_complete,
        "failing_required_checks": list(report.failing_required_checks),
        "warnings": list(report.warnings),
        "quarantine_reasons": list(report.quarantine_reasons),
        "rejection_reasons": list(report.rejection_reasons),
    }
