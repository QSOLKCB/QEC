"""v137.21.7 — Benchmark Vault + Poisoning Resistance Pack.

Deterministic benchmark custody boundary for benchmark corpora and evaluation
packs. This module is additive and decoder-untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

BENCHMARK_VAULT_VERSION = "v137.21.7"

DECISION_ALLOW = "allow"
DECISION_WARN = "warn"
DECISION_QUARANTINE = "quarantine"
DECISION_REJECT = "reject"

CHECK_CATEGORY_ORDER: Tuple[str, ...] = (
    "benchmark_identity",
    "manifest_integrity",
    "metadata_policy",
    "provenance_integrity",
    "contract_compliance",
    "source_admissibility",
    "corpus_classification",
    "sealed_public_separation",
    "poisoning_resistance",
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


def _canonicalize(value: Any, *, field: str, strict_string_keys: bool = True) -> Any:
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


def _freeze_snapshot(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_snapshot(item) for key, item in value.items()})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_snapshot(item) for item in value)
    return value


def _iter_mapping_paths(value: Any, base_path: str = "") -> Tuple[Tuple[str, str], ...]:
    rows: list[Tuple[str, str]] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node.keys()):
                child = f"{path}.{key}" if path else key
                rows.append((child, key))
                _walk(node[key], child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for idx, item in enumerate(node):
                _walk(item, f"{path}[{idx}]")

    _walk(value, base_path)
    return tuple(rows)


def _expect_bool(raw: Mapping[str, Any], field: str, default: bool) -> bool:
    value = raw.get(field, default)
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool")
    return value


def _tuple_str(raw: Mapping[str, Any], field: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    value = raw.get(field, default)
    if isinstance(value, (str, bytes, bytearray)):
        seq = (value,)
    elif isinstance(value, Sequence):
        seq = tuple(value)
    else:
        raise TypeError(f"{field} must be sequence of str")
    normalized = []
    for item in seq:
        normalized.append(_normalize_text(item, field=field))
    return tuple(sorted(set(normalized)))


def _classification_tokens(classification: str) -> Tuple[str, ...]:
    lowered = classification.lower()
    for token in ("|", "+", "/", ","):
        lowered = lowered.replace(token, " ")
    parts = tuple(part for part in lowered.split() if part)
    if not parts:
        return (lowered.strip(),)
    return parts


@dataclass(frozen=True)
class BenchmarkArtifact:
    benchmark_id: str
    benchmark_type: str
    manifest: Mapping[str, Any]
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    declared_contract: str
    source_channel: str
    corpus_classification: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_type": self.benchmark_type,
            "manifest": _canonicalize(dict(self.manifest), field="manifest"),
            "metadata": _canonicalize(dict(self.metadata), field="metadata"),
            "provenance": _canonicalize(dict(self.provenance), field="provenance"),
            "declared_contract": self.declared_contract,
            "source_channel": self.source_channel,
            "corpus_classification": self.corpus_classification,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkVaultPolicy:
    policy_version: str
    allowed_benchmark_types: Tuple[str, ...]
    allowed_corpus_classifications: Tuple[str, ...]
    sealed_classifications: Tuple[str, ...]
    public_classifications: Tuple[str, ...]
    required_provenance_fields: Tuple[str, ...]
    required_manifest_fields: Tuple[str, ...]
    forbidden_metadata_keys: Tuple[str, ...]
    forbidden_manifest_field_names: Tuple[str, ...]
    allowed_source_channels: Tuple[str, ...]
    allowed_declared_contracts: Tuple[str, ...]
    require_manifest_hash: bool
    require_lineage_hash: bool
    quarantine_incomplete_provenance: bool
    reject_sealed_public_mixing: bool
    reject_manifest_identity_conflicts: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "allowed_benchmark_types": list(self.allowed_benchmark_types),
            "allowed_corpus_classifications": list(self.allowed_corpus_classifications),
            "sealed_classifications": list(self.sealed_classifications),
            "public_classifications": list(self.public_classifications),
            "required_provenance_fields": list(self.required_provenance_fields),
            "required_manifest_fields": list(self.required_manifest_fields),
            "forbidden_metadata_keys": list(self.forbidden_metadata_keys),
            "forbidden_manifest_field_names": list(self.forbidden_manifest_field_names),
            "allowed_source_channels": list(self.allowed_source_channels),
            "allowed_declared_contracts": list(self.allowed_declared_contracts),
            "require_manifest_hash": self.require_manifest_hash,
            "require_lineage_hash": self.require_lineage_hash,
            "quarantine_incomplete_provenance": self.quarantine_incomplete_provenance,
            "reject_sealed_public_mixing": self.reject_sealed_public_mixing,
            "reject_manifest_identity_conflicts": self.reject_manifest_identity_conflicts,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkVaultCheck:
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
            "observed_value": _canonicalize(self.observed_value, field="observed_value"),
            "policy_value": _canonicalize(self.policy_value, field="policy_value"),
            "explanation": self.explanation,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkManifestIdentity:
    benchmark_id: str
    manifest_hash: str
    lineage_hash: str
    corpus_classification: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "manifest_hash": self.manifest_hash,
            "lineage_hash": self.lineage_hash,
            "corpus_classification": self.corpus_classification,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkVaultReport:
    benchmark_id: str
    checks: Tuple[BenchmarkVaultCheck, ...]
    counts_by_status: Mapping[str, int]
    failing_required_checks: Tuple[str, ...]
    warnings: Tuple[str, ...]
    quarantine_reasons: Tuple[str, ...]
    rejection_reasons: Tuple[str, ...]
    decision: str
    vault_admissible: bool
    quarantined: bool
    rejected: bool
    sealed_corpus: bool
    public_corpus: bool
    provenance_complete: bool
    manifest_complete: bool
    policy_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "counts_by_status", MappingProxyType(dict(self.counts_by_status)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "checks": [check.to_dict() for check in self.checks],
            "counts_by_status": dict(self.counts_by_status),
            "failing_required_checks": list(self.failing_required_checks),
            "warnings": list(self.warnings),
            "quarantine_reasons": list(self.quarantine_reasons),
            "rejection_reasons": list(self.rejection_reasons),
            "decision": self.decision,
            "vault_admissible": self.vault_admissible,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "sealed_corpus": self.sealed_corpus,
            "public_corpus": self.public_corpus,
            "provenance_complete": self.provenance_complete,
            "manifest_complete": self.manifest_complete,
            "policy_version": self.policy_version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkVaultReceipt:
    benchmark_hash: str
    policy_hash: str
    report_hash: str
    decision: str
    vault_admissible: bool
    quarantined: bool
    rejected: bool
    sealed_corpus: bool
    public_corpus: bool
    rationale: Tuple[str, ...]
    version: str = BENCHMARK_VAULT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "benchmark_hash": self.benchmark_hash,
            "policy_hash": self.policy_hash,
            "report_hash": self.report_hash,
            "decision": self.decision,
            "vault_admissible": self.vault_admissible,
            "quarantined": self.quarantined,
            "rejected": self.rejected,
            "sealed_corpus": self.sealed_corpus,
            "public_corpus": self.public_corpus,
            "rationale": list(self.rationale),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class BenchmarkVaultKernel:
    policy: BenchmarkVaultPolicy

    @staticmethod
    def default_policy() -> BenchmarkVaultPolicy:
        return BenchmarkVaultPolicy(
            policy_version=BENCHMARK_VAULT_VERSION,
            allowed_benchmark_types=("benchmark_corpus", "evaluation_pack"),
            allowed_corpus_classifications=("public", "sealed"),
            sealed_classifications=("sealed",),
            public_classifications=("public",),
            required_provenance_fields=("origin", "source_id", "chain_of_custody"),
            required_manifest_fields=("benchmark_id",),
            forbidden_metadata_keys=(
                "benchmark_override",
                "sealed_corpus_override",
                "public_corpus_relabel",
                "evaluation_status_override",
            ),
            forbidden_manifest_field_names=(
                "benchmark_override",
                "sealed_corpus_override",
                "public_corpus_relabel",
            ),
            allowed_source_channels=("benchmark_harness", "verification_suite"),
            allowed_declared_contracts=("benchmark.formal.v1", "benchmark.vault.v1"),
            require_manifest_hash=True,
            require_lineage_hash=True,
            quarantine_incomplete_provenance=True,
            reject_sealed_public_mixing=True,
            reject_manifest_identity_conflicts=True,
        )

    @staticmethod
    def build_policy(raw: Any) -> BenchmarkVaultPolicy:
        if isinstance(raw, BenchmarkVaultPolicy):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("policy must be mapping")
        default = BenchmarkVaultKernel.default_policy()
        return BenchmarkVaultPolicy(
            policy_version=_normalize_text(raw.get("policy_version", BENCHMARK_VAULT_VERSION), field="policy_version"),
            allowed_benchmark_types=_tuple_str(raw, "allowed_benchmark_types", default.allowed_benchmark_types),
            allowed_corpus_classifications=_tuple_str(raw, "allowed_corpus_classifications", default.allowed_corpus_classifications),
            sealed_classifications=_tuple_str(raw, "sealed_classifications", default.sealed_classifications),
            public_classifications=_tuple_str(raw, "public_classifications", default.public_classifications),
            required_provenance_fields=_tuple_str(raw, "required_provenance_fields", default.required_provenance_fields),
            required_manifest_fields=_tuple_str(raw, "required_manifest_fields", default.required_manifest_fields),
            forbidden_metadata_keys=_tuple_str(raw, "forbidden_metadata_keys", default.forbidden_metadata_keys),
            forbidden_manifest_field_names=_tuple_str(raw, "forbidden_manifest_field_names", default.forbidden_manifest_field_names),
            allowed_source_channels=_tuple_str(raw, "allowed_source_channels", default.allowed_source_channels),
            allowed_declared_contracts=_tuple_str(raw, "allowed_declared_contracts", default.allowed_declared_contracts),
            require_manifest_hash=_expect_bool(raw, "require_manifest_hash", default.require_manifest_hash),
            require_lineage_hash=_expect_bool(raw, "require_lineage_hash", default.require_lineage_hash),
            quarantine_incomplete_provenance=_expect_bool(raw, "quarantine_incomplete_provenance", default.quarantine_incomplete_provenance),
            reject_sealed_public_mixing=_expect_bool(raw, "reject_sealed_public_mixing", default.reject_sealed_public_mixing),
            reject_manifest_identity_conflicts=_expect_bool(raw, "reject_manifest_identity_conflicts", default.reject_manifest_identity_conflicts),
        )

    @staticmethod
    def build_artifact(raw: Any) -> BenchmarkArtifact:
        if not isinstance(raw, Mapping):
            raise TypeError("artifact must be mapping")
        required = (
            "benchmark_id",
            "benchmark_type",
            "manifest",
            "metadata",
            "provenance",
            "declared_contract",
            "source_channel",
            "corpus_classification",
        )
        for field in required:
            if field not in raw:
                raise ValueError(f"artifact missing required field: {field}")
        unknown_keys = sorted(str(key) for key in raw.keys() if str(key) not in set(required))
        if unknown_keys:
            raise ValueError(f"artifact contains unknown top-level keys: {unknown_keys}")

        manifest = _canonicalize(raw["manifest"], field="manifest", strict_string_keys=True)
        metadata = _canonicalize(raw["metadata"], field="metadata", strict_string_keys=True)
        provenance = _canonicalize(raw["provenance"], field="provenance", strict_string_keys=True)
        if not isinstance(manifest, Mapping):
            raise TypeError("manifest must be mapping")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be mapping")
        if not isinstance(provenance, Mapping):
            raise TypeError("provenance must be mapping")

        return BenchmarkArtifact(
            benchmark_id=_normalize_text(raw["benchmark_id"], field="benchmark_id"),
            benchmark_type=_normalize_text(raw["benchmark_type"], field="benchmark_type"),
            manifest=manifest,
            metadata=metadata,
            provenance=provenance,
            declared_contract=_normalize_text(raw["declared_contract"], field="declared_contract"),
            source_channel=_normalize_text(raw["source_channel"], field="source_channel"),
            corpus_classification=_normalize_text(raw["corpus_classification"], field="corpus_classification"),
        )

    def _check(
        self,
        *,
        name: str,
        category: str,
        required: bool,
        passed: bool,
        decision_effect: str,
        observed_value: Any,
        policy_value: Any,
        explanation: str,
    ) -> BenchmarkVaultCheck:
        observed_snapshot = _freeze_snapshot(_canonicalize(observed_value, field="observed_value"))
        policy_snapshot = _freeze_snapshot(_canonicalize(policy_value, field="policy_value"))
        return BenchmarkVaultCheck(
            name=name,
            category=category,
            required=required,
            passed=passed,
            decision_effect=decision_effect,
            observed_value=observed_snapshot,
            policy_value=policy_snapshot,
            explanation=explanation,
        )

    def evaluate(self, artifact: Any, *, intake_firewall_context: Any = None) -> Tuple[BenchmarkVaultReport, BenchmarkVaultReceipt]:
        checks: list[BenchmarkVaultCheck] = []
        try:
            normalized = self.build_artifact(artifact)
            envelope_ok = True
            envelope_error = ""
        except Exception as exc:
            normalized = BenchmarkArtifact(
                benchmark_id="unknown",
                benchmark_type="unknown",
                manifest={},
                metadata={},
                provenance={},
                declared_contract="unknown",
                source_channel="unknown",
                corpus_classification="unknown",
            )
            envelope_ok = False
            envelope_error = str(exc)

        checks.append(
            self._check(
                name="benchmark_envelope_structure",
                category="benchmark_identity",
                required=True,
                passed=envelope_ok,
                decision_effect=DECISION_REJECT,
                observed_value="valid" if envelope_ok else envelope_error,
                policy_value="required_fields_and_types",
                explanation="Benchmark artifact envelope must satisfy required deterministic schema.",
            )
        )

        provenance_complete = False
        manifest_complete = False
        sealed_corpus = False
        public_corpus = False

        if envelope_ok:
            type_valid = normalized.benchmark_type in self.policy.allowed_benchmark_types
            checks.append(self._check(name="benchmark_type_allowed", category="benchmark_identity", required=True, passed=type_valid, decision_effect=DECISION_REJECT, observed_value=normalized.benchmark_type, policy_value=list(self.policy.allowed_benchmark_types), explanation="Benchmark type must be explicitly allowed."))

            effective_required_manifest_fields = tuple(
                sorted(
                    set(field for field in self.policy.required_manifest_fields if field not in ("benchmark_id", "manifest_hash", "lineage_hash"))
                    .union({"benchmark_id"})
                    .union({"manifest_hash"} if self.policy.require_manifest_hash else set())
                    .union({"lineage_hash"} if self.policy.require_lineage_hash else set())
                )
            )
            missing_manifest_fields = tuple(sorted(field for field in effective_required_manifest_fields if field not in normalized.manifest))
            manifest_complete = len(missing_manifest_fields) == 0
            checks.append(self._check(name="required_manifest_fields", category="manifest_integrity", required=True, passed=manifest_complete, decision_effect=DECISION_REJECT, observed_value=list(missing_manifest_fields), policy_value=list(effective_required_manifest_fields), explanation="Manifest required fields must be present under active policy requirements."))

            manifest_hash_present = isinstance(normalized.manifest.get("manifest_hash"), str) and bool(str(normalized.manifest.get("manifest_hash")).strip())
            checks.append(self._check(name="manifest_hash_presence", category="manifest_integrity", required=self.policy.require_manifest_hash, passed=(manifest_hash_present or (not self.policy.require_manifest_hash)), decision_effect=DECISION_REJECT, observed_value=manifest_hash_present, policy_value=self.policy.require_manifest_hash, explanation="Manifest hash presence is policy-controlled."))

            lineage_hash_present = isinstance(normalized.manifest.get("lineage_hash"), str) and bool(str(normalized.manifest.get("lineage_hash")).strip())
            lineage_required = self.policy.require_lineage_hash
            checks.append(self._check(name="lineage_hash_presence", category="manifest_integrity", required=lineage_required, passed=(lineage_hash_present or (not lineage_required)), decision_effect=DECISION_REJECT, observed_value=lineage_hash_present, policy_value=lineage_required, explanation="Lineage hash presence is policy-controlled."))

            manifest_paths = _iter_mapping_paths(normalized.manifest)
            forbidden_manifest_hits = tuple(sorted(path for path, key in manifest_paths if key in self.policy.forbidden_manifest_field_names))
            checks.append(self._check(name="forbidden_manifest_field_names", category="manifest_integrity", required=True, passed=len(forbidden_manifest_hits) == 0, decision_effect=DECISION_REJECT, observed_value=list(forbidden_manifest_hits), policy_value=list(self.policy.forbidden_manifest_field_names), explanation="Forbidden manifest fields are rejected."))

            metadata_paths = _iter_mapping_paths(normalized.metadata)
            forbidden_metadata_hits = tuple(sorted(path for path, key in metadata_paths if key in self.policy.forbidden_metadata_keys))
            checks.append(self._check(name="forbidden_metadata_keys", category="metadata_policy", required=True, passed=len(forbidden_metadata_hits) == 0, decision_effect=DECISION_REJECT, observed_value=list(forbidden_metadata_hits), policy_value=list(self.policy.forbidden_metadata_keys), explanation="Forbidden metadata keys are rejected."))

            missing_provenance = tuple(sorted(field for field in self.policy.required_provenance_fields if field not in normalized.provenance))
            provenance_complete = len(missing_provenance) == 0
            provenance_effect = DECISION_QUARANTINE if self.policy.quarantine_incomplete_provenance else DECISION_REJECT
            checks.append(self._check(name="required_provenance_fields", category="provenance_integrity", required=True, passed=provenance_complete, decision_effect=provenance_effect, observed_value=list(missing_provenance), policy_value=list(self.policy.required_provenance_fields), explanation="Provenance must be complete."))

            contract_valid = normalized.declared_contract in self.policy.allowed_declared_contracts
            checks.append(self._check(name="declared_contract_allowed", category="contract_compliance", required=True, passed=contract_valid, decision_effect=DECISION_REJECT, observed_value=normalized.declared_contract, policy_value=list(self.policy.allowed_declared_contracts), explanation="Declared benchmark contract must be allowed."))

            source_valid = normalized.source_channel in self.policy.allowed_source_channels
            checks.append(self._check(name="source_channel_allowed", category="source_admissibility", required=True, passed=source_valid, decision_effect=DECISION_REJECT, observed_value=normalized.source_channel, policy_value=list(self.policy.allowed_source_channels), explanation="Source channel must be allowed."))

            classification_valid = normalized.corpus_classification in self.policy.allowed_corpus_classifications
            checks.append(self._check(name="corpus_classification_allowed", category="corpus_classification", required=True, passed=classification_valid, decision_effect=DECISION_REJECT, observed_value=normalized.corpus_classification, policy_value=list(self.policy.allowed_corpus_classifications), explanation="Corpus classification must be explicitly allowed."))

            tokens = set(_classification_tokens(normalized.corpus_classification))
            sealed_corpus = normalized.corpus_classification in self.policy.sealed_classifications
            public_corpus = normalized.corpus_classification in self.policy.public_classifications
            token_mixing = bool(tokens.intersection(set(self.policy.sealed_classifications))) and bool(tokens.intersection(set(self.policy.public_classifications)))
            classification_mixing = (sealed_corpus and public_corpus) or token_mixing
            mixing_passed = (not classification_mixing) or (not self.policy.reject_sealed_public_mixing)
            checks.append(self._check(name="sealed_public_mixing", category="sealed_public_separation", required=self.policy.reject_sealed_public_mixing, passed=mixing_passed, decision_effect=DECISION_REJECT, observed_value=normalized.corpus_classification, policy_value=self.policy.reject_sealed_public_mixing, explanation="Sealed/public mixing is disallowed by policy."))

            manifest_identity_conflicts = []
            manifest_benchmark_id = normalized.manifest.get("benchmark_id")
            if isinstance(manifest_benchmark_id, str) and manifest_benchmark_id.strip() and manifest_benchmark_id != normalized.benchmark_id:
                manifest_identity_conflicts.append("benchmark_id_conflict")
            manifest_classification = normalized.manifest.get("corpus_classification")
            if isinstance(manifest_classification, str) and manifest_classification.strip() and manifest_classification != normalized.corpus_classification:
                manifest_identity_conflicts.append("corpus_classification_conflict")
            identity_passed = (len(manifest_identity_conflicts) == 0) or (not self.policy.reject_manifest_identity_conflicts)
            checks.append(self._check(name="manifest_identity_conflicts", category="benchmark_identity", required=self.policy.reject_manifest_identity_conflicts, passed=identity_passed, decision_effect=DECISION_REJECT, observed_value=manifest_identity_conflicts, policy_value=self.policy.reject_manifest_identity_conflicts, explanation="Manifest identity conflicts are rejected by policy."))

            override_hits = tuple(
                sorted(
                    path
                    for path, key in (metadata_paths + manifest_paths)
                    if "override" in key.lower()
                )
            )
            checks.append(self._check(name="undeclared_override_fields", category="poisoning_resistance", required=True, passed=len(override_hits) == 0, decision_effect=DECISION_REJECT, observed_value=list(override_hits), policy_value="no_override_fields", explanation="Override fields are rejected to prevent undeclared benchmark mutation."))

            intake_decision = ""
            if isinstance(intake_firewall_context, Mapping):
                maybe_decision = intake_firewall_context.get("decision")
                if isinstance(maybe_decision, str):
                    intake_decision = maybe_decision.strip().lower()
            intake_ok = intake_decision in ("", DECISION_ALLOW, DECISION_WARN)
            checks.append(self._check(name="intake_firewall_compatibility", category="poisoning_resistance", required=False, passed=intake_ok, decision_effect=DECISION_WARN, observed_value=intake_decision if intake_decision else "unspecified", policy_value=[DECISION_ALLOW, DECISION_WARN], explanation="Intake firewall context should indicate an admissible intake lane."))

            if sealed_corpus:
                sealed_lanes = tuple(sorted(path for path, key in metadata_paths if key == "public_lane"))
                checks.append(self._check(name="sealed_forbids_public_lane_metadata", category="sealed_public_separation", required=True, passed=len(sealed_lanes) == 0, decision_effect=DECISION_REJECT, observed_value=list(sealed_lanes), policy_value="metadata.public_lane_forbidden_for_sealed", explanation="Sealed corpora cannot include public-only lane metadata."))
            elif public_corpus:
                sealed_lanes = tuple(sorted(path for path, key in metadata_paths if key == "sealed_lane"))
                checks.append(self._check(name="public_forbids_sealed_lane_metadata", category="sealed_public_separation", required=True, passed=len(sealed_lanes) == 0, decision_effect=DECISION_REJECT, observed_value=list(sealed_lanes), policy_value="metadata.sealed_lane_forbidden_for_public", explanation="Public corpora cannot include sealed-only lane metadata."))

        checks = sorted(checks, key=lambda c: (CHECK_CATEGORY_ORDER.index(c.category), c.name))

        failing_required = tuple(check.name for check in checks if check.required and not check.passed)
        warnings = tuple(check.name for check in checks if (not check.required) and (not check.passed) and check.decision_effect == DECISION_WARN)
        quarantine_reasons = tuple(check.name for check in checks if (not check.passed) and check.decision_effect == DECISION_QUARANTINE)
        rejection_reasons = tuple(check.name for check in checks if (not check.passed) and check.decision_effect == DECISION_REJECT)

        quarantined = len(quarantine_reasons) > 0 and len(rejection_reasons) == 0
        rejected = len(rejection_reasons) > 0
        if rejected:
            decision = DECISION_REJECT
        elif quarantined:
            decision = DECISION_QUARANTINE
        elif warnings:
            decision = DECISION_WARN
        else:
            decision = DECISION_ALLOW

        counts_by_status = {
            "passed": sum(1 for check in checks if check.passed),
            "failed": sum(1 for check in checks if check.required and (not check.passed)),
            "advisory_failed": sum(1 for check in checks if (not check.required) and (not check.passed)),
            "total": len(checks),
        }

        report = BenchmarkVaultReport(
            benchmark_id=normalized.benchmark_id,
            checks=tuple(checks),
            counts_by_status=counts_by_status,
            failing_required_checks=failing_required,
            warnings=warnings,
            quarantine_reasons=quarantine_reasons,
            rejection_reasons=rejection_reasons,
            decision=decision,
            vault_admissible=decision in (DECISION_ALLOW, DECISION_WARN),
            quarantined=quarantined,
            rejected=rejected,
            sealed_corpus=sealed_corpus,
            public_corpus=public_corpus,
            provenance_complete=provenance_complete,
            manifest_complete=manifest_complete,
            policy_version=self.policy.policy_version,
        )

        rationale = tuple(report.rejection_reasons + report.quarantine_reasons + report.warnings)
        receipt = BenchmarkVaultReceipt(
            benchmark_hash=normalized.stable_hash(),
            policy_hash=self.policy.stable_hash(),
            report_hash=report.stable_hash(),
            decision=report.decision,
            vault_admissible=report.vault_admissible,
            quarantined=report.quarantined,
            rejected=report.rejected,
            sealed_corpus=report.sealed_corpus,
            public_corpus=report.public_corpus,
            rationale=rationale,
        )
        return report, receipt



def run_benchmark_vault(
    *,
    benchmark_artifact: Any,
    benchmark_vault_policy: Any = None,
    provenance_metadata: Any = None,
    corpus_classification: Any = None,
    intake_firewall_context: Any = None,
) -> Tuple[BenchmarkVaultReport, BenchmarkVaultReceipt]:
    policy = BenchmarkVaultKernel.default_policy() if benchmark_vault_policy is None else BenchmarkVaultKernel.build_policy(benchmark_vault_policy)
    merged: Dict[str, Any]
    if isinstance(benchmark_artifact, Mapping):
        merged = dict(benchmark_artifact)
    else:
        merged = {"benchmark_id": "unknown"}

    if provenance_metadata is not None:
        merged["provenance"] = provenance_metadata
    if corpus_classification is not None:
        merged["corpus_classification"] = corpus_classification

    kernel = BenchmarkVaultKernel(policy=policy)
    return kernel.evaluate(merged, intake_firewall_context=intake_firewall_context)



def validate_benchmark_artifact(benchmark_artifact: Any) -> Dict[str, Any]:
    try:
        BenchmarkVaultKernel.build_artifact(benchmark_artifact)
    except Exception as exc:
        return {"valid": False, "violations": (str(exc),)}
    return {"valid": True, "violations": ()}



def summarize_benchmark_vault_report(report: BenchmarkVaultReport) -> Dict[str, Any]:
    return {
        "benchmark_id": report.benchmark_id,
        "decision": report.decision,
        "vault_admissible": report.vault_admissible,
        "quarantined": report.quarantined,
        "rejected": report.rejected,
        "sealed_corpus": report.sealed_corpus,
        "public_corpus": report.public_corpus,
        "provenance_complete": report.provenance_complete,
        "manifest_complete": report.manifest_complete,
        "failing_required_checks": list(report.failing_required_checks),
        "warnings": list(report.warnings),
    }
