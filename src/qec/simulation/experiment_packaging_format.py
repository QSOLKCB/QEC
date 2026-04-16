# SPDX-License-Identifier: MIT
"""v138.1.1 — Deterministic experiment packaging format.

Canonical, replay-safe package schema for correlated-noise simulation artifacts.
This module is pure data-model + validation + deterministic hashing; it does
not perform filesystem archive operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

EXPERIMENT_PACKAGING_FORMAT_VERSION = "v138.1.1"

_HEX_64 = frozenset("0123456789abcdef")


class ExperimentPackageValidationError(ValueError):
    """Raised when experiment package content violates deterministic schema."""



def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)



def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()



def _is_hex_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in _HEX_64 for ch in value)



def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ExperimentPackageValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise ExperimentPackageValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ExperimentPackageValidationError(f"{field} contains unsupported type: {type(value).__name__}")



def _normalize_text(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ExperimentPackageValidationError(f"{field} must be non-empty")
    return text



def _normalize_hash(value: Any, *, field: str) -> str:
    hash_value = _normalize_text(value, field=field).lower()
    if not _is_hex_sha256(hash_value):
        raise ExperimentPackageValidationError(f"{field} must be 64-char lowercase SHA-256 hex")
    return hash_value



def _normalize_hash_tuple(values: Sequence[str], *, field: str) -> Tuple[str, ...]:
    normalized = tuple(sorted(_normalize_hash(item, field=field) for item in values))
    return normalized



def _artifact_sort_key(artifact: "ExperimentPackageArtifact") -> Tuple[Any, ...]:
    return (
        artifact.artifact_role,
        artifact.artifact_kind,
        artifact.serialization_format,
        artifact.artifact_hash,
        artifact.lineage_hash or "",
        artifact.content_bytes if artifact.content_bytes is not None else -1,
        _stable_hash(artifact.metadata),
    )



def _package_hash_payload(*, manifest_hash: str, artifact_set_hash: str, upstream_receipt_hashes: Tuple[str, ...], package_version: str) -> Dict[str, Any]:
    return {
        "manifest_hash": manifest_hash,
        "artifact_set_hash": artifact_set_hash,
        "upstream_receipt_hashes": list(upstream_receipt_hashes),
        "package_version": package_version,
    }


@dataclass(frozen=True)
class ExperimentPackageArtifact:
    artifact_role: str
    artifact_hash: str
    artifact_kind: str
    serialization_format: str
    content_bytes: int | None
    lineage_hash: str | None
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_role": self.artifact_role,
            "artifact_hash": self.artifact_hash,
            "artifact_kind": self.artifact_kind,
            "serialization_format": self.serialization_format,
            "content_bytes": self.content_bytes,
            "lineage_hash": self.lineage_hash,
            "metadata": _canonicalize_value(dict(self.metadata), field="artifact.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ExperimentPackageManifest:
    format_version: str
    package_kind: str
    experiment_id: str
    simulator_release: str
    simulator_module: str
    scenario_hash: str
    realization_hashes: Tuple[str, ...]
    topology_family: str
    code_family: str
    seed: int | str
    parameter_hash: str
    policy_flags: Tuple[str, ...]
    benchmark_id: str | None
    manifest_lineage_hash: str | None
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.format_version,
            "package_kind": self.package_kind,
            "experiment_id": self.experiment_id,
            "simulator_release": self.simulator_release,
            "simulator_module": self.simulator_module,
            "scenario_hash": self.scenario_hash,
            "realization_hashes": list(self.realization_hashes),
            "topology_family": self.topology_family,
            "code_family": self.code_family,
            "seed": self.seed,
            "parameter_hash": self.parameter_hash,
            "policy_flags": list(self.policy_flags),
            "benchmark_id": self.benchmark_id,
            "manifest_lineage_hash": self.manifest_lineage_hash,
            "notes": list(self.notes),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ExperimentPackageReceipt:
    manifest_hash: str
    artifact_set_hash: str
    upstream_receipt_hashes: Tuple[str, ...]
    package_hash: str
    validation_passed: bool
    validation_error_count: int
    package_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_hash": self.manifest_hash,
            "artifact_set_hash": self.artifact_set_hash,
            "upstream_receipt_hashes": list(self.upstream_receipt_hashes),
            "package_hash": self.package_hash,
            "validation_passed": self.validation_passed,
            "validation_error_count": self.validation_error_count,
            "package_version": self.package_version,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return _package_hash_payload(
            manifest_hash=self.manifest_hash,
            artifact_set_hash=self.artifact_set_hash,
            upstream_receipt_hashes=self.upstream_receipt_hashes,
            package_version=self.package_version,
        )

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class ExperimentPackage:
    manifest: ExperimentPackageManifest
    artifacts: Tuple[ExperimentPackageArtifact, ...]
    receipt: ExperimentPackageReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "receipt": self.receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ExperimentPackageValidationReport:
    valid: bool
    errors: Tuple[str, ...]

    @property
    def error_count(self) -> int:
        return len(self.errors)

    def to_dict(self) -> Dict[str, Any]:
        return {"valid": self.valid, "error_count": self.error_count, "errors": list(self.errors)}



def _manifest_from_any(raw: ExperimentPackageManifest | Mapping[str, Any]) -> ExperimentPackageManifest:
    if isinstance(raw, ExperimentPackageManifest):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise ExperimentPackageValidationError("manifest must be mapping or ExperimentPackageManifest")

    benchmark_id = raw.get("benchmark_id")
    lineage_hash = raw.get("manifest_lineage_hash")
    seed = raw["seed"]
    if not isinstance(seed, (str, int)):
        raise ExperimentPackageValidationError("manifest.seed must be int or str")

    return ExperimentPackageManifest(
        format_version=_normalize_text(raw.get("format_version", EXPERIMENT_PACKAGING_FORMAT_VERSION), field="manifest.format_version"),
        package_kind=_normalize_text(raw["package_kind"], field="manifest.package_kind"),
        experiment_id=_normalize_text(raw["experiment_id"], field="manifest.experiment_id"),
        simulator_release=_normalize_text(raw["simulator_release"], field="manifest.simulator_release"),
        simulator_module=_normalize_text(raw["simulator_module"], field="manifest.simulator_module"),
        scenario_hash=_normalize_hash(raw["scenario_hash"], field="manifest.scenario_hash"),
        realization_hashes=_normalize_hash_tuple(tuple(raw.get("realization_hashes", ())), field="manifest.realization_hashes"),
        topology_family=_normalize_text(raw["topology_family"], field="manifest.topology_family"),
        code_family=_normalize_text(raw["code_family"], field="manifest.code_family"),
        seed=str(seed) if isinstance(seed, str) else int(seed),
        parameter_hash=_normalize_hash(raw["parameter_hash"], field="manifest.parameter_hash"),
        policy_flags=tuple(sorted(_normalize_text(flag, field="manifest.policy_flags") for flag in tuple(raw.get("policy_flags", ())))),
        benchmark_id=None if benchmark_id is None else _normalize_text(benchmark_id, field="manifest.benchmark_id"),
        manifest_lineage_hash=None if lineage_hash is None else _normalize_hash(lineage_hash, field="manifest.manifest_lineage_hash"),
        notes=tuple(sorted(_normalize_text(note, field="manifest.notes") for note in tuple(raw.get("notes", ())))),
    )



def _artifact_from_any(raw: ExperimentPackageArtifact | Mapping[str, Any]) -> ExperimentPackageArtifact:
    if isinstance(raw, ExperimentPackageArtifact):
        metadata = _canonicalize_value(dict(raw.metadata), field="artifact.metadata")
        return ExperimentPackageArtifact(
            artifact_role=_normalize_text(raw.artifact_role, field="artifact.artifact_role").lower(),
            artifact_hash=_normalize_hash(raw.artifact_hash, field="artifact.artifact_hash"),
            artifact_kind=_normalize_text(raw.artifact_kind, field="artifact.artifact_kind"),
            serialization_format=_normalize_text(raw.serialization_format, field="artifact.serialization_format"),
            content_bytes=None if raw.content_bytes is None else int(raw.content_bytes),
            lineage_hash=None if raw.lineage_hash is None else _normalize_hash(raw.lineage_hash, field="artifact.lineage_hash"),
            metadata=metadata,
        )
    if not isinstance(raw, Mapping):
        raise ExperimentPackageValidationError("artifact must be mapping or ExperimentPackageArtifact")

    content_bytes = raw.get("content_bytes")
    if content_bytes is not None and int(content_bytes) < 0:
        raise ExperimentPackageValidationError("artifact.content_bytes must be >= 0")
    lineage_hash = raw.get("lineage_hash")
    metadata = _canonicalize_value(dict(raw.get("metadata", {})), field="artifact.metadata")

    return ExperimentPackageArtifact(
        artifact_role=_normalize_text(raw["artifact_role"], field="artifact.artifact_role").lower(),
        artifact_hash=_normalize_hash(raw["artifact_hash"], field="artifact.artifact_hash"),
        artifact_kind=_normalize_text(raw["artifact_kind"], field="artifact.artifact_kind"),
        serialization_format=_normalize_text(raw["serialization_format"], field="artifact.serialization_format"),
        content_bytes=None if content_bytes is None else int(content_bytes),
        lineage_hash=None if lineage_hash is None else _normalize_hash(lineage_hash, field="artifact.lineage_hash"),
        metadata=metadata,
    )



def build_experiment_package(
    *,
    manifest: ExperimentPackageManifest | Mapping[str, Any],
    artifacts: Sequence[ExperimentPackageArtifact | Mapping[str, Any]],
    upstream_receipt_hashes: Sequence[str] = (),
    package_version: str = EXPERIMENT_PACKAGING_FORMAT_VERSION,
) -> ExperimentPackage:
    """Build canonical deterministic experiment package + validated receipt."""
    normalized_manifest = _manifest_from_any(manifest)
    normalized_artifacts = tuple(sorted((_artifact_from_any(artifact) for artifact in artifacts), key=_artifact_sort_key))
    normalized_upstream = _normalize_hash_tuple(tuple(upstream_receipt_hashes), field="receipt.upstream_receipt_hashes")

    manifest_hash = normalized_manifest.stable_hash()
    artifact_set_hash = _stable_hash([artifact.to_dict() for artifact in normalized_artifacts])
    package_hash = _stable_hash(
        _package_hash_payload(
            manifest_hash=manifest_hash,
            artifact_set_hash=artifact_set_hash,
            upstream_receipt_hashes=normalized_upstream,
            package_version=package_version,
        )
    )

    provisional = ExperimentPackage(
        manifest=normalized_manifest,
        artifacts=normalized_artifacts,
        receipt=ExperimentPackageReceipt(
            manifest_hash=manifest_hash,
            artifact_set_hash=artifact_set_hash,
            upstream_receipt_hashes=normalized_upstream,
            package_hash=package_hash,
            validation_passed=False,
            validation_error_count=0,
            package_version=package_version,
        ),
    )
    report = validate_experiment_package(provisional)
    if not report.valid:
        raise ExperimentPackageValidationError("; ".join(report.errors))

    return ExperimentPackage(
        manifest=normalized_manifest,
        artifacts=normalized_artifacts,
        receipt=ExperimentPackageReceipt(
            manifest_hash=manifest_hash,
            artifact_set_hash=artifact_set_hash,
            upstream_receipt_hashes=normalized_upstream,
            package_hash=package_hash,
            validation_passed=True,
            validation_error_count=0,
            package_version=package_version,
        ),
    )



def validate_experiment_package(package: ExperimentPackage) -> ExperimentPackageValidationReport:
    """Validate canonical package schema and hash consistency."""
    errors = []

    # Manifest field invariants.
    if not package.manifest.format_version.strip():
        errors.append("manifest.format_version must be non-empty")
    if not package.manifest.package_kind.strip():
        errors.append("manifest.package_kind must be non-empty")
    if not package.manifest.experiment_id.strip():
        errors.append("manifest.experiment_id must be non-empty")
    if not package.manifest.simulator_release.strip():
        errors.append("manifest.simulator_release must be non-empty")
    if not package.manifest.simulator_module.strip():
        errors.append("manifest.simulator_module must be non-empty")
    if not package.manifest.topology_family.strip():
        errors.append("manifest.topology_family must be non-empty")
    if not package.manifest.code_family.strip():
        errors.append("manifest.code_family must be non-empty")

    for field_name, hash_value in (
        ("manifest.scenario_hash", package.manifest.scenario_hash),
        ("manifest.parameter_hash", package.manifest.parameter_hash),
    ):
        if not _is_hex_sha256(hash_value):
            errors.append(f"{field_name} must be 64-char lowercase SHA-256 hex")

    for index, value in enumerate(package.manifest.realization_hashes):
        if not _is_hex_sha256(value):
            errors.append(f"manifest.realization_hashes[{index}] must be 64-char lowercase SHA-256 hex")

    # Artifact invariants + deterministic ordering.
    seen_roles = set()
    canonical_roles = []
    for idx, artifact in enumerate(package.artifacts):
        canonical_role = artifact.artifact_role.strip().lower()
        canonical_roles.append(canonical_role)
        if canonical_role in seen_roles:
            errors.append(f"duplicate artifact role after canonical normalization: {canonical_role!r}")
        seen_roles.add(canonical_role)

        if not _is_hex_sha256(artifact.artifact_hash):
            errors.append(f"artifacts[{idx}].artifact_hash must be 64-char lowercase SHA-256 hex")
        if artifact.lineage_hash is not None and not _is_hex_sha256(artifact.lineage_hash):
            errors.append(f"artifacts[{idx}].lineage_hash must be 64-char lowercase SHA-256 hex")
        try:
            _canonicalize_value(dict(artifact.metadata), field=f"artifacts[{idx}].metadata")
        except ExperimentPackageValidationError as exc:
            errors.append(str(exc))

    sorted_artifacts = tuple(sorted(package.artifacts, key=_artifact_sort_key))
    if package.artifacts != sorted_artifacts:
        errors.append("artifacts must be pre-sorted in canonical deterministic order")

    sorted_upstream = tuple(sorted(package.receipt.upstream_receipt_hashes))
    if package.receipt.upstream_receipt_hashes != sorted_upstream:
        errors.append("receipt.upstream_receipt_hashes must be pre-sorted")

    for idx, upstream_hash in enumerate(package.receipt.upstream_receipt_hashes):
        if not _is_hex_sha256(upstream_hash):
            errors.append(f"receipt.upstream_receipt_hashes[{idx}] must be 64-char lowercase SHA-256 hex")

    expected_manifest_hash = package.manifest.stable_hash()
    if package.receipt.manifest_hash != expected_manifest_hash:
        errors.append("receipt.manifest_hash mismatch")

    expected_artifact_set_hash = _stable_hash([artifact.to_dict() for artifact in package.artifacts])
    if package.receipt.artifact_set_hash != expected_artifact_set_hash:
        errors.append("receipt.artifact_set_hash mismatch")

    expected_package_hash = _stable_hash(package.receipt.to_hash_payload_dict())
    if package.receipt.package_hash != expected_package_hash:
        errors.append("receipt.package_hash mismatch")

    ordered_errors = tuple(sorted(errors))
    return ExperimentPackageValidationReport(valid=not ordered_errors, errors=ordered_errors)



def package_replay_identity(package: ExperimentPackage) -> Dict[str, Any]:
    """Expose deterministic replay identity for v138.1.2 orchestration handoff."""
    return {
        "package_hash": package.receipt.package_hash,
        "manifest_hash": package.receipt.manifest_hash,
        "artifact_set_hash": package.receipt.artifact_set_hash,
        "scenario_hash": package.manifest.scenario_hash,
        "realization_hashes": package.manifest.realization_hashes,
        "upstream_receipt_hashes": package.receipt.upstream_receipt_hashes,
    }


__all__ = [
    "EXPERIMENT_PACKAGING_FORMAT_VERSION",
    "ExperimentPackageValidationError",
    "ExperimentPackageArtifact",
    "ExperimentPackageManifest",
    "ExperimentPackageReceipt",
    "ExperimentPackage",
    "ExperimentPackageValidationReport",
    "build_experiment_package",
    "validate_experiment_package",
    "package_replay_identity",
]
