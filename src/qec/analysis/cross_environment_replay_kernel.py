"""v148.6 — Cross-Environment Replay Kernel.

Deterministic analysis-layer comparator for replay artifacts emitted by the
same workload across multiple environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

SCHEMA_VERSION = "1"
MODULE_VERSION = "v148.6"

_VALID_SHA256_CHARS = frozenset("0123456789abcdef")


def _validate_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _validate_sha256_hex(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _VALID_SHA256_CHARS for ch in value):
        raise ValueError(f"{field_name} must be a valid SHA-256 hex string")
    return value


@dataclass(frozen=True)
class EnvironmentReplayArtifact:
    environment_id: str
    workload_id: str
    artifact_hash: str
    canonical_payload_hash: str
    receipt_hash: str
    platform_label: str
    python_label: str
    metadata_hash: str
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.environment_id, field_name="environment_id")
        _validate_non_empty_string(self.workload_id, field_name="workload_id")
        _validate_sha256_hex(self.artifact_hash, field_name="artifact_hash")
        _validate_sha256_hex(self.canonical_payload_hash, field_name="canonical_payload_hash")
        _validate_sha256_hex(self.receipt_hash, field_name="receipt_hash")
        _validate_string(self.platform_label, field_name="platform_label")
        _validate_string(self.python_label, field_name="python_label")
        _validate_sha256_hex(self.metadata_hash, field_name="metadata_hash")
        object.__setattr__(self, "stable_hash", sha256_hex(self._hash_payload()))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "workload_id": self.workload_id,
            "artifact_hash": self.artifact_hash,
            "canonical_payload_hash": self.canonical_payload_hash,
            "receipt_hash": self.receipt_hash,
            "platform_label": self.platform_label,
            "python_label": self.python_label,
            "metadata_hash": self.metadata_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class EnvironmentReplayComparison:
    workload_id: str
    comparison_status: str
    reference_environment_id: str
    matching_environment_ids: tuple[str, ...]
    mismatching_environment_ids: tuple[str, ...]
    mismatch_classification: str
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.workload_id, field_name="workload_id")
        _validate_non_empty_string(self.comparison_status, field_name="comparison_status")
        _validate_non_empty_string(self.reference_environment_id, field_name="reference_environment_id")
        if not isinstance(self.matching_environment_ids, tuple) or any(
            not isinstance(v, str) or v == "" for v in self.matching_environment_ids
        ):
            raise ValueError("matching_environment_ids must be a tuple of non-empty strings")
        if not isinstance(self.mismatching_environment_ids, tuple) or any(
            not isinstance(v, str) or v == "" for v in self.mismatching_environment_ids
        ):
            raise ValueError("mismatching_environment_ids must be a tuple of non-empty strings")
        _validate_non_empty_string(self.mismatch_classification, field_name="mismatch_classification")
        object.__setattr__(self, "stable_hash", sha256_hex(self._hash_payload()))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "comparison_status": self.comparison_status,
            "reference_environment_id": self.reference_environment_id,
            "matching_environment_ids": self.matching_environment_ids,
            "mismatching_environment_ids": self.mismatching_environment_ids,
            "mismatch_classification": self.mismatch_classification,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


@dataclass(frozen=True)
class CrossEnvironmentReplayReceipt:
    schema_version: str
    module_version: str
    receipt_status: str
    workload_id: str
    environment_count: int
    comparison: EnvironmentReplayComparison
    determinism_preserved: bool
    failure_recorded: bool
    failure_classified: bool
    stable_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version, field_name="schema_version")
        _validate_non_empty_string(self.module_version, field_name="module_version")
        _validate_non_empty_string(self.receipt_status, field_name="receipt_status")
        _validate_non_empty_string(self.workload_id, field_name="workload_id")
        if not isinstance(self.environment_count, int) or self.environment_count < 0:
            raise ValueError("environment_count must be a non-negative integer")
        if not isinstance(self.comparison, EnvironmentReplayComparison):
            raise ValueError("comparison must be an EnvironmentReplayComparison")
        if not isinstance(self.determinism_preserved, bool):
            raise ValueError("determinism_preserved must be a bool")
        if not isinstance(self.failure_recorded, bool):
            raise ValueError("failure_recorded must be a bool")
        if not isinstance(self.failure_classified, bool):
            raise ValueError("failure_classified must be a bool")
        object.__setattr__(self, "stable_hash", sha256_hex(self._hash_payload()))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "receipt_status": self.receipt_status,
            "workload_id": self.workload_id,
            "environment_count": self.environment_count,
            "comparison": self.comparison.to_dict(),
            "determinism_preserved": self.determinism_preserved,
            "failure_recorded": self.failure_recorded,
            "failure_classified": self.failure_classified,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["stable_hash"] = self.stable_hash
        return payload

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def compare_cross_environment_replay(
    artifacts: Sequence[EnvironmentReplayArtifact],
) -> CrossEnvironmentReplayReceipt:
    if not isinstance(artifacts, Sequence):
        raise ValueError("artifacts must be a sequence of EnvironmentReplayArtifact")

    normalized = tuple(artifacts)
    if any(not isinstance(artifact, EnvironmentReplayArtifact) for artifact in normalized):
        raise ValueError("artifacts must contain only EnvironmentReplayArtifact entries")

    if len(normalized) == 0:
        raise ValueError("artifacts must not be empty")

    workload_ids = tuple(artifact.workload_id for artifact in normalized)
    workload_id = workload_ids[0]

    if len(normalized) < 2:
        comparison = EnvironmentReplayComparison(
            workload_id=workload_id,
            comparison_status="INSUFFICIENT",
            reference_environment_id=normalized[0].environment_id,
            matching_environment_ids=(normalized[0].environment_id,),
            mismatching_environment_ids=(),
            mismatch_classification="INSUFFICIENT_ENVIRONMENTS",
        )
        return CrossEnvironmentReplayReceipt(
            schema_version=SCHEMA_VERSION,
            module_version=MODULE_VERSION,
            receipt_status="INSUFFICIENT_ENVIRONMENTS",
            workload_id=workload_id,
            environment_count=len(normalized),
            comparison=comparison,
            determinism_preserved=False,
            failure_recorded=True,
            failure_classified=True,
        )

    if len(set(workload_ids)) != 1:
        sorted_artifacts = tuple(sorted(normalized, key=lambda item: item.environment_id))
        reference_environment = sorted_artifacts[0].environment_id
        comparison = EnvironmentReplayComparison(
            workload_id=workload_id,
            comparison_status="MISMATCH",
            reference_environment_id=reference_environment,
            matching_environment_ids=(),
            mismatching_environment_ids=tuple(a.environment_id for a in sorted_artifacts),
            mismatch_classification="WORKLOAD_ID_MISMATCH",
        )
        return CrossEnvironmentReplayReceipt(
            schema_version=SCHEMA_VERSION,
            module_version=MODULE_VERSION,
            receipt_status="CROSS_ENV_MISMATCH",
            workload_id=workload_id,
            environment_count=len(normalized),
            comparison=comparison,
            determinism_preserved=False,
            failure_recorded=True,
            failure_classified=True,
        )

    sorted_artifacts = tuple(sorted(normalized, key=lambda item: item.environment_id))
    reference = sorted_artifacts[0]
    matching_ids: list[str] = [reference.environment_id]
    mismatching_ids: list[str] = []

    artifact_mismatch = False
    payload_mismatch = False
    receipt_mismatch = False

    for candidate in sorted_artifacts[1:]:
        candidate_mismatch = False
        if candidate.artifact_hash != reference.artifact_hash:
            artifact_mismatch = True
            candidate_mismatch = True
        if candidate.canonical_payload_hash != reference.canonical_payload_hash:
            payload_mismatch = True
            candidate_mismatch = True
        if candidate.receipt_hash != reference.receipt_hash:
            receipt_mismatch = True
            candidate_mismatch = True

        if candidate_mismatch:
            mismatching_ids.append(candidate.environment_id)
        else:
            matching_ids.append(candidate.environment_id)

    mismatch_categories = sum((artifact_mismatch, payload_mismatch, receipt_mismatch))
    if mismatch_categories == 0:
        mismatch_classification = "NONE"
        comparison_status = "MATCH"
        receipt_status = "CROSS_ENV_MATCH"
        determinism_preserved = True
        failure_recorded = False
    elif mismatch_categories == 1:
        if artifact_mismatch:
            mismatch_classification = "ARTIFACT_HASH_MISMATCH"
        elif payload_mismatch:
            mismatch_classification = "CANONICAL_PAYLOAD_MISMATCH"
        else:
            mismatch_classification = "RECEIPT_HASH_MISMATCH"
        comparison_status = "MISMATCH"
        receipt_status = "CROSS_ENV_MISMATCH"
        determinism_preserved = False
        failure_recorded = True
    else:
        mismatch_classification = "MIXED_HASH_MISMATCH"
        comparison_status = "MISMATCH"
        receipt_status = "CROSS_ENV_MISMATCH"
        determinism_preserved = False
        failure_recorded = True

    comparison = EnvironmentReplayComparison(
        workload_id=workload_id,
        comparison_status=comparison_status,
        reference_environment_id=reference.environment_id,
        matching_environment_ids=tuple(sorted(matching_ids)),
        mismatching_environment_ids=tuple(sorted(mismatching_ids)),
        mismatch_classification=mismatch_classification,
    )

    return CrossEnvironmentReplayReceipt(
        schema_version=SCHEMA_VERSION,
        module_version=MODULE_VERSION,
        receipt_status=receipt_status,
        workload_id=workload_id,
        environment_count=len(sorted_artifacts),
        comparison=comparison,
        determinism_preserved=determinism_preserved,
        failure_recorded=failure_recorded,
        failure_classified=True,
    )
