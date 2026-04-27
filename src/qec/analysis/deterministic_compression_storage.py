"""v149.4 — Deterministic compression/storage descriptors (analysis layer only)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
import math
from typing import Final

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION: Final[str] = "v149.4"

_ALLOWED_RETENTION_CLASSES: Final[tuple[str, ...]] = ("EPHEMERAL", "SESSION", "RELEASE", "ARCHIVAL")
_ALLOWED_STORAGE_CLASSES: Final[tuple[str, ...]] = ("HOT", "WARM", "COLD", "ARCHIVAL")
_ALLOWED_PLAN_STATUSES: Final[tuple[str, ...]] = ("EMPTY", "COMPRESSED", "NO_GAIN", "INVALID")

_RETENTION_RANK: Final[dict[str, int]] = {
    "EPHEMERAL": 0,
    "SESSION": 1,
    "RELEASE": 2,
    "ARCHIVAL": 3,
}

_COMPRESSION_FACTOR_BY_RETENTION: Final[dict[str, float]] = {
    "ARCHIVAL": 0.55,
    "RELEASE": 0.65,
    "SESSION": 0.80,
    "EPHEMERAL": 1.00,
}

_STORAGE_CLASS_BY_RETENTION: Final[dict[str, str]] = {
    "ARCHIVAL": "ARCHIVAL",
    "RELEASE": "COLD",
    "SESSION": "WARM",
    "EPHEMERAL": "HOT",
}

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _round_public_metric(value: float) -> float:
    return float(round(float(value), 12))


def _require_canonical_token(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty canonical string")
    token = value.strip()
    if not token or token != value:
        raise ValueError(f"{name} must be a non-empty canonical string")
    return token


def _require_sha256_hex(value: str, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid SHA-256 hex") from exc
    if value != value.lower():
        raise ValueError(f"{name} must be a valid SHA-256 hex")
    return value


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_probability(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be bounded [0.0, 1.0]")
    return number


def _normalize_unique_sorted_tokens(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be tuple")
    normalized = tuple(sorted(_require_canonical_token(v, name=name) for v in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


def _normalize_unique_sorted_hashes(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be tuple")
    normalized = tuple(sorted(_require_sha256_hex(v, name=name) for v in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return normalized


@dataclass(frozen=True)
class CanonicalStorageArtifact:
    artifact_id: str
    artifact_type: str
    artifact_hash: str
    canonical_size_bytes: int
    lineage_hashes: tuple[str, ...]
    priority: int
    retention_class: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.artifact_id, name="artifact_id")
        _require_canonical_token(self.artifact_type, name="artifact_type")
        _require_sha256_hex(self.artifact_hash, name="artifact_hash")
        _require_non_negative_int(self.canonical_size_bytes, name="canonical_size_bytes")
        object.__setattr__(self, "lineage_hashes", _normalize_unique_sorted_hashes(self.lineage_hashes, name="lineage_hashes"))
        _require_non_negative_int(self.priority, name="priority")
        if self.retention_class not in _ALLOWED_RETENTION_CLASSES:
            raise ValueError("retention_class must be one of EPHEMERAL|SESSION|RELEASE|ARCHIVAL")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "artifact_hash": self.artifact_hash,
            "canonical_size_bytes": int(self.canonical_size_bytes),
            "lineage_hashes": self.lineage_hashes,
            "priority": int(self.priority),
            "retention_class": self.retention_class,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class CompressionSegment:
    segment_id: str
    artifact_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    segment_hash: str
    compressed_size_bytes: int
    compression_ratio: float
    storage_class: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        _require_canonical_token(self.segment_id, name="segment_id")
        object.__setattr__(self, "artifact_ids", _normalize_unique_sorted_tokens(self.artifact_ids, name="artifact_ids"))
        object.__setattr__(self, "source_hashes", _normalize_unique_sorted_hashes(self.source_hashes, name="source_hashes"))
        _require_sha256_hex(self.segment_hash, name="segment_hash")
        _require_non_negative_int(self.compressed_size_bytes, name="compressed_size_bytes")
        object.__setattr__(self, "compression_ratio", _round_public_metric(_require_probability(self.compression_ratio, name="compression_ratio")))
        if self.storage_class not in _ALLOWED_STORAGE_CLASSES:
            raise ValueError("storage_class must be one of HOT|WARM|COLD|ARCHIVAL")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "segment_id": self.segment_id,
            "artifact_ids": self.artifact_ids,
            "source_hashes": self.source_hashes,
            "segment_hash": self.segment_hash,
            "compressed_size_bytes": int(self.compressed_size_bytes),
            "compression_ratio": _round_public_metric(float(self.compression_ratio)),
            "storage_class": self.storage_class,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class CompressionStoragePlan:
    plan_status: str
    artifact_count: int
    segment_count: int
    segments: tuple[CompressionSegment, ...]
    total_canonical_size_bytes: int
    total_compressed_size_bytes: int
    overall_compression_ratio: float
    storage_integrity_hash: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.plan_status not in _ALLOWED_PLAN_STATUSES:
            raise ValueError("plan_status must be one of EMPTY|COMPRESSED|NO_GAIN|INVALID")
        _require_non_negative_int(self.artifact_count, name="artifact_count")
        _require_non_negative_int(self.segment_count, name="segment_count")
        if not isinstance(self.segments, tuple):
            raise ValueError("segments must be tuple")
        for item in self.segments:
            if not isinstance(item, CompressionSegment):
                raise ValueError("segments must contain CompressionSegment")
        sorted_segments = tuple(sorted(self.segments, key=lambda v: v.segment_id))
        if sorted_segments != self.segments:
            raise ValueError("segments must be sorted by segment_id")
        segment_ids = tuple(item.segment_id for item in sorted_segments)
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("segment_id must be unique")

        _require_non_negative_int(self.total_canonical_size_bytes, name="total_canonical_size_bytes")
        _require_non_negative_int(self.total_compressed_size_bytes, name="total_compressed_size_bytes")
        object.__setattr__(
            self,
            "overall_compression_ratio",
            _round_public_metric(_require_probability(self.overall_compression_ratio, name="overall_compression_ratio")),
        )
        _require_sha256_hex(self.storage_integrity_hash, name="storage_integrity_hash")

        segment_artifact_ids = tuple(artifact_id for segment in sorted_segments for artifact_id in segment.artifact_ids)
        if len(set(segment_artifact_ids)) != len(segment_artifact_ids):
            raise ValueError("segments must not duplicate artifact_ids")
        if self.artifact_count != len(segment_artifact_ids):
            raise ValueError("artifact_count must match segment artifact_ids")
        if self.segment_count != len(sorted_segments):
            raise ValueError("segment_count must match segments length")

        if self.plan_status == "EMPTY":
            if any(
                (
                    self.artifact_count != 0,
                    self.segment_count != 0,
                    self.total_canonical_size_bytes != 0,
                    self.total_compressed_size_bytes != 0,
                    float(self.overall_compression_ratio) != 0.0,
                )
            ):
                raise ValueError("EMPTY plan_status requires zero counts/sizes and ratio 0.0")
        elif self.plan_status == "COMPRESSED":
            if self.segment_count == 0 or float(self.overall_compression_ratio) >= 1.0:
                raise ValueError("COMPRESSED plan_status requires segment_count > 0 and ratio < 1.0")
        elif self.plan_status == "NO_GAIN":
            if self.segment_count == 0:
                raise ValueError("NO_GAIN plan_status requires segment_count > 0")
            if self.total_canonical_size_bytes == 0:
                if float(self.overall_compression_ratio) != 0.0:
                    raise ValueError("NO_GAIN with zero canonical size requires ratio 0.0")
            elif float(self.overall_compression_ratio) != 1.0:
                raise ValueError("NO_GAIN plan_status requires ratio 1.0 when canonical size is non-zero")

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "plan_status": self.plan_status,
            "artifact_count": int(self.artifact_count),
            "segment_count": int(self.segment_count),
            "segments": tuple(item.to_dict() for item in self.segments),
            "total_canonical_size_bytes": int(self.total_canonical_size_bytes),
            "total_compressed_size_bytes": int(self.total_compressed_size_bytes),
            "overall_compression_ratio": _round_public_metric(float(self.overall_compression_ratio)),
            "storage_integrity_hash": self.storage_integrity_hash,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


@dataclass(frozen=True)
class CompressionStorageReceipt:
    module_version: str
    artifact_count: int
    plan: CompressionStoragePlan
    compression_enabled: bool
    invariant_preserved: bool
    receipt_reason: str
    stable_hash_input: InitVar[str | None] = None
    _stable_hash: str = field(init=False, repr=False)

    def __post_init__(self, stable_hash_input: str | None) -> None:
        if self.module_version != DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION:
            raise ValueError("module_version must match DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION")
        _require_non_negative_int(self.artifact_count, name="artifact_count")
        if self.artifact_count != self.plan.artifact_count:
            raise ValueError("artifact_count must match plan.artifact_count")
        if not isinstance(self.compression_enabled, bool):
            raise ValueError("compression_enabled must be bool")
        if not isinstance(self.invariant_preserved, bool):
            raise ValueError("invariant_preserved must be bool")
        _require_canonical_token(self.receipt_reason, name="receipt_reason")
        if not self.invariant_preserved and self.receipt_reason == "invariant_preserved":
            raise ValueError('receipt_reason must not be "invariant_preserved" when invariant_preserved is False')

        computed = sha256_hex(self._payload_without_hash())
        if stable_hash_input is None:
            object.__setattr__(self, "_stable_hash", computed)
            return
        provided = _require_sha256_hex(stable_hash_input, name="stable_hash")
        if provided != computed:
            raise ValueError("stable_hash mismatch")
        object.__setattr__(self, "_stable_hash", provided)

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "module_version": self.module_version,
            "artifact_count": int(self.artifact_count),
            "plan": self.plan.to_dict(),
            "compression_enabled": self.compression_enabled,
            "invariant_preserved": self.invariant_preserved,
            "receipt_reason": self.receipt_reason,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self._stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash(self) -> str:
        return self._stable_hash


def _compression_sort_key(artifact: CanonicalStorageArtifact) -> tuple[int, int, str, str, str]:
    return (
        -_RETENTION_RANK[artifact.retention_class],
        -artifact.priority,
        artifact.artifact_type,
        artifact.artifact_id,
        artifact.artifact_hash,
    )


def _build_segment(
    *,
    retention_class: str,
    artifact_type: str,
    artifacts: tuple[CanonicalStorageArtifact, ...],
    compression_enabled: bool,
) -> CompressionSegment:
    artifact_ids = tuple(sorted(artifact.artifact_id for artifact in artifacts))
    source_hashes = tuple(sorted(artifact.artifact_hash for artifact in artifacts))
    total_group_size = sum(artifact.canonical_size_bytes for artifact in artifacts)

    if compression_enabled:
        factor = _COMPRESSION_FACTOR_BY_RETENTION[retention_class]
        if total_group_size == 0:
            compressed_size = 0
            compression_ratio = 0.0
        else:
            compressed_size = max(1, int(total_group_size * factor))
            compression_ratio = _round_public_metric(compressed_size / total_group_size)
    else:
        compressed_size = total_group_size
        compression_ratio = 1.0

    segment_seed_hash = sha256_hex(
        {
            "retention_class": retention_class,
            "artifact_type": artifact_type,
            "artifact_ids": artifact_ids,
            "source_hashes": source_hashes,
        }
    )
    segment_id = f"seg-{retention_class.lower()}-{artifact_type}-{segment_seed_hash[:12]}"
    storage_class = _STORAGE_CLASS_BY_RETENTION[retention_class]

    segment_hash = sha256_hex(
        {
            "segment_id": segment_id,
            "artifact_ids": artifact_ids,
            "source_hashes": source_hashes,
            "compressed_size_bytes": int(compressed_size),
            "compression_ratio": _round_public_metric(compression_ratio),
            "storage_class": storage_class,
        }
    )

    return CompressionSegment(
        segment_id=segment_id,
        artifact_ids=artifact_ids,
        source_hashes=source_hashes,
        segment_hash=segment_hash,
        compressed_size_bytes=compressed_size,
        compression_ratio=compression_ratio,
        storage_class=storage_class,
    )


def _storage_integrity_hash(
    *,
    artifacts: tuple[CanonicalStorageArtifact, ...],
    segments: tuple[CompressionSegment, ...],
    total_canonical_size_bytes: int,
    total_compressed_size_bytes: int,
    overall_compression_ratio: float,
) -> str:
    return sha256_hex(
        {
            "artifact_hashes": tuple(sorted(artifact.artifact_hash for artifact in artifacts)),
            "segment_hashes": tuple(sorted(segment.segment_hash for segment in segments)),
            "total_canonical_size_bytes": int(total_canonical_size_bytes),
            "total_compressed_size_bytes": int(total_compressed_size_bytes),
            "overall_compression_ratio": _round_public_metric(overall_compression_ratio),
        }
    )


def plan_deterministic_compression_storage(
    artifacts: Sequence[CanonicalStorageArtifact],
    *,
    compression_enabled: bool = True,
) -> CompressionStorageReceipt:
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise ValueError("artifacts must be a sequence of CanonicalStorageArtifact")
    if not isinstance(compression_enabled, bool):
        raise ValueError("compression_enabled must be bool")

    normalized = tuple(artifacts)
    for item in normalized:
        if not isinstance(item, CanonicalStorageArtifact):
            raise ValueError("artifacts must contain CanonicalStorageArtifact")

    artifact_ids = tuple(item.artifact_id for item in normalized)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("duplicate artifact_id is not allowed")

    artifact_hashes = tuple(item.artifact_hash for item in normalized)
    if len(set(artifact_hashes)) != len(artifact_hashes):
        raise ValueError("duplicate artifact_hash is not allowed")

    if not normalized:
        plan = CompressionStoragePlan(
            plan_status="EMPTY",
            artifact_count=0,
            segment_count=0,
            segments=tuple(),
            total_canonical_size_bytes=0,
            total_compressed_size_bytes=0,
            overall_compression_ratio=0.0,
            storage_integrity_hash=_storage_integrity_hash(
                artifacts=tuple(),
                segments=tuple(),
                total_canonical_size_bytes=0,
                total_compressed_size_bytes=0,
                overall_compression_ratio=0.0,
            ),
        )
        return CompressionStorageReceipt(
            module_version=DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION,
            artifact_count=0,
            plan=plan,
            compression_enabled=compression_enabled,
            invariant_preserved=True,
            receipt_reason="empty",
        )

    sorted_artifacts = tuple(sorted(normalized, key=_compression_sort_key))

    segments_by_group: dict[tuple[str, str], list[CanonicalStorageArtifact]] = {}
    if compression_enabled:
        for artifact in sorted_artifacts:
            group_key = (artifact.retention_class, artifact.artifact_type)
            segments_by_group.setdefault(group_key, []).append(artifact)
    else:
        for artifact in sorted_artifacts:
            group_key = (artifact.retention_class, artifact.artifact_id)
            segments_by_group[group_key] = [artifact]

    segment_list: list[CompressionSegment] = []
    for group_key in sorted(segments_by_group.keys()):
        retention_class = group_key[0]
        if compression_enabled:
            artifact_type = group_key[1]
        else:
            artifact_type = segments_by_group[group_key][0].artifact_type
        segment_list.append(
            _build_segment(
                retention_class=retention_class,
                artifact_type=artifact_type,
                artifacts=tuple(segments_by_group[group_key]),
                compression_enabled=compression_enabled,
            )
        )

    segments = tuple(sorted(segment_list, key=lambda v: v.segment_id))
    total_canonical_size_bytes = sum(item.canonical_size_bytes for item in sorted_artifacts)
    total_compressed_size_bytes = sum(item.compressed_size_bytes for item in segments)
    if total_canonical_size_bytes == 0:
        overall_compression_ratio = 0.0
    else:
        overall_compression_ratio = _round_public_metric(total_compressed_size_bytes / total_canonical_size_bytes)

    if total_compressed_size_bytes < total_canonical_size_bytes:
        plan_status = "COMPRESSED"
    else:
        plan_status = "NO_GAIN"

    if not compression_enabled:
        overall_compression_ratio = 1.0
        total_compressed_size_bytes = total_canonical_size_bytes
        plan_status = "NO_GAIN"

    plan = CompressionStoragePlan(
        plan_status=plan_status,
        artifact_count=len(sorted_artifacts),
        segment_count=len(segments),
        segments=segments,
        total_canonical_size_bytes=total_canonical_size_bytes,
        total_compressed_size_bytes=total_compressed_size_bytes,
        overall_compression_ratio=overall_compression_ratio,
        storage_integrity_hash=_storage_integrity_hash(
            artifacts=sorted_artifacts,
            segments=segments,
            total_canonical_size_bytes=total_canonical_size_bytes,
            total_compressed_size_bytes=total_compressed_size_bytes,
            overall_compression_ratio=overall_compression_ratio,
        ),
    )

    return CompressionStorageReceipt(
        module_version=DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION,
        artifact_count=len(sorted_artifacts),
        plan=plan,
        compression_enabled=compression_enabled,
        invariant_preserved=True,
        receipt_reason="invariant_preserved" if compression_enabled else "compression_disabled",
    )


__all__ = [
    "DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION",
    "CanonicalStorageArtifact",
    "CompressionSegment",
    "CompressionStoragePlan",
    "CompressionStorageReceipt",
    "plan_deterministic_compression_storage",
]
