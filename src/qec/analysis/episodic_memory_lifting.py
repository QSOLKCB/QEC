"""v137.7.0 — Raw → Episode Hierarchy.

Deterministic Layer-4 memory lifting from raw records into bounded episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

EPISODIC_MEMORY_LIFTING_LAW = "EPISODIC_MEMORY_LIFTING_LAW"
DETERMINISTIC_EPISODE_BOUNDARY_RULE = "DETERMINISTIC_EPISODE_BOUNDARY_RULE"
REPLAY_SAFE_EPISODE_CHAIN_INVARIANT = "REPLAY_SAFE_EPISODE_CHAIN_INVARIANT"
BOUNDED_EPISODE_AGGREGATION_RULE = "BOUNDED_EPISODE_AGGREGATION_RULE"

_BOUNDARY_REASON_ORDER: tuple[str, ...] = (
    "max_length",
    "reset_marker",
    "task_completion",
    "source_discontinuity",
    "provenance_discontinuity",
    "state_discontinuity",
    "timestamp_gap",
)



def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")



def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )



def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")



def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()



def _validate_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped



def _validate_optional_str(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _validate_non_empty_str(value, field_name=field_name)



def _validate_optional_bool(value: object, *, field_name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be bool when provided")
    return value



def _validate_optional_timestamp(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timestamp must be numeric when provided")
    ts = float(value)
    if not math.isfinite(ts):
        raise ValueError("timestamp must be finite when provided")
    return ts


@dataclass(frozen=True)
class EpisodeBoundaryConfig:
    max_episode_length: int = 8
    timestamp_gap_seconds: float | None = None
    normalize_by_sequence_index: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.max_episode_length, int) or self.max_episode_length <= 0:
            raise ValueError("max_episode_length must be a positive integer")
        if self.timestamp_gap_seconds is not None:
            if isinstance(self.timestamp_gap_seconds, bool) or not isinstance(self.timestamp_gap_seconds, (int, float)):
                raise ValueError("timestamp_gap_seconds must be numeric when provided")
            gap = float(self.timestamp_gap_seconds)
            if not math.isfinite(gap) or gap <= 0.0:
                raise ValueError("timestamp_gap_seconds must be finite and > 0")
            object.__setattr__(self, "timestamp_gap_seconds", gap)
        if not isinstance(self.normalize_by_sequence_index, bool):
            raise ValueError("normalize_by_sequence_index must be bool")


@dataclass(frozen=True)
class RawMemoryRecord:
    record_id: str
    sequence_index: int
    source_id: str | None = None
    provenance_id: str | None = None
    state_token: str | None = None
    timestamp: float | None = None
    is_reset: bool = False
    task_completed: bool = False

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], *, index: int) -> "RawMemoryRecord":
        if not isinstance(payload, Mapping):
            raise ValueError("records must be mappings")
        record_id = _validate_non_empty_str(payload.get("record_id"), field_name="record_id")

        raw_sequence = payload.get("sequence_index", index)
        if isinstance(raw_sequence, bool) or not isinstance(raw_sequence, int):
            raise ValueError("sequence_index must be an integer")
        if raw_sequence < 0:
            raise ValueError("sequence_index must be >= 0")

        return cls(
            record_id=record_id,
            sequence_index=raw_sequence,
            source_id=_validate_optional_str(payload.get("source_id"), field_name="source_id"),
            provenance_id=_validate_optional_str(payload.get("provenance_id"), field_name="provenance_id"),
            state_token=_validate_optional_str(payload.get("state_token"), field_name="state_token"),
            timestamp=_validate_optional_timestamp(payload.get("timestamp")),
            is_reset=_validate_optional_bool(payload.get("is_reset"), field_name="is_reset"),
            task_completed=_validate_optional_bool(payload.get("task_completed"), field_name="task_completed"),
        )

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "record_id": self.record_id,
            "sequence_index": self.sequence_index,
            "source_id": self.source_id,
            "provenance_id": self.provenance_id,
            "state_token": self.state_token,
            "timestamp": self.timestamp,
            "is_reset": self.is_reset,
            "task_completed": self.task_completed,
        }


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    episode_index: int
    record_ids: tuple[str, ...]
    start_sequence_index: int
    end_sequence_index: int
    boundary_reasons: tuple[str, ...]
    parent_episode_hash: str
    episode_hash: str
    replay_identity_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "record_ids": self.record_ids,
            "start_sequence_index": self.start_sequence_index,
            "end_sequence_index": self.end_sequence_index,
            "boundary_reasons": self.boundary_reasons,
            "parent_episode_hash": self.parent_episode_hash,
            "episode_hash": self.episode_hash,
            "replay_identity_hash": self.replay_identity_hash,
        }


@dataclass(frozen=True)
class EpisodicMemoryArtifact:
    schema_version: int
    source_sequence_hash: str
    total_records: int
    episode_count: int
    episode_ids: tuple[str, ...]
    episodes: tuple[EpisodeRecord, ...]
    law_invariants: tuple[str, ...]
    artifact_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_sequence_hash": self.source_sequence_hash,
            "total_records": self.total_records,
            "episode_count": self.episode_count,
            "episode_ids": self.episode_ids,
            "episodes": tuple(ep.to_dict() for ep in self.episodes),
            "law_invariants": self.law_invariants,
            "artifact_hash": self.artifact_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class EpisodicMemoryReceipt:
    schema_version: int
    artifact_hash: str
    source_sequence_hash: str
    replay_chain_head: str
    episode_hashes: tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "artifact_hash": self.artifact_hash,
            "source_sequence_hash": self.source_sequence_hash,
            "replay_chain_head": self.replay_chain_head,
            "episode_hashes": self.episode_hashes,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")



def _normalize_records(
    records: Sequence[Mapping[str, object]],
    *,
    normalize_by_sequence_index: bool,
) -> tuple[RawMemoryRecord, ...]:
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        raise ValueError("records must be a sequence of mappings")
    if len(records) == 0:
        raise ValueError("records must be non-empty")

    parsed_records: list[RawMemoryRecord] = []
    for i, item in enumerate(records):
        if isinstance(item, RawMemoryRecord):
            parsed_records.append(item)
        else:
            parsed_records.append(RawMemoryRecord.from_mapping(item, index=i))
    parsed = tuple(parsed_records)
    if normalize_by_sequence_index:
        parsed = tuple(sorted(parsed, key=lambda item: (item.sequence_index, item.record_id)))

    ids = tuple(item.record_id for item in parsed)
    if len(set(ids)) != len(ids):
        raise ValueError("record_id values must be unique")

    if not normalize_by_sequence_index:
        expected = tuple(item.sequence_index for item in parsed)
        if any(expected[i] > expected[i + 1] for i in range(len(expected) - 1)):
            raise ValueError("sequence_index must be non-decreasing without normalization")
    else:
        seq = tuple(item.sequence_index for item in parsed)
        if any(seq[i] > seq[i + 1] for i in range(len(seq) - 1)):
            raise ValueError("failed to normalize sequence_index ordering")

    return parsed



def _boundary_reasons(
    prev_record: RawMemoryRecord,
    current_record: RawMemoryRecord,
    *,
    current_episode_length: int,
    config: EpisodeBoundaryConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if current_episode_length >= config.max_episode_length:
        reasons.append("max_length")
    if current_record.is_reset:
        reasons.append("reset_marker")
    if prev_record.task_completed:
        reasons.append("task_completion")
    if prev_record.source_id != current_record.source_id:
        reasons.append("source_discontinuity")
    if prev_record.provenance_id != current_record.provenance_id:
        reasons.append("provenance_discontinuity")
    if prev_record.state_token != current_record.state_token:
        reasons.append("state_discontinuity")
    if (
        config.timestamp_gap_seconds is not None
        and prev_record.timestamp is not None
        and current_record.timestamp is not None
        and (current_record.timestamp - prev_record.timestamp) > config.timestamp_gap_seconds
    ):
        reasons.append("timestamp_gap")

    if len(reasons) == 0:
        return ()
    reason_set = set(reasons)
    return tuple(reason for reason in _BOUNDARY_REASON_ORDER if reason in reason_set)



def detect_episode_boundaries(
    records: Sequence[Mapping[str, object]],
    *,
    config: EpisodeBoundaryConfig = EpisodeBoundaryConfig(),
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    normalized = _normalize_records(records, normalize_by_sequence_index=config.normalize_by_sequence_index)
    boundaries: list[tuple[int, tuple[str, ...]]] = []
    current_len = 1
    for i in range(1, len(normalized)):
        reasons = _boundary_reasons(
            normalized[i - 1],
            normalized[i],
            current_episode_length=current_len,
            config=config,
        )
        if len(reasons) > 0:
            boundaries.append((i, reasons))
            current_len = 1
        else:
            current_len += 1
    return tuple(boundaries)



def lift_raw_records_to_episodic_memory(
    records: Sequence[Mapping[str, object]],
    *,
    config: EpisodeBoundaryConfig = EpisodeBoundaryConfig(),
) -> EpisodicMemoryArtifact:
    normalized = _normalize_records(records, normalize_by_sequence_index=config.normalize_by_sequence_index)
    source_sequence_hash = _sha256_hex(tuple(item.to_dict() for item in normalized))
    boundaries = detect_episode_boundaries(normalized, config=config)
    split_points = (0,) + tuple(idx for idx, _ in boundaries) + (len(normalized),)

    episodes: list[EpisodeRecord] = []
    parent_hash = source_sequence_hash
    for ep_index in range(len(split_points) - 1):
        start = split_points[ep_index]
        end = split_points[ep_index + 1]
        chunk = normalized[start:end]
        reasons = () if ep_index == 0 else boundaries[ep_index - 1][1]
        episode_id = f"episode-{ep_index:06d}"
        episode_body = {
            "episode_id": episode_id,
            "episode_index": ep_index,
            "record_ids": tuple(item.record_id for item in chunk),
            "start_sequence_index": chunk[0].sequence_index,
            "end_sequence_index": chunk[-1].sequence_index,
            "boundary_reasons": reasons,
            "parent_episode_hash": parent_hash,
        }
        episode_hash = _sha256_hex(episode_body)
        replay_identity_hash = _sha256_hex({"parent": parent_hash, "episode_hash": episode_hash})
        episodes.append(
            EpisodeRecord(
                episode_id=episode_id,
                episode_index=ep_index,
                record_ids=episode_body["record_ids"],
                start_sequence_index=chunk[0].sequence_index,
                end_sequence_index=chunk[-1].sequence_index,
                boundary_reasons=reasons,
                parent_episode_hash=parent_hash,
                episode_hash=episode_hash,
                replay_identity_hash=replay_identity_hash,
            )
        )
        parent_hash = replay_identity_hash

    law_invariants = (
        EPISODIC_MEMORY_LIFTING_LAW,
        DETERMINISTIC_EPISODE_BOUNDARY_RULE,
        REPLAY_SAFE_EPISODE_CHAIN_INVARIANT,
        BOUNDED_EPISODE_AGGREGATION_RULE,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_sequence_hash": source_sequence_hash,
        "total_records": len(normalized),
        "episode_count": len(episodes),
        "episode_ids": tuple(item.episode_id for item in episodes),
        "episodes": tuple(item.to_dict() for item in episodes),
        "law_invariants": law_invariants,
    }
    artifact_hash = _sha256_hex(payload)
    return EpisodicMemoryArtifact(
        schema_version=_SCHEMA_VERSION,
        source_sequence_hash=source_sequence_hash,
        total_records=len(normalized),
        episode_count=len(episodes),
        episode_ids=tuple(item.episode_id for item in episodes),
        episodes=tuple(episodes),
        law_invariants=law_invariants,
        artifact_hash=artifact_hash,
    )



def generate_episodic_memory_receipt(artifact: EpisodicMemoryArtifact) -> EpisodicMemoryReceipt:
    if not isinstance(artifact, EpisodicMemoryArtifact):
        raise ValueError("artifact must be an EpisodicMemoryArtifact")
    replay_chain_head = artifact.source_sequence_hash if artifact.episode_count == 0 else artifact.episodes[-1].replay_identity_hash
    payload = {
        "schema_version": artifact.schema_version,
        "artifact_hash": artifact.artifact_hash,
        "source_sequence_hash": artifact.source_sequence_hash,
        "replay_chain_head": replay_chain_head,
        "episode_hashes": tuple(ep.episode_hash for ep in artifact.episodes),
    }
    receipt_hash = _sha256_hex(payload)
    return EpisodicMemoryReceipt(
        schema_version=artifact.schema_version,
        artifact_hash=artifact.artifact_hash,
        source_sequence_hash=artifact.source_sequence_hash,
        replay_chain_head=replay_chain_head,
        episode_hashes=tuple(ep.episode_hash for ep in artifact.episodes),
        receipt_hash=receipt_hash,
    )



def export_episodic_memory_bytes(artifact: EpisodicMemoryArtifact) -> bytes:
    if not isinstance(artifact, EpisodicMemoryArtifact):
        raise ValueError("artifact must be an EpisodicMemoryArtifact")
    return artifact.to_canonical_bytes()


__all__ = [
    "BOUNDED_EPISODE_AGGREGATION_RULE",
    "DETERMINISTIC_EPISODE_BOUNDARY_RULE",
    "EPISODIC_MEMORY_LIFTING_LAW",
    "REPLAY_SAFE_EPISODE_CHAIN_INVARIANT",
    "EpisodeBoundaryConfig",
    "EpisodeRecord",
    "EpisodicMemoryArtifact",
    "EpisodicMemoryReceipt",
    "detect_episode_boundaries",
    "export_episodic_memory_bytes",
    "generate_episodic_memory_receipt",
    "lift_raw_records_to_episodic_memory",
]
