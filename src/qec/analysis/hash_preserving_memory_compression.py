"""v137.7.2 — Hash-Preserving Compression Chain.

Deterministic Layer-4 compression of semantic themes into a replay-safe chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.semantic_theme_compaction import SemanticThemeArtifact, ThemeRecord

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

HASH_PRESERVING_COMPRESSION_LAW = "HASH_PRESERVING_COMPRESSION_LAW"
DETERMINISTIC_MEMORY_COMPRESSION_RULE = "DETERMINISTIC_MEMORY_COMPRESSION_RULE"
REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT = "REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT"


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


@dataclass(frozen=True)
class CompressionRecord:
    theme_id: str
    theme_index: int
    source_theme_hash: str
    source_replay_identity_hash: str
    source_parent_theme_hash: str
    signature_ref: int
    reason_ref: int
    episode_hashes_ref: int
    compression_record_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "theme_id": self.theme_id,
            "theme_index": self.theme_index,
            "source_theme_hash": self.source_theme_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "source_parent_theme_hash": self.source_parent_theme_hash,
            "signature_ref": self.signature_ref,
            "reason_ref": self.reason_ref,
            "episode_hashes_ref": self.episode_hashes_ref,
            "compression_record_hash": self.compression_record_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "theme_id": self.theme_id,
            "theme_index": self.theme_index,
            "source_theme_hash": self.source_theme_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "source_parent_theme_hash": self.source_parent_theme_hash,
            "signature_ref": self.signature_ref,
            "reason_ref": self.reason_ref,
            "episode_hashes_ref": self.episode_hashes_ref,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CompressedMemoryArtifact:
    schema_version: int
    source_artifact_hash: str
    source_theme_count: int
    compressed_record_count: int
    signature_table: tuple[str, ...]
    reason_table: tuple[tuple[str, ...], ...]
    episode_hash_chain_table: tuple[tuple[str, ...], ...]
    records: tuple[CompressionRecord, ...]
    preserved_theme_hashes: tuple[str, ...]
    source_compaction_ratio: float
    compression_ratio: float
    compression_chain_head: str
    replay_identity_hash: str
    law_invariants: tuple[str, ...]
    compression_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "source_theme_count": self.source_theme_count,
            "compressed_record_count": self.compressed_record_count,
            "signature_table": self.signature_table,
            "reason_table": self.reason_table,
            "episode_hash_chain_table": self.episode_hash_chain_table,
            "records": tuple(record.to_dict() for record in self.records),
            "preserved_theme_hashes": self.preserved_theme_hashes,
            "source_compaction_ratio": self.source_compaction_ratio,
            "compression_ratio": self.compression_ratio,
            "compression_chain_head": self.compression_chain_head,
            "replay_identity_hash": self.replay_identity_hash,
            "law_invariants": self.law_invariants,
            "compression_hash": self.compression_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "source_theme_count": self.source_theme_count,
            "compressed_record_count": self.compressed_record_count,
            "signature_table": self.signature_table,
            "reason_table": self.reason_table,
            "episode_hash_chain_table": self.episode_hash_chain_table,
            "records": tuple(record.to_dict() for record in self.records),
            "preserved_theme_hashes": self.preserved_theme_hashes,
            "source_compaction_ratio": self.source_compaction_ratio,
            "compression_ratio": self.compression_ratio,
            "compression_chain_head": self.compression_chain_head,
            "replay_identity_hash": self.replay_identity_hash,
            "law_invariants": self.law_invariants,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CompressionReceipt:
    schema_version: int
    source_artifact_hash: str
    compression_hash: str
    compression_chain_head: str
    replay_identity_hash: str
    preserved_theme_hashes: tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "compression_hash": self.compression_hash,
            "compression_chain_head": self.compression_chain_head,
            "replay_identity_hash": self.replay_identity_hash,
            "preserved_theme_hashes": self.preserved_theme_hashes,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_artifact_hash": self.source_artifact_hash,
            "compression_hash": self.compression_hash,
            "compression_chain_head": self.compression_chain_head,
            "replay_identity_hash": self.replay_identity_hash,
            "preserved_theme_hashes": self.preserved_theme_hashes,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _validate_theme_record(theme: ThemeRecord, *, expected_index: int) -> None:
    _validate_non_empty_str(theme.theme_id, field_name="theme_id")
    if not isinstance(theme.theme_index, int) or theme.theme_index != expected_index:
        raise ValueError("themes must be ordered with contiguous theme_index")
    _validate_non_empty_str(theme.theme_signature, field_name="theme_signature")
    _validate_non_empty_str(theme.theme_hash, field_name="theme_hash")
    _validate_non_empty_str(theme.replay_identity_hash, field_name="replay_identity_hash")
    _validate_non_empty_str(theme.parent_theme_hash, field_name="parent_theme_hash")
    if len(theme.episode_hashes) == 0:
        raise ValueError("theme episode_hashes must be non-empty")
    if len(theme.compaction_reasons) == 0:
        raise ValueError("theme compaction_reasons must be non-empty")


def _validate_semantic_theme_artifact(artifact: SemanticThemeArtifact) -> tuple[ThemeRecord, ...]:
    if not isinstance(artifact, SemanticThemeArtifact):
        raise ValueError("artifact must be a SemanticThemeArtifact")
    _validate_non_empty_str(artifact.artifact_hash, field_name="artifact_hash")
    if artifact.theme_count != len(artifact.themes):
        raise ValueError("theme_count must match themes length")
    if artifact.theme_count != len(artifact.theme_ids):
        raise ValueError("theme_count must match theme_ids length")
    if tuple(theme.theme_id for theme in artifact.themes) != artifact.theme_ids:
        raise ValueError("theme_ids must align with themes ordering")
    themes = artifact.themes
    for idx, theme in enumerate(themes):
        _validate_theme_record(theme, expected_index=idx)
    return themes


def compress_semantic_theme_memory(artifact: SemanticThemeArtifact) -> CompressedMemoryArtifact:
    themes = _validate_semantic_theme_artifact(artifact)

    signatures: list[str] = []
    signature_ref_map: dict[str, int] = {}
    reasons: list[tuple[str, ...]] = []
    reason_ref_map: dict[tuple[str, ...], int] = {}
    episode_hash_chains: list[tuple[str, ...]] = []
    episode_chain_ref_map: dict[tuple[str, ...], int] = {}

    records: list[CompressionRecord] = []
    for theme in themes:
        signature = theme.theme_signature
        signature_ref = signature_ref_map.get(signature)
        if signature_ref is None:
            signature_ref = len(signatures)
            signature_ref_map[signature] = signature_ref
            signatures.append(signature)

        reason_tuple = tuple(theme.compaction_reasons)
        reason_ref = reason_ref_map.get(reason_tuple)
        if reason_ref is None:
            reason_ref = len(reasons)
            reason_ref_map[reason_tuple] = reason_ref
            reasons.append(reason_tuple)

        episode_hashes = tuple(theme.episode_hashes)
        episode_ref = episode_chain_ref_map.get(episode_hashes)
        if episode_ref is None:
            episode_ref = len(episode_hash_chains)
            episode_chain_ref_map[episode_hashes] = episode_ref
            episode_hash_chains.append(episode_hashes)

        record_payload = {
            "theme_id": theme.theme_id,
            "theme_index": theme.theme_index,
            "source_theme_hash": theme.theme_hash,
            "source_replay_identity_hash": theme.replay_identity_hash,
            "source_parent_theme_hash": theme.parent_theme_hash,
            "signature_ref": signature_ref,
            "reason_ref": reason_ref,
            "episode_hashes_ref": episode_ref,
        }
        records.append(
            CompressionRecord(
                theme_id=theme.theme_id,
                theme_index=theme.theme_index,
                source_theme_hash=theme.theme_hash,
                source_replay_identity_hash=theme.replay_identity_hash,
                source_parent_theme_hash=theme.parent_theme_hash,
                signature_ref=signature_ref,
                reason_ref=reason_ref,
                episode_hashes_ref=episode_ref,
                compression_record_hash=_sha256_hex(record_payload),
            )
        )

    preserved_theme_hashes = tuple(theme.theme_hash for theme in themes)
    compression_chain_head = artifact.source_artifact_hash if len(records) == 0 else records[-1].source_replay_identity_hash
    replay_identity_hash = _sha256_hex(
        {
            "source_artifact_hash": artifact.artifact_hash,
            "compression_chain_head": compression_chain_head,
            "preserved_theme_hashes": preserved_theme_hashes,
        }
    )

    source_payload = {
        "themes": tuple(theme.to_dict() for theme in themes),
        "mapping": artifact.episode_to_theme,
    }
    compressed_payload_for_ratio = {
        "records": tuple(record.to_dict() for record in records),
        "signature_table": tuple(signatures),
        "reason_table": tuple(reasons),
        "episode_hash_chain_table": tuple(episode_hash_chains),
    }
    source_bytes_len = len(_canonical_bytes(source_payload))
    compressed_bytes_len = len(_canonical_bytes(compressed_payload_for_ratio))
    if source_bytes_len <= 0:
        raise ValueError("source bytes length must be positive")
    raw_ratio = float((source_bytes_len - compressed_bytes_len) / source_bytes_len)
    compression_ratio = min(1.0, max(0.0, raw_ratio))

    law_invariants = (
        HASH_PRESERVING_COMPRESSION_LAW,
        DETERMINISTIC_MEMORY_COMPRESSION_RULE,
        REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT,
    )

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_artifact_hash": artifact.artifact_hash,
        "source_theme_count": artifact.theme_count,
        "compressed_record_count": len(records),
        "signature_table": tuple(signatures),
        "reason_table": tuple(reasons),
        "episode_hash_chain_table": tuple(episode_hash_chains),
        "records": tuple(record.to_dict() for record in records),
        "preserved_theme_hashes": preserved_theme_hashes,
        "source_compaction_ratio": artifact.compaction_ratio,
        "compression_ratio": compression_ratio,
        "compression_chain_head": compression_chain_head,
        "replay_identity_hash": replay_identity_hash,
        "law_invariants": law_invariants,
    }
    compression_hash = _sha256_hex(payload)

    return CompressedMemoryArtifact(
        schema_version=_SCHEMA_VERSION,
        source_artifact_hash=artifact.artifact_hash,
        source_theme_count=artifact.theme_count,
        compressed_record_count=len(records),
        signature_table=tuple(signatures),
        reason_table=tuple(reasons),
        episode_hash_chain_table=tuple(episode_hash_chains),
        records=tuple(records),
        preserved_theme_hashes=preserved_theme_hashes,
        source_compaction_ratio=artifact.compaction_ratio,
        compression_ratio=compression_ratio,
        compression_chain_head=compression_chain_head,
        replay_identity_hash=replay_identity_hash,
        law_invariants=law_invariants,
        compression_hash=compression_hash,
    )


def generate_compression_receipt(artifact: CompressedMemoryArtifact) -> CompressionReceipt:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("artifact must be a CompressedMemoryArtifact")
    payload = {
        "schema_version": artifact.schema_version,
        "source_artifact_hash": artifact.source_artifact_hash,
        "compression_hash": artifact.compression_hash,
        "compression_chain_head": artifact.compression_chain_head,
        "replay_identity_hash": artifact.replay_identity_hash,
        "preserved_theme_hashes": artifact.preserved_theme_hashes,
    }
    receipt_hash = _sha256_hex(payload)
    return CompressionReceipt(
        schema_version=artifact.schema_version,
        source_artifact_hash=artifact.source_artifact_hash,
        compression_hash=artifact.compression_hash,
        compression_chain_head=artifact.compression_chain_head,
        replay_identity_hash=artifact.replay_identity_hash,
        preserved_theme_hashes=artifact.preserved_theme_hashes,
        receipt_hash=receipt_hash,
    )


def export_compressed_memory_bytes(artifact: CompressedMemoryArtifact) -> bytes:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("artifact must be a CompressedMemoryArtifact")
    return artifact.to_canonical_bytes()


__all__ = [
    "DETERMINISTIC_MEMORY_COMPRESSION_RULE",
    "HASH_PRESERVING_COMPRESSION_LAW",
    "REPLAY_SAFE_COMPRESSION_CHAIN_INVARIANT",
    "CompressedMemoryArtifact",
    "CompressionReceipt",
    "CompressionRecord",
    "compress_semantic_theme_memory",
    "export_compressed_memory_bytes",
    "generate_compression_receipt",
]
