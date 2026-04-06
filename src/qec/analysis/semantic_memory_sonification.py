"""v137.7.2 — Deterministic Sonification Projection for Compressed Semantic Memory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, CompressionRecord

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

DETERMINISTIC_SONIFICATION_PROJECTION_RULE = "DETERMINISTIC_SONIFICATION_PROJECTION_RULE"

_FREQUENCY_BINS_HZ: tuple[float, ...] = (
    220.0,
    246.94,
    261.63,
    293.66,
    329.63,
    349.23,
    392.0,
    440.0,
    493.88,
    523.25,
    587.33,
    659.25,
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


@dataclass(frozen=True)
class SonificationProjectionSpec:
    schema_version: int
    source_compression_hash: str
    sample_rate_hz: int
    bit_depth_pcm: int
    channel_count: int
    total_duration_ms: int
    event_sequence: tuple[tuple[int, float, int, float, int], ...]
    wav_render_spec: dict[str, _JSONValue]
    law_invariants: tuple[str, ...]
    sonification_spec_hash: str
    audio_projection_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "sample_rate_hz": self.sample_rate_hz,
            "bit_depth_pcm": self.bit_depth_pcm,
            "channel_count": self.channel_count,
            "total_duration_ms": self.total_duration_ms,
            "event_sequence": self.event_sequence,
            "wav_render_spec": self.wav_render_spec,
            "law_invariants": self.law_invariants,
            "sonification_spec_hash": self.sonification_spec_hash,
            "audio_projection_hash": self.audio_projection_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "sample_rate_hz": self.sample_rate_hz,
            "bit_depth_pcm": self.bit_depth_pcm,
            "channel_count": self.channel_count,
            "total_duration_ms": self.total_duration_ms,
            "event_sequence": self.event_sequence,
            "wav_render_spec": self.wav_render_spec,
            "law_invariants": self.law_invariants,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SonificationReceipt:
    schema_version: int
    source_compression_hash: str
    sonification_spec_hash: str
    audio_projection_hash: str
    event_count: int
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "sonification_spec_hash": self.sonification_spec_hash,
            "audio_projection_hash": self.audio_projection_hash,
            "event_count": self.event_count,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "sonification_spec_hash": self.sonification_spec_hash,
            "audio_projection_hash": self.audio_projection_hash,
            "event_count": self.event_count,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def _validate_compressed_artifact(artifact: CompressedMemoryArtifact) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("artifact must be a CompressedMemoryArtifact")
    _validate_non_empty_str(artifact.compression_hash, field_name="compression_hash")
    if artifact.compressed_record_count != len(artifact.records):
        raise ValueError("compressed_record_count must match records length")
    records = artifact.records
    for idx, record in enumerate(records):
        if record.theme_index != idx:
            raise ValueError("record theme_index must be contiguous and ordered")
        _validate_non_empty_str(record.theme_id, field_name="theme_id")
        _validate_non_empty_str(record.source_theme_hash, field_name="source_theme_hash")
    return records


def _frequency_for_hash(hash_hex: str, *, signature_ref: int) -> float:
    bucket = (int(hash_hex[:8], 16) + signature_ref) % len(_FREQUENCY_BINS_HZ)
    return float(_FREQUENCY_BINS_HZ[bucket])


def project_compressed_memory_to_sonification(
    artifact: CompressedMemoryArtifact,
    *,
    sample_rate_hz: int = 48_000,
    bit_depth_pcm: int = 16,
) -> SonificationProjectionSpec:
    records = _validate_compressed_artifact(artifact)
    if sample_rate_hz != 48_000:
        raise ValueError("sample_rate_hz must be fixed at 48000 for deterministic replay")
    if bit_depth_pcm != 16:
        raise ValueError("bit_depth_pcm must be fixed at 16 for deterministic replay")

    events: list[tuple[int, float, int, float, int]] = []
    offset_ms = 0
    ratio = artifact.compression_ratio
    for idx, record in enumerate(records):
        frequency_hz = _frequency_for_hash(record.source_theme_hash, signature_ref=record.signature_ref)
        duration_ms = 180 + (record.reason_ref * 25) + int(round(ratio * 80.0))
        amplitude = round(0.25 + min(0.6, ratio * 0.6) + (record.signature_ref % 4) * 0.03, 6)
        channel = 0 if idx == 0 or record.source_parent_theme_hash == records[idx - 1].source_replay_identity_hash else 1
        events.append((offset_ms, frequency_hz, duration_ms, amplitude, channel))
        offset_ms += duration_ms

    total_duration_ms = offset_ms
    wav_render_spec: dict[str, _JSONValue] = {
        "encoding": "pcm_s16le",
        "sample_rate_hz": sample_rate_hz,
        "bit_depth_pcm": bit_depth_pcm,
        "channel_count": 2,
        "total_duration_ms": total_duration_ms,
        "deterministic_gain_normalization": "none",
    }
    law_invariants = (DETERMINISTIC_SONIFICATION_PROJECTION_RULE,)

    spec_payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_compression_hash": artifact.compression_hash,
        "sample_rate_hz": sample_rate_hz,
        "bit_depth_pcm": bit_depth_pcm,
        "channel_count": 2,
        "total_duration_ms": total_duration_ms,
        "event_sequence": tuple(events),
        "wav_render_spec": wav_render_spec,
        "law_invariants": law_invariants,
    }
    sonification_spec_hash = _sha256_hex(spec_payload)
    audio_projection_hash = _sha256_hex({"event_sequence": tuple(events), "wav_render_spec": wav_render_spec})

    return SonificationProjectionSpec(
        schema_version=_SCHEMA_VERSION,
        source_compression_hash=artifact.compression_hash,
        sample_rate_hz=sample_rate_hz,
        bit_depth_pcm=bit_depth_pcm,
        channel_count=2,
        total_duration_ms=total_duration_ms,
        event_sequence=tuple(events),
        wav_render_spec=wav_render_spec,
        law_invariants=law_invariants,
        sonification_spec_hash=sonification_spec_hash,
        audio_projection_hash=audio_projection_hash,
    )


def generate_sonification_receipt(spec: SonificationProjectionSpec) -> SonificationReceipt:
    if not isinstance(spec, SonificationProjectionSpec):
        raise ValueError("spec must be a SonificationProjectionSpec")
    payload = {
        "schema_version": spec.schema_version,
        "source_compression_hash": spec.source_compression_hash,
        "sonification_spec_hash": spec.sonification_spec_hash,
        "audio_projection_hash": spec.audio_projection_hash,
        "event_count": len(spec.event_sequence),
    }
    receipt_hash = _sha256_hex(payload)
    return SonificationReceipt(
        schema_version=spec.schema_version,
        source_compression_hash=spec.source_compression_hash,
        sonification_spec_hash=spec.sonification_spec_hash,
        audio_projection_hash=spec.audio_projection_hash,
        event_count=len(spec.event_sequence),
        receipt_hash=receipt_hash,
    )


def export_sonification_spec_bytes(spec: SonificationProjectionSpec) -> bytes:
    if not isinstance(spec, SonificationProjectionSpec):
        raise ValueError("spec must be a SonificationProjectionSpec")
    return spec.to_canonical_bytes()


def export_mp3_projection_manifest(spec: SonificationProjectionSpec, *, include_mp3_manifest: bool = False) -> bytes:
    if not isinstance(spec, SonificationProjectionSpec):
        raise ValueError("spec must be a SonificationProjectionSpec")
    if not include_mp3_manifest:
        raise ValueError("include_mp3_manifest must be explicitly True")
    manifest = {
        "source_compression_hash": spec.source_compression_hash,
        "sonification_spec_hash": spec.sonification_spec_hash,
        "audio_projection_hash": spec.audio_projection_hash,
        "codec": "mp3",
        "authoritative": False,
    }
    return _canonical_bytes(manifest)


__all__ = [
    "DETERMINISTIC_SONIFICATION_PROJECTION_RULE",
    "SonificationProjectionSpec",
    "SonificationReceipt",
    "export_mp3_projection_manifest",
    "export_sonification_spec_bytes",
    "generate_sonification_receipt",
    "project_compressed_memory_to_sonification",
]
