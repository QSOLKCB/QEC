"""v137.4.2 — Signed Provenance Artifact Chain.

Deterministic Layer 4 provenance chain with append-only lineage,
canonical export, replay-safe verification, and stable root hashing.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_ARTIFACT_SCHEMA_VERSION = 1
_CHAIN_SCHEMA_VERSION = 1
_GENESIS_PARENT_HASH = "0" * 64


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_non_empty_str(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _validate_hash_hex(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field_name} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc
    return value


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not permitted in canonical payload")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        canonical_dict: dict[str, _JSONValue] = {}
        for key in sorted(value.keys()):
            if not isinstance(key, str):
                raise ValueError("payload keys must be strings")
            canonical_dict[key] = _canonicalize_json(value[key])
        return canonical_dict
    raise ValueError(f"unsupported payload value type: {type(value)!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class ProvenanceArtifact:
    """Immutable signed provenance artifact in an append-only chain."""

    index: int
    schema_version: int
    parent_artifact_hash: str
    originating_sovereignty_event_hash: str
    originating_privilege_decision_hash: str
    payload: Mapping[str, _JSONValue]
    artifact_hash: str
    signer_key_id: str
    artifact_signature: str


@dataclass(frozen=True)
class ProvenanceChain:
    """Immutable deterministic provenance chain."""

    artifacts: tuple[ProvenanceArtifact, ...]
    chain_root: str
    chain_schema_version: int = _CHAIN_SCHEMA_VERSION


def _artifact_hash(
    *,
    parent_artifact_hash: str,
    payload: Mapping[str, _JSONValue],
    index: int,
    schema_version: int,
    originating_sovereignty_event_hash: str,
    originating_privilege_decision_hash: str,
) -> str:
    payload_bytes = _canonical_json_bytes(payload)
    digest_input = b"|".join(
        (
            parent_artifact_hash.encode("ascii"),
            payload_bytes,
            str(index).encode("ascii"),
            str(schema_version).encode("ascii"),
            originating_sovereignty_event_hash.encode("ascii"),
            originating_privilege_decision_hash.encode("ascii"),
        )
    )
    return _sha256_hex(digest_input)


def _artifact_signature(*, signer_key_id: str, artifact_hash: str) -> str:
    return _sha256_hex(b"|".join((b"qec-provenance-signature-v1", signer_key_id.encode("utf-8"), artifact_hash.encode("ascii"))))


def _artifact_to_dict(artifact: ProvenanceArtifact) -> dict[str, Any]:
    return {
        "index": artifact.index,
        "schema_version": artifact.schema_version,
        "parent_artifact_hash": artifact.parent_artifact_hash,
        "originating_sovereignty_event_hash": artifact.originating_sovereignty_event_hash,
        "originating_privilege_decision_hash": artifact.originating_privilege_decision_hash,
        "payload": _canonicalize_json(artifact.payload),
        "artifact_hash": artifact.artifact_hash,
        "signer_key_id": artifact.signer_key_id,
        "artifact_signature": artifact.artifact_signature,
    }


def compute_provenance_root(artifacts: Sequence[ProvenanceArtifact]) -> str:
    """Compute stable chain root hash from deterministic artifact ordering."""
    if not isinstance(artifacts, Sequence):
        raise ValueError("artifacts must be a sequence")
    if len(artifacts) == 0:
        return _sha256_hex(b"provenance-empty-root")

    ordered = tuple(sorted(artifacts, key=lambda item: (int(item.index), str(item.artifact_hash))))
    digest = hashlib.sha256()
    for artifact in ordered:
        digest.update(artifact.artifact_hash.encode("ascii"))
    return digest.hexdigest()


def verify_provenance_chain(chain: ProvenanceChain) -> bool:
    """Fail-fast replay-safe verification of append-only provenance chain."""
    if not isinstance(chain, ProvenanceChain):
        raise ValueError("chain must be a ProvenanceChain")
    if chain.chain_schema_version != _CHAIN_SCHEMA_VERSION:
        raise ValueError("unsupported chain schema version")

    expected_parent = _GENESIS_PARENT_HASH
    seen_artifact_hashes: set[str] = set()

    for i, artifact in enumerate(chain.artifacts):
        if not isinstance(artifact, ProvenanceArtifact):
            raise ValueError("artifacts must contain ProvenanceArtifact entries")
        if artifact.index != i:
            raise ValueError("artifact index sequence is not append-only")
        if artifact.schema_version <= 0:
            raise ValueError("artifact schema_version must be positive")

        _validate_hash_hex(artifact.parent_artifact_hash, field_name="parent_artifact_hash")
        _validate_hash_hex(
            artifact.originating_sovereignty_event_hash,
            field_name="originating_sovereignty_event_hash",
        )
        _validate_hash_hex(
            artifact.originating_privilege_decision_hash,
            field_name="originating_privilege_decision_hash",
        )
        _validate_hash_hex(artifact.artifact_hash, field_name="artifact_hash")
        _validate_hash_hex(artifact.artifact_signature, field_name="artifact_signature")
        _validate_non_empty_str(artifact.signer_key_id, field_name="signer_key_id")

        if artifact.parent_artifact_hash != expected_parent:
            raise ValueError("artifact parent hash mismatch")

        canonical_payload = _canonicalize_json(artifact.payload)
        if not isinstance(canonical_payload, dict):
            raise ValueError("artifact payload must be an object")

        expected_hash = _artifact_hash(
            parent_artifact_hash=artifact.parent_artifact_hash,
            payload=canonical_payload,
            index=artifact.index,
            schema_version=artifact.schema_version,
            originating_sovereignty_event_hash=artifact.originating_sovereignty_event_hash,
            originating_privilege_decision_hash=artifact.originating_privilege_decision_hash,
        )
        if artifact.artifact_hash != expected_hash:
            raise ValueError("artifact hash mismatch")

        expected_signature = _artifact_signature(signer_key_id=artifact.signer_key_id, artifact_hash=artifact.artifact_hash)
        if artifact.artifact_signature != expected_signature:
            raise ValueError("artifact signature mismatch")

        if artifact.artifact_hash in seen_artifact_hashes:
            raise ValueError("replay detected: duplicate artifact hash")
        seen_artifact_hashes.add(artifact.artifact_hash)

        expected_parent = artifact.artifact_hash

    expected_root = compute_provenance_root(chain.artifacts)
    if chain.chain_root != expected_root:
        raise ValueError("chain_root mismatch")

    return True


def append_provenance_artifact(
    chain: ProvenanceChain,
    payload: Mapping[str, Any],
    *,
    originating_sovereignty_event_hash: str,
    originating_privilege_decision_hash: str,
    signer_key_id: str,
    schema_version: int = _ARTIFACT_SCHEMA_VERSION,
) -> ProvenanceChain:
    """Append immutable provenance artifact with deterministic signed lineage."""
    if not isinstance(chain, ProvenanceChain):
        raise ValueError("chain must be a ProvenanceChain")
    if schema_version <= 0:
        raise ValueError("schema_version must be positive")

    verify_provenance_chain(chain)

    canonical_payload = _canonicalize_json(payload)
    if not isinstance(canonical_payload, dict):
        raise ValueError("payload must be a mapping object")

    event_hash = _validate_hash_hex(
        _validate_non_empty_str(originating_sovereignty_event_hash, field_name="originating_sovereignty_event_hash"),
        field_name="originating_sovereignty_event_hash",
    )
    privilege_hash = _validate_hash_hex(
        _validate_non_empty_str(originating_privilege_decision_hash, field_name="originating_privilege_decision_hash"),
        field_name="originating_privilege_decision_hash",
    )
    key_id = _validate_non_empty_str(signer_key_id, field_name="signer_key_id")

    index = len(chain.artifacts)
    parent_artifact_hash = chain.artifacts[-1].artifact_hash if chain.artifacts else _GENESIS_PARENT_HASH

    artifact_hash = _artifact_hash(
        parent_artifact_hash=parent_artifact_hash,
        payload=canonical_payload,
        index=index,
        schema_version=schema_version,
        originating_sovereignty_event_hash=event_hash,
        originating_privilege_decision_hash=privilege_hash,
    )
    signature = _artifact_signature(signer_key_id=key_id, artifact_hash=artifact_hash)

    artifact = ProvenanceArtifact(
        index=index,
        schema_version=schema_version,
        parent_artifact_hash=parent_artifact_hash,
        originating_sovereignty_event_hash=event_hash,
        originating_privilege_decision_hash=privilege_hash,
        payload=canonical_payload,
        artifact_hash=artifact_hash,
        signer_key_id=key_id,
        artifact_signature=signature,
    )
    artifacts = chain.artifacts + (artifact,)
    return ProvenanceChain(artifacts=artifacts, chain_root=compute_provenance_root(artifacts))


def export_provenance_bytes(chain: ProvenanceChain) -> bytes:
    """Export canonical bytes for deterministic provenance replay."""
    verify_provenance_chain(chain)
    payload = {
        "chain_schema_version": chain.chain_schema_version,
        "chain_root": chain.chain_root,
        "artifacts": [_artifact_to_dict(artifact) for artifact in chain.artifacts],
    }
    return _canonical_json_bytes(payload)


def generate_provenance_receipt(chain: ProvenanceChain) -> dict[str, Any]:
    """Generate immutable deterministic provenance receipt."""
    canonical_bytes = export_provenance_bytes(chain)
    tip_hash = chain.artifacts[-1].artifact_hash if chain.artifacts else _GENESIS_PARENT_HASH
    return {
        "chain_schema_version": chain.chain_schema_version,
        "artifact_count": len(chain.artifacts),
        "tip_artifact_hash": tip_hash,
        "chain_root": chain.chain_root,
        "provenance_digest_sha256": _sha256_hex(canonical_bytes),
    }


__all__ = [
    "ProvenanceArtifact",
    "ProvenanceChain",
    "append_provenance_artifact",
    "verify_provenance_chain",
    "compute_provenance_root",
    "generate_provenance_receipt",
    "export_provenance_bytes",
]
