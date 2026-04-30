from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.canonicalization_engine import CanonicalDocument

_VERSION = "v151.2"
_STATUS = "SEMANTIC_FIELD_CONSTRUCTED"
_ALLOWED_CONSTRAINT_TYPES = {
    "CANONICAL_DOCUMENT_HASH",
    "CANONICAL_SCHEMA_HASH",
    "CANONICAL_LOCALE_HASH",
    "CANONICAL_EXTRACTION_HASH",
    "FIELD_SET_HASH",
}


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _sha256_hex(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _is_finite_json_number(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    return True


def _validate_json_safety(value: Any) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str) or k == "":
                raise _invalid()
            _validate_json_safety(v)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_safety(item)
        return
    if not _is_finite_json_number(value):
        raise _invalid()
    _canon(value)


def _governance_context_payload(*, context_id: str, schema_version: str, context_payload: Mapping[str, Any], allowed_keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        "context_id": context_id,
        "schema_version": schema_version,
        "context_payload": context_payload,
        "allowed_keys": allowed_keys,
    }


def _canon(value: Any) -> Any:
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _non_empty_string(value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise _invalid()
    return value


def _evidence_payload(field_name: str, canonical_value: Any) -> dict[str, Any]:
    return {"field_name": field_name, "canonical_value": canonical_value}


def _source_constraint_payload(constraint_type: str, constraint_value: Any) -> dict[str, Any]:
    return {"constraint_type": constraint_type, "constraint_value": constraint_value}


def _res_payload(
    version: str,
    canonical_document_hash: str,
    grounded_field_hash: str,
    evidence_fields: tuple["EvidenceField", ...],
    source_constraints: tuple["SourceConstraint", ...],
) -> dict[str, Any]:
    return {
        "version": version,
        "canonical_document_hash": canonical_document_hash,
        "grounded_field_hash": grounded_field_hash,
        "evidence_fields": tuple(e.to_dict() for e in evidence_fields),
        "source_constraints": tuple(c.to_dict() for c in source_constraints),
    }


def _rag_payload(
    version: str,
    canonical_document_hash: str,
    interpretation_hash: str,
    generated_claims: tuple["GeneratedClaim", ...],
    governance_context_hash: str,
) -> dict[str, Any]:
    return {
        "version": version,
        "canonical_document_hash": canonical_document_hash,
        "interpretation_hash": interpretation_hash,
        "generated_claims": tuple(c.to_dict() for c in generated_claims),
        "governance_context_hash": governance_context_hash,
    }


def _interpretation_payload(generated_claims: tuple["GeneratedClaim", ...], governance_context_hash: str) -> dict[str, Any]:
    return {"generated_claims": tuple(c.to_dict() for c in generated_claims), "governance_context_hash": governance_context_hash}


@dataclass(frozen=True)
class EvidenceField:
    """Grounded evidence from one top-level canonical payload key.

    Nested structures are preserved in canonical_value and are never flattened.
    """
    field_name: str
    canonical_value: Any
    value_hash: str

    def __post_init__(self) -> None:
        _non_empty_string(self.field_name)
        object.__setattr__(self, "canonical_value", _canon(self.canonical_value))
        if self.computed_stable_hash() != self.value_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {"field_name": self.field_name, "canonical_value": self.canonical_value, "value_hash": self.value_hash}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex(_evidence_payload(self.field_name, self.canonical_value))


@dataclass(frozen=True)
class SourceConstraint:
    constraint_type: str
    constraint_value: Any
    constraint_hash: str

    def __post_init__(self) -> None:
        if self.constraint_type not in _ALLOWED_CONSTRAINT_TYPES:
            raise _invalid()
        object.__setattr__(self, "constraint_value", _canon(self.constraint_value))
        if self.computed_stable_hash() != self.constraint_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {"constraint_type": self.constraint_type, "constraint_value": self.constraint_value, "constraint_hash": self.constraint_hash}

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex(_source_constraint_payload(self.constraint_type, self.constraint_value))


@dataclass(frozen=True)
class GeneratedClaim:
    """Untrusted generated claim.

    claim_payload is canonicalized for hashing/JSON safety only; semantic
    validation is explicitly out of scope for this layer.
    """
    claim_id: str
    claim_text: str
    claim_payload: Any
    claim_hash: str

    def __post_init__(self) -> None:
        _non_empty_string(self.claim_id)
        _non_empty_string(self.claim_text)
        object.__setattr__(self, "claim_payload", _canon(self.claim_payload))
        if self.computed_stable_hash() != self.claim_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "claim_payload": self.claim_payload,
            "claim_hash": self.claim_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({"claim_id": self.claim_id, "claim_text": self.claim_text, "claim_payload": self.claim_payload})


@dataclass(frozen=True)
class GovernanceContext:
    """Deterministic governance context identity (no governance execution)."""

    context_id: str
    context_payload: Mapping[str, Any]
    governance_context_hash: str
    schema_version: str = "v151.3"
    allowed_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _non_empty_string(self.context_id)
        _non_empty_string(self.schema_version)
        if not isinstance(self.context_payload, Mapping):
            raise _invalid()
        payload = _canon(dict(self.context_payload))
        if not isinstance(payload, Mapping):
            raise _invalid()
        _validate_json_safety(payload)
        object.__setattr__(self, "context_payload", payload)
        if self.allowed_keys is None:
            allowed_keys = tuple(sorted(payload.keys()))
        else:
            allowed_keys = self.allowed_keys
        if not isinstance(allowed_keys, tuple) or not allowed_keys:
            raise _invalid()
        if any((not isinstance(k, str) or k == "") for k in allowed_keys):
            raise _invalid()
        if len(set(allowed_keys)) != len(allowed_keys):
            raise _invalid()
        if not set(payload.keys()).issubset(set(allowed_keys)):
            raise _invalid()
        object.__setattr__(self, "allowed_keys", allowed_keys)
        legacy_hash = _sha256_hex({"context_id": self.context_id, "context_payload": self.context_payload})
        if self.computed_stable_hash() != self.governance_context_hash and legacy_hash != self.governance_context_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "schema_version": self.schema_version,
            "context_payload": self.context_payload,
            "allowed_keys": self.allowed_keys,
            "governance_context_hash": self.governance_context_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex(_governance_context_payload(
            context_id=self.context_id,
            schema_version=self.schema_version,
            context_payload=self.context_payload,
            allowed_keys=self.allowed_keys if self.allowed_keys is not None else tuple(sorted(self.context_payload.keys())),
        ))


@dataclass(frozen=True)
class RESState:
    version: str
    canonical_document_hash: str
    grounded_field_hash: str
    evidence_fields: tuple[EvidenceField, ...]
    source_constraints: tuple[SourceConstraint, ...]
    res_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION or not self.evidence_fields or not self.source_constraints:
            raise _invalid()
        if tuple(sorted(self.evidence_fields, key=lambda e: (e.field_name, e.value_hash))) != self.evidence_fields:
            raise _invalid()
        if tuple(sorted(self.source_constraints, key=lambda c: (c.constraint_type, c.constraint_hash))) != self.source_constraints:
            raise _invalid()
        if self.grounded_field_hash != _sha256_hex(tuple(e.to_dict() for e in self.evidence_fields)):
            raise _invalid()
        if self.computed_stable_hash() != self.res_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canonical_document_hash": self.canonical_document_hash,
            "grounded_field_hash": self.grounded_field_hash,
            "evidence_fields": tuple(e.to_dict() for e in self.evidence_fields),
            "source_constraints": tuple(c.to_dict() for c in self.source_constraints),
            "res_hash": self.res_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "res_hash"})


@dataclass(frozen=True)
class RAGState:
    version: str
    canonical_document_hash: str
    interpretation_hash: str
    generated_claims: tuple[GeneratedClaim, ...]
    governance_context_hash: str
    rag_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION:
            raise _invalid()
        if tuple(sorted(self.generated_claims, key=lambda c: (c.claim_id, c.claim_hash))) != self.generated_claims:
            raise _invalid()
        if len({claim.claim_id for claim in self.generated_claims}) != len(self.generated_claims):
            raise _invalid()
        if self.interpretation_hash != _sha256_hex(_interpretation_payload(self.generated_claims, self.governance_context_hash)):
            raise _invalid()
        if self.computed_stable_hash() != self.rag_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canonical_document_hash": self.canonical_document_hash,
            "interpretation_hash": self.interpretation_hash,
            "generated_claims": tuple(c.to_dict() for c in self.generated_claims),
            "governance_context_hash": self.governance_context_hash,
            "rag_hash": self.rag_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex(
            _rag_payload(
                self.version,
                self.canonical_document_hash,
                self.interpretation_hash,
                self.generated_claims,
                self.governance_context_hash,
            )
        )


@dataclass(frozen=True)
class SemanticFieldReceipt:
    version: str
    canonical_hash: str
    res_hash: str
    rag_hash: str
    semantic_field_hash: str
    status: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status != _STATUS:
            raise _invalid()
        if self.semantic_field_hash != _sha256_hex({"canonical_hash": self.canonical_hash, "res_hash": self.res_hash, "rag_hash": self.rag_hash}):
            raise _invalid()
        if self.computed_stable_hash() != self.stable_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "canonical_hash": self.canonical_hash,
            "res_hash": self.res_hash,
            "rag_hash": self.rag_hash,
            "semantic_field_hash": self.semantic_field_hash,
            "status": self.status,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def run_res_rag_semantic_field(canonical_document: CanonicalDocument, generated_claims: Sequence[GeneratedClaim], governance_context: GovernanceContext) -> SemanticFieldReceipt:
    """Construct deterministic RES/RAG semantic field receipt.

    EvidenceField instances are derived only from top-level
    canonical_document.canonical_payload keys. Nested values remain embedded in
    canonical_value.

    FIELD_SET_HASH = SHA256(sorted tuple of evidence field names).

    Validates CanonicalDocument hash recomputation integrity before derivation.
    """
    if not isinstance(canonical_document, CanonicalDocument) or not isinstance(governance_context, GovernanceContext) or not isinstance(generated_claims, Sequence):
        raise _invalid()
    if canonical_document.canonical_hash != _sha256_hex(canonical_document.canonical_payload):
        raise _invalid()

    evidence = tuple(
        sorted(
            (
                EvidenceField(field_name=k, canonical_value=v, value_hash=_sha256_hex(_evidence_payload(k, v)))
                for k, v in canonical_document.canonical_payload.items()
            ),
            key=lambda e: (e.field_name, e.value_hash),
        )
    )
    if not evidence:
        raise _invalid()
    field_set_hash = _sha256_hex(tuple(sorted(e.field_name for e in evidence)))
    constraints = tuple(
        sorted(
            [
                SourceConstraint("CANONICAL_DOCUMENT_HASH", canonical_document.canonical_hash, _sha256_hex(_source_constraint_payload("CANONICAL_DOCUMENT_HASH", canonical_document.canonical_hash))),
                SourceConstraint("CANONICAL_SCHEMA_HASH", canonical_document.schema_hash, _sha256_hex(_source_constraint_payload("CANONICAL_SCHEMA_HASH", canonical_document.schema_hash))),
                SourceConstraint("CANONICAL_LOCALE_HASH", canonical_document.locale_hash, _sha256_hex(_source_constraint_payload("CANONICAL_LOCALE_HASH", canonical_document.locale_hash))),
                SourceConstraint("CANONICAL_EXTRACTION_HASH", canonical_document.extraction_hash, _sha256_hex(_source_constraint_payload("CANONICAL_EXTRACTION_HASH", canonical_document.extraction_hash))),
                SourceConstraint("FIELD_SET_HASH", field_set_hash, _sha256_hex(_source_constraint_payload("FIELD_SET_HASH", field_set_hash))),
            ],
            key=lambda c: (c.constraint_type, c.constraint_hash),
        )
    )
    grounded_field_hash = _sha256_hex(tuple(e.to_dict() for e in evidence))
    res_hash = _sha256_hex(_res_payload(_VERSION, canonical_document.canonical_hash, grounded_field_hash, evidence, constraints))
    res = RESState(_VERSION, canonical_document.canonical_hash, grounded_field_hash, evidence, constraints, res_hash)

    if any(not isinstance(claim, GeneratedClaim) for claim in generated_claims):
        raise _invalid()
    sorted_claims = tuple(sorted(generated_claims, key=lambda c: (c.claim_id, c.claim_hash)))
    for i in range(1, len(sorted_claims)):
        if sorted_claims[i - 1].claim_id == sorted_claims[i].claim_id:
            raise _invalid()
    interp_hash = _sha256_hex(_interpretation_payload(sorted_claims, governance_context.governance_context_hash))
    rag_hash = _sha256_hex(_rag_payload(_VERSION, canonical_document.canonical_hash, interp_hash, sorted_claims, governance_context.governance_context_hash))
    rag = RAGState(_VERSION, canonical_document.canonical_hash, interp_hash, sorted_claims, governance_context.governance_context_hash, rag_hash)

    semantic_field_hash = _sha256_hex({"canonical_hash": canonical_document.canonical_hash, "res_hash": res.res_hash, "rag_hash": rag.rag_hash})
    stable_hash = _sha256_hex({"version": _VERSION, "canonical_hash": canonical_document.canonical_hash, "res_hash": res.res_hash, "rag_hash": rag.rag_hash, "semantic_field_hash": semantic_field_hash, "status": _STATUS})
    receipt = SemanticFieldReceipt(_VERSION, canonical_document.canonical_hash, res.res_hash, rag.rag_hash, semantic_field_hash, _STATUS, stable_hash)

    if receipt.computed_stable_hash() != receipt.stable_hash:
        raise _invalid()
    return receipt
