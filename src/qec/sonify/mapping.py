"""Mapping schema definitions for v167.0 symbolic sonification."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

from .canonical import canonical_sha256, require_exact_bool, require_nonempty_text, require_text, sorted_unique_string_tuple, validate_sha256
from .events import ALLOWED_EVENT_TYPES

SCHEMA_VERSION = "v167.0"
SCHEMA_KIND = "SYMBOLIC_SONIFICATION_MAPPING_SCHEMA"
MAPPING_POLICY = "SYMBOLIC_TOKENS_TO_EVENT_STREAM_SCHEMA_ONLY"
REQUIRED_SYMBOL_FIELDS = ("authority_allowed", "claim_scope", "creative_status", "symbol_id", "symbol_label", "symbolic_role")
REQUIRED_EVENT_FIELDS = ("authority_allowed", "claim_scope", "creative_status", "duration_ticks", "event_hash", "event_id", "event_type", "lane", "parameters", "start_tick", "symbolic_token", "tags")

@dataclass(frozen=True)
class SymbolicMappingSchema:
    schema_id: str
    schema_version: str
    schema_kind: str
    required_symbol_fields: tuple[str, ...]
    required_event_fields: tuple[str, ...]
    allowed_event_types: tuple[str, ...]
    mapping_policy: str
    creative_status_required: bool
    authority_allowed: bool
    schema_hash: str


def build_symbolic_mapping_schema(schema_id: str, required_symbol_fields: Iterable[str] = REQUIRED_SYMBOL_FIELDS, required_event_fields: Iterable[str] = REQUIRED_EVENT_FIELDS, allowed_event_types: Iterable[str] = ALLOWED_EVENT_TYPES, schema_version: str = SCHEMA_VERSION, schema_kind: str = SCHEMA_KIND, mapping_policy: str = MAPPING_POLICY, creative_status_required: bool = True, authority_allowed: bool = False) -> SymbolicMappingSchema:
    payload = {
        "schema_id": require_nonempty_text(schema_id, "schema_id"),
        "schema_version": require_text(schema_version, "schema_version"),
        "schema_kind": require_text(schema_kind, "schema_kind"),
        "required_symbol_fields": sorted_unique_string_tuple(tuple(required_symbol_fields), "required_symbol_fields"),
        "required_event_fields": sorted_unique_string_tuple(tuple(required_event_fields), "required_event_fields"),
        "allowed_event_types": sorted_unique_string_tuple(tuple(allowed_event_types), "allowed_event_types"),
        "mapping_policy": require_text(mapping_policy, "mapping_policy"),
        "creative_status_required": require_exact_bool(creative_status_required, "creative_status_required"),
        "authority_allowed": require_exact_bool(authority_allowed, "authority_allowed"),
    }
    return validate_symbolic_mapping_schema(SymbolicMappingSchema(**payload, schema_hash=canonical_sha256(payload)))


def default_symbolic_mapping_schema() -> SymbolicMappingSchema:
    return build_symbolic_mapping_schema("symbolic_mapping_schema_v167_0")


def validate_symbolic_mapping_schema(schema: SymbolicMappingSchema) -> SymbolicMappingSchema:
    if not isinstance(schema, SymbolicMappingSchema):
        raise TypeError("schema must be SymbolicMappingSchema")
    if schema.schema_version != SCHEMA_VERSION or schema.schema_kind != SCHEMA_KIND or schema.mapping_policy != MAPPING_POLICY:
        raise ValueError("mapping schema metadata is invalid")
    if schema.creative_status_required is not True or schema.authority_allowed is not False:
        raise ValueError("mapping schema claim metadata is invalid")
    if set(REQUIRED_SYMBOL_FIELDS) - set(schema.required_symbol_fields):
        raise ValueError("mapping schema is missing required symbol fields")
    if set(REQUIRED_EVENT_FIELDS) - set(schema.required_event_fields):
        raise ValueError("mapping schema is missing required event fields")
    if tuple(schema.allowed_event_types) != tuple(sorted(ALLOWED_EVENT_TYPES)):
        raise ValueError("mapping schema event types do not match v167.0")
    if validate_sha256(schema.schema_hash, "schema_hash") != symbolic_mapping_schema_hash(schema):
        raise ValueError("schema_hash does not match canonical mapping schema payload")
    return schema


def symbolic_mapping_schema_payload(schema: SymbolicMappingSchema | Mapping[str, Any], *, include_hash: bool = True) -> dict[str, Any]:
    if isinstance(schema, Mapping):
        source = schema
    else:
        source = {
            "schema_id": schema.schema_id,
            "schema_version": schema.schema_version,
            "schema_kind": schema.schema_kind,
            "required_symbol_fields": schema.required_symbol_fields,
            "required_event_fields": schema.required_event_fields,
            "allowed_event_types": schema.allowed_event_types,
            "mapping_policy": schema.mapping_policy,
            "creative_status_required": schema.creative_status_required,
            "authority_allowed": schema.authority_allowed,
            "schema_hash": schema.schema_hash,
        }
    payload = {
        "schema_id": require_nonempty_text(source["schema_id"], "schema_id"),
        "schema_version": require_text(source["schema_version"], "schema_version"),
        "schema_kind": require_text(source["schema_kind"], "schema_kind"),
        "required_symbol_fields": list(sorted_unique_string_tuple(source["required_symbol_fields"], "required_symbol_fields")),
        "required_event_fields": list(sorted_unique_string_tuple(source["required_event_fields"], "required_event_fields")),
        "allowed_event_types": list(sorted_unique_string_tuple(source["allowed_event_types"], "allowed_event_types")),
        "mapping_policy": require_text(source["mapping_policy"], "mapping_policy"),
        "creative_status_required": require_exact_bool(source["creative_status_required"], "creative_status_required"),
        "authority_allowed": require_exact_bool(source["authority_allowed"], "authority_allowed"),
    }
    if include_hash:
        payload["schema_hash"] = validate_sha256(source["schema_hash"], "schema_hash")
    return payload


def symbolic_mapping_schema_hash(schema_or_payload: Any) -> str:
    return canonical_sha256(symbolic_mapping_schema_payload(schema_or_payload, include_hash=False))
