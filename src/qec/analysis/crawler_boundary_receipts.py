from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.skill_library_manifest import SkillLibraryManifest, validate_skill_library_manifest
from qec.analysis.tool_dispatch_telemetry_receipts import (
    ToolDispatchTelemetryReceipt,
    validate_tool_dispatch_telemetry_receipt,
)

_SCHEMA_VERSION = "CRAWLER_BOUNDARY_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_SCOPE_ENTRIES = 4096

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_CRAWLER_TYPES = {
    "DECLARED_AUDIT_CRAWLER",
    "DECLARED_CONTEXT_CRAWLER",
    "DECLARED_REPLAY_CRAWLER",
    "DECLARED_OBSERVATION_CRAWLER",
    "DECLARED_RESEARCH_CRAWLER",
    "DECLARED_CUSTOM_CRAWLER",
}
_ALLOWED_CRAWL_SCOPE_MODES = {
    "NO_NETWORK_SCOPE",
    "DECLARED_STATIC_SCOPE",
    "DECLARED_CONTEXT_SCOPE",
    "DECLARED_AUDIT_SCOPE",
    "DECLARED_CUSTOM_SCOPE",
}
_ALLOWED_CRAWL_PERMISSION_MODES = {
    "NETWORK_DISABLED",
    "AUDIT_ONLY_PERMISSION",
    "REPLAY_ONLY_PERMISSION",
    "CONTEXT_ONLY_PERMISSION",
    "DECLARED_CUSTOM_PERMISSION",
}
_ALLOWED_CRAWL_REPLAY_MODES = {
    "REPLAY_SAFE_CRAWL",
    "CONTEXTUAL_CRAWL",
    "NON_REPLAY_SAFE_CRAWL",
    "RESEARCH_CRAWL_ONLY",
    "DECLARED_CUSTOM_REPLAY",
}
_ALLOWED_CRAWL_AUDIT_MODES = {
    "STRICT_AUDIT_BOUNDARY",
    "REPLAY_AUDIT_BOUNDARY",
    "CONTEXT_AUDIT_BOUNDARY",
    "DECLARED_CUSTOM_AUDIT",
}

_REPLAY_SAFE_SCOPE_MODES = {"NO_NETWORK_SCOPE", "DECLARED_STATIC_SCOPE", "DECLARED_AUDIT_SCOPE"}
_REPLAY_SAFE_PERMISSION_MODES = {"NETWORK_DISABLED", "AUDIT_ONLY_PERMISSION", "REPLAY_ONLY_PERMISSION"}
_REPLAY_SAFE_AUDIT_MODES = {"STRICT_AUDIT_BOUNDARY", "REPLAY_AUDIT_BOUNDARY"}

_FORBIDDEN_TOKENS = (
    "live crawler",
    "network request succeeded",
    "runtime dispatch",
    "autonomous crawling",
    "agent output is evidence",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "internet access confirmed",
    "autonomous evaluation",
    "hidden network semantics",
    "hidden crawler execution",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable crawler",
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_text(value: str, field_name: str, max_len: int) -> None:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ValueError(f"{field_name} must be non-empty and bounded")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower().replace("_", " ").replace("-", " ")
    for token in _FORBIDDEN_TOKENS:
        if token.lower().replace("_", " ").replace("-", " ") in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_crawler_boundary_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    cls(**value.__dict__)


@dataclass(frozen=True)
class CrawlerIdentity:
    crawler_name: str
    crawler_version: str
    crawler_type: str
    crawler_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CrawlerIdentity:
            raise _invalid_input()
        _check_text(self.crawler_name, "crawler_name", _MAX_NAME_LENGTH)
        _check_text(self.crawler_version, "crawler_version", _MAX_NAME_LENGTH)
        if self.crawler_type not in _ALLOWED_CRAWLER_TYPES:
            raise ValueError("invalid crawler_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.crawler_identity_hash, "crawler_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "crawler_identity_hash")) != self.crawler_identity_hash:
            raise ValueError("crawler_identity_hash mismatch")


@dataclass(frozen=True)
class CrawlScopeBoundary:
    scope_mode: str
    scope_reason: str
    crawl_scope_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CrawlScopeBoundary:
            raise _invalid_input()
        if self.scope_mode not in _ALLOWED_CRAWL_SCOPE_MODES:
            raise ValueError("invalid scope_mode")
        _check_text(self.scope_reason, "scope_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.crawl_scope_boundary_hash, "crawl_scope_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "crawl_scope_boundary_hash")) != self.crawl_scope_boundary_hash:
            raise ValueError("crawl_scope_boundary_hash mismatch")


@dataclass(frozen=True)
class CrawlPermissionBoundary:
    permission_mode: str
    permission_reason: str
    crawl_permission_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CrawlPermissionBoundary:
            raise _invalid_input()
        if self.permission_mode not in _ALLOWED_CRAWL_PERMISSION_MODES:
            raise ValueError("invalid permission_mode")
        _check_text(self.permission_reason, "permission_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.crawl_permission_boundary_hash, "crawl_permission_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "crawl_permission_boundary_hash")) != self.crawl_permission_boundary_hash:
            raise ValueError("crawl_permission_boundary_hash mismatch")


@dataclass(frozen=True)
class CrawlReplayBoundary:
    replay_mode: str
    replay_reason: str
    crawl_replay_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CrawlReplayBoundary:
            raise _invalid_input()
        if self.replay_mode not in _ALLOWED_CRAWL_REPLAY_MODES:
            raise ValueError("invalid replay_mode")
        _check_text(self.replay_reason, "replay_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.crawl_replay_boundary_hash, "crawl_replay_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "crawl_replay_boundary_hash")) != self.crawl_replay_boundary_hash:
            raise ValueError("crawl_replay_boundary_hash mismatch")


@dataclass(frozen=True)
class CrawlAuditBoundary:
    audit_mode: str
    audit_reason: str
    crawl_audit_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CrawlAuditBoundary:
            raise _invalid_input()
        if self.audit_mode not in _ALLOWED_CRAWL_AUDIT_MODES:
            raise ValueError("invalid audit_mode")
        _check_text(self.audit_reason, "audit_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.crawl_audit_boundary_hash, "crawl_audit_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "crawl_audit_boundary_hash")) != self.crawl_audit_boundary_hash:
            raise ValueError("crawl_audit_boundary_hash mismatch")


@dataclass(frozen=True)
class CrawlerBoundaryReceipt:
    schema_version: str
    tool_dispatch_telemetry_receipt_hash: str
    skill_library_manifest_hash: str
    crawler_identity: CrawlerIdentity
    crawl_scope_boundary: CrawlScopeBoundary
    crawl_permission_boundary: CrawlPermissionBoundary
    crawl_replay_boundary: CrawlReplayBoundary
    crawl_audit_boundary: CrawlAuditBoundary
    replay_safe_crawler: bool
    adapter_only: bool
    crawler_boundary_receipt_hash: str


def build_crawler_identity(crawler_name: str, crawler_version: str, crawler_type: str) -> CrawlerIdentity:
    payload = {"crawler_name": crawler_name, "crawler_version": crawler_version, "crawler_type": crawler_type}
    return CrawlerIdentity(**payload, crawler_identity_hash=_hash_payload(payload))


def build_crawl_scope_boundary(scope_mode: str, scope_reason: str) -> CrawlScopeBoundary:
    payload = {"scope_mode": scope_mode, "scope_reason": scope_reason}
    return CrawlScopeBoundary(**payload, crawl_scope_boundary_hash=_hash_payload(payload))


def build_crawl_permission_boundary(permission_mode: str, permission_reason: str) -> CrawlPermissionBoundary:
    payload = {"permission_mode": permission_mode, "permission_reason": permission_reason}
    return CrawlPermissionBoundary(**payload, crawl_permission_boundary_hash=_hash_payload(payload))


def build_crawl_replay_boundary(replay_mode: str, replay_reason: str) -> CrawlReplayBoundary:
    payload = {"replay_mode": replay_mode, "replay_reason": replay_reason}
    return CrawlReplayBoundary(**payload, crawl_replay_boundary_hash=_hash_payload(payload))


def build_crawl_audit_boundary(audit_mode: str, audit_reason: str) -> CrawlAuditBoundary:
    payload = {"audit_mode": audit_mode, "audit_reason": audit_reason}
    return CrawlAuditBoundary(**payload, crawl_audit_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_crawler(
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    skill_library_manifest: SkillLibraryManifest,
    crawler_identity: CrawlerIdentity,
    crawl_scope_boundary: CrawlScopeBoundary,
    crawl_permission_boundary: CrawlPermissionBoundary,
    crawl_replay_boundary: CrawlReplayBoundary,
    crawl_audit_boundary: CrawlAuditBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and tool_dispatch_telemetry_receipt.replay_safe_dispatch is True
        and skill_library_manifest.replay_safe_skill_library is True
        and crawler_identity.crawler_type != "DECLARED_CUSTOM_CRAWLER"
        and crawl_scope_boundary.scope_mode in _REPLAY_SAFE_SCOPE_MODES
        and crawl_permission_boundary.permission_mode in _REPLAY_SAFE_PERMISSION_MODES
        and crawl_replay_boundary.replay_mode == "REPLAY_SAFE_CRAWL"
        and crawl_audit_boundary.audit_mode in _REPLAY_SAFE_AUDIT_MODES
    )


def build_crawler_boundary_receipt(
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    skill_library_manifest: SkillLibraryManifest,
    crawler_identity: CrawlerIdentity,
    crawl_scope_boundary: CrawlScopeBoundary,
    crawl_permission_boundary: CrawlPermissionBoundary,
    crawl_replay_boundary: CrawlReplayBoundary,
    crawl_audit_boundary: CrawlAuditBoundary,
    adapter_only: bool,
) -> CrawlerBoundaryReceipt:
    replay_safe_crawler = _recompute_replay_safe_crawler(
        tool_dispatch_telemetry_receipt,
        skill_library_manifest,
        crawler_identity,
        crawl_scope_boundary,
        crawl_permission_boundary,
        crawl_replay_boundary,
        crawl_audit_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "tool_dispatch_telemetry_receipt_hash": tool_dispatch_telemetry_receipt.tool_dispatch_telemetry_receipt_hash,
        "skill_library_manifest_hash": skill_library_manifest.skill_library_manifest_hash,
        "crawler_identity": crawler_identity,
        "crawl_scope_boundary": crawl_scope_boundary,
        "crawl_permission_boundary": crawl_permission_boundary,
        "crawl_replay_boundary": crawl_replay_boundary,
        "crawl_audit_boundary": crawl_audit_boundary,
        "replay_safe_crawler": replay_safe_crawler,
        "adapter_only": adapter_only,
    }
    return CrawlerBoundaryReceipt(**payload, crawler_boundary_receipt_hash=_hash_payload(payload))


def validate_crawler_identity(value: CrawlerIdentity) -> CrawlerIdentity:
    _revalidate_exact_instance(value, CrawlerIdentity)
    return value


def validate_crawl_scope_boundary(value: CrawlScopeBoundary) -> CrawlScopeBoundary:
    _revalidate_exact_instance(value, CrawlScopeBoundary)
    return value


def validate_crawl_permission_boundary(value: CrawlPermissionBoundary) -> CrawlPermissionBoundary:
    _revalidate_exact_instance(value, CrawlPermissionBoundary)
    return value


def validate_crawl_replay_boundary(value: CrawlReplayBoundary) -> CrawlReplayBoundary:
    _revalidate_exact_instance(value, CrawlReplayBoundary)
    return value


def validate_crawl_audit_boundary(value: CrawlAuditBoundary) -> CrawlAuditBoundary:
    _revalidate_exact_instance(value, CrawlAuditBoundary)
    return value


def validate_crawler_boundary_receipt(
    receipt: CrawlerBoundaryReceipt,
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    skill_library_manifest: SkillLibraryManifest,
    **upstream_kwargs: Any,
) -> CrawlerBoundaryReceipt:
    _revalidate_exact_instance(receipt, CrawlerBoundaryReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if isinstance(receipt.replay_safe_crawler, bool) is False:
        raise ValueError("replay_safe_crawler must be bool")

    validate_crawler_identity(receipt.crawler_identity)
    validate_crawl_scope_boundary(receipt.crawl_scope_boundary)
    validate_crawl_permission_boundary(receipt.crawl_permission_boundary)
    validate_crawl_replay_boundary(receipt.crawl_replay_boundary)
    validate_crawl_audit_boundary(receipt.crawl_audit_boundary)

    validate_tool_dispatch_telemetry_receipt(tool_dispatch_telemetry_receipt, skill_library_manifest, **upstream_kwargs)
    validate_skill_library_manifest(skill_library_manifest, **upstream_kwargs)

    if receipt.tool_dispatch_telemetry_receipt_hash != tool_dispatch_telemetry_receipt.tool_dispatch_telemetry_receipt_hash:
        raise ValueError("tool_dispatch_telemetry_receipt_hash mismatch")
    if receipt.skill_library_manifest_hash != skill_library_manifest.skill_library_manifest_hash:
        raise ValueError("skill_library_manifest_hash mismatch")

    _validate_crawler_boundary_semantics(
        receipt.crawl_scope_boundary.scope_reason,
        receipt.crawl_permission_boundary.permission_reason,
        receipt.crawl_replay_boundary.replay_reason,
        receipt.crawl_audit_boundary.audit_reason,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    recomputed = _recompute_replay_safe_crawler(
        tool_dispatch_telemetry_receipt,
        skill_library_manifest,
        receipt.crawler_identity,
        receipt.crawl_scope_boundary,
        receipt.crawl_permission_boundary,
        receipt.crawl_replay_boundary,
        receipt.crawl_audit_boundary,
        receipt.adapter_only,
    )
    if receipt.replay_safe_crawler != recomputed:
        raise ValueError("replay_safe_crawler must be recomputed")

    _validate_hash_format(receipt.crawler_boundary_receipt_hash, "crawler_boundary_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "crawler_boundary_receipt_hash")) != receipt.crawler_boundary_receipt_hash:
        raise ValueError("crawler_boundary_receipt_hash mismatch")
    return receipt
