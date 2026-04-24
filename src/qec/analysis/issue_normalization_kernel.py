# SPDX-License-Identifier: MIT
"""Deterministic normalization of raw review issues into canonical receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

_ALLOWED_SOURCES = frozenset(
    {
        "GITHUB_REVIEW",
        "COPILOT",
        "CODEX",
        "CHATGPT_CODEX_CONNECTOR",
        "MANUAL",
        "UNKNOWN",
    }
)

_ALLOWED_CATEGORIES = frozenset(
    {
        "DETERMINISM",
        "CANONICALIZATION",
        "HASH_INTEGRITY",
        "IMMUTABILITY",
        "VALIDATION",
        "BOUNDS",
        "ORDERING",
        "NAMING",
        "DOCS",
        "TEST_COVERAGE",
        "SCOPE_GUARDRAIL",
        "UNKNOWN",
    }
)

_ALLOWED_SEVERITIES = frozenset({"LOW", "MEDIUM", "HIGH", "CRITICAL"})

_ALLOWED_INVARIANTS = frozenset(
    {
        "CANONICAL_JSON",
        "STABLE_HASH",
        "FAIL_FAST_VALIDATION",
        "FROZEN_DATACLASS",
        "BOUNDED_OUTPUT",
        "DETERMINISTIC_ORDERING",
        "DECODER_IMMUTABILITY",
        "ANALYSIS_LAYER_ONLY",
        "REPLAY_STABILITY",
        "UNKNOWN",
    }
)

_SEVERITY_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

_CATEGORY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("stable_hash", "HASH_INTEGRITY"),
    ("sha256", "HASH_INTEGRITY"),
    ("hash", "HASH_INTEGRITY"),
    ("serialization", "CANONICALIZATION"),
    ("canonical", "CANONICALIZATION"),
    ("json", "CANONICALIZATION"),
    ("duplicate", "ORDERING"),
    ("ordering", "ORDERING"),
    ("sort", "ORDERING"),
    ("overwrite", "ORDERING"),
    ("frozen", "IMMUTABILITY"),
    ("mutation", "IMMUTABILITY"),
    ("mutable", "IMMUTABILITY"),
    ("validate", "VALIDATION"),
    ("invalid", "VALIDATION"),
    ("malformed", "VALIDATION"),
    ("reject", "VALIDATION"),
    ("bounded", "BOUNDS"),
    ("score", "BOUNDS"),
    ("range", "BOUNDS"),
    ("decoder", "SCOPE_GUARDRAIL"),
    ("test", "TEST_COVERAGE"),
    ("coverage", "TEST_COVERAGE"),
    ("regression", "TEST_COVERAGE"),
)

_INVARIANT_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("stable_hash", "STABLE_HASH"),
    ("sha256", "STABLE_HASH"),
    ("hash", "STABLE_HASH"),
    ("canonical", "CANONICAL_JSON"),
    ("json", "CANONICAL_JSON"),
    ("invalid", "FAIL_FAST_VALIDATION"),
    ("reject", "FAIL_FAST_VALIDATION"),
    ("validate", "FAIL_FAST_VALIDATION"),
    ("fail", "FAIL_FAST_VALIDATION"),
    ("frozen", "FROZEN_DATACLASS"),
    ("mutable", "FROZEN_DATACLASS"),
    ("mutation", "FROZEN_DATACLASS"),
    ("bounded", "BOUNDED_OUTPUT"),
    ("range", "BOUNDED_OUTPUT"),
    ("score", "BOUNDED_OUTPUT"),
    ("ordering", "DETERMINISTIC_ORDERING"),
    ("sort", "DETERMINISTIC_ORDERING"),
    ("duplicate", "DETERMINISTIC_ORDERING"),
    ("decoder", "DECODER_IMMUTABILITY"),
    ("analysis-layer", "ANALYSIS_LAYER_ONLY"),
    ("scope", "ANALYSIS_LAYER_ONLY"),
    ("replay", "REPLAY_STABILITY"),
    ("stability", "REPLAY_STABILITY"),
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.strip().split())


def _classify_by_keywords(text: str, keywords: tuple[tuple[str, str], ...], default: str) -> str:
    lowered = text.lower()
    for keyword, output in keywords:
        if keyword in lowered:
            return output
    return default


def _normalize_target_path(path: str) -> str:
    if "\x00" in path:
        raise ValueError("target_path must not contain null bytes")
    return path.replace("\\", "/")


def _derive_summary_from_body(body: str) -> str:
    lines = tuple(line.strip() for line in body.splitlines() if line.strip())
    if not lines:
        raise ValueError("issue summary/body must produce a non-empty summary")
    first_line = lines[0]
    sentence_cut = len(first_line)
    for marker in (".", "!", "?"):
        idx = first_line.find(marker)
        if idx != -1:
            sentence_cut = min(sentence_cut, idx)
    summary = _normalize_whitespace(first_line[:sentence_cut])
    if not summary:
        summary = _normalize_whitespace(first_line)
    if not summary:
        raise ValueError("issue summary/body must produce a non-empty summary")
    return summary


@dataclass(frozen=True)
class CanonicalIssue:
    issue_id: str
    source: str
    category: str
    severity: str
    target_path: str
    summary: str
    invariant: str

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "issue_id": self.issue_id,
            "source": self.source,
            "category": self.category,
            "severity": self.severity,
            "target_path": self.target_path,
            "summary": self.summary,
            "invariant": self.invariant,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CanonicalIssueSet:
    issues: tuple[CanonicalIssue, ...]
    issue_count: int
    issue_hash: str

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "issue_count": self.issue_count,
            "issue_hash": self.issue_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IssueNormalizationReceipt:
    schema_version: str
    module_version: str
    normalization_status: str
    source_count: int
    issue_count: int
    canonical_issue_set: CanonicalIssueSet
    issue_set_hash: str

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "normalization_status": self.normalization_status,
            "source_count": self.source_count,
            "issue_count": self.issue_count,
            "canonical_issue_set": self.canonical_issue_set.to_dict(),
            "issue_set_hash": self.issue_set_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload_dict()
        payload["stable_hash"] = self.stable_hash()
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._payload_dict()).encode("utf-8")).hexdigest()


def _build_issue_id(
    source: str,
    category: str,
    severity: str,
    target_path: str,
    summary: str,
    invariant: str,
) -> str:
    payload = {
        "source": source,
        "category": category,
        "severity": severity,
        "target_path": target_path,
        "summary": summary,
        "invariant": invariant,
    }
    return f"ISSUE-{_sha256_hex(payload)[:16]}"


def _validate_allowed(value: str, allowed: frozenset[str], field_name: str) -> str:
    if value not in allowed:
        raise ValueError(f"invalid {field_name}: {value}")
    return value


def _normalize_raw_issue(raw_issue: Mapping[str, Any]) -> CanonicalIssue:
    if not isinstance(raw_issue, Mapping):
        raise ValueError("each raw issue must be a mapping")

    for required_key in ("summary", "body", "source", "severity", "category", "target_path", "invariant"):
        if required_key in raw_issue and raw_issue[required_key] is not None and not isinstance(raw_issue[required_key], str):
            raise ValueError(f"{required_key} must be a string when provided")

    source = _validate_allowed(raw_issue.get("source", "UNKNOWN") or "UNKNOWN", _ALLOWED_SOURCES, "source")
    severity = _validate_allowed(raw_issue.get("severity", "MEDIUM") or "MEDIUM", _ALLOWED_SEVERITIES, "severity")

    target_path = _normalize_target_path(raw_issue.get("target_path", "") or "")

    summary_raw = raw_issue.get("summary", "") or ""
    body_raw = raw_issue.get("body", "") or ""
    summary = _normalize_whitespace(summary_raw) if summary_raw else _derive_summary_from_body(body_raw)
    if not summary:
        raise ValueError("summary must be non-empty after normalization")

    classification_text = " ".join((summary, body_raw)).strip()

    category_raw = raw_issue.get("category")
    if category_raw:
        category = _validate_allowed(category_raw, _ALLOWED_CATEGORIES, "category")
    else:
        category = _classify_by_keywords(classification_text, _CATEGORY_KEYWORDS, "UNKNOWN")

    invariant_raw = raw_issue.get("invariant")
    if invariant_raw:
        invariant = _validate_allowed(invariant_raw, _ALLOWED_INVARIANTS, "invariant")
    else:
        invariant_text = " ".join((category, classification_text))
        invariant = _classify_by_keywords(invariant_text, _INVARIANT_KEYWORDS, "UNKNOWN")

    issue_id = _build_issue_id(
        source=source,
        category=category,
        severity=severity,
        target_path=target_path,
        summary=summary,
        invariant=invariant,
    )

    return CanonicalIssue(
        issue_id=issue_id,
        source=source,
        category=category,
        severity=severity,
        target_path=target_path,
        summary=summary,
        invariant=invariant,
    )


def _issue_sort_key(item: tuple[CanonicalIssue, str]) -> tuple[int, str, str, str, str]:
    issue, stable_hash = item
    return (
        -_SEVERITY_RANK[issue.severity],
        issue.category,
        issue.target_path,
        issue.summary,
        stable_hash,
    )


def _build_issue_set(issues: Sequence[CanonicalIssue]) -> CanonicalIssueSet:
    issue_hash_pairs = tuple((issue, issue.stable_hash()) for issue in issues)
    sorted_issue_hash_pairs = tuple(sorted(issue_hash_pairs, key=_issue_sort_key))
    sorted_issues = tuple(issue for issue, _ in sorted_issue_hash_pairs)
    hashes = tuple(stable_hash for _, stable_hash in sorted_issue_hash_pairs)
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate normalized issues are not allowed")
    issue_hash = hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()
    return CanonicalIssueSet(issues=sorted_issues, issue_count=len(sorted_issues), issue_hash=issue_hash)


def normalize_review_issues(raw_issues: Sequence[Mapping[str, Any]]) -> IssueNormalizationReceipt:
    if isinstance(raw_issues, (str, bytes, bytearray)):
        raise ValueError("raw_issues must be a sequence of mappings, not a string-like value")
    if not isinstance(raw_issues, Sequence):
        raise ValueError("raw_issues must be a sequence")

    raw_issue_items = tuple(raw_issues)
    for index, raw_issue in enumerate(raw_issue_items):
        if not isinstance(raw_issue, Mapping):
            raise ValueError(f"raw_issues[{index}] must be a mapping")

    normalized_issues = tuple(_normalize_raw_issue(raw_issue) for raw_issue in raw_issue_items)
    issue_set = _build_issue_set(normalized_issues)

    status = "EMPTY" if len(raw_issue_items) == 0 else "NORMALIZED"

    return IssueNormalizationReceipt(
        schema_version="1.0",
        module_version="v148.1",
        normalization_status=status,
        source_count=len(raw_issue_items),
        issue_count=issue_set.issue_count,
        canonical_issue_set=issue_set,
        issue_set_hash=issue_set.issue_hash,
    )


__all__ = [
    "CanonicalIssue",
    "CanonicalIssueSet",
    "IssueNormalizationReceipt",
    "normalize_review_issues",
]
