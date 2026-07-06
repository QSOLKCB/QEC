"""Canonical JSON and validation helpers for v167.0 symbolic sonification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def assert_json_safe(value: Any, field_name: str = "value") -> None:
    """Reject values that are not deterministic JSON identity data."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        raise TypeError(f"{field_name} must not contain floats")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} object keys must be strings")
            assert_json_safe(item, f"{field_name}.{key}")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            assert_json_safe(item, f"{field_name}[{index}]")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_safe(item, f"{field_name}[{index}]")
        return
    raise TypeError(f"{field_name} contains non-JSON-safe value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    assert_json_safe(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_sha256(value: Any, field_name: str = "hash") -> str:
    text = require_text(value, field_name)
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


def require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    return value


def require_nonempty_text(value: Any, field_name: str) -> str:
    text = require_text(value, field_name)
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def require_exact_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be an exact bool")
    return value


def require_int(value: Any, field_name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def require_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{field_name}[{index}] must be text")
        result.append(item)
    return tuple(result)


def sorted_unique_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    items = require_string_tuple(value, field_name)
    ordered = tuple(sorted(items))
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"{field_name} must contain unique strings")
    return ordered
