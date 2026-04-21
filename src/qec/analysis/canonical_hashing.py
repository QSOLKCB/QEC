# SPDX-License-Identifier: MIT
"""Shared canonical JSON serialization and SHA-256 helpers for analysis modules."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | Mapping[str, "_JSONValue"]


class CanonicalHashingError(ValueError):
    """Raised when payloads cannot be represented canonically."""


def canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalHashingError("non-finite float values are not allowed")
        return float(value)
    if isinstance(value, (tuple, list)):
        return tuple(canonicalize_json(item) for item in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise CanonicalHashingError("payload keys must be strings")
        return {key: canonicalize_json(value[key]) for key in sorted(keys)}
    raise CanonicalHashingError(f"unsupported canonical payload type: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


__all__ = [
    "CanonicalHashingError",
    "canonicalize_json",
    "canonical_json",
    "canonical_bytes",
    "sha256_hex",
]
