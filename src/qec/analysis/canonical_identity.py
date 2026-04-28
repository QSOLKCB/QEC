"""Canonical hash identity validation helpers for deterministic analysis receipts."""

from __future__ import annotations

from collections.abc import Sequence
import re

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _require_sha256_hex(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or _SHA256_HEX_RE.fullmatch(value) is None:
        raise _invalid_input()
    return value


def canonical_hash_identity(hashes: Sequence[str]) -> tuple[str, ...]:
    """Validate canonical sorted-unique lowercase SHA-256 identity tuples."""

    validated = tuple(_require_sha256_hex(value) for value in hashes)
    if validated != tuple(sorted(set(validated))):
        raise _invalid_input()
    return validated


__all__ = ["canonical_hash_identity"]
