"""
Canonical JSON writer and artifact hash computation for interop records.

Ensures byte-identical output for identical logical content by enforcing
sorted keys, compact separators, and numpy-to-Python canonicalization.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from utils.canonicalize import canonicalize


def canonical_json(obj: Any) -> str:
    """Serialize *obj* to a canonical JSON string.

    Uses ``sort_keys=True`` and ``separators=(",", ":")`` for
    byte-identical output.  All numpy types are converted to Python.
    """
    clean = canonicalize(obj)
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def artifact_hash(obj: Any) -> str:
    """Compute the SHA-256 hash of the canonical JSON representation.

    This hash is stable: same logical content always produces the same hash.
    """
    payload = canonical_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_hash(config_dict: dict[str, Any]) -> str:
    """Compute a stable hash of a benchmark configuration dict.

    Used as ``stable_sweep_hash`` in result records.
    """
    clean = canonicalize(config_dict)
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
