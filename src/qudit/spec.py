"""
QuditSpec — dimension specification for high-dimensional readiness.

A lightweight, frozen dataclass that captures the local dimension of a
quantum system.  The default (dimension=2, encoding="qubit") matches
standard qubit operation so all existing logic is unaffected.

Invariants
----------
* ``dimension`` is an int >= 2.
* ``encoding`` is a non-empty str.
* ``metadata`` is a dict whose canonicalized form is JSON-serializable.
* Two specs with the same fields produce byte-identical JSON via
  :meth:`to_dict` + :func:`json.dumps(sort_keys=True)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from utils.canonicalize import canonicalize as _canonicalize_meta


def _is_json_serializable(obj: Any) -> bool:
    """Return True if *obj* round-trips through JSON without error."""
    try:
        json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return True
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class QuditSpec:
    """Immutable dimension specification for a quantum system.

    Parameters
    ----------
    dimension:
        Local Hilbert-space dimension (``>= 2``).  Default is ``2``
        (qubit).
    encoding:
        Human-readable encoding label (e.g. ``"qubit"``, ``"ququart"``).
        Must be a non-empty string.
    metadata:
        Arbitrary JSON-serializable metadata.  Canonicalized on
        construction (sorted keys, numpy types converted).
    """

    dimension: int = 2
    encoding: str = "qubit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate dimension.
        if not isinstance(self.dimension, int) or self.dimension < 2:
            raise ValueError(
                f"dimension must be an int >= 2, got {self.dimension!r}"
            )
        # Validate encoding.
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError(
                f"encoding must be a non-empty str, got {self.encoding!r}"
            )
        # Canonicalize + validate metadata.
        canon = _canonicalize_meta(self.metadata)
        if not _is_json_serializable(canon):
            raise ValueError("metadata must be JSON-serializable after canonicalization")
        # frozen=True prevents normal assignment; use object.__setattr__.
        object.__setattr__(self, "metadata", canon)

    # ── Serialization helpers ──────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-Python dict with sorted keys.

        The result is safe for ``json.dumps(sort_keys=True)``.
        """
        return {
            "dimension": self.dimension,
            "encoding": self.encoding,
            "metadata": _canonicalize_meta(self.metadata),
        }

    def to_json(self) -> str:
        """Deterministic, compact JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuditSpec":
        """Construct from a plain dict (e.g. parsed from JSON config)."""
        return cls(
            dimension=data.get("dimension", 2),
            encoding=data.get("encoding", "qubit"),
            metadata=data.get("metadata", {}),
        )

    # ── Convenience ────────────────────────────────────────────────

    @staticmethod
    def qubit() -> "QuditSpec":
        """Return the default qubit spec (dimension=2)."""
        return QuditSpec()

    @property
    def is_qubit(self) -> bool:
        """True when dimension == 2."""
        return self.dimension == 2
