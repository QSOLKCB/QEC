"""
Shared canonicalization utility.

Converts nested Python/numpy structures into deterministic, JSON-safe
plain-Python objects with sorted dict keys.  Used by both the benchmark
schema layer and the qudit spec layer to avoid logic duplication.
"""

from __future__ import annotations

import copy
from typing import Any


def canonicalize(obj: Any) -> Any:
    """Deep-copy *obj* converting numpy types to plain Python.

    * ``numpy.integer``  → ``int``
    * ``numpy.floating`` → ``float``
    * ``numpy.bool_``    → ``bool``
    * ``numpy.ndarray``  → ``list`` (recursively)
    * ``tuple``          → ``list``
    * ``dict``           → ``dict`` with sorted keys (recursive)
    * signed floating zero → canonical ``0.0``

    The result is safe for :func:`json.dumps` and normalizes ``-0.0`` to
    ``0.0`` so numerically identical zero values have one deterministic JSON
    representation.
    """
    try:
        import numpy as np
        _has_numpy = True
    except ImportError:  # pragma: no cover
        _has_numpy = False

    def _canonical_float(value: float) -> float:
        return 0.0 if value == 0.0 else value

    def _convert(v: Any) -> Any:
        if _has_numpy:
            if isinstance(v, np.ndarray):
                return [_convert(x) for x in v.tolist()]
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return _canonical_float(float(v))
        if isinstance(v, float):
            return _canonical_float(v)
        if isinstance(v, dict):
            return {str(k): _convert(val) for k, val in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        return v

    return _convert(copy.deepcopy(obj))
