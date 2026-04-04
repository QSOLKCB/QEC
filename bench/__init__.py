"""Compatibility package exposing modules from src/bench at top-level `bench`."""

from __future__ import annotations

import os
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_bench = os.path.join(_repo_root, "src", "bench")
if os.path.isdir(_src_bench) and _src_bench not in __path__:
    __path__.append(_src_bench)
