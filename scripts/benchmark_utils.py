"""Shared utilities for the benchmark harness.

Kept minimal — only add helpers here when they are needed by both
benchmark_runner and benchmark_analyze.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_run(path: Path) -> dict:
    """Load a single benchmark run JSON file."""
    return json.loads(path.read_text())
