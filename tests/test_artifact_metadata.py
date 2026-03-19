"""
Tests for the v10.0.0 artifact metadata stamp.

Verifies:
  - metadata contains all required fields
  - timestamp is ISO 8601
  - versions are strings
  - seed is correctly recorded
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.utils.artifact_metadata import generate_run_metadata


class TestArtifactMetadata:
    """Tests for generate_run_metadata."""

    def test_required_keys(self):
        meta = generate_run_metadata(seed=42)
        assert "repo_version" in meta
        assert "numpy_version" in meta
        assert "scipy_version" in meta
        assert "seed" in meta
        assert "timestamp" in meta

    def test_seed_recorded(self):
        meta = generate_run_metadata(seed=123)
        assert meta["seed"] == 123

    def test_numpy_version(self):
        meta = generate_run_metadata(seed=0)
        assert meta["numpy_version"] == np.__version__

    def test_timestamp_iso8601(self):
        meta = generate_run_metadata(seed=0)
        ts = meta["timestamp"]
        # Should parse as ISO 8601
        parsed = datetime.fromisoformat(ts)
        assert parsed is not None

    def test_repo_version_is_string(self):
        meta = generate_run_metadata(seed=0)
        assert isinstance(meta["repo_version"], str)
        assert len(meta["repo_version"]) > 0

    def test_different_seeds_same_structure(self):
        m1 = generate_run_metadata(seed=1)
        m2 = generate_run_metadata(seed=2)
        assert set(m1.keys()) == set(m2.keys())
        assert m1["seed"] != m2["seed"]
