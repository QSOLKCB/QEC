"""Tests for v84.5.0 — Phase Map JSON Export."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from qec.experiments.hybrid_target_sweep import (
    phase_map_to_json,
    save_phase_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_phase_map():
    return {
        "nodes": [
            {"id": 0, "label": "stable", "score": 0.95},
            {"id": 1, "label": "boundary", "score": 0.50},
        ],
        "edges": [
            {"source": 0, "target": 1, "weight": 0.45},
        ],
    }


# ---------------------------------------------------------------------------
# Tests — save_phase_map
# ---------------------------------------------------------------------------


class TestSavePhaseMap:
    def test_file_created(self, tmp_path):
        """Output file is created on disk."""
        out = tmp_path / "map.json"
        save_phase_map(_sample_phase_map(), out)
        assert out.exists()

    def test_round_trip(self, tmp_path):
        """JSON contents round-trip correctly."""
        pm = _sample_phase_map()
        out = tmp_path / "map.json"
        save_phase_map(pm, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == pm

    def test_metadata_n_nodes(self, tmp_path):
        """Metadata reports correct n_nodes."""
        out = tmp_path / "map.json"
        meta = save_phase_map(_sample_phase_map(), out)
        assert meta["n_nodes"] == 2

    def test_metadata_n_edges(self, tmp_path):
        """Metadata reports correct n_edges."""
        out = tmp_path / "map.json"
        meta = save_phase_map(_sample_phase_map(), out)
        assert meta["n_edges"] == 1

    def test_metadata_output_path(self, tmp_path):
        """Metadata output_path matches requested path."""
        out = tmp_path / "map.json"
        meta = save_phase_map(_sample_phase_map(), out)
        assert meta["output_path"] == str(out)

    def test_sort_keys(self, tmp_path):
        """Output uses sort_keys=True (deterministic key ordering)."""
        pm = {"zebra": 1, "alpha": 2, "nodes": [], "edges": []}
        out = tmp_path / "map.json"
        save_phase_map(pm, out)
        text = out.read_text(encoding="utf-8")
        keys = list(json.loads(text).keys())
        assert keys == sorted(keys)

    def test_empty_phase_map(self, tmp_path):
        """Empty phase map (no nodes, no edges) is supported."""
        out = tmp_path / "map.json"
        meta = save_phase_map({}, out)
        assert meta["n_nodes"] == 0
        assert meta["n_edges"] == 0
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded == {}

    def test_utf8_encoding(self, tmp_path):
        """File is written with UTF-8 encoding."""
        pm = {"nodes": [{"label": "\u03b1\u03b2\u03b3"}], "edges": []}
        out = tmp_path / "map.json"
        save_phase_map(pm, out)
        text = out.read_text(encoding="utf-8")
        assert "\u03b1\u03b2\u03b3" in text


# ---------------------------------------------------------------------------
# Tests — phase_map_to_json
# ---------------------------------------------------------------------------


class TestPhaseMapToJson:
    def test_deterministic(self):
        """Repeated calls produce identical output."""
        pm = _sample_phase_map()
        assert phase_map_to_json(pm) == phase_map_to_json(pm)

    def test_sort_keys(self):
        """Keys are sorted in output string."""
        pm = {"z": 1, "a": 2}
        parsed = json.loads(phase_map_to_json(pm))
        assert list(parsed.keys()) == ["a", "z"]
