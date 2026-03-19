"""Tests for Tanner graph export formats (v9.3.0)."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
from scipy.io import mmread

from qec.io.export_graph import (
    export_matrix_market,
    export_parity_check,
    export_json_adjacency,
)


@pytest.fixture
def sample_H():
    """Small parity-check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


class TestExportMatrixMarket:
    def test_creates_file(self, sample_H, tmp_path):
        path = str(tmp_path / "test.mtx")
        export_matrix_market(sample_H, path)
        assert os.path.exists(path)

    def test_roundtrip(self, sample_H, tmp_path):
        path = str(tmp_path / "test.mtx")
        export_matrix_market(sample_H, path)
        loaded = mmread(path).toarray()
        np.testing.assert_array_equal(loaded, sample_H)

    def test_creates_parent_dirs(self, sample_H, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "test.mtx")
        export_matrix_market(sample_H, path)
        assert os.path.exists(path)


class TestExportParityCheck:
    def test_creates_file(self, sample_H, tmp_path):
        path = str(tmp_path / "test.txt")
        export_parity_check(sample_H, path)
        assert os.path.exists(path)

    def test_format_correct(self, sample_H, tmp_path):
        path = str(tmp_path / "test.txt")
        export_parity_check(sample_H, path)
        with open(path) as f:
            lines = f.readlines()
        # First line: n m
        assert lines[0].strip() == "6 3"
        # Row 0: columns 0, 1, 3
        assert lines[1].strip() == "0: 0 1 3"
        # Row 1: columns 1, 2, 4
        assert lines[2].strip() == "1: 1 2 4"
        # Row 2: columns 0, 2, 5
        assert lines[3].strip() == "2: 0 2 5"

    def test_line_count(self, sample_H, tmp_path):
        path = str(tmp_path / "test.txt")
        export_parity_check(sample_H, path)
        with open(path) as f:
            lines = f.readlines()
        # 1 header + m rows
        assert len(lines) == 1 + sample_H.shape[0]


class TestExportJsonAdjacency:
    def test_creates_file(self, sample_H, tmp_path):
        path = str(tmp_path / "test.json")
        export_json_adjacency(sample_H, path)
        assert os.path.exists(path)

    def test_structure_correct(self, sample_H, tmp_path):
        path = str(tmp_path / "test.json")
        export_json_adjacency(sample_H, path)
        with open(path) as f:
            data = json.load(f)
        assert data["num_variables"] == 6
        assert data["num_checks"] == 3
        assert isinstance(data["edges"], list)

    def test_edge_count(self, sample_H, tmp_path):
        path = str(tmp_path / "test.json")
        export_json_adjacency(sample_H, path)
        with open(path) as f:
            data = json.load(f)
        expected_edges = int(np.sum(sample_H != 0))
        assert len(data["edges"]) == expected_edges

    def test_edges_match_matrix(self, sample_H, tmp_path):
        path = str(tmp_path / "test.json")
        export_json_adjacency(sample_H, path)
        with open(path) as f:
            data = json.load(f)
        # Reconstruct matrix from edges
        m, n = sample_H.shape
        reconstructed = np.zeros((m, n), dtype=np.float64)
        for i, j in data["edges"]:
            reconstructed[i, j] = 1.0
        np.testing.assert_array_equal(reconstructed, sample_H)

    def test_deterministic(self, sample_H, tmp_path):
        path1 = str(tmp_path / "test1.json")
        path2 = str(tmp_path / "test2.json")
        export_json_adjacency(sample_H, path1)
        export_json_adjacency(sample_H, path2)
        with open(path1) as f1, open(path2) as f2:
            assert f1.read() == f2.read()
