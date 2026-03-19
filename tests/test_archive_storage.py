"""
Tests for the v10.0.0 persistent discovery archive.

Verifies:
  - SQLite storage persists and retrieves codes correctly
  - content-addressable storage works via matrix hash
  - elite retrieval respects fitness ordering
  - diverse code selection works
  - archive supports population seeding
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.archive.storage import DiscoveryArchive
from qec.archive.seeding import seed_population_from_archive


def _small_H():
    """Create a small parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _another_H():
    """Create a different parity-check matrix."""
    return np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1],
    ], dtype=np.float64)


class TestDiscoveryArchive:
    """Tests for the SQLite-backed archive."""

    def test_add_and_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                assert archive.count() == 0
                archive.add_code(_small_H(), fitness=1.0)
                assert archive.count() == 1
            finally:
                archive.close()

    def test_content_addressable(self):
        """Same matrix should produce same code_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                id1 = archive.add_code(_small_H(), fitness=1.0)
                id2 = archive.add_code(_small_H(), fitness=2.0)
                assert id1 == id2
                assert archive.count() == 1
            finally:
                archive.close()

    def test_different_matrices_different_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                id1 = archive.add_code(_small_H(), fitness=1.0)
                id2 = archive.add_code(_another_H(), fitness=2.0)
                assert id1 != id2
                assert archive.count() == 2
            finally:
                archive.close()

    def test_get_elite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                archive.add_code(_small_H(), fitness=1.0)
                archive.add_code(_another_H(), fitness=5.0)
                elites = archive.get_elite(top_k=2)
                assert len(elites) == 2
                # Best fitness first
                assert elites[0]["fitness"] >= elites[1]["fitness"]
            finally:
                archive.close()

    def test_matrix_round_trip(self):
        """Matrix should survive serialization/deserialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                H = _small_H()
                archive.add_code(H, fitness=1.0)
                elites = archive.get_elite(top_k=1)
                np.testing.assert_array_equal(elites[0]["H"], H)
            finally:
                archive.close()

    def test_genealogy_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                archive.add_code(
                    _small_H(),
                    fitness=1.0,
                    parent_code="parent_123",
                    mutation_operator="cycle_pressure",
                    generation=5,
                )
                elites = archive.get_elite(top_k=1)
                assert elites[0]["parent_code"] == "parent_123"
                assert elites[0]["mutation_operator"] == "cycle_pressure"
                assert elites[0]["generation"] == 5
            finally:
                archive.close()

    def test_metrics_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                metrics = {"girth": 6, "spectral_radius": 1.5}
                archive.add_code(_small_H(), fitness=1.0, metrics=metrics)
                elites = archive.get_elite(top_k=1)
                assert elites[0]["metrics"]["girth"] == 6
            finally:
                archive.close()

    def test_persistence(self):
        """Data should persist across connections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            archive.add_code(_small_H(), fitness=1.0)
            archive.close()

            archive2 = DiscoveryArchive(db_path=db_path)
            try:
                assert archive2.count() == 1
            finally:
                archive2.close()

    def test_get_diverse_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                archive.add_code(_small_H(), fitness=3.0)
                archive.add_code(_another_H(), fitness=2.0)
                diverse = archive.get_diverse_codes(top_k=2)
                assert len(diverse) == 2
            finally:
                archive.close()

    def test_seed_population(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            try:
                archive.add_code(_small_H(), fitness=3.0)
                archive.add_code(_another_H(), fitness=2.0)
                matrices = archive.seed_population(size=2)
                assert len(matrices) == 2
                assert all(isinstance(m, np.ndarray) for m in matrices)
            finally:
                archive.close()


class TestSeedPopulationFromArchive:
    """Tests for the seeding utility."""

    def test_seed_from_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            archive.add_code(_small_H(), fitness=3.0)
            archive.add_code(_another_H(), fitness=2.0)
            archive.close()

            matrices = seed_population_from_archive(db_path, size=2)
            assert len(matrices) == 2

    def test_empty_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            archive = DiscoveryArchive(db_path=db_path)
            archive.close()

            matrices = seed_population_from_archive(db_path, size=5)
            assert len(matrices) == 0
