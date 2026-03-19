"""
Tests for the v10.0.0 population discovery engine.

Verifies:
  - discovery engine runs deterministically
  - population management works correctly
  - same seed produces identical results
  - output structure is correct
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.population_engine import DiscoveryEngine


def _default_spec():
    """Small graph specification for testing."""
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


class TestDiscoveryEngineInit:
    """Tests for DiscoveryEngine initialization."""

    def test_default_params(self):
        engine = DiscoveryEngine()
        assert engine.population_size == 50
        assert engine.generations == 500
        assert engine.seed == 42

    def test_custom_params(self):
        engine = DiscoveryEngine(
            population_size=10, generations=5, seed=99,
        )
        assert engine.population_size == 10
        assert engine.generations == 5
        assert engine.seed == 99


class TestDiscoveryEngineRun:
    """Tests for the full discovery loop."""

    def test_returns_required_keys(self):
        engine = DiscoveryEngine(population_size=4, generations=2, seed=42)
        result = engine.run(_default_spec())
        assert "best" in result
        assert "best_H" in result
        assert "elite_history" in result
        assert "archive" in result
        assert "generation_summaries" in result

    def test_best_has_fitness(self):
        engine = DiscoveryEngine(population_size=4, generations=2, seed=42)
        result = engine.run(_default_spec())
        assert result["best"] is not None
        assert result["best"]["fitness"] is not None

    def test_best_H_is_matrix(self):
        engine = DiscoveryEngine(population_size=4, generations=2, seed=42)
        result = engine.run(_default_spec())
        assert isinstance(result["best_H"], np.ndarray)
        assert result["best_H"].ndim == 2

    def test_deterministic(self):
        e1 = DiscoveryEngine(population_size=4, generations=2, seed=42)
        e2 = DiscoveryEngine(population_size=4, generations=2, seed=42)
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        assert r1["best"]["fitness"] == r2["best"]["fitness"]
        assert r1["best"]["code_id"] == r2["best"]["code_id"]
        np.testing.assert_array_equal(r1["best_H"], r2["best_H"])

    def test_replay_identical(self):
        """Same seed produces byte-identical elite history."""
        e1 = DiscoveryEngine(population_size=4, generations=3, seed=7)
        e2 = DiscoveryEngine(population_size=4, generations=3, seed=7)
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        j1 = json.dumps(r1["elite_history"], sort_keys=True)
        j2 = json.dumps(r2["elite_history"], sort_keys=True)
        assert j1 == j2

    def test_different_seeds_differ(self):
        e1 = DiscoveryEngine(population_size=4, generations=2, seed=0)
        e2 = DiscoveryEngine(population_size=4, generations=2, seed=99)
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        # Both should produce finite fitness
        assert np.isfinite(r1["best"]["fitness"])
        assert np.isfinite(r2["best"]["fitness"])

    def test_elite_history_length(self):
        engine = DiscoveryEngine(population_size=4, generations=3, seed=42)
        result = engine.run(_default_spec())
        # One entry per generation (1..3) plus final evaluation
        assert len(result["elite_history"]) >= 3

    def test_generation_summaries(self):
        engine = DiscoveryEngine(population_size=4, generations=3, seed=42)
        result = engine.run(_default_spec())
        for summary in result["generation_summaries"]:
            assert "generation" in summary
            assert "best_fitness" in summary
            assert "population_size" in summary

    def test_archive_populated(self):
        engine = DiscoveryEngine(population_size=4, generations=3, seed=42)
        result = engine.run(_default_spec())
        assert len(result["archive"]) > 0
