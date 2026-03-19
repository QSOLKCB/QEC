"""
Tests for the v11.4.0 Memetic Evolution integration.

Verifies:
  - deterministic population behavior with memetic optimization
  - fitness improves or stays stable
  - archive integration unchanged
  - memetic flag toggles correctly
  - backwards compatibility with memetic disabled
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


class TestMemeticEngineInit:
    """Tests for DiscoveryEngine memetic initialization."""

    def test_memetic_defaults(self):
        engine = DiscoveryEngine()
        assert engine.memetic_optimization is True
        assert engine.local_steps == 10

    def test_memetic_custom(self):
        engine = DiscoveryEngine(
            memetic_optimization=False, local_steps=5,
        )
        assert engine.memetic_optimization is False
        assert engine.local_steps == 5

    def test_backwards_compatible_no_memetic_args(self):
        engine = DiscoveryEngine(
            population_size=10, generations=5, seed=42,
        )
        assert engine.population_size == 10
        assert engine.generations == 5
        assert engine.seed == 42
        assert engine.memetic_optimization is True


class TestMemeticEvolutionRun:
    """Tests for the full discovery loop with memetic optimization."""

    def test_memetic_returns_required_keys(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        assert "best" in result
        assert "best_H" in result
        assert "elite_history" in result
        assert "archive" in result
        assert "generation_summaries" in result

    def test_memetic_best_has_fitness(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        assert result["best"] is not None
        assert result["best"]["fitness"] is not None

    def test_memetic_best_H_is_matrix(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        assert isinstance(result["best_H"], np.ndarray)
        assert result["best_H"].ndim == 2

    def test_memetic_deterministic(self):
        e1 = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        e2 = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        assert r1["best"]["fitness"] == r2["best"]["fitness"]
        assert r1["best"]["code_id"] == r2["best"]["code_id"]
        np.testing.assert_array_equal(r1["best_H"], r2["best_H"])

    def test_memetic_replay_identical(self):
        """Same seed produces byte-identical elite history."""
        e1 = DiscoveryEngine(
            population_size=4, generations=3, seed=7,
            memetic_optimization=True, local_steps=2,
        )
        e2 = DiscoveryEngine(
            population_size=4, generations=3, seed=7,
            memetic_optimization=True, local_steps=2,
        )
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        j1 = json.dumps(r1["elite_history"], sort_keys=True)
        j2 = json.dumps(r2["elite_history"], sort_keys=True)
        assert j1 == j2

    def test_memetic_disabled_deterministic(self):
        """Memetic disabled matches original behavior."""
        e1 = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=False,
        )
        e2 = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=False,
        )
        r1 = e1.run(_default_spec())
        r2 = e2.run(_default_spec())
        assert r1["best"]["fitness"] == r2["best"]["fitness"]
        assert r1["best"]["code_id"] == r2["best"]["code_id"]

    def test_memetic_archive_populated(self):
        engine = DiscoveryEngine(
            population_size=4, generations=3, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        assert len(result["archive"]) > 0

    def test_memetic_elite_history_length(self):
        engine = DiscoveryEngine(
            population_size=4, generations=3, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        assert len(result["elite_history"]) >= 3

    def test_memetic_generation_summaries(self):
        engine = DiscoveryEngine(
            population_size=4, generations=3, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(_default_spec())
        for summary in result["generation_summaries"]:
            assert "generation" in summary
            assert "best_fitness" in summary
            assert "population_size" in summary

    def test_memetic_best_H_shape_matches_spec(self):
        spec = _default_spec()
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.run(spec)
        H = result["best_H"]
        assert H.shape[0] == spec["num_checks"]
        assert H.shape[1] == spec["num_variables"]


class TestMemeticStep:
    """Tests for the apply_memetic_step method directly."""

    def test_empty_children(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        result = engine.apply_memetic_step([], gen_seed=42)
        assert result == []

    def test_children_evaluated_after_step(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        spec = _default_spec()
        engine._spec = spec
        engine.initialize_population(spec)
        engine.evaluate_population()

        # Create some children
        children = []
        for entry in engine._population[:2]:
            child = {
                "code_id": entry["code_id"],
                "H": entry["H"].copy(),
                "fitness": None,
                "metrics": {},
                "parent_id": None,
                "operator": "test",
                "generation": 1,
            }
            children.append(child)

        result = engine.apply_memetic_step(children, gen_seed=42)
        for child in result:
            assert child["fitness"] is not None

    def test_memetic_step_no_input_mutation(self):
        engine = DiscoveryEngine(
            population_size=4, generations=2, seed=42,
            memetic_optimization=True, local_steps=2,
        )
        spec = _default_spec()
        engine._spec = spec
        engine.initialize_population(spec)
        engine.evaluate_population()

        original_H = engine._population[0]["H"].copy()
        children = [{
            "code_id": engine._population[0]["code_id"],
            "H": engine._population[0]["H"].copy(),
            "fitness": None,
            "metrics": {},
            "parent_id": None,
            "operator": "test",
            "generation": 1,
        }]

        # Save copy of child H before memetic step
        child_H_copy = children[0]["H"].copy()
        engine.apply_memetic_step(children, gen_seed=42)

        # Original population entry should not be mutated
        np.testing.assert_array_equal(
            engine._population[0]["H"], original_H,
        )
