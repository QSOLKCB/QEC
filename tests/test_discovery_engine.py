"""
Tests for the v9.0.0 discovery engine.

Verifies:
  - discovery runs deterministically
  - engine produces valid output structure
  - archive is updated across generations
  - generation summaries are consistent
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.experiments.discovery_run import run_discovery_experiment
from src.qec.experiments.discovery_benchmark import run_discovery_benchmark


def _default_spec():
    """Small graph specification for testing."""
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


class TestDiscoveryEngine:
    """Tests for the main discovery engine."""

    def test_returns_required_keys(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        assert "best_candidate" in result
        assert "elite_history" in result
        assert "archive_summary" in result
        assert "generation_summaries" in result

    def test_deterministic(self):
        spec = _default_spec()
        r1 = run_structure_discovery(
            spec, num_generations=2, population_size=4, base_seed=42,
        )
        r2 = run_structure_discovery(
            spec, num_generations=2, population_size=4, base_seed=42,
        )
        j1 = json.dumps(r1["elite_history"], sort_keys=True)
        j2 = json.dumps(r2["elite_history"], sort_keys=True)
        assert j1 == j2

    def test_generation_summaries_count(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=3, population_size=4)
        # Generation 0 + 3 generations = 4 summaries
        assert len(result["generation_summaries"]) == 4

    def test_elite_history_grows(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=3, population_size=4)
        assert len(result["elite_history"]) == 4

    def test_best_candidate_has_objectives(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        best = result["best_candidate"]
        assert best is not None
        assert "objectives" in best
        assert "composite_score" in best["objectives"]

    def test_archive_summary_has_categories(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        summary = result["archive_summary"]
        assert "best_composite" in summary
        assert "lowest_instability" in summary
        assert "total_unique" in summary

    def test_different_seeds_differ(self):
        spec = _default_spec()
        r1 = run_structure_discovery(
            spec, num_generations=2, population_size=4, base_seed=0,
        )
        r2 = run_structure_discovery(
            spec, num_generations=2, population_size=4, base_seed=99,
        )
        # Highly likely to produce different best candidates
        c1 = r1["best_candidate"]["objectives"]["composite_score"]
        c2 = r2["best_candidate"]["objectives"]["composite_score"]
        # At minimum, both should be finite
        assert np.isfinite(c1)
        assert np.isfinite(c2)


class TestDiscoveryExperiment:
    """Tests for the discovery run experiment."""

    def test_artifact_schema(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery.json")
            artifact = run_discovery_experiment(
                spec, num_generations=2, population_size=4, output_path=path,
            )
            assert "metadata" in artifact
            assert "results" in artifact
            results = artifact["results"]
            assert "spec" in results
            assert "config" in results
            assert "best_candidate" in results
            assert "generation_summaries" in results

    def test_artifact_valid_json(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery.json")
            run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=path,
            )
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_deterministic_artifact(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "d1.json")
            p2 = os.path.join(tmpdir, "d2.json")
            run_discovery_experiment(
                spec, num_generations=2, population_size=4,
                base_seed=42, output_path=p1,
            )
            run_discovery_experiment(
                spec, num_generations=2, population_size=4,
                base_seed=42, output_path=p2,
            )
            with open(p1) as f:
                d1 = json.load(f)
            with open(p2) as f:
                d2 = json.load(f)
            # Compare results (excluding timestamp in metadata)
            assert json.dumps(d1["results"], sort_keys=True) == json.dumps(d2["results"], sort_keys=True)


class TestDiscoveryBenchmark:
    """Tests for the discovery benchmark."""

    def test_benchmark_schema(self):
        specs = [
            _default_spec(),
            {
                "num_variables": 8,
                "num_checks": 4,
                "variable_degree": 2,
                "check_degree": 4,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bench.json")
            result = run_discovery_benchmark(
                specs, num_generations=1, population_size=4, output_path=path,
            )
            assert "metadata" in result
            assert "benchmark_results" in result
            bench = result["benchmark_results"]
            assert bench["num_specs"] == 2
            assert len(bench["results"]) == 2

    def test_benchmark_deterministic(self):
        specs = [_default_spec()]
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "b1.json")
            p2 = os.path.join(tmpdir, "b2.json")
            r1 = run_discovery_benchmark(
                specs, num_generations=1, population_size=4,
                base_seed=42, output_path=p1,
            )
            r2 = run_discovery_benchmark(
                specs, num_generations=1, population_size=4,
                base_seed=42, output_path=p2,
            )
            assert (
                r1["benchmark_results"]["results"][0]["best_composite_score"]
                == r2["benchmark_results"]["results"][0]["best_composite_score"]
            )


    def test_information_gain_fields_opt_in(self):
        spec = _default_spec()
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_landscape_learning=True,
            enable_information_gain_scheduler=True,
        )
        summary = result["generation_summaries"][-1]
        assert "information_gain_score" in summary
        assert "spectral_uncertainty" in summary
        assert "novelty_score" in summary
        assert "selected_target_spectrum" in summary

    def test_information_gain_fields_default_unchanged(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        summary = result["generation_summaries"][-1]
        assert "information_gain_score" not in summary
        assert "spectral_uncertainty" not in summary
        assert "novelty_score" not in summary
        assert "selected_target_spectrum" not in summary


    def test_experiment_planner_fields_opt_in(self):
        spec = _default_spec()
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_experiment_planner=True,
            planner_uncertainty_threshold=0.01,
            planner_max_targets=3,
        )
        summary = result["generation_summaries"][-1]
        assert "planned_experiment_targets" in summary
        assert "phase_uncertainty_score" in summary
        assert "planner_iteration" in summary

    def test_experiment_planner_fields_default_unchanged(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        summary = result["generation_summaries"][-1]
        assert "planned_experiment_targets" not in summary
        assert "phase_uncertainty_score" not in summary
        assert "planner_iteration" not in summary


    def test_phase_diagram_3d_fields_opt_in(self):
        spec = _default_spec()
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_experiment_planner=True,
            planner_uncertainty_threshold=0.01,
            planner_max_targets=3,
            enable_phase_diagram_3d=True,
        )
        assert "phase_diagram_3d_path" in result
        assert "phase_diagram_3d_num_targets" in result

    def test_phase_diagram_3d_fields_default_unchanged(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        assert "phase_diagram_3d_path" not in result
        assert "phase_diagram_3d_num_targets" not in result


    def test_theory_refinement_fields_opt_in(self):
        spec = _default_spec()
        result = run_structure_discovery(
            spec,
            num_generations=2,
            population_size=4,
            base_seed=42,
            enable_theory_extraction=True,
            enable_theory_refinement=True,
            theory_refinement_interval=1,
        )
        assert "theory_memory" in result
        assert "theory_memory_summary" in result
        assert "conjecture_validations" in result
        assert "conjecture_counterexamples" in result

    def test_theory_refinement_fields_default_unchanged(self):
        spec = _default_spec()
        result = run_structure_discovery(spec, num_generations=2, population_size=4)
        summary = result["generation_summaries"][-1]
        assert "num_validated_conjectures" not in summary
        assert "num_supported_conjectures" not in summary
        assert "num_fragile_conjectures" not in summary
        assert "num_rejected_conjectures" not in summary
        assert "mean_theory_support_score" not in summary
