"""
Tests for v9.2.0 reproducibility metadata.

Verifies:
  - metadata exists in all discovery artifacts
  - required fields are present
  - artifact structure is valid (metadata + results/benchmark_results/archive)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.utils.reproducibility import collect_environment_metadata
from qec.experiments.discovery_run import run_discovery_experiment
from qec.experiments.discovery_benchmark import run_discovery_benchmark


_REQUIRED_METADATA_FIELDS = [
    "repo_version",
    "git_commit",
    "python_version",
    "numpy_version",
    "scipy_version",
    "timestamp",
]


def _default_spec():
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


class TestCollectEnvironmentMetadata:
    """Tests for the collect_environment_metadata utility."""

    def test_required_fields_present(self):
        meta = collect_environment_metadata()
        for field in _REQUIRED_METADATA_FIELDS:
            assert field in meta, f"Missing required field: {field}"

    def test_repo_version(self):
        meta = collect_environment_metadata()
        assert meta["repo_version"] == "9.2.0"

    def test_python_version_format(self):
        meta = collect_environment_metadata()
        parts = meta["python_version"].split(".")
        assert len(parts) >= 2

    def test_timestamp_format(self):
        meta = collect_environment_metadata()
        assert meta["timestamp"].endswith("Z")
        assert "T" in meta["timestamp"]

    def test_optional_spec(self):
        spec = {"num_variables": 6}
        meta = collect_environment_metadata(spec=spec)
        assert meta["spec"] == spec

    def test_optional_generation_count(self):
        meta = collect_environment_metadata(generation_count=10)
        assert meta["generation_count"] == 10

    def test_optional_population_size(self):
        meta = collect_environment_metadata(population_size=8)
        assert meta["population_size"] == 8

    def test_omitted_optionals(self):
        meta = collect_environment_metadata()
        assert "spec" not in meta
        assert "generation_count" not in meta
        assert "population_size" not in meta

    def test_git_commit_is_string(self):
        meta = collect_environment_metadata()
        assert isinstance(meta["git_commit"], str)
        assert len(meta["git_commit"]) > 0


class TestDiscoveryRunMetadata:
    """Tests that discovery run artifacts contain metadata."""

    def test_artifact_has_metadata(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery_run.json")
            artifact = run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=path,
            )
            assert "metadata" in artifact
            for field in _REQUIRED_METADATA_FIELDS:
                assert field in artifact["metadata"]

    def test_artifact_has_results(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery_run.json")
            artifact = run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=path,
            )
            assert "results" in artifact
            results = artifact["results"]
            assert "best_candidate" in results
            assert "generation_summaries" in results

    def test_artifact_json_valid(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery_run.json")
            run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=path,
            )
            with open(path) as f:
                data = json.load(f)
            assert "metadata" in data
            assert "results" in data

    def test_archive_artifact_created(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_path = os.path.join(tmpdir, "discovery_run.json")
            run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=run_path,
            )
            archive_path = os.path.join(tmpdir, "discovery_archive.json")
            assert os.path.exists(archive_path)
            with open(archive_path) as f:
                data = json.load(f)
            assert "metadata" in data
            assert "archive" in data

    def test_metadata_includes_spec(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "discovery_run.json")
            artifact = run_discovery_experiment(
                spec, num_generations=1, population_size=4, output_path=path,
            )
            assert artifact["metadata"]["spec"] == spec


class TestDiscoveryBenchmarkMetadata:
    """Tests that benchmark artifacts contain metadata."""

    def test_benchmark_has_metadata(self):
        specs = [_default_spec()]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bench.json")
            result = run_discovery_benchmark(
                specs, num_generations=1, population_size=4, output_path=path,
            )
            assert "metadata" in result
            for field in _REQUIRED_METADATA_FIELDS:
                assert field in result["metadata"]

    def test_benchmark_has_benchmark_results(self):
        specs = [_default_spec()]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bench.json")
            result = run_discovery_benchmark(
                specs, num_generations=1, population_size=4, output_path=path,
            )
            assert "benchmark_results" in result
            assert "results" in result["benchmark_results"]

    def test_benchmark_json_valid(self):
        specs = [_default_spec()]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bench.json")
            run_discovery_benchmark(
                specs, num_generations=1, population_size=4, output_path=path,
            )
            with open(path) as f:
                data = json.load(f)
            assert "metadata" in data
            assert "benchmark_results" in data
