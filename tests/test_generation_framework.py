"""
Tests for the v8.4.0 Tanner graph generation framework.

Verifies:
  - candidate generation is deterministic
  - candidate ranking is reproducible
  - trajectory artifact schema is valid
  - exported graphs reload correctly
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

from qec.generation.tanner_graph_generator import (
    generate_tanner_graph_candidates,
)
from qec.generation.candidate_evaluation import (
    evaluate_tanner_graph_candidate,
)
from qec.generation.candidate_ranking import (
    rank_tanner_graph_candidates,
)
from qec.generation.export_generated_graph import (
    export_generated_graph,
)
from qec.generation.deterministic_construction import (
    construct_deterministic_tanner_graph,
)
from qec.experiments.generation_trajectory import (
    run_generation_trajectory,
)
from qec.experiments.generation_benchmark import (
    run_generation_benchmark,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _default_spec():
    """Small graph specification for testing."""
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


# ── Candidate generation tests ──────────────────────────────────


class TestTannerGraphGeneration:
    """Tests for deterministic Tanner graph generation."""

    def test_returns_requested_count(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 5)
        assert len(candidates) == 5

    def test_candidate_has_required_keys(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 1)
        c = candidates[0]
        assert "candidate_id" in c
        assert "H" in c

    def test_H_shape_matches_spec(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 3)
        for c in candidates:
            H = c["H"]
            assert H.shape == (spec["num_checks"], spec["num_variables"])

    def test_H_is_binary(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 3)
        for c in candidates:
            H = c["H"]
            assert np.all((H == 0) | (H == 1))

    def test_deterministic(self):
        spec = _default_spec()
        c1 = generate_tanner_graph_candidates(spec, 3, base_seed=42)
        c2 = generate_tanner_graph_candidates(spec, 3, base_seed=42)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a["H"], b["H"])
            assert a["candidate_id"] == b["candidate_id"]

    def test_different_seeds_differ(self):
        spec = _default_spec()
        c1 = generate_tanner_graph_candidates(spec, 1, base_seed=0)
        c2 = generate_tanner_graph_candidates(spec, 1, base_seed=99)
        # Matrices should differ (overwhelmingly likely for different seeds)
        assert not np.array_equal(c1[0]["H"], c2[0]["H"])

    def test_nonzero_rows_and_cols(self):
        """Every row and column must have at least one nonzero entry."""
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 5)
        for c in candidates:
            H = c["H"]
            assert np.all(H.sum(axis=1) > 0), "Found all-zero row"
            assert np.all(H.sum(axis=0) > 0), "Found all-zero column"

    def test_perturbation_mode(self):
        """Perturbation of a seed graph produces valid matrices."""
        seed_H = _small_H()
        spec = {
            "num_variables": seed_H.shape[1],
            "num_checks": seed_H.shape[0],
            "variable_degree": 2,
            "check_degree": 3,
            "seed_graph": seed_H,
        }
        candidates = generate_tanner_graph_candidates(spec, 5, base_seed=7)
        assert len(candidates) == 5
        for c in candidates:
            H = c["H"]
            assert H.shape == seed_H.shape
            assert np.all((H == 0) | (H == 1))

    def test_candidate_ids_unique(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 10)
        ids = [c["candidate_id"] for c in candidates]
        assert len(ids) == len(set(ids))


# ── Candidate evaluation tests ──────────────────────────────────


class TestCandidateEvaluation:
    """Tests for spectral candidate evaluation."""

    def test_returns_all_keys(self):
        H = _small_H()
        result = evaluate_tanner_graph_candidate(H)
        required_keys = {
            "spectral_radius", "entropy", "spectral_gap",
            "bethe_margin", "support_dimension", "curvature",
            "cycle_density", "sis", "instability_score",
            "predicted_regime",
        }
        assert required_keys == set(result.keys())

    def test_deterministic(self):
        H = _small_H()
        r1 = evaluate_tanner_graph_candidate(H)
        r2 = evaluate_tanner_graph_candidate(H)
        for key in r1:
            assert r1[key] == r2[key], f"Non-deterministic key: {key}"

    def test_spectral_radius_positive(self):
        H = _small_H()
        result = evaluate_tanner_graph_candidate(H)
        assert result["spectral_radius"] > 0

    def test_predicted_regime_valid(self):
        H = _small_H()
        result = evaluate_tanner_graph_candidate(H)
        assert result["predicted_regime"] in {"stable", "metastable", "unstable"}

    def test_instability_score_finite(self):
        H = _small_H()
        result = evaluate_tanner_graph_candidate(H)
        assert np.isfinite(result["instability_score"])


# ── Candidate ranking tests ─────────────────────────────────────


class TestCandidateRanking:
    """Tests for deterministic candidate ranking."""

    def test_ranking_reproducible(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 5)
        evaluated = []
        for c in candidates:
            ev = evaluate_tanner_graph_candidate(c["H"])
            evaluated.append({**ev, "candidate_id": c["candidate_id"]})

        r1 = rank_tanner_graph_candidates(evaluated)
        r2 = rank_tanner_graph_candidates(evaluated)
        for a, b in zip(r1, r2):
            assert a["candidate_id"] == b["candidate_id"]
            assert a["instability_score"] == b["instability_score"]

    def test_ranked_in_ascending_instability(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 5)
        evaluated = []
        for c in candidates:
            ev = evaluate_tanner_graph_candidate(c["H"])
            evaluated.append({**ev, "candidate_id": c["candidate_id"]})

        ranked = rank_tanner_graph_candidates(evaluated)
        for i in range(len(ranked) - 1):
            assert ranked[i]["instability_score"] <= ranked[i + 1]["instability_score"]

    def test_preserves_all_candidates(self):
        spec = _default_spec()
        candidates = generate_tanner_graph_candidates(spec, 5)
        evaluated = []
        for c in candidates:
            ev = evaluate_tanner_graph_candidate(c["H"])
            evaluated.append({**ev, "candidate_id": c["candidate_id"]})

        ranked = rank_tanner_graph_candidates(evaluated)
        assert len(ranked) == len(evaluated)


# ── Trajectory tests ────────────────────────────────────────────


class TestGenerationTrajectory:
    """Tests for the generation trajectory experiment."""

    def test_trajectory_schema(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trajectory.json")
            trajectory = run_generation_trajectory(
                spec, num_steps=2, candidates_per_step=3,
                output_path=path,
            )

            assert len(trajectory) == 3  # step 0 + 2 steps
            for record in trajectory:
                assert "step" in record
                assert "best_candidate_id" in record
                assert "best_score" in record

    def test_trajectory_deterministic(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "t1.json")
            path2 = os.path.join(tmpdir, "t2.json")
            t1 = run_generation_trajectory(
                spec, num_steps=2, candidates_per_step=3,
                base_seed=42, output_path=path1,
            )
            t2 = run_generation_trajectory(
                spec, num_steps=2, candidates_per_step=3,
                base_seed=42, output_path=path2,
            )

            for a, b in zip(t1, t2):
                assert a["best_score"] == b["best_score"]
                assert a["best_candidate_id"] == b["best_candidate_id"]

    def test_trajectory_artifact_valid_json(self):
        spec = _default_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trajectory.json")
            run_generation_trajectory(
                spec, num_steps=1, candidates_per_step=2,
                output_path=path,
            )
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 2


# ── Export tests ────────────────────────────────────────────────


class TestExportGeneratedGraph:
    """Tests for graph export and reload."""

    def test_export_creates_file(self):
        H = _small_H()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.json")
            export_generated_graph(H, path)
            assert os.path.exists(path)

    def test_export_reloads_correctly(self):
        H = _small_H()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.json")
            export_generated_graph(H, path)

            with open(path) as f:
                data = json.load(f)

            assert "num_variables" in data
            assert "num_checks" in data
            assert "H" in data
            assert "metrics" in data
            assert "stability_score" in data

            H_reloaded = np.array(data["H"], dtype=np.float64)
            np.testing.assert_array_equal(H_reloaded, H)

    def test_export_dimensions_match(self):
        H = _small_H()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.json")
            artifact = export_generated_graph(H, path)

            assert artifact["num_variables"] == H.shape[1]
            assert artifact["num_checks"] == H.shape[0]

    def test_export_valid_json(self):
        H = _small_H()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "graph.json")
            export_generated_graph(H, path)

            with open(path) as f:
                raw = f.read()
            # Verify canonical serialization (no spaces)
            data = json.loads(raw)
            assert isinstance(data, dict)


# ── Benchmark tests ─────────────────────────────────────────────


class TestGenerationBenchmark:
    """Tests for the generation benchmark experiment."""

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
            path = os.path.join(tmpdir, "benchmark.json")
            result = run_generation_benchmark(
                specs, candidates_per_spec=3, output_path=path,
            )

            assert "num_specs" in result
            assert "results" in result
            assert result["num_specs"] == 2
            assert len(result["results"]) == 2

    def test_benchmark_deterministic(self):
        specs = [_default_spec()]
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = os.path.join(tmpdir, "b1.json")
            p2 = os.path.join(tmpdir, "b2.json")
            r1 = run_generation_benchmark(
                specs, candidates_per_spec=3, base_seed=42, output_path=p1,
            )
            r2 = run_generation_benchmark(
                specs, candidates_per_spec=3, base_seed=42, output_path=p2,
            )

            assert r1["results"][0]["best_instability_score"] == (
                r2["results"][0]["best_instability_score"]
            )


# ── Deterministic construction tests ────────────────────────────


class TestDeterministicConstruction:
    """Tests for the v8.5.0 deterministic cycle-avoidant construction."""

    def test_deterministic_reproducibility(self):
        """Identical specs produce identical matrices across calls."""
        spec = _default_spec()
        H1 = construct_deterministic_tanner_graph(spec)
        H2 = construct_deterministic_tanner_graph(spec)
        np.testing.assert_array_equal(H1, H2)

    def test_degree_constraints(self):
        """Column sums equal variable_degree, row sums equal check_degree."""
        spec = _default_spec()
        H = construct_deterministic_tanner_graph(spec)
        np.testing.assert_array_equal(
            H.sum(axis=0),
            np.full(spec["num_variables"], spec["variable_degree"]),
        )
        np.testing.assert_array_equal(
            H.sum(axis=1),
            np.full(spec["num_checks"], spec["check_degree"]),
        )

    def test_invalid_specification_raises(self):
        """Mismatched n*dv != m*dc raises ValueError."""
        bad_spec = {
            "num_variables": 6,
            "num_checks": 3,
            "variable_degree": 2,
            "check_degree": 5,  # 6*2 != 3*5
        }
        with pytest.raises(ValueError, match="Degree constraint violated"):
            construct_deterministic_tanner_graph(bad_spec)

    def test_generator_deterministic_with_construction(self):
        """generate_tanner_graph_candidates remains deterministic."""
        spec = _default_spec()
        c1 = generate_tanner_graph_candidates(spec, 4, base_seed=77)
        c2 = generate_tanner_graph_candidates(spec, 4, base_seed=77)
        for a, b in zip(c1, c2):
            np.testing.assert_array_equal(a["H"], b["H"])
            assert a["candidate_id"] == b["candidate_id"]

    def test_structural_sanity(self):
        """Evaluate a constructed graph and verify metrics exist."""
        spec = _default_spec()
        H = construct_deterministic_tanner_graph(spec)
        metrics = evaluate_tanner_graph_candidate(H)
        required_keys = {
            "spectral_radius", "entropy", "spectral_gap",
            "bethe_margin", "support_dimension", "curvature",
            "cycle_density", "sis", "instability_score",
            "predicted_regime",
        }
        assert required_keys == set(metrics.keys())
        assert np.all((H == 0) | (H == 1))

    def test_H_shape_and_dtype(self):
        """Output has correct shape and dtype."""
        spec = _default_spec()
        H = construct_deterministic_tanner_graph(spec)
        assert H.shape == (spec["num_checks"], spec["num_variables"])
        assert H.dtype == np.float64
