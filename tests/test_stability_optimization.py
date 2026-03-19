"""
Tests for v8.3.0 stability optimization and spectral invariant discovery.

Verifies:
  - Repair candidates deterministic
  - Optimizer improves stability score
  - Invariant discovery returns ranked expressions
  - Stability landscape dataset schema valid
  - API integration
  - No decoder import from diagnostics layer
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


# ── Test fixtures ─────────────────────────────────────────────────


def _small_H():
    """Return a small test parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
    ], dtype=np.float64)


def _another_H():
    """Return another small test matrix with different structure."""
    return np.array([
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0],
    ], dtype=np.float64)


def _sample_dataset():
    """Return a minimal stability dataset for testing."""
    return [
        {
            "spectral_radius": 1.0,
            "entropy": 2.0,
            "spectral_gap": 0.1,
            "bethe_margin": 0.5,
            "support_dimension": 3.0,
            "curvature": 0.2,
            "cycle_density": 0.3,
            "sis": 0.01,
            "bp_converged": 1,
        },
        {
            "spectral_radius": 1.5,
            "entropy": 1.0,
            "spectral_gap": 0.05,
            "bethe_margin": -0.1,
            "support_dimension": 2.0,
            "curvature": 0.8,
            "cycle_density": 0.6,
            "sis": 0.05,
            "bp_converged": 0,
        },
        {
            "spectral_radius": 0.8,
            "entropy": 2.5,
            "spectral_gap": 0.2,
            "bethe_margin": 0.8,
            "support_dimension": 4.0,
            "curvature": 0.1,
            "cycle_density": 0.2,
            "sis": 0.005,
            "bp_converged": 1,
        },
        {
            "spectral_radius": 2.0,
            "entropy": 0.5,
            "spectral_gap": 0.01,
            "bethe_margin": -0.5,
            "support_dimension": 1.5,
            "curvature": 1.2,
            "cycle_density": 0.9,
            "sis": 0.1,
            "bp_converged": 0,
        },
    ]


# ── Part 1: Repair Candidate Generator ─────────────────────────────


class TestRepairCandidates:
    def test_generate_repair_candidates_returns_list(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )

        candidates = generate_repair_candidates(_small_H())
        assert isinstance(candidates, list)
        for c in candidates:
            assert "remove_edge" in c
            assert "add_edge" in c
            assert "predicted_effect" in c

    def test_repair_candidates_determinism(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )

        r1 = generate_repair_candidates(_small_H())
        r2 = generate_repair_candidates(_small_H())
        assert r1 == r2

    def test_repair_candidates_max_candidates(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )

        candidates = generate_repair_candidates(
            _small_H(), max_candidates=3,
        )
        assert len(candidates) <= 3

    def test_repair_candidates_another_matrix(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )

        candidates = generate_repair_candidates(_another_H())
        assert isinstance(candidates, list)


# ── Part 2: Repair Scoring ──────────────────────────────────────────


class TestRepairScoring:
    def test_score_repair_candidate_returns_deltas(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )
        from qec.diagnostics.repair_scoring import score_repair_candidate

        candidates = generate_repair_candidates(_small_H())
        if candidates:
            score = score_repair_candidate(_small_H(), candidates[0])
            assert "delta_spectral_radius" in score
            assert "delta_entropy" in score
            assert "delta_curvature" in score
            assert "delta_cycle_density" in score
            assert "delta_sis" in score
            for val in score.values():
                assert isinstance(val, float)

    def test_score_repair_determinism(self):
        from qec.diagnostics.repair_candidates import (
            generate_repair_candidates,
        )
        from qec.diagnostics.repair_scoring import score_repair_candidate

        candidates = generate_repair_candidates(_small_H())
        if candidates:
            s1 = score_repair_candidate(_small_H(), candidates[0])
            s2 = score_repair_candidate(_small_H(), candidates[0])
            assert s1 == s2


# ── Part 3: Stability Optimizer ──────────────────────────────────────


class TestStabilityOptimizer:
    def test_optimize_returns_trajectory(self, tmp_path):
        from qec.diagnostics.stability_optimizer import (
            optimize_tanner_graph_stability,
        )

        output_path = str(tmp_path / "trajectory.json")
        trajectory = optimize_tanner_graph_stability(
            _small_H(), steps=2, output_path=output_path,
        )

        assert isinstance(trajectory, list)
        assert len(trajectory) == 3  # step 0, 1, 2
        for entry in trajectory:
            assert "step" in entry
            assert "score" in entry
            assert isinstance(entry["step"], int)
            assert isinstance(entry["score"], float)

    def test_optimize_trajectory_monotonic_or_flat(self, tmp_path):
        from qec.diagnostics.stability_optimizer import (
            optimize_tanner_graph_stability,
        )

        output_path = str(tmp_path / "trajectory.json")
        trajectory = optimize_tanner_graph_stability(
            _small_H(), steps=3, output_path=output_path,
        )

        # Score should be non-decreasing (improvements only)
        for i in range(1, len(trajectory)):
            assert trajectory[i]["score"] >= trajectory[i - 1]["score"] - 1e-10

    def test_optimize_artifact_saved(self, tmp_path):
        from qec.diagnostics.stability_optimizer import (
            optimize_tanner_graph_stability,
        )

        output_path = str(tmp_path / "trajectory.json")
        optimize_tanner_graph_stability(
            _small_H(), steps=1, output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)

    def test_optimize_determinism(self, tmp_path):
        from qec.diagnostics.stability_optimizer import (
            optimize_tanner_graph_stability,
        )

        p1 = str(tmp_path / "t1.json")
        p2 = str(tmp_path / "t2.json")
        t1 = optimize_tanner_graph_stability(_small_H(), steps=2, output_path=p1)
        t2 = optimize_tanner_graph_stability(_small_H(), steps=2, output_path=p2)
        assert t1 == t2


# ── Part 4: Spectral Invariant Discovery ─────────────────────────────


class TestSpectralInvariantDiscovery:
    def test_discover_returns_ranked_list(self, tmp_path):
        from qec.diagnostics.spectral_invariant_discovery import (
            discover_spectral_invariants,
        )

        output_path = str(tmp_path / "invariants.json")
        invariants = discover_spectral_invariants(
            _sample_dataset(), output_path=output_path,
        )

        assert isinstance(invariants, list)
        assert len(invariants) > 0
        for inv in invariants:
            assert "expression" in inv
            assert "correlation" in inv
            assert "accuracy" in inv
            assert isinstance(inv["expression"], str)
            assert isinstance(inv["correlation"], float)
            assert isinstance(inv["accuracy"], float)

    def test_discover_sorted_by_correlation(self, tmp_path):
        from qec.diagnostics.spectral_invariant_discovery import (
            discover_spectral_invariants,
        )

        output_path = str(tmp_path / "invariants.json")
        invariants = discover_spectral_invariants(
            _sample_dataset(), output_path=output_path,
        )

        # Sorted by absolute correlation descending
        for i in range(1, len(invariants)):
            assert abs(invariants[i - 1]["correlation"]) >= abs(
                invariants[i]["correlation"]
            ) - 1e-12

    def test_discover_empty_dataset(self, tmp_path):
        from qec.diagnostics.spectral_invariant_discovery import (
            discover_spectral_invariants,
        )

        output_path = str(tmp_path / "invariants.json")
        invariants = discover_spectral_invariants([], output_path=output_path)
        assert invariants == []

    def test_discover_artifact_saved(self, tmp_path):
        from qec.diagnostics.spectral_invariant_discovery import (
            discover_spectral_invariants,
        )

        output_path = str(tmp_path / "invariants.json")
        discover_spectral_invariants(
            _sample_dataset(), output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)

    def test_discover_determinism(self, tmp_path):
        from qec.diagnostics.spectral_invariant_discovery import (
            discover_spectral_invariants,
        )

        p1 = str(tmp_path / "i1.json")
        p2 = str(tmp_path / "i2.json")
        i1 = discover_spectral_invariants(_sample_dataset(), output_path=p1)
        i2 = discover_spectral_invariants(_sample_dataset(), output_path=p2)
        assert i1 == i2


# ── Part 5: Stability Landscape Explorer ──────────────────────────────


class TestStabilityLandscape:
    def test_explore_returns_dataset(self, tmp_path):
        from qec.experiments.stability_landscape import (
            explore_stability_landscape,
        )

        output_path = str(tmp_path / "landscape.json")
        dataset = explore_stability_landscape(
            _small_H(),
            num_perturbations=3,
            base_seed=42,
            output_path=output_path,
        )

        assert isinstance(dataset, list)
        assert len(dataset) >= 1  # at least the original graph
        for sample in dataset:
            assert "spectral_radius" in sample
            assert "entropy" in sample
            assert "spectral_gap" in sample
            assert "bethe_margin" in sample
            assert "support_dimension" in sample
            assert "curvature" in sample
            assert "cycle_density" in sample
            assert "sis" in sample
            assert "predicted_stable" in sample
            assert isinstance(sample["predicted_stable"], bool)

    def test_explore_artifact_saved(self, tmp_path):
        from qec.experiments.stability_landscape import (
            explore_stability_landscape,
        )

        output_path = str(tmp_path / "landscape.json")
        explore_stability_landscape(
            _small_H(),
            num_perturbations=2,
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)

    def test_explore_determinism(self, tmp_path):
        from qec.experiments.stability_landscape import (
            explore_stability_landscape,
        )

        p1 = str(tmp_path / "l1.json")
        p2 = str(tmp_path / "l2.json")
        d1 = explore_stability_landscape(
            _small_H(), num_perturbations=3, base_seed=42, output_path=p1,
        )
        d2 = explore_stability_landscape(
            _small_H(), num_perturbations=3, base_seed=42, output_path=p2,
        )
        assert d1 == d2


# ── Part 6: Optimization Benchmark ────────────────────────────────────


class TestStabilityOptimizationBenchmark:
    def test_benchmark_runs(self, tmp_path):
        from qec.experiments.stability_optimization_benchmark import (
            run_stability_optimization_benchmark,
        )

        output_path = str(tmp_path / "benchmark.json")
        result = run_stability_optimization_benchmark(
            num_graphs=2,
            base_seed=42,
            optimization_steps=1,
            output_path=output_path,
        )

        assert "initial_stability_score" in result
        assert "final_stability_score" in result
        assert "critical_radius_shift" in result
        assert "repair_steps" in result
        assert isinstance(result["initial_stability_score"], float)
        assert isinstance(result["final_stability_score"], float)

    def test_benchmark_artifact_saved(self, tmp_path):
        from qec.experiments.stability_optimization_benchmark import (
            run_stability_optimization_benchmark,
        )

        output_path = str(tmp_path / "benchmark.json")
        run_stability_optimization_benchmark(
            num_graphs=2,
            base_seed=42,
            optimization_steps=1,
            output_path=output_path,
        )

        assert os.path.exists(output_path)


# ── Part 7: API Integration ──────────────────────────────────────────


class TestAPIIntegrationV83:
    def test_optimize_tanner_graph_stability_in_api(self, tmp_path):
        from qec.diagnostics.api import optimize_tanner_graph_stability

        output_path = str(tmp_path / "trajectory.json")
        trajectory = optimize_tanner_graph_stability(
            _small_H(), steps=1, output_path=output_path,
        )
        assert isinstance(trajectory, list)
        assert len(trajectory) >= 1

    def test_generate_repair_candidates_in_api(self):
        from qec.diagnostics.api import generate_repair_candidates

        candidates = generate_repair_candidates(_small_H())
        assert isinstance(candidates, list)

    def test_score_repair_candidate_in_api(self):
        from qec.diagnostics.api import (
            generate_repair_candidates,
            score_repair_candidate,
        )

        candidates = generate_repair_candidates(_small_H())
        if candidates:
            score = score_repair_candidate(_small_H(), candidates[0])
            assert "delta_spectral_radius" in score

    def test_discover_spectral_invariants_in_api(self, tmp_path):
        from qec.diagnostics.api import discover_spectral_invariants

        output_path = str(tmp_path / "invariants.json")
        invariants = discover_spectral_invariants(
            _sample_dataset(), output_path=output_path,
        )
        assert isinstance(invariants, list)
        assert len(invariants) > 0


# ── Layer discipline: diagnostics must not import decoder ─────────


class TestLayerDisciplineV83:
    def test_repair_candidates_no_decoder_import(self):
        import qec.diagnostics.repair_candidates as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_repair_scoring_no_decoder_import(self):
        import qec.diagnostics.repair_scoring as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_stability_optimizer_no_decoder_import(self):
        import qec.diagnostics.stability_optimizer as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_spectral_invariant_discovery_no_decoder_import(self):
        import qec.diagnostics.spectral_invariant_discovery as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
