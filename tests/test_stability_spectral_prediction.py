"""
Tests for v8.2.0 stability boundary and spectral prediction modules.

Verifies:
  - Dataset builder schema
  - Boundary estimation returns valid weights
  - Predictor deterministic output
  - Trapping-set detector returns clusters
  - Critical radius estimator behaves correctly
  - Spectral critical line estimator returns valid value
  - Repair trajectory artifact schema valid
  - ASCII boundary rendering
  - API integration
  - Benchmark experiment
  - Determinism (repeated runs produce identical output)
  - No decoder import from diagnostics layer
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


# ── Part 1: Dataset Builder ──────────────────────────────────────


class TestStabilityDataset:
    def test_build_dataset_schema(self, tmp_path):
        from qec.experiments.stability_dataset import build_stability_dataset

        graphs = [_small_H(), _another_H()]
        output_path = str(tmp_path / "dataset.json")

        dataset = build_stability_dataset(
            graphs, output_path=output_path, base_seed=42,
        )

        assert len(dataset) == 2
        for obs in dataset:
            assert "spectral_radius" in obs
            assert "entropy" in obs
            assert "spectral_gap" in obs
            assert "bethe_margin" in obs
            assert "support_dimension" in obs
            assert "curvature" in obs
            assert "cycle_density" in obs
            assert "sis" in obs
            assert "bp_converged" in obs
            assert obs["bp_converged"] in (0, 1)

    def test_dataset_artifact_saved(self, tmp_path):
        from qec.experiments.stability_dataset import build_stability_dataset

        output_path = str(tmp_path / "dataset.json")
        build_stability_dataset(
            [_small_H()], output_path=output_path, base_seed=42,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 1

    def test_dataset_determinism(self, tmp_path):
        from qec.experiments.stability_dataset import build_stability_dataset

        graphs = [_small_H(), _another_H()]
        path1 = str(tmp_path / "d1.json")
        path2 = str(tmp_path / "d2.json")

        d1 = build_stability_dataset(graphs, output_path=path1, base_seed=42)
        d2 = build_stability_dataset(graphs, output_path=path2, base_seed=42)

        assert d1 == d2


# ── Part 2: Stability Boundary Estimator ─────────────────────────


class TestStabilityBoundary:
    def test_estimate_boundary_returns_valid_weights(self):
        from qec.diagnostics.stability_boundary import (
            estimate_stability_boundary,
        )

        dataset = _sample_dataset()
        result = estimate_stability_boundary(dataset)

        assert "weights" in result
        assert "bias" in result
        assert "accuracy" in result
        assert isinstance(result["weights"], list)
        assert len(result["weights"]) == 5
        assert isinstance(result["bias"], float)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_estimate_boundary_empty_dataset(self):
        from qec.diagnostics.stability_boundary import (
            estimate_stability_boundary,
        )

        result = estimate_stability_boundary([])
        assert result["weights"] == [0.0] * 5
        assert result["bias"] == 0.0
        assert result["accuracy"] == 0.0

    def test_estimate_boundary_determinism(self):
        from qec.diagnostics.stability_boundary import (
            estimate_stability_boundary,
        )

        dataset = _sample_dataset()
        r1 = estimate_stability_boundary(dataset)
        r2 = estimate_stability_boundary(dataset)
        assert r1 == r2


# ── Part 3: Stability Predictor ──────────────────────────────────


class TestStabilityPredictor:
    def test_predict_bp_stability(self):
        from qec.diagnostics.stability_predictor import predict_bp_stability

        boundary = {
            "weights": [0.1, 0.2, -0.1, -0.3, 0.5],
            "bias": 0.0,
        }

        result = predict_bp_stability(_small_H(), boundary)

        assert "score" in result
        assert "predicted_converged" in result
        assert "metrics" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["predicted_converged"], bool)

    def test_predict_determinism(self):
        from qec.diagnostics.stability_predictor import predict_bp_stability

        boundary = {"weights": [0.1, 0.2, -0.1, -0.3, 0.5], "bias": 0.0}
        r1 = predict_bp_stability(_small_H(), boundary)
        r2 = predict_bp_stability(_small_H(), boundary)
        assert r1 == r2


# ── Part 4: NB Sign Trapping Set Detector ────────────────────────


class TestNBSignTrappingSets:
    def test_detect_returns_clusters(self):
        from qec.diagnostics.nb_sign_trapping_sets import (
            detect_nb_sign_trapping_sets,
        )

        result = detect_nb_sign_trapping_sets(_small_H())

        assert "candidate_trapping_sets" in result
        assert isinstance(result["candidate_trapping_sets"], list)
        for candidate in result["candidate_trapping_sets"]:
            assert "edge_indices" in candidate
            assert "nodes" in candidate
            assert "size" in candidate
            assert "mean_energy" in candidate

    def test_detect_determinism(self):
        from qec.diagnostics.nb_sign_trapping_sets import (
            detect_nb_sign_trapping_sets,
        )

        r1 = detect_nb_sign_trapping_sets(_small_H())
        r2 = detect_nb_sign_trapping_sets(_small_H())
        assert r1 == r2


# ── Part 5: Critical Radius Estimator ────────────────────────────


class TestCriticalRadius:
    def test_estimate_critical_radius(self):
        from qec.diagnostics.critical_radius import (
            estimate_critical_spectral_radius,
        )

        dataset = _sample_dataset()
        result = estimate_critical_spectral_radius(dataset)

        assert "critical_radius" in result
        assert "transition_width" in result
        assert isinstance(result["critical_radius"], float)
        assert isinstance(result["transition_width"], float)
        assert result["critical_radius"] > 0.0

    def test_critical_radius_empty_dataset(self):
        from qec.diagnostics.critical_radius import (
            estimate_critical_spectral_radius,
        )

        result = estimate_critical_spectral_radius([])
        assert result["critical_radius"] == 0.0
        assert result["transition_width"] == 0.0

    def test_critical_radius_all_converged(self):
        from qec.diagnostics.critical_radius import (
            estimate_critical_spectral_radius,
        )

        dataset = [
            {"spectral_radius": 1.0, "bp_converged": 1},
            {"spectral_radius": 1.5, "bp_converged": 1},
        ]
        result = estimate_critical_spectral_radius(dataset)
        assert result["transition_width"] == 0.0

    def test_critical_radius_all_failed(self):
        from qec.diagnostics.critical_radius import (
            estimate_critical_spectral_radius,
        )

        dataset = [
            {"spectral_radius": 1.0, "bp_converged": 0},
            {"spectral_radius": 1.5, "bp_converged": 0},
        ]
        result = estimate_critical_spectral_radius(dataset)
        assert result["transition_width"] == 0.0

    def test_critical_radius_midpoint(self):
        from qec.diagnostics.critical_radius import (
            estimate_critical_spectral_radius,
        )

        dataset = [
            {"spectral_radius": 1.0, "bp_converged": 1},
            {"spectral_radius": 2.0, "bp_converged": 0},
        ]
        result = estimate_critical_spectral_radius(dataset)
        assert result["critical_radius"] == 1.5
        assert result["transition_width"] == 1.0


# ── Part 6: Spectral Critical Line Estimator ─────────────────────


class TestSpectralCriticalLine:
    def test_predict_spectral_critical_radius(self):
        from qec.diagnostics.spectral_critical_line import (
            predict_spectral_critical_radius,
        )

        result = predict_spectral_critical_radius(_small_H())

        assert "predicted_critical_radius" in result
        assert isinstance(result["predicted_critical_radius"], float)

    def test_spectral_critical_determinism(self):
        from qec.diagnostics.spectral_critical_line import (
            predict_spectral_critical_radius,
        )

        r1 = predict_spectral_critical_radius(_small_H())
        r2 = predict_spectral_critical_radius(_small_H())
        assert r1 == r2


# ── Part 7: Repair Stability Trajectory ──────────────────────────


class TestRepairStabilityTrajectory:
    def test_trajectory_schema(self, tmp_path):
        from qec.experiments.repair_stability_trajectory import (
            track_repair_stability_trajectory,
        )

        output_path = str(tmp_path / "trajectory.json")
        trajectory = track_repair_stability_trajectory(
            _small_H(),
            repair_steps=2,
            base_seed=42,
            samples_per_step=2,
            output_path=output_path,
        )

        assert len(trajectory) == 3  # step 0, 1, 2
        for entry in trajectory:
            assert "step" in entry
            assert "critical_radius" in entry
            assert "transition_width" in entry

    def test_trajectory_artifact_saved(self, tmp_path):
        from qec.experiments.repair_stability_trajectory import (
            track_repair_stability_trajectory,
        )

        output_path = str(tmp_path / "trajectory.json")
        track_repair_stability_trajectory(
            _small_H(),
            repair_steps=1,
            samples_per_step=2,
            output_path=output_path,
        )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)


# ── Part 8: ASCII Boundary Rendering ─────────────────────────────


class TestASCIIBoundaryRendering:
    def test_render_ascii_stability_boundary(self):
        from qec.experiments.stability_phase_diagram import (
            render_ascii_stability_boundary,
        )

        grid_results = [
            {
                "spectral_radius_bin": 0,
                "sis_bin": 0,
                "spectral_radius_center": 0.9,
                "sis_center": 0.01,
                "convergence_rate": 1.0,
            },
            {
                "spectral_radius_bin": 1,
                "sis_bin": 0,
                "spectral_radius_center": 1.1,
                "sis_center": 0.01,
                "convergence_rate": 0.0,
            },
        ]

        diagram = render_ascii_stability_boundary(
            grid_results, grid_resolution=2, critical_radius=1.0,
        )

        assert "Stability Phase Diagram" in diagram
        assert "|" in diagram
        assert "Critical radius" in diagram


# ── Part 10: Benchmark Experiment ─────────────────────────────────


class TestStabilityPredictionBenchmark:
    def test_benchmark_runs(self, tmp_path):
        from qec.experiments.stability_prediction_benchmark import (
            run_stability_prediction_benchmark,
        )

        output_path = str(tmp_path / "benchmark.json")
        result = run_stability_prediction_benchmark(
            num_graphs=3,
            base_seed=42,
            output_path=output_path,
        )

        assert "accuracy" in result
        assert "false_positive_rate" in result
        assert "false_negative_rate" in result
        assert "critical_radius" in result
        assert "spectral_prediction_error" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_benchmark_artifact_saved(self, tmp_path):
        from qec.experiments.stability_prediction_benchmark import (
            run_stability_prediction_benchmark,
        )

        output_path = str(tmp_path / "benchmark.json")
        run_stability_prediction_benchmark(
            num_graphs=2,
            base_seed=42,
            output_path=output_path,
        )

        assert os.path.exists(output_path)


# ── Part 11: API Integration ─────────────────────────────────────


class TestAPIIntegration:
    def test_estimate_stability_boundary_in_api(self):
        from qec.diagnostics.api import estimate_stability_boundary

        result = estimate_stability_boundary(_sample_dataset())
        assert "weights" in result

    def test_predict_bp_stability_in_api(self):
        from qec.diagnostics.api import predict_bp_stability

        boundary = {"weights": [0.1, 0.2, -0.1, -0.3, 0.5], "bias": 0.0}
        result = predict_bp_stability(_small_H(), boundary)
        assert "predicted_converged" in result

    def test_detect_nb_sign_trapping_sets_in_api(self):
        from qec.diagnostics.api import detect_nb_sign_trapping_sets

        result = detect_nb_sign_trapping_sets(_small_H())
        assert "candidate_trapping_sets" in result

    def test_estimate_critical_spectral_radius_in_api(self):
        from qec.diagnostics.api import estimate_critical_spectral_radius

        result = estimate_critical_spectral_radius(_sample_dataset())
        assert "critical_radius" in result

    def test_predict_spectral_critical_radius_in_api(self):
        from qec.diagnostics.api import predict_spectral_critical_radius

        result = predict_spectral_critical_radius(_small_H())
        assert "predicted_critical_radius" in result


# ── Layer discipline: diagnostics must not import decoder ─────────


class TestLayerDiscipline:
    def test_stability_boundary_no_decoder_import(self):
        import qec.diagnostics.stability_boundary as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_stability_predictor_no_decoder_import(self):
        import qec.diagnostics.stability_predictor as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_nb_sign_trapping_sets_no_decoder_import(self):
        import qec.diagnostics.nb_sign_trapping_sets as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_critical_radius_no_decoder_import(self):
        import qec.diagnostics.critical_radius as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_spectral_critical_line_no_decoder_import(self):
        import qec.diagnostics.spectral_critical_line as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source
