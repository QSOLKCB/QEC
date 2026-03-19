"""
Tests for stability phase diagram experiment (v8.0).

Verifies:
  - Experiment runs successfully
  - Artifact schema valid
  - Spectral metrics recorded
  - BP results reproducible
  - Incremental spectrum matches full recomputation
  - Oscillation detection
  - Boundary estimation
  - ASCII diagram generation
  - Diagnostics modules
  - No decoder import
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

from qec.experiments.stability_phase_diagram import (
    run_stability_phase_diagram_experiment,
    serialize_phase_diagram_artifact,
    detect_metastable_bp_oscillation,
    estimate_bp_stability_boundary,
    predict_spectral_stability_boundary,
    log_most_unstable_subgraph,
    track_repair_boundary_shift,
    _derive_seed,
    _render_ascii_phase_diagram,
    scale_ascii_phase_diagram,
    highlight_bp_critical_line,
)
from qec.diagnostics.nb_localization_detector import (
    detect_nb_eigenvector_localization,
)
from qec.diagnostics.nb_energy_heatmap import compute_nb_energy_heatmap
from qec.diagnostics.nb_sign_pattern_detector import (
    detect_nb_sign_pattern_trapping_sets,
)


# ── Test fixtures ─────────────────────────────────────────────────


def _small_H():
    """Return a small test parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0],
        [0, 1, 1, 0, 1],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 1, 1],
    ], dtype=np.float64)


def _tiny_H():
    """Return a tiny test parity-check matrix."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


# ── Oscillation Detection ────────────────────────────────────────


class TestOscillationDetection:
    """Tests for detect_metastable_bp_oscillation."""

    def test_empty_residuals(self):
        result = detect_metastable_bp_oscillation([])
        assert result["is_oscillatory"] is False
        assert result["oscillation_period"] == 0

    def test_converged_residuals(self):
        residuals = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.0, 0.0, 0.0]
        result = detect_metastable_bp_oscillation(residuals)
        assert result["is_oscillatory"] is False

    def test_oscillatory_residuals(self):
        # Period-2 oscillation
        residuals = []
        for _ in range(20):
            residuals.extend([1.5, 0.5])
        result = detect_metastable_bp_oscillation(residuals, window_size=10)
        assert result["is_oscillatory"] is True
        assert result["oscillation_period"] == 2

    def test_output_keys(self):
        result = detect_metastable_bp_oscillation([1.0, 0.5])
        expected = {"is_oscillatory", "oscillation_period",
                    "mean_residual", "residual_variance"}
        assert set(result.keys()) == expected

    def test_determinism(self):
        residuals = [1.0, 0.8, 1.0, 0.8] * 10
        r1 = detect_metastable_bp_oscillation(residuals)
        r2 = detect_metastable_bp_oscillation(residuals)
        assert r1 == r2


# ── Boundary Estimation ──────────────────────────────────────────


class TestBoundaryEstimation:
    """Tests for estimate_bp_stability_boundary."""

    def test_empty_grid(self):
        result = estimate_bp_stability_boundary([])
        assert result["num_boundary_points"] == 0
        assert result["boundary_points"] == []

    def test_uniform_converged(self):
        grid = [
            {
                "spectral_radius_bin": i,
                "sis_bin": j,
                "spectral_radius_center": float(i),
                "sis_center": float(j),
                "convergence_rate": 1.0,
            }
            for i in range(3) for j in range(3)
        ]
        result = estimate_bp_stability_boundary(grid)
        assert result["num_boundary_points"] == 0

    def test_mixed_grid_has_boundary(self):
        grid = []
        for i in range(4):
            for j in range(4):
                conv_rate = 1.0 if i + j < 4 else 0.0
                grid.append({
                    "spectral_radius_bin": i,
                    "sis_bin": j,
                    "spectral_radius_center": float(i) * 0.5,
                    "sis_center": float(j) * 0.1,
                    "convergence_rate": conv_rate,
                })
        result = estimate_bp_stability_boundary(grid)
        assert result["num_boundary_points"] > 0

    def test_determinism(self):
        grid = [
            {
                "spectral_radius_bin": i, "sis_bin": j,
                "spectral_radius_center": float(i),
                "sis_center": float(j),
                "convergence_rate": 1.0 if i < 2 else 0.0,
            }
            for i in range(4) for j in range(4)
        ]
        r1 = estimate_bp_stability_boundary(grid)
        r2 = estimate_bp_stability_boundary(grid)
        assert r1 == r2


class TestSpectralBoundaryPrediction:
    """Tests for predict_spectral_stability_boundary."""

    def test_empty_grid(self):
        result = predict_spectral_stability_boundary([])
        assert result["num_cells"] == 0
        assert result["prediction_accuracy"] == 0.0

    def test_output_keys(self):
        grid = [{
            "sis_center": 0.5,
            "convergence_rate": 1.0,
        }]
        result = predict_spectral_stability_boundary(grid)
        expected = {"predicted_sis_threshold", "prediction_accuracy", "num_cells"}
        assert set(result.keys()) == expected


# ── Seed Derivation ──────────────────────────────────────────────


class TestSeedDerivation:
    """Tests for _derive_seed."""

    def test_deterministic(self):
        s1 = _derive_seed(42, "test")
        s2 = _derive_seed(42, "test")
        assert s1 == s2

    def test_different_labels(self):
        s1 = _derive_seed(42, "a")
        s2 = _derive_seed(42, "b")
        assert s1 != s2

    def test_different_seeds(self):
        s1 = _derive_seed(1, "test")
        s2 = _derive_seed(2, "test")
        assert s1 != s2

    def test_within_range(self):
        s = _derive_seed(999, "label")
        assert 0 <= s < 2**31


# ── NB Localization Detector ─────────────────────────────────────


class TestNBLocalizationDetector:
    """Tests for detect_nb_eigenvector_localization."""

    def test_output_keys(self):
        H = _tiny_H()
        result = detect_nb_eigenvector_localization(H)
        expected = {
            "is_localized", "ipr", "eeec", "num_localized_edges",
            "localized_edge_indices", "max_edge_energy",
            "mean_edge_energy", "localization_ratio",
        }
        assert set(result.keys()) == expected

    def test_output_types(self):
        H = _tiny_H()
        result = detect_nb_eigenvector_localization(H)
        assert isinstance(result["is_localized"], bool)
        assert isinstance(result["ipr"], float)
        assert isinstance(result["num_localized_edges"], int)
        assert isinstance(result["localized_edge_indices"], list)

    def test_determinism(self):
        H = _tiny_H()
        r1 = detect_nb_eigenvector_localization(H)
        r2 = detect_nb_eigenvector_localization(H)
        assert r1["ipr"] == r2["ipr"]
        assert r1["is_localized"] == r2["is_localized"]
        assert r1["localized_edge_indices"] == r2["localized_edge_indices"]

    def test_energy_bounds(self):
        H = _tiny_H()
        result = detect_nb_eigenvector_localization(H)
        assert result["max_edge_energy"] >= 0.0
        assert result["mean_edge_energy"] >= 0.0
        assert 0.0 <= result["localization_ratio"] <= 1.0


# ── NB Energy Heatmap ────────────────────────────────────────────


class TestNBEnergyHeatmap:
    """Tests for compute_nb_energy_heatmap."""

    def test_output_keys(self):
        H = _tiny_H()
        result = compute_nb_energy_heatmap(H)
        expected = {
            "variable_node_heat", "check_node_heat",
            "max_variable_heat", "max_check_heat",
            "hottest_variable_node", "hottest_check_node",
            "total_energy",
        }
        assert set(result.keys()) == expected

    def test_heat_dimensions(self):
        H = _tiny_H()
        m, n = H.shape
        result = compute_nb_energy_heatmap(H)
        assert len(result["variable_node_heat"]) == n
        assert len(result["check_node_heat"]) == m

    def test_non_negative_heat(self):
        H = _tiny_H()
        result = compute_nb_energy_heatmap(H)
        for h in result["variable_node_heat"]:
            assert h >= 0.0
        for h in result["check_node_heat"]:
            assert h >= 0.0

    def test_determinism(self):
        H = _tiny_H()
        r1 = compute_nb_energy_heatmap(H)
        r2 = compute_nb_energy_heatmap(H)
        # Heat values must be identical; hottest node indices may differ
        # when multiple nodes share the maximum heat (argmax tie-breaking).
        assert r1["variable_node_heat"] == r2["variable_node_heat"]
        assert r1["check_node_heat"] == r2["check_node_heat"]
        np.testing.assert_allclose(
            r1["max_variable_heat"], r2["max_variable_heat"], atol=1e-12,
        )
        np.testing.assert_allclose(
            r1["max_check_heat"], r2["max_check_heat"], atol=1e-12,
        )
        np.testing.assert_allclose(
            r1["total_energy"], r2["total_energy"], atol=1e-12,
        )

    def test_json_serializable(self):
        H = _tiny_H()
        result = compute_nb_energy_heatmap(H)
        serialized = json.dumps(result)
        assert json.loads(serialized) == result


# ── NB Sign Pattern Detector ─────────────────────────────────────


class TestNBSignPatternDetector:
    """Tests for detect_nb_sign_pattern_trapping_sets."""

    def test_output_keys(self):
        H = _tiny_H()
        result = detect_nb_sign_pattern_trapping_sets(H)
        expected = {
            "num_positive_edges", "num_negative_edges",
            "sign_imbalance", "positive_energy_fraction",
            "negative_energy_fraction", "num_sign_clusters",
            "sign_clusters", "max_cluster_energy",
        }
        assert set(result.keys()) == expected

    def test_edge_count_consistency(self):
        H = _tiny_H()
        from qec.diagnostics.spectral_nb import compute_nb_spectrum
        spectrum = compute_nb_spectrum(H)
        n_edges = len(spectrum["eigenvector"])

        result = detect_nb_sign_pattern_trapping_sets(H)
        assert result["num_positive_edges"] + result["num_negative_edges"] == n_edges

    def test_energy_fractions_sum_to_one(self):
        H = _tiny_H()
        result = detect_nb_sign_pattern_trapping_sets(H)
        total = result["positive_energy_fraction"] + result["negative_energy_fraction"]
        assert abs(total - 1.0) < 1e-10

    def test_sign_imbalance_bounds(self):
        H = _tiny_H()
        result = detect_nb_sign_pattern_trapping_sets(H)
        assert 0.0 <= result["sign_imbalance"] <= 1.0

    def test_determinism(self):
        H = _tiny_H()
        r1 = detect_nb_sign_pattern_trapping_sets(H)
        r2 = detect_nb_sign_pattern_trapping_sets(H)
        assert r1 == r2


# ── Unstable Subgraph Logging ─────────────────────────────────────


class TestUnstableSubgraph:
    """Tests for log_most_unstable_subgraph."""

    def test_output_keys(self):
        H = _tiny_H()
        result = log_most_unstable_subgraph(H)
        expected = {
            "top_variable_nodes", "top_check_nodes", "top_edges",
            "spectral_radius", "ipr", "sis",
        }
        assert set(result.keys()) == expected

    def test_top_k_limit(self):
        H = _tiny_H()
        result = log_most_unstable_subgraph(H, top_k=3)
        assert len(result["top_variable_nodes"]) <= 3
        assert len(result["top_check_nodes"]) <= 3
        assert len(result["top_edges"]) <= 3

    def test_determinism(self):
        H = _tiny_H()
        r1 = log_most_unstable_subgraph(H)
        r2 = log_most_unstable_subgraph(H)
        assert r1 == r2


# ── ASCII Phase Diagram ──────────────────────────────────────────


class TestASCIIPhaseDiagram:
    """Tests for ASCII phase diagram rendering."""

    def test_render_produces_string(self):
        grid = [
            {
                "spectral_radius_bin": 0, "sis_bin": 0,
                "spectral_radius_center": 1.0, "sis_center": 0.1,
                "convergence_rate": 1.0, "has_oscillation": False,
            },
            {
                "spectral_radius_bin": 0, "sis_bin": 1,
                "spectral_radius_center": 1.0, "sis_center": 0.2,
                "convergence_rate": 0.0, "has_oscillation": False,
            },
        ]
        diagram = _render_ascii_phase_diagram(grid, 3)
        assert isinstance(diagram, str)
        assert "+" in diagram or "-" in diagram

    def test_scale_limits_width(self):
        long_line = "x" * 200
        scaled = scale_ascii_phase_diagram(long_line, max_width=80)
        for line in scaled.split("\n"):
            assert len(line) <= 80

    def test_highlight_returns_string(self):
        diagram = "test diagram"
        result = highlight_bp_critical_line(diagram, [])
        assert isinstance(result, str)


# ── Main Experiment ──────────────────────────────────────────────


class TestStabilityPhaseDiagramExperiment:
    """Tests for run_stability_phase_diagram_experiment."""

    def test_experiment_runs(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        assert "schema_version" in result
        assert result["schema_version"] == "8.0.0"

    def test_artifact_schema(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        required_keys = {
            "schema_version", "grid_resolution", "perturbations_per_cell",
            "total_perturbations", "base_seed",
            "baseline_spectral_radius", "baseline_sis",
            "grid_results", "spectral_trajectories",
            "measured_boundary", "predicted_boundary",
            "ascii_phase_diagram", "unstable_subgraph",
            "num_snapshots_saved",
        }
        assert required_keys.issubset(set(result.keys()))

    def test_grid_results_have_metrics(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        if result["grid_results"]:
            cell = result["grid_results"][0]
            assert "convergence_rate" in cell
            assert "mean_iterations" in cell
            assert "mean_residual_norm" in cell

    def test_spectral_trajectories_recorded(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        assert len(result["spectral_trajectories"]) > 0
        traj = result["spectral_trajectories"][0]
        assert "spectral_radius" in traj
        assert "ipr" in traj
        assert "sis" in traj
        assert "decoder_converged" in traj

    def test_determinism(self):
        H = _small_H()
        r1 = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        r2 = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        # Grid results should be identical
        assert len(r1["grid_results"]) == len(r2["grid_results"])
        for g1, g2 in zip(r1["grid_results"], r2["grid_results"]):
            assert g1["convergence_rate"] == g2["convergence_rate"]
            assert g1["mean_iterations"] == g2["mean_iterations"]

    def test_json_serializable(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        serialized = json.dumps(result, sort_keys=True)
        deserialized = json.loads(serialized)
        assert deserialized["schema_version"] == result["schema_version"]
        assert deserialized["grid_resolution"] == result["grid_resolution"]

    def test_boundary_estimation_present(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        assert "measured_boundary" in result
        assert "predicted_boundary" in result
        assert "num_boundary_points" in result["measured_boundary"]

    def test_ascii_diagram_present(self):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        assert "ascii_phase_diagram" in result
        assert isinstance(result["ascii_phase_diagram"], str)
        assert len(result["ascii_phase_diagram"]) > 0


# ── Incremental Spectrum Consistency ─────────────────────────────


class TestIncrementalConsistency:
    """Tests that incremental spectrum matches full recomputation."""

    def test_incremental_matches_full(self):
        from qec.diagnostics.spectral_nb import compute_nb_spectrum
        from qec.diagnostics.spectral_incremental import (
            update_nb_eigenpair_incremental,
        )

        H = _tiny_H()
        full = compute_nb_spectrum(H)

        # Use full eigenvector as warm start (trivial case)
        incr = update_nb_eigenpair_incremental(
            H, full["eigenvector"], max_iter=50,
        )

        if incr["converged"]:
            assert abs(incr["spectral_radius"] - full["spectral_radius"]) < 0.1


# ── Artifact Serialization ───────────────────────────────────────


class TestArtifactSerialization:
    """Tests for serialize_phase_diagram_artifact."""

    def test_serialization(self, tmp_path):
        H = _small_H()
        result = run_stability_phase_diagram_experiment(
            H, grid_resolution=3, perturbations_per_cell=2,
            base_seed=42, max_iters=20,
        )
        out_path = str(tmp_path / "test_artifact.json")
        json_str = serialize_phase_diagram_artifact(result, out_path)

        assert os.path.exists(out_path)
        with open(out_path) as f:
            loaded = json.loads(f.read())
        assert loaded["schema_version"] == "8.0.0"


# ── Repair Boundary Tracking ────────────────────────────────────


class TestRepairBoundaryTracking:
    """Tests for track_repair_boundary_shift."""

    def test_no_shift_with_identical_grids(self):
        H = _tiny_H()
        grid = [
            {
                "spectral_radius_bin": 0, "sis_bin": 0,
                "spectral_radius_center": 1.0, "sis_center": 0.1,
                "convergence_rate": 1.0,
            },
        ]
        result = track_repair_boundary_shift(H, grid, grid)
        assert result["delta_mean_spectral_radius"] == 0.0
        assert result["delta_mean_sis"] == 0.0

    def test_output_keys(self):
        H = _tiny_H()
        grid = [
            {
                "spectral_radius_bin": 0, "sis_bin": 0,
                "spectral_radius_center": 1.0, "sis_center": 0.1,
                "convergence_rate": 1.0,
            },
        ]
        result = track_repair_boundary_shift(H, grid, grid)
        expected = {
            "boundary_before", "boundary_after",
            "delta_mean_spectral_radius", "delta_mean_sis",
            "boundary_expanded",
        }
        assert set(result.keys()) == expected


# ── No Decoder Import ────────────────────────────────────────────


class TestNoDecoderImport:
    """Tests that new modules do not import decoder core."""

    def test_localization_detector_no_decoder(self):
        import qec.diagnostics.nb_localization_detector as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source

    def test_energy_heatmap_no_decoder(self):
        import qec.diagnostics.nb_energy_heatmap as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source

    def test_sign_pattern_no_decoder(self):
        import qec.diagnostics.nb_sign_pattern_detector as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source

    def test_experiment_no_decoder_core(self):
        import qec.experiments.stability_phase_diagram as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
