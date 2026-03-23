"""Tests for phase spectral analysis (v86.0.0)."""

import numpy as np
import pytest

from qec.experiments.phase_spectral_analysis import (
    analyze_phase_spectrum,
    build_phase_matrix,
    run_phase_spectral_analysis,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_phase_map():
    """Three-node linear graph: 0 --1.0-- 1 --2.0-- 2."""
    return {
        "nodes": [
            {"id": 0, "mean_score": 0.5},
            {"id": 1, "mean_score": 0.6},
            {"id": 2, "mean_score": 0.7},
        ],
        "edges": [
            {"source": 0, "target": 1, "weight": 1.0},
            {"source": 1, "target": 2, "weight": 2.0},
        ],
    }


def _single_node_phase_map():
    return {"nodes": [{"id": 0}], "edges": []}


def _empty_phase_map():
    return {"nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# Tests — matrix construction
# ---------------------------------------------------------------------------

class TestBuildPhaseMatrix:
    def test_shape(self):
        result = build_phase_matrix(_simple_phase_map())
        assert result["matrix"].shape == (3, 3)
        assert result["n"] == 3

    def test_symmetry(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        np.testing.assert_array_equal(m, m.T)

    def test_weights_correct(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        assert m[0, 1] == 1.0
        assert m[1, 0] == 1.0
        assert m[1, 2] == 2.0
        assert m[2, 1] == 2.0
        assert m[0, 2] == 0.0

    def test_empty(self):
        result = build_phase_matrix(_empty_phase_map())
        assert result["n"] == 0
        assert result["matrix"].shape == (0, 0)

    def test_single_node(self):
        result = build_phase_matrix(_single_node_phase_map())
        assert result["n"] == 1
        assert result["matrix"].shape == (1, 1)
        assert result["matrix"][0, 0] == 0.0


# ---------------------------------------------------------------------------
# Tests — spectral analysis
# ---------------------------------------------------------------------------

class TestAnalyzePhaseSpectrum:
    def test_rank_simple(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        spec = analyze_phase_spectrum(m)
        assert spec["rank"] == 2  # 3x3 matrix with rank 2

    def test_eigenvalues_sorted_descending(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        spec = analyze_phase_spectrum(m)
        eigvals = spec["eigenvalues"]
        for i in range(len(eigvals) - 1):
            assert eigvals[i] >= eigvals[i + 1]

    def test_singular_values_nonnegative(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        spec = analyze_phase_spectrum(m)
        for sv in spec["singular_values"]:
            assert sv >= -1e-15  # numerical tolerance

    def test_degeneracy_identity(self):
        # Identity matrix: all eigenvalues = 1, no zero modes.
        m = np.eye(3, dtype=np.float64)
        spec = analyze_phase_spectrum(m)
        assert spec["n_degenerate_modes"] == 0

    def test_degeneracy_zero_matrix(self):
        m = np.zeros((3, 3), dtype=np.float64)
        spec = analyze_phase_spectrum(m)
        assert spec["n_degenerate_modes"] == 3

    def test_spectral_gap(self):
        m = build_phase_matrix(_simple_phase_map())["matrix"]
        spec = analyze_phase_spectrum(m)
        eigvals = spec["eigenvalues"]
        expected_gap = eigvals[0] - eigvals[1]
        assert abs(spec["spectral_gap"] - expected_gap) < 1e-12

    def test_empty_spectrum(self):
        m = np.zeros((0, 0), dtype=np.float64)
        spec = analyze_phase_spectrum(m)
        assert spec["rank"] == 0
        assert spec["eigenvalues"] == []
        assert spec["max_eigenvalue"] == 0.0


# ---------------------------------------------------------------------------
# Tests — wrapper
# ---------------------------------------------------------------------------

class TestRunPhaseSpectralAnalysis:
    def test_output_keys(self):
        result = run_phase_spectral_analysis(_simple_phase_map())
        assert "matrix_shape" in result
        assert "spectrum" in result
        assert result["matrix_shape"] == (3, 3)

    def test_determinism(self):
        r1 = run_phase_spectral_analysis(_simple_phase_map())
        r2 = run_phase_spectral_analysis(_simple_phase_map())
        assert r1["spectrum"]["eigenvalues"] == r2["spectrum"]["eigenvalues"]
        assert r1["spectrum"]["singular_values"] == r2["spectrum"]["singular_values"]
        assert r1["spectrum"]["rank"] == r2["spectrum"]["rank"]
