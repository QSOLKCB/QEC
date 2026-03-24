"""
Tests for v74.3.0 — Dual-Lattice Invariant Visualizer.

Covers:
- determinism (same input → identical output)
- no mutation of input data
- correct Rubik lattice structure
- correct Sierpinski lattice structure
- valid classification → valid node activation
- no NaN / invalid values
"""

from __future__ import annotations

import copy
import json
import os
import tempfile

import numpy as np
import pytest

from qec.experiments.sonic_visualizer import (
    ACTIVATION_THRESHOLD,
    CLASSIFICATION_COLORS,
    RUBIK_SIZE,
    SIERPINSKI_NODES,
    SIERPINSKI_SIZE,
    SPECTRAL_BAND_COLORS,
    build_rubik_lattice,
    build_sierpinski_lattice,
    load_analysis,
    load_sequence_analysis,
    render_combined,
    render_rubik,
    render_sierpinski,
    rubik_colors,
    visualize,
)


# ---------------------------------------------------------------------------
# Fixtures — deterministic synthetic data
# ---------------------------------------------------------------------------

def _sample_analysis() -> dict:
    """Create a minimal synthetic analysis.json matching v74.0 format."""
    return {
        "duration_seconds": 1.0,
        "sample_rate": 44100,
        "n_samples": 44100,
        "rms_energy": 0.3,
        "peak_amplitude": 0.9,
        "zero_crossing_rate": 0.05,
        "spectral_centroid_hz": 1200.0,
        "spectral_spread_hz": 400.0,
        "fft_top_peaks": [
            {"frequency_hz": 440.0, "magnitude": 0.8},
            {"frequency_hz": 880.0, "magnitude": 0.5},
            {"frequency_hz": 1320.0, "magnitude": 0.3},
            {"frequency_hz": 1760.0, "magnitude": 0.2},
            {"frequency_hz": 2200.0, "magnitude": 0.15},
            {"frequency_hz": 3000.0, "magnitude": 0.1},
            {"frequency_hz": 4000.0, "magnitude": 0.08},
            {"frequency_hz": 5000.0, "magnitude": 0.05},
            {"frequency_hz": 6000.0, "magnitude": 0.03},
            {"frequency_hz": 8000.0, "magnitude": 0.02},
        ],
        "source_file": "test_state.wav",
    }


def _sample_sequence() -> dict:
    """Create a minimal synthetic sequence_analysis.json matching v74.2 format."""
    return {
        "n_states": 3,
        "transitions": [
            {
                "from": "state_0",
                "to": "state_1",
                "from_index": 0,
                "to_index": 1,
                "metrics": {
                    "centroid_delta": 100.0,
                    "energy_delta": 0.1,
                    "energy_ratio": 1.1,
                    "fft_peak_count_change": 0,
                    "fft_similarity": 0.8,
                    "spread_delta": 50.0,
                    "spread_ratio": 1.2,
                    "zcr_delta": 0.01,
                },
                "classification": "stable",
            },
            {
                "from": "state_1",
                "to": "state_2",
                "from_index": 1,
                "to_index": 2,
                "metrics": {
                    "centroid_delta": 500.0,
                    "energy_delta": -0.3,
                    "energy_ratio": 0.5,
                    "fft_peak_count_change": -3,
                    "fft_similarity": 0.15,
                    "spread_delta": 200.0,
                    "spread_ratio": 2.0,
                    "zcr_delta": 0.05,
                },
                "classification": "collapse",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Rubik Lattice Tests
# ---------------------------------------------------------------------------

class TestRubikLattice:
    """Tests for the 8x8x8 Rubik measurement lattice."""

    def test_shape(self):
        grid = build_rubik_lattice(_sample_analysis())
        assert grid.shape == (8, 8, 8)

    def test_value_range(self):
        grid = build_rubik_lattice(_sample_analysis())
        assert np.all(grid >= 0.0)
        assert np.all(grid <= 1.0)

    def test_no_nan(self):
        grid = build_rubik_lattice(_sample_analysis())
        assert not np.any(np.isnan(grid))
        assert not np.any(np.isinf(grid))

    def test_determinism(self):
        """Same input must produce identical grids."""
        a = _sample_analysis()
        g1 = build_rubik_lattice(a)
        g2 = build_rubik_lattice(a)
        np.testing.assert_array_equal(g1, g2)

    def test_non_empty_for_valid_input(self):
        grid = build_rubik_lattice(_sample_analysis())
        assert np.any(grid > 0.0), "Grid should have active voxels"

    def test_empty_peaks(self):
        """Analysis with no peaks should produce an empty grid."""
        a = _sample_analysis()
        a["fft_top_peaks"] = []
        grid = build_rubik_lattice(a)
        assert np.all(grid == 0.0)

    def test_input_not_mutated(self):
        """Input dict must not be modified."""
        a = _sample_analysis()
        original = copy.deepcopy(a)
        build_rubik_lattice(a)
        assert a == original


class TestRubikColors:
    """Tests for Rubik voxel coloring."""

    def test_shape(self):
        grid = build_rubik_lattice(_sample_analysis())
        colors = rubik_colors(grid)
        assert colors.shape == (8, 8, 8, 4)

    def test_alpha_zero_below_threshold(self):
        grid = np.zeros((8, 8, 8), dtype=np.float64)
        colors = rubik_colors(grid)
        assert np.all(colors[:, :, :, 3] == 0.0)

    def test_spectral_band_count(self):
        assert len(SPECTRAL_BAND_COLORS) == RUBIK_SIZE

    def test_determinism(self):
        grid = build_rubik_lattice(_sample_analysis())
        c1 = rubik_colors(grid)
        c2 = rubik_colors(grid)
        np.testing.assert_array_equal(c1, c2)


# ---------------------------------------------------------------------------
# Sierpinski Lattice Tests
# ---------------------------------------------------------------------------

class TestSierpinskiLattice:
    """Tests for the 3x3x3 Sierpinski interpretation lattice."""

    def test_node_count(self):
        lattice = build_sierpinski_lattice(_sample_sequence())
        assert len(lattice["nodes"]) == len(SIERPINSKI_NODES)

    def test_node_positions_in_range(self):
        for x, y, z in SIERPINSKI_NODES:
            assert 0 <= x < SIERPINSKI_SIZE
            assert 0 <= y < SIERPINSKI_SIZE
            assert 0 <= z < SIERPINSKI_SIZE

    def test_valid_classifications(self):
        lattice = build_sierpinski_lattice(_sample_sequence())
        valid = set(CLASSIFICATION_COLORS.keys())
        for c in lattice["classifications"]:
            assert c in valid

    def test_activations_range(self):
        lattice = build_sierpinski_lattice(_sample_sequence())
        for a in lattice["activations"]:
            assert 0.0 <= a <= 1.0

    def test_colors_rgba(self):
        lattice = build_sierpinski_lattice(_sample_sequence())
        for r, g, b, a in lattice["colors"]:
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0
            assert 0.0 <= a <= 1.0

    def test_determinism(self):
        s = _sample_sequence()
        l1 = build_sierpinski_lattice(s)
        l2 = build_sierpinski_lattice(s)
        assert l1 == l2

    def test_input_not_mutated(self):
        s = _sample_sequence()
        original = copy.deepcopy(s)
        build_sierpinski_lattice(s)
        assert s == original

    def test_empty_transitions(self):
        """No transitions should still produce valid lattice."""
        s = {"n_states": 0, "transitions": []}
        lattice = build_sierpinski_lattice(s)
        assert len(lattice["nodes"]) == len(SIERPINSKI_NODES)
        # All nodes default to stable.
        assert all(c == "stable" for c in lattice["classifications"])

    def test_no_color_mixing(self):
        """Each node color must match exactly one classification color."""
        lattice = build_sierpinski_lattice(_sample_sequence())
        for i, (r, g, b, a) in enumerate(lattice["colors"]):
            label = lattice["classifications"][i]
            expected_r, expected_g, expected_b = CLASSIFICATION_COLORS[label]
            assert r == expected_r
            assert g == expected_g
            assert b == expected_b


# ---------------------------------------------------------------------------
# Data Loading Tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests for JSON loading functions."""

    def test_load_analysis_returns_copy(self, tmp_path):
        data = _sample_analysis()
        p = tmp_path / "analysis.json"
        p.write_text(json.dumps(data, sort_keys=True))
        loaded = load_analysis(str(p))
        loaded["rms_energy"] = 999.0
        # Reload should return original value.
        reloaded = load_analysis(str(p))
        assert reloaded["rms_energy"] == data["rms_energy"]

    def test_load_sequence_returns_copy(self, tmp_path):
        data = _sample_sequence()
        p = tmp_path / "sequence.json"
        p.write_text(json.dumps(data, sort_keys=True))
        loaded = load_sequence_analysis(str(p))
        loaded["transitions"].clear()
        reloaded = load_sequence_analysis(str(p))
        assert len(reloaded["transitions"]) == 2


# ---------------------------------------------------------------------------
# Rendering Tests
# ---------------------------------------------------------------------------

class TestRendering:
    """Tests for PNG rendering output."""

    def test_render_rubik_creates_file(self, tmp_path):
        grid = build_rubik_lattice(_sample_analysis())
        out = str(tmp_path / "rubik.png")
        result = render_rubik(grid, out)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_render_sierpinski_creates_file(self, tmp_path):
        lattice = build_sierpinski_lattice(_sample_sequence())
        out = str(tmp_path / "sierpinski.png")
        result = render_sierpinski(lattice, out)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_render_combined_creates_file(self, tmp_path):
        grid = build_rubik_lattice(_sample_analysis())
        lattice = build_sierpinski_lattice(_sample_sequence())
        out = str(tmp_path / "combined.png")
        result = render_combined(grid, lattice, out)
        assert os.path.isfile(result)
        assert os.path.getsize(result) > 0

    def test_render_determinism(self, tmp_path):
        """Same input should produce identical PNG bytes."""
        grid = build_rubik_lattice(_sample_analysis())
        out1 = str(tmp_path / "r1.png")
        out2 = str(tmp_path / "r2.png")
        render_rubik(grid, out1)
        render_rubik(grid, out2)
        with open(out1, "rb") as f1, open(out2, "rb") as f2:
            assert f1.read() == f2.read()


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------

class TestVisualizePipeline:
    """Tests for the full visualize() pipeline."""

    def test_full_pipeline(self, tmp_path):
        # Write synthetic input files.
        analysis = _sample_analysis()
        sequence = _sample_sequence()

        state_dir = tmp_path / "sonic" / "test_state"
        state_dir.mkdir(parents=True)
        (state_dir / "analysis.json").write_text(
            json.dumps(analysis, sort_keys=True)
        )
        (tmp_path / "sonic" / "sequence_analysis.json").write_text(
            json.dumps(sequence, sort_keys=True)
        )

        out_dir = str(tmp_path / "output")
        outputs = visualize(
            [str(state_dir / "analysis.json")],
            str(tmp_path / "sonic" / "sequence_analysis.json"),
            output_dir=out_dir,
        )

        assert "sierpinski_state" in outputs
        assert "combined_view" in outputs
        for path in outputs.values():
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
