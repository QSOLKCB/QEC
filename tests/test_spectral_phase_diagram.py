"""
Tests for v12.5.0 — Spectral Instability Phase Diagram Generator.

Validates:
  - Deterministic outputs (identical results on repeated runs)
  - Correct metric extraction
  - Correct JSON schema of phase diagram points
  - Stable spectral ordering
  - FER computation correctness
  - ASCII heatmap rendering
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from src.qec.experiments.spectral_phase_diagram import (
    SpectralPhaseDiagramGenerator,
    extract_spectral_metrics,
    _derive_seed,
    _run_decoder_trial,
)
from src.qec.experiments.phase_diagram_plot import (
    render_ascii_heatmap,
)


# ── Test fixtures ────────────────────────────────────────────────


def _make_small_H() -> np.ndarray:
    """Create a small deterministic parity-check matrix."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def _make_two_graphs() -> list[np.ndarray]:
    """Create two small deterministic parity-check matrices."""
    H1 = np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)
    H2 = np.array([
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 1],
    ], dtype=np.float64)
    return [H1, H2]


# ── Seed derivation tests ────────────────────────────────────────


class TestDeriveSeeed:
    def test_deterministic(self):
        s1 = _derive_seed(42, "test_label")
        s2 = _derive_seed(42, "test_label")
        assert s1 == s2

    def test_different_labels_differ(self):
        s1 = _derive_seed(42, "label_a")
        s2 = _derive_seed(42, "label_b")
        assert s1 != s2

    def test_different_seeds_differ(self):
        s1 = _derive_seed(42, "label")
        s2 = _derive_seed(43, "label")
        assert s1 != s2

    def test_within_range(self):
        s = _derive_seed(42, "test")
        assert 0 <= s < 2**31


# ── Spectral metric extraction tests ─────────────────────────────


class TestExtractSpectralMetrics:
    def test_returns_expected_keys(self):
        H = _make_small_H()
        metrics = extract_spectral_metrics(H)
        assert "spectral_radius" in metrics
        assert "ipr" in metrics
        assert "flow_alignment" in metrics

    def test_deterministic(self):
        H = _make_small_H()
        m1 = extract_spectral_metrics(H)
        m2 = extract_spectral_metrics(H)
        assert m1 == m2

    def test_spectral_radius_nonnegative(self):
        H = _make_small_H()
        metrics = extract_spectral_metrics(H)
        assert metrics["spectral_radius"] >= 0.0

    def test_ipr_in_range(self):
        H = _make_small_H()
        metrics = extract_spectral_metrics(H)
        assert 0.0 <= metrics["ipr"] <= 1.0


# ── Decoder trial tests ──────────────────────────────────────────


class TestDecoderTrial:
    def test_deterministic(self):
        H = _make_small_H()
        r1 = _run_decoder_trial(H, 0.05, 42)
        r2 = _run_decoder_trial(H, 0.05, 42)
        assert r1 == r2

    def test_returns_expected_keys(self):
        H = _make_small_H()
        result = _run_decoder_trial(H, 0.05, 42)
        assert "success" in result
        assert "iterations" in result
        assert "residual_norm" in result

    def test_success_is_bool(self):
        H = _make_small_H()
        result = _run_decoder_trial(H, 0.05, 42)
        assert isinstance(result["success"], bool)


# ── Phase diagram generator tests ────────────────────────────────


class TestSpectralPhaseDiagramGenerator:
    def test_basic_generation(self):
        graphs = _make_two_graphs()
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.02, 0.05],
            trials_per_point=3,
        )
        assert "points" in result
        assert len(result["points"]) == 4  # 2 graphs x 2 error rates

    def test_deterministic(self):
        graphs = _make_two_graphs()
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        r1 = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.05],
            trials_per_point=3,
        )
        r2 = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.05],
            trials_per_point=3,
        )
        assert r1 == r2

    def test_point_schema(self):
        graphs = [_make_small_H()]
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.05],
            trials_per_point=3,
        )
        point = result["points"][0]
        assert "spectral_radius" in point
        assert "error_rate" in point
        assert "FER" in point
        assert "IPR" in point
        assert "flow_alignment" in point

    def test_fer_in_range(self):
        graphs = [_make_small_H()]
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.05],
            trials_per_point=5,
        )
        for point in result["points"]:
            assert 0.0 <= point["FER"] <= 1.0

    def test_spectral_ordering(self):
        """Points are sorted by (spectral_radius, error_rate)."""
        graphs = _make_two_graphs()
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.02, 0.05, 0.10],
            trials_per_point=2,
        )
        points = result["points"]
        for i in range(len(points) - 1):
            key_i = (points[i]["spectral_radius"], points[i]["error_rate"])
            key_j = (points[i + 1]["spectral_radius"], points[i + 1]["error_rate"])
            assert key_i <= key_j

    def test_json_serializable(self):
        graphs = [_make_small_H()]
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.05],
            trials_per_point=2,
        )
        json_str = json.dumps(result, sort_keys=True)
        roundtrip = json.loads(json_str)
        assert roundtrip == result

    def test_empty_graphs(self):
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=[],
            error_rates=[0.05],
            trials_per_point=2,
        )
        assert result["points"] == []

    def test_empty_error_rates(self):
        graphs = [_make_small_H()]
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[],
            trials_per_point=2,
        )
        assert result["points"] == []


# ── ASCII heatmap tests ──────────────────────────────────────────


class TestAsciiHeatmap:
    def test_renders_nonempty(self):
        graphs = _make_two_graphs()
        gen = SpectralPhaseDiagramGenerator(base_seed=42)
        result = gen.generate_phase_diagram(
            graphs=graphs,
            error_rates=[0.02, 0.05],
            trials_per_point=2,
        )
        ascii_out = render_ascii_heatmap(result)
        assert "Spectral Instability Phase Diagram" in ascii_out
        assert len(ascii_out) > 0

    def test_empty_data(self):
        ascii_out = render_ascii_heatmap({"points": []})
        assert "(no data)" in ascii_out
