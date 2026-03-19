"""
Tests for phase boundary analysis (v5.9.0).

Validates:
  - Adjacency-based boundary detection
  - Mixed-region detection via phase entropy
  - Critical-cell detection via boundary fraction / metastability / eps
  - Determinism across repeated runs
  - JSON roundtrip stability
  - Edge cases (single cell, uniform grid)
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from qec.diagnostics.phase_boundary_analysis import (
    analyze_phase_boundaries,
    DEFAULT_ENTROPY_THRESHOLD,
    DEFAULT_BOUNDARY_FRACTION_THRESHOLD,
    DEFAULT_METASTABILITY_THRESHOLD,
    DEFAULT_BOUNDARY_EPS_THRESHOLD,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _make_cell(
    x: float | int,
    y: float | int,
    dominant_phase: int,
    phase_entropy: float = 0.0,
    success_fraction: float = 0.0,
    boundary_fraction: float = 0.0,
    failure_fraction: float = 0.0,
    mean_metastability_score: float | None = None,
    mean_boundary_eps: float | None = None,
) -> dict[str, Any]:
    return {
        "x": x,
        "y": y,
        "trial_count": 10,
        "success_fraction": success_fraction,
        "boundary_fraction": boundary_fraction,
        "failure_fraction": failure_fraction,
        "dominant_phase": dominant_phase,
        "phase_entropy": phase_entropy,
        "mean_boundary_eps": mean_boundary_eps,
        "mean_barrier_eps": None,
        "mean_metastability_score": mean_metastability_score,
        "mean_oscillation_score": None,
        "mean_alignment_max": None,
        "mean_cluster_count": None,
    }


def _make_diagram(
    x_values: list,
    y_values: list,
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "grid_axes": {
            "x_name": "p",
            "x_values": x_values,
            "y_name": "d",
            "y_values": y_values,
        },
        "cells": cells,
    }


# ── Boundary detection tests ─────────────────────────────────────────


class TestBoundaryDetection:
    def test_uniform_grid_no_boundary(self):
        """All cells same phase → no boundary cells."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0),
            _make_cell(0.01, 5, 1, success_fraction=1.0),
            _make_cell(0.02, 3, 1, success_fraction=1.0),
            _make_cell(0.02, 5, 1, success_fraction=1.0),
        ]
        diagram = _make_diagram([0.01, 0.02], [3, 5], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_boundary_cells"] == 0
        assert result["boundary_cells"] == []

    def test_horizontal_boundary(self):
        """Phase transition along x-axis → boundary cells detected."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0),
            _make_cell(0.01, 5, 1, success_fraction=1.0),
            _make_cell(0.02, 3, -1, failure_fraction=1.0),
            _make_cell(0.02, 5, -1, failure_fraction=1.0),
        ]
        diagram = _make_diagram([0.01, 0.02], [3, 5], cells)
        result = analyze_phase_boundaries(diagram)
        # All 4 cells are adjacent to a different-phase cell.
        assert result["boundary_summary"]["num_boundary_cells"] == 4

    def test_vertical_boundary(self):
        """Phase transition along y-axis → boundary cells detected."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0),
            _make_cell(0.01, 5, -1, failure_fraction=1.0),
            _make_cell(0.02, 3, 1, success_fraction=1.0),
            _make_cell(0.02, 5, -1, failure_fraction=1.0),
        ]
        diagram = _make_diagram([0.01, 0.02], [3, 5], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_boundary_cells"] == 4

    def test_single_cell_no_boundary(self):
        """Single cell → no neighbors → no boundary."""
        cells = [_make_cell(0.01, 3, 1, success_fraction=1.0)]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_boundary_cells"] == 0

    def test_corner_boundary(self):
        """3x3 grid with center different → boundary cells."""
        x_vals = [1, 2, 3]
        y_vals = [1, 2, 3]
        cells = []
        for x in x_vals:
            for y in y_vals:
                if x == 2 and y == 2:
                    cells.append(_make_cell(x, y, -1, failure_fraction=1.0))
                else:
                    cells.append(_make_cell(x, y, 1, success_fraction=1.0))
        diagram = _make_diagram(x_vals, y_vals, cells)
        result = analyze_phase_boundaries(diagram)
        # Center cell and its 4 neighbors are boundary cells.
        boundary_coords = [(c["x"], c["y"]) for c in result["boundary_cells"]]
        assert (2, 2) in boundary_coords
        assert (1, 2) in boundary_coords
        assert (3, 2) in boundary_coords
        assert (2, 1) in boundary_coords
        assert (2, 3) in boundary_coords


# ── Mixed-region detection tests ──────────────────────────────────────


class TestMixedRegionDetection:
    def test_high_entropy_detected(self):
        """Cell with entropy > threshold is mixed."""
        cells = [
            _make_cell(0.01, 3, 0, phase_entropy=0.8, boundary_fraction=0.5),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_mixed_cells"] == 1

    def test_low_entropy_not_detected(self):
        """Cell with entropy < threshold is not mixed."""
        cells = [
            _make_cell(0.01, 3, 1, phase_entropy=0.1, success_fraction=1.0),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_mixed_cells"] == 0

    def test_custom_threshold(self):
        """Custom entropy threshold works."""
        cells = [
            _make_cell(0.01, 3, 0, phase_entropy=0.3, boundary_fraction=0.5),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram, entropy_threshold=0.2)
        assert result["boundary_summary"]["num_mixed_cells"] == 1


# ── Critical-cell detection tests ────────────────────────────────────


class TestCriticalCellDetection:
    def test_high_boundary_fraction(self):
        """High boundary_fraction → critical."""
        cells = [
            _make_cell(0.01, 3, 0, boundary_fraction=0.5),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_critical_cells"] == 1

    def test_high_metastability(self):
        """High metastability_score → critical."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0,
                       mean_metastability_score=0.8),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_critical_cells"] == 1

    def test_near_zero_boundary_eps(self):
        """Near-zero mean_boundary_eps → critical."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0,
                       mean_boundary_eps=0.005),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_critical_cells"] == 1

    def test_not_critical(self):
        """Low values everywhere → not critical."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0,
                       mean_metastability_score=0.1,
                       mean_boundary_eps=0.5),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_critical_cells"] == 0

    def test_none_metastability_not_critical(self):
        """None metastability → not critical from that criterion."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0,
                       mean_metastability_score=None,
                       mean_boundary_eps=None),
        ]
        diagram = _make_diagram([0.01], [3], cells)
        result = analyze_phase_boundaries(diagram)
        assert result["boundary_summary"]["num_critical_cells"] == 0


# ── Determinism and JSON tests ───────────────────────────────────────


class TestDeterminismAndJSON:
    def test_determinism(self):
        """Repeated analysis produces identical output."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0),
            _make_cell(0.01, 5, -1, failure_fraction=1.0, phase_entropy=0.6),
            _make_cell(0.02, 3, 0, boundary_fraction=0.5, phase_entropy=0.8),
            _make_cell(0.02, 5, -1, failure_fraction=1.0),
        ]
        diagram = _make_diagram([0.01, 0.02], [3, 5], cells)
        r1 = analyze_phase_boundaries(diagram)
        r2 = analyze_phase_boundaries(diagram)
        assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)

    def test_json_roundtrip(self):
        """Output survives JSON serialization."""
        cells = [
            _make_cell(0.01, 3, 1, success_fraction=1.0),
            _make_cell(0.02, 3, -1, failure_fraction=1.0),
        ]
        diagram = _make_diagram([0.01, 0.02], [3], cells)
        result = analyze_phase_boundaries(diagram)
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized == result


# ── Validation tests ─────────────────────────────────────────────────


class TestValidation:
    def test_missing_grid_axes(self):
        with pytest.raises(ValueError, match="grid_axes"):
            analyze_phase_boundaries({"cells": []})

    def test_missing_cells(self):
        with pytest.raises(ValueError, match="cells"):
            analyze_phase_boundaries({
                "grid_axes": {
                    "x_name": "p", "x_values": [0.01],
                    "y_name": "d", "y_values": [3],
                },
            })
