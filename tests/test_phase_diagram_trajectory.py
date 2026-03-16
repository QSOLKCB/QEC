"""Determinism tests for 3D phase-diagram trajectory overlay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import numpy as np

from src.qec.analysis.spectral_phase_diagram_3d import generate_phase_surface_3d
from src.qec.discovery.discovery_engine import run_structure_discovery


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_trajectory_overlay_deterministic_png(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    _ = matplotlib

    pts = np.asarray(
        [
            [1.00, -0.30, 0.20],
            [1.05, -0.25, 0.22],
            [1.10, -0.20, 0.24],
            [1.15, -0.18, 0.26],
        ],
        dtype=np.float64,
    )
    out1 = tmp_path / "phase1.png"
    out2 = tmp_path / "phase2.png"

    r1 = generate_phase_surface_3d(pts, output_path=str(out1), trajectory_points=[pts[0], pts[1], pts[2], pts[3]])
    r2 = generate_phase_surface_3d(pts, output_path=str(out2), trajectory_points=[pts[0], pts[1], pts[2], pts[3]])

    assert r1["rendered"] is True
    assert r2["rendered"] is True
    assert r1["trajectory_length"] == 4
    assert r2["trajectory_length"] == 4
    assert _sha256(out1) == _sha256(out2)


def test_discovery_engine_phase_trajectory_opt_in() -> None:
    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    result = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=11,
        enable_phase_trajectory=True,
    )
    assert "phase_diagram_trajectory_length" in result
    assert int(result["phase_diagram_trajectory_length"]) >= 1

    result_default = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=11,
    )
    assert "phase_diagram_trajectory_length" not in result_default
