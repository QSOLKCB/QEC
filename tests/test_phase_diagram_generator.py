from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

import numpy as np

from src.qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from src.qec.analysis.spectral_phase_diagram import (
    build_phase_diagram_dataset,
    construct_phase_grid,
    detect_phase_boundaries,
    estimate_stability_surface,
    generate_phase_heatmap,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _archive_fixture() -> dict:
    return {
        "top_k": 3,
        "categories": {
            "best_composite": [
                {
                    "candidate_id": "c2",
                    "objectives": {
                        "spectral_radius": 1.3,
                        "bethe_min_eigenvalue": -0.2,
                        "bp_stability_score": 0.55,
                    },
                },
                {
                    "candidate_id": "c1",
                    "objectives": {
                        "spectral_radius": 1.1,
                        "bethe_min_eigenvalue": -0.4,
                        "bp_stability_score": 0.72,
                    },
                },
            ],
            "most_novel": [
                {
                    "candidate_id": "c0",
                    "objectives": {
                        "spectral_radius": 0.9,
                        "bethe_min_eigenvalue": -0.6,
                        "bp_stability_score": 0.88,
                    },
                },
                {
                    "candidate_id": "c1",
                    "objectives": {
                        "spectral_radius": 1.1,
                        "bethe_min_eigenvalue": -0.4,
                        "bp_stability_score": 0.72,
                    },
                },
            ],
        },
    }


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_dataset_determinism() -> None:
    d1 = build_phase_diagram_dataset(_archive_fixture())
    d2 = build_phase_diagram_dataset(_archive_fixture())
    assert d1["x"].dtype == np.float64
    assert d1["y"].dtype == np.float64
    assert d1["z"].dtype == np.float64
    np.testing.assert_allclose(d1["x"], d2["x"])
    np.testing.assert_allclose(d1["y"], d2["y"])
    np.testing.assert_allclose(d1["z"], d2["z"])


def test_grid_construction_determinism() -> None:
    dataset = build_phase_diagram_dataset(_archive_fixture())
    g1 = construct_phase_grid(dataset, grid_resolution=16)
    g2 = construct_phase_grid(dataset, grid_resolution=16)
    assert g1["grid_x"].dtype == np.float64
    assert g1["grid_y"].dtype == np.float64
    assert g1["grid_z"].dtype == np.float64
    np.testing.assert_allclose(g1["grid_x"], g2["grid_x"])
    np.testing.assert_allclose(g1["grid_y"], g2["grid_y"])
    np.testing.assert_allclose(g1["grid_z"], g2["grid_z"])


def test_stability_surface_reproducibility() -> None:
    dataset = build_phase_diagram_dataset(_archive_fixture())
    grid = construct_phase_grid(dataset, grid_resolution=12)
    s1 = estimate_stability_surface(dataset, grid)
    s2 = estimate_stability_surface(dataset, grid)
    np.testing.assert_allclose(s1["phase_surface"], s2["phase_surface"])
    np.testing.assert_allclose(s1["sample_counts"], s2["sample_counts"])


def test_phase_boundary_integration() -> None:
    memory = SpectralLandscapeMemory(dim=4)
    memory.add([0.9, -0.6, 0.1, 0.2])
    memory.add([1.1, -0.4, 0.2, 0.3])
    memory.add([1.3, -0.2, 0.3, 0.4])
    boundaries = detect_phase_boundaries(memory)
    assert isinstance(boundaries, list)
    assert all(set(b.keys()) == {"x0", "y0", "x1", "y1"} for b in boundaries)


def test_artifact_generation_reproducibility() -> None:
    dataset = build_phase_diagram_dataset(_archive_fixture())
    grid = construct_phase_grid(dataset, grid_resolution=10)
    surface = estimate_stability_surface(dataset, grid)
    boundaries = [{"x0": 0.9, "y0": -0.6, "x1": 1.3, "y1": -0.2}]

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            payload = {
                "grid_x": surface["grid_x"],
                "grid_y": surface["grid_y"],
                "phase_surface": surface["phase_surface"],
                "phase_boundaries": boundaries,
            }
            p1 = generate_phase_heatmap(payload, output_path="phase1.png")
            p2 = generate_phase_heatmap(payload, output_path="phase2.png")
            assert os.path.exists(p1)
            assert os.path.exists(p2)
            with open(p1, "rb") as f:
                b1 = f.read()
            with open(p2, "rb") as f:
                b2 = f.read()
            assert b1 == b2
        finally:
            os.chdir(cwd)


def test_discovery_engine_phase_diagram_opt_in_reproducible() -> None:
    spec = _spec()
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            r1 = run_structure_discovery(
                spec,
                num_generations=1,
                population_size=4,
                base_seed=42,
                enable_landscape_learning=True,
                enable_phase_diagram=True,
                phase_diagram_resolution=12,
            )
            r2 = run_structure_discovery(
                spec,
                num_generations=1,
                population_size=4,
                base_seed=42,
                enable_landscape_learning=True,
                enable_phase_diagram=True,
                phase_diagram_resolution=12,
            )
        finally:
            os.chdir(cwd)

    assert "phase_diagram_surface" in r1
    assert "phase_diagram_grid" in r1
    assert "phase_boundaries" in r1
    assert "phase_heatmap_path" in r1
    assert r1["phase_heatmap_path"] == "spectral_phase_diagram.png"
    assert json.dumps(r1["phase_diagram_grid"], sort_keys=True) == json.dumps(r2["phase_diagram_grid"], sort_keys=True)
    assert json.dumps(r1["phase_diagram_surface"], sort_keys=True) == json.dumps(r2["phase_diagram_surface"], sort_keys=True)
    assert json.dumps(r1["phase_boundaries"], sort_keys=True) == json.dumps(r2["phase_boundaries"], sort_keys=True)


def test_heatmap_fallback_without_matplotlib() -> None:
    dataset = build_phase_diagram_dataset(_archive_fixture())
    grid = construct_phase_grid(dataset, grid_resolution=8)
    surface = estimate_stability_surface(dataset, grid)
    payload = {
        "grid_x": surface["grid_x"],
        "grid_y": surface["grid_y"],
        "phase_surface": surface["phase_surface"],
        "phase_boundaries": [],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "fallback.png")
        with mock.patch("importlib.util.find_spec", return_value=None):
            path = generate_phase_heatmap(payload, output_path=output_path)
        assert path == output_path
        assert os.path.exists(output_path)
        with open(output_path, "rb") as f:
            data = f.read()
        assert data.startswith(b"\x89PNG\r\n\x1a\n")
