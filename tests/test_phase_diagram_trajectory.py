from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qec.analysis.spectral_phase_diagram_3d import generate_phase_surface_3d
from qec.discovery.discovery_engine import run_structure_discovery


def _surface() -> dict[str, np.ndarray]:
    gx = np.asarray([0.8, 1.0], dtype=np.float64)
    gy = np.asarray([-0.2, -0.1], dtype=np.float64)
    gz = np.asarray([[0.3, 0.4], [0.5, 0.6]], dtype=np.float64)
    return {"grid_x": gx, "grid_y": gy, "grid_z": gz}


def _traj() -> list[np.ndarray]:
    return [
        np.asarray([0.8, -0.2, 0.3], dtype=np.float64),
        np.asarray([1.0, -0.1, 0.6], dtype=np.float64),
    ]


def test_deterministic_png_with_trajectory(tmp_path: Path) -> None:
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    r1 = generate_phase_surface_3d(_surface(), _traj(), str(p1))
    r2 = generate_phase_surface_3d(_surface(), _traj(), str(p2))
    assert p1.read_bytes() == p2.read_bytes()
    assert r1["trajectory_length"] == 2
    assert r1["trajectory_points"] == r2["trajectory_points"]


def test_fallback_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    p1 = tmp_path / "f1.png"
    p2 = tmp_path / "f2.png"
    orig = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "matplotlib.pyplot":
            return None
        return orig(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    r1 = generate_phase_surface_3d(_surface(), _traj(), str(p1))
    r2 = generate_phase_surface_3d(_surface(), _traj(), str(p2))
    assert p1.read_bytes() == p2.read_bytes()
    assert r1["trajectory_points"] == r2["trajectory_points"]


def test_engine_phase_trajectory_opt_in() -> None:
    spec = {"num_variables": 6, "num_checks": 3, "variable_degree": 2, "check_degree": 4}
    result = run_structure_discovery(spec, num_generations=2, population_size=4, base_seed=7, enable_phase_trajectory=True)
    assert "phase_diagram_trajectory_length" in result
    assert int(result["phase_diagram_trajectory_length"]) >= 1
