from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np

from qec.analysis.spectral_phase_diagram_3d import generate_phase_surface_3d


def _phase_surface() -> dict[str, np.ndarray]:
    gx = np.asarray([0.8, 1.0, 1.2], dtype=np.float64)
    gy = np.asarray([-0.3, -0.2, -0.1], dtype=np.float64)
    gz = np.asarray(
        [
            [0.2, 0.3, 0.4],
            [0.3, 0.5, 0.7],
            [0.1, 0.4, 0.6],
        ],
        dtype=np.float64,
    )
    return {"grid_x": gx, "grid_y": gy, "grid_z": gz}


def test_phase_diagram_3d_file_reproducible(tmp_path: Path) -> None:
    out1 = tmp_path / "surface_1.png"
    out2 = tmp_path / "surface_2.png"

    r1 = generate_phase_surface_3d(None, _phase_surface(), str(out1))
    r2 = generate_phase_surface_3d(None, _phase_surface(), str(out2))

    assert out1.exists()
    assert out2.exists()
    assert r1["num_targets"] == 0
    assert r2["num_targets"] == 0
    assert out1.read_bytes() == out2.read_bytes()


def test_overlay_target_ordering_is_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "overlay_1.png"
    out2 = tmp_path / "overlay_2.png"

    t1 = [
        np.asarray([1.2, -0.1, 0.5, 0.0], dtype=np.float64),
        np.asarray([0.8, -0.3, 0.6, 0.0], dtype=np.float64),
        np.asarray([1.0, -0.2, 0.7, 0.0], dtype=np.float64),
    ]
    t2 = [t1[2], t1[0], t1[1]]

    r1 = generate_phase_surface_3d(None, _phase_surface(), str(out1), planned_targets=t1)
    r2 = generate_phase_surface_3d(None, _phase_surface(), str(out2), planned_targets=t2)

    assert r1["target_points"] == r2["target_points"]
    assert r1["num_targets"] == 3
    assert r2["num_targets"] == 3
    assert out1.read_bytes() == out2.read_bytes()


def test_phase_diagram_3d_handles_none_targets(tmp_path: Path) -> None:
    out = tmp_path / "surface_none.png"
    result = generate_phase_surface_3d(None, _phase_surface(), str(out), planned_targets=None)
    assert out.exists()
    assert result["surface_path"] == str(out)
    assert result["num_targets"] == 0
    assert result["target_points"] == []


def test_phase_diagram_3d_fallback_without_matplotlib(tmp_path: Path, monkeypatch) -> None:
    out1 = tmp_path / "fallback_1.png"
    out2 = tmp_path / "fallback_2.png"

    original_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name == "matplotlib.pyplot":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)

    r1 = generate_phase_surface_3d(None, _phase_surface(), str(out1), planned_targets=[np.asarray([1.0, -0.2], dtype=np.float64)])
    r2 = generate_phase_surface_3d(None, _phase_surface(), str(out2), planned_targets=[np.asarray([1.0, -0.2], dtype=np.float64)])

    assert out1.exists()
    assert out2.exists()
    assert r1["surface_path"] == str(out1)
    assert r2["surface_path"] == str(out2)
    assert r1["target_points"] == r2["target_points"]
    assert out1.read_bytes() == out2.read_bytes()
