"""Tests for v86.2.0 — spectral trajectory visualization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from qec.visualization.trajectory_plot import (
    TRAJECTORY_COLORS,
    plot_spectral_trajectory,
)


# ── fixtures ─────────────────────────────────────────────────────────


def _make_traj(
    trajectory_type: str = "convergent",
    n_steps: int = 5,
) -> dict:
    drift = [0.8, 0.5, 0.3, 0.1][:n_steps - 1]
    return {
        "n_steps": n_steps,
        "drift": drift,
        "lambda_max": [1.0 + 0.1 * i for i in range(n_steps)],
        "rank_evolution": [4] * n_steps,
        "degeneracy_evolution": [1] * n_steps,
        "temporal_transitions": [{"time_index": 1, "drift": 0.5}],
        "trajectory_type": trajectory_type,
    }


# ── determinism ──────────────────────────────────────────────────────


def test_deterministic_plot():
    """Same input produces identical PNG bytes."""
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        p1 = Path(tmp) / "a.png"
        p2 = Path(tmp) / "b.png"
        plot_spectral_trajectory(traj, output_path=p1)
        plot_spectral_trajectory(traj, output_path=p2)
        assert p1.read_bytes() == p2.read_bytes()


# ── n_steps ──────────────────────────────────────────────────────────


def test_n_steps_in_result():
    traj = _make_traj(n_steps=7)
    result = plot_spectral_trajectory(traj)
    assert result["n_steps"] == 7


# ── file creation ────────────────────────────────────────────────────


def test_png_created():
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "test.png"
        result = plot_spectral_trajectory(traj, output_path=out)
        assert out.exists()
        assert result["output_path"] == str(out)


def test_no_output_path_returns_none():
    traj = _make_traj()
    result = plot_spectral_trajectory(traj)
    assert result["output_path"] is None


# ── transitions rendered ─────────────────────────────────────────────


def test_transitions_rendered():
    """Plot with transitions does not error."""
    traj = _make_traj()
    traj["temporal_transitions"] = [
        {"time_index": 0, "drift": 0.8},
        {"time_index": 2, "drift": 0.3},
    ]
    result = plot_spectral_trajectory(traj)
    assert result["n_steps"] == 5


# ── empty input ──────────────────────────────────────────────────────


def test_empty_trajectory():
    traj = {
        "n_steps": 0,
        "drift": [],
        "lambda_max": [],
        "rank_evolution": [],
        "degeneracy_evolution": [],
        "temporal_transitions": [],
        "trajectory_type": "undetermined",
    }
    result = plot_spectral_trajectory(traj)
    assert result["n_steps"] == 0


# ── mode handling ────────────────────────────────────────────────────


def test_paper_mode():
    traj = _make_traj()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "paper.png"
        result = plot_spectral_trajectory(traj, output_path=out, mode="paper")
        assert out.exists()
        assert result["output_path"] == str(out)


def test_invalid_mode_raises():
    traj = _make_traj()
    with pytest.raises(ValueError, match="mode must be"):
        plot_spectral_trajectory(traj, mode="invalid")


# ── all trajectory types ─────────────────────────────────────────────


@pytest.mark.parametrize("ttype", list(TRAJECTORY_COLORS.keys()))
def test_trajectory_types(ttype):
    traj = _make_traj(trajectory_type=ttype)
    result = plot_spectral_trajectory(traj)
    assert result["n_steps"] == 5
