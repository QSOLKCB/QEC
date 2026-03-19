from __future__ import annotations

import builtins
import numpy as np

from qec.analysis.nb_instability_gradient import NBInstabilityGradientAnalyzer
from qec.experiments.constants import ABLATION_METRICS, MUTATION_STRATEGIES
from qec.experiments.mutation_trajectory_plot import (
    plot_trajectory_matplotlib,
    render_trajectory_ascii,
)
from qec.experiments.spectral_basin_visualizer import SpectralBasinVisualizer


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def test_constants_stable() -> None:
    assert MUTATION_STRATEGIES == [
        "baseline", "random_swap", "nb_swap", "nb_ipr_swap", "nb_gradient",
    ]
    assert ABLATION_METRICS == ["fer", "spectral_radius", "ipr", "runtime"]


def test_spectral_basin_trajectory_deterministic() -> None:
    H = _matrix()
    vis = SpectralBasinVisualizer()

    t1 = vis.trace_mutation_trajectory(H, iterations=4)
    t2 = vis.trace_mutation_trajectory(H, iterations=4)

    assert t1 == t2
    assert len(t1) == 5
    assert set(t1[0]) == {"iteration", "spectral_radius", "ipr", "nb_energy"}
    for point in t1:
        assert round(point["spectral_radius"], 12) == point["spectral_radius"]
        assert round(point["ipr"], 12) == point["ipr"]
        assert round(point["nb_energy"], 12) == point["nb_energy"]


def test_trajectory_ascii_render() -> None:
    trajectory = [
        {"iteration": 0, "spectral_radius": 3.12, "ipr": 0.042, "nb_energy": 1.0},
        {"iteration": 1, "spectral_radius": 3.05, "ipr": 0.05, "nb_energy": 0.9},
    ]
    text = render_trajectory_ascii(trajectory)
    assert "Spectral Basin Trajectory" in text
    assert "iter   radius   IPR" in text


def test_trajectory_ascii_render_empty() -> None:
    text = render_trajectory_ascii([])
    assert "Spectral Basin Trajectory" in text
    assert "(no data)" in text


def test_trajectory_plot_fallback_or_figure() -> None:
    trajectory = [{"iteration": 0, "spectral_radius": 1.0, "ipr": 0.1, "nb_energy": 1.0}]
    result = plot_trajectory_matplotlib(trajectory)
    assert isinstance(result, str) or hasattr(result, "savefig")


def test_trajectory_plot_importerror_fallback(monkeypatch) -> None:
    orig_import = builtins.__import__

    def _import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "matplotlib.pyplot":
            raise ImportError("forced for test")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    trajectory = [{"iteration": 0, "spectral_radius": 1.0, "ipr": 0.1, "nb_energy": 1.0}]
    result = plot_trajectory_matplotlib(trajectory)
    assert isinstance(result, str)
    assert "Spectral Basin Trajectory" in result


def test_trajectory_plot_empty_returns_figure_with_no_data() -> None:
    fig = plot_trajectory_matplotlib([])
    if isinstance(fig, str):
        assert "(no data)" in fig
        return

    assert hasattr(fig, "axes")
    assert fig.axes[0].get_title() == "Spectral Basin Trajectory"
    assert any(text.get_text() == "(no data)" for text in fig.axes[0].texts)


def test_nb_flow_mismatch_guard() -> None:
    analyzer = NBInstabilityGradientAnalyzer()

    def _bad_flow(_H: np.ndarray) -> dict:
        return {
            "directed_edges": [(0, 1)],
            "directed_edge_flow": np.zeros(0, dtype=np.float64),
        }

    analyzer._flow_analyzer.compute_flow = _bad_flow  # type: ignore[assignment]

    H = _matrix()
    try:
        analyzer.compute_gradient(H)
        assert False, "Expected ValueError for flow mismatch"
    except ValueError as exc:
        assert "NB flow vector mismatch" in str(exc)
