from __future__ import annotations

import json

import numpy as np

from qec.analysis.spectral_phase_diagram_surrogate import (
    PHASE_GRID,
    SpectralPhaseDiagramSurrogate,
    spectral_feature_vector,
    surrogate_threshold,
)
from qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search
from qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_surrogate_prediction_is_deterministic() -> None:
    surrogate = SpectralPhaseDiagramSurrogate()
    metrics = {
        "nb_spectral_radius": 0.7,
        "bethe_hessian_min_eigenvalue": -0.2,
        "bp_stability_score": 0.6,
        "ipr_localization": 0.1,
    }
    features = spectral_feature_vector(metrics)

    c1 = surrogate.predict(features)
    c2 = surrogate.predict(features)

    assert c1.dtype == np.float64
    assert np.array_equal(c1, c2)


def test_surrogate_threshold_extraction() -> None:
    curve = np.array([0.9, 0.8, 0.6, 0.49, 0.2, 0.1, 0.05, 0.01], dtype=np.float64)
    value = surrogate_threshold(PHASE_GRID, curve)
    assert value == 0.04


def test_search_pipeline_supports_phase_diagram_surrogate(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    from qec.discovery import threshold_search as mod

    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.045}
        },
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        rank_by_prediction=True,
        enable_phase_diagram_surrogate=True,
    )
    result = run_spectral_threshold_search(H0, config=cfg)

    assert "history" in result
    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    metrics = payload["candidates"][0]
    assert "spectral_phase_curve" in metrics
    assert len(metrics["spectral_phase_curve"]) == len(PHASE_GRID)
    assert "surrogate_predicted_threshold" in metrics
    assert metrics["predicted_threshold"] == metrics["surrogate_predicted_threshold"]


def test_phase_curve_artifact_serialization(tmp_path, monkeypatch) -> None:
    H0 = _small_graph()

    from qec.discovery import threshold_search as mod

    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.045}
        },
    )

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        enable_phase_diagram_surrogate=True,
    )
    run_spectral_threshold_search(H0, config=cfg)

    text = (tmp_path / "candidate_metrics.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    curve = payload["candidates"][0]["spectral_phase_curve"]

    assert isinstance(curve, list)
    assert all(isinstance(x, float) for x in curve)
