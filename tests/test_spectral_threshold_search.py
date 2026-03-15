from __future__ import annotations

import json

import numpy as np

from src.qec.discovery.threshold_search import (
    BPThresholdEstimator,
    PhaseDiagramOrchestrator,
    SpectralSearchConfig,
    run_spectral_threshold_search,
)
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_deterministic_search_history(tmp_path, monkeypatch):
    H0 = _small_graph()

    def _fake_eval(self, H, *, max_phase_diagram_size, seed):
        return {"measured_boundary": {"mean_boundary_spectral_radius": round(float(np.sum(H)) / 1000.0, 12)}}

    monkeypatch.setattr(PhaseDiagramOrchestrator, "evaluate", _fake_eval)

    cfg1 = SpectralSearchConfig(iterations=2, max_phase_diagram_size=1, output_dir=str(tmp_path / "a"))
    cfg2 = SpectralSearchConfig(iterations=2, max_phase_diagram_size=1, output_dir=str(tmp_path / "b"))

    r1 = run_spectral_threshold_search(H0, config=cfg1)
    r2 = run_spectral_threshold_search(H0, config=cfg2)

    assert r1["history"] == r2["history"]


def test_mutation_integration_uses_nbgradient(tmp_path, monkeypatch):
    H0 = _small_graph()
    calls = {"count": 0}

    from src.qec.discovery import threshold_search as mod

    def _spy_mutate(self, H, steps=1):
        calls["count"] += 1
        return np.asarray(H, dtype=np.float64).copy(), [{"operator": "spy", "steps": int(steps)}]

    monkeypatch.setattr(mod.NBGradientMutator, "mutate", _spy_mutate)
    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {"measured_boundary": {"mean_boundary_spectral_radius": 0.5}},
    )

    cfg = SpectralSearchConfig(iterations=1, max_phase_diagram_size=1, output_dir=str(tmp_path))
    run_spectral_threshold_search(H0, config=cfg)

    assert calls["count"] >= 1


def test_threshold_evaluation_uses_bp_threshold():
    est = BPThresholdEstimator()
    phase = {"measured_boundary": {"mean_boundary_spectral_radius": 0.0481234567899}}
    assert est.estimate(phase) == 0.04812345679


def test_artifact_determinism(tmp_path, monkeypatch):
    H0 = _small_graph()

    monkeypatch.setattr(
        PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {"measured_boundary": {"mean_boundary_spectral_radius": 0.045}},
    )

    cfg = SpectralSearchConfig(iterations=2, max_phase_diagram_size=1, output_dir=str(tmp_path))
    run_spectral_threshold_search(H0, config=cfg)

    first = (tmp_path / "search_history.json").read_text(encoding="utf-8")
    run_spectral_threshold_search(H0, config=cfg)
    second = (tmp_path / "search_history.json").read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first) == json.loads(second)
