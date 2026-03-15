from __future__ import annotations

import json

import numpy as np

from src.qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search
from src.qec.generation.deterministic_construction import construct_deterministic_tanner_graph


def _small_graph() -> np.ndarray:
    spec = {
        "num_variables": 8,
        "num_checks": 4,
        "variable_degree": 2,
        "check_degree": 4,
    }
    return construct_deterministic_tanner_graph(spec)


def test_prediction_ranking_is_deterministic_and_limited(tmp_path, monkeypatch):
    from src.qec.discovery import threshold_search as mod

    H0 = _small_graph()
    eval_calls: list[int] = []

    class _DummyMutator:
        def __init__(self, *, enabled, enable_spectral_beam_search):
            self.enable_spectral_beam_search = bool(enable_spectral_beam_search)

    def _mutate(self, H, steps=1):
        H_new = np.asarray(H, dtype=np.float64).copy()
        if self.enable_spectral_beam_search:
            H_new[[0, 1]] = H_new[[1, 0]]
            label = "beam"
        else:
            H_new[:, [0, 1]] = H_new[:, [1, 0]]
            label = "nb"
        return H_new, [{"operator": label, "steps": int(steps)}]

    def _compute_nb(H):
        signature = float(np.argmax(np.asarray(H, dtype=np.float64).ravel()))
        return {"spectral_radius": np.float64(0.01 + signature / 1000.0)}

    def _predict(*, spectral_radius, bethe_negative_mass, flow_ipr, spectral_entropy_value, trap_similarity):
        class _Pred:
            predicted_threshold = float(spectral_radius)
            score = float(spectral_radius * 10.0)

        return _Pred()

    def _evaluate(self, H, *, max_phase_diagram_size, seed):
        eval_calls.append(int(seed))
        return {"measured_boundary": {"mean_boundary_spectral_radius": 0.05}}

    monkeypatch.setattr(mod, "NBGradientMutator", _DummyMutator)
    monkeypatch.setattr(_DummyMutator, "mutate", _mutate, raising=False)
    monkeypatch.setattr(mod, "compute_nb_spectrum", _compute_nb)
    monkeypatch.setattr(mod, "predict_threshold_quality", _predict)
    monkeypatch.setattr(mod.PhaseDiagramOrchestrator, "evaluate", _evaluate)

    cfg = SpectralSearchConfig(
        iterations=1,
        max_phase_diagram_size=1,
        output_dir=str(tmp_path),
        rank_by_prediction=True,
        max_bp_candidates=1,
    )

    run_spectral_threshold_search(H0, config=cfg)
    first_metrics_text = (tmp_path / "candidate_metrics.json").read_text(encoding="utf-8")

    run_spectral_threshold_search(H0, config=cfg)
    second_metrics_text = (tmp_path / "candidate_metrics.json").read_text(encoding="utf-8")

    payload = json.loads(second_metrics_text)
    non_rejected = [c for c in payload["candidates"] if not c["rejected"]]

    assert len(eval_calls) == 2
    assert first_metrics_text == second_metrics_text
    assert non_rejected[0]["prediction_rank"] == 1
    assert non_rejected[1]["prediction_rank"] == 2


def test_prediction_fields_present_without_ranking(tmp_path, monkeypatch):
    from src.qec.discovery import threshold_search as mod

    H0 = _small_graph()

    class _DummyMutator:
        def __init__(self, *, enabled, enable_spectral_beam_search):
            self.enable_spectral_beam_search = bool(enable_spectral_beam_search)

        def mutate(self, H, steps=1):
            return np.asarray(H, dtype=np.float64).copy(), [{"operator": "noop", "steps": int(steps)}]

    monkeypatch.setattr(mod, "NBGradientMutator", _DummyMutator)

    monkeypatch.setattr(
        mod.PhaseDiagramOrchestrator,
        "evaluate",
        lambda self, H, *, max_phase_diagram_size, seed: {
            "measured_boundary": {"mean_boundary_spectral_radius": 0.04}
        },
    )

    cfg = SpectralSearchConfig(iterations=1, max_phase_diagram_size=1, output_dir=str(tmp_path))
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    metrics = payload["candidates"][0]
    assert "predicted_threshold" in metrics
    assert "prediction_score" in metrics
    assert "prediction_rank" in metrics
    assert metrics["prediction_rank"] is None
