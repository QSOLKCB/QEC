from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_phase_predictor import SpectralPhasePredictor, spectral_feature_vector
from src.qec.discovery.threshold_search import SpectralSearchConfig, run_spectral_threshold_search


def test_predictor_is_deterministic() -> None:
    predictor = SpectralPhasePredictor()
    features = np.array([2.0, -0.5, -0.25, 0.3], dtype=np.float64)
    first = predictor.predict(features)
    second = predictor.predict(features)

    assert first == second


def test_spectral_feature_vector_extraction() -> None:
    metrics = {
        "nb_spectral_radius": 2.5,
        "bethe_min_eigenvalue": -0.75,
        "bp_stability_score": -0.3,
        "ipr_localization_score": 0.2,
    }
    features = spectral_feature_vector(metrics)

    np.testing.assert_array_equal(features, np.array([2.5, -0.75, -0.3, 0.2], dtype=np.float64))


def test_phase_predictor_ranking_and_bp_budget(tmp_path, monkeypatch) -> None:
    from src.qec.discovery import threshold_search as mod

    H0 = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

    mutated = [
        np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64),
    ]

    def _mutate(self, H, steps=1):
        idx = _mutate.calls
        _mutate.calls += 1
        if idx < len(mutated):
            return mutated[idx], [{"operator": "spy", "i": idx}]
        return np.asarray(H, dtype=np.float64).copy(), [{"operator": "spy", "i": idx}]

    _mutate.calls = 0

    def _compute_nb(H):
        score = float(np.sum(H))
        return {"spectral_radius": score / 10.0, "eigenvector": np.array([1.0, 0.0], dtype=np.float64)}

    class _Fr:
        def __init__(self, v):
            self.frustration_score = v / 20.0
            self.negative_modes = 1
            self.max_ipr = v / 10.0
            self.transport_imbalance = 0.0
            self.trap_modes = tuple()

    def _compute_frustration(self, H):
        v = float(np.sum(H))
        return _Fr(v)

    eval_calls = {"count": 0}

    def _fake_eval(self, H, *, max_phase_diagram_size, seed):
        eval_calls["count"] += 1
        return {"measured_boundary": {"mean_boundary_spectral_radius": round(float(np.sum(H)) / 100.0, 12)}}

    monkeypatch.setattr(mod.NBGradientMutator, "mutate", _mutate)
    monkeypatch.setattr(mod, "_is_degree_preserving", lambda H_ref, H_new: True)
    monkeypatch.setattr(mod, "compute_nb_spectrum", _compute_nb)
    monkeypatch.setattr(mod.SpectralFrustrationAnalyzer, "compute_frustration", _compute_frustration)
    monkeypatch.setattr(mod.PhaseDiagramOrchestrator, "evaluate", _fake_eval)

    cfg = SpectralSearchConfig(
        iterations=1,
        output_dir=str(tmp_path),
        enable_beam_mutations=True,
        rank_by_prediction=False,
        enable_spectral_phase_predictor=True,
        bp_evaluation_budget=1,
        trap_similarity_reject=2.0,
    )
    run_spectral_threshold_search(H0, config=cfg)

    payload = json.loads((tmp_path / "candidate_metrics.json").read_text(encoding="utf-8"))
    candidates = payload["candidates"]

    scored = [c for c in candidates if not c["rejected"]]
    assert scored
    assert all("spectral_predicted_threshold" in c for c in scored)
    assert eval_calls["count"] == 1
    assert sum(1 for c in scored if c["bp_evaluated"]) == 1

    predicted = [c["spectral_predicted_threshold"] for c in scored]
    top = max(predicted)
    chosen = [c for c in scored if c["bp_evaluated"]][0]
    assert chosen["spectral_predicted_threshold"] == top
