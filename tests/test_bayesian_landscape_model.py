from __future__ import annotations

import json

import numpy as np

from qec.analysis.bayesian_landscape_model import BayesianSpectralModel
from qec.analysis.expected_improvement import (
    expected_improvement,
    rank_candidates_bayesian,
)
from qec.analysis.spectral_dataset import build_spectral_dataset
from qec.discovery.discovery_engine import run_structure_discovery


def _mock_archive() -> dict:
    return {
        "top_k": 2,
        "categories": {
            "best_composite": [
                {
                    "candidate_id": "c0",
                    "objectives": {
                        "spectral_radius": 1.2,
                        "bethe_margin": 0.3,
                        "ipr_localization": 0.1,
                        "entropy": 0.9,
                        "composite_score": 0.4,
                        "threshold_estimate": 0.8,
                    },
                },
                {
                    "candidate_id": "c1",
                    "objectives": {
                        "spectral_radius": 1.1,
                        "bethe_margin": 0.2,
                        "ipr_localization": 0.2,
                        "entropy": 0.8,
                        "composite_score": 0.5,
                        "threshold_estimate": 0.7,
                    },
                },
            ],
            "most_novel": [
                {
                    "candidate_id": "c1",
                    "objectives": {
                        "spectral_radius": 1.1,
                        "bethe_margin": 0.2,
                        "ipr_localization": 0.2,
                        "entropy": 0.8,
                        "composite_score": 0.5,
                        "threshold_estimate": 0.7,
                    },
                }
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


def test_dataset_extraction_correctness() -> None:
    X, y = build_spectral_dataset(_mock_archive())
    assert X.dtype == np.float64
    assert y.dtype == np.float64
    assert X.shape == (2, 4)
    assert y.shape == (2,)
    np.testing.assert_allclose(y, np.array([0.4, 0.2], dtype=np.float64))


def test_model_prediction_determinism() -> None:
    X, y = build_spectral_dataset(_mock_archive())
    model = BayesianSpectralModel(length_scale=1.0, noise=1e-6)
    model.fit(X, y)
    p1 = model.predict(X[0])
    p2 = model.predict(X[0])
    assert p1 == p2


def test_expected_improvement_stability() -> None:
    ei1 = expected_improvement(0.75, 0.1, 0.5)
    ei2 = expected_improvement(0.75, 0.1, 0.5)
    assert np.float64(ei1) == np.float64(ei2)
    assert ei1 > 0.0




def test_model_predict_untrained_is_neutral() -> None:
    model = BayesianSpectralModel(length_scale=1.0, noise=1e-6)
    mean, sigma = model.predict(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert mean == 0.0
    assert sigma == 1.0


def test_expected_improvement_zero_sigma() -> None:
    assert expected_improvement(1.0, 0.0, 0.5) == 0.0

def test_candidate_ranking_reproducibility() -> None:
    X, y = build_spectral_dataset(_mock_archive())
    model = BayesianSpectralModel(length_scale=0.75, noise=1e-6)
    model.fit(X, y)
    candidates = [
        {"candidate_id": "a", "objectives": {"spectral_radius": 1.2, "bethe_margin": 0.3, "ipr_localization": 0.1, "entropy": 0.9}},
        {"candidate_id": "b", "objectives": {"spectral_radius": 1.1, "bethe_margin": 0.2, "ipr_localization": 0.2, "entropy": 0.8}},
    ]
    r1 = rank_candidates_bayesian(candidates, model, float(np.max(y)))
    r2 = rank_candidates_bayesian(candidates, model, float(np.max(y)))
    assert [c["candidate_id"] for c in r1] == [c["candidate_id"] for c in r2]
    assert all("expected_improvement" in c for c in r1)


def test_engine_integration_determinism() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=123,
        enable_bayesian_model=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=123,
        enable_bayesian_model=True,
    )
    j1 = json.dumps(r1["generation_summaries"], sort_keys=True)
    j2 = json.dumps(r2["generation_summaries"], sort_keys=True)
    assert j1 == j2
    assert "expected_improvement" in r1["generation_summaries"][0]
    assert "bayesian_prediction_mean" in r1["generation_summaries"][0]
    assert "bayesian_prediction_uncertainty" in r1["generation_summaries"][0]
