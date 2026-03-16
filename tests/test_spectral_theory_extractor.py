"""Tests for v53.0.0 spectral theory extractor."""

from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_conjectures import generate_conjectures, rank_conjectures
from src.qec.analysis.spectral_theory_dataset import build_theory_dataset
from src.qec.analysis.spectral_theory_models import fit_theory_models
from src.qec.discovery.discovery_engine import run_structure_discovery


def _archive_fixture() -> dict:
    base_entries = [
        {
            "candidate_id": "c0",
            "objectives": {
                "spectral_radius": 1.0,
                "bethe_min_eigenvalue": -0.5,
                "bp_stability": 0.3,
                "cycle_density": 0.2,
                "motif_frequency": 0.1,
                "mean_degree": 2.0,
                "decoding_threshold": 0.55,
                "composite_score": 0.9,
            },
        },
        {
            "candidate_id": "c1",
            "objectives": {
                "spectral_radius": 1.2,
                "bethe_min_eigenvalue": -0.4,
                "bp_stability": 0.4,
                "cycle_density": 0.22,
                "motif_frequency": 0.12,
                "mean_degree": 2.1,
                "decoding_threshold": 0.59,
                "composite_score": 0.85,
            },
        },
        {
            "candidate_id": "c2",
            "objectives": {
                "spectral_radius": 1.4,
                "bethe_min_eigenvalue": -0.35,
                "bp_stability": 0.45,
                "cycle_density": 0.28,
                "motif_frequency": 0.15,
                "mean_degree": 2.2,
                "decoding_threshold": 0.63,
                "composite_score": 0.8,
            },
        },
    ]
    return {
        "top_k": 5,
        "categories": {
            "best_composite": [base_entries[2], base_entries[1], base_entries[0]],
            "lowest_instability": [base_entries[1], base_entries[0]],
            "lowest_spectral_radius": [base_entries[0]],
        },
    }


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_dataset_construction_determinism() -> None:
    archive = _archive_fixture()
    X1, y1 = build_theory_dataset(archive)
    X2, y2 = build_theory_dataset(archive)
    assert X1.dtype == np.float64
    assert y1.dtype == np.float64
    assert X1.shape == (3, 6)
    assert y1.shape == (3,)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_model_fitting_reproducibility() -> None:
    X, y = build_theory_dataset(_archive_fixture())
    m1 = fit_theory_models(X, y)
    m2 = fit_theory_models(X, y)
    assert len(m1) > 0
    assert [m["model_name"] for m in m1] == [m["model_name"] for m in m2]
    for a, b in zip(m1, m2):
        assert a["fit_parameters"] == b["fit_parameters"]
        assert a["r2"] == b["r2"]
        assert a["mse"] == b["mse"]
        assert a["correlation"] == b["correlation"]
        assert np.array_equal(a["predictions"], b["predictions"])


def test_conjecture_generation_and_ranking_determinism() -> None:
    X, y = build_theory_dataset(_archive_fixture())
    models = fit_theory_models(X, y)
    c1 = generate_conjectures(models)
    c2 = generate_conjectures(models)
    assert c1 == c2

    r1 = rank_conjectures(c1)
    r2 = rank_conjectures(c2)
    assert r1 == r2
    assert all("ranking_score" in row for row in r1)


def test_engine_integration_determinism() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=4,
        population_size=4,
        base_seed=7,
        enable_theory_extraction=True,
        theory_extraction_interval=2,
        max_conjectures=3,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=4,
        population_size=4,
        base_seed=7,
        enable_theory_extraction=True,
        theory_extraction_interval=2,
        max_conjectures=3,
    )

    assert "spectral_conjectures" in r1
    assert "conjecture_rankings" in r1
    assert "theory_dataset_size" in r1
    assert "best_conjecture_equation" in r1
    assert "best_conjecture_score" in r1
    assert "num_conjectures" in r1
    assert json.dumps(r1["spectral_conjectures"], sort_keys=True) == json.dumps(
        r2["spectral_conjectures"], sort_keys=True,
    )
    assert json.dumps(r1["conjecture_rankings"], sort_keys=True) == json.dumps(
        r2["conjecture_rankings"], sort_keys=True,
    )

    summary = r1["generation_summaries"][-1]
    assert "num_conjectures" in summary
    assert "best_conjecture_score" in summary
    assert "theory_extraction_iteration" in summary


def test_equation_string_and_metadata_determinism() -> None:
    X, y = build_theory_dataset(_archive_fixture())
    models = fit_theory_models(X, y)
    conjectures = generate_conjectures(models)
    assert len(conjectures) > 0
    c0 = conjectures[0]
    assert "equation_string" in c0
    assert "threshold ≈" in str(c0["equation_string"])
    assert "model_type" in c0
    assert "features_used" in c0
    assert "dataset_size" in c0
    assert "fit_metrics" in c0


def test_feature_subset_model_reproducibility() -> None:
    X, y = build_theory_dataset(_archive_fixture())
    models = fit_theory_models(X, y)
    model_names = [str(m.get("model_name", "")) for m in models]
    assert any(name.startswith("linear_0") for name in model_names)
    assert any(name.startswith("linear_0_1") for name in model_names)
    assert any(name.startswith("linear_0_1_2") for name in model_names)

    models_2 = fit_theory_models(X, y)
    assert [m["model_name"] for m in models] == [m["model_name"] for m in models_2]


def test_conjecture_ranking_stability_with_ties() -> None:
    conjectures = [
        {"conjecture_id": 2, "fit_score": 0.5, "confidence_score": 0.5},
        {"conjecture_id": 1, "fit_score": 0.5, "confidence_score": 0.5},
    ]
    ranked = rank_conjectures(conjectures)
    assert ranked[0]["conjecture_id"] == 1
    assert ranked[1]["conjecture_id"] == 2
    assert float(np.float64(ranked[0]["ranking_score"])) == float(np.float64(ranked[1]["ranking_score"]))
