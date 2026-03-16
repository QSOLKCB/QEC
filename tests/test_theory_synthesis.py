from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.theory_synthesis import (
    build_phase_dataset,
    fit_spectral_models,
    generate_spectral_conjectures,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _phase_profiles() -> list[dict[str, float]]:
    return [
        {
            "phase_id": 2,
            "spectral_radius": 1.10,
            "bethe_min_eigenvalue": -0.15,
            "bp_stability_score": 0.52,
            "trapping_density": 0.31,
            "estimated_threshold": 0.42,
        },
        {
            "phase_id": 0,
            "spectral_radius": 0.95,
            "bethe_hessian_min_eigenvalue": -0.05,
            "bp_stability_score": 0.71,
            "trapping_density": 0.22,
            "estimated_threshold": 0.57,
        },
        {
            "phase_id": 1,
            "spectral_radius": 1.02,
            "bethe_hessian_min_eigenvalue": -0.09,
            "bp_stability_score": 0.64,
            "trapping_density": 0.26,
            "estimated_threshold": 0.50,
        },
    ]


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_build_phase_dataset_deterministic_order_and_dtype() -> None:
    dataset = build_phase_dataset(_phase_profiles())
    assert dataset["feature_names"] == [
        "spectral_radius",
        "bethe_hessian_min_eigenvalue",
        "bp_stability_score",
        "trapping_density",
    ]
    assert dataset["X"].dtype == np.float64
    assert dataset["y"].dtype == np.float64
    # Must be sorted by phase_id: 0, 1, 2
    assert np.allclose(dataset["X"][0], np.array([0.95, -0.05, 0.71, 0.22], dtype=np.float64))
    assert np.allclose(dataset["X"][1], np.array([1.02, -0.09, 0.64, 0.26], dtype=np.float64))
    assert np.allclose(dataset["X"][2], np.array([1.10, -0.15, 0.52, 0.31], dtype=np.float64))


def test_model_fitting_and_conjectures_are_deterministic() -> None:
    dataset = build_phase_dataset(_phase_profiles())

    models_a = fit_spectral_models(dataset["X"], dataset["y"])
    models_b = fit_spectral_models(dataset["X"], dataset["y"])
    assert len(models_a) == 5
    for rec in models_a:
        assert isinstance(rec["coefficients"], np.ndarray)
        assert rec["coefficients"].dtype == np.float64

    assert json.dumps(
        [{**m, "coefficients": m["coefficients"].tolist()} for m in models_a],
        sort_keys=True,
    ) == json.dumps(
        [{**m, "coefficients": m["coefficients"].tolist()} for m in models_b],
        sort_keys=True,
    )

    conjectures_a = generate_spectral_conjectures(models_a, dataset["feature_names"])
    conjectures_b = generate_spectral_conjectures(models_b, dataset["feature_names"])
    assert conjectures_a == conjectures_b
    scores = [float(c["fit_quality"]) for c in conjectures_a]
    assert scores == sorted(scores, reverse=True)


def test_engine_theory_synthesis_is_deterministic_opt_in() -> None:
    run_kwargs = dict(
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_basin_hopping=True,
        basin_detection_interval=1,
        enable_phase_novelty_discovery=True,
        phase_novelty_interval=1,
        phase_novelty_threshold=-1.0,
        enable_phase_characterization=True,
        phase_characterization_interval=1,
        enable_theory_synthesis=True,
        theory_synthesis_interval=1,
    )
    r1 = run_structure_discovery(_spec(), **run_kwargs)
    r2 = run_structure_discovery(_spec(), **run_kwargs)

    assert "spectral_theory_models" in r1
    assert "spectral_conjectures" in r1
    assert "num_generated_conjectures" in r1
    assert "best_conjecture_score" in r1
    assert r1["num_generated_conjectures"] == len(r1["spectral_conjectures"])

    s1 = r1["generation_summaries"][-1]
    assert "num_generated_conjectures" in s1
    assert "best_conjecture_score" in s1

    stable_1 = json.dumps(r1["spectral_conjectures"], sort_keys=True)
    stable_2 = json.dumps(r2["spectral_conjectures"], sort_keys=True)
    assert stable_1 == stable_2
