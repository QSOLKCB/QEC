from __future__ import annotations

import numpy as np

from qec.analysis.spectral_conjectures import generate_conjectures, rank_conjectures
from qec.analysis.spectral_theory_dataset import build_theory_dataset
from qec.analysis.spectral_theory_models import fit_theory_models
from qec.discovery.discovery_engine import run_structure_discovery


def _archive() -> dict:
    return {
        "categories": {
            "lowest_cycle_density": [
                {"candidate_id": "c2", "objectives": {"spectral_radius": 1.2, "bethe_min_eigenvalue": -0.2, "bp_stability": 0.3, "cycle_density": 0.1, "motif_frequency": 0.2, "mean_degree": 3.0, "threshold": 0.08}},
                {"candidate_id": "c1", "objectives": {"spectral_radius": 1.0, "bethe_min_eigenvalue": -0.3, "bp_stability": 0.4, "cycle_density": 0.2, "motif_frequency": 0.3, "mean_degree": 2.0, "decoding_threshold": 0.09}},
            ],
            "most_novel": [
                {"candidate_id": "c3", "objectives": {"spectral_radius": 1.1, "bethe_min_eigenvalue": -0.1, "bp_stability": 0.5, "cycle_density": 0.15, "motif_frequency": 0.25, "mean_degree": 2.5, "threshold_estimate": 0.07}},
            ],
        }
    }


def test_dataset_is_deterministic_and_ordered() -> None:
    d1 = build_theory_dataset(_archive())
    d2 = build_theory_dataset(_archive())
    assert d1["feature_names"] == [
        "spectral_radius", "bethe_min_eigenvalue", "bp_stability", "cycle_density", "motif_frequency", "mean_degree",
    ]
    assert [r["candidate_id"] for r in d1["rows"]] == ["c1", "c2", "c3"]
    assert np.array_equal(d1["X"], d2["X"])
    assert np.array_equal(d1["y"], d2["y"])


def test_models_conjectures_and_ranking_are_deterministic() -> None:
    dataset = build_theory_dataset(_archive())
    m1 = fit_theory_models(dataset)
    m2 = fit_theory_models(dataset)
    assert len(m1) > 0
    assert m1 == m2

    c1 = generate_conjectures(m1)
    c2 = generate_conjectures(m2)
    assert c1 == c2

    # tie stability via lexsort on id
    tied = [
        {"conjecture_id": "conj_0002", "ranking_score": 1.0},
        {"conjecture_id": "conj_0001", "ranking_score": 1.0},
    ]
    ranked = rank_conjectures(tied)
    assert [x["conjecture_id"] for x in ranked] == ["conj_0001", "conj_0002"]


def test_engine_theory_extraction_opt_in() -> None:
    spec = {"num_variables": 6, "num_checks": 3, "variable_degree": 2, "check_degree": 4}
    result = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_theory_extraction=True,
        theory_extraction_interval=1,
        max_conjectures=5,
    )
    assert "spectral_conjectures" in result
    assert "theory_dataset_size" in result
    assert result["num_conjectures"] <= 5
