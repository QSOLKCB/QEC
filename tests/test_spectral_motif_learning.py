from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.operator_statistics import update_operator_success
from src.qec.analysis.spectral_motif_extraction import extract_spectral_motifs
from src.qec.discovery.adaptive_operator_weights import (
    compute_operator_weights,
    deterministic_weighted_choice,
)
from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.discovery.motif_library import SpectralMotifLibrary


def _archive_fixture() -> dict:
    e1 = {
        "candidate_id": "a",
        "objectives": {
            "composite_score": 0.1,
            "spectral_radius": 1.2,
            "bethe_margin": 0.3,
            "ipr_localization": 0.2,
            "entropy": 0.5,
        },
    }
    e2 = {
        "candidate_id": "b",
        "objectives": {
            "composite_score": 0.2,
            "spectral_radius": 1.3,
            "bethe_margin": 0.4,
            "ipr_localization": 0.1,
            "entropy": 0.6,
        },
    }
    return {"categories": {"best_composite": [e1, e2]}}


def test_motif_extraction_determinism() -> None:
    archive = _archive_fixture()
    m1 = extract_spectral_motifs(archive)
    m2 = extract_spectral_motifs(archive)
    assert len(m1) > 0
    assert json.dumps([
        {"motif_id": int(m["motif_id"]), "frequency": int(m["frequency"]), "spectral_signature": m["spectral_signature"].tolist()}
        for m in m1
    ], sort_keys=True) == json.dumps([
        {"motif_id": int(m["motif_id"]), "frequency": int(m["frequency"]), "spectral_signature": m["spectral_signature"].tolist()}
        for m in m2
    ], sort_keys=True)


def test_motif_similarity_matching() -> None:
    lib = SpectralMotifLibrary()
    lib.add_motifs([
        {"motif_id": 1, "spectral_signature": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), "frequency": 2},
        {"motif_id": 2, "spectral_signature": np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float64), "frequency": 1},
    ])
    found = lib.nearest_motif(np.array([1.1, 0.0, 0.0, 0.0], dtype=np.float64))
    assert found is not None
    assert int(found["motif_id"]) == 1


def test_operator_weight_updates() -> None:
    stats = {}
    stats = update_operator_success(stats, "edge_swap", 0.2)
    stats = update_operator_success(stats, "edge_swap", -0.1)
    assert float(stats["edge_swap"]["attempts"]) == 2.0
    assert float(stats["edge_swap"]["successes"]) == 1.0
    assert np.isclose(float(stats["edge_swap"]["success_rate"]), 0.5)


def test_mutation_weighting_stability() -> None:
    stats = {
        "edge_swap": {"success_rate": 0.2},
    }
    regional_similarity = {"edge_swap": 1.0}
    w1 = compute_operator_weights(stats, regional_similarity)
    w2 = compute_operator_weights(stats, regional_similarity)
    assert json.dumps(w1, sort_keys=True) == json.dumps(w2, sort_keys=True)
    assert np.isclose(sum(w1.values()), 1.0)
    assert deterministic_weighted_choice(w1) in w1


def test_engine_integration_reproducibility() -> None:
    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_adaptive_mutation=True,
        enable_motif_learning=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_adaptive_mutation=True,
        enable_motif_learning=True,
    )
    assert json.dumps(r1["generation_summaries"], sort_keys=True) == json.dumps(r2["generation_summaries"], sort_keys=True)
    assert r1.get("motif_library_size", 0) == r2.get("motif_library_size", 0)
