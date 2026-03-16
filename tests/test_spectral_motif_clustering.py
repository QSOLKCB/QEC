from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.spectral_motif_clustering import cluster_spectral_motifs
from src.qec.discovery.motif_library import MotifLibrary
from src.qec.discovery.adaptive_operator_weights import (
    compute_adaptive_operator_weights,
    deterministic_weighted_choice,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _motifs() -> list[dict[str, object]]:
    return [
        {"motif_id": 3, "spectrum": [1.0, 1.0, 1.0, 1.0]},
        {"motif_id": 1, "spectrum": [0.0, 0.1, 0.0, 0.1]},
        {"motif_id": 2, "spectrum": [0.1, 0.0, 0.1, 0.0]},
    ]


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_clustering_is_deterministic() -> None:
    c1 = cluster_spectral_motifs(_motifs(), k=None)
    c2 = cluster_spectral_motifs(_motifs(), k=None)
    assert len(c1) == len(c2)
    j1 = json.dumps([{"cluster_id": c["cluster_id"], "motif_ids": c["motif_ids"], "frequency": c["frequency"]} for c in c1], sort_keys=True)
    j2 = json.dumps([{"cluster_id": c["cluster_id"], "motif_ids": c["motif_ids"], "frequency": c["frequency"]} for c in c2], sort_keys=True)
    assert j1 == j2


def test_cluster_centroids_stable_float64() -> None:
    clusters = cluster_spectral_motifs(_motifs(), k=2)
    assert len(clusters) == 2
    for c in clusters:
        centroid = np.asarray(c["centroid"])
        assert centroid.dtype == np.float64


def test_cluster_lookup_deterministic() -> None:
    lib = MotifLibrary()
    lib.set_motifs(_motifs())
    lib.cluster_motifs(k=2)
    s = np.asarray([0.05, 0.05, 0.05, 0.05], dtype=np.float64)
    c1 = lib.get_cluster(s)
    c2 = lib.get_cluster(s)
    assert c1 is not None and c2 is not None
    assert int(c1["cluster_id"]) == int(c2["cluster_id"])


def test_adaptive_weighting_reproducible() -> None:
    base = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
    succ = np.asarray([0.1, 0.2, 0.3], dtype=np.float64)
    w1 = compute_adaptive_operator_weights(base, succ, 0.5, cluster_similarity=0.25, enable_stability_guard=True)
    w2 = compute_adaptive_operator_weights(base, succ, 0.5, cluster_similarity=0.25, enable_stability_guard=True)
    assert np.array_equal(w1, w2)
    c1 = deterministic_weighted_choice(["a", "b", "c"], w1, seed=123)
    c2 = deterministic_weighted_choice(["a", "b", "c"], w2, seed=123)
    assert c1 == c2


def test_stability_guard_prevents_zero_weights() -> None:
    base = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    succ = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    w = compute_adaptive_operator_weights(base, succ, 0.0, cluster_similarity=0.0, enable_stability_guard=True)
    assert np.all(w > 0.0)


def test_engine_reproducibility_with_motif_clustering() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=11,
        enable_motif_clustering=True,
        enable_operator_stability_guard=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=11,
        enable_motif_clustering=True,
        enable_operator_stability_guard=True,
    )
    assert json.dumps(r1["generation_summaries"], sort_keys=True) == json.dumps(r2["generation_summaries"], sort_keys=True)
    assert "motif_cluster_count" in r1
    assert "cluster_frequencies" in r1
    assert "cluster_centroids" in r1
    assert any("selected_operator" in s for s in r1["generation_summaries"])
    assert any("operator_success_rate" in s for s in r1["generation_summaries"])
