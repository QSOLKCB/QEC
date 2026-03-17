from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.operator_statistics import update_operator_success
from src.qec.analysis.spectral_motif_extraction import extract_spectral_motifs
from src.qec.discovery.adaptive_operator_weights import (
    compute_operator_weights,
    deterministic_weighted_choice,
)
import src.qec.discovery.discovery_engine as discovery_engine
from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.discovery.motif_library import SpectralMotifLibrary


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


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


def test_operator_statistics_update_after_evaluation(monkeypatch) -> None:
    state = {"evaluation_calls": 0, "last_eval_calls": -1, "updates": []}

    def fake_compute_discovery_objectives(H, seed=0):
        state["evaluation_calls"] += 1
        score = float(np.float64(100.0 - np.sum(np.asarray(H, dtype=np.float64), dtype=np.float64)))
        return {
            "composite_score": score,
            "instability_score": float(np.float64(score)),
        }

    def fake_mutate_tanner_graph(H, operator, generation, seed, target_edges=None, target_spectrum=None):
        H2 = np.asarray(H, dtype=np.float64).copy()
        H2[0, 0] = 1.0 - H2[0, 0]
        return H2, operator

    def fake_ace_gate_mutation(*args, **kwargs):
        return {"accept": True}

    def fake_repair_tanner_graph(H, target_variable_degree=None, target_check_degree=None):
        return np.asarray(H, dtype=np.float64), {"is_valid": True}

    real_update = discovery_engine.update_operator_success

    def checked_update(operator_stats, operator_name, improvement):
        state["last_eval_calls"] = state["evaluation_calls"]
        state["updates"].append(float(np.float64(improvement)))
        return real_update(operator_stats, operator_name, improvement)

    monkeypatch.setattr(discovery_engine, "compute_discovery_objectives", fake_compute_discovery_objectives)
    monkeypatch.setattr(discovery_engine, "mutate_tanner_graph", fake_mutate_tanner_graph)
    monkeypatch.setattr(discovery_engine, "ace_gate_mutation", fake_ace_gate_mutation)
    monkeypatch.setattr(discovery_engine, "repair_tanner_graph", fake_repair_tanner_graph)
    monkeypatch.setattr(discovery_engine, "update_operator_success", checked_update)

    result_a = run_structure_discovery(_default_spec(), num_generations=1, population_size=4, base_seed=7)
    result_b = run_structure_discovery(_default_spec(), num_generations=1, population_size=4, base_seed=7)

    assert state["updates"]
    assert state["last_eval_calls"] > 0

    assert result_a["operator_stats"] == result_b["operator_stats"]
    assert result_a["operator_weights"] == result_b["operator_weights"]

    for stats in result_a["operator_stats"].values():
        trials = float(np.float64(stats["trials"]))
        successes = float(np.float64(stats["successes"]))
        success_rate = float(np.float64(stats["success_rate"]))
        assert trials >= successes
        expected_rate = float(np.float64(successes / max(trials, 1.0)))
        assert np.isclose(success_rate, expected_rate)
        assert success_rate >= 0.0


def test_operator_success_requires_improvement() -> None:
    from src.qec.analysis.operator_statistics import update_operator_success

    stats = {}
    try:
        update_operator_success(stats, "edge_swap", None)
    except ValueError as exc:
        assert "requires computed improvement" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing improvement")
    spec = _default_spec()
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
