"""Regression tests for deterministic operator-learning update correctness."""

from __future__ import annotations

import numpy as np

from src.qec.discovery import discovery_engine
from src.qec.discovery.discovery_engine import run_structure_discovery


def _default_spec() -> dict[str, int]:
    return {
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
        attempts = float(np.float64(stats["attempts"]))
        successes = float(np.float64(stats["successes"]))
        total_improvement = float(np.float64(stats["total_improvement"]))
        mean_improvement = float(np.float64(stats["mean_improvement"]))
        assert attempts >= successes
        assert np.isclose(mean_improvement, total_improvement / attempts)
        assert mean_improvement >= 0.0


def test_operator_success_requires_improvement() -> None:
    from src.qec.analysis.operator_statistics import update_operator_success

    stats = {}
    try:
        update_operator_success(stats, "edge_swap", None)
    except ValueError as exc:
        assert "requires computed improvement" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing improvement")
