from __future__ import annotations

import pytest

from qec.analysis import neural_acceleration_sim as sim


def _stub_kernel(*, nodes, edges):
    return {"proposals": [{"target_nodes": [nodes[0]]}], "nodes": nodes, "edges": edges}


def _stub_terminate_false(*, kernel_result):
    del kernel_result
    return {"decision": {"terminate_early": False}}


def test_deterministic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    scenarios = [
        {"id": "s1", "nodes": ["a", "b"], "edges": [("a", "b")]},
        {"id": "s2", "nodes": ["x", "y", "z"], "edges": [("x", "y")]},
    ]

    first = sim.run_neural_acceleration_simulation(scenarios)
    second = sim.run_neural_acceleration_simulation(scenarios)

    assert first == second


def test_baseline_equals_accelerated_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    result = sim.run_neural_acceleration_simulation(
        [{"id": "s1", "nodes": ["n0", "n1"], "edges": [("n0", "n1")]}]
    )

    assert result["mean_latency_baseline"] == 4.0
    assert result["mean_latency_accelerated"] == 2.0


def test_latency_accelerated_less_than_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    result = sim.run_neural_acceleration_simulation(
        [{"id": "s1", "nodes": ["n0", "n1"], "edges": [("n0", "n1")]}]
    )

    assert result["mean_latency_accelerated"] < result["mean_latency_baseline"]
    assert result["mean_latency_improvement"] == 2.0


def test_speedup_is_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    result = sim.run_neural_acceleration_simulation(
        [
            {"id": "s1", "nodes": ["a", "b"], "edges": [("a", "b")]},
            {"id": "s2", "nodes": ["x", "y", "z"], "edges": [("x", "y")]},
        ]
    )

    expected = ((2.0 / 4.0) + (2.0 / 5.0)) / 2.0
    assert result["mean_normalized_speedup"] == expected


def test_mismatch_triggers_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    def _bad_accelerated_path(*, nodes, edges):
        del edges
        return "DIFF", False, len(nodes)

    monkeypatch.setattr(sim, "_run_accelerated_path", _bad_accelerated_path)

    with pytest.raises(
        ValueError, match="acceleration changed top proposal node for scenario 's1'"
    ):
        sim.run_neural_acceleration_simulation(
            [{"id": "s1", "nodes": ["n0", "n1"], "edges": [("n0", "n1")]}]
        )


def test_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sim, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        sim, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    with pytest.raises(ValueError, match="scenarios must be a non-empty sequence"):
        sim.run_neural_acceleration_simulation([])

    with pytest.raises(ValueError, match="scenario missing required fields"):
        sim.run_neural_acceleration_simulation([{"id": "s1"}])
