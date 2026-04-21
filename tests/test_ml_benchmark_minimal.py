from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from qec.analysis import ml_benchmark_minimal as bench


def _stub_kernel(*, nodes, edges):
    return {"proposals": [{"target_nodes": [nodes[0]]}], "nodes": nodes, "edges": edges}


def _stub_terminate_false(*, kernel_result):
    return {"decision": {"terminate_early": False}}


def _stub_terminate_true(*, kernel_result):
    return {"decision": {"terminate_early": True}}


def _stub_kernel_malformed(*, nodes, edges):
    return {"proposals": []}


def _stub_terminate_malformed(*, kernel_result):
    return {"decision": {"terminate_early": "yes"}}


def test_deterministic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    scenarios = [
        {
            "id": "b",
            "nodes": ["n1", "n2"],
            "edges": [("n1", "n2")],
            "expected_top_node": "n1",
            "expected_terminate": False,
        },
        {
            "id": "a",
            "nodes": ["x", "y", "z"],
            "edges": [("x", "y")],
            "expected_top_node": "x",
            "expected_terminate": False,
        },
    ]

    first = bench.run_minimal_ml_benchmark(scenarios)
    second = bench.run_minimal_ml_benchmark(scenarios)
    assert first == second


def test_correct_match(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    result = bench.run_minimal_ml_benchmark(
        [
            {
                "id": "s1",
                "nodes": ["n0"],
                "edges": [],
                "expected_top_node": "n0",
                "expected_terminate": False,
            }
        ]
    )
    assert result["mean_top_match"] == 1.0
    assert result["mean_termination_correct"] == 1.0


def test_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_true
    )
    result = bench.run_minimal_ml_benchmark(
        [
            {
                "id": "s1",
                "nodes": ["n0"],
                "edges": [],
                "expected_top_node": "nX",
                "expected_terminate": False,
            }
        ]
    )
    assert result["mean_top_match"] == 0.0
    assert result["mean_termination_correct"] == 0.0


def test_early_termination_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_true
    )
    result = bench.run_minimal_ml_benchmark(
        [
            {
                "id": "s1",
                "nodes": ["n0", "n1"],
                "edges": [("n0", "n1")],
                "expected_top_node": "n0",
                "expected_terminate": True,
            }
        ]
    )
    assert result["mean_termination_correct"] == 1.0


def test_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    with pytest.raises(ValueError, match="scenarios must be a non-empty list"):
        bench.run_minimal_ml_benchmark([])

    with pytest.raises(ValueError, match="missing required fields"):
        bench.run_minimal_ml_benchmark([{"id": "s1"}])

    with pytest.raises(
        ValueError, match="scenario nodes must be a non-empty list\\[str\\]"
    ):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": [],
                    "edges": [],
                    "expected_top_node": "a",
                    "expected_terminate": False,
                }
            ]
        )

    with pytest.raises(ValueError, match="scenario edges must be list of 2-tuples"):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": ["a"],
                    "edges": [("a", "b"), ("b",)],
                    "expected_top_node": "a",
                    "expected_terminate": False,
                }
            ]
        )

    with pytest.raises(ValueError, match="scenario expected_top_node must be str"):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": ["a"],
                    "edges": [],
                    "expected_top_node": 1,
                    "expected_terminate": False,
                }
            ]
        )

    with pytest.raises(ValueError, match="scenario expected_terminate must be bool"):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": ["a"],
                    "edges": [],
                    "expected_top_node": "a",
                    "expected_terminate": "false",
                }
            ]
        )

    with pytest.raises(ValueError, match="duplicate scenario ids"):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "dup",
                    "nodes": ["a"],
                    "edges": [],
                    "expected_top_node": "a",
                    "expected_terminate": False,
                },
                {
                    "id": "dup",
                    "nodes": ["b"],
                    "edges": [],
                    "expected_top_node": "b",
                    "expected_terminate": False,
                },
            ]
        )


def test_malformed_kernel_result_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bench, "deterministic_gnn_decoder_kernel", _stub_kernel_malformed
    )
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )

    with pytest.raises(
        ValueError, match="kernel_result must contain non-empty list at key 'proposals'"
    ):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": ["a"],
                    "edges": [],
                    "expected_top_node": "a",
                    "expected_terminate": False,
                }
            ]
        )


def test_malformed_termination_result_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_malformed
    )

    with pytest.raises(
        ValueError,
        match="termination_result decision\\['terminate_early'\\] must be bool",
    ):
        bench.run_minimal_ml_benchmark(
            [
                {
                    "id": "s1",
                    "nodes": ["a"],
                    "edges": [],
                    "expected_top_node": "a",
                    "expected_terminate": False,
                }
            ]
        )


def test_dataclass_immutability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    full = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a"],
                "edges": [],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )

    with pytest.raises(FrozenInstanceError):
        full.aggregate.mean_top_match = 0.0


def test_canonical_json_and_stable_hash_are_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    scenarios = [
        {
            "id": "s2",
            "nodes": ["x", "y"],
            "edges": [("x", "y")],
            "expected_top_node": "x",
            "expected_terminate": False,
        },
        {
            "id": "s1",
            "nodes": ["a"],
            "edges": [],
            "expected_top_node": "a",
            "expected_terminate": False,
        },
    ]
    full_a = bench.build_ml_benchmark_full_result(scenarios)
    full_b = bench.build_ml_benchmark_full_result(scenarios)

    assert full_a.to_canonical_json() == full_b.to_canonical_json()
    assert full_a.stable_hash() == full_b.stable_hash()
    assert full_a.aggregate.stable_hash() == full_b.aggregate.stable_hash()
    assert full_a.case_results[0].stable_hash() == full_b.case_results[0].stable_hash()


def test_receipt_hash_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    full = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a"],
                "edges": [],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )
    assert full.receipt.receipt_hash == full.receipt.stable_hash()


def test_replay_identity_changes_when_input_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    baseline = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a"],
                "edges": [],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )
    changed = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a", "b"],
                "edges": [("a", "b")],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )
    assert baseline.receipt.replay_identity != changed.receipt.replay_identity


def test_aggregate_hash_changes_when_scenario_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bench, "deterministic_gnn_decoder_kernel", _stub_kernel)
    monkeypatch.setattr(
        bench, "early_termination_via_dark_state_proofs", _stub_terminate_false
    )
    baseline = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a"],
                "edges": [],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )
    changed = bench.build_ml_benchmark_full_result(
        [
            {
                "id": "s1",
                "nodes": ["a", "b"],
                "edges": [("a", "b")],
                "expected_top_node": "a",
                "expected_terminate": False,
            }
        ]
    )
    assert baseline.aggregate.stable_hash() != changed.aggregate.stable_hash()
