from __future__ import annotations

import pytest

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
