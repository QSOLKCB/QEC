from __future__ import annotations

import inspect

import pytest

from qec.analysis import canonical_hashing
from qec.analysis import deterministic_gnn_decoder_kernel
from qec.analysis.deterministic_gnn_decoder_kernel import (
    DeterministicGNNKernelConfig,
    DeterministicGNNKernelReceipt,
    SyndromeGraphEdge,
    SyndromeGraphNode,
    build_deterministic_gnn_decoder_kernel,
)


def _base_config(**overrides: object) -> DeterministicGNNKernelConfig:
    payload: dict[str, object] = {
        "num_rounds": 5,
        "self_weight": 0.2,
        "neighbor_weight": 0.3,
        "syndrome_weight": 0.4,
        "hardware_weight": 0.05,
        "residual_weight": 0.05,
        "damping_factor": 0.25,
        "score_round_digits": 8,
        "top_k": 3,
        "convergence_epsilon": 1e-9,
        "normalization_policy": "clamp_0_1",
    }
    payload.update(overrides)
    return DeterministicGNNKernelConfig(**payload)


def _base_nodes() -> list[dict[str, object]]:
    return [
        {
            "node_id": "n0",
            "syndrome": 0.9,
            "parity": 0.5,
            "defect": 0.1,
            "hardware_sideband": {"latency": 0.2, "thermal": 0.4},
        },
        {
            "node_id": "n1",
            "syndrome": 0.3,
            "parity": 0.2,
            "defect": 0.2,
            "hardware_sideband": {"latency": 0.1, "thermal": 0.1},
        },
        {
            "node_id": "n2",
            "syndrome": 0.7,
            "parity": 0.3,
            "defect": 0.0,
            "hardware_sideband": {"latency": 0.3, "thermal": 0.2},
        },
    ]


def _base_edges() -> list[dict[str, object]]:
    return [
        {
            "edge_id": "e0",
            "source_node_id": "n0",
            "target_node_id": "n1",
            "coupling_weight": 0.8,
            "edge_sideband": {"link_quality": 0.9},
        },
        {
            "edge_id": "e1",
            "source_node_id": "n1",
            "target_node_id": "n2",
            "coupling_weight": 0.6,
            "edge_sideband": {"link_quality": 0.8},
        },
    ]


def test_determinism_same_input_same_bytes_hash_and_ranking() -> None:
    config = _base_config()
    nodes = _base_nodes()
    edges = _base_edges()

    a = build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes, edges=edges)
    b = build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes, edges=edges)

    assert a.to_canonical_bytes() == b.to_canonical_bytes()
    assert a.stable_hash() == b.stable_hash()
    assert tuple(p.proposal_id for p in a.proposals) == tuple(p.proposal_id for p in b.proposals)


def test_validation_rejects_duplicate_nodes_unknown_nodes_nan_and_noncanonical_order() -> None:
    config = _base_config()

    bad_nodes = _base_nodes()
    bad_nodes[1]["node_id"] = "n0"
    with pytest.raises(ValueError, match="duplicate node ids"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=bad_nodes, edges=_base_edges())

    bad_edges = _base_edges()
    bad_edges[0]["target_node_id"] = "missing"
    with pytest.raises(ValueError, match="unknown node"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=_base_nodes(), edges=bad_edges)

    bad_nan = _base_nodes()
    bad_nan[0]["syndrome"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=bad_nan, edges=_base_edges())

    unsorted_nodes = list(reversed(_base_nodes()))
    with pytest.raises(ValueError, match="canonical"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=unsorted_nodes, edges=_base_edges())


def test_behavior_ordering_tie_breaking_bounds_and_convergence_shape() -> None:
    config = _base_config(top_k=2, num_rounds=3)
    nodes = [
        {"node_id": "n0", "syndrome": 0.5, "parity": 0.5, "defect": 0.0, "hardware_sideband": {"latency": 0.0}},
        {"node_id": "n1", "syndrome": 0.5, "parity": 0.5, "defect": 0.0, "hardware_sideband": {"latency": 0.0}},
    ]
    edges = [
        {
            "edge_id": "e0",
            "source_node_id": "n0",
            "target_node_id": "n1",
            "coupling_weight": 0.0,
            "edge_sideband": {},
        }
    ]

    result = build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes, edges=edges)
    assert [proposal.proposal_id for proposal in result.proposals] == ["proposal::n0", "proposal::n1"]
    assert len(result.message_snapshots) == 3
    assert isinstance(result.converged, bool)
    for proposal in result.proposals:
        assert 0.0 <= proposal.proposal_score <= 1.0
        assert 0.0 <= proposal.confidence <= 1.0


def test_empty_graph_and_malformed_hardware_rejected() -> None:
    config = _base_config()
    with pytest.raises(ValueError, match="at least one node"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=[], edges=[])

    malformed = _base_nodes()
    malformed[0]["hardware_sideband"] = {"latency": "slow"}
    with pytest.raises(ValueError, match="must be numeric"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=malformed, edges=_base_edges())


def test_receipt_integrity_and_meaningful_change_hash_delta() -> None:
    baseline = build_deterministic_gnn_decoder_kernel(config=_base_config(), nodes=_base_nodes(), edges=_base_edges())
    replay = build_deterministic_gnn_decoder_kernel(config=_base_config(), nodes=_base_nodes(), edges=_base_edges())
    changed = build_deterministic_gnn_decoder_kernel(
        config=_base_config(self_weight=0.21),
        nodes=_base_nodes(),
        edges=_base_edges(),
    )

    assert baseline.receipt.stable_hash() == replay.receipt.stable_hash()
    assert baseline.receipt.receipt_hash == baseline.receipt.stable_hash()
    assert baseline.receipt.kernel_result_hash == baseline.result_hash
    assert baseline.stable_hash() != changed.stable_hash()
    assert baseline.receipt.stable_hash() != changed.receipt.stable_hash()
    assert baseline.receipt.top_proposal_hash is not None
    assert baseline.receipt.kernel_result_hash


def test_config_validation_rejects_boolean_numeric_fields() -> None:
    with pytest.raises(ValueError, match="must not be a bool"):
        build_deterministic_gnn_decoder_kernel(config=_base_config(num_rounds=True), nodes=_base_nodes(), edges=_base_edges())
    with pytest.raises(ValueError, match="must not be a bool"):
        build_deterministic_gnn_decoder_kernel(
            config={**_base_config().to_dict(), "damping_factor": True},
            nodes=_base_nodes(),
            edges=_base_edges(),
        )


def test_dataclass_inputs_are_revalidated_for_finiteness_and_sideband_shape() -> None:
    config = _base_config()
    nodes = (
        SyndromeGraphNode(node_id="n0", syndrome=float("nan"), parity=0.5, defect=0.1, hardware_sideband={"latency": 0.2}),
        SyndromeGraphNode(node_id="n1", syndrome=0.3, parity=0.2, defect=0.2, hardware_sideband={"latency": 0.1}),
    )
    edges = (SyndromeGraphEdge(edge_id="e0", source_node_id="n0", target_node_id="n1", coupling_weight=0.8, edge_sideband={}),)
    with pytest.raises(ValueError, match="node.syndrome must be finite"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes, edges=edges)

    nodes_ok = (
        SyndromeGraphNode(node_id="n0", syndrome=0.9, parity=0.5, defect=0.1, hardware_sideband={"latency": "slow"}),
        SyndromeGraphNode(node_id="n1", syndrome=0.3, parity=0.2, defect=0.2, hardware_sideband={"latency": 0.1}),
    )
    with pytest.raises(ValueError, match="must be numeric"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes_ok, edges=edges)

    edges_nan = (SyndromeGraphEdge(edge_id="e0", source_node_id="n0", target_node_id="n1", coupling_weight=float("nan"), edge_sideband={}),)
    nodes_clean = (
        SyndromeGraphNode(node_id="n0", syndrome=0.9, parity=0.5, defect=0.1, hardware_sideband={"latency": 0.2}),
        SyndromeGraphNode(node_id="n1", syndrome=0.3, parity=0.2, defect=0.2, hardware_sideband={"latency": 0.1}),
    )
    with pytest.raises(ValueError, match="edge.coupling_weight must be finite"):
        build_deterministic_gnn_decoder_kernel(config=config, nodes=nodes_clean, edges=edges_nan)


def test_guardrail_kernel_analysis_module_has_no_decoder_core_dependency() -> None:
    source = inspect.getsource(deterministic_gnn_decoder_kernel)
    assert "qec.decoder" not in source


def test_canonical_hashing_helpers_match_shared_canonical_hashing_module() -> None:
    payload = {"z": [1, 2.0], "a": {"k": "v"}}
    assert deterministic_gnn_decoder_kernel._canonical_json(payload) == canonical_hashing.canonical_json(payload)
    assert deterministic_gnn_decoder_kernel._canonical_bytes(payload) == canonical_hashing.canonical_bytes(payload)
    assert deterministic_gnn_decoder_kernel._sha256_hex(payload) == canonical_hashing.sha256_hex(payload)


def test_hash_mismatch_invariants_raise_for_receipt_dataclass() -> None:
    baseline = build_deterministic_gnn_decoder_kernel(config=_base_config(), nodes=_base_nodes(), edges=_base_edges())
    with pytest.raises(ValueError, match="receipt_hash must match stable_hash payload"):
        DeterministicGNNKernelReceipt(**{**baseline.receipt.to_dict(), "receipt_hash": "0" * 64})
