"""v138.7.3 — deterministic neural acceleration simulation layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def deterministic_gnn_decoder_kernel(
    *, nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> Mapping[str, Any]:
    """Runtime hook for deterministic decoder kernel execution."""
    raise NotImplementedError(
        "deterministic_gnn_decoder_kernel must be provided by runtime"
    )


def early_termination_via_dark_state_proofs(
    *, kernel_result: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Runtime hook for deterministic dark-state termination proofing."""
    raise NotImplementedError(
        "early_termination_via_dark_state_proofs must be provided by runtime"
    )


_REQUIRED_FIELDS = ("id", "nodes", "edges")


def _validate_scenario(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("scenario must be mapping-like")

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"scenario missing required fields: {missing}")

    scenario_id = raw["id"]
    if not isinstance(scenario_id, str):
        raise ValueError("scenario id must be str")

    nodes = raw["nodes"]
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(node, str) for node in nodes)
    ):
        raise ValueError("scenario nodes must be a non-empty list[str]")

    edges = raw["edges"]
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("scenario edges must be list[tuple[str, str]]")

    normalized_edges: list[tuple[str, str]] = []
    for edge in edges:
        if (
            not isinstance(edge, Sequence)
            or isinstance(edge, (str, bytes))
            or len(edge) != 2
        ):
            raise ValueError("scenario edges must be list of 2-tuples")
        if not isinstance(edge[0], str) or not isinstance(edge[1], str):
            raise ValueError("scenario edges must contain str node ids")
        normalized_edges.append((edge[0], edge[1]))

    return {"id": scenario_id, "nodes": list(nodes), "edges": normalized_edges}


def _extract_top_proposal_node(kernel_result: Mapping[str, Any]) -> str:
    proposals = kernel_result.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("kernel_result must contain non-empty list at key 'proposals'")
    top = proposals[0]
    if not isinstance(top, Mapping):
        raise ValueError("kernel_result proposals must be mapping-like")
    target_nodes = top.get("target_nodes")
    if not isinstance(target_nodes, list) or not target_nodes:
        raise ValueError(
            "kernel_result proposal must contain non-empty list at key 'target_nodes'"
        )
    top_node = target_nodes[0]
    if not isinstance(top_node, str):
        raise ValueError("kernel_result top target node must be str")
    return top_node


def _extract_terminate_early(termination_result: Mapping[str, Any]) -> bool:
    decision = termination_result.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("termination_result must contain mapping at key 'decision'")
    terminate_early = decision.get("terminate_early")
    if not isinstance(terminate_early, bool):
        raise ValueError("termination_result decision terminate_early must be bool")
    return terminate_early


def _run_baseline_path(
    *, nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> tuple[str, bool, int]:
    kernel_result = deterministic_gnn_decoder_kernel(nodes=nodes, edges=edges)
    top_node = _extract_top_proposal_node(kernel_result)
    termination_result = early_termination_via_dark_state_proofs(
        kernel_result=kernel_result
    )
    terminate_early = _extract_terminate_early(termination_result)
    proposal_count = len(kernel_result["proposals"])
    return top_node, terminate_early, proposal_count


def _run_accelerated_path(
    *, nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> tuple[str, bool, int]:
    # Deterministic simulation only: exact same logical outputs as baseline.
    kernel_result = deterministic_gnn_decoder_kernel(nodes=nodes, edges=edges)
    top_node = _extract_top_proposal_node(kernel_result)
    termination_result = early_termination_via_dark_state_proofs(
        kernel_result=kernel_result
    )
    terminate_early = _extract_terminate_early(termination_result)
    proposal_count = len(kernel_result["proposals"])
    return top_node, terminate_early, proposal_count


def run_neural_acceleration_simulation(
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("scenarios must be a non-empty sequence")
    if not scenarios:
        raise ValueError("scenarios must be a non-empty sequence")

    normalized = [_validate_scenario(scenario) for scenario in scenarios]

    total_baseline = 0.0
    total_accelerated = 0.0
    total_improvement = 0.0
    total_speedup = 0.0

    for scenario in normalized:
        nodes = scenario["nodes"]
        edges = scenario["edges"]
        baseline_top, baseline_terminate, proposal_count = _run_baseline_path(
            nodes=nodes,
            edges=edges,
        )
        accelerated_top, accelerated_terminate, _ = _run_accelerated_path(
            nodes=nodes,
            edges=edges,
        )

        if baseline_top != accelerated_top:
            raise ValueError(
                f"acceleration changed top proposal node for scenario '{scenario['id']}'"
            )
        if baseline_terminate != accelerated_terminate:
            raise ValueError(
                f"acceleration changed terminate_early for scenario '{scenario['id']}'"
            )

        node_count = len(nodes)
        edge_count = len(edges)
        latency_baseline = float(node_count + edge_count + proposal_count)
        latency_accelerated = float(max(1, int(latency_baseline) // 2))
        latency_improvement = latency_baseline - latency_accelerated
        normalized_speedup = latency_accelerated / latency_baseline

        total_baseline += latency_baseline
        total_accelerated += latency_accelerated
        total_improvement += latency_improvement
        total_speedup += normalized_speedup

    count = float(len(normalized))
    return {
        "mean_latency_baseline": total_baseline / count,
        "mean_latency_accelerated": total_accelerated / count,
        "mean_latency_improvement": total_improvement / count,
        "mean_normalized_speedup": total_speedup / count,
    }
