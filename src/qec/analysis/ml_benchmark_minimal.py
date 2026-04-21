"""v138.7.2 bootstrap: deterministic minimal ML benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any





def deterministic_gnn_decoder_kernel(*, nodes, edges):
    raise NotImplementedError("deterministic_gnn_decoder_kernel must be provided by runtime")


def early_termination_via_dark_state_proofs(*, kernel_result):
    raise NotImplementedError("early_termination_via_dark_state_proofs must be provided by runtime")
REQUIRED_FIELDS = (
    "id",
    "nodes",
    "edges",
    "expected_top_node",
    "expected_terminate",
)


def _validate_scenario(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("scenario must be mapping-like")

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ValueError(f"scenario missing required fields: {missing}")

    scenario_id = raw["id"]
    if not isinstance(scenario_id, str):
        raise ValueError("scenario id must be str")

    nodes = raw["nodes"]
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) for node in nodes):
        raise ValueError("scenario nodes must be a non-empty list[str]")

    edges = raw["edges"]
    if not isinstance(edges, list):
        raise ValueError("scenario edges must be list[tuple[str, str]]")
    for edge in edges:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError("scenario edges must be list of 2-tuples")
        if not isinstance(edge[0], str) or not isinstance(edge[1], str):
            raise ValueError("scenario edges must contain str node ids")

    expected_top_node = raw["expected_top_node"]
    if not isinstance(expected_top_node, str):
        raise ValueError("scenario expected_top_node must be str")

    expected_terminate = raw["expected_terminate"]
    if not isinstance(expected_terminate, bool):
        raise ValueError("scenario expected_terminate must be bool")

    return {
        "id": scenario_id,
        "nodes": list(nodes),
        "edges": list(edges),
        "expected_top_node": expected_top_node,
        "expected_terminate": expected_terminate,
    }


def run_minimal_ml_benchmark(scenarios: list[dict]) -> dict:
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")

    normalized = [_validate_scenario(s) for s in scenarios]
    scenario_ids = [s["id"] for s in normalized]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("duplicate scenario ids")

    ordered = sorted(normalized, key=lambda item: item["id"])

    top_match_sum = 0.0
    termination_correct_sum = 0.0
    latency_sum = 0.0

    for scenario in ordered:
        kernel_result = deterministic_gnn_decoder_kernel(nodes=scenario["nodes"], edges=scenario["edges"])
        predicted_top = kernel_result["proposals"][0]["target_nodes"][0]

        termination_result = early_termination_via_dark_state_proofs(kernel_result=kernel_result)
        predicted_terminate = termination_result["decision"]["terminate_early"]

        top_match_sum += 1.0 if predicted_top == scenario["expected_top_node"] else 0.0
        termination_correct_sum += 1.0 if predicted_terminate == scenario["expected_terminate"] else 0.0
        latency_sum += float(len(scenario["nodes"]) + len(scenario["edges"]) + len(kernel_result["proposals"]))

    count = float(len(ordered))
    return {
        "mean_top_match": top_match_sum / count,
        "mean_termination_correct": termination_correct_sum / count,
        "mean_latency_units": latency_sum / count,
    }
