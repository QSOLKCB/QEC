"""v138.7.2 bootstrap: deterministic minimal ML benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar


def deterministic_gnn_decoder_kernel(
    *, nodes: Sequence[str], edges: Sequence[tuple[str, str]]
) -> Mapping[str, Any]:
    """Return kernel_result with shape: {'proposals': [{'target_nodes': list[str]}], ...}."""
    raise NotImplementedError(
        "deterministic_gnn_decoder_kernel must be provided by runtime"
    )


def early_termination_via_dark_state_proofs(
    *, kernel_result: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return termination_result with shape: {'decision': {'terminate_early': bool}, ...}."""
    raise NotImplementedError(
        "early_termination_via_dark_state_proofs must be provided by runtime"
    )


REQUIRED_FIELDS = (
    "id",
    "nodes",
    "edges",
    "expected_top_node",
    "expected_terminate",
)

ROUNDING_DECIMALS = 12


@dataclass(frozen=True)
class _CanonicalDataclass:
    """Base helpers for canonical serialization and stable hashing."""

    _hash_excluded_fields: ClassVar[tuple[str, ...]] = ()

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        raise NotImplementedError

    def to_canonical_json(self, *, include_hash_fields: bool = True) -> str:
        return json.dumps(
            self.to_dict(include_hash_fields=include_hash_fields),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def stable_hash(self) -> str:
        payload = self.to_canonical_json(include_hash_fields=False).encode("ascii")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class MLBenchmarkConfig(_CanonicalDataclass):
    api_version: str = "v138.7.2.1"
    message_rounds: int = 1
    rounding_decimals: int = ROUNDING_DECIMALS

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        del include_hash_fields
        return {
            "api_version": self.api_version,
            "message_rounds": self.message_rounds,
            "rounding_decimals": self.rounding_decimals,
        }


@dataclass(frozen=True)
class MLBenchmarkScenario(_CanonicalDataclass):
    id: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    expected_top_node: str
    expected_terminate: bool

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        del include_hash_fields
        return {
            "id": self.id,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "expected_top_node": self.expected_top_node,
            "expected_terminate": self.expected_terminate,
        }


@dataclass(frozen=True)
class MLBenchmarkCaseResult(_CanonicalDataclass):
    scenario_id: str
    predicted_top_node: str
    predicted_terminate: bool
    top_match: float
    termination_correct: float
    latency_units: float
    proposal_count: float
    node_count: float
    edge_count: float
    message_rounds: float
    normalized_latency_score: float

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        del include_hash_fields
        return {
            "scenario_id": self.scenario_id,
            "predicted_top_node": self.predicted_top_node,
            "predicted_terminate": self.predicted_terminate,
            "top_match": self.top_match,
            "termination_correct": self.termination_correct,
            "latency_units": self.latency_units,
            "proposal_count": self.proposal_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "message_rounds": self.message_rounds,
            "normalized_latency_score": self.normalized_latency_score,
        }


@dataclass(frozen=True)
class MLBenchmarkAggregate(_CanonicalDataclass):
    mean_top_match: float
    mean_termination_correct: float
    mean_latency_units: float
    mean_normalized_latency_score: float

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        del include_hash_fields
        return {
            "mean_top_match": self.mean_top_match,
            "mean_termination_correct": self.mean_termination_correct,
            "mean_latency_units": self.mean_latency_units,
            "mean_normalized_latency_score": self.mean_normalized_latency_score,
        }


@dataclass(frozen=True)
class MLBenchmarkReceipt(_CanonicalDataclass):
    _hash_excluded_fields: ClassVar[tuple[str, ...]] = ("receipt_hash",)

    config_hash: str = ""
    scenario_manifest_hash: str = ""
    aggregate_hash: str = ""
    benchmark_result_hash: str = ""
    replay_identity: str = ""
    receipt_hash: str = ""

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        payload = {
            "config_hash": self.config_hash,
            "scenario_manifest_hash": self.scenario_manifest_hash,
            "aggregate_hash": self.aggregate_hash,
            "benchmark_result_hash": self.benchmark_result_hash,
            "replay_identity": self.replay_identity,
            "receipt_hash": self.receipt_hash,
        }
        if include_hash_fields:
            return payload
        return {
            key: value
            for key, value in payload.items()
            if key not in self._hash_excluded_fields
        }


@dataclass(frozen=True)
class MLAccuracyLatencyBenchmarkResult(_CanonicalDataclass):
    config: MLBenchmarkConfig
    scenarios: tuple[MLBenchmarkScenario, ...]
    case_results: tuple[MLBenchmarkCaseResult, ...]
    aggregate: MLBenchmarkAggregate
    receipt: MLBenchmarkReceipt

    def to_dict(self, *, include_hash_fields: bool = True) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(include_hash_fields=include_hash_fields),
            "scenarios": [
                scenario.to_dict(include_hash_fields=include_hash_fields)
                for scenario in self.scenarios
            ],
            "case_results": [
                case.to_dict(include_hash_fields=include_hash_fields)
                for case in self.case_results
            ],
            "aggregate": self.aggregate.to_dict(
                include_hash_fields=include_hash_fields
            ),
            "receipt": self.receipt.to_dict(include_hash_fields=include_hash_fields),
        }


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
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(not isinstance(node, str) for node in nodes)
    ):
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


def _round_api_float(value: float, *, rounding_decimals: int = ROUNDING_DECIMALS) -> float:
    """Round API-facing floats using the configured decimal precision."""
    if not isinstance(rounding_decimals, int):
        raise ValueError("rounding_decimals must be int")
    if rounding_decimals < 0:
        raise ValueError("rounding_decimals must be >= 0")
    return round(float(value), rounding_decimals)


def _normalize_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
) -> tuple[MLBenchmarkScenario, ...]:
    normalized = [_validate_scenario(s) for s in scenarios]
    scenario_ids = [s["id"] for s in normalized]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("duplicate scenario ids")

    ordered = sorted(normalized, key=lambda item: item["id"])
    return tuple(
        MLBenchmarkScenario(
            id=scenario["id"],
            nodes=tuple(scenario["nodes"]),
            edges=tuple((edge[0], edge[1]) for edge in scenario["edges"]),
            expected_top_node=scenario["expected_top_node"],
            expected_terminate=scenario["expected_terminate"],
        )
        for scenario in ordered
    )


def _scenario_manifest_hash(scenarios: tuple[MLBenchmarkScenario, ...]) -> str:
    payload = {"scenarios": [scenario.to_dict() for scenario in scenarios]}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def build_ml_benchmark_full_result(
    scenarios: Sequence[Mapping[str, Any]],
) -> MLAccuracyLatencyBenchmarkResult:
    """Run deterministic benchmark and return full canonical result dataclass."""
    if not isinstance(scenarios, Sequence) or not scenarios:
        raise ValueError("scenarios must be a non-empty sequence")

    config = MLBenchmarkConfig()
    normalized_scenarios = _normalize_scenarios(scenarios)

    top_match_sum = 0.0
    termination_correct_sum = 0.0
    latency_sum = 0.0
    normalized_latency_sum = 0.0
    case_results: list[MLBenchmarkCaseResult] = []

    for scenario in normalized_scenarios:
        kernel_result = deterministic_gnn_decoder_kernel(
            nodes=scenario.nodes, edges=scenario.edges
        )
        if not isinstance(kernel_result, Mapping):
            raise ValueError("kernel_result must be mapping-like")
        predicted_top = _extract_predicted_top(kernel_result)

        termination_result = early_termination_via_dark_state_proofs(
            kernel_result=kernel_result
        )
        if not isinstance(termination_result, Mapping):
            raise ValueError("termination_result must be mapping-like")
        predicted_terminate = _extract_predicted_terminate(termination_result)

        proposal_count = float(len(kernel_result["proposals"]))
        node_count = float(len(scenario.nodes))
        edge_count = float(len(scenario.edges))
        message_rounds = float(config.message_rounds)
        latency_units = node_count + edge_count + proposal_count
        normalized_latency_score = 1.0 / (1.0 + latency_units)

        top_match = 1.0 if predicted_top == scenario.expected_top_node else 0.0
        termination_correct = (
            1.0 if predicted_terminate == scenario.expected_terminate else 0.0
        )

        top_match_sum += top_match
        termination_correct_sum += termination_correct
        latency_sum += latency_units
        normalized_latency_sum += normalized_latency_score

        case_results.append(
            MLBenchmarkCaseResult(
                scenario_id=scenario.id,
                predicted_top_node=predicted_top,
                predicted_terminate=predicted_terminate,
                top_match=top_match,
                termination_correct=termination_correct,
                latency_units=latency_units,
                proposal_count=proposal_count,
                node_count=node_count,
                edge_count=edge_count,
                message_rounds=message_rounds,
                normalized_latency_score=normalized_latency_score,
            )
        )

    count = float(len(normalized_scenarios))
    aggregate = MLBenchmarkAggregate(
        mean_top_match=top_match_sum / count,
        mean_termination_correct=termination_correct_sum / count,
        mean_latency_units=latency_sum / count,
        mean_normalized_latency_score=normalized_latency_sum / count,
    )

    base_result = MLAccuracyLatencyBenchmarkResult(
        config=config,
        scenarios=normalized_scenarios,
        case_results=tuple(case_results),
        aggregate=aggregate,
        receipt=MLBenchmarkReceipt(),
    )
    config_hash = config.stable_hash()
    scenario_manifest_hash = _scenario_manifest_hash(normalized_scenarios)
    aggregate_hash = aggregate.stable_hash()
    benchmark_result_hash = base_result.stable_hash()
    replay_identity = hashlib.sha256(
        f"{config_hash}:{scenario_manifest_hash}".encode("ascii")
    ).hexdigest()
    provisional_receipt = MLBenchmarkReceipt(
        config_hash=config_hash,
        scenario_manifest_hash=scenario_manifest_hash,
        aggregate_hash=aggregate_hash,
        benchmark_result_hash=benchmark_result_hash,
        replay_identity=replay_identity,
        receipt_hash="",
    )
    receipt = MLBenchmarkReceipt(
        config_hash=config_hash,
        scenario_manifest_hash=scenario_manifest_hash,
        aggregate_hash=aggregate_hash,
        benchmark_result_hash=benchmark_result_hash,
        replay_identity=replay_identity,
        receipt_hash=provisional_receipt.stable_hash(),
    )
    return MLAccuracyLatencyBenchmarkResult(
        config=config,
        scenarios=normalized_scenarios,
        case_results=tuple(case_results),
        aggregate=aggregate,
        receipt=receipt,
    )


def _extract_predicted_top(kernel_result: Mapping[str, Any]) -> str:
    proposals = kernel_result.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise ValueError("kernel_result must contain non-empty list at key 'proposals'")

    first_proposal = proposals[0]
    if not isinstance(first_proposal, Mapping):
        raise ValueError("kernel_result proposals[0] must be mapping-like")

    target_nodes = first_proposal.get("target_nodes")
    if (
        not isinstance(target_nodes, Sequence)
        or isinstance(target_nodes, (str, bytes))
        or not target_nodes
    ):
        raise ValueError(
            "kernel_result proposals[0]['target_nodes'] must be non-empty sequence[str]"
        )
    if any(not isinstance(node, str) for node in target_nodes):
        raise ValueError(
            "kernel_result proposals[0]['target_nodes'] must be non-empty sequence[str]"
        )

    return target_nodes[0]


def _extract_predicted_terminate(termination_result: Mapping[str, Any]) -> bool:
    decision = termination_result.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("termination_result must contain mapping-like key 'decision'")

    terminate_early = decision.get("terminate_early")
    if not isinstance(terminate_early, bool):
        raise ValueError("termination_result decision['terminate_early'] must be bool")

    return terminate_early


def run_minimal_ml_benchmark(
    scenarios: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Run deterministic benchmark over scenarios and return mean float metrics."""
    full_result = build_ml_benchmark_full_result(scenarios)
    aggregate = full_result.aggregate
    return {
        "mean_top_match": _round_api_float(aggregate.mean_top_match),
        "mean_termination_correct": _round_api_float(
            aggregate.mean_termination_correct
        ),
        "mean_latency_units": _round_api_float(aggregate.mean_latency_units),
        "mean_normalized_latency_score": _round_api_float(
            aggregate.mean_normalized_latency_score
        ),
    }
