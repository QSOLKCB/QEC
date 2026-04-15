"""v137.19.0 — Governance Benchmark Battery.

Deterministic benchmark + metrics + receipt layer over
DeterministicAgentSimulationSandbox outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from qec.orchestration.deterministic_agent_simulation_sandbox import (
    DeterministicAgentSimulationSandbox,
)


REQUIRED_METRIC_NAMES: Tuple[str, ...] = (
    "allow_count",
    "deny_count",
    "allow_rate",
    "deny_rate",
    "replay_stability_rate",
    "continuity_success_rate",
    "boundary_failure_rate",
    "policy_drift_rate",
    "mean_trace_length",
    "max_trace_length",
    "receipt_reproducibility",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _deep_canonical_copy(data: Any) -> Any:
    return json.loads(_canonical_json(data))


def _stable_str_tuple(values: Sequence[Any]) -> Tuple[str, ...]:
    normalized = {str(v).strip() for v in values}
    normalized.discard("")
    return tuple(sorted(normalized))


def _normalized_mapping_tuple(values: Sequence[Any]) -> Tuple[Dict[str, Any], ...]:
    normalized: list[Dict[str, Any]] = []
    for raw in values:
        if isinstance(raw, Mapping):
            clean = {
                str(k): _deep_canonical_copy(v)
                for k, v in sorted(raw.items(), key=lambda item: str(item[0]))
                if isinstance(k, str) and str(k)
            }
            normalized.append(clean)
    return tuple(sorted(normalized, key=lambda item: _canonical_json(item)))


def _safe_rate(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if denominator <= 0.0:
        return default
    return float(numerator) / float(denominator)


@dataclass(frozen=True)
class GovernanceBenchmarkScenario:
    simulation_set: Tuple[DeterministicAgentSimulationSandbox, ...]
    benchmark_rules: Dict[str, Any]
    firewall_policy_variants: Tuple[Dict[str, Any], ...]
    covenant_variants: Tuple[Dict[str, Any], ...]
    boundary_rule_variants: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_set": [item.to_dict() for item in self.simulation_set],
            "benchmark_rules": _deep_canonical_copy(self.benchmark_rules),
            "firewall_policy_variants": [
                _deep_canonical_copy(item) for item in self.firewall_policy_variants
            ],
            "covenant_variants": [_deep_canonical_copy(item) for item in self.covenant_variants],
            "boundary_rule_variants": [_deep_canonical_copy(item) for item in self.boundary_rule_variants],
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class GovernanceBenchmarkResult:
    simulation_index: int
    simulation_hash: str
    simulation_scenario_hash: str
    sandbox_receipt_hash: str
    trace_length: int
    allow_count: int
    deny_count: int
    continuity_success_count: int
    boundary_failure_count: int
    replay_step_signature: str
    policy_signature: str
    validation_violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_index": self.simulation_index,
            "simulation_hash": self.simulation_hash,
            "simulation_scenario_hash": self.simulation_scenario_hash,
            "sandbox_receipt_hash": self.sandbox_receipt_hash,
            "trace_length": self.trace_length,
            "allow_count": self.allow_count,
            "deny_count": self.deny_count,
            "continuity_success_count": self.continuity_success_count,
            "boundary_failure_count": self.boundary_failure_count,
            "replay_step_signature": self.replay_step_signature,
            "policy_signature": self.policy_signature,
            "validation_violations": list(self.validation_violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class GovernanceBenchmarkReceipt:
    scenario_hash: str
    benchmark_result_hashes: Tuple[str, ...]
    aggregate_metrics: Dict[str, float]
    validation_violations: Tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "benchmark_result_hashes": list(self.benchmark_result_hashes),
            "aggregate_metrics": _deep_canonical_copy(self.aggregate_metrics),
            "validation_violations": list(self.validation_violations),
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class GovernanceBenchmarkBattery:
    scenario: GovernanceBenchmarkScenario
    benchmark_results: Tuple[GovernanceBenchmarkResult, ...]
    benchmark_receipt: GovernanceBenchmarkReceipt
    validation_violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "benchmark_results": [row.to_dict() for row in self.benchmark_results],
            "benchmark_receipt": self.benchmark_receipt.to_dict(),
            "validation_violations": list(self.validation_violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


def build_governance_benchmark_scenario(
    *,
    simulation_set: Sequence[Any],
    firewall_policy_variants: Sequence[Any] = (),
    covenant_variants: Sequence[Any] = (),
    boundary_rule_variants: Sequence[Any] = (),
) -> GovernanceBenchmarkScenario:
    simulations: Tuple[DeterministicAgentSimulationSandbox, ...] = tuple(
        sorted(
            (item for item in simulation_set if isinstance(item, DeterministicAgentSimulationSandbox)),
            key=lambda item: (item.scenario.stable_hash(), item.stable_hash()),
        )
    )
    benchmark_rules = {
        "required_metric_names": list(REQUIRED_METRIC_NAMES),
        "deterministic_ordering": True,
        "non_mutating": True,
    }
    return GovernanceBenchmarkScenario(
        simulation_set=simulations,
        benchmark_rules=benchmark_rules,
        firewall_policy_variants=_normalized_mapping_tuple(firewall_policy_variants),
        covenant_variants=_normalized_mapping_tuple(covenant_variants),
        boundary_rule_variants=_normalized_mapping_tuple(boundary_rule_variants),
    )


def validate_governance_benchmark(scenario: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        if not isinstance(scenario, GovernanceBenchmarkScenario):
            return ("malformed_benchmark:type",)

        if not isinstance(scenario.simulation_set, tuple):
            violations.append("malformed_benchmark:simulation_set")
        for idx, item in enumerate(scenario.simulation_set):
            if not isinstance(item, DeterministicAgentSimulationSandbox):
                violations.append(f"malformed_benchmark:simulation:{idx}")

        if not isinstance(scenario.benchmark_rules, Mapping):
            violations.append("malformed_benchmark:benchmark_rules")
        else:
            metrics = scenario.benchmark_rules.get("required_metric_names", ())
            if tuple(metrics) != REQUIRED_METRIC_NAMES:
                violations.append("malformed_benchmark:required_metric_names")

        for field_name, variants in (
            ("firewall_policy_variants", scenario.firewall_policy_variants),
            ("covenant_variants", scenario.covenant_variants),
            ("boundary_rule_variants", scenario.boundary_rule_variants),
        ):
            if not isinstance(variants, tuple):
                violations.append(f"malformed_benchmark:{field_name}")
                continue
            for idx, item in enumerate(variants):
                if not isinstance(item, Mapping):
                    violations.append(f"malformed_benchmark:{field_name}:{idx}")

    except Exception as exc:  # pragma: no cover
        violations.append(f"validator_error:{type(exc).__name__}")

    return tuple(sorted(set(violations)))


def _build_result(simulation_index: int, simulation: DeterministicAgentSimulationSandbox) -> GovernanceBenchmarkResult:
    ordered_trace = tuple(sorted(simulation.simulated_execution_trace, key=lambda row: row.step_index))
    allow_count = sum(1 for step in ordered_trace if step.allow)
    deny_count = len(ordered_trace) - allow_count
    continuity_success_count = sum(
        1
        for step in ordered_trace
        if bool(step.simulated_boundary_result.get("audit_receipt", {}).get("continuity_ok", False))
    )
    boundary_failure_count = sum(
        1
        for step in ordered_trace
        if (
            bool(step.simulated_boundary_result.get("violated_rules", ()))
            or bool(step.simulated_boundary_result.get("findings", ()))
            or not bool(step.simulated_boundary_result.get("within_boundary", True))
        )
    )
    replay_step_signature = _sha256_hex(
        _canonical_json([step.step_replay_identity for step in ordered_trace]).encode("utf-8")
    )
    policy_signature = _sha256_hex(_canonical_json([bool(step.allow) for step in ordered_trace]).encode("utf-8"))

    return GovernanceBenchmarkResult(
        simulation_index=simulation_index,
        simulation_hash=simulation.stable_hash(),
        simulation_scenario_hash=simulation.scenario.stable_hash(),
        sandbox_receipt_hash=simulation.sandbox_receipt.receipt_hash,
        trace_length=len(ordered_trace),
        allow_count=allow_count,
        deny_count=deny_count,
        continuity_success_count=continuity_success_count,
        boundary_failure_count=boundary_failure_count,
        replay_step_signature=replay_step_signature,
        policy_signature=policy_signature,
        validation_violations=_stable_str_tuple(simulation.validation_violations),
    )


def _aggregate_metrics(results: Sequence[GovernanceBenchmarkResult]) -> Dict[str, float]:
    ordered_results = tuple(sorted(results, key=lambda row: (row.simulation_scenario_hash, row.simulation_index, row.simulation_hash)))
    total_steps = float(sum(row.trace_length for row in ordered_results))
    allow_count = float(sum(row.allow_count for row in ordered_results))
    deny_count = float(sum(row.deny_count for row in ordered_results))
    continuity_success = float(sum(row.continuity_success_count for row in ordered_results))
    boundary_failures = float(sum(row.boundary_failure_count for row in ordered_results))

    groups: Dict[str, list[GovernanceBenchmarkResult]] = {}
    for row in ordered_results:
        groups.setdefault(row.simulation_scenario_hash, []).append(row)

    replay_comparisons = 0
    replay_matches = 0
    policy_drift_events = 0
    receipt_matches = 0
    for scenario_hash in sorted(groups.keys()):
        group = sorted(groups[scenario_hash], key=lambda row: (row.simulation_index, row.simulation_hash))
        if len(group) <= 1:
            continue
        baseline = group[0]
        for current in group[1:]:
            replay_comparisons += 1
            if current.replay_step_signature == baseline.replay_step_signature:
                replay_matches += 1
            if current.policy_signature != baseline.policy_signature:
                policy_drift_events += 1
            if current.sandbox_receipt_hash == baseline.sandbox_receipt_hash:
                receipt_matches += 1

    replay_default = 1.0
    receipt_default = 1.0
    return {
        "allow_count": allow_count,
        "deny_count": deny_count,
        "allow_rate": _safe_rate(allow_count, total_steps),
        "deny_rate": _safe_rate(deny_count, total_steps),
        "replay_stability_rate": _safe_rate(replay_matches, replay_comparisons, default=replay_default),
        "continuity_success_rate": _safe_rate(continuity_success, total_steps),
        "boundary_failure_rate": _safe_rate(boundary_failures, total_steps),
        "policy_drift_rate": _safe_rate(policy_drift_events, replay_comparisons, default=0.0),
        "mean_trace_length": _safe_rate(total_steps, float(len(ordered_results))),
        "max_trace_length": float(max((row.trace_length for row in ordered_results), default=0)),
        "receipt_reproducibility": _safe_rate(receipt_matches, replay_comparisons, default=receipt_default),
    }


def build_governance_benchmark_receipt(
    *,
    scenario: GovernanceBenchmarkScenario,
    benchmark_results: Sequence[GovernanceBenchmarkResult],
    validation_violations: Sequence[str] = (),
) -> GovernanceBenchmarkReceipt:
    try:
        scenario_hash = scenario.stable_hash()
    except Exception:
        scenario_hash = "malformed_benchmark"
    ordered_results = tuple(
        sorted(
            benchmark_results,
            key=lambda row: (row.simulation_scenario_hash, row.simulation_index, row.simulation_hash),
        )
    )
    result_hashes = tuple(row.stable_hash() for row in ordered_results)
    metrics = _aggregate_metrics(ordered_results)
    ordered_violations = _stable_str_tuple(validation_violations)
    preimage = {
        "scenario_hash": scenario_hash,
        "benchmark_result_hashes": list(result_hashes),
        "aggregate_metrics": _deep_canonical_copy(metrics),
        "validation_violations": list(ordered_violations),
    }
    receipt_hash = _sha256_hex(_canonical_json(preimage).encode("utf-8"))
    return GovernanceBenchmarkReceipt(
        scenario_hash=scenario_hash,
        benchmark_result_hashes=result_hashes,
        aggregate_metrics=metrics,
        validation_violations=ordered_violations,
        receipt_hash=receipt_hash,
    )


def run_governance_benchmark_battery(
    scenario: GovernanceBenchmarkScenario,
) -> GovernanceBenchmarkBattery:
    violations = validate_governance_benchmark(scenario)
    if violations:
        receipt = build_governance_benchmark_receipt(
            scenario=scenario,
            benchmark_results=(),
            validation_violations=violations,
        )
        return GovernanceBenchmarkBattery(
            scenario=scenario,
            benchmark_results=(),
            benchmark_receipt=receipt,
            validation_violations=violations,
        )

    ordered_simulations = tuple(
        sorted(
            scenario.simulation_set,
            key=lambda item: (item.scenario.stable_hash(), item.stable_hash()),
        )
    )
    results = tuple(_build_result(index, simulation) for index, simulation in enumerate(ordered_simulations))
    receipt = build_governance_benchmark_receipt(
        scenario=scenario,
        benchmark_results=results,
        validation_violations=violations,
    )
    return GovernanceBenchmarkBattery(
        scenario=scenario,
        benchmark_results=results,
        benchmark_receipt=receipt,
        validation_violations=violations,
    )


def compare_governance_benchmark_replay(
    baseline: GovernanceBenchmarkBattery,
    replay: GovernanceBenchmarkBattery,
) -> Dict[str, Any]:
    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario")
    if baseline.benchmark_receipt.receipt_hash != replay.benchmark_receipt.receipt_hash:
        mismatches.append("benchmark_receipt")
    if tuple(row.stable_hash() for row in baseline.benchmark_results) != tuple(
        row.stable_hash() for row in replay.benchmark_results
    ):
        mismatches.append("benchmark_results")
    if baseline.validation_violations != replay.validation_violations:
        mismatches.append("validation_violations")

    return {
        "match": len(mismatches) == 0,
        "mismatch_fields": tuple(mismatches),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_governance_benchmark(
    battery: GovernanceBenchmarkBattery,
) -> Dict[str, Any]:
    ordered_results = tuple(
        sorted(
            battery.benchmark_results,
            key=lambda row: (row.simulation_scenario_hash, row.simulation_index, row.simulation_hash),
        )
    )
    return {
        "scenario_hash": battery.scenario.stable_hash(),
        "benchmark_hash": battery.stable_hash(),
        "receipt_hash": battery.benchmark_receipt.receipt_hash,
        "validation_violations": list(battery.validation_violations),
        "result_count": len(ordered_results),
        "aggregate_metrics": _deep_canonical_copy(battery.benchmark_receipt.aggregate_metrics),
        "results": [
            {
                "simulation_index": row.simulation_index,
                "simulation_hash": row.simulation_hash,
                "simulation_scenario_hash": row.simulation_scenario_hash,
                "trace_length": row.trace_length,
                "allow_count": row.allow_count,
                "deny_count": row.deny_count,
                "continuity_success_count": row.continuity_success_count,
                "boundary_failure_count": row.boundary_failure_count,
                "policy_signature": row.policy_signature,
                "replay_step_signature": row.replay_step_signature,
                "result_hash": row.stable_hash(),
            }
            for row in ordered_results
        ],
    }


__all__ = (
    "GovernanceBenchmarkScenario",
    "GovernanceBenchmarkResult",
    "GovernanceBenchmarkReceipt",
    "GovernanceBenchmarkBattery",
    "build_governance_benchmark_scenario",
    "run_governance_benchmark_battery",
    "validate_governance_benchmark",
    "build_governance_benchmark_receipt",
    "compare_governance_benchmark_replay",
    "summarize_governance_benchmark",
)
