from __future__ import annotations

"""v137.6.4 deterministic planning replay certification battery.

Release laws:
- PLANNING_REPLAY_CERTIFICATION_LAW:
  Identical full planning input must replay byte-identically.
- END_TO_END_FRONTIER_STABILITY_RULE:
  Frontier ordering remains stable across full-pipeline replays.
- PRUNING_POLICY_RECEIPT_REPLAY_INVARIANT:
  v137.6.2 pruning and v137.6.3 policy receipts remain hash-stable.
- ADVERSARIAL_REPLAY_FAILURE_CONTAINMENT_RULE:
  Malformed input failures are deterministic and replay-safe.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from qec.analysis.autonomous_planning_search_kernel import (
    generate_plan_receipt,
    synthesize_plan_ir,
)
from qec.analysis.policy_constrained_planner import (
    analyze_policy_constrained_frontier,
    generate_policy_decision_receipt,
)
from qec.analysis.route_dead_end_pruning import (
    analyze_dead_end_pruning,
    generate_dead_end_pruning_receipt,
)
from qec.analysis.route_graph_execution_runtime import (
    execute_route_graph,
    generate_execution_receipt,
)

PLANNING_REPLAY_BATTERY_VERSION: str = "v137.6.4"
GENESIS_HASH: str = "0" * 64
_PRECISION: int = 12


@dataclass(frozen=True)
class PlanningReplayRunRecord:
    run_index: int
    plan_hash: str
    execution_hash: str
    pruning_hash: str
    policy_hash: str
    artifact_bytes_hash: str
    frontier_order: tuple[str, ...]
    chain_hash: str


@dataclass(frozen=True)
class PlanningReplayCertificationArtifact:
    schema_version: str
    source_plan_hash: str
    route_graph_hash: str
    replay_run_count: int
    byte_identity_verified: bool
    frontier_order_stable: bool
    pruning_hash_stable: bool
    policy_hash_stable: bool
    adversarial_cases_passed: bool
    replay_identity_chain: tuple[str, ...]
    certification_score: float
    stable_certification_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "route_graph_hash": self.route_graph_hash,
            "replay_run_count": int(self.replay_run_count),
            "byte_identity_verified": bool(self.byte_identity_verified),
            "frontier_order_stable": bool(self.frontier_order_stable),
            "pruning_hash_stable": bool(self.pruning_hash_stable),
            "policy_hash_stable": bool(self.policy_hash_stable),
            "adversarial_cases_passed": bool(self.adversarial_cases_passed),
            "replay_identity_chain": list(self.replay_identity_chain),
            "certification_score": _round64(self.certification_score),
            "stable_certification_hash": self.stable_certification_hash,
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class PlanningReplayCertificationReceipt:
    schema_version: str
    stable_certification_hash: str
    report_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "stable_certification_hash": self.stable_certification_hash,
            "report_hash": self.report_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_replay_run_count(replay_run_count: int) -> int:
    value = int(replay_run_count)
    if value < 2:
        raise ValueError("replay_run_count must be >= 2")
    return value


def _validate_stress_run_count(stress_run_count: int) -> int:
    value = int(stress_run_count)
    if value < 1:
        raise ValueError("stress_run_count must be >= 1")
    return value


def _failure_signature(error: Exception) -> str:
    return _sha256_hex_mapping({"error_type": type(error).__name__, "message": str(error)})


def _execute_single_replay_run(
    world_state: Mapping[str, Any],
    objective: Mapping[str, Any],
    candidate_routes: Sequence[Sequence[str]],
    route_graph: Mapping[str, Sequence[str]],
    *,
    initial_node: str,
    search_depth: int,
    max_path_length: int,
    policy_rules: Mapping[str, Any],
    run_index: int,
) -> PlanningReplayRunRecord:
    plan = synthesize_plan_ir(
        world_state,
        objective,
        candidate_routes,
        search_depth=search_depth,
        enable_v137_6_search=True,
    )
    plan_receipt = generate_plan_receipt(plan)

    execution = execute_route_graph(
        plan.stable_plan_hash,
        route_graph,
        initial_node=initial_node,
        world_state=world_state,
        max_path_length=max_path_length,
        enable_v137_6_route_runtime=True,
    )
    execution_receipt = generate_execution_receipt(execution)

    pruning = analyze_dead_end_pruning(
        plan.stable_plan_hash,
        route_graph,
        current_path=execution.executed_route,
        max_path_length=max_path_length,
        enable_v137_6_2_dead_end_pruning=True,
    )
    pruning_receipt = generate_dead_end_pruning_receipt(pruning)

    policy = analyze_policy_constrained_frontier(
        plan.stable_plan_hash,
        route_graph,
        current_path=execution.executed_route,
        frontier_candidates=pruning.examined_candidates,
        max_path_length=max_path_length,
        policy_rules=policy_rules,
        enable_v137_6_3_policy_constraints=True,
    )
    policy_receipt = generate_policy_decision_receipt(policy)

    artifact_payload = {
        "plan": plan.to_dict(),
        "plan_receipt": plan_receipt.to_dict(),
        "execution": execution.to_dict(),
        "execution_receipt": execution_receipt.to_dict(),
        "pruning": pruning.to_dict(),
        "pruning_receipt": pruning_receipt.to_dict(),
        "policy": policy.to_dict(),
        "policy_receipt": policy_receipt.to_dict(),
    }
    artifact_bytes = _canonical_json(artifact_payload).encode("utf-8")
    artifact_bytes_hash = _sha256_hex_bytes(artifact_bytes)

    chain_hash = _sha256_hex_mapping(
        {
            "schema_version": PLANNING_REPLAY_BATTERY_VERSION,
            "run_index": int(run_index),
            "prior_hash": GENESIS_HASH,
            "plan_hash": plan.stable_plan_hash,
            "execution_hash": execution.stable_execution_hash,
            "pruning_hash": pruning.stable_pruning_hash,
            "policy_hash": policy.stable_policy_hash,
            "artifact_bytes_hash": artifact_bytes_hash,
            "plan_receipt_hash": plan_receipt.receipt_hash,
            "execution_receipt_hash": execution_receipt.receipt_hash,
            "pruning_receipt_hash": pruning_receipt.receipt_hash,
            "policy_receipt_hash": policy_receipt.receipt_hash,
        }
    )

    return PlanningReplayRunRecord(
        run_index=run_index,
        plan_hash=plan.stable_plan_hash,
        execution_hash=execution.stable_execution_hash,
        pruning_hash=pruning.stable_pruning_hash,
        policy_hash=policy.stable_policy_hash,
        artifact_bytes_hash=artifact_bytes_hash,
        frontier_order=policy.examined_candidates,
        chain_hash=chain_hash,
    )


def run_adversarial_replay_harness(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    current_path: Sequence[str],
    frontier_candidates: Sequence[str],
    max_path_length: int,
) -> tuple[str, ...]:
    signatures: list[str] = []

    def _record_failure(case_name: str, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as error:  # noqa: BLE001
            signatures.append(
                _sha256_hex_mapping(
                    {
                        "case": case_name,
                        "signature": _failure_signature(error),
                    }
                )
            )
            return
        raise ValueError(f"{case_name} expected deterministic failure")

    _record_failure(
        "unknown_frontier_node",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=current_path,
            frontier_candidates=tuple(frontier_candidates) + ("unknown_frontier_node",),
            max_path_length=max_path_length,
            policy_rules={},
            enable_v137_6_3_policy_constraints=True,
        ),
    )
    _record_failure(
        "invalid_route_graph",
        lambda: analyze_dead_end_pruning(
            source_plan_hash,
            {"start": "goal"},
            current_path=("start",),
            max_path_length=max_path_length,
            enable_v137_6_2_dead_end_pruning=True,
        ),
    )
    _record_failure(
        "invalid_path",
        lambda: analyze_dead_end_pruning(
            source_plan_hash,
            route_graph,
            current_path=("missing",),
            max_path_length=max_path_length,
            enable_v137_6_2_dead_end_pruning=True,
        ),
    )
    _record_failure(
        "invalid_max_depth",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=current_path,
            frontier_candidates=frontier_candidates,
            max_path_length=max_path_length,
            policy_rules={"max_depth": 0},
            enable_v137_6_3_policy_constraints=True,
        ),
    )
    _record_failure(
        "invalid_policy_rule_schema",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=current_path,
            frontier_candidates=frontier_candidates,
            max_path_length=max_path_length,
            policy_rules={"unsupported_rule": True},
            enable_v137_6_3_policy_constraints=True,
        ),
    )
    _record_failure(
        "frontier_duplicate_nodes",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=current_path,
            frontier_candidates=tuple(frontier_candidates) + (frontier_candidates[0],),
            max_path_length=max_path_length,
            policy_rules={},
            enable_v137_6_3_policy_constraints=True,
        ),
    )
    _record_failure(
        "invalid_terminal_targets",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=current_path,
            frontier_candidates=frontier_candidates,
            max_path_length=max_path_length,
            policy_rules={"required_terminal_subset": tuple()},
            enable_v137_6_3_policy_constraints=True,
        ),
    )
    _record_failure(
        "zero_length_path",
        lambda: analyze_policy_constrained_frontier(
            source_plan_hash,
            route_graph,
            current_path=tuple(),
            frontier_candidates=frontier_candidates,
            max_path_length=max_path_length,
            policy_rules={},
            enable_v137_6_3_policy_constraints=True,
        ),
    )

    return tuple(signatures)


def certify_planning_replay_battery(
    world_state: Mapping[str, Any],
    objective: Mapping[str, Any],
    candidate_routes: Sequence[Sequence[str]],
    route_graph: Mapping[str, Sequence[str]],
    *,
    initial_node: str,
    search_depth: int,
    max_path_length: int,
    policy_rules: Mapping[str, Any],
    replay_run_count: int = 2,
    stress_run_count: int = 10,
    enable_v137_6_4_replay_battery: bool = False,
) -> PlanningReplayCertificationArtifact:
    if not enable_v137_6_4_replay_battery:
        raise ValueError("enable_v137_6_4_replay_battery must be True to enable v137.6.4 replay battery")

    run_count = _validate_replay_run_count(replay_run_count)
    stress_count = _validate_stress_run_count(stress_run_count)

    run_records = tuple(
        _execute_single_replay_run(
            world_state,
            objective,
            candidate_routes,
            route_graph,
            initial_node=initial_node,
            search_depth=search_depth,
            max_path_length=max_path_length,
            policy_rules=policy_rules,
            run_index=index,
        )
        for index in range(run_count)
    )

    byte_identity_verified = len({record.artifact_bytes_hash for record in run_records}) == 1
    frontier_order_stable = len({record.frontier_order for record in run_records}) == 1
    pruning_hash_stable = len({record.pruning_hash for record in run_records}) == 1
    policy_hash_stable = len({record.policy_hash for record in run_records}) == 1

    baseline = run_records[0]
    stress_hashes = tuple(
        _execute_single_replay_run(
            world_state,
            objective,
            candidate_routes,
            route_graph,
            initial_node=initial_node,
            search_depth=search_depth,
            max_path_length=max_path_length,
            policy_rules=policy_rules,
            run_index=run_count + idx,
        ).artifact_bytes_hash
        for idx in range(stress_count)
    )
    if any(hash_value != baseline.artifact_bytes_hash for hash_value in stress_hashes):
        raise ValueError("stress replay byte identity failed")

    adversarial_signatures_a = run_adversarial_replay_harness(
        baseline.plan_hash,
        route_graph,
        current_path=(initial_node,),
        frontier_candidates=baseline.frontier_order,
        max_path_length=max_path_length,
    )
    adversarial_signatures_b = run_adversarial_replay_harness(
        baseline.plan_hash,
        route_graph,
        current_path=(initial_node,),
        frontier_candidates=baseline.frontier_order,
        max_path_length=max_path_length,
    )
    adversarial_cases_passed = adversarial_signatures_a == adversarial_signatures_b

    checks = (
        bool(byte_identity_verified),
        bool(frontier_order_stable),
        bool(pruning_hash_stable),
        bool(policy_hash_stable),
        bool(adversarial_cases_passed),
    )
    certification_score = _round64(float(sum(1 for item in checks if item)) / float(len(checks)))

    replay_identity_chain: list[str] = [GENESIS_HASH]
    for record in run_records:
        replay_identity_chain.append(
            _sha256_hex_mapping(
                {
                    "prior_hash": replay_identity_chain[-1],
                    "run_index": int(record.run_index),
                    "run_chain_hash": record.chain_hash,
                }
            )
        )

    source_plan_hash = baseline.plan_hash
    route_graph_hash = baseline.execution_hash
    artifact_payload = {
        "schema_version": PLANNING_REPLAY_BATTERY_VERSION,
        "source_plan_hash": source_plan_hash,
        "route_graph_hash": route_graph_hash,
        "replay_run_count": run_count,
        "byte_identity_verified": byte_identity_verified,
        "frontier_order_stable": frontier_order_stable,
        "pruning_hash_stable": pruning_hash_stable,
        "policy_hash_stable": policy_hash_stable,
        "adversarial_cases_passed": adversarial_cases_passed,
        "replay_identity_chain": replay_identity_chain,
        "certification_score": certification_score,
    }
    stable_certification_hash = _sha256_hex_mapping(artifact_payload)

    return PlanningReplayCertificationArtifact(
        schema_version=PLANNING_REPLAY_BATTERY_VERSION,
        source_plan_hash=source_plan_hash,
        route_graph_hash=route_graph_hash,
        replay_run_count=run_count,
        byte_identity_verified=byte_identity_verified,
        frontier_order_stable=frontier_order_stable,
        pruning_hash_stable=pruning_hash_stable,
        policy_hash_stable=policy_hash_stable,
        adversarial_cases_passed=adversarial_cases_passed,
        replay_identity_chain=tuple(replay_identity_chain),
        certification_score=certification_score,
        stable_certification_hash=stable_certification_hash,
    )


def export_planning_replay_certification_bytes(artifact: PlanningReplayCertificationArtifact) -> bytes:
    if not isinstance(artifact, PlanningReplayCertificationArtifact):
        raise TypeError("artifact must be a PlanningReplayCertificationArtifact instance")
    return artifact.to_canonical_bytes()


def generate_planning_replay_certification_receipt(
    artifact: PlanningReplayCertificationArtifact,
) -> PlanningReplayCertificationReceipt:
    if not isinstance(artifact, PlanningReplayCertificationArtifact):
        raise TypeError("artifact must be a PlanningReplayCertificationArtifact instance")

    report_hash = _sha256_hex_mapping(artifact.to_dict())
    receipt_hash = _sha256_hex_mapping(
        {
            "schema_version": artifact.schema_version,
            "stable_certification_hash": artifact.stable_certification_hash,
            "report_hash": report_hash,
        }
    )
    return PlanningReplayCertificationReceipt(
        schema_version=artifact.schema_version,
        stable_certification_hash=artifact.stable_certification_hash,
        report_hash=report_hash,
        receipt_hash=receipt_hash,
    )


__all__ = [
    "PLANNING_REPLAY_BATTERY_VERSION",
    "GENESIS_HASH",
    "PlanningReplayCertificationArtifact",
    "PlanningReplayCertificationReceipt",
    "certify_planning_replay_battery",
    "export_planning_replay_certification_bytes",
    "generate_planning_replay_certification_receipt",
    "run_adversarial_replay_harness",
]
