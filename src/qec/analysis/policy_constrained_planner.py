from __future__ import annotations

"""v137.6.3 deterministic policy-constrained planner governance.

Release laws:
- POLICY_ADMISSIBILITY_LAW:
  Candidate policy verdicts are explicit, truthful, and derived from declared
  policy constraints only.
- DETERMINISTIC_ROUTE_GOVERNANCE_RULE:
  Same route graph + frontier + policy input yields identical verdict ordering,
  candidate decisions, and hashes.
- BOUNDED_POLICY_PRESSURE_INVARIANT:
  policy_pressure_score is always in [0.0, 1.0] and equals
  rejected_candidates / examined_candidates when examined_candidates > 0.
- REPLAY_SAFE_POLICY_RECEIPT_CHAIN:
  Artifacts and receipts export canonical JSON/bytes and include stable
  SHA-256 hashes for replay-safe auditing.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

POLICY_CONSTRAINED_PLANNER_VERSION: str = "v137.6.3"
GENESIS_HASH: str = "0" * 64
_PRECISION: int = 12
_ALLOWED_POLICY_KEYS: frozenset[str] = frozenset(
    {
        "max_depth",
        "forbidden_nodes",
        "forbidden_transitions",
        "required_terminal_subset",
        "route_cost_ceiling",
        "must_pass_through_nodes",
        "node_costs",
    }
)


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_hex_mapping(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _norm_token(value: str, *, name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be a non-empty token")
    return token


def _validate_hash(token: str, *, name: str) -> str:
    value = _norm_token(token, name=name)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_max_path_length(max_path_length: int) -> int:
    value = int(max_path_length)
    if value < 1:
        raise ValueError("max_path_length must be >= 1")
    return value


def _normalize_path(current_path: Sequence[str]) -> tuple[str, ...]:
    if not current_path:
        raise ValueError("current_path must not be empty")
    return tuple(_norm_token(node, name="path node") for node in current_path)


def _normalize_frontier(frontier_candidates: Sequence[str]) -> tuple[str, ...]:
    if isinstance(frontier_candidates, (str, bytes, bytearray)):
        raise ValueError("frontier_candidates must be a sequence of node tokens")
    canonical = tuple(sorted(_norm_token(node, name="frontier candidate") for node in frontier_candidates))
    if len(canonical) != len(set(canonical)):
        raise ValueError("frontier_candidates must not contain duplicates")
    return canonical


def _normalize_route_graph(route_graph: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not route_graph:
        raise ValueError("route_graph must not be empty")

    normalized: dict[str, tuple[str, ...]] = {}
    all_nodes: set[str] = set()
    for raw_node, raw_neighbors in route_graph.items():
        node = _norm_token(str(raw_node), name="route node")
        if node in normalized:
            raise ValueError(f"route node key collision after normalization: {node!r}")
        if isinstance(raw_neighbors, (str, bytes, bytearray)):
            raise ValueError(f"route_graph[{node}] neighbors must not be a string or bytes sequence")
        if not isinstance(raw_neighbors, Sequence):
            raise ValueError(f"route_graph[{node}] must be a sequence of next nodes")

        next_nodes = tuple(sorted(_norm_token(str(raw_next), name="next route node") for raw_next in raw_neighbors))
        if len(next_nodes) != len(set(next_nodes)):
            raise ValueError(f"route_graph[{node}] contains duplicate branch targets")

        normalized[node] = next_nodes
        all_nodes.add(node)
        for branch in next_nodes:
            all_nodes.add(branch)

    for node in sorted(all_nodes):
        normalized.setdefault(node, tuple())

    return normalized


def _collect_reachable(start_node: str, route_graph: Mapping[str, tuple[str, ...]], *, max_steps: int) -> frozenset[str]:
    if max_steps < 0:
        return frozenset()
    stack: list[tuple[str, int]] = [(start_node, max_steps)]
    visited: set[tuple[str, int]] = set()
    reached: set[str] = set()
    while stack:
        node, steps_left = stack.pop()
        state = (node, steps_left)
        if state in visited:
            continue
        visited.add(state)
        reached.add(node)
        if steps_left == 0:
            continue
        for nxt in reversed(route_graph.get(node, tuple())):
            stack.append((nxt, steps_left - 1))
    return frozenset(reached)


def _collect_reachable_terminals(
    start_node: str, route_graph: Mapping[str, tuple[str, ...]], *, max_steps: int
) -> frozenset[str]:
    if max_steps < 0:
        return frozenset()
    stack: list[tuple[str, int]] = [(start_node, max_steps)]
    visited: set[tuple[str, int]] = set()
    terminals: set[str] = set()
    while stack:
        node, steps_left = stack.pop()
        state = (node, steps_left)
        if state in visited:
            continue
        visited.add(state)
        branches = route_graph.get(node, tuple())
        if not branches:
            terminals.add(node)
            continue
        if steps_left == 0:
            continue
        for nxt in reversed(branches):
            stack.append((nxt, steps_left - 1))
    return frozenset(terminals)


def _normalize_policy_rules(policy_rules: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy_rules, Mapping):
        raise ValueError("policy_rules must be a mapping")
    unknown = sorted(set(policy_rules.keys()) - _ALLOWED_POLICY_KEYS)
    if unknown:
        raise ValueError(f"unsupported policy_rules keys: {unknown}")

    normalized: dict[str, Any] = {}

    if "max_depth" in policy_rules:
        depth = int(policy_rules["max_depth"])
        if depth < 1:
            raise ValueError("policy_rules.max_depth must be >= 1")
        normalized["max_depth"] = depth

    if "forbidden_nodes" in policy_rules:
        raw = policy_rules["forbidden_nodes"]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("policy_rules.forbidden_nodes must be a sequence")
        normalized["forbidden_nodes"] = tuple(sorted({_norm_token(str(x), name="forbidden node") for x in raw}))

    if "forbidden_transitions" in policy_rules:
        raw = policy_rules["forbidden_transitions"]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("policy_rules.forbidden_transitions must be a sequence")
        transitions: set[tuple[str, str]] = set()
        for item in raw:
            if isinstance(item, (str, bytes, bytearray)) or not isinstance(item, Sequence) or len(item) != 2:
                raise ValueError("policy_rules.forbidden_transitions entries must be 2-item sequences")
            frm = _norm_token(str(item[0]), name="forbidden transition from")
            to = _norm_token(str(item[1]), name="forbidden transition to")
            transitions.add((frm, to))
        normalized["forbidden_transitions"] = tuple(sorted(transitions))

    if "required_terminal_subset" in policy_rules:
        raw = policy_rules["required_terminal_subset"]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("policy_rules.required_terminal_subset must be a sequence")
        required = tuple(sorted({_norm_token(str(x), name="required terminal node") for x in raw}))
        if not required:
            raise ValueError("policy_rules.required_terminal_subset must not be empty")
        normalized["required_terminal_subset"] = required

    if "route_cost_ceiling" in policy_rules:
        ceiling = float(policy_rules["route_cost_ceiling"])
        if not math.isfinite(ceiling):
            raise ValueError("policy_rules.route_cost_ceiling must be finite")
        if ceiling < 0.0:
            raise ValueError("policy_rules.route_cost_ceiling must be >= 0")
        normalized["route_cost_ceiling"] = _round64(ceiling)

    if "must_pass_through_nodes" in policy_rules:
        raw = policy_rules["must_pass_through_nodes"]
        if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
            raise ValueError("policy_rules.must_pass_through_nodes must be a sequence")
        required = tuple(sorted({_norm_token(str(x), name="must-pass-through node") for x in raw}))
        if not required:
            raise ValueError("policy_rules.must_pass_through_nodes must not be empty")
        normalized["must_pass_through_nodes"] = required

    if "node_costs" in policy_rules:
        raw_costs = policy_rules["node_costs"]
        if not isinstance(raw_costs, Mapping):
            raise ValueError("policy_rules.node_costs must be a mapping")
        costs: dict[str, float] = {}
        for raw_node, raw_cost in raw_costs.items():
            node = _norm_token(str(raw_node), name="node_cost key")
            cost = float(raw_cost)
            if not math.isfinite(cost):
                raise ValueError("policy_rules.node_costs values must be finite")
            if cost < 0.0:
                raise ValueError("policy_rules.node_costs values must be >= 0")
            costs[node] = _round64(cost)
        normalized["node_costs"] = {k: costs[k] for k in sorted(costs)}

    if "node_costs" in normalized and "route_cost_ceiling" not in normalized:
        raise ValueError("policy_rules.node_costs requires policy_rules.route_cost_ceiling")

    return normalized


def compute_policy_pressure(total_candidates_examined: int, rejected_candidates: int) -> float:
    examined = int(total_candidates_examined)
    rejected = int(rejected_candidates)
    if examined < 0:
        raise ValueError("total_candidates_examined must be >= 0")
    if rejected < 0:
        raise ValueError("rejected_candidates must be >= 0")
    if rejected > examined:
        raise ValueError("rejected_candidates must be <= total_candidates_examined")
    if examined == 0:
        return 0.0
    return _round64(float(rejected) / float(examined))


@dataclass(frozen=True)
class CandidatePolicyDecision:
    candidate_node: str
    status: str
    admissible: bool
    violated_rules: tuple[str, ...]
    warning_pressure: float
    reachable_terminal_nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_node": self.candidate_node,
            "status": self.status,
            "admissible": bool(self.admissible),
            "violated_rules": list(self.violated_rules),
            "warning_pressure": _round64(self.warning_pressure),
            "reachable_terminal_nodes": list(self.reachable_terminal_nodes),
        }


@dataclass(frozen=True)
class PolicyDecisionArtifact:
    schema_version: str
    source_plan_hash: str
    current_path: tuple[str, ...]
    route_graph_hash: str
    examined_candidates: tuple[str, ...]
    admitted_candidates: tuple[str, ...]
    rejected_candidates: tuple[str, ...]
    constrained_candidates: tuple[str, ...]
    violated_rules: tuple[str, ...]
    candidate_decisions: tuple[CandidatePolicyDecision, ...]
    policy_rules: Mapping[str, Any]
    policy_pressure_score: float
    policy_identity_chain: tuple[str, ...]
    stable_policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "current_path": list(self.current_path),
            "route_graph_hash": self.route_graph_hash,
            "examined_candidates": list(self.examined_candidates),
            "admitted_candidates": list(self.admitted_candidates),
            "rejected_candidates": list(self.rejected_candidates),
            "constrained_candidates": list(self.constrained_candidates),
            "violated_rules": list(self.violated_rules),
            "candidate_decisions": [decision.to_dict() for decision in self.candidate_decisions],
            "policy_rules": self.policy_rules,
            "policy_pressure_score": _round64(self.policy_pressure_score),
            "policy_identity_chain": list(self.policy_identity_chain),
            "stable_policy_hash": self.stable_policy_hash,
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class PolicyDecisionReceipt:
    schema_version: str
    source_plan_hash: str
    route_graph_hash: str
    examined_candidates: int
    admitted_candidates: int
    rejected_candidates: int
    constrained_candidates: int
    policy_pressure_score: float
    stable_policy_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "route_graph_hash": self.route_graph_hash,
            "examined_candidates": int(self.examined_candidates),
            "admitted_candidates": int(self.admitted_candidates),
            "rejected_candidates": int(self.rejected_candidates),
            "constrained_candidates": int(self.constrained_candidates),
            "policy_pressure_score": _round64(self.policy_pressure_score),
            "stable_policy_hash": self.stable_policy_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def analyze_policy_constrained_frontier(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    current_path: Sequence[str],
    frontier_candidates: Sequence[str],
    max_path_length: int,
    policy_rules: Mapping[str, Any],
    enable_v137_6_3_policy_constraints: bool = False,
) -> PolicyDecisionArtifact:
    if not enable_v137_6_3_policy_constraints:
        raise ValueError("enable_v137_6_3_policy_constraints must be True to enable v137.6.3 policy constraints")

    normalized_source_plan_hash = _validate_hash(source_plan_hash, name="source_plan_hash")
    normalized_graph = _normalize_route_graph(route_graph)
    normalized_path = _normalize_path(current_path)
    if normalized_path[-1] not in normalized_graph:
        raise ValueError("current_path terminal node must exist in route_graph")

    normalized_frontier = _normalize_frontier(frontier_candidates)
    bounded_max_path_length = _validate_max_path_length(max_path_length)
    if len(normalized_path) > bounded_max_path_length:
        raise ValueError("current_path length must be <= max_path_length")

    normalized_policy_rules = _normalize_policy_rules(policy_rules)
    route_graph_hash = _sha256_hex_mapping({"route_graph": {k: list(v) for k, v in sorted(normalized_graph.items())}})

    forbidden_nodes = frozenset(normalized_policy_rules.get("forbidden_nodes", tuple()))
    forbidden_transitions = frozenset(normalized_policy_rules.get("forbidden_transitions", tuple()))
    required_terminals = frozenset(normalized_policy_rules.get("required_terminal_subset", tuple()))
    required_nodes = frozenset(normalized_policy_rules.get("must_pass_through_nodes", tuple()))
    max_depth = normalized_policy_rules.get("max_depth")
    route_cost_ceiling = normalized_policy_rules.get("route_cost_ceiling")
    node_costs = normalized_policy_rules.get("node_costs", {})

    prior_node = normalized_path[-1]
    remaining_after_step = bounded_max_path_length - (len(normalized_path) + 1)
    identity_chain: list[str] = [GENESIS_HASH]
    decisions: list[CandidatePolicyDecision] = []
    admitted: list[str] = []
    rejected: list[str] = []
    constrained: list[str] = []
    violated_rule_set: set[str] = set()

    for idx, candidate in enumerate(normalized_frontier):
        tentative_path = normalized_path + (candidate,)
        candidate_violations: list[str] = []
        candidate_constrained = False

        if max_depth is not None and len(tentative_path) > int(max_depth):
            candidate_violations.append("max_depth")
        if candidate in forbidden_nodes:
            candidate_violations.append("forbidden_nodes")
        if (prior_node, candidate) in forbidden_transitions:
            candidate_violations.append("forbidden_transitions")

        reachable_terminals = _collect_reachable_terminals(candidate, normalized_graph, max_steps=remaining_after_step)
        if required_terminals and reachable_terminals.isdisjoint(required_terminals):
            candidate_violations.append("required_terminal_subset")

        if route_cost_ceiling is not None:
            route_cost = 0.0
            for node in tentative_path:
                route_cost += float(node_costs.get(node, 1.0))
            route_cost = _round64(route_cost)
            if route_cost > float(route_cost_ceiling):
                candidate_violations.append("route_cost_ceiling")

        if required_nodes:
            covered_nodes = frozenset(tentative_path)
            missing_required = tuple(sorted(required_nodes - covered_nodes))
            if missing_required:
                reachable_nodes = _collect_reachable(candidate, normalized_graph, max_steps=remaining_after_step)
                if any(node not in reachable_nodes for node in missing_required):
                    candidate_violations.append("must_pass_through_nodes")
                else:
                    candidate_constrained = True

        if candidate_violations:
            status = "rejected"
            rejected.append(candidate)
            violated_rule_set.update(candidate_violations)
        elif candidate_constrained:
            status = "constrained"
            admitted.append(candidate)
            constrained.append(candidate)
        else:
            status = "admissible"
            admitted.append(candidate)

        step_hash = _sha256_hex_mapping(
            {
                "prior_hash": identity_chain[-1],
                "candidate_index": idx,
                "candidate_node": candidate,
                "candidate_status": status,
                "violated_rules": sorted(set(candidate_violations)),
                "source_plan_hash": normalized_source_plan_hash,
                "route_graph_hash": route_graph_hash,
            }
        )
        identity_chain.append(step_hash)

        decisions.append(
            CandidatePolicyDecision(
                candidate_node=candidate,
                status=status,
                admissible=status != "rejected",
                violated_rules=tuple(sorted(set(candidate_violations))),
                warning_pressure=0.0,
                reachable_terminal_nodes=tuple(sorted(reachable_terminals)),
            )
        )

    pressure = compute_policy_pressure(len(normalized_frontier), len(rejected))
    decisions_with_pressure = tuple(
        CandidatePolicyDecision(
            candidate_node=decision.candidate_node,
            status=decision.status,
            admissible=decision.admissible,
            violated_rules=decision.violated_rules,
            warning_pressure=(pressure if decision.admissible and pressure > 0.0 else 0.0),
            reachable_terminal_nodes=decision.reachable_terminal_nodes,
        )
        for decision in decisions
    )

    artifact_payload = {
        "schema_version": POLICY_CONSTRAINED_PLANNER_VERSION,
        "source_plan_hash": normalized_source_plan_hash,
        "current_path": list(normalized_path),
        "route_graph_hash": route_graph_hash,
        "examined_candidates": list(normalized_frontier),
        "admitted_candidates": list(admitted),
        "rejected_candidates": list(rejected),
        "constrained_candidates": list(constrained),
        "violated_rules": sorted(violated_rule_set),
        "candidate_decisions": [decision.to_dict() for decision in decisions_with_pressure],
        "policy_rules": normalized_policy_rules,
        "policy_pressure_score": pressure,
        "policy_identity_chain": list(identity_chain),
    }
    stable_policy_hash = _sha256_hex_mapping(artifact_payload)

    return PolicyDecisionArtifact(
        schema_version=POLICY_CONSTRAINED_PLANNER_VERSION,
        source_plan_hash=normalized_source_plan_hash,
        current_path=normalized_path,
        route_graph_hash=route_graph_hash,
        examined_candidates=normalized_frontier,
        admitted_candidates=tuple(admitted),
        rejected_candidates=tuple(rejected),
        constrained_candidates=tuple(constrained),
        violated_rules=tuple(sorted(violated_rule_set)),
        candidate_decisions=decisions_with_pressure,
        policy_rules=normalized_policy_rules,
        policy_pressure_score=pressure,
        policy_identity_chain=tuple(identity_chain),
        stable_policy_hash=stable_policy_hash,
    )


def admit_policy_constrained_frontier(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    current_path: Sequence[str],
    frontier_candidates: Sequence[str],
    max_path_length: int,
    policy_rules: Mapping[str, Any],
    enable_v137_6_3_policy_constraints: bool = False,
) -> tuple[tuple[str, ...], PolicyDecisionArtifact]:
    artifact = analyze_policy_constrained_frontier(
        source_plan_hash,
        route_graph,
        current_path=current_path,
        frontier_candidates=frontier_candidates,
        max_path_length=max_path_length,
        policy_rules=policy_rules,
        enable_v137_6_3_policy_constraints=enable_v137_6_3_policy_constraints,
    )
    return artifact.admitted_candidates, artifact


def generate_policy_decision_receipt(artifact: PolicyDecisionArtifact) -> PolicyDecisionReceipt:
    if not isinstance(artifact, PolicyDecisionArtifact):
        raise TypeError("artifact must be a PolicyDecisionArtifact instance")

    receipt_payload = {
        "schema_version": artifact.schema_version,
        "source_plan_hash": artifact.source_plan_hash,
        "route_graph_hash": artifact.route_graph_hash,
        "examined_candidates": len(artifact.examined_candidates),
        "admitted_candidates": len(artifact.admitted_candidates),
        "rejected_candidates": len(artifact.rejected_candidates),
        "constrained_candidates": len(artifact.constrained_candidates),
        "policy_pressure_score": _round64(artifact.policy_pressure_score),
        "stable_policy_hash": artifact.stable_policy_hash,
    }
    receipt_hash = _sha256_hex_mapping(receipt_payload)

    return PolicyDecisionReceipt(
        schema_version=artifact.schema_version,
        source_plan_hash=artifact.source_plan_hash,
        route_graph_hash=artifact.route_graph_hash,
        examined_candidates=len(artifact.examined_candidates),
        admitted_candidates=len(artifact.admitted_candidates),
        rejected_candidates=len(artifact.rejected_candidates),
        constrained_candidates=len(artifact.constrained_candidates),
        policy_pressure_score=artifact.policy_pressure_score,
        stable_policy_hash=artifact.stable_policy_hash,
        receipt_hash=receipt_hash,
    )


def export_policy_decision_bytes(artifact: PolicyDecisionArtifact) -> bytes:
    if not isinstance(artifact, PolicyDecisionArtifact):
        raise TypeError("artifact must be a PolicyDecisionArtifact instance")
    return artifact.to_canonical_bytes()


__all__ = [
    "POLICY_CONSTRAINED_PLANNER_VERSION",
    "GENESIS_HASH",
    "CandidatePolicyDecision",
    "PolicyDecisionArtifact",
    "PolicyDecisionReceipt",
    "analyze_policy_constrained_frontier",
    "admit_policy_constrained_frontier",
    "compute_policy_pressure",
    "generate_policy_decision_receipt",
    "export_policy_decision_bytes",
]
