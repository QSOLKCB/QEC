"""v137.18.5 — Deterministic Agent Simulation Sandbox.

Deterministic, non-mutating dry-run sandbox that simulates governance stack
execution without performing external actions or persisting state.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from typing import Any, Dict, Mapping, Sequence, Tuple

from qec.orchestration.deterministic_covenant_engine import (
    CovenantRule,
    build_covenant_rule,
    execute_covenant_transition,
)
from qec.orchestration.governance_memory_io_boundary_auditor import (
    GovernanceMemoryIOBoundaryAuditReport,
    audit_governance_memory_io_boundaries,
)
from qec.orchestration.policy_execution_firewall_kernel import (
    FirewallPolicyRule,
    build_firewall_policy_rule,
    evaluate_policy_execution_firewall,
)
from qec.orchestration.proof_carrying_agent_action_capsule import (
    ProofCarryingAgentActionCapsule,
    build_proof_carrying_agent_action_capsule,
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


def _state_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    ordered: Dict[str, Any] = {}
    for key in sorted(value.keys(), key=lambda item: str(item)):
        if isinstance(key, str) and key:
            ordered[key] = _deep_canonical_copy(value[key])
    return ordered


@dataclass(frozen=True)
class AgentSimulationScenario:
    action_capsule: ProofCarryingAgentActionCapsule
    covenant_rule: CovenantRule
    sandbox_policy_rules: Tuple[FirewallPolicyRule, ...]
    initial_state: Dict[str, Any]
    simulation_steps: int
    io_surface: str
    prior_transition_receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_capsule": self.action_capsule.to_dict(),
            "covenant_rule": self.covenant_rule.to_dict(),
            "sandbox_policy_rules": [rule.to_dict() for rule in self.sandbox_policy_rules],
            "initial_state": _deep_canonical_copy(self.initial_state),
            "simulation_steps": self.simulation_steps,
            "io_surface": self.io_surface,
            "prior_transition_receipt_hash": self.prior_transition_receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class AgentSimulationStep:
    step_index: int
    simulated_state: Dict[str, Any]
    simulated_covenant_receipt: Dict[str, Any]
    simulated_boundary_result: Dict[str, Any]
    simulated_firewall_decision: Dict[str, Any]
    allow: bool
    rule_hit_reasons: Tuple[str, ...]
    step_replay_identity: str
    continuity_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "simulated_state": _deep_canonical_copy(self.simulated_state),
            "simulated_covenant_receipt": _deep_canonical_copy(self.simulated_covenant_receipt),
            "simulated_boundary_result": _deep_canonical_copy(self.simulated_boundary_result),
            "simulated_firewall_decision": _deep_canonical_copy(self.simulated_firewall_decision),
            "allow": self.allow,
            "rule_hit_reasons": list(self.rule_hit_reasons),
            "step_replay_identity": self.step_replay_identity,
            "continuity_hash": self.continuity_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


@dataclass(frozen=True)
class AgentSimulationReceipt:
    scenario_hash: str
    step_hashes: Tuple[str, ...]
    allow_count: int
    deny_count: int
    validation_violations: Tuple[str, ...]
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_hash": self.scenario_hash,
            "step_hashes": list(self.step_hashes),
            "allow_count": self.allow_count,
            "deny_count": self.deny_count,
            "validation_violations": list(self.validation_violations),
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.receipt_hash


@dataclass(frozen=True)
class DeterministicAgentSimulationSandbox:
    scenario: AgentSimulationScenario
    simulated_execution_trace: Tuple[AgentSimulationStep, ...]
    sandbox_receipt: AgentSimulationReceipt
    validation_violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "simulated_execution_trace": [step.to_dict() for step in self.simulated_execution_trace],
            "sandbox_receipt": self.sandbox_receipt.to_dict(),
            "validation_violations": list(self.validation_violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_json().encode("utf-8"))


def build_agent_simulation_scenario(
    *,
    action_capsule: Any,
    covenant_rule: Any,
    sandbox_policy_rules: Sequence[Any],
    initial_state: Any,
    simulation_steps: int = 1,
    io_surface: str = "none",
    prior_transition_receipt_hash: str = "",
) -> AgentSimulationScenario:
    capsule = (
        action_capsule
        if isinstance(action_capsule, ProofCarryingAgentActionCapsule)
        else build_proof_carrying_agent_action_capsule(**dict(action_capsule))
    )
    rule = (
        covenant_rule
        if isinstance(covenant_rule, CovenantRule)
        else build_covenant_rule(**dict(covenant_rule))
    )
    policies: Tuple[FirewallPolicyRule, ...] = tuple(
        sorted(
            (
                policy if isinstance(policy, FirewallPolicyRule) else build_firewall_policy_rule(**dict(policy))
                for policy in sandbox_policy_rules
            ),
            key=lambda item: (item.rule_id, item.stable_hash()),
        )
    )

    steps = int(simulation_steps)
    normalized_steps = steps if steps >= 0 else 0
    return AgentSimulationScenario(
        action_capsule=capsule,
        covenant_rule=rule,
        sandbox_policy_rules=policies,
        initial_state=_state_mapping(initial_state),
        simulation_steps=normalized_steps,
        io_surface=str(io_surface),
        prior_transition_receipt_hash=str(prior_transition_receipt_hash),
    )


def validate_agent_simulation(scenario: Any) -> Tuple[str, ...]:
    violations: list[str] = []
    try:
        if not isinstance(scenario, AgentSimulationScenario):
            return ("malformed_scenario:type",)

        if scenario.simulation_steps < 0:
            violations.append("malformed_scenario:simulation_steps_negative")
        if scenario.io_surface not in ("none", "internal"):
            violations.append("malformed_scenario:io_surface")
        if not isinstance(scenario.initial_state, Mapping):
            violations.append("malformed_scenario:initial_state")
        if not isinstance(scenario.sandbox_policy_rules, tuple):
            violations.append("malformed_scenario:sandbox_policy_rules")

        if not isinstance(scenario.action_capsule, ProofCarryingAgentActionCapsule):
            violations.append("malformed_scenario:action_capsule")
        if not isinstance(scenario.covenant_rule, CovenantRule):
            violations.append("malformed_scenario:covenant_rule")
        for idx, rule in enumerate(scenario.sandbox_policy_rules):
            if not isinstance(rule, FirewallPolicyRule):
                violations.append(f"malformed_scenario:sandbox_policy_rule:{idx}")

    except Exception as exc:  # pragma: no cover
        violations.append(f"validator_error:{type(exc).__name__}")

    return tuple(sorted(set(violations)))


def run_deterministic_agent_simulation(
    scenario: AgentSimulationScenario,
) -> DeterministicAgentSimulationSandbox:
    violations = validate_agent_simulation(scenario)
    if violations:
        empty_trace: Tuple[AgentSimulationStep, ...] = ()
        receipt = build_agent_simulation_receipt(
            scenario=scenario,
            simulated_execution_trace=empty_trace,
            validation_violations=violations,
        )
        return DeterministicAgentSimulationSandbox(
            scenario=scenario,
            simulated_execution_trace=empty_trace,
            sandbox_receipt=receipt,
            validation_violations=violations,
        )

    state_t = copy.deepcopy(scenario.initial_state)
    prior_transition_hash = scenario.prior_transition_receipt_hash
    prior_continuity_hash = _sha256_hex(scenario.stable_hash().encode("utf-8"))
    trace: list[AgentSimulationStep] = []

    for step_index in range(scenario.simulation_steps):
        covenant_execution = execute_covenant_transition(
            state_t=state_t,
            action_capsule=scenario.action_capsule.to_dict(),
            covenant_rule=scenario.covenant_rule,
        )

        current_receipt_dict = {
            **covenant_execution.receipt.to_dict(),
            "prior_receipt_hash": prior_transition_hash,
        }
        prior_receipt_dict = {"receipt_hash": prior_transition_hash}
        boundary_report: GovernanceMemoryIOBoundaryAuditReport = audit_governance_memory_io_boundaries(
            covenant_metadata={
                "covenant_id": scenario.covenant_rule.rule_id,
                "capsule_id": scenario.action_capsule.action_id,
            },
            payload=scenario.action_capsule.action_payload,
            state=covenant_execution.next_state.state_data,
            proof_chain=tuple(ob.stable_hash() for ob in scenario.action_capsule.proof_obligations),
            action_scope=scenario.action_capsule.action_scope,
            io_surface=scenario.io_surface,
            replay_identity=scenario.action_capsule.replay_identity,
            declared_replay_identity=scenario.action_capsule.replay_identity,
            transition_receipt=current_receipt_dict,
            prior_transition_receipt=prior_receipt_dict,
            boundary_rules=None,
        )

        firewall_decision = evaluate_policy_execution_firewall(
            action_capsule=scenario.action_capsule,
            covenant_execution=covenant_execution,
            boundary_audit_report=boundary_report,
            policy_rules=scenario.sandbox_policy_rules,
        )

        allow = firewall_decision.decision_value == "allow"
        reasons = _stable_str_tuple(reason.reason_code for reason in firewall_decision.reasons)
        step_identity_payload = {
            "prior_continuity_hash": prior_continuity_hash,
            "step_index": step_index,
            "state_hash": covenant_execution.next_state.state_hash,
            "transition_receipt_hash": covenant_execution.receipt.receipt_hash,
            "boundary_report_hash": boundary_report.report_hash,
            "firewall_decision_hash": firewall_decision.decision_hash,
        }
        step_replay_identity = _sha256_hex(_canonical_json(step_identity_payload).encode("utf-8"))
        continuity_hash = _sha256_hex(
            _canonical_json(
                {
                    "allow": allow,
                    "prior_continuity_hash": prior_continuity_hash,
                    "step_replay_identity": step_replay_identity,
                }
            ).encode("utf-8")
        )

        step = AgentSimulationStep(
            step_index=step_index,
            simulated_state=_deep_canonical_copy(covenant_execution.next_state.state_data),
            simulated_covenant_receipt=_deep_canonical_copy(covenant_execution.receipt.to_dict()),
            simulated_boundary_result=_deep_canonical_copy(boundary_report.to_dict()),
            simulated_firewall_decision=_deep_canonical_copy(firewall_decision.to_dict()),
            allow=allow,
            rule_hit_reasons=reasons,
            step_replay_identity=step_replay_identity,
            continuity_hash=continuity_hash,
        )
        trace.append(step)

        state_t = _state_mapping(covenant_execution.next_state.state_data)
        prior_transition_hash = covenant_execution.receipt.receipt_hash
        prior_continuity_hash = continuity_hash

    simulated_trace = tuple(sorted(trace, key=lambda item: item.step_index))
    receipt = build_agent_simulation_receipt(
        scenario=scenario,
        simulated_execution_trace=simulated_trace,
        validation_violations=violations,
    )
    return DeterministicAgentSimulationSandbox(
        scenario=scenario,
        simulated_execution_trace=simulated_trace,
        sandbox_receipt=receipt,
        validation_violations=violations,
    )


def build_agent_simulation_receipt(
    *,
    scenario: AgentSimulationScenario,
    simulated_execution_trace: Sequence[AgentSimulationStep],
    validation_violations: Sequence[str] = (),
) -> AgentSimulationReceipt:
    try:
        scenario_hash = scenario.stable_hash()
    except (AttributeError, TypeError):
        scenario_hash = "malformed_scenario"
    step_hashes = tuple(step.stable_hash() for step in simulated_execution_trace)
    allow_count = sum(1 for step in simulated_execution_trace if step.allow)
    deny_count = sum(1 for step in simulated_execution_trace if not step.allow)
    ordered_violations = tuple(sorted({str(v) for v in validation_violations if str(v)}))
    preimage = {
        "scenario_hash": scenario_hash,
        "step_hashes": list(step_hashes),
        "allow_count": allow_count,
        "deny_count": deny_count,
        "validation_violations": list(ordered_violations),
    }
    receipt_hash = _sha256_hex(_canonical_json(preimage).encode("utf-8"))
    return AgentSimulationReceipt(
        scenario_hash=scenario_hash,
        step_hashes=step_hashes,
        allow_count=allow_count,
        deny_count=deny_count,
        validation_violations=ordered_violations,
        receipt_hash=receipt_hash,
    )


def compare_agent_simulation_replay(
    baseline: DeterministicAgentSimulationSandbox,
    replay: DeterministicAgentSimulationSandbox,
) -> Dict[str, Any]:
    mismatches: list[str] = []
    if baseline.scenario.stable_hash() != replay.scenario.stable_hash():
        mismatches.append("scenario")
    if baseline.sandbox_receipt.receipt_hash != replay.sandbox_receipt.receipt_hash:
        mismatches.append("sandbox_receipt")

    baseline_step_hashes = tuple(step.stable_hash() for step in baseline.simulated_execution_trace)
    replay_step_hashes = tuple(step.stable_hash() for step in replay.simulated_execution_trace)
    if baseline_step_hashes != replay_step_hashes:
        mismatches.append("simulated_execution_trace")

    if baseline.validation_violations != replay.validation_violations:
        mismatches.append("validation_violations")

    return {
        "match": len(mismatches) == 0,
        "mismatch_fields": tuple(mismatches),
        "baseline_hash": baseline.stable_hash(),
        "replay_hash": replay.stable_hash(),
    }


def summarize_agent_simulation(
    sandbox: DeterministicAgentSimulationSandbox,
) -> Dict[str, Any]:
    ordered_steps = tuple(sorted(sandbox.simulated_execution_trace, key=lambda item: item.step_index))
    return {
        "scenario_hash": sandbox.scenario.stable_hash(),
        "sandbox_hash": sandbox.stable_hash(),
        "receipt_hash": sandbox.sandbox_receipt.receipt_hash,
        "validation_violations": list(sandbox.validation_violations),
        "step_count": len(ordered_steps),
        "allow_count": sandbox.sandbox_receipt.allow_count,
        "deny_count": sandbox.sandbox_receipt.deny_count,
        "steps": [
            {
                "step_index": step.step_index,
                "allow": step.allow,
                "rule_hit_reasons": list(step.rule_hit_reasons),
                "step_replay_identity": step.step_replay_identity,
                "continuity_hash": step.continuity_hash,
                "step_hash": step.stable_hash(),
            }
            for step in ordered_steps
        ],
    }


__all__ = (
    "AgentSimulationScenario",
    "AgentSimulationStep",
    "AgentSimulationReceipt",
    "DeterministicAgentSimulationSandbox",
    "build_agent_simulation_scenario",
    "run_deterministic_agent_simulation",
    "validate_agent_simulation",
    "build_agent_simulation_receipt",
    "compare_agent_simulation_replay",
    "summarize_agent_simulation",
)
