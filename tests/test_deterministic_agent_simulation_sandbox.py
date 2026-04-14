from __future__ import annotations

import json

from qec.orchestration.deterministic_agent_simulation_sandbox import (
    AgentSimulationReceipt,
    AgentSimulationScenario,
    AgentSimulationStep,
    build_agent_simulation_receipt,
    build_agent_simulation_scenario,
    compare_agent_simulation_replay,
    run_deterministic_agent_simulation,
    summarize_agent_simulation,
    validate_agent_simulation,
)
from qec.orchestration.deterministic_covenant_engine import build_covenant_rule
from qec.orchestration.policy_execution_firewall_kernel import build_firewall_policy_rule
from qec.orchestration.proof_carrying_agent_action_capsule import (
    AgentActionProofObligation,
    build_proof_carrying_agent_action_capsule,
)


def _capsule() -> object:
    return build_proof_carrying_agent_action_capsule(
        action_id="a1",
        action_type="validate",
        action_scope="validate",
        action_payload={"k": 1},
        invariants=("shape:stable",),
        proof_obligations=(
            AgentActionProofObligation(
                obligation_id="o1",
                obligation_kind="determinism",
                obligation_statement="same input same bytes",
                obligation_scope="simulation",
                obligation_epoch=0,
            ),
        ),
        validation_flags=("deterministic_only", "replay_safe"),
    )


def _allow_scenario(steps: int = 3) -> AgentSimulationScenario:
    capsule = _capsule()
    rule = build_covenant_rule(
        rule_id="r1",
        target_key="counter",
        delta=1.0,
        min_value=0.0,
        max_value=100.0,
        replay_identity=capsule.replay_identity,
    )
    policy = build_firewall_policy_rule(
        rule_id="p1",
        allowed_action_types=("validate",),
        allowed_action_scopes=("validate",),
        require_within_boundary=False,
        max_allowed_severity="critical",
        require_receipt_continuity=False,
        required_replay_identity_prefix="agent-action::",
        required_covenant_rule_id="r1",
    )
    return build_agent_simulation_scenario(
        action_capsule=capsule,
        covenant_rule=rule,
        sandbox_policy_rules=(policy,),
        initial_state={"counter": 0.0},
        simulation_steps=steps,
        io_surface="none",
    )


def _deny_scenario() -> AgentSimulationScenario:
    scenario = _allow_scenario(steps=2)
    policy = build_firewall_policy_rule(
        rule_id="p-deny",
        allowed_action_types=("certify",),
        allowed_action_scopes=("validate",),
        require_within_boundary=False,
        max_allowed_severity="critical",
        require_receipt_continuity=False,
        required_replay_identity_prefix="agent-action::",
        required_covenant_rule_id="r1",
    )
    return build_agent_simulation_scenario(
        action_capsule=scenario.action_capsule,
        covenant_rule=scenario.covenant_rule,
        sandbox_policy_rules=(policy,),
        initial_state={"counter": 0.0},
        simulation_steps=2,
        io_surface="none",
    )


def test_deterministic_repeated_simulation():
    scenario = _allow_scenario()
    a = run_deterministic_agent_simulation(scenario)
    b = run_deterministic_agent_simulation(scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_stable_hash_reproducibility_for_frozen_dataclasses():
    scenario = _allow_scenario(steps=1)
    run = run_deterministic_agent_simulation(scenario)
    step = run.simulated_execution_trace[0]
    receipt = run.sandbox_receipt

    assert scenario.stable_hash() == scenario.stable_hash()
    assert step.stable_hash() == step.stable_hash()
    assert receipt.stable_hash() == receipt.stable_hash()
    assert run.stable_hash() == run.stable_hash()


def test_allow_path_simulation():
    run = run_deterministic_agent_simulation(_allow_scenario(steps=2))
    assert run.validation_violations == ()
    assert all(step.allow for step in run.simulated_execution_trace)
    assert run.sandbox_receipt.deny_count == 0


def test_deny_path_simulation():
    run = run_deterministic_agent_simulation(_deny_scenario())
    assert any(not step.allow for step in run.simulated_execution_trace)
    assert run.sandbox_receipt.deny_count == len(run.simulated_execution_trace)


def test_malformed_input_handling_is_deterministic():
    violations = validate_agent_simulation({"bad": "shape"})
    assert violations == ("malformed_scenario:type",)


def test_validator_never_raises_for_weird_object():
    class Weird:
        pass

    violations = validate_agent_simulation(Weird())
    assert isinstance(violations, tuple)
    assert "malformed_scenario:type" in violations


def test_replay_comparison_stability():
    scenario = _allow_scenario(steps=2)
    a = run_deterministic_agent_simulation(scenario)
    b = run_deterministic_agent_simulation(scenario)
    cmp_report = compare_agent_simulation_replay(a, b)
    assert cmp_report["match"] is True
    assert cmp_report["mismatch_fields"] == ()


def test_canonical_json_round_trip():
    scenario = _allow_scenario(steps=1)
    run = run_deterministic_agent_simulation(scenario)
    data = json.loads(run.to_canonical_json())
    assert data["sandbox_receipt"]["receipt_hash"] == run.sandbox_receipt.receipt_hash


def test_deterministic_summary_ordering():
    run = run_deterministic_agent_simulation(_allow_scenario(steps=3))
    summary = summarize_agent_simulation(run)
    indices = [row["step_index"] for row in summary["steps"]]
    assert indices == sorted(indices)


def test_continuity_trace_reproducibility():
    scenario = _allow_scenario(steps=3)
    a = run_deterministic_agent_simulation(scenario)
    b = run_deterministic_agent_simulation(scenario)
    assert [s.continuity_hash for s in a.simulated_execution_trace] == [
        s.continuity_hash for s in b.simulated_execution_trace
    ]


def test_step_ordering_stability():
    run = run_deterministic_agent_simulation(_allow_scenario(steps=4))
    assert tuple(step.step_index for step in run.simulated_execution_trace) == (0, 1, 2, 3)


def test_simulated_receipt_reproducibility():
    scenario = _allow_scenario(steps=2)
    run = run_deterministic_agent_simulation(scenario)
    receipt_a = build_agent_simulation_receipt(
        scenario=run.scenario,
        simulated_execution_trace=run.simulated_execution_trace,
        validation_violations=run.validation_violations,
    )
    receipt_b = build_agent_simulation_receipt(
        scenario=run.scenario,
        simulated_execution_trace=run.simulated_execution_trace,
        validation_violations=run.validation_violations,
    )
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_no_input_mutation_of_initial_state():
    scenario = _allow_scenario(steps=1)
    original = {"counter": 0.0}
    scenario2 = build_agent_simulation_scenario(
        action_capsule=scenario.action_capsule,
        covenant_rule=scenario.covenant_rule,
        sandbox_policy_rules=scenario.sandbox_policy_rules,
        initial_state=original,
        simulation_steps=1,
        io_surface="none",
    )
    _ = run_deterministic_agent_simulation(scenario2)
    assert original == {"counter": 0.0}


def test_malformed_scenario_returns_empty_trace_with_violations():
    scenario = _allow_scenario(steps=1)
    broken = AgentSimulationScenario(
        action_capsule=scenario.action_capsule,
        covenant_rule=scenario.covenant_rule,
        sandbox_policy_rules=scenario.sandbox_policy_rules,
        initial_state=scenario.initial_state,
        simulation_steps=1,
        io_surface="external",
        prior_transition_receipt_hash="",
    )
    run = run_deterministic_agent_simulation(broken)
    assert run.simulated_execution_trace == ()
    assert "malformed_scenario:io_surface" in run.validation_violations


def test_dataclass_canonical_json_and_to_dict_shape():
    scenario = _allow_scenario(steps=1)
    run = run_deterministic_agent_simulation(scenario)
    step = run.simulated_execution_trace[0]
    receipt = run.sandbox_receipt

    assert isinstance(step, AgentSimulationStep)
    assert isinstance(receipt, AgentSimulationReceipt)
    assert "step_index" in step.to_dict()
    assert "receipt_hash" in receipt.to_dict()
    assert "scenario" in run.to_dict()


def test_boundary_continuity_ok_for_valid_replay_chain():
    """Regression: boundary auditor receives correct mapping shapes so
    continuity_ok is True for steps 2+ in a valid replay chain."""
    scenario = _allow_scenario(steps=3)
    run = run_deterministic_agent_simulation(scenario)
    assert run.validation_violations == ()
    # Steps 1 and 2 (index 1, 2) have a real prior receipt hash —
    # the boundary audit must confirm continuity_ok=True for them.
    for step in run.simulated_execution_trace[1:]:
        audit_receipt = step.simulated_boundary_result.get("audit_receipt", {})
        assert audit_receipt.get("continuity_ok") is True, (
            f"step {step.step_index}: expected continuity_ok=True, got {audit_receipt}"
        )


def test_malformed_action_capsule_type_does_not_crash():
    """Regression: malformed AgentSimulationScenario with wrong action_capsule
    type must not raise during receipt construction; returns empty trace + violations."""
    scenario = _allow_scenario(steps=1)
    # Bypass type enforcement by constructing dataclass directly with wrong type.
    broken = AgentSimulationScenario(
        action_capsule="not_a_capsule",  # type: ignore[arg-type]
        covenant_rule=scenario.covenant_rule,
        sandbox_policy_rules=scenario.sandbox_policy_rules,
        initial_state=scenario.initial_state,
        simulation_steps=1,
        io_surface="none",
        prior_transition_receipt_hash="",
    )
    run = run_deterministic_agent_simulation(broken)
    assert run.simulated_execution_trace == ()
    assert "malformed_scenario:action_capsule" in run.validation_violations
    # Receipt hash must be deterministic even for this malformed case.
    assert isinstance(run.sandbox_receipt.receipt_hash, str)
    assert len(run.sandbox_receipt.receipt_hash) == 64
    assert run.sandbox_receipt.scenario_hash == "malformed_scenario"
