from __future__ import annotations

import json

from qec.orchestration.deterministic_agent_simulation_sandbox import (
    AgentSimulationScenario,
    build_agent_simulation_scenario,
    run_deterministic_agent_simulation,
)
from qec.orchestration.deterministic_covenant_engine import build_covenant_rule
from qec.orchestration.governance_benchmark_battery import (
    GovernanceBenchmarkScenario,
    build_governance_benchmark_receipt,
    build_governance_benchmark_scenario,
    compare_governance_benchmark_replay,
    run_governance_benchmark_battery,
    summarize_governance_benchmark,
    validate_governance_benchmark,
)
from qec.orchestration.policy_execution_firewall_kernel import build_firewall_policy_rule
from qec.orchestration.proof_carrying_agent_action_capsule import (
    AgentActionProofObligation,
    build_proof_carrying_agent_action_capsule,
)


def _capsule(action_type: str = "validate") -> object:
    return build_proof_carrying_agent_action_capsule(
        action_id=f"a-{action_type}",
        action_type=action_type,
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


def _scenario(*, action_type: str = "validate", steps: int = 3) -> AgentSimulationScenario:
    capsule = _capsule(action_type=action_type)
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


def _sandbox(*, action_type: str = "validate", steps: int = 3):
    return run_deterministic_agent_simulation(_scenario(action_type=action_type, steps=steps))


def test_deterministic_repeated_benchmark_runs() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(), _sandbox()))
    a = run_governance_benchmark_battery(scenario)
    b = run_governance_benchmark_battery(scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_stable_hash_reproducibility() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(),))
    run = run_governance_benchmark_battery(scenario)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert run.benchmark_receipt.stable_hash() == run.benchmark_receipt.stable_hash()
    assert run.stable_hash() == run.stable_hash()


def test_allow_deny_metric_correctness() -> None:
    allow = _sandbox(action_type="validate", steps=2)
    deny = _sandbox(action_type="certify", steps=3)
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(allow, deny))
    )
    metrics = battery.benchmark_receipt.aggregate_metrics
    assert metrics["allow_count"] == 2.0
    assert metrics["deny_count"] == 3.0
    assert metrics["allow_rate"] == 0.4
    assert metrics["deny_rate"] == 0.6


def test_replay_stability_metric_correctness() -> None:
    stable_a = _sandbox(action_type="validate", steps=2)
    stable_b = _sandbox(action_type="validate", steps=2)
    drift = _sandbox(action_type="certify", steps=2)
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(stable_a, stable_b, drift))
    )
    metrics = battery.benchmark_receipt.aggregate_metrics
    assert metrics["replay_stability_rate"] == 1.0
    assert metrics["policy_drift_rate"] == 0.0
    assert metrics["receipt_reproducibility"] == 1.0


def test_continuity_metric_correctness() -> None:
    stable = _sandbox(action_type="validate", steps=3)
    battery = run_governance_benchmark_battery(build_governance_benchmark_scenario(simulation_set=(stable,)))
    assert battery.benchmark_receipt.aggregate_metrics["continuity_success_rate"] == (2.0 / 3.0)


def test_malformed_input_handling() -> None:
    assert validate_governance_benchmark({"bad": "shape"}) == ("malformed_benchmark:type",)


def test_validator_never_raises() -> None:
    class Weird:
        pass

    violations = validate_governance_benchmark(Weird())
    assert isinstance(violations, tuple)
    assert violations == ("malformed_benchmark:type",)


def test_replay_comparison_stability() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(), _sandbox()))
    a = run_governance_benchmark_battery(scenario)
    b = run_governance_benchmark_battery(scenario)
    cmp_report = compare_governance_benchmark_replay(a, b)
    assert cmp_report["match"] is True
    assert cmp_report["mismatch_fields"] == ()


def test_canonical_json_round_trip() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(),))
    run = run_governance_benchmark_battery(scenario)
    data = json.loads(run.to_canonical_json())
    assert data["benchmark_receipt"]["receipt_hash"] == run.benchmark_receipt.receipt_hash


def test_deterministic_metric_ordering() -> None:
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(_sandbox(steps=1),))
    )
    assert tuple(battery.benchmark_receipt.aggregate_metrics.keys()) == (
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


def test_identical_input_benchmark_reproducibility() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(), _sandbox()))
    artifacts = tuple(run_governance_benchmark_battery(scenario).to_canonical_json() for _ in range(4))
    assert len(set(artifacts)) == 1


def test_boundary_failure_rate_for_internal_io_is_one() -> None:
    sc = _scenario(steps=2)
    internal = build_agent_simulation_scenario(
        action_capsule=sc.action_capsule,
        covenant_rule=sc.covenant_rule,
        sandbox_policy_rules=sc.sandbox_policy_rules,
        initial_state={"counter": 0.0},
        simulation_steps=2,
        io_surface="internal",
    )
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(run_deterministic_agent_simulation(internal),))
    )
    assert battery.benchmark_receipt.aggregate_metrics["boundary_failure_rate"] == 0.5


def test_boundary_failure_rate_all_invalid_is_one() -> None:
    # A single-step simulation always produces within_boundary=False on step 0
    # (no prior receipt hash exists), yielding boundary_failure_rate == 1.0.
    sim = run_deterministic_agent_simulation(_scenario(steps=1))
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(sim,))
    )
    assert battery.benchmark_receipt.aggregate_metrics["boundary_failure_rate"] == 1.0


def test_boundary_failure_rate_fully_valid_is_zero() -> None:
    # Chain a second simulation that receives a valid prior receipt hash so that
    # all steps satisfy within_boundary=True, yielding boundary_failure_rate == 0.0.
    sc = _scenario(steps=1)
    first = run_deterministic_agent_simulation(sc)
    prior_hash = first.simulated_execution_trace[0].simulated_boundary_result.get(
        "audit_receipt", {}
    ).get("receipt_hash", "")
    chained_sc = build_agent_simulation_scenario(
        action_capsule=sc.action_capsule,
        covenant_rule=sc.covenant_rule,
        sandbox_policy_rules=sc.sandbox_policy_rules,
        initial_state={"counter": 1.0},
        simulation_steps=2,
        io_surface="none",
        prior_transition_receipt_hash=prior_hash,
    )
    second = run_deterministic_agent_simulation(chained_sc)
    battery = run_governance_benchmark_battery(
        build_governance_benchmark_scenario(simulation_set=(second,))
    )
    assert battery.benchmark_receipt.aggregate_metrics["boundary_failure_rate"] == 0.0


def test_summarize_ordering_by_index() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(steps=2), _sandbox(steps=1)))
    battery = run_governance_benchmark_battery(scenario)
    summary = summarize_governance_benchmark(battery)
    indices = [row["simulation_index"] for row in summary["results"]]
    assert indices == sorted(indices)


def test_build_receipt_reproducibility() -> None:
    scenario = build_governance_benchmark_scenario(simulation_set=(_sandbox(),))
    battery = run_governance_benchmark_battery(scenario)
    a = build_governance_benchmark_receipt(
        scenario=scenario,
        benchmark_results=battery.benchmark_results,
        validation_violations=battery.validation_violations,
    )
    b = build_governance_benchmark_receipt(
        scenario=scenario,
        benchmark_results=battery.benchmark_results,
        validation_violations=battery.validation_violations,
    )
    assert a.receipt_hash == b.receipt_hash


def test_variant_input_not_mutated() -> None:
    variants = [{"rule": "z"}, {"rule": "a"}]
    snapshot = json.dumps(variants, sort_keys=True)
    _ = build_governance_benchmark_scenario(simulation_set=(_sandbox(),), firewall_policy_variants=variants)
    assert json.dumps(variants, sort_keys=True) == snapshot


def test_malformed_dataclass_returns_violations_without_raise() -> None:
    malformed = GovernanceBenchmarkScenario(
        simulation_set=("bad",),  # type: ignore[arg-type]
        benchmark_rules={"required_metric_names": []},
        firewall_policy_variants=(),
        covenant_variants=(),
        boundary_rule_variants=(),
    )
    battery = run_governance_benchmark_battery(malformed)
    assert battery.benchmark_results == ()
    assert "malformed_benchmark:simulation:0" in battery.validation_violations
