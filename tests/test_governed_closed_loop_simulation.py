from __future__ import annotations

import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.closed_loop_simulation_kernel import SimulationConfig
from qec.analysis.deterministic_stress_lattice import StressAxis
from qec.analysis.governed_orchestration_layer import (
    GovernedOrchestrationReceipt,
    OrchestrationVerdict,
)
from qec.analysis.governed_closed_loop_simulation import (
    GovernedClosedLoopReceipt,
    GovernedCycleRecord,
    GovernedSimulationSummary,
    run_governed_closed_loop,
)
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.state_conditioned_filter_mesh import FilterOrdering


def _axes() -> tuple[StressAxis, ...]:
    return (
        StressAxis("thermal_pressure", 0.0, 1.0),
        StressAxis("latency_drift", 0.0, 1.0),
        StressAxis("timing_skew", 0.0, 1.0),
        StressAxis("power_pressure", 0.0, 1.0),
        StressAxis("consensus_instability", 0.0, 1.0),
    )


def _orderings() -> tuple[FilterOrdering, ...]:
    return (
        FilterOrdering.build(("thermal_stabilize", "parity_gate"), ("boundary_control",)),
        FilterOrdering.build(("latency_buffer", "spectral_phase"), ("surface_sync",)),
        FilterOrdering.build(("power_budget", "consensus_stabilize"), ("timing_sync",)),
    )


def _config(*, cycle_count: int = 8, recurrence_window: int = 6, point_count: int = 4) -> SimulationConfig:
    axes = tuple(sorted(_axes(), key=lambda axis: axis.name))
    candidate_orderings = tuple(sorted(_orderings(), key=lambda o: (o.ordering_signature, o.stable_hash)))
    payload = {
        "axes": tuple(axis.to_dict() for axis in axes),
        "point_count": point_count,
        "stress_method": "lattice",
        "cycle_count": cycle_count,
        "candidate_orderings": tuple(item.to_dict() for item in candidate_orderings),
        "recurrence_window": recurrence_window,
    }
    return SimulationConfig(
        axes=axes,
        point_count=point_count,
        stress_method="lattice",
        cycle_count=cycle_count,
        candidate_orderings=candidate_orderings,
        recurrence_window=recurrence_window,
        stable_hash=sha256_hex(payload),
    )


def _policy(
    *,
    min_required_score: float = 0.55,
    min_required_confidence: float = 0.95,
    min_required_margin: float = 0.45,
    min_required_convergence: float = 0.90,
    allow_tie_break: bool = False,
    allow_no_improvement: bool = False,
    require_stable_transition: bool = True,
) -> GovernancePolicy:
    payload = {
        "min_required_score": round(min_required_score, 12),
        "min_required_confidence": round(min_required_confidence, 12),
        "min_required_margin": round(min_required_margin, 12),
        "min_required_convergence": round(min_required_convergence, 12),
        "allow_tie_break": allow_tie_break,
        "allow_no_improvement": allow_no_improvement,
        "require_stable_transition": require_stable_transition,
    }
    return GovernancePolicy(
        min_required_score=min_required_score,
        min_required_confidence=min_required_confidence,
        min_required_margin=min_required_margin,
        min_required_convergence=min_required_convergence,
        allow_tie_break=allow_tie_break,
        allow_no_improvement=allow_no_improvement,
        require_stable_transition=require_stable_transition,
        stable_hash=sha256_hex(payload),
    )


def test_deterministic_replay() -> None:
    config = _config()
    policy = _policy()
    first = run_governed_closed_loop(config, policy)
    second = run_governed_closed_loop(config, policy)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_full_pipeline_integration_and_linkage() -> None:
    config = _config(cycle_count=6)
    policy = _policy()
    receipt = run_governed_closed_loop(config, policy)
    assert len(receipt.cycle_records) == config.cycle_count
    for cycle in receipt.cycle_records:
        assert len(cycle.mesh_receipt_hash) == 64
        assert len(cycle.transition_receipt_hash) == 64
        assert len(cycle.refinement_receipt_hash) == 64
        assert len(cycle.governance_receipt_hash) == 64


def test_governance_distribution_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    import qec.analysis.governed_closed_loop_simulation as gcls

    original_eval = gcls.evaluate_governed_orchestration
    verdict_by_cycle = ("allow", "hold", "reject", "allow", "hold", "reject", "allow", "hold", "reject", "allow")
    call_index = {"value": 0}

    def _cycled(policy: GovernancePolicy, transition, refinement):
        idx = call_index["value"]
        call_index["value"] += 1
        base = original_eval(policy, transition, refinement)
        verdict_value = verdict_by_cycle[idx]
        reason = "within_policy" if verdict_value == "allow" else ("low_confidence" if verdict_value == "hold" else "score_too_low")
        admissible = verdict_value == "allow"
        verdict_payload = {
            "verdict": verdict_value,
            "admissible": admissible,
            "reason_code": reason,
            "selected_ordering_signature": base.verdict.selected_ordering_signature,
            "decision_type": base.verdict.decision_type,
            "transition_classification": base.verdict.transition_classification,
            "refinement_classification": base.verdict.refinement_classification,
        }
        verdict = OrchestrationVerdict(**{**verdict_payload, "stable_hash": sha256_hex(verdict_payload)})
        receipt_payload = {
            "policy": base.policy.to_dict(),
            "input_transition_hash": base.input_transition_hash,
            "input_refinement_hash": base.input_refinement_hash,
            "checks": tuple(check.to_dict() for check in base.checks),
            "verdict": verdict.to_dict(),
        }
        return GovernedOrchestrationReceipt(
            policy=base.policy,
            input_transition_hash=base.input_transition_hash,
            input_refinement_hash=base.input_refinement_hash,
            checks=base.checks,
            verdict=verdict,
            stable_hash=sha256_hex(receipt_payload),
        )

    monkeypatch.setattr(gcls, "evaluate_governed_orchestration", _cycled)
    receipt = run_governed_closed_loop(_config(cycle_count=10), _policy())
    verdicts = {record.governance_verdict for record in receipt.cycle_records}
    assert {"allow", "hold", "reject"}.issubset(verdicts)


def test_admissible_count_equals_allow_count() -> None:
    summary = run_governed_closed_loop(_config(cycle_count=9), _policy()).summary
    assert summary.admissible_count == summary.allow_count


def test_summary_consistency_counts() -> None:
    summary = run_governed_closed_loop(_config(cycle_count=11), _policy()).summary
    assert summary.allow_count + summary.hold_count + summary.reject_count == summary.cycle_count


def test_linkage_validation_tampered_governance_input_hashes_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import qec.analysis.governed_closed_loop_simulation as gcls

    original_eval = gcls.evaluate_governed_orchestration

    def _tampered(policy: GovernancePolicy, transition, refinement):
        receipt = original_eval(policy, transition, refinement)
        object.__setattr__(receipt, "input_transition_hash", "0" * 64)
        payload = {
            "policy": receipt.policy.to_dict(),
            "input_transition_hash": receipt.input_transition_hash,
            "input_refinement_hash": receipt.input_refinement_hash,
            "checks": tuple(check.to_dict() for check in receipt.checks),
            "verdict": receipt.verdict.to_dict(),
        }
        object.__setattr__(receipt, "stable_hash", sha256_hex(payload))
        return receipt

    monkeypatch.setattr(gcls, "evaluate_governed_orchestration", _tampered)
    with pytest.raises(ValueError, match="input_transition_hash"):
        run_governed_closed_loop(_config(), _policy())


def test_invalid_policy_rejection() -> None:
    with pytest.raises(ValueError, match="min_required_score"):
        _policy(min_required_score=1.1)


def test_enum_validation_invalid_verdict() -> None:
    payload = {
        "cycle_index": 0,
        "transition_classification": "stable_transition",
        "refinement_classification": "bounded",
        "governance_verdict": "maybe",
        "governance_admissible": False,
        "governance_reason": "x",
        "convergence_metric": 0.5,
        "mesh_receipt_hash": "0" * 64,
        "transition_receipt_hash": "1" * 64,
        "refinement_receipt_hash": "2" * 64,
        "governance_receipt_hash": "3" * 64,
    }
    with pytest.raises(ValueError, match="governance_verdict"):
        GovernedCycleRecord(**payload, stable_hash=sha256_hex(payload))


def test_governance_invariant_rejects_verdict_admissible_mismatch() -> None:
    payload = {
        "cycle_index": 0,
        "transition_classification": "stable_transition",
        "refinement_classification": "bounded",
        "governance_verdict": "allow",
        "governance_admissible": False,
        "governance_reason": "within_policy",
        "convergence_metric": 0.5,
        "mesh_receipt_hash": "0" * 64,
        "transition_receipt_hash": "1" * 64,
        "refinement_receipt_hash": "2" * 64,
        "governance_receipt_hash": "3" * 64,
    }
    with pytest.raises(ValueError, match="governance_admissible must equal"):
        GovernedCycleRecord(**payload, stable_hash=sha256_hex(payload))


def test_canonical_reconstruction_stability() -> None:
    receipt = run_governed_closed_loop(_config(cycle_count=7), _policy())
    payload = json.loads(receipt.to_canonical_json())
    config_payload = payload["config"]
    config = SimulationConfig(
        axes=tuple(StressAxis(**axis) for axis in config_payload["axes"]),
        point_count=config_payload["point_count"],
        stress_method=config_payload["stress_method"],
        cycle_count=config_payload["cycle_count"],
        candidate_orderings=tuple(
            FilterOrdering(
                input_filters=tuple(item["input_filters"]),
                control_filters=tuple(item["control_filters"]),
                ordering_signature=item["ordering_signature"],
                stable_hash=item["stable_hash"],
            )
            for item in config_payload["candidate_orderings"]
        ),
        recurrence_window=config_payload["recurrence_window"],
        stable_hash=config_payload["stable_hash"],
    )
    policy = GovernancePolicy(**payload["policy"])
    records = tuple(GovernedCycleRecord(**record) for record in payload["cycle_records"])
    summary = GovernedSimulationSummary(**payload["summary"])
    rebuilt = GovernedClosedLoopReceipt(
        config=config,
        policy=policy,
        stress_receipt_hash=payload["stress_receipt_hash"],
        cycle_records=records,
        summary=summary,
        stable_hash=payload["stable_hash"],
    )
    assert rebuilt.stable_hash == receipt.stable_hash
    assert rebuilt.to_canonical_json() == receipt.to_canonical_json()


def test_policy_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import qec.analysis.governed_closed_loop_simulation as gcls

    original_eval = gcls.evaluate_governed_orchestration

    def _mismatched_policy(policy: GovernancePolicy, transition, refinement):
        receipt = original_eval(policy, transition, refinement)
        other_policy = _policy(min_required_score=0.0)
        object.__setattr__(receipt, "policy", other_policy)
        payload = {
            "policy": other_policy.to_dict(),
            "input_transition_hash": receipt.input_transition_hash,
            "input_refinement_hash": receipt.input_refinement_hash,
            "checks": tuple(check.to_dict() for check in receipt.checks),
            "verdict": receipt.verdict.to_dict(),
        }
        object.__setattr__(receipt, "stable_hash", sha256_hex(payload))
        return receipt

    monkeypatch.setattr(gcls, "evaluate_governed_orchestration", _mismatched_policy)
    with pytest.raises(ValueError, match="policy mismatch"):
        run_governed_closed_loop(_config(), _policy())


def test_admissible_count_derived_from_record_flags() -> None:
    records = (
        GovernedCycleRecord(
            cycle_index=0,
            transition_classification="stable_transition",
            refinement_classification="bounded",
            governance_verdict="allow",
            governance_admissible=True,
            governance_reason="within_policy",
            convergence_metric=0.5,
            mesh_receipt_hash="0" * 64,
            transition_receipt_hash="1" * 64,
            refinement_receipt_hash="2" * 64,
            governance_receipt_hash="3" * 64,
            stable_hash=sha256_hex(
                {
                    "cycle_index": 0,
                    "transition_classification": "stable_transition",
                    "refinement_classification": "bounded",
                    "governance_verdict": "allow",
                    "governance_admissible": True,
                    "governance_reason": "within_policy",
                    "convergence_metric": 0.5,
                    "mesh_receipt_hash": "0" * 64,
                    "transition_receipt_hash": "1" * 64,
                    "refinement_receipt_hash": "2" * 64,
                    "governance_receipt_hash": "3" * 64,
                }
            ),
        ),
        GovernedCycleRecord(
            cycle_index=1,
            transition_classification="uncertain_transition",
            refinement_classification="no_improvement",
            governance_verdict="hold",
            governance_admissible=False,
            governance_reason="low_confidence",
            convergence_metric=0.25,
            mesh_receipt_hash="4" * 64,
            transition_receipt_hash="5" * 64,
            refinement_receipt_hash="6" * 64,
            governance_receipt_hash="7" * 64,
            stable_hash=sha256_hex(
                {
                    "cycle_index": 1,
                    "transition_classification": "uncertain_transition",
                    "refinement_classification": "no_improvement",
                    "governance_verdict": "hold",
                    "governance_admissible": False,
                    "governance_reason": "low_confidence",
                    "convergence_metric": 0.25,
                    "mesh_receipt_hash": "4" * 64,
                    "transition_receipt_hash": "5" * 64,
                    "refinement_receipt_hash": "6" * 64,
                    "governance_receipt_hash": "7" * 64,
                }
            ),
        ),
    )

    summary = GovernedSimulationSummary(
        cycle_count=2,
        allow_count=1,
        hold_count=1,
        reject_count=0,
        admissible_count=sum(1 for record in records if record.governance_admissible),
        non_admissible_count=sum(1 for record in records if not record.governance_admissible),
        mean_convergence_metric=0.375,
        stable_transition_count=1,
        uncertain_transition_count=1,
        recurrence_classification="aperiodic",
        dominant_recurrence_period=None,
        stable_hash=sha256_hex(
            {
                "cycle_count": 2,
                "allow_count": 1,
                "hold_count": 1,
                "reject_count": 0,
                "admissible_count": 1,
                "non_admissible_count": 1,
                "mean_convergence_metric": 0.375,
                "stable_transition_count": 1,
                "uncertain_transition_count": 1,
                "recurrence_classification": "aperiodic",
                "dominant_recurrence_period": None,
            }
        ),
    )
    assert summary.admissible_count == 1
    assert summary.non_admissible_count == 1


def test_governance_verdicts_do_not_influence_loop_progression() -> None:
    config = _config(cycle_count=8)
    strict = _policy(
        min_required_score=0.8,
        min_required_confidence=0.99,
        min_required_margin=0.8,
        min_required_convergence=0.99,
        allow_tie_break=False,
        allow_no_improvement=False,
        require_stable_transition=True,
    )
    permissive = _policy(
        min_required_score=0.0,
        min_required_confidence=0.0,
        min_required_margin=0.0,
        min_required_convergence=0.0,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
    )
    strict_receipt = run_governed_closed_loop(config, strict)
    permissive_receipt = run_governed_closed_loop(config, permissive)

    strict_artifacts = tuple(
        (r.mesh_receipt_hash, r.transition_receipt_hash, r.refinement_receipt_hash)
        for r in strict_receipt.cycle_records
    )
    permissive_artifacts = tuple(
        (r.mesh_receipt_hash, r.transition_receipt_hash, r.refinement_receipt_hash)
        for r in permissive_receipt.cycle_records
    )
    assert strict_artifacts == permissive_artifacts
    assert tuple(r.governance_verdict for r in strict_receipt.cycle_records) != tuple(
        r.governance_verdict for r in permissive_receipt.cycle_records
    )
