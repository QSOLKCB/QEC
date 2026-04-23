from __future__ import annotations

import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.closed_loop_simulation_kernel import SimulationConfig
from qec.analysis.deterministic_stress_lattice import StressAxis
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.policy_sensitivity_analysis_kernel import (
    MIN_POLICY_COUNT,
    PolicyComparisonRecord,
    PolicyRunRecord,
    PolicySensitivityReceipt,
    PolicySensitivitySummary,
    analyze_policy_sensitivity,
)
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
    policies = (
        _policy(min_required_score=0.85, min_required_confidence=0.99, min_required_margin=0.70),
        _policy(min_required_score=0.10, min_required_confidence=0.10, min_required_margin=0.05, allow_tie_break=True),
        _policy(),
    )
    first = analyze_policy_sensitivity(config, policies)
    second = analyze_policy_sensitivity(config, policies)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_policy_count_validation() -> None:
    with pytest.raises(ValueError, match="MIN_POLICY_COUNT"):
        analyze_policy_sensitivity(_config(), (_policy(),))


def test_duplicate_policy_rejection() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="duplicate policy"):
        analyze_policy_sensitivity(_config(), (policy, policy))


def test_governed_linkage() -> None:
    receipt = analyze_policy_sensitivity(_config(), (_policy(), _policy(min_required_score=0.1)))
    for record in receipt.policy_run_records:
        assert len(record.governed_receipt_hash) == 64


def test_comparison_count_correctness() -> None:
    policies = (_policy(), _policy(min_required_score=0.1), _policy(min_required_score=0.8))
    receipt = analyze_policy_sensitivity(_config(), policies)
    assert len(receipt.comparison_records) == len(policies) * (len(policies) - 1) // 2


def test_permissive_restrictive_identification() -> None:
    strict = _policy(
        min_required_score=0.9,
        min_required_confidence=0.99,
        min_required_margin=0.9,
        min_required_convergence=0.99,
        allow_tie_break=False,
        allow_no_improvement=False,
        require_stable_transition=True,
    )
    loose = _policy(
        min_required_score=0.0,
        min_required_confidence=0.0,
        min_required_margin=0.0,
        min_required_convergence=0.0,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
    )
    receipt = analyze_policy_sensitivity(_config(cycle_count=10), (strict, loose))
    by_policy = {record.policy_hash: record for record in receipt.policy_run_records}
    assert by_policy[loose.stable_hash].admissible_count >= by_policy[strict.stable_hash].admissible_count
    assert receipt.summary.most_permissive_policy_hash == loose.stable_hash
    assert receipt.summary.most_restrictive_policy_hash == strict.stable_hash


def test_convergence_extrema_identification() -> None:
    policies = (
        _policy(min_required_score=0.8),
        _policy(min_required_score=0.3),
        _policy(min_required_score=0.1, allow_tie_break=True),
    )
    receipt = analyze_policy_sensitivity(_config(), policies)
    expected_low = min(record.policy_hash for record in receipt.policy_run_records)
    assert receipt.summary.highest_convergence_policy_hash == expected_low
    assert receipt.summary.lowest_convergence_policy_hash == expected_low


def test_sensitivity_classification_equivalent_and_shift() -> None:
    base = _policy(min_required_score=0.0, min_required_confidence=0.0, min_required_margin=0.0, min_required_convergence=0.0)
    equivalent = _policy(
        min_required_score=0.01,
        min_required_confidence=0.0,
        min_required_margin=0.0,
        min_required_convergence=0.0,
        allow_tie_break=False,
        allow_no_improvement=False,
        require_stable_transition=True,
    )
    strict = _policy(min_required_score=0.95, min_required_confidence=0.99, min_required_margin=0.95)
    loose = _policy(
        min_required_score=0.0,
        min_required_confidence=0.0,
        min_required_margin=0.0,
        min_required_convergence=0.0,
        allow_tie_break=True,
        allow_no_improvement=True,
        require_stable_transition=False,
    )

    low_receipt = analyze_policy_sensitivity(_config(), (base, equivalent))
    assert low_receipt.summary.global_sensitivity_classification == "low"
    assert low_receipt.comparison_records[0].sensitivity_classification == "equivalent"

    shift_receipt = analyze_policy_sensitivity(_config(cycle_count=10), (strict, loose))
    assert shift_receipt.summary.global_sensitivity_classification in {"moderate", "high"}
    assert shift_receipt.comparison_records[0].sensitivity_classification in {"moderate_shift", "high_shift"}


def test_canonical_ordering_from_unordered_policies() -> None:
    p1 = _policy(min_required_score=0.2)
    p2 = _policy(min_required_score=0.7)
    p3 = _policy(min_required_score=0.4)
    receipt = analyze_policy_sensitivity(_config(), (p2, p1, p3))
    hashes = tuple(record.policy_hash for record in receipt.policy_run_records)
    assert hashes == tuple(sorted((p1.stable_hash, p2.stable_hash, p3.stable_hash)))
    pair_keys = tuple((record.left_policy_hash, record.right_policy_hash) for record in receipt.comparison_records)
    assert pair_keys == tuple(sorted(pair_keys))


def test_tampered_policy_rejection() -> None:
    tampered = _policy(min_required_score=0.2)
    object.__setattr__(tampered, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="stable_hash"):
        analyze_policy_sensitivity(_config(), (_policy(), tampered))


def test_canonical_reconstruction() -> None:
    policies = (_policy(min_required_score=0.8), _policy(min_required_score=0.1, allow_tie_break=True))
    receipt = analyze_policy_sensitivity(_config(), policies)
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

    run_records = tuple(PolicyRunRecord(**record) for record in payload["policy_run_records"])
    comparisons = tuple(PolicyComparisonRecord(**record) for record in payload["comparison_records"])
    summary = PolicySensitivitySummary(**payload["summary"])

    rebuilt = PolicySensitivityReceipt(
        config=config,
        policy_run_records=run_records,
        comparison_records=comparisons,
        summary=summary,
        stable_hash=payload["stable_hash"],
    )
    assert rebuilt.stable_hash == receipt.stable_hash
    assert rebuilt.to_canonical_json() == receipt.to_canonical_json()


def test_no_external_state_dependence() -> None:
    config = _config(cycle_count=7)
    policies = (
        _policy(min_required_score=0.9),
        _policy(min_required_score=0.1, allow_tie_break=True, allow_no_improvement=True, require_stable_transition=False),
    )
    baseline = analyze_policy_sensitivity(config, policies)
    for _ in range(3):
        replay = analyze_policy_sensitivity(config, policies)
        assert replay.stable_hash == baseline.stable_hash
        assert replay.to_canonical_bytes() == baseline.to_canonical_bytes()


def test_tuple_type_required() -> None:
    with pytest.raises(ValueError, match="policies must be tuple"):
        analyze_policy_sensitivity(_config(), [_policy(), _policy(min_required_score=0.1)])  # type: ignore[arg-type]




def test_duplicate_pair_rejection() -> None:
    policies = (_policy(min_required_score=0.2), _policy(min_required_score=0.5), _policy(min_required_score=0.8))
    receipt = analyze_policy_sensitivity(_config(), policies)
    duplicated_pairs = (
        receipt.comparison_records[0],
        receipt.comparison_records[0],
        receipt.comparison_records[2],
    )
    with pytest.raises(ValueError, match="cover all unordered policy pairs exactly once"):
        PolicySensitivityReceipt(
            config=receipt.config,
            policy_run_records=receipt.policy_run_records,
            comparison_records=duplicated_pairs,
            summary=receipt.summary,
            stable_hash=receipt.stable_hash,
        )


def test_missing_pair_rejection() -> None:
    policies = (_policy(min_required_score=0.2), _policy(min_required_score=0.5), _policy(min_required_score=0.8))
    receipt = analyze_policy_sensitivity(_config(), policies)
    missing_pair_payload = [record.to_dict() for record in receipt.comparison_records]
    missing_pair_payload[1] = missing_pair_payload[0]
    comparisons = tuple(PolicyComparisonRecord(**record) for record in missing_pair_payload)
    summary_payload = receipt.summary.to_dict()
    summary_payload["stable_hash"] = sha256_hex(
        {
            "policy_count": summary_payload["policy_count"],
            "comparison_count": summary_payload["comparison_count"],
            "most_permissive_policy_hash": summary_payload["most_permissive_policy_hash"],
            "most_restrictive_policy_hash": summary_payload["most_restrictive_policy_hash"],
            "highest_convergence_policy_hash": summary_payload["highest_convergence_policy_hash"],
            "lowest_convergence_policy_hash": summary_payload["lowest_convergence_policy_hash"],
            "global_sensitivity_classification": summary_payload["global_sensitivity_classification"],
        }
    )
    summary = PolicySensitivitySummary(**summary_payload)
    payload = {
        "config": receipt.config.to_dict(),
        "policy_run_records": tuple(item.to_dict() for item in receipt.policy_run_records),
        "comparison_records": tuple(item.to_dict() for item in comparisons),
        "summary": summary.to_dict(),
    }
    with pytest.raises(ValueError, match="cover all unordered policy pairs exactly once"):
        PolicySensitivityReceipt(
            config=receipt.config,
            policy_run_records=receipt.policy_run_records,
            comparison_records=comparisons,
            summary=summary,
            stable_hash=sha256_hex(payload),
        )


def test_summary_mismatch_rejection() -> None:
    receipt = analyze_policy_sensitivity(_config(), (_policy(min_required_score=0.9), _policy(min_required_score=0.1)))
    summary_payload = receipt.summary.to_dict()
    summary_payload["global_sensitivity_classification"] = (
        "high" if receipt.summary.global_sensitivity_classification != "high" else "low"
    )
    summary_payload["stable_hash"] = sha256_hex(
        {
            "policy_count": summary_payload["policy_count"],
            "comparison_count": summary_payload["comparison_count"],
            "most_permissive_policy_hash": summary_payload["most_permissive_policy_hash"],
            "most_restrictive_policy_hash": summary_payload["most_restrictive_policy_hash"],
            "highest_convergence_policy_hash": summary_payload["highest_convergence_policy_hash"],
            "lowest_convergence_policy_hash": summary_payload["lowest_convergence_policy_hash"],
            "global_sensitivity_classification": summary_payload["global_sensitivity_classification"],
        }
    )
    summary = PolicySensitivitySummary(**summary_payload)
    payload = {
        "config": receipt.config.to_dict(),
        "policy_run_records": tuple(item.to_dict() for item in receipt.policy_run_records),
        "comparison_records": tuple(item.to_dict() for item in receipt.comparison_records),
        "summary": summary.to_dict(),
    }
    with pytest.raises(ValueError, match="global_sensitivity_classification mismatch"):
        PolicySensitivityReceipt(
            config=receipt.config,
            policy_run_records=receipt.policy_run_records,
            comparison_records=receipt.comparison_records,
            summary=summary,
            stable_hash=sha256_hex(payload),
        )

def test_min_policy_count_constant() -> None:
    assert MIN_POLICY_COUNT == 2
