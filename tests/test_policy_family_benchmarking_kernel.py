from __future__ import annotations

import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.closed_loop_simulation_kernel import SimulationConfig
from qec.analysis.deterministic_stress_lattice import StressAxis
from qec.analysis.governed_closed_loop_simulation import (
    GovernedClosedLoopReceipt,
    GovernedCycleRecord,
    GovernedSimulationSummary,
)
from qec.analysis.governed_orchestration_layer import GovernancePolicy
from qec.analysis.policy_family_benchmarking_kernel import (
    FAMILY_CLASS_HIGH_VARIATION,
    FAMILY_CLASS_STABLE,
    GeneratedPolicyRecord,
    PolicyBenchmarkComparison,
    PolicyFamilyBenchmarkReceipt,
    PolicyFamilyBenchmarkSummary,
    PolicyFamilySpec,
    PolicySweepAxis,
    benchmark_policy_family,
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


def _axis(parameter_name: str, values: tuple[float | bool, ...]) -> PolicySweepAxis:
    payload = {
        "parameter_name": parameter_name,
        "values": tuple(round(float(v), 12) if isinstance(v, (int, float)) and not isinstance(v, bool) else v for v in values),
    }
    return PolicySweepAxis(parameter_name=parameter_name, values=values, stable_hash=sha256_hex(payload))


def _family_spec(axes: tuple[PolicySweepAxis, ...], max_family_size: int = 16) -> PolicyFamilySpec:
    canonical_axes = tuple(sorted(axes, key=lambda axis: axis.parameter_name))
    payload = {
        "axes": tuple(axis.to_dict() for axis in canonical_axes),
        "max_family_size": max_family_size,
    }
    return PolicyFamilySpec(axes=axes, max_family_size=max_family_size, stable_hash=sha256_hex(payload))


def _synthetic_governed_receipt(config: SimulationConfig, policy: GovernancePolicy) -> GovernedClosedLoopReceipt:
    permissiveness = (
        (1.0 - policy.min_required_score)
        + (1.0 - policy.min_required_confidence)
        + (1.0 - policy.min_required_margin)
        + (1.0 - policy.min_required_convergence)
        + (1.0 if policy.allow_tie_break else 0.0)
        + (1.0 if policy.allow_no_improvement else 0.0)
        + (1.0 if not policy.require_stable_transition else 0.0)
    ) / 7.0
    allow_count = int(round(permissiveness * float(config.cycle_count)))
    if allow_count < 0:
        allow_count = 0
    if allow_count > config.cycle_count:
        allow_count = config.cycle_count
    reject_count = config.cycle_count - allow_count
    convergence_metric = round(min(1.0, max(0.0, permissiveness)), 12)

    cycle_records: list[GovernedCycleRecord] = []
    for idx in range(config.cycle_count):
        allow = idx < allow_count
        payload = {
            "cycle_index": idx,
            "transition_classification": "stable_transition",
            "refinement_classification": "converged",
            "governance_verdict": "allow" if allow else "reject",
            "governance_admissible": allow,
            "governance_reason": "within_policy" if allow else "score_too_low",
            "convergence_metric": convergence_metric,
            "mesh_receipt_hash": "a" * 64,
            "transition_receipt_hash": "b" * 64,
            "refinement_receipt_hash": "c" * 64,
            "governance_receipt_hash": "d" * 64,
        }
        cycle_records.append(GovernedCycleRecord(**payload, stable_hash=sha256_hex(payload)))

    summary_payload = {
        "cycle_count": config.cycle_count,
        "allow_count": allow_count,
        "hold_count": 0,
        "reject_count": reject_count,
        "admissible_count": allow_count,
        "non_admissible_count": reject_count,
        "mean_convergence_metric": convergence_metric,
        "stable_transition_count": config.cycle_count,
        "uncertain_transition_count": 0,
        "recurrence_classification": "not_evaluated",
        "dominant_recurrence_period": None,
    }
    summary = GovernedSimulationSummary(**summary_payload, stable_hash=sha256_hex(summary_payload))
    receipt_payload = {
        "config": config.to_dict(),
        "policy": policy.to_dict(),
        "stress_receipt_hash": "e" * 64,
        "cycle_records": tuple(record.to_dict() for record in cycle_records),
        "summary": summary.to_dict(),
    }
    return GovernedClosedLoopReceipt(
        config=config,
        policy=policy,
        stress_receipt_hash="e" * 64,
        cycle_records=tuple(cycle_records),
        summary=summary,
        stable_hash=sha256_hex(receipt_payload),
    )


def test_deterministic_replay() -> None:
    config = _config()
    baseline = _policy()
    family_spec = _family_spec(
        (
            _axis("min_required_score", (0.3, 0.7)),
            _axis("allow_tie_break", (False, True)),
        )
    )
    first = benchmark_policy_family(config, baseline, family_spec)
    second = benchmark_policy_family(config, baseline, family_spec)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_invalid_family_size_rejection() -> None:
    family_spec = _family_spec(
        (
            _axis("min_required_score", (0.2, 0.3, 0.4)),
            _axis("allow_tie_break", (False, True)),
        ),
        max_family_size=2,
    )
    with pytest.raises(ValueError, match="generated family size"):
        benchmark_policy_family(_config(), _policy(), family_spec)


def test_duplicate_axis_rejection() -> None:
    axis_a = _axis("min_required_score", (0.2, 0.3))
    axis_b = _axis("min_required_score", (0.4, 0.5))
    payload = {
        "axes": tuple(axis.to_dict() for axis in (axis_a, axis_b)),
        "max_family_size": 8,
    }
    with pytest.raises(ValueError, match="duplicate"):
        PolicyFamilySpec(axes=(axis_a, axis_b), max_family_size=8, stable_hash=sha256_hex(payload))


def test_invalid_parameter_name_rejection() -> None:
    payload = {
        "parameter_name": "unsupported_parameter",
        "values": (0.1, 0.2),
    }
    with pytest.raises(ValueError, match="invalid"):
        PolicySweepAxis(parameter_name="unsupported_parameter", values=(0.1, 0.2), stable_hash=sha256_hex(payload))


def test_policy_family_generation_correctness() -> None:
    baseline = _policy(min_required_margin=0.61, allow_no_improvement=True)
    family_spec = _family_spec(
        (
            _axis("min_required_score", (0.2, 0.4)),
            _axis("allow_tie_break", (False, True)),
        ),
        max_family_size=8,
    )
    receipt = benchmark_policy_family(_config(), baseline, family_spec)
    assert len(receipt.generated_policy_records) == 4
    combos = {(r.parameter_overrides[0][1], r.parameter_overrides[1][1]) for r in receipt.generated_policy_records}
    assert combos == {(False, 0.2), (False, 0.4), (True, 0.2), (True, 0.4)}
    for record in receipt.generated_policy_records:
        assert baseline.min_required_margin == 0.61


def test_generated_ordering_canonicality() -> None:
    baseline = _policy()
    ordered = _family_spec((_axis("allow_tie_break", (False, True)), _axis("min_required_score", (0.2, 0.6))))
    unordered = _family_spec((_axis("min_required_score", (0.2, 0.6)), _axis("allow_tie_break", (False, True))))
    ordered_receipt = benchmark_policy_family(_config(), baseline, ordered)
    unordered_receipt = benchmark_policy_family(_config(), baseline, unordered)
    assert ordered_receipt.to_canonical_json() == unordered_receipt.to_canonical_json()
    hashes = tuple(record.policy_hash for record in unordered_receipt.generated_policy_records)
    assert hashes == tuple(sorted(hashes))


def test_governed_linkage() -> None:
    receipt = benchmark_policy_family(
        _config(),
        _policy(),
        _family_spec((_axis("min_required_score", (0.1, 0.9)),)),
    )
    for record in receipt.generated_policy_records:
        assert len(record.governed_receipt_hash) == 64


def test_comparison_count_correctness() -> None:
    receipt = benchmark_policy_family(
        _config(),
        _policy(),
        _family_spec((_axis("min_required_score", (0.1, 0.5, 0.9)),)),
    )
    n = len(receipt.generated_policy_records)
    assert len(receipt.comparison_records) == n * (n - 1) // 2


def test_permissive_restrictive_identification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.policy_family_benchmarking_kernel.run_governed_closed_loop",
        _synthetic_governed_receipt,
    )
    baseline = _policy(min_required_score=0.7)
    receipt = benchmark_policy_family(
        _config(cycle_count=10),
        baseline,
        _family_spec((_axis("min_required_score", (0.0, 0.95)),)),
    )
    by_hash = {r.policy_hash: r for r in receipt.generated_policy_records}
    low_hash = max(by_hash, key=lambda h: by_hash[h].admissible_count)
    high_hash = min(by_hash, key=lambda h: by_hash[h].admissible_count)
    assert by_hash[low_hash].admissible_count >= by_hash[high_hash].admissible_count
    assert receipt.summary.most_permissive_policy_hash == low_hash
    assert receipt.summary.most_restrictive_policy_hash == high_hash


def test_family_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.policy_family_benchmarking_kernel.run_governed_closed_loop",
        _synthetic_governed_receipt,
    )
    stable_receipt = benchmark_policy_family(
        _config(),
        _policy(),
        _family_spec((_axis("min_required_score", (0.20, 0.21)),)),
    )
    assert stable_receipt.summary.family_behavior_classification == FAMILY_CLASS_STABLE

    high_variation_receipt = benchmark_policy_family(
        _config(cycle_count=10),
        _policy(),
        _family_spec(
            (
                _axis("min_required_score", (0.0, 1.0)),
                _axis("allow_tie_break", (False, True)),
                _axis("allow_no_improvement", (False, True)),
                _axis("require_stable_transition", (True, False)),
            ),
        ),
    )
    assert high_variation_receipt.summary.family_behavior_classification == FAMILY_CLASS_HIGH_VARIATION


def test_duplicate_generated_policy_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _axis("min_required_score", (0.25, 0.2500000000001))


def test_tampered_policy_or_receipt_rejection() -> None:
    tampered = _policy(min_required_score=0.4)
    object.__setattr__(tampered, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="stable_hash"):
        benchmark_policy_family(_config(), tampered, _family_spec((_axis("min_required_score", (0.2, 0.8)),)))


def test_canonical_reconstruction() -> None:
    receipt = benchmark_policy_family(
        _config(),
        _policy(),
        _family_spec((_axis("min_required_score", (0.2, 0.8)), _axis("allow_tie_break", (False, True)))),
    )
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
    family_spec_payload = payload["family_spec"]
    family_spec = PolicyFamilySpec(
        axes=tuple(PolicySweepAxis(**{**axis, "values": tuple(axis["values"])}) for axis in family_spec_payload["axes"]),
        max_family_size=family_spec_payload["max_family_size"],
        stable_hash=family_spec_payload["stable_hash"],
    )
    generated_records = tuple(
        GeneratedPolicyRecord(
            **{
                **record,
                "parameter_overrides": tuple((item[0], item[1]) for item in record["parameter_overrides"]),
            }
        )
        for record in payload["generated_policy_records"]
    )
    comparisons = tuple(PolicyBenchmarkComparison(**record) for record in payload["comparison_records"])
    summary = PolicyFamilyBenchmarkSummary(**payload["summary"])
    rebuilt = PolicyFamilyBenchmarkReceipt(
        config=config,
        baseline_policy_hash=payload["baseline_policy_hash"],
        family_spec=family_spec,
        generated_policy_records=generated_records,
        comparison_records=comparisons,
        summary=summary,
        stable_hash=payload["stable_hash"],
    )
    assert rebuilt.stable_hash == receipt.stable_hash
    assert rebuilt.to_canonical_json() == receipt.to_canonical_json()


def test_no_external_state_dependence() -> None:
    config = _config(cycle_count=7)
    baseline = _policy()
    family_spec = _family_spec((_axis("min_required_score", (0.1, 0.9)), _axis("allow_tie_break", (False, True))))
    first = benchmark_policy_family(config, baseline, family_spec)
    for _ in range(3):
        replay = benchmark_policy_family(config, baseline, family_spec)
        assert replay.stable_hash == first.stable_hash
        assert replay.to_canonical_bytes() == first.to_canonical_bytes()
